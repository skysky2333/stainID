"""Click-to-outline with Segment Anything, for people outlining objects in the browser ("Fix the field").

Segment Anything works in two parts. Its image encoder turns a 1024-px window of a field into features: the expensive
part, about 40 ms on a GPU and 5 s on four CPU cores. Its small mask decoder turns those features and the clicks into
outlines in a tenth of a second on the CPU. The features are made ahead and kept on disk
(`stainid.api.sam_embeddings`), so a click usually runs only the decoder.

One model per server process (`outliner`). The encoder runs on the GPU with the most free memory when there is one
(fp16 weights: about 170 MB, 1.5 GB while it encodes, given back after each window, so a step started later still has
the card), else on the CPU with a few threads, so it never takes the processors from running steps;
`$STAINID_SAM_DEVICE` (`cpu`, `cuda`, `cuda:1`) chooses instead. When the GPU fails (it filled up), windows are encoded
on the CPU and the GPU is tried again after `GPU_RETRY_SECONDS`. The decoder always runs on the CPU. Encoding and
decoding each take turns.
"""
from __future__ import annotations

import logging
import os
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage

from stainid.imaging.color import rgb_to_hed
from stainid.masks.objects import MASK_RULES, select_mask

log = logging.getLogger(__name__)

THREADS = 4
DEVICE_ENV = "STAINID_SAM_DEVICE"
MIN_FREE_GPU_BYTES = 3 * 2**30  # a GPU with less free memory than this is left alone
GPU_RETRY_SECONDS = 300.0
EPSILON_PX = 1.0
MIN_PIXELS = 12

_LOCK = threading.Lock()
_MODELS: dict[tuple, ClickOutliner] = {}


@dataclass(frozen=True)
class Embedding:
    """The encoder's features of one window (float16, 1 × 256 × 64 × 64) and the sizes the decoder needs back."""
    features: np.ndarray
    original_size: tuple[int, int]
    input_size: tuple[int, int]


def polygon(mask: np.ndarray, origin: tuple[int, int] = (0, 0)) -> list[list[int]] | None:
    """The outline of a mask's largest piece (holes filled), simplified, in field pixels."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if count < 2:
        return None
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[largest, cv2.CC_STAT_AREA] < MIN_PIXELS:
        return None
    x, y, w, h = (int(v) for v in stats[largest, :4])  # only its box (and a pixel around it): the same outline, faster
    piece = np.pad(ndimage.binary_fill_holes(labels[y: y + h, x: x + w] == largest), 1).astype(np.uint8)
    contours, _ = cv2.findContours(piece, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = cv2.approxPolyDP(max(contours, key=cv2.contourArea), EPSILON_PX, True)[:, 0, :]
    if len(contour) < 3:
        return None
    return (contour + np.asarray(origin) + np.array([x - 1, y - 1])).astype(int).tolist()


def build_model(checkpoint: Path):
    """Segment Anything ViT-B from its weights, on the CPU in fp32 (tests swap in a tiny one)."""
    from segment_anything import sam_model_registry

    return sam_model_registry["vit_b"](checkpoint=str(checkpoint)).eval()


def _free_by_card() -> dict[int, int]:
    """Free memory of each GPU in bytes, without starting CUDA on any of them ({} when nvidia-smi cannot tell)."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10, check=True).stdout
        return {int(i): int(free) * 2**20 for i, free in (line.split(",") for line in out.strip().splitlines())}
    except (OSError, subprocess.SubprocessError, ValueError):
        return {}


def choose_device() -> str:
    """`$STAINID_SAM_DEVICE` when set, else the GPU with the most free memory (at least `MIN_FREE_GPU_BYTES`), else
    the CPU."""
    wanted = os.environ.get(DEVICE_ENV, "").strip()
    if wanted:
        return wanted
    import torch

    if not torch.cuda.is_available():
        return "cpu"
    free = _free_by_card() if torch.cuda.device_count() > 1 and not os.environ.get("CUDA_VISIBLE_DEVICES") else {}
    index = max(free, key=free.get) if free else 0
    try:
        available = torch.cuda.mem_get_info(index)[0]
    except RuntimeError:
        return "cpu"
    return f"cuda:{index}" if available >= MIN_FREE_GPU_BYTES else "cpu"


class ClickOutliner:
    def __init__(self, checkpoint: Path, device: str | None = None):
        import torch
        from segment_anything import SamPredictor

        torch.set_num_threads(THREADS)
        self.checkpoint = checkpoint
        self.model = build_model(checkpoint)
        self.predictor = SamPredictor(self.model)  # the decoder, on the CPU
        self.encoding = threading.RLock()
        self.decoding = threading.Lock()
        self.gpu: str | None = None
        self._cpu_encoder = self.model.image_encoder
        self._gpu_failed = -GPU_RETRY_SECONDS
        device = device or choose_device()
        if device != "cpu":
            try:
                self._gpu_encoder = self.model.image_encoder.half().to(torch.device(device))
                self.gpu = device
                self._cpu_encoder = None  # made again from the weights if the GPU ever fails
                self.encode(np.zeros((64, 64, 3), np.uint8))  # the card's first run is slow: not on someone's click
            except (RuntimeError, AssertionError) as error:
                log.warning("Segment Anything stays on the CPU: %s", error)
                self.model.image_encoder.float().cpu()
        log.info("Segment Anything for click-to-outline on %s", self.gpu or "the CPU")

    @property
    def on_gpu(self) -> bool:
        return self.gpu is not None and time.monotonic() - self._gpu_failed >= GPU_RETRY_SECONDS

    def _cpu(self):
        if self._cpu_encoder is None:
            self._cpu_encoder = build_model(self.checkpoint).image_encoder
        return self._cpu_encoder

    def encode(self, window: np.ndarray) -> Embedding:
        """The features of one window (an RGB array of at most 1024 × 1024)."""
        import torch

        transformed = self.predictor.transform.apply_image(window)
        x = self.model.preprocess(torch.as_tensor(transformed).permute(2, 0, 1).contiguous()[None].float())
        with self.encoding, torch.inference_mode():
            features = None
            if self.on_gpu:
                try:
                    features = self._gpu_encoder(x.to(self.gpu, torch.float16)).float().cpu()
                except RuntimeError as error:  # out of memory, or the card went away
                    log.warning("Segment Anything on %s failed, using the CPU for %.0f s: %s", self.gpu, GPU_RETRY_SECONDS, error)
                    self._gpu_failed = time.monotonic()
                finally:
                    torch.cuda.empty_cache()
            if features is None:
                features = self._cpu()(x)
        return Embedding(features.numpy().astype(np.float16), tuple(window.shape[:2]), tuple(transformed.shape[:2]))

    def outline(self, embedding: Embedding, origin: tuple[int, int], rgb: np.ndarray, stain: str,
                points: list[tuple[float, float, bool]], box: tuple[float, float, float, float] | None = None,
                sizes: bool = False, avoid: list[list[list[float]]] | None = None) -> dict:
        """Outlines for one object from clicks (x, y, on the object or not; the first one on it) in the window at
        `origin` of the field `rgb`: with one click the three sizes Segment Anything offers (smallest first) and the
        one this stain's outline step would choose, with more clicks the one outline that fits them all. `sizes`: the
        three sizes whatever the clicks; `box` (x0, y0, x1, y1, field pixels) bounds the object; `avoid` (outlines in
        field pixels) is cut out of every result, which keeps its largest part (a split piece's siblings)."""
        import torch

        x0, y0 = origin
        height, width = embedding.original_size
        inside = [(px, py, on) for px, py, on in points if x0 <= px < x0 + width and y0 <= py < y0 + height]
        coords = np.array([[px - x0, py - y0] for px, py, _ in inside], float)
        labels = np.array([1 if on else 0 for _, _, on in inside])
        many = len(inside) > 1 and not sizes
        framed = None if box is None else np.array([box[0] - x0, box[1] - y0, box[2] - x0, box[3] - y0], float)
        with self.decoding:
            predictor = self.predictor
            predictor.features = torch.from_numpy(embedding.features.astype(np.float32))
            predictor.original_size, predictor.input_size = embedding.original_size, embedding.input_size
            predictor.is_image_set = True
            masks, scores, _ = predictor.predict(point_coords=coords, point_labels=labels, box=framed, multimask_output=not many)
        masks = [m.astype(bool) for m in masks]
        if avoid:
            others = np.zeros(masks[0].shape, np.uint8)
            for ring in avoid:
                if ring and len(ring) >= 3:
                    cv2.fillPoly(others, [np.rint(np.asarray(ring, float) - (x0, y0)).astype(np.int32)], 1)
            masks = [m & (others == 0) for m in masks]
        if many:
            chosen = 0
        else:
            window = rgb[y0: y0 + masks[0].shape[0], x0: x0 + masks[0].shape[1]]
            dab = np.maximum(rgb_to_hed(window)[..., 2], 0).astype(np.float32) if MASK_RULES[stain] == "contrast" else None
            _, chosen = select_mask(masks, dab, stain)
        options = [(int(m.sum()), polygon(m, origin), float(s), i == chosen) for i, (m, s) in enumerate(zip(masks, scores))]
        options = sorted([o for o in options if o[1]], key=lambda o: o[0])
        return {"options": [{"points": p, "area_px": a, "score": round(s, 3)} for a, p, s, _ in options],
                "chosen": next((i for i, o in enumerate(options) if o[3]), 0)}


def _key(checkpoint: Path) -> tuple[str, int, int]:
    stat = checkpoint.stat()
    return checkpoint.name, stat.st_size, stat.st_mtime_ns  # weights put in again (even of the same size): loaded again


LOAD_RETRY_SECONDS = 30.0  # after weights that would not load, the next try waits this long (not until a restart)
_LOAD_FAILED: dict[tuple, tuple[float, str]] = {}


class NotLoaded(RuntimeError):
    """The Segment Anything weights would not load (the detail is in the log); tried again after `LOAD_RETRY_SECONDS`."""


def outliner(checkpoint: Path) -> ClickOutliner:
    """The server's one model (loaded on first use), shared by every project with the same published weights. Weights
    that would not load are tried again on a request after `LOAD_RETRY_SECONDS`, or at once when the file changes."""
    with _LOCK:
        key = _key(checkpoint)
        if key not in _MODELS:
            failed = _LOAD_FAILED.get(key)
            if failed and time.monotonic() - failed[0] < LOAD_RETRY_SECONDS:
                raise NotLoaded(failed[1])
            try:
                _MODELS[key] = ClickOutliner(checkpoint)
            except Exception as error:
                log.warning("Segment Anything weights %s would not load: %s: %s", checkpoint, type(error).__name__, error)
                _LOAD_FAILED[key] = (time.monotonic(), f"{type(error).__name__}: {error}")
                raise NotLoaded(f"{type(error).__name__}: {error}") from error
            _LOAD_FAILED.pop(key, None)
        return _MODELS[key]


_CHOSEN: list = [None, -GPU_RETRY_SECONDS]  # the device a model would be loaded on, and when that was looked at


def encodes_on_gpu(checkpoint: Path) -> bool:
    """Whether the model encodes on a GPU (or would, once loaded): without loading it on a server without one."""
    model = _MODELS.get(_key(checkpoint))
    if model is not None:
        return model.on_gpu
    if time.monotonic() - _CHOSEN[1] >= GPU_RETRY_SECONDS:
        _CHOSEN[:] = [choose_device(), time.monotonic()]
    return _CHOSEN[0] != "cpu"
