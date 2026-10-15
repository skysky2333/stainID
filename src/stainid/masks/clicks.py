"""Click-to-outline with Segment Anything, for people outlining objects in the browser (outline sessions).

One model per server process, on the CPU with a few threads, so it never takes the GPU or the processors from running
steps. Computing the picture's features for a 1024-px window takes a few seconds; they are kept for the last
`KEEP` windows, so further clicks in the same part of a field answer at once. Requests take turns.
"""
from __future__ import annotations

import threading
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage

from stainid.imaging.color import rgb_to_hed
from stainid.masks.objects import select_mask
from stainid.masks.sam import WindowedSam

KEEP = 12
THREADS = 4
EPSILON_PX = 1.0
MIN_PIXELS = 12

_LOCK = threading.Lock()
_MODELS: dict[str, ClickOutliner] = {}


def polygon(mask: np.ndarray, origin: tuple[int, int] = (0, 0)) -> list[list[int]] | None:
    """The outline of a mask's largest piece (holes filled), simplified, in field pixels."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if count < 2:
        return None
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[largest, cv2.CC_STAT_AREA] < MIN_PIXELS:
        return None
    piece = ndimage.binary_fill_holes(labels == largest).astype(np.uint8)
    contours, _ = cv2.findContours(piece, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = cv2.approxPolyDP(max(contours, key=cv2.contourArea), EPSILON_PX, True)[:, 0, :]
    if len(contour) < 3:
        return None
    return (contour + np.asarray(origin)).astype(int).tolist()


class ClickOutliner:
    def __init__(self, checkpoint: Path):
        import torch

        torch.set_num_threads(THREADS)
        self.sam = WindowedSam(checkpoint, device="cpu")
        self.features: OrderedDict[tuple, tuple] = OrderedDict()

    def _window(self, key: str, rgb: np.ndarray, x: float, y: float) -> tuple[int, int]:
        """Make the window around (x, y) the predictor's image, from the kept features when there are some."""
        predictor = self.sam.predictor
        self.sam.set_field(rgb)
        origin = self.sam._origin(x, y)
        kept = self.features.get((key, origin))
        if kept is None:
            x0, y0 = origin
            predictor.set_image(rgb[y0: y0 + self.sam.window, x0: x0 + self.sam.window])
            kept = (predictor.features, predictor.original_size, predictor.input_size)
            self.features[(key, origin)] = kept
            while len(self.features) > KEEP:
                self.features.popitem(last=False)
        else:
            self.features.move_to_end((key, origin))
            predictor.features, predictor.original_size, predictor.input_size = kept
            predictor.is_image_set = True
        return origin

    def outline(self, key: str, rgb: np.ndarray, stain: str, points: list[tuple[float, float, bool]]) -> dict:
        """Outlines for one object from clicks (x, y, on the object or not; the first one on it): with one click the
        three sizes Segment Anything offers (smallest first) and the one this stain's outline step would choose, with
        more clicks the one outline that fits them all."""
        x, y, _ = points[0]
        origin = self._window(key, rgb, x, y)
        x0, y0 = origin
        size = self.sam.window
        inside = [(px, py, on) for px, py, on in points if x0 <= px < x0 + size and y0 <= py < y0 + size]
        coords = np.array([[px - x0, py - y0] for px, py, _ in inside], float)
        labels = np.array([1 if on else 0 for _, _, on in inside])
        many = len(inside) > 1
        masks, scores, _ = self.sam.predictor.predict(point_coords=coords, point_labels=labels, multimask_output=not many)
        masks = [m.astype(bool) for m in masks]
        if many:
            chosen = 0
        else:
            window = rgb[y0: y0 + masks[0].shape[0], x0: x0 + masks[0].shape[1]]
            _, chosen = select_mask(masks, np.maximum(rgb_to_hed(window)[..., 2], 0).astype(np.float32), stain)
        options = [(int(m.sum()), polygon(m, origin), float(s), i == chosen) for i, (m, s) in enumerate(zip(masks, scores))]
        options = sorted([o for o in options if o[1]], key=lambda o: o[0])
        return {"options": [{"points": p, "area_px": a, "score": round(s, 3)} for a, p, s, _ in options],
                "chosen": next((i for i, o in enumerate(options) if o[3]), 0)}


def outliner(checkpoint: Path) -> ClickOutliner:
    with _LOCK:
        if str(checkpoint) not in _MODELS:
            _MODELS[str(checkpoint)] = ClickOutliner(checkpoint)
        return _MODELS[str(checkpoint)]


def outline(checkpoint: Path, key: str, rgb: np.ndarray, stain: str, points: list[tuple[float, float, bool]]) -> dict:
    """`ClickOutliner.outline` with the server's one model (loaded on first use), one request at a time."""
    model = outliner(checkpoint)
    with _LOCK:
        return model.outline(key, rgb, stain, points)
