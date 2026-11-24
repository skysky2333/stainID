"""Stage 4a, network scores: a network model's score for every candidate the stain's rules leave to the model (GPU; the
core image is read), stored per core image for stage 4 (calls), which then decides from these scores instead of a
forest on the stored features.

A network model is a bundle with `kind: network` (2026-11-20, the AT8 tau+ neuron: the benchmark's two-scale
ConvNeXt-Tiny, public/benchmark/comparison ident_ours3.py, run df3d82785f, trained on label set 4; the user: "switch the
study's AT8 calls to the network"). Its weights (`network_file`, next to the bundle) hold the training settings (`cfg`).
Inputs, exactly as in training: the candidate's 192 um surroundings at the scan's 0.2738 um/px (701 px, white beyond the
image), resampled to 384 px (0.5 um/px; antialiased bilinear), of which the central `window_um` (160 um) is the first
view, and the central 72 um at the scan's scale (263 px), of which the central 48 um is the second view, both resized to
`px` (288) and normalised as ImageNet; the score is the mean sigmoid of four flips (none, x, y, both) in bfloat16.
The bundle keeps the forest it replaced (`classifier`, `feature_names`, `fallback_threshold`) for the places that can only
score stored features (the labelling pool, legacy imports), and the subtype rule (`mature_feature`,
`mature_p90_threshold`) and duplicate radius (`nms_radius_um`) of stage 4.

Files: <outputs.v2>/network/<stain>/<core_id>.parquet (candidate_id, score), manifests/network/<stain>/<core_id>.json
(the proposals' digest, the stain's rules, the weights' sha256). Candidates the rules reject get no score.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from stainid import candidates as store
from stainid.v2 import manifest
from stainid.v2.params import stage_params

SCAN_UM = 0.2738  # the scale the network was trained at (the study's scans)
WIN_UM, WIN_PX = 192.0, 384  # the stored surroundings: 192 um at 0.5 um/px
CELL_UM = 72.0  # the stored cell crop, at the scan's scale
CELL_VIEW_UM = 48.0  # the cell view: its central 48 um
WIN_SIDE = int(round(WIN_UM / SCAN_UM))  # 701 px of the core image
CELL_PX = int(round(CELL_UM / SCAN_UM))  # 263 px
CHUNK = 256  # candidates cut and scored at once
FLIPS = ((), (3,), (2,), (2, 3))


def is_network(bundle: dict) -> bool:
    return isinstance(bundle, dict) and bundle.get("kind") == "network"


def weights_file(model) -> Path:
    return Path(model.path).parent / str(model.bundle["network_file"])


def folder(project, stain: str) -> Path:
    return store.root(project) / "network" / stain


def scores_path(project, stain: str, core_id: str) -> Path:
    return store.path(project, stain, core_id, "network")


def expected(project, stain: str, core_id: str, model, settings: dict | None = None, upstream=manifest.READ) -> dict | None:
    """What this stage's manifest of a core would be now (None: the model is no network, or the proposals are missing)."""
    if model is None or not is_network(model.bundle):
        return None
    upstream = manifest.read(project, "proposals", stain, core_id) if upstream is manifest.READ else upstream
    if upstream is None:
        return None
    rules = stage_params(project, "calls", stain) if settings is None else settings
    return manifest.make("network", {"rules": rules, "input": {"window_um": WIN_UM, "cell_um": CELL_UM, "scan_um": SCAN_UM}},
                         {"proposals": upstream["digest"], "weights": manifest.sha256(weights_file(model))})


def eligible(project, stain: str, frame: pd.DataFrame) -> np.ndarray:
    """The candidates the stain's rules leave to the model (stage 4's `pre_rules`)."""
    from stainid.v2.classify import pre_rules

    return pre_rules(stain, frame, stage_params(project, "calls", stain)) == ""


def read_scores(project, stain: str, frame: pd.DataFrame, core_id: str) -> np.ndarray:
    """The stored score of each candidate of `frame` (NaN where none is stored)."""
    scores = store.read_table(scores_path(project, stain, core_id))
    if scores.empty:
        return np.full(len(frame), np.nan)
    return frame.candidate_id.map(scores.set_index("candidate_id").score).to_numpy(float)


class Scorer:
    """The network on one device, scoring windows cut from a core image."""

    def __init__(self, model, device: str | None = None):
        import torch

        self.torch = torch
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        saved = torch.load(weights_file(model), map_location="cpu", weights_only=False)
        self.cfg = saved["cfg"]
        self.net = build(self.cfg["arch"], float(self.cfg.get("drop", 0.0)))
        self.net.load_state_dict(saved["state"])
        self.net = self.net.to(self.device).to(memory_format=torch.channels_last).eval()
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1) * 255
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1) * 255
        torch.backends.cudnn.benchmark = True

    def _view(self, src, res: float, src_um: float, um: float, px: int):
        import torch.nn.functional as F

        side = int(round(min(src_um, um) / res))
        o = (src.shape[1] - side) // 2
        x = src[:, o:o + side, o:o + side].permute(0, 3, 1, 2).float()
        x = F.interpolate(x, size=(px, px), mode="bilinear", antialias=side > px, align_corners=False)
        return ((x - self.mean) / self.std).contiguous(memory_format=self.torch.channels_last)

    def score(self, windows: np.ndarray) -> np.ndarray:
        """Scores of (n, 701, 701, 3) uint8 windows centred on the candidates."""
        import torch.nn.functional as F

        torch, out = self.torch, np.zeros(len(windows), np.float32)
        two = str(self.cfg["arch"]).startswith("two:")
        with torch.no_grad():
            for s in range(0, len(windows), 128):
                part = windows[s:s + 128]
                x = torch.from_numpy(np.ascontiguousarray(part)).to(self.device).permute(0, 3, 1, 2).float()
                win = F.interpolate(x, size=(WIN_PX, WIN_PX), mode="bilinear", antialias=True, align_corners=False)
                win = win.round().clamp(0, 255).byte().permute(0, 2, 3, 1).contiguous()
                del x
                views = [self._view(win, WIN_UM / WIN_PX, WIN_UM, float(self.cfg["window_um"]), int(self.cfg["px"]))]
                if two:
                    co = (WIN_SIDE - CELL_PX) // 2
                    cell = torch.from_numpy(np.ascontiguousarray(part[:, co:co + CELL_PX, co:co + CELL_PX])).to(self.device)
                    views.append(self._view(cell, SCAN_UM, CELL_UM, CELL_VIEW_UM, int(self.cfg["px"])))
                p = 0
                for flips in FLIPS:
                    with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                        p = p + torch.sigmoid(self.net(*[v.flip(flips) if flips else v for v in views]).float())
                out[s:s + len(part)] = (p / len(FLIPS)).cpu().numpy()
        return out


def build(arch: str, drop: float):
    """The two-scale network (the study's tau_network_search.Net), without downloading its starting weights."""
    import torch
    import torchvision.models as tv

    two = arch.startswith("two:")
    base = arch.split(":", 1)[1] if two else arch
    if base != "convnext_tiny":
        raise ValueError(f"network {arch}: only ConvNeXt-Tiny is supported here")

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.body = tv.convnext_tiny(weights=None)
            self.body.classifier[2] = torch.nn.Identity()
            self.two = two
            self.drop = torch.nn.Dropout(drop)
            self.head = torch.nn.Linear(768 * (2 if two else 1), 1)

        def forward(self, x, cell=None):
            z = self.body(x)
            if self.two:
                z = torch.cat([z, self.body(cell)], 1)
            return self.head(self.drop(z)).squeeze(1)

    return Model()


def window(image, x: float, y: float) -> np.ndarray:
    """The 701 px around (x, y) of a core image, white beyond its edge."""
    half = WIN_SIDE // 2
    x0, y0 = int(round(x)) - half, int(round(y)) - half
    out = np.full((WIN_SIDE, WIN_SIDE, 3), 255, np.uint8)
    w, h = image.size
    a0, b0, a1, b1 = max(0, x0), max(0, y0), min(w, x0 + WIN_SIDE), min(h, y0 + WIN_SIDE)
    if a1 > a0 and b1 > b0:
        part = np.asarray(image.crop((a0, b0, a1, b1)).convert("RGB"))
        out[b0 - y0: b0 - y0 + part.shape[0], a0 - x0: a0 - x0 + part.shape[1]] = part
    return out


def score_core(project, stain: str, core_id: str, tiles: list[dict], model, scorer: Scorer) -> dict:
    from stainid.imaging.core_images import open_image

    started = time.time()
    frame = store.read_table(store.path(project, stain, core_id, "proposals"))
    made = expected(project, stain, core_id, model)
    keep = eligible(project, stain, frame) if not frame.empty else np.zeros(0, bool)
    rows = frame[keep]
    scores = np.zeros(len(rows), np.float32)
    if len(rows):
        pixel = float(np.sqrt(float(tiles[0]["pixel_width_um"]) * float(tiles[0]["pixel_height_um"])))
        if abs(pixel - SCAN_UM) > 0.01 * SCAN_UM:
            raise ValueError(f"{core_id} {stain}: {pixel:.4f} um/px, the network expects {SCAN_UM} um/px")
        xy = rows[["x_core", "y_core"]].to_numpy(float)
        with open_image(project.root / tiles[0]["image_path"]) as image:
            image.load()
            for s in range(0, len(rows), CHUNK):
                windows = np.stack([window(image, x, y) for x, y in xy[s:s + CHUNK]])
                scores[s:s + CHUNK] = scorer.score(windows)
    store.write_table(folder(project, stain) / f"{core_id}.csv.gz", pd.DataFrame({"candidate_id": rows.candidate_id.to_numpy(), "score": scores}))
    seconds = time.time() - started
    manifest.write(project, "network", stain, core_id, made, scored=int(len(rows)), candidates=int(len(frame)), seconds=round(seconds, 2))
    return {"scored": int(len(rows)), "candidates": int(len(frame)), "seconds": seconds}
