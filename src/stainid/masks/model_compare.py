"""Outline models compared on a small panel of candidates, each prompted exactly as stage 3 (`v2.outlines`) prompts SAM.

    python -m stainid.masks.model_compare --project ROOT --out DIR [--candidates-root ROOT/data/analysis/v2.before_v2b]
        [--size 100] [--cores-per-stain 6] [--models sam_vit_b,sam2.1_hiera_b+,pathosam_vit_b]

Methods: Segment Anything ViT-B (the current weights), SAM 2.1 Hiera base+, PathoSAM ViT-B (the generalist
histopathology weights) and the stain outline (the baseline). The panel, per stain: every candidate whose outline people
changed in Fix the field (`labels.outline_fixes`: precise edits, splits, merges) plus a fixed random sample (seed 0) from
a few core images of different TMAs, stratified by size (terciles) and accepted vs rejected, to make up about `--size`.

Each candidate is prompted as stage 3 does: one positive point at the candidate in the 1024-px window nearest to it in
its field crop (`project.context_px` of context), all three outputs decoded, the default chosen with stage 3's rule
(`v2.outlines.choose`: with no valid output stage 3 keeps the stain outline), the outputs turned into stage 3's polygons.

Results in DIR: panel.csv; objects.csv (per candidate and method: area, ratio and IoU to the stain outline, IoU with the
human outline where one exists, the stage-3 status; and how well the SAM ViT-B re-run reproduces the stored stage-3
outline: the prompting check); summary.csv, truth.csv, pairwise.csv, timing.csv (s per object, encoder s per window,
peak GPU memory; fp16 encoder); grid_<stain>_NN.png (one row per candidate: raw | stain | SAM | SAM 2 | PathoSAM);
summary.md; manifest.json (weights: URL, SHA256). Weights are loaded as tensors only (`torch.load(weights_only=True)`).
Progress goes to $STAINID_PROGRESS_FILE ("done/total" candidates x models). Nothing here changes a default.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

MODELS = {
    "sam_vit_b": {"kind": "sam1", "arch": "vit_b", "file": "sam/sam_vit_b_01ec64.pth", "params": "91M",
                  "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth", "name": "SAM ViT-B (current)"},
    "sam2.1_hiera_b+": {"kind": "sam2", "config": "configs/sam2.1/sam2.1_hiera_b+.yaml", "file": "sam2/sam2.1_hiera_base_plus.pt", "params": "81M",
                        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt", "name": "SAM 2.1 Hiera base+"},
    "pathosam_vit_b": {"kind": "sam1", "arch": "vit_b", "file": "pathosam/vit_b_histopathology.pt", "params": "91M",
                       "url": "https://owncloud.gwdg.de/index.php/s/sBB4H8CTmIoBZsQ/download", "name": "PathoSAM ViT-B (generalist)"},
}
SIZES = ("small", "medium", "large")
PROGRESS_ENV = "STAINID_PROGRESS_FILE"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


# ---------------------------------------------------------------- models (stage 3's interface: encode a window, decode points)

def _state_dict(path: Path) -> dict:
    """The weights as tensors only: `weights_only=True` refuses anything else in the file."""
    import torch

    found = torch.load(path, map_location="cpu", weights_only=True)
    for key in ("model_state", "model", "state_dict"):
        if isinstance(found, dict) and key in found and isinstance(found[key], dict):
            found = found[key]
    state = {k.removeprefix("sam."): v for k, v in found.items() if hasattr(v, "shape")}
    return state


class Sam1Outliner:
    """SAM (ViT-B/L, Meta's or PathoSAM's weights) through stage 3's own `BatchOutliner.decode` (fp16 encoder)."""

    def __init__(self, spec: dict, path: Path, device: str):
        import torch
        from segment_anything import sam_model_registry
        from segment_anything.utils.transforms import ResizeLongestSide

        from stainid.v2.outlines import BatchOutliner

        model = sam_model_registry[spec["arch"]](checkpoint=None)
        missing, unexpected = model.load_state_dict(_state_dict(path), strict=False)
        if missing:
            raise ValueError(f"{path.name}: {len(missing)} weights missing (e.g. {missing[:3]}): not a SAM {spec['arch']} checkpoint")
        self.unexpected = list(unexpected)
        inner = BatchOutliner.__new__(BatchOutliner)
        inner.torch, inner.device, inner.model = torch, torch.device(device), model
        inner.half = inner.device.type == "cuda"
        if inner.half:
            model.image_encoder.half()
        model.to(inner.device).eval()
        inner.transform = ResizeLongestSide(model.image_encoder.img_size)
        self.inner = inner

    def encode(self, window: np.ndarray):
        return self.inner.encode(window)

    def decode(self, embedding, points: np.ndarray):
        return self.inner.decode(embedding, points)


class Sam2Outliner:
    """SAM 2.1 on single images: the image encoder under fp16 autocast, the same point prompt, three outputs per point
    (as `Output`, in the model's order)."""

    def __init__(self, spec: dict, path: Path, device: str):
        import torch
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        self.torch = torch
        model = build_sam2(spec["config"], ckpt_path=None, device="cpu")
        missing, _ = model.load_state_dict(_state_dict(path), strict=False)
        if missing:
            raise ValueError(f"{path.name}: {len(missing)} weights missing: not a SAM 2.1 checkpoint for {spec['config']}")
        model.to(device).eval()
        self.predictor = SAM2ImagePredictor(model)
        self.device = torch.device(device)

    def _autocast(self):
        return self.torch.autocast("cuda", dtype=self.torch.float16) if self.device.type == "cuda" else self.torch.autocast("cpu", enabled=False)

    def encode(self, window: np.ndarray):
        with self.torch.inference_mode(), self._autocast():
            self.predictor.set_image(window)
        return {"features": self.predictor._features, "hw": self.predictor._orig_hw}

    def decode(self, embedding, points: np.ndarray):
        from segment_anything.utils.amg import calculate_stability_score

        from stainid.v2.outlines import BATCH, Output

        torch, predictor = self.torch, self.predictor
        predictor._features, predictor._orig_hw, predictor._is_image_set = embedding["features"], embedding["hw"], True
        out = []
        for start in range(0, len(points), BATCH):
            chunk = points[start: start + BATCH].astype(np.float32)
            with torch.inference_mode(), self._autocast():
                _, coords, labels, _ = predictor._prep_prompts(chunk[:, None, :], np.ones((len(chunk), 1), np.int32), None, None, normalize_coords=True)
                logits, iou, _ = predictor._predict(coords, labels, None, None, multimask_output=True, return_logits=True)
            logits = logits.float()
            stability = calculate_stability_score(logits, predictor.mask_threshold, 1.0).cpu().numpy()
            masks = logits > predictor.mask_threshold
            area = masks.sum(dim=(2, 3)).cpu().numpy()
            edge = (masks[:, :, 0, :].any(-1) | masks[:, :, -1, :].any(-1) | masks[:, :, :, 0].any(-1) | masks[:, :, :, -1].any(-1)).cpu().numpy()
            rows, cols = masks.any(dim=3).cpu().numpy(), masks.any(dim=2).cpu().numpy()
            iou = iou.float().cpu().numpy()
            for b in range(len(chunk)):
                outputs = []
                for k in range(masks.shape[1]):
                    ys, xs = np.flatnonzero(rows[b, k]), np.flatnonzero(cols[b, k])
                    if not len(ys):
                        outputs.append(Output(0, False, (0, 0, 0, 0), np.zeros((0, 0), bool), float(iou[b, k]), float(stability[b, k])))
                        continue
                    y0, y1 = max(0, ys[0] - 1), min(masks.shape[2], ys[-1] + 2)
                    x0, x1 = max(0, xs[0] - 1), min(masks.shape[3], xs[-1] + 2)
                    outputs.append(Output(int(area[b, k]), bool(edge[b, k]), (int(x0), int(y0), int(x1), int(y1)),
                                          masks[b, k, y0:y1, x0:x1].cpu().numpy(), float(iou[b, k]), float(stability[b, k])))
                out.append(outputs)
        return out


def load_model(name: str, models_dir: Path, device: str):
    spec = MODELS[name]
    path = models_dir / spec["file"]
    if not path.exists():
        raise FileNotFoundError(f"{name}: {path} is not there (see the comparison's downloads)")
    return (Sam2Outliner if spec["kind"] == "sam2" else Sam1Outliner)(spec, path, device)


# ---------------------------------------------------------------- the panel

PANEL_COLUMNS = ["candidate_id", "core_id", "tile_id", "x_core", "y_core", "area_um2", "outline", "decision", "probability", "sam_outline", "sam_status"]


def _candidates(project, stain: str, core: str) -> pd.DataFrame:
    """A core image's candidates with their calls (from the calls table when the candidates table has none)."""
    from stainid import candidates

    frame = candidates.read_table(candidates.path(project, stain, core), PANEL_COLUMNS).reindex(columns=PANEL_COLUMNS)
    if frame.decision.isna().all():
        calls = candidates.read_table(candidates.path(project, stain, core, "calls"), ["candidate_id", "decision", "probability"])
        if len(calls):
            frame = frame.drop(columns=["decision", "probability"]).merge(calls, on="candidate_id", how="left").reindex(columns=PANEL_COLUMNS)
    return frame


def _tma_of(tiles: pd.DataFrame) -> dict[str, str]:
    return dict(zip(tiles.core_id.astype(str), tiles.tma.astype(str)))


def fixed_objects(project, tiles: pd.DataFrame, stain: str) -> list[dict]:
    """Objects whose outline people changed in Fix the field (`labels.outline_fixes`): candidate ids and their outlines."""
    from stainid.labels import outline_fixes

    try:
        objects, _ = outline_fixes._fixed(project, stain, tiles)
    except Exception as error:  # noqa: BLE001 - no fixes readable: the panel is the sample alone
        print(f"{stain}: fixed outlines not read ({error})", flush=True)
        return []
    return objects


def panel(project, tiles: pd.DataFrame, stain: str, size: int, cores_per_stain: int, seed: int = 0) -> pd.DataFrame:
    """The stain's panel: every candidate with a human-changed outline, plus a fixed random sample (seed) from a few core
    images of different TMAs, stratified by size (the stain's terciles in those cores) and accepted vs rejected, to make
    up about `size`."""
    from stainid import candidates

    rng = np.random.default_rng(seed)
    tma = _tma_of(tiles)
    known = set(tiles.tile_id.astype(str))
    rows = []
    fixed = fixed_objects(project, tiles, stain)
    by_core: dict[str, list] = {}
    for o in fixed:
        for i, cid in enumerate(o["candidates"]):
            by_core.setdefault(candidates.parse_id(cid)[1], []).append((cid, o, i))
    for core, entries in by_core.items():
        frame = _candidates(project, stain, core)
        for cid, o, i in entries:
            row = frame[frame.candidate_id == cid]
            if len(row) and str(row.tile_id.iloc[0]) in known:
                rows.append(row.iloc[0].to_dict() | {"group": f"fixed ({o['kind']})", "truth": candidates.encode_rings(o["drawn"]) if o["drawn"] else "",
                                                     "person": o["person"]})
    cores = sorted(candidates.cores(project, stain))
    by_tma: dict[str, list[str]] = {}
    for core in cores:
        by_tma.setdefault(tma.get(core, "?"), []).append(core)
    order = list(by_tma)
    rng.shuffle(order)
    pool = []
    for name in order:
        options = list(by_tma[name])
        rng.shuffle(options)
        for core in options[:40]:
            frame = _candidates(project, stain, core)
            frame = frame[frame.tile_id.astype(str).isin(known) & frame.decision.isin(["accepted", "rejected"])]
            if frame.decision.nunique() == 2 and len(frame) >= 20:
                pool.append(frame)
                break
        if len(pool) >= cores_per_stain:
            break
    pool = pd.concat(pool, ignore_index=True) if pool else pd.DataFrame(columns=PANEL_COLUMNS)
    pool = pool[~pool.candidate_id.isin({r["candidate_id"] for r in rows})]
    want = max(0, size - len(rows))
    if want and len(pool):
        terciles = np.quantile(pool.area_um2.astype(float), [1 / 3, 2 / 3])
        pool = pool.assign(size_class=np.select([pool.area_um2 < terciles[0], pool.area_um2 < terciles[1]], ["small", "medium"], "large"))
        strata = [(s, d) for s in ("small", "medium", "large") for d in ("accepted", "rejected")]
        groups = {k: g.sample(frac=1, random_state=seed) for k, g in pool.groupby(["size_class", "decision"])}
        taken = {k: 0 for k in strata}
        picked = []
        while len(picked) < want and any(taken[k] < len(groups.get(k, ())) for k in strata):
            for k in strata:
                if len(picked) >= want:
                    break
                group = groups.get(k)
                if group is not None and taken[k] < len(group):
                    picked.append(group.iloc[taken[k]].to_dict() | {"group": f"sample ({k[0]}, {k[1]})", "truth": "", "person": ""})
                    taken[k] += 1
        rows += picked
    out = pd.DataFrame(rows)
    out["stain"] = stain
    out["tma"] = out.core_id.astype(str).map(tma)
    return out


# ---------------------------------------------------------------- running a model over the panel, as stage 3 does

def run_model(name: str, outliner, project, tiles: pd.DataFrame, frame: pd.DataFrame, progress=None) -> tuple[dict, dict]:
    """Per candidate id: the three polygons (small → large, core px), the stage-3 default's index among them (-1: stage 3
    keeps the stain outline) and its status; with the time spent and the peak GPU memory."""
    import torch

    from stainid.imaging.color import rgb_to_hed
    from stainid.imaging.core_images import open_image
    from stainid.masks.sam import window_origin
    from stainid.pipelines.cohort_v1 import context_crop
    from stainid.v2.outlines import _polygon, choose
    from stainid.v2.params import params

    settings = params(project)["outlines"]
    by_tile = tiles.drop_duplicates("tile_id").set_index("tile_id")
    results, encode_s, decode_s, windows = {}, 0.0, 0.0, 0
    cuda = torch.cuda.is_available() and outliner_device(outliner) == "cuda"
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    for (stain, core_id), group in frame.groupby(["stain", "core_id"], sort=True):
        first = by_tile.loc[group.tile_id.iloc[0]]
        with open_image(project.root / first["image_path"]) as image:
            image.load()
            for tile_id, rows in group.groupby("tile_id"):
                tile = by_tile.loc[tile_id]
                x, y = int(tile["x_px"]), int(tile["y_px"])
                rgb, inner = context_crop(image, x, y, int(tile["width_px"]), int(tile["height_px"]), project.context_px)
                pixel = float(np.sqrt(float(tile["pixel_width_um"]) * float(tile["pixel_height_um"])))
                fx, fy = x - inner[1].start, y - inner[0].start
                dab = np.maximum(rgb_to_hed(rgb)[..., 2], 0).astype(np.float32)
                xs, ys = rows.x_core.to_numpy(float) - fx, rows.y_core.to_numpy(float) - fy
                origins = [window_origin(rgb.shape[:2], a, b) for a, b in zip(xs, ys)]
                for origin in sorted(set(origins)):
                    index = [i for i, w in enumerate(origins) if w == origin]
                    x0, y0 = origin
                    window = np.ascontiguousarray(rgb[y0: y0 + 1024, x0: x0 + 1024])
                    if cuda:
                        torch.cuda.synchronize()
                    started = time.perf_counter()
                    embedding = outliner.encode(window)
                    if cuda:
                        torch.cuda.synchronize()
                    middle = time.perf_counter()
                    decoded = outliner.decode(embedding, np.column_stack((xs[index] - x0, ys[index] - y0)))
                    if cuda:
                        torch.cuda.synchronize()
                    encode_s += middle - started
                    decode_s += time.perf_counter() - middle
                    windows += 1
                    local_dab = dab[y0: y0 + window.shape[0], x0: x0 + window.shape[1]]
                    for i, outputs in zip(index, decoded):
                        row = rows.iloc[i]
                        chosen, status = choose(outputs, float(row.area_um2) / pixel ** 2, stain, pixel, local_dab, float(settings["runaway_factor"]),
                                                float(settings.get("shrink_factor", 0.0)), float(settings.get("shrink_min_um2", np.inf)))
                        polygons = [_polygon(o, (fx + x0, fy + y0)) for o in outputs]
                        order = list(np.argsort([o.area for o in outputs], kind="stable"))
                        results[row.candidate_id] = {"polygons": [polygons[j] for j in order], "default": order.index(chosen) if chosen >= 0 else -1,
                                                     "status": status, "pixel_um": pixel}
                    if progress is not None:
                        progress(len(index))
    timing = {"model": name, "objects": len(results), "windows": windows,
              "s_per_object": round((encode_s + decode_s) / len(results), 4) if results else None,
              "encode_s_per_window": round(encode_s / windows, 4) if windows else None,
              "decode_s_per_object": round(decode_s / len(results), 4) if results else None,
              "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2) if cuda else None}
    return results, timing


def outliner_device(outliner) -> str:
    inner = getattr(outliner, "inner", None)
    device = inner.device if inner is not None else outliner.device
    return device.type


# ---------------------------------------------------------------- scoring

def _rings(text) -> list:
    from stainid import candidates

    if not isinstance(text, str) or not text:
        return []
    return [r for r in candidates.rings(text) if len(r) >= 6]


def _area_px(rings: list) -> float:
    return float(sum(abs(cv2.contourArea(np.asarray(r, np.float32).reshape(-1, 2))) for r in rings))


def _iou(a: list, b: list) -> float | None:
    from stainid.labels.outline_fixes import overlap

    got = overlap(a, b) if a and b else None
    return got[0] if got else None


def outlines_of(frame: pd.DataFrame, found: dict[str, dict]) -> dict[str, dict[str, list]]:
    """Per candidate, each method's outline (rings, core px): the stain outline, and per model the stage-3 default (the
    stain outline where stage 3 would fall back to it) and its three sizes."""
    out = {}
    for row in frame.itertuples():
        stain_rings = _rings(row.outline)
        entry = {"stain outline": stain_rings}
        for model, results in found.items():
            got = results.get(row.candidate_id)
            if got is None:
                continue
            entry[model] = [got["polygons"][got["default"]]] if got["default"] >= 0 and len(got["polygons"][got["default"]]) >= 6 else stain_rings
            for k, size in enumerate(SIZES):
                entry[f"{model} {size}"] = [got["polygons"][k]] if len(got["polygons"][k]) >= 6 else []
        out[row.candidate_id] = entry
    return out


def score(frame: pd.DataFrame, found: dict[str, dict]) -> pd.DataFrame:
    """One row per candidate: per method its area (µm²), area ratio to the stain outline, IoU with the stain outline,
    with the human outline (where one exists) and, for SAM ViT-B, with the outline stage 3 stored (the prompting check);
    per model the stage-3 status and the best of its three sizes against the human outline."""
    outlines = outlines_of(frame, found)
    rows = []
    for row in frame.itertuples():
        entry = outlines[row.candidate_id]
        pixel = next((r["pixel_um"] for r in (res.get(row.candidate_id) for res in found.values()) if r), None) or 0.0
        truth = _rings(row.truth)
        stain_area = _area_px(entry["stain outline"])
        out = {"stain": row.stain, "candidate_id": row.candidate_id, "core_id": row.core_id, "tma": row.tma, "group": row.group, "decision": row.decision,
               "probability": row.probability, "has_truth": bool(truth), "stain_area_um2": round(stain_area * pixel ** 2, 2)}
        if truth:
            out["iou_truth_stain outline"] = _iou(truth, entry["stain outline"])
            out["ratio_truth_stain outline"] = round(stain_area / _area_px(truth), 3) if _area_px(truth) else None
        for model in found:
            if model not in entry:
                continue
            got = found[model][row.candidate_id]
            area = _area_px(entry[model])
            out[f"{model}_status"] = got["status"]
            out[f"{model}_area_um2"] = round(area * pixel ** 2, 2)
            out[f"{model}_ratio_to_stain"] = round(area / stain_area, 3) if stain_area else None
            out[f"{model}_iou_stain"] = _iou(entry[model], entry["stain outline"])
            if truth:
                out[f"iou_truth_{model}"] = _iou(truth, entry[model])
                out[f"ratio_truth_{model}"] = round(area / _area_px(truth), 3) if _area_px(truth) else None
                sizes = [_iou(truth, entry[f"{model} {s}"]) for s in SIZES]
                out[f"iou_truth_{model} best of three"] = max((v for v in sizes if v is not None), default=None)
        if "sam_vit_b" in entry and isinstance(row.sam_outline, str) and row.sam_outline:
            out["sam_vit_b_iou_stored_stage3"] = _iou(entry["sam_vit_b"], _rings(row.sam_outline) or [])
        rows.append(out)
    return pd.DataFrame(rows)


def pairwise(frame: pd.DataFrame, found: dict[str, dict]) -> pd.DataFrame:
    """Per stain and pair of methods: the median IoU of their outlines and the median area ratio (second / first)."""
    outlines = outlines_of(frame, found)
    methods = ["stain outline", *found]
    rows = []
    for stain, part in frame.groupby("stain"):
        for i, a in enumerate(methods):
            for b in methods[i + 1:]:
                ious, ratios = [], []
                for cid in part.candidate_id:
                    ea, eb = outlines[cid].get(a), outlines[cid].get(b)
                    if not ea or not eb:
                        continue
                    v = _iou(ea, eb)
                    if v is not None:
                        ious.append(v)
                    if _area_px(ea):
                        ratios.append(_area_px(eb) / _area_px(ea))
                rows.append({"stain": stain, "a": a, "b": b, "n": len(ious), "iou_median": round(float(np.median(ious)), 3) if ious else None,
                             "iou_mean": round(float(np.mean(ious)), 3) if ious else None,
                             "area_ratio_b_over_a_median": round(float(np.median(ratios)), 3) if ratios else None})
    return pd.DataFrame(rows)


def summarise(scored: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """Per stain, decision and method: objects, median IoU with the stain outline, median area ratio to it, the ratio of
    summed areas (what a positive-area measure would change by), stage-3 fallbacks; and against the human outlines."""
    rows = []
    parts = [(stain, decision, part) for stain, whole in scored.groupby("stain")
             for decision, part in [("all", whole), *whole.groupby("decision")]]
    for stain, decision, part in parts:
        base = {"stain": stain, "decision": decision, "objects": len(part)}
        rows.append({**base, "method": "stain outline", "area_um2_median": round(float(part.stain_area_um2.median()), 1)})
        for model in models:
            if f"{model}_area_um2" not in part:
                continue
            total_stain = part.stain_area_um2.sum()
            rows.append({**base, "method": model, "area_um2_median": round(float(part[f"{model}_area_um2"].median()), 1),
                         "iou_with_stain_median": round(float(part[f"{model}_iou_stain"].median()), 3),
                         "area_ratio_to_stain_median": round(float(part[f"{model}_ratio_to_stain"].median()), 3),
                         "summed_area_ratio_to_stain": round(float(part[f"{model}_area_um2"].sum() / total_stain), 3) if total_stain else None,
                         "stage3_fallback_share": round(float(part[f"{model}_status"].astype(str).str.startswith("fallback").mean()), 3)})
    return pd.DataFrame(rows)


def truth_table(scored: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    rows = []
    with_truth = scored[scored.has_truth] if "has_truth" in scored else scored.iloc[:0]
    for stain, part in with_truth.groupby("stain"):
        for method in ["stain outline", *models, *(f"{m} best of three" for m in models)]:
            column = f"iou_truth_{method}"
            if column not in part:
                continue
            iou = pd.to_numeric(part[column], errors="coerce").dropna()
            ratio = pd.to_numeric(part.get(f"ratio_truth_{method}", pd.Series(dtype=float)), errors="coerce").dropna()
            rows.append({"stain": stain, "method": method, "objects": len(iou), "iou_median": round(float(iou.median()), 3) if len(iou) else None,
                         "iou_mean": round(float(iou.mean()), 3) if len(iou) else None,
                         "area_ratio_median": round(float(ratio.median()), 3) if len(ratio) else None})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- the picture grid

GRID_COLOURS = {"stain outline": (255, 230, 0), "sam_vit_b": (0, 255, 255), "sam2.1_hiera_b+": (255, 0, 255), "pathosam_vit_b": (60, 255, 60)}
CELL = 192
LABEL_W = 230
ROWS_PER_PAGE = 25


def _pixel_um(found: dict[str, dict], candidate_id: str) -> float:
    return next((r[candidate_id]["pixel_um"] for r in found.values() if candidate_id in r), 0.0)


def grid(project, tiles: pd.DataFrame, frame: pd.DataFrame, found: dict[str, dict], scored: pd.DataFrame, out: Path) -> list[Path]:
    """Per stain, pages of one row per candidate: raw | stain outline | each model's stage-3 default (the human outline,
    where one exists, dashed in white on the raw picture)."""
    from PIL import Image, ImageDraw

    from stainid.imaging.core_images import open_image

    outlines = outlines_of(frame, found)
    methods = ["stain outline", *found]
    names = {"stain outline": "stain", "sam_vit_b": "SAM ViT-B", "sam2.1_hiera_b+": "SAM 2.1 b+", "pathosam_vit_b": "PathoSAM ViT-B"}
    by_tile = tiles.drop_duplicates("tile_id").set_index("tile_id")
    info = scored.set_index("candidate_id")
    pictures: dict[str, Image.Image] = {}
    boxes: dict[str, tuple[int, int, int]] = {}
    for core_id, part in frame.groupby("core_id"):
        with open_image(project.root / by_tile.loc[part.tile_id.iloc[0]]["image_path"]) as image:
            for row in part.itertuples():
                rings = [r for m in methods for r in outlines[row.candidate_id].get(m, [])] + _rings(row.truth)
                points = np.vstack([np.asarray(r, float).reshape(-1, 2) for r in rings]) if rings else np.array([[row.x_core, row.y_core]])
                cx, cy = float(row.x_core), float(row.y_core)
                half = int(max(32, 1.25 * np.abs(points - (cx, cy)).max()))
                half = min(half, 600)
                x0, y0 = int(cx) - half, int(cy) - half
                pictures[row.candidate_id] = image.crop((x0, y0, x0 + 2 * half, y0 + 2 * half)).convert("RGB")
                boxes[row.candidate_id] = (x0, y0, 2 * half)
    written = []
    header = 22
    for stain, part in frame.groupby("stain"):
        part = part.reset_index(drop=True)
        for page in range(0, len(part), ROWS_PER_PAGE):
            chunk = part.iloc[page: page + ROWS_PER_PAGE]
            sheet = Image.new("RGB", (LABEL_W + CELL * (1 + len(methods)), header + CELL * len(chunk)), (24, 24, 24))
            draw = ImageDraw.Draw(sheet)
            for c, title in enumerate(["raw", *(names.get(m, m) for m in methods)]):
                draw.text((LABEL_W + c * CELL + 6, 4), title, fill=(230, 230, 230))
            for r, row in enumerate(chunk.itertuples()):
                top = header + r * CELL
                x0, y0, side = boxes[row.candidate_id]
                scale = CELL / side
                picture = pictures[row.candidate_id].resize((CELL, CELL), Image.LANCZOS)
                for c in range(1 + len(methods)):
                    cell = picture.copy()
                    pen = ImageDraw.Draw(cell)
                    rings = _rings(row.truth) if c == 0 else outlines[row.candidate_id].get(methods[c - 1], [])
                    colour = (255, 255, 255) if c == 0 else GRID_COLOURS.get(methods[c - 1], (255, 128, 0))
                    for ring in rings:
                        xy = [((x - x0) * scale, (y - y0) * scale) for x, y in np.asarray(ring, float).reshape(-1, 2)]
                        pen.line(xy + xy[:1], fill=colour, width=2)
                    if c == 0:
                        px, py = (row.x_core - x0) * scale, (row.y_core - y0) * scale
                        pen.line([(px - 4, py), (px + 4, py)], fill=(255, 60, 60), width=1)
                        pen.line([(px, py - 4), (px, py + 4)], fill=(255, 60, 60), width=1)
                    sheet.paste(cell, (LABEL_W + c * CELL, top))
                s = info.loc[row.candidate_id]
                lines = [row.candidate_id.split(":", 1)[1][:30], f"{row.group}", f"{row.decision}  p={float(row.probability):.2f}" if row.probability == row.probability
                         else str(row.decision), f"stain {s.stain_area_um2:.0f} um2", f"crop {side * _pixel_um(found, row.candidate_id):.0f} um"]
                for m in found:
                    if f"{m}_ratio_to_stain" in s:
                        ratio = s[f"{m}_ratio_to_stain"]
                        status = str(s[f"{m}_status"])
                        lines.append(f"{names.get(m, m)[:10]} x{ratio:.2f}" + (" fb" if status.startswith("fallback") else ""))
                draw.multiline_text((6, top + 6), "\n".join(lines), fill=(220, 220, 220), spacing=3)
            path = out / f"grid_{stain}_{page // ROWS_PER_PAGE + 1:02d}.png"
            sheet.save(path, optimize=True)
            written.append(path)
    return written


# ---------------------------------------------------------------- report

def _table(frame: pd.DataFrame) -> list[str]:
    if frame is None or not len(frame):
        return ["(none)", ""]
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for row in frame.itertuples(index=False):
        lines.append("| " + " | ".join("" if v is None or (isinstance(v, float) and v != v) else str(v) for v in row) + " |")
    return lines + [""]


def markdown(counts: dict, summary: pd.DataFrame, truth: pd.DataFrame, pairs: pd.DataFrame, timing: pd.DataFrame, check: dict, grids: list[Path]) -> str:
    lines = ["# Outline models compared on a candidate panel", "", "Panel: " + json.dumps(counts), "",
             "Prompting check (SAM ViT-B re-run vs the outline stage 3 stored, median IoU): " + json.dumps(check), "",
             "## Against the human outlines (Fix the field)", "", *_table(truth),
             "## Each method against the stain outline (stage-3 default; the stain outline where stage 3 falls back)", "", *_table(summary),
             "## Pairwise agreement (median IoU; area ratio b / a)", "", *_table(pairs),
             "## Time and GPU memory (fp16 encoder; one window encoded per candidate's field window)", "", *_table(timing),
             "## Picture grids", "", *(f"- {p}" for p in grids), ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    import torch

    from stainid import candidates
    from stainid.api import state

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--project", required=True)
    parser.add_argument("--candidates-root", default=None, help="the v2 folder to read candidates from (e.g. data/analysis/v2.before_v2b)")
    parser.add_argument("--stains", default="6E10,AT8,NeuN")
    parser.add_argument("--size", type=int, default=100, help="candidates per stain (the human-changed ones first)")
    parser.add_argument("--cores-per-stain", type=int, default=6, help="core images (each from another TMA) the sample is taken from")
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--models-dir", default=None, help="default: the project's data/models")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threads", type=int, default=2, help="processor threads (the queue's budget)")
    args = parser.parse_args(argv)

    torch.set_num_threads(args.threads)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.candidates_root:
        fixed_root = Path(args.candidates_root)
        candidates.root = lambda project: fixed_root  # read-only: this tool never writes candidates
    loaded = state._load(args.project)
    project, tiles = loaded.project, loaded.tiles
    models_dir = Path(args.models_dir) if args.models_dir else project.root / "data" / "models"
    names = [m for m in args.models.split(",") if m]
    frames = [panel(project, tiles, stain, args.size, args.cores_per_stain) for stain in args.stains.split(",")]
    frame = pd.concat([f for f in frames if len(f)], ignore_index=True)
    frame.drop(columns=[c for c in ("sam_outline_x",) if c in frame]).to_csv(out / "panel.csv", index=False)
    counts = {stain: {"objects": int((frame.stain == stain).sum()), "fixed": int(frame[frame.stain == stain].group.str.startswith("fixed").sum()),
                      "cores": sorted(frame[frame.stain == stain].core_id.unique().tolist()),
                      "groups": frame[frame.stain == stain].group.value_counts().to_dict()} for stain in args.stains.split(",")}
    print("panel: " + json.dumps(counts), flush=True)

    total = len(frame) * len(names)
    done = [0]
    progress_file = os.environ.get(PROGRESS_ENV)

    def progress(k: int) -> None:
        done[0] += k
        if progress_file:
            with open(progress_file, "a") as handle:
                handle.write(f"{done[0]}/{total}\n")

    manifest = {"models": {}, "device": args.device, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "project": str(project.root), "candidates_root": str(candidates.root(project)), "counts": counts,
                "started": time.strftime("%Y-%m-%d %H:%M:%S"), "torch": torch.__version__}
    found, timings = {}, []
    for name in names:
        spec = MODELS[name]
        path = models_dir / spec["file"]
        manifest["models"][name] = {**spec, "path": str(path), "sha256": sha256(path) if path.exists() else None}
        try:
            outliner = load_model(name, models_dir, args.device)
        except Exception as error:  # noqa: BLE001 - one model that cannot be loaded does not stop the others
            manifest["models"][name]["error"] = f"{type(error).__name__}: {error}"
            print(f"{name}: not compared ({error})", flush=True)
            continue
        run_model(name, outliner, project, tiles, frame.head(2))  # warm-up (cuDNN autotuning, first allocations): not timed
        found[name], timing = run_model(name, outliner, project, tiles, frame, progress)
        timings.append(timing)
        print(f"{name}: {timing}", flush=True)
        del outliner
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    scored = score(frame, found)
    scored.to_csv(out / "objects.csv", index=False)
    summary = summarise(scored, list(found))
    summary.to_csv(out / "summary.csv", index=False)
    truth = truth_table(scored, list(found))
    truth.to_csv(out / "truth.csv", index=False)
    pairs = pairwise(frame, found)
    pairs.to_csv(out / "pairwise.csv", index=False)
    timing = pd.DataFrame(timings)
    timing.to_csv(out / "timing.csv", index=False)
    check = {}
    if "sam_vit_b_iou_stored_stage3" in scored:
        for stain, part in scored.groupby("stain"):
            values = pd.to_numeric(part.sam_vit_b_iou_stored_stage3, errors="coerce").dropna()
            check[stain] = round(float(values.median()), 3) if len(values) else None
    grids = grid(project, tiles, frame, found, scored, out)
    (out / "summary.md").write_text(markdown(counts, summary, truth, pairs, timing, check, grids))
    manifest["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
    print(f"written: {out}", flush=True)


if __name__ == "__main__":
    main()
