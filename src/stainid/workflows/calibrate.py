"""Per-slide DAB threshold calibration (one threshold per TMA x stain, blind to diagnosis)."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from stainid.imaging.calibration import calibrate_threshold
from stainid.imaging.core_images import open_image
from stainid.imaging.tissue import core_tissue_darkness
from stainid.jobs import units
from stainid.project import Project
from stainid.tables import read_csv, write_records
from stainid.v2 import params as v2_params

Image.MAX_IMAGE_PIXELS = None
STRIDE = 64  # every 64th tissue pixel of every tile: about a million pixels per core, 60 million per slide


def excluded_of(field, stain: str) -> np.ndarray:
    """The pixels the analysis leaves out of a field for this stain (`stainid.v2.field.Field`): linear artifacts and
    folds (NeuN: linear artifacts only), the same masks stage 2 finds on the same crop. None of them depends on the slide
    threshold (darkness, hematoxylin and shape only), so the threshold is calibrated on the tissue they leave."""
    return field.neun_exclusion if stain == "NeuN" else field.exclusion


def sample_core(root: str, image_path: str, tiles: list[dict], stride: int = STRIDE, context: int = 128,
                leave_out: bool = True, linear_contrast: float | None = None) -> dict[str, np.ndarray | int]:
    """One core image's share of its slide's calibration sample: every `stride`-th tissue pixel's DAB (optical density) of
    each tile's own area, cut with the context margin stage 2 uses (so the exclusion masks are the same); `values` leave
    the excluded pixels out (`leave_out`), `all` keep them (the sample as it was made before). `linear_contrast`: the
    project's `v2.proposals.linear_artifact_contrast`, so the linear-artifact mask is stage 2's."""
    from stainid.pipelines.cohort_v1 import context_crop  # noqa: PLC0415
    from stainid.v2.field import Field  # noqa: PLC0415

    kept, everything, tissue_px, excluded_px = [], [], 0, 0
    with open_image(Path(root) / image_path) as image:
        image.load()
        background = None
        if linear_contrast is not None:  # the core image's tissue darkness, as stage 2 takes it (`CoreWork.background`)
            background = core_tissue_darkness(np.asarray(image if image.mode == "RGB" else image.convert("RGB")))
        for tile in tiles:
            x, y = int(tile["x_px"]), int(tile["y_px"])
            rgb, inner = context_crop(image, x, y, int(tile["width_px"]), int(tile["height_px"]), context)
            size = float(np.sqrt(float(tile["pixel_width_um"]) * float(tile["pixel_height_um"])))
            field = Field({**tile, "context_px": context}, rgb, inner, (x - inner[1].start, y - inner[0].start), float("nan"), size,
                          np.zeros(rgb.shape[:2], bool), None, linear_contrast=linear_contrast,
                          linear_background=background)
            tissue = field.tissue[inner]
            if not tissue.any():
                continue
            dab = field.dab[inner]
            everything.append(dab[tissue][::stride].astype(np.float32))
            excluded = excluded_of(field, str(tile["stain"]))[inner] & tissue if leave_out else np.zeros_like(tissue)
            clean = tissue & ~excluded
            kept.append(dab[clean][::stride].astype(np.float32))
            tissue_px += int(tissue.sum())
            excluded_px += int(excluded.sum())
            field.release()
    empty = np.zeros(0, np.float32)
    return {"values": np.concatenate(kept) if kept else empty, "all": np.concatenate(everything) if everything else empty,
            "tissue_px": tissue_px, "excluded_px": excluded_px}


def sample_dab(root: Path, tiles: list[dict[str, str]], stride: int = STRIDE, context: int = 128, leave_out: bool = True,
               workers: int = 1, linear_contrast: float | None = None) -> dict[str, np.ndarray | int]:
    """The calibration sample of these tiles (`sample_core` per core image, `workers` core images at once): `values`
    (tissue outside the excluded areas), `all` (all tissue), and the tissue and excluded pixel counts."""
    by_image = defaultdict(list)
    for tile in tiles:
        by_image[tile["image_path"]].append(tile)
    jobs = sorted(by_image.items())
    if workers > 1 and len(jobs) > 1:
        from concurrent.futures import ProcessPoolExecutor  # noqa: PLC0415
        from multiprocessing import get_context  # noqa: PLC0415

        with ProcessPoolExecutor(min(workers, len(jobs)), mp_context=get_context("spawn")) as pool:
            parts = list(pool.map(sample_core, [str(root)] * len(jobs), [j[0] for j in jobs], [j[1] for j in jobs], [stride] * len(jobs),
                                  [context] * len(jobs), [leave_out] * len(jobs), [linear_contrast] * len(jobs)))
    else:
        parts = [sample_core(str(root), image_path, image_tiles, stride, context, leave_out, linear_contrast) for image_path, image_tiles in jobs]
    empty = np.zeros(0, np.float32)
    return {"values": np.concatenate([p["values"] for p in parts]) if parts else empty,
            "all": np.concatenate([p["all"] for p in parts]) if parts else empty,
            "tissue_px": sum(p["tissue_px"] for p in parts), "excluded_px": sum(p["excluded_px"] for p in parts)}


def bounded(project: Project, row: dict) -> dict:
    """A slide's calibration row with its threshold held to the stain's bound (`v2.calibration.max_threshold_dab_od`):
    `threshold_dab_od` is what every step uses, at most the bound; `calibrated_threshold_dab_od` keeps what the tissue
    gave (a new row: its threshold; an older row: the threshold it had). A kept threshold is only ever lowered to the
    bound, never raised (one set by hand stays); lifting a bound takes calibrating again (`--redo`)."""
    threshold = float(row["threshold_dab_od"])
    bound = v2_params.max_threshold(project, str(row["stain"]))
    calibrated = row.get("calibrated_threshold_dab_od")
    return {**row, "threshold_dab_od": threshold if bound is None else min(threshold, bound),
            "calibrated_threshold_dab_od": float(calibrated) if calibrated not in (None, "") else threshold}


def calibrate_slides(project: Project, stains: list[str] | None = None, redo: bool = False, stride: int = STRIDE,
                     output: Path | None = None, cores: set[str] | None = None, workers: int = 1) -> Path:
    """Pool tissue DAB from every whole-core tile of a slide, outside the excluded areas (linear artifacts, folds; the
    masks stage 2 uses, `excluded_of`), and apply `calibrate_threshold` (Otsu / median + 6 MAD / 0.01 floor). Slides that
    already have a threshold keep it unless `redo`. With `cores`, only the slides of these core
    images are calibrated, from these core images' tiles (all of a slide's for the same threshold as without). Each
    threshold is held to the stain's bound (`bounded`), the kept ones too."""
    output = output or project.input("calibration")
    wanted = lambda stain: stains is None or stain in stains  # noqa: E731
    kept = [r for r in (read_csv(output) if output.exists() else []) if not (redo and wanted(r["stain"]))]
    have = {(r["tma"], r["stain"]) for r in kept}
    groups = defaultdict(list)
    for tile in read_csv(project.analysis_tiles()):
        if wanted(tile["stain"]) and (tile["tma"], tile["stain"]) not in have and (cores is None or tile["core_id"] in cores):
            groups[(tile["tma"], tile["stain"])].append(tile)
    if not groups and not have:
        raise ValueError("None of these core images has tiles" if cores is not None else "No tiles to calibrate on: run “Cut cores into tiles” first")
    if have:
        print(f"{len(have)} slides already have a threshold (kept)", flush=True)
    kept = [bounded(project, r) for r in kept]  # a changed bound applies to the kept slides too, without sampling them again
    rows = []
    for number, ((tma, stain), group) in enumerate(sorted(groups.items()), start=1):
        sample = sample_dab(project.root, group, stride, project.context_px, workers=workers, linear_contrast=v2_params.linear_contrast(project))
        values = sample["values"]
        threshold, quantiles = calibrate_threshold(values)
        rows.append(bounded(project, {"tma": tma, "stain": stain, "threshold_dab_od": threshold, "sampled_tissue_pixels": int(values.size),
                                      "excluded_tissue_fraction": round(sample["excluded_px"] / max(1, sample["tissue_px"]), 6),
                                      **{f"dab_p{q}": v for q, v in zip(("50", "75", "90", "95", "99", "999"), quantiles)}}))
        threshold = float(rows[-1]["threshold_dab_od"])
        write_records(output, kept + rows)  # each slide's threshold is kept as soon as it is known: a new run goes on from here
        print(f"[{number}/{len(groups)}] TMA {tma} {stain}: threshold {threshold:.4f} OD", flush=True)
        units.boundary(f"TMA {tma} {stain}")
    if not rows:  # nothing to sample: the kept thresholds under the current bound
        write_records(output, kept)
    return output
