"""Per-slide DAB threshold calibration (one threshold per TMA x stain, blind to diagnosis)."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from stainid.imaging.calibration import calibrate_threshold
from stainid.imaging.color import rgb_to_hed
from stainid.imaging.core_images import open_image
from stainid.imaging.tissue import field_tissue_mask
from stainid.jobs import units
from stainid.project import Project
from stainid.tables import read_csv, write_records

Image.MAX_IMAGE_PIXELS = None
STRIDE = 64  # every 64th tissue pixel of every tile: about a million pixels per core, 60 million per slide


def sample_dab(root: Path, tiles: list[dict[str, str]], stride: int = STRIDE) -> np.ndarray:
    """Tissue DAB (optical density) of every tile of these cores, every `stride`-th tissue pixel of each tile."""
    values = []
    by_image = defaultdict(list)
    for tile in tiles:
        by_image[tile["image_path"]].append(tile)
    for image_path, image_tiles in sorted(by_image.items()):
        with open_image(root / image_path) as image:
            image.load()
            for tile in image_tiles:
                x, y = int(tile["x_px"]), int(tile["y_px"])
                rgb = np.asarray(image.crop((x, y, x + int(tile["width_px"]), y + int(tile["height_px"]))).convert("RGB"))
                tissue = field_tissue_mask(rgb)
                if tissue.any():
                    values.append(np.maximum(rgb_to_hed(rgb)[..., 2], 0)[tissue][::stride].astype(np.float32))
    return np.concatenate(values) if values else np.zeros(0, np.float32)


def calibrate_slides(project: Project, stains: list[str] | None = None, redo: bool = False, stride: int = STRIDE,
                     output: Path | None = None, cores: set[str] | None = None) -> Path:
    """Pool tissue DAB from every whole-core tile of a slide and apply `calibrate_threshold` (Otsu / median + 6 MAD /
    0.01 floor). Slides that already have a threshold keep it unless `redo`. With `cores`, only the slides of these core
    images are calibrated, from these core images' tiles (all of a slide's for the same threshold as without)."""
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
    rows = []
    for number, ((tma, stain), group) in enumerate(sorted(groups.items()), start=1):
        values = sample_dab(project.root, group, stride)
        threshold, quantiles = calibrate_threshold(values)
        rows.append({"tma": tma, "stain": stain, "threshold_dab_od": threshold, "sampled_tissue_pixels": int(values.size),
                     **{f"dab_p{q}": v for q, v in zip(("50", "75", "90", "95", "99", "999"), quantiles)}})
        write_records(output, kept + rows)  # each slide's threshold is kept as soon as it is known: a new run goes on from here
        print(f"[{number}/{len(groups)}] TMA {tma} {stain}: threshold {threshold:.4f} OD", flush=True)
        units.boundary(f"TMA {tma} {stain}")
    return output
