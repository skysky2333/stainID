"""Per-slide DAB threshold calibration (one threshold per TMA x stain, blind to diagnosis)."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from stainid.imaging.calibration import calibrate_threshold
from stainid.imaging.color import rgb_to_hed
from stainid.imaging.tissue import field_tissue_mask
from stainid.project import Project
from stainid.tables import read_csv, write_records

Image.MAX_IMAGE_PIXELS = None


def sample_dab(tiles: list[dict[str, str]], fields_per_core: int, stride: int) -> np.ndarray:
    values = []
    by_core = defaultdict(list)
    for tile in tiles:
        by_core[tile["core_id"]].append(tile)
    for core_tiles in by_core.values():
        chosen = sorted(core_tiles, key=lambda t: int(t["selection_order"]))[:fields_per_core]
        with Image.open(chosen[0]["image_path"]) as image:
            for tile in chosen:
                x, y = int(tile["x_px"]), int(tile["y_px"])
                rgb = np.asarray(image.crop((x, y, x + int(tile["width_px"]), y + int(tile["height_px"]))).convert("RGB"))
                dab = np.maximum(rgb_to_hed(rgb)[..., 2], 0)
                values.append(dab[field_tissue_mask(rgb)][::stride])
    return np.concatenate(values)


def calibrate_slides(project: Project, stains: list[str] | None = None, fields_per_core: int = 1, stride: int = 7,
                     output: Path | None = None) -> Path:
    """Pool tissue DAB from every core of a slide and apply `calibrate_threshold` (Otsu / median + 6 MAD / 0.01 floor)."""
    output = output or project.input("calibration")
    tiles = [t for t in read_csv(project.input("tile_manifest")) if stains is None or t["stain"] in stains]
    groups = defaultdict(list)
    for tile in tiles:
        groups[(tile["tma"], tile["stain"])].append(tile)
    rows = []
    for number, ((tma, stain), group) in enumerate(sorted(groups.items()), start=1):
        values = sample_dab(group, fields_per_core, stride)
        threshold, quantiles = calibrate_threshold(values)
        rows.append({"tma": tma, "stain": stain, "threshold_dab_od": threshold, "sampled_tissue_pixels": int(values.size),
                     **{f"dab_p{q}": v for q, v in zip(("50", "75", "90", "95", "99", "999"), quantiles)}})
        print(f"[{number}/{len(groups)}] TMA {tma} {stain}: threshold {threshold:.4f} OD", flush=True)
    write_records(output, rows)
    return output
