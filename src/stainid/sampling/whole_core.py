"""Whole-core tiles: the core image cut into the same 2048 px grid the sampled fields come from, every square that
holds tissue (partial squares at the right and bottom edges included), so detection covers the whole core without
overlap. Squares that are sampled fields keep their field ID (and selection order), so results and labels line up;
the others are named `<core>_<stain>_W<row><column>` with selection order 100 and up.
"""
from __future__ import annotations

import cv2
import numpy as np

from stainid.project import Project
from stainid.qc.core_qc import tissue_mask
from stainid.tables import read_csv, write_records

MINIMUM_TISSUE = 0.10
MINIMUM_EDGE_PX = 256
KEEP = ("core_id", "tma", "core_label", "donor_id", "sample_region_id", "region", "disease_group", "technical_replicate", "stain",
        "image_path", "pixel_width_um", "pixel_height_um")


def grid(width: int, field: int) -> list[tuple[int, int]]:
    """Start and size of each grid square along one side; a last partial square is kept if it is wide enough."""
    out = [(start, field) for start in range(0, width - field + 1, field)]
    rest = width - len(out) * field
    return out + ([(len(out) * field, rest)] if rest >= MINIMUM_EDGE_PX else [])


def core_tiles(project: Project, row: dict, sampled: dict[tuple[int, int], dict], field: int) -> list[dict]:
    preview = cv2.cvtColor(cv2.imread(str(project.root / row["image_path"]), cv2.IMREAD_REDUCED_COLOR_8), cv2.COLOR_BGR2RGB)
    width, height = int(row["native_width_px"]), int(row["native_height_px"])
    tissue, _, _ = tissue_mask(preview)
    sy, sx = tissue.shape[0] / height, tissue.shape[1] / width
    out = []
    for r, (y, h) in enumerate(grid(height, field)):
        for c, (x, w) in enumerate(grid(width, field)):
            patch = tissue[int(y * sy): max(int(y * sy) + 1, int((y + h) * sy)), int(x * sx): max(int(x * sx) + 1, int((x + w) * sx))]
            fraction = float(patch.mean())
            if fraction < MINIMUM_TISSUE:
                continue
            known = sampled.get((x, y))
            out.append({"tile_id": known["tile_id"] if known else f"{row['core_id']}_{row['stain']}_W{r:02d}{c:02d}", **{k: row[k] for k in KEEP},
                        "selection_role": "whole_core_grid", "selection_order": int(known["selection_order"]) if known else 100 + r * 100 + c,
                        "nested_sample": known["nested_sample"] if known else "whole_core", "x_px": x, "y_px": y, "width_px": w, "height_px": h,
                        "preview_tissue_fraction": round(fraction, 4)})
    return out


def write_whole_core_manifest(project: Project, field_size_px: int = 2048) -> list:
    cores = [r for r in read_csv(project.input("core_images")) if r.get("tissue_status", "") != "no_tissue"]
    sampled: dict[tuple[str, str], dict[tuple[int, int], dict]] = {}
    for row in read_csv(project.input("field_manifest")):
        sampled.setdefault((row["core_id"], row["stain"]), {})[(int(row["x_px"]), int(row["y_px"]))] = row
    rows = []
    for number, row in enumerate(cores, start=1):
        rows += core_tiles(project, row, sampled.get((row["core_id"], row["stain"]), {}), field_size_px)
        print(f"[{number}/{len(cores)}] {row['core_id']} {row['stain']}: {len(rows)} tiles so far", flush=True)
    write_records(project.input("whole_core_manifest"), rows)
    counts = np.array([sum(r["core_id"] == c["core_id"] and r["stain"] == c["stain"] for r in rows) for c in cores])
    print(f"{len(rows)} tiles, {counts.mean():.1f} per core image -> {project.relative(project.input('whole_core_manifest'))}", flush=True)
    return [project.input("whole_core_manifest")]
