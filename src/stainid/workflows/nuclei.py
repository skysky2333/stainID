"""Cellpose-SAM nucleus segmentation for analysis fields (hematoxylin counterstain, any stain)."""
from __future__ import annotations

import time

import numpy as np
from PIL import Image

from stainid.nuclei import CELLPOSE_SETTINGS, cellpose_masks, load_cellpose
from stainid.pipelines.cohort_v1 import context_crop
from stainid.project import Project
from stainid.tables import read_csv

Image.MAX_IMAGE_PIXELS = None


def run_nuclei(project: Project, stains: list[str], device: str = "mps", batch_size: int = 8, threads: int = 4,
               shard_index: int = 0, shard_count: int = 1) -> None:
    project.apply_environment()
    output = project.output("nuclei")
    output.mkdir(parents=True, exist_ok=True)
    rows = [r for r in read_csv(project.input("tile_manifest")) if r["stain"] in set(stains)]
    rows = sorted(rows, key=lambda r: (r["image_path"], int(r["selection_order"])))[shard_index::shard_count]
    pending = [r for r in rows if not (output / f"{r['tile_id']}.npz").exists()]
    print(f"{len(pending)} of {len(rows)} fields pending", flush=True)
    CELLPOSE_SETTINGS["batch_size"] = batch_size
    model = load_cellpose(project.model("cellpose"), threads, device)
    image_path, image = None, None
    for index, row in enumerate(pending, start=1):
        if row["image_path"] != image_path:
            image_path = row["image_path"]
            image = Image.open(project.root / image_path)
            image.load()
        rgb, _ = context_crop(image, int(row["x_px"]), int(row["y_px"]), int(row["width_px"]), int(row["height_px"]), project.context_px)
        started = time.time()
        masks = cellpose_masks(model, rgb)
        np.savez_compressed(output / f"{row['tile_id']}.npz", masks=masks.astype(np.uint32))
        print(f"[{index}/{len(pending)}] {row['tile_id']}: {int(masks.max())} nuclei, {time.time() - started:.0f}s", flush=True)
