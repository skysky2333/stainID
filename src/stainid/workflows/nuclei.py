"""Cellpose-SAM nucleus segmentation for analysis fields (hematoxylin counterstain, any stain)."""
from __future__ import annotations

import itertools
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

from stainid.imaging.core_images import open_image
from stainid.nuclei import CELLPOSE_SETTINGS, cellpose_masks, load_cellpose
from stainid.pipelines.cohort_v1 import context_crop
from stainid.project import Project
from stainid.tables import file_names, read_csv, save_npz_atomic

Image.MAX_IMAGE_PIXELS = None


class CoreImages:
    """Core images for the tiles in flight: each is decoded once (tiles come in image order) and the two most recent kept."""

    def __init__(self):
        self.lock, self.loaded = threading.Lock(), {}

    def get(self, path: Path) -> Image.Image:
        with self.lock:
            if path not in self.loaded:
                image = open_image(path)
                image.load()
                self.loaded = {**dict(list(self.loaded.items())[-1:]), path: image}
            return self.loaded[path]


def run_nuclei(project: Project, stains: list[str], device: str = "mps", batch_size: int = 8, threads: int = 4,
               shard_index: int = 0, shard_count: int = 1, workers: int = 2, cores: set[str] | None = None) -> None:
    """Segment every pending tile (of these core images only, with `cores`). `workers` tiles are in flight at once and share one model, so the GPU runs one tile's
    network while another tile's CPU steps (normalisation, mask building) run; the masks are the same as one at a time."""
    project.apply_environment()
    output = project.output("nuclei")
    output.mkdir(parents=True, exist_ok=True)
    rows = [r for r in read_csv(project.analysis_tiles()) if r["stain"] in set(stains) and (cores is None or r["core_id"] in cores)]
    rows = sorted(rows, key=lambda r: (r["image_path"], int(r["selection_order"])))[shard_index::shard_count]
    done = file_names(output)
    pending = [r for r in rows if f"{r['tile_id']}.npz" not in done]
    print(f"[0/{len(pending)}] {len(pending)} of {len(rows)} fields still to do", flush=True)
    CELLPOSE_SETTINGS["batch_size"] = batch_size
    model = load_cellpose(project.model("cellpose"), threads, device, project.precision("cellpose"))
    images, finished = CoreImages(), itertools.count(1)

    def segment(row: dict) -> None:
        if (output / f"{row['tile_id']}.npz").exists():  # another process (an overlapping shard) got there first
            return
        image = images.get(project.root / row["image_path"])
        rgb, _ = context_crop(image, int(row["x_px"]), int(row["y_px"]), int(row["width_px"]), int(row["height_px"]), project.context_px)
        started = time.time()
        masks = cellpose_masks(model, rgb)
        save_npz_atomic(output / f"{row['tile_id']}.npz", masks=masks.astype("uint32"))
        print(f"[{next(finished)}/{len(pending)}] {row['tile_id']}: {int(masks.max())} nuclei, {time.time() - started:.0f}s", flush=True)

    with ThreadPoolExecutor(max(1, workers)) as pool:
        for _ in pool.map(segment, pending):
            pass
