"""Prompted SAM ViT-B outlines for accepted objects (NeuN neurons, plaques, tau+ neurons)."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
from joblib import load
from PIL import Image

from stainid.masks.objects import segment_objects
from stainid.masks.sam import WindowedSam
from stainid.pipelines.cohort_v1 import context_crop
from stainid.project import Project
from stainid.tables import read_csv, write_records
from stainid.workflows import calibration_table, tiles_for

Image.MAX_IMAGE_PIXELS = None
TILE_KEYS = ("tile_id", "core_id", "tma", "donor_id", "sample_region_id", "region", "disease_group", "stain", "selection_order")


def load_seeds(stain: str, tiles: list[dict[str, str]], fields_parts: Path, neun_tiles: Path, context_px: int) -> dict[str, list[dict]]:
    """Object centroids (field-crop coordinates) that SAM is prompted with."""
    seeds: dict[str, list[dict]] = defaultdict(list)
    if stain == "NeuN":
        for tile in tiles:
            crop_x0, crop_y0 = max(0, int(tile["x_px"]) - context_px), max(0, int(tile["y_px"]) - context_px)
            for row in read_csv(neun_tiles / f"{tile['tile_id']}_objects.csv"):
                if row["candidate_class"] == "positive":
                    seeds[tile["tile_id"]].append({"x": float(row["core_centroid_x_px"]) - crop_x0, "y": float(row["core_centroid_y_px"]) - crop_y0,
                                                   "seed_source": row["source_kind"], "seed_probability": row["context_positive_probability"]})
        return seeds
    for core_id in {tile["core_id"] for tile in tiles}:
        for row in read_csv(fields_parts / f"{core_id}_{stain}_objects.csv"):
            if not row.get("tile_id"):
                continue
            if stain == "6E10":
                seeds[row["tile_id"]].append({"x": float(row["centroid_x_px"]), "y": float(row["centroid_y_px"]),
                                              "seed_source": row["plaque_class"], "seed_probability": row["plaque_probability"]})
            else:
                seeds[row["tile_id"]].append({"x": float(row["centroid_x_px"]), "y": float(row["centroid_y_px"]), "seed_source": row["candidate_source"],
                                              "seed_probability": row["tau_neuron_probability"], "seed_mature": row["tau_neuron_mature"]})
    return seeds


def run_masks(project: Project, stain: str, device: str = "cpu", threads: int = 4, shard_index: int = 0, shard_count: int = 1) -> None:
    import torch

    torch.set_num_threads(threads)
    neun_tiles = project.output("neun") / "tiles"
    groups = tiles_for(project.input("tile_manifest"), {stain}, shard_index, shard_count)
    if stain == "NeuN":
        groups = {k: [t for t in v if (neun_tiles / f"{t['tile_id']}_objects.csv").exists()] for k, v in groups.items()}
    calibration = calibration_table(project.input("calibration"))
    core_multiplier = float(load(project.model("amyloid"))["morphotype_threshold"]) if stain == "6E10" else None
    sam = WindowedSam(project.model("sam"), device=device)
    out = project.output("masks") / stain
    (out / "parts").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    for number, ((core_id, _), tiles) in enumerate(groups.items(), start=1):
        target = out / "parts" / f"{core_id}_{stain}_objects.csv"
        if target.exists() or not tiles:
            continue
        seeds = load_seeds(stain, tiles, project.output("fields") / "parts", neun_tiles, project.context_px)
        rows = []
        with Image.open(project.root / tiles[0]["image_path"]) as image:
            image.load()
            for tile in tiles:
                if not seeds.get(tile["tile_id"]):
                    continue
                rgb, inner = context_crop(image, int(tile["x_px"]), int(tile["y_px"]), int(tile["width_px"]), int(tile["height_px"]), project.context_px)
                pixel_size = float(np.sqrt(float(tile["pixel_width_um"]) * float(tile["pixel_height_um"])))
                threshold = calibration[(tile["tma"], stain)]
                objects, labels = segment_objects(sam, rgb, seeds[tile["tile_id"]], stain, threshold, pixel_size,
                                                  None if core_multiplier is None else core_multiplier * threshold)
                np.savez_compressed(out / "labels" / f"{tile['tile_id']}.npz", labels=labels.astype(np.uint16), inner=np.array([inner[0].start, inner[1].start]))
                rows += [{**{k: tile[k] for k in TILE_KEYS}, **obj} for obj in objects]
        write_records(target, rows, ["tile_id"])
        print(f"[{number}/{len(groups)}] {core_id} {stain}: {sum(r['mask_status'] == 'ok' for r in rows)} masks", flush=True)
