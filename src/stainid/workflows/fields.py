"""6E10 plaque and AT8 tau field pipelines (per core, per stain; outputs `parts/<core>_<stain>_{features,objects}.csv`)."""
from __future__ import annotations

import json

import numpy as np
from joblib import load
from PIL import Image

from stainid.imaging.tissue import fold_mask, linear_artifact_mask
from stainid.pipelines.cohort_v1 import context_crop
from stainid.project import Project
from stainid.qc.exclusions import rasterize_core_exclusions
from stainid.stains.amyloid.field_pipeline import analyze_6e10_v2
from stainid.stains.tau.field_pipeline import analyze_at8_v2
from stainid.tables import write_records
from stainid.workflows import calibration_table, tiles_for

Image.MAX_IMAGE_PIXELS = None
TILE_KEYS = ("tile_id", "core_id", "tma", "donor_id", "sample_region_id", "region", "disease_group", "technical_replicate", "stain", "selection_order")


def check_models(project: Project, path, models: dict) -> None:
    """Refuse to mix results from different models in one results folder."""
    used = {key: project.relative(value) for key, value in models.items()}
    if path.exists() and json.loads(path.read_text()) != used:
        raise ValueError("These results were made with a different model. Run the step again with 'Start over' to redo them with the current model.")
    path.write_text(json.dumps(used, indent=1))


def exclusion_mask(rgb, tile, inner, pixel_size, manual):
    x, y = int(tile["x_px"]), int(tile["y_px"])
    return (linear_artifact_mask(rgb) | fold_mask(rgb, pixel_size)
            | rasterize_core_exclusions(manual.get(tile["image_path"], []), x - inner[1].start, y - inner[0].start, rgb.shape[:2]))


def analyze_field(project: Project, image, tile, stain, calibration, manual, bundles, nuclei_dir):
    rgb, inner = context_crop(image, int(tile["x_px"]), int(tile["y_px"]), int(tile["width_px"]), int(tile["height_px"]), project.context_px)
    pixel_size = float(np.sqrt(float(tile["pixel_width_um"]) * float(tile["pixel_height_um"])))
    exclusion = exclusion_mask(rgb, tile, inner, pixel_size, manual)
    nuclei_path = nuclei_dir / f"{tile['tile_id']}.npz"
    nuclei = np.load(nuclei_path)["masks"] if nuclei_path.exists() else np.zeros(rgb.shape[:2], dtype=np.uint32)
    threshold = calibration[(tile["tma"], stain)]
    if stain == "AT8":
        summary, items = analyze_at8_v2(rgb, inner, nuclei, threshold, pixel_size, exclusion, bundles["tau"])
    else:
        summary, items = analyze_6e10_v2(rgb, inner, nuclei, threshold, pixel_size, exclusion, bundles["amyloid"])
    common = {k: tile[k] for k in TILE_KEYS}
    feature = {**common, "nuclei_available": nuclei_path.exists(), "fold_exclusion_fraction": float(fold_mask(rgb, pixel_size)[inner].mean()), **summary}
    return feature, [{**common, **item} for item in items]


def run_fields(project: Project, stains: list[str] = ("6E10", "AT8"), shard_index: int = 0, shard_count: int = 1,
               core_ids: set[str] | None = None) -> None:
    project.apply_environment()
    groups = tiles_for(project.input("tile_manifest"), set(stains), shard_index, shard_count, core_ids)
    calibration = calibration_table(project.input("calibration"))
    manual = json.loads(project.input("manual_exclusions").read_text())
    bundles = {"tau": load(project.model("tau")), "amyloid": load(project.model("amyloid"))}
    parts = project.output("fields") / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    check_models(project, project.output("fields") / "provenance.json", {"amyloid": project.model("amyloid"), "tau": project.model("tau")})
    nuclei_dir = project.output("nuclei")
    for number, ((core_id, stain), tiles) in enumerate(groups.items(), start=1):
        feature_path = parts / f"{core_id}_{stain}_features.csv"
        if feature_path.exists():
            continue
        if stain == "AT8" and not all((nuclei_dir / f"{t['tile_id']}.npz").exists() for t in tiles):
            print(f"skip {core_id} {stain}: nuclei pending", flush=True)
            continue
        features, objects = [], []
        with Image.open(project.root / tiles[0]["image_path"]) as image:
            image.load()
            for tile in tiles:
                feature, items = analyze_field(project, image, tile, stain, calibration, manual, bundles, nuclei_dir)
                features.append(feature)
                objects += items
        write_records(parts / f"{core_id}_{stain}_objects.csv", objects, ["tile_id"])
        write_records(feature_path, features)
        print(f"[{number}/{len(groups)}] {core_id} {stain}", flush=True)
