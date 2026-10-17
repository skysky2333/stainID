"""6E10 plaque and AT8 tau field pipelines (per core, per stain; outputs `parts/<core>_<stain>_{features,objects}.csv`)."""
from __future__ import annotations

import json

import numpy as np
from joblib import load
from PIL import Image

from stainid.imaging.core_images import open_image
from stainid.imaging.tissue import core_glass_darkness, fold_mask, linear_artifact_mask
from stainid.pipelines.cohort_v1 import context_crop
from stainid.project import Project
from stainid.qc.exclusions import rasterize_core_exclusions
from stainid.stains.amyloid.field_pipeline import analyze_6e10_v2
from stainid.stains.tau.field_pipeline import analyze_at8_v2
from stainid.tables import claim_field, file_names, write_records, write_text_atomic
from stainid.workflows import calibration_table, follow_nuclei, tiles_for

Image.MAX_IMAGE_PIXELS = None
TILE_KEYS = ("tile_id", "core_id", "tma", "donor_id", "sample_region_id", "region", "disease_group", "technical_replicate", "stain", "selection_order")


def check_models(project: Project, path, models: dict) -> None:
    """Refuse to mix results from different models in one results folder."""
    used = {key: project.relative(value) for key, value in models.items()}
    if path.exists():
        if json.loads(path.read_text()) != used:
            raise ValueError("These results were made with a different model. Run the step again with 'Start over' to redo them with the current model.")
        return
    write_text_atomic(path, json.dumps(used, indent=1))


def exclusion_mask(rgb, tile, inner, pixel_size, manual):
    x, y = int(tile["x_px"]), int(tile["y_px"])
    return (linear_artifact_mask(rgb) | fold_mask(rgb, pixel_size)
            | rasterize_core_exclusions(manual.get(tile["image_path"], []), x - inner[1].start, y - inner[0].start, rgb.shape[:2]))


def analyze_field(project: Project, image, tile, stain, calibration, manual, bundles, nuclei_dir, glass=None):
    rgb, inner = context_crop(image, int(tile["x_px"]), int(tile["y_px"]), int(tile["width_px"]), int(tile["height_px"]), project.context_px)
    pixel_size = float(np.sqrt(float(tile["pixel_width_um"]) * float(tile["pixel_height_um"])))
    exclusion = exclusion_mask(rgb, tile, inner, pixel_size, manual)
    nuclei_path = nuclei_dir / f"{tile['tile_id']}.npz"
    nuclei = np.load(nuclei_path)["masks"] if nuclei_path.exists() else np.zeros(rgb.shape[:2], dtype=np.uint32)
    threshold = calibration[(tile["tma"], stain)]
    if stain == "AT8":
        summary, items = analyze_at8_v2(rgb, inner, nuclei, threshold, pixel_size, exclusion, bundles["tau"], glass=glass)
    else:
        summary, items = analyze_6e10_v2(rgb, inner, nuclei, threshold, pixel_size, exclusion, bundles["amyloid"], glass=glass)
    common = {k: tile[k] for k in TILE_KEYS}
    feature = {**common, "nuclei_available": nuclei_path.exists(), "fold_exclusion_fraction": float(fold_mask(rgb, pixel_size)[inner].mean()), **summary}
    return feature, [{**common, **item} for item in items]


def run_fields(project: Project, stains: list[str] = ("6E10", "AT8"), shard_index: int = 0, shard_count: int = 1,
               core_ids: set[str] | None = None) -> None:
    project.apply_environment()
    groups = tiles_for(project.analysis_tiles(), set(stains), shard_index, shard_count, core_ids)
    calibration = calibration_table(project.input("calibration"))
    manual: dict = {}  # detection's own areas only (the hand-drawn polygons are people's areas: training and evaluation)
    tau = load(project.model("tau"))
    bundles = {"tau": {**tau, "probability_threshold": project.acceptance("tau", tau)}, "amyloid": load(project.model("amyloid"))}
    parts = project.output("fields") / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    models = {"amyloid": project.model("amyloid"), "tau": project.model("tau")}
    if project.config["acceptance"].get("tau") is not None:
        models["tau_acceptance"] = str(bundles["tau"]["probability_threshold"])
    check_models(project, project.output("fields") / "provenance.json", models)
    nuclei_dir = project.output("nuclei")
    done, nuclei = file_names(parts), file_names(nuclei_dir)
    print(f"[0/{len(groups)}] {sum(f'{c}_{s}_features.csv' not in done for c, s in groups)} of {len(groups)} core images still to do", flush=True)
    skipped = []
    for number, ((core_id, stain), tiles) in enumerate(groups.items(), start=1):
        feature_path = parts / f"{core_id}_{stain}_features.csv"
        if feature_path.exists():
            continue
        if stain == "AT8" and not all(f"{t['tile_id']}.npz" in nuclei for t in tiles):
            skipped.append(core_id)
            print(f"[{number}/{len(groups)}] {core_id} AT8 skipped: its nuclei are missing", flush=True)
            continue
        claim = parts / f"{core_id}_{stain}.claim"
        if not claim_field(claim):  # another process is on this core image
            continue
        try:
            if feature_path.exists():  # finished by another process meanwhile
                continue
            features, objects = [], []
            with open_image(project.root / tiles[0]["image_path"]) as image:
                image.load()
                glass = core_glass_darkness(image)
                for tile in tiles:
                    feature, items = analyze_field(project, image, tile, stain, calibration, manual, bundles, nuclei_dir, glass)
                    features.append(feature)
                    objects += items
            write_records(parts / f"{core_id}_{stain}_objects.csv", objects, ["tile_id"])
            write_records(feature_path, features)
            print(f"[{number}/{len(groups)}] {core_id} {stain}", flush=True)
        finally:
            claim.unlink(missing_ok=True)
    if skipped:
        print(f"{len(skipped)} AT8 cores were skipped because Find nuclei has not processed them yet; run Find nuclei for AT8, "
              "then run this step again to finish them", flush=True)


def follow_fields(project: Project, stains: list[str] = ("6E10", "AT8"), shard_index: int = 0, shard_count: int = 1) -> None:
    """Detect plaques and tau on each core image as soon as Find nuclei has finished it (see `follow_nuclei`)."""
    follow_nuclei(project, set(stains), lambda stain, core_ids: run_fields(project, [stain], core_ids=core_ids), shard_index, shard_count)
