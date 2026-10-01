from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from joblib import load
from PIL import Image

from stainid.imaging.tissue import field_tissue_mask, linear_artifact_mask
from stainid.nuclei import CELLPOSE_SETTINGS, cellpose_masks, load_cellpose
from stainid.pipelines.cohort_v1 import context_crop, local_profile_density
from stainid.qc.exclusions import rasterize_core_exclusions, read_manual_exclusions
from stainid.stains.neun.cellpose import classify_cellpose_masks
from stainid.stains.neun.classifier import build_matrix_from_contexts, object_contexts
from stainid.stains.neun.profiles import combine_profile_and_cellpose_reviews, review_table_rows, segment_dab_profiles
from stainid.tables import format_csv_value, write_csv, write_text_atomic

Image.MAX_IMAGE_PIXELS = None


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def neun_candidates(
    rgb: np.ndarray,
    raw_masks: np.ndarray,
    threshold: float,
    pixel_width_um: float,
    pixel_height_um: float,
    manual_mask: np.ndarray,
    tile_id: str,
) -> tuple[list[dict[str, object]], np.ndarray, np.ndarray]:
    """All NeuN candidate profiles in a field (stain contours + Cellpose cells) and their label images."""
    _, cellpose_objects, cellpose_labels = classify_cellpose_masks(
        raw_masks, rgb, threshold, pixel_width_um, pixel_height_um
    )
    cellpose_objects = [
        {
            **{
                key: str(format_csv_value(value))
                for key, value in row.items()
                if key != "label"
            },
            "label": label,
        }
        for label, row in enumerate(cellpose_objects, start=1)
    ]
    profile_objects, profile_labels = segment_dab_profiles(
        rgb, threshold, pixel_width_um, pixel_height_um, manual_mask
    )
    combined = combine_profile_and_cellpose_reviews(
        threshold,
        profile_objects,
        profile_labels,
        cellpose_labels,
        cellpose_objects,
        manual_mask,
    )
    rows = [
        {
            **row,
            "selection_area_um2": row["profile_area_um2"]
            if row["source_kind"] == "dab_profile"
            else row["area_um2"],
        }
        for row in review_table_rows(combined, tile_id)
    ]
    return rows, profile_labels, cellpose_labels


def neun_feature_matrix(rgb: np.ndarray, rows: list[dict[str, object]], pixel_size_um: float, context_um: float = 55.0) -> tuple[np.ndarray, list[str]]:
    contexts = object_contexts(rgb[:, :, ::-1], rows, pixel_size_um, context_um)
    return build_matrix_from_contexts(rows, contexts)


def classify_neun_tile(
    rgb: np.ndarray,
    raw_masks: np.ndarray,
    inner: tuple[slice, slice],
    threshold: float,
    pixel_width_um: float,
    pixel_height_um: float,
    manual_mask: np.ndarray,
    bundle: dict[str, object],
    tile_id: str,
    context_um: float = 55.0,
) -> tuple[dict[str, float | int], list[dict[str, object]]]:
    rows, profile_labels, cellpose_labels = neun_candidates(rgb, raw_masks, threshold, pixel_width_um, pixel_height_um, manual_mask, tile_id)
    artifact = linear_artifact_mask(rgb) | manual_mask
    tissue = field_tissue_mask(rgb) & ~artifact
    inner_tissue = tissue[inner]
    pixel_area_um2 = pixel_width_um * pixel_height_um
    tissue_area_mm2 = inner_tissue.sum() * pixel_area_um2 / 1_000_000.0
    y_slice, x_slice = inner
    selected_rows = [
        row
        for row in rows
        if x_slice.start <= float(row["centroid_x_px"]) < x_slice.stop
        and y_slice.start <= float(row["centroid_y_px"]) < y_slice.stop
    ]
    context_threshold = float(bundle["context_threshold"])
    addition_threshold = float(bundle["addition_threshold"])
    if selected_rows:
        pixel_size_um = float(np.sqrt(pixel_width_um * pixel_height_um))
        matrix, names = neun_feature_matrix(rgb, selected_rows, pixel_size_um, context_um)
        if names != bundle["feature_names"]:
            raise ValueError("NeuN cohort features do not match the frozen model")
        probability = bundle["classifier"].predict_proba(
            matrix[:, bundle["usable_features"]]
        )[:, 1]
    else:
        probability = np.zeros(0)
    conservative = np.asarray(
        [float(row["candidate_class"] == "positive_profile") for row in selected_rows]
    )
    union = np.maximum(probability, conservative) if selected_rows else probability
    objects = []
    positive_mask = np.zeros(inner_tissue.shape, dtype=bool)
    for row, context_value, union_value in zip(selected_rows, probability, union):
        positive = bool(context_value >= context_threshold)
        if positive:
            labels = (
                profile_labels
                if row["source_kind"] == "dab_profile"
                else cellpose_labels
            )
            positive_mask |= labels[inner] == int(row["source_label"])
        objects.append(
            {
                **{
                    key: value
                    for key, value in row.items()
                    if key not in {"image_id", "centroid_x_px", "centroid_y_px"}
                },
                "proposal_class": row["candidate_class"],
                "candidate_class": "positive" if positive else "negative",
                "centroid_x_px": float(row["centroid_x_px"]),
                "centroid_y_px": float(row["centroid_y_px"]),
                "area_um2": float(row["selection_area_um2"]),
                "mean_dab_od": row["mean_dab_od"],
                "context_positive_probability": float(context_value),
                "union_positive_probability": float(union_value),
                "union_positive": bool(union_value >= addition_threshold),
                "triage": "positive"
                if context_value >= 0.8
                else "negative"
                if context_value <= 0.2
                else "review",
            }
        )
    positive = [row for row in objects if row["candidate_class"] == "positive"]
    union_count = sum(bool(row["union_positive"]) for row in objects)
    summary = {
        "tissue_area_mm2": tissue_area_mm2,
        "candidate_profile_count": len(objects),
        "neun_positive_profile_count": len(positive),
        "neun_union_positive_profile_count": union_count,
        "neun_review_profile_count": sum(row["triage"] == "review" for row in objects),
        "candidate_profile_density_mm2": (
            len(objects) / tissue_area_mm2 if tissue_area_mm2 else float("nan")
        ),
        "neun_positive_profile_density_mm2": (
            len(positive) / tissue_area_mm2 if tissue_area_mm2 else float("nan")
        ),
        "neun_union_positive_profile_density_mm2": (
            union_count / tissue_area_mm2 if tissue_area_mm2 else float("nan")
        ),
        "neun_positive_fraction_of_candidates": (
            len(positive) / len(objects) if objects else float("nan")
        ),
        "neun_positive_profile_area_fraction": (
            (positive_mask & inner_tissue).sum() / inner_tissue.sum()
            if inner_tissue.any()
            else float("nan")
        ),
        **local_profile_density(
            positive, tissue, inner, pixel_width_um, pixel_height_um
        ),
        "automatic_linear_artifact_area_fraction": float(
            linear_artifact_mask(rgb)[inner].mean()
        ),
        "manual_exclusion_area_fraction": float(manual_mask[inner].mean()),
        "total_artifact_exclusion_area_fraction": float(artifact[inner].mean()),
    }
    return summary, objects


def claim_field(claim: Path) -> bool:
    """Take a field for this process; a claim left by a process that no longer runs (stopped or crashed) is taken over."""
    try:
        descriptor = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        owner = claim.read_text().strip()
        try:
            os.kill(int(owner), 0)
            return False
        except (ValueError, ProcessLookupError):
            claim.unlink(missing_ok=True)
            return claim_field(claim)
        except PermissionError:
            return False
    os.write(descriptor, str(os.getpid()).encode())
    os.close(descriptor)
    return True


def run_neun_cohort(
    tile_manifest: Path,
    calibration_path: Path,
    model_bundle_path: Path,
    cellpose_model_path: Path,
    output_dir: Path,
    manual_exclusions_path: Path,
    tile_ids: set[str] | None = None,
    nested_samples: set[str] | None = None,
    shard_index: int = 0,
    shard_count: int = 1,
    threads: int = 8,
    context_px: int = 128,
    device: str = "cpu",
    batch_size: int = 1,
    reverse: bool = False,
) -> int:
    rows = [row for row in read_csv(tile_manifest) if row["stain"] == "NeuN"]
    if nested_samples:
        rows = [row for row in rows if row["nested_sample"] in nested_samples]
    if tile_ids:
        rows = [row for row in rows if row["tile_id"] in tile_ids]
    image_paths = sorted({row["image_path"] for row in rows})[shard_index::shard_count]
    rows = sorted(
        (row for row in rows if row["image_path"] in set(image_paths)),
        key=lambda row: (row["image_path"], int(row["selection_order"])),
    )
    tile_dir = output_dir / "tiles"
    mask_dir = output_dir / "cellpose_masks"
    tile_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    pending = [
        row for row in rows if not (tile_dir / f"{row['tile_id']}_features.csv").exists()
    ]
    if reverse:
        pending.reverse()
    provenance_path = output_dir / "provenance.json"
    provenance = {
        "neun_model_bundle": str(model_bundle_path),
        "neun_model_bundle_sha256": sha256(model_bundle_path),
        "cellpose_model": str(cellpose_model_path),
        "cellpose_model_sha256": sha256(cellpose_model_path),
        "cellpose_settings": {
            key: value for key, value in CELLPOSE_SETTINGS.items() if key != "batch_size"
        }
        | {"precision": "float32"},
        "context_px": context_px,
        "primary_rule": "context model probability >= context_threshold",
        "sensitivity_rule": "stain contour or context probability >= addition_threshold",
    }
    if provenance_path.exists():
        recorded = json.loads(provenance_path.read_text(encoding="utf-8"))
        if recorded != provenance:
            raise ValueError("These NeuN results were made with a different model or settings. "
                             f"Run the step again with 'Start over' to redo them ({provenance_path})")
    else:
        write_text_atomic(provenance_path, json.dumps(provenance, indent=2) + "\n")
    print(f"[0/{len(pending)}] {len(pending)} of {len(rows)} fields still to do (part {shard_index + 1} of {shard_count})", flush=True)
    if not pending:
        return 0
    bundle = load(model_bundle_path)
    calibrations = {
        row["tma"]: float(row["threshold_dab_od"])
        for row in read_csv(calibration_path)
        if row["stain"] == "NeuN"
    }
    manual_exclusions = read_manual_exclusions(manual_exclusions_path)
    CELLPOSE_SETTINGS["batch_size"] = batch_size
    model = load_cellpose(cellpose_model_path, threads, device)
    image_path = None
    image = None
    for index, row in enumerate(pending, start=1):
        claim = tile_dir / f"{row['tile_id']}.claim"
        if (tile_dir / f"{row['tile_id']}_features.csv").exists():
            continue
        if not claim_field(claim):
            continue
        if row["image_path"] != image_path:
            image_path = row["image_path"]
            image = Image.open(image_path)
            image.load()
        x, y = int(row["x_px"]), int(row["y_px"])
        width, height = int(row["width_px"]), int(row["height_px"])
        rgb, inner = context_crop(image, x, y, width, height, context_px)
        crop_x = x - inner[1].start
        crop_y = y - inner[0].start
        manual_mask = rasterize_core_exclusions(
            manual_exclusions.get(row["image_path"], []),
            crop_x,
            crop_y,
            rgb.shape[:2],
        )
        mask_path = mask_dir / f"{row['tile_id']}.npz"
        if mask_path.exists():
            with np.load(mask_path) as loaded:
                raw_masks = loaded["masks"]
        else:
            raw_masks = cellpose_masks(model, rgb)
            np.savez_compressed(mask_path, masks=raw_masks.astype(np.uint32))
        threshold = calibrations[row["tma"]]
        summary, objects = classify_neun_tile(
            rgb,
            raw_masks,
            inner,
            threshold,
            float(row["pixel_width_um"]),
            float(row["pixel_height_um"]),
            manual_mask,
            bundle,
            row["tile_id"],
        )
        common = {
            key: value
            for key, value in row.items()
            if key not in {"preview_x_px", "preview_y_px"}
        }
        object_rows = [
            {
                **common,
                "object_id": f"{row['tile_id']}_O{number:05d}",
                "tile_centroid_x_px": obj["centroid_x_px"] - inner[1].start,
                "tile_centroid_y_px": obj["centroid_y_px"] - inner[0].start,
                "core_centroid_x_px": crop_x + obj["centroid_x_px"],
                "core_centroid_y_px": crop_y + obj["centroid_y_px"],
                **{
                    key: value
                    for key, value in obj.items()
                    if key not in {"centroid_x_px", "centroid_y_px"}
                },
            }
            for number, obj in enumerate(objects, start=1)
        ]
        feature_row = {
            **common,
            "analysis_context_px": context_px,
            "cellpose_device": device,
            "calibration_threshold_dab_od": threshold,
            **summary,
        }
        write_csv(
            tile_dir / f"{row['tile_id']}_objects.csv",
            [{k: format_csv_value(v) for k, v in obj.items()} for obj in object_rows],
            list(object_rows[0]) if object_rows else ["tile_id", "object_id"],
        )
        write_csv(
            tile_dir / f"{row['tile_id']}_features.csv",
            [{k: format_csv_value(v) for k, v in feature_row.items()}],
        )
        claim.unlink()
        print(
            f"[{index}/{len(pending)}] {device} {row['tile_id']}: "
            f"{summary['neun_positive_profile_count']} positive of "
            f"{summary['candidate_profile_count']} candidates",
            flush=True,
        )
    return len(pending)


def merge_neun_cohort(output_dir: Path, feature_output: Path, object_output: Path) -> tuple[Path, Path]:
    features: list[dict[str, str]] = []
    objects: list[dict[str, str]] = []
    for path in sorted((output_dir / "tiles").glob("*_features.csv")):
        features.extend(read_csv(path))
        objects.extend(read_csv(path.with_name(path.name.replace("_features", "_objects"))))
    if not features:
        raise ValueError(f"No NeuN cohort tiles found in {output_dir}")
    write_csv(feature_output, features, list(dict.fromkeys(k for r in features for k in r)))
    write_csv(
        object_output,
        objects,
        list(dict.fromkeys(k for r in objects for k in r)) or ["tile_id", "object_id"],
    )
    return feature_output, object_output


__all__ = [
    "classify_neun_tile",
    "merge_neun_cohort",
    "run_neun_cohort",
]
