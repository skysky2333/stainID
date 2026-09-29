from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from joblib import load
from PIL import Image

from stainid.imaging.tissue import field_tissue_mask, linear_artifact_mask
from stainid.qc.exclusions import rasterize_core_exclusions
from stainid.stains.amyloid.model import classify_amyloid_objects
from stainid.stains.amyloid.segmentation import segment_amyloid_candidates, write_amyloid_overlay
from stainid.stains.neun.candidates import segment_neun_candidates, write_candidate_overlay
from stainid.stains.tau.network import segment_at8_network, write_at8_overlay
from stainid.tables import format_csv_value, write_csv

Image.MAX_IMAGE_PIXELS = None


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def union_columns(rows: list[dict[str, object]]) -> list[str]:
    columns: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for column in row:
            if column not in seen:
                columns.append(column)
                seen.add(column)
    return columns


def context_crop(
    image: Image.Image,
    x: int,
    y: int,
    width: int,
    height: int,
    context_px: int,
) -> tuple[np.ndarray, tuple[slice, slice]]:
    x0 = max(0, x - context_px)
    y0 = max(0, y - context_px)
    x1 = min(image.width, x + width + context_px)
    y1 = min(image.height, y + height + context_px)
    rgb = np.asarray(image.crop((x0, y0, x1, y1)).convert("RGB"))
    inner = (slice(y - y0, y - y0 + height), slice(x - x0, x - x0 + width))
    return rgb, inner


def centered_objects(
    objects: list[dict[str, float | int | bool | str]],
    inner: tuple[slice, slice],
) -> list[dict[str, float | int | bool | str]]:
    y_slice, x_slice = inner
    return [
        row
        for row in objects
        if x_slice.start <= float(row["centroid_x_px"]) < x_slice.stop
        and y_slice.start <= float(row["centroid_y_px"]) < y_slice.stop
    ]


def local_profile_density(
    objects: list[dict[str, float | int | bool | str]],
    tissue: np.ndarray,
    inner: tuple[slice, slice],
    pixel_width_um: float,
    pixel_height_um: float,
    window_um: float = 100.0,
    minimum_tissue_fraction: float = 0.5,
) -> dict[str, float | int]:
    y_slice, x_slice = inner
    window_width = max(1, round(window_um / pixel_width_um))
    window_height = max(1, round(window_um / pixel_height_um))
    pixel_area_mm2 = pixel_width_um * pixel_height_um / 1_000_000.0
    nominal_pixels = window_width * window_height
    densities = []
    for y0 in range(y_slice.start, y_slice.stop - window_height + 1, window_height):
        for x0 in range(x_slice.start, x_slice.stop - window_width + 1, window_width):
            local_tissue = tissue[y0 : y0 + window_height, x0 : x0 + window_width]
            tissue_pixels = int(local_tissue.sum())
            if tissue_pixels < minimum_tissue_fraction * nominal_pixels:
                continue
            count = sum(
                x0 <= float(row["centroid_x_px"]) < x0 + window_width
                and y0 <= float(row["centroid_y_px"]) < y0 + window_height
                for row in objects
            )
            densities.append(count / (tissue_pixels * pixel_area_mm2))
    mean_density = float(np.mean(densities)) if densities else float("nan")
    return {
        "neun_local_density_window_count": len(densities),
        "neun_local_density_mean_mm2": mean_density,
        "neun_local_density_cv_100um": (
            float(np.std(densities, ddof=1) / mean_density)
            if len(densities) >= 4 and mean_density > 0
            else float("nan")
        ),
    }


def analyze_tile(
    rgb: np.ndarray,
    inner: tuple[slice, slice],
    stain: str,
    threshold: float,
    pixel_width_um: float,
    pixel_height_um: float,
    overlay_path: Path | None = None,
    manual_exclusion_mask: np.ndarray | None = None,
    amyloid_bundle: dict[str, object] | None = None,
) -> tuple[dict[str, float | int | bool | str], list[dict[str, object]]]:
    pixel_area_um2 = pixel_width_um * pixel_height_um
    automatic_artifact = linear_artifact_mask(rgb)
    manual_artifact = (
        np.zeros(automatic_artifact.shape, dtype=bool)
        if manual_exclusion_mask is None
        else manual_exclusion_mask
    )
    if manual_artifact.shape != automatic_artifact.shape:
        raise ValueError("Manual artifact mask dimensions do not match the tile")
    artifact = automatic_artifact | manual_artifact
    automatic_artifact_fraction = float(automatic_artifact[inner].mean())
    manual_artifact_fraction = float(manual_artifact[inner].mean())
    artifact_fraction = float(artifact[inner].mean())
    artifact_summary = {
        "automatic_linear_artifact_area_fraction": automatic_artifact_fraction,
        "manual_exclusion_area_fraction": manual_artifact_fraction,
        "total_artifact_exclusion_area_fraction": artifact_fraction,
    }
    if stain == "NeuN":
        raw, objects, labels = segment_neun_candidates(
            rgb, threshold, pixel_width_um, pixel_height_um, artifact
        )
        selected = centered_objects(objects, inner)
        tissue = field_tissue_mask(rgb) & ~artifact
        inner_tissue = tissue[inner]
        tissue_area_mm2 = inner_tissue.sum() * pixel_area_um2 / 1_000_000.0
        complete = [row for row in selected if not bool(row["touches_edge"])]
        positive = [row for row in complete if bool(row["neun_positive"])]
        review = [row for row in complete if row["candidate_class"] == "review"]
        positive_labels = [int(row["label"]) for row in positive]
        positive_area = np.isin(labels[inner], positive_labels) & inner_tissue
        summary = {
            "tissue_area_mm2": tissue_area_mm2,
            "candidate_profile_count": len(complete),
            "neun_positive_profile_count": len(positive),
            "neun_review_profile_count": len(review),
            "candidate_profile_density_mm2": (
                len(complete) / tissue_area_mm2 if tissue_area_mm2 else float("nan")
            ),
            "neun_positive_profile_density_mm2": (
                len(positive) / tissue_area_mm2 if tissue_area_mm2 else float("nan")
            ),
            "neun_positive_fraction_of_candidates": (
                len(positive) / len(complete) if complete else float("nan")
            ),
            "neun_positive_profile_area_fraction": (
                positive_area.sum() / inner_tissue.sum()
                if inner_tissue.any()
                else float("nan")
            ),
            **local_profile_density(
                positive,
                tissue,
                inner,
                pixel_width_um,
                pixel_height_um,
            ),
            "seed_threshold_od": raw["seed_threshold_od"],
            **artifact_summary,
        }
        if overlay_path is not None:
            write_candidate_overlay(rgb, labels, objects, overlay_path, artifact)
    elif stain == "6E10":
        if amyloid_bundle is None:
            raise ValueError("6E10 inference requires the frozen amyloid bundle")
        raw, objects, plaque_labels, core_labels, excluded = (
            segment_amyloid_candidates(
                rgb, threshold, pixel_width_um, pixel_height_um, artifact
            )
        )
        selected = classify_amyloid_objects(
            rgb,
            centered_objects(objects, inner),
            threshold,
            float(np.sqrt(pixel_width_um * pixel_height_um)),
            amyloid_bundle,
        )
        valid = field_tissue_mask(rgb) & ~excluded
        inner_valid = valid[inner]
        tissue_area_mm2 = inner_valid.sum() * pixel_area_um2 / 1_000_000.0
        accepted = [
            row
            for row in selected
            if row["candidate_class"] in {"compact", "diffuse", "small_plaque"}
        ]
        eligible = [row for row in accepted if row["candidate_class"] != "small_plaque"]
        compact_count = sum(row["candidate_class"] == "compact" for row in eligible)
        accepted_labels = [int(row["label"]) for row in accepted]
        accepted_area = np.isin(plaque_labels[inner], accepted_labels) & inner_valid
        summary = {
            "tissue_area_mm2": tissue_area_mm2,
            "accepted_plaque_candidate_count": len(accepted),
            "morphotype_eligible_plaque_count": len(eligible),
            "compact_candidate_count": compact_count,
            "diffuse_candidate_count": len(eligible) - compact_count,
            "small_plaque_count": len(accepted) - len(eligible),
            "rejected_candidate_count": sum(
                row["candidate_class"] == "rejected"
                and row["proposal_class"] != "artifact"
                for row in selected
            ),
            "artifact_candidate_count": sum(
                row["proposal_class"] == "artifact" for row in selected
            ),
            "accepted_plaque_candidate_density_mm2": (
                len(accepted) / tissue_area_mm2 if tissue_area_mm2 else float("nan")
            ),
            "accepted_plaque_deposit_area_fraction": (
                accepted_area.sum() / inner_valid.sum()
                if inner_valid.any()
                else float("nan")
            ),
            "amyloid_positive_area_fraction": (
                (plaque_labels[inner] > 0)[inner_valid].mean()
                if inner_valid.any()
                else float("nan")
            ),
            "compact_plaque_fraction": (
                compact_count / len(eligible) if eligible else float("nan")
            ),
            "seed_threshold_dab_od": raw["seed_threshold_dab_od"],
            "extent_threshold_dab_od": raw["extent_threshold_dab_od"],
            **artifact_summary,
        }
        if overlay_path is not None:
            write_amyloid_overlay(
                rgb, plaque_labels, core_labels, selected, excluded, overlay_path
            )
    elif stain == "AT8":
        raw, objects, maps = segment_at8_network(
            rgb, threshold, pixel_width_um, pixel_height_um, artifact
        )
        selected = centered_objects(objects, inner)
        valid = maps["valid"][inner]
        positive = maps["positive"][inner] & valid
        thread = maps["thread"][inner] & valid
        tissue_area_mm2 = valid.sum() * pixel_area_um2 / 1_000_000.0
        positive_count = int(positive.sum())
        compact_count = sum(row["source_kind"] == "compact_profile" for row in selected)
        skeleton_length_mm = (
            maps["thread_skeleton"][inner].sum()
            * np.sqrt(pixel_area_um2)
            / 1_000.0
        )
        summary = {
            "tissue_area_mm2": tissue_area_mm2,
            "at8_positive_area_fraction": (
                positive_count / valid.sum() if valid.any() else float("nan")
            ),
            "compact_profile_count": compact_count,
            "compact_profile_density_mm2": (
                compact_count / tissue_area_mm2 if tissue_area_mm2 else float("nan")
            ),
            "noncompact_at8_area_fraction": (
                thread.sum() / valid.sum() if valid.any() else float("nan")
            ),
            "thread_area_fraction": (
                thread.sum() / valid.sum() if valid.any() else float("nan")
            ),
            "tau_noncompact_area_fraction_of_at8": (
                thread.sum() / positive_count if positive_count else float("nan")
            ),
            "tau_thread_area_fraction_of_at8": (
                thread.sum() / positive_count if positive_count else float("nan")
            ),
            "thread_skeleton_length_mm_per_mm2": (
                skeleton_length_mm / tissue_area_mm2
                if tissue_area_mm2
                else float("nan")
            ),
            "thread_branchpoint_density_mm2": (
                maps["branchpoints"][inner].sum() / tissue_area_mm2
                if tissue_area_mm2
                else float("nan")
            ),
            "thread_endpoint_density_mm2": (
                maps["endpoints"][inner].sum() / tissue_area_mm2
                if tissue_area_mm2
                else float("nan")
            ),
            "thread_cluster_count": sum(
                row["source_kind"] == "thread_cluster" for row in selected
            ),
            "positive_threshold_dab_od": raw["positive_threshold_dab_od"],
            "edge_guard_triggered": raw["edge_guard_triggered"],
            "edge_positive_fraction": raw["edge_positive_fraction"],
            "edge_positive_enrichment": raw["edge_positive_enrichment"],
            "edge_exclusion_fraction": raw["edge_exclusion_fraction"],
            "hole_rim_removed_area_fraction": float(
                maps["hole_rim_removed"][inner][valid].mean() if valid.any() else float("nan")
            ),
            "field_qc_status": raw["field_qc_status"],
            "biological_training_eligible": raw["biological_training_eligible"],
            **artifact_summary,
        }
        if overlay_path is not None:
            write_at8_overlay(rgb, objects, maps, overlay_path)
    else:
        raise ValueError(f"Unsupported stain: {stain}")
    return summary, [dict(row) for row in selected]


def run_cohort_inference(
    tile_manifest: Path,
    calibration_path: Path,
    feature_output: Path,
    object_output: Path,
    core_ids: set[str] | None = None,
    limit_images: int | None = None,
    context_px: int = 128,
    overlay_dir: Path | None = None,
    manual_exclusions_path: Path | None = None,
    amyloid_bundle_path: Path | None = None,
    stains: set[str] | None = None,
    shard_index: int = 0,
    shard_count: int = 1,
) -> tuple[Path, Path]:
    amyloid_bundle = load(amyloid_bundle_path) if amyloid_bundle_path else None
    rows = read_csv(tile_manifest)
    if core_ids:
        rows = [row for row in rows if row["core_id"] in core_ids]
    if stains:
        rows = [row for row in rows if row["stain"] in stains]
    shard_cores = set(sorted({row["core_id"] for row in rows})[shard_index::shard_count])
    rows = [row for row in rows if row["core_id"] in shard_cores]
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["image_path"], row["core_id"], row["stain"])].append(row)
    groups = sorted(grouped.items())
    if limit_images is not None:
        groups = groups[:limit_images]
    if not groups:
        raise ValueError("No cohort tile images matched the requested filters")
    calibrations = {
        (row["tma"], row["stain"]): float(row["threshold_dab_od"])
        for row in read_csv(calibration_path)
    }
    manual_exclusions: dict[str, list[dict[str, object]]] = {}
    if manual_exclusions_path is not None:
        with manual_exclusions_path.open(encoding="utf-8") as handle:
            manual_exclusions = json.load(handle)
    parts_dir = feature_output.with_name(f"{feature_output.stem}_parts")
    parts_dir.mkdir(parents=True, exist_ok=True)
    for image_index, ((image_path, core_id, stain), image_rows) in enumerate(
        groups, start=1
    ):
        part_features = parts_dir / f"{core_id}_{stain}_features.csv"
        part_objects = parts_dir / f"{core_id}_{stain}_objects.csv"
        if part_features.exists():
            continue
        feature_rows: list[dict[str, object]] = []
        object_rows: list[dict[str, object]] = []
        path = Path(image_path)
        if not path.is_absolute():
            path = Path.cwd() / path
        with Image.open(path) as image:
            for row in sorted(image_rows, key=lambda item: int(item["selection_order"])):
                x = int(row["x_px"])
                y = int(row["y_px"])
                width = int(row["width_px"])
                height = int(row["height_px"])
                rgb, inner = context_crop(image, x, y, width, height, context_px)
                crop_x = x - inner[1].start
                crop_y = y - inner[0].start
                manual_mask = rasterize_core_exclusions(
                    manual_exclusions.get(image_path, []),
                    crop_x,
                    crop_y,
                    rgb.shape[:2],
                )
                threshold = calibrations[(row["tma"], stain)]
                overlay_path = (
                    overlay_dir / stain / f"{row['tile_id']}.jpg"
                    if overlay_dir is not None
                    else None
                )
                summary, objects = analyze_tile(
                    rgb,
                    inner,
                    stain,
                    threshold,
                    float(row["pixel_width_um"]),
                    float(row["pixel_height_um"]),
                    overlay_path,
                    manual_mask,
                    amyloid_bundle,
                )
                common: dict[str, object] = {
                    key: value
                    for key, value in row.items()
                    if key not in {"preview_x_px", "preview_y_px"}
                }
                feature_rows.append(
                    {
                        **common,
                        "analysis_context_px": context_px,
                        "calibration_threshold_dab_od": threshold,
                        **summary,
                    }
                )
                inner_y, inner_x = inner
                for object_index, obj in enumerate(objects, start=1):
                    local_x = float(obj["centroid_x_px"]) - inner_x.start
                    local_y = float(obj["centroid_y_px"]) - inner_y.start
                    object_rows.append(
                        {
                            **common,
                            "object_id": f"{row['tile_id']}_O{object_index:05d}",
                            "tile_centroid_x_px": local_x,
                            "tile_centroid_y_px": local_y,
                            "core_centroid_x_px": x + local_x,
                            "core_centroid_y_px": y + local_y,
                            **{
                                key: value
                                for key, value in obj.items()
                                if key not in {"centroid_x_px", "centroid_y_px"}
                            },
                        }
                    )
        formatted_objects = [
            {key: format_csv_value(value) for key, value in row.items()}
            for row in object_rows
        ]
        write_csv(
            part_objects,
            formatted_objects,
            union_columns(formatted_objects)
            if formatted_objects
            else ["tile_id", "core_id", "stain", "object_id"],
        )
        formatted_features = [
            {key: format_csv_value(value) for key, value in row.items()}
            for row in feature_rows
        ]
        write_csv(part_features, formatted_features, union_columns(formatted_features))
        print(
            f"Analyzed {image_index}/{len(groups)} stain-core images: {core_id} {stain}",
            flush=True,
        )
    all_features: list[dict[str, str]] = []
    all_objects: list[dict[str, str]] = []
    for (_, core_id, stain), _ in groups:
        all_features += read_csv(parts_dir / f"{core_id}_{stain}_features.csv")
        all_objects += read_csv(parts_dir / f"{core_id}_{stain}_objects.csv")
    write_csv(feature_output, all_features, union_columns(all_features))
    write_csv(
        object_output,
        all_objects,
        union_columns(all_objects) if all_objects else ["tile_id", "core_id", "stain", "object_id"],
    )
    return feature_output, object_output


__all__ = [
    "analyze_tile",
    "centered_objects",
    "context_crop",
    "local_profile_density",
    "run_cohort_inference",
]
