from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from skimage.morphology import h_maxima
from skimage.segmentation import watershed

from stainid.imaging.color import rgb_to_hed
from stainid.imaging.tissue import field_tissue_mask
from stainid.stains.neun.candidates import mask_feature
from stainid.tables import format_csv_value, write_csv

CLASS_PROPERTIES = {
    "diffuse": ("Diffuse plaque", (20, 170, 120)),
    "compact": ("Compact or cored plaque", (220, 35, 35)),
    "review": ("Ambiguous amyloid deposit", (245, 145, 25)),
    "artifact": ("6E10 artifact", (180, 50, 190)),
    "compact_core": ("Compact plaque core", (245, 205, 35)),
    "small_plaque": ("Plaque below morphotype size", (60, 110, 220)),
    "speck": ("Deposit below plaque size", (170, 170, 170)),
    "rejected": ("Rejected amyloid candidate", (120, 120, 120)),
}


def apply_amyloid_class_overrides(
    objects: list[dict[str, float | int | bool | str]],
    summary: dict[str, float | int | str],
    image_id: str,
    overrides: dict[str, str],
) -> None:
    allowed = {"diffuse", "compact", "review", "artifact"}
    indexed = {
        f"{image_id}_AI_A{index:05d}": row
        for index, row in enumerate(objects, start=1)
    }
    unknown = set(overrides) - set(indexed)
    if unknown:
        raise ValueError(f"Unknown amyloid candidate overrides: {sorted(unknown)}")
    invalid = set(overrides.values()) - allowed
    if invalid:
        raise ValueError(f"Invalid amyloid override classes: {sorted(invalid)}")
    for candidate_id, candidate_class in overrides.items():
        indexed[candidate_id]["candidate_class"] = candidate_class
        indexed[candidate_id]["decision_basis"] = "ai_visual_adjudication_development"
        indexed[candidate_id]["review_status"] = "ai_visual_adjudicated_development"

    for name in allowed:
        summary[f"{name}_candidate_count"] = sum(
            row["candidate_class"] == name for row in objects
        )
    accepted = [
        row for row in objects if row["candidate_class"] in {"diffuse", "compact"}
    ]
    tissue_area_um2 = float(summary["tissue_area_mm2"]) * 1_000_000.0
    summary["accepted_plaque_candidate_count"] = len(accepted)
    summary["ambiguous_review_candidate_count"] = sum(
        row["candidate_class"] == "review" for row in objects
    )
    summary["accepted_plaque_candidate_density_mm2"] = (
        len(accepted) / float(summary["tissue_area_mm2"])
        if float(summary["tissue_area_mm2"])
        else float("nan")
    )
    summary["accepted_plaque_deposit_area_fraction"] = (
        sum(float(row["deposit_area_um2"]) for row in accepted) / tissue_area_um2
        if tissue_area_um2
        else float("nan")
    )
    if overrides:
        summary["review_status"] = "partially_ai_visual_adjudicated_development"


def _remove_small_components(mask: np.ndarray, minimum_pixels: int) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), 8
    )
    clean = np.zeros(mask.shape, dtype=np.uint8)
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] >= minimum_pixels:
            clean[labels == label] = 1
    return clean.astype(bool)


def _shape_features(mask: np.ndarray, pixel_area_um2: float) -> dict[str, float]:
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contour = max(contours, key=cv2.contourArea)
    perimeter_px = float(cv2.arcLength(contour, True))
    contour_area_px = float(cv2.contourArea(contour))
    hull_area_px = float(cv2.contourArea(cv2.convexHull(contour)))
    x, y, width, height = cv2.boundingRect(contour)
    ys, xs = np.nonzero(mask)
    coordinates = np.column_stack((xs, ys)).astype(np.float64)
    covariance = np.cov(coordinates, rowvar=False)
    eigenvalues = np.sort(np.linalg.eigvalsh(covariance))[::-1]
    major_axis_um = 4.0 * np.sqrt(max(eigenvalues[0], 0.0) * pixel_area_um2)
    minor_axis_um = 4.0 * np.sqrt(max(eigenvalues[-1], 0.0) * pixel_area_um2)
    eccentricity = (
        np.sqrt(max(0.0, 1.0 - eigenvalues[-1] / eigenvalues[0]))
        if eigenvalues[0] > 0
        else 0.0
    )
    area_px = int(mask.sum())
    return {
        "perimeter_um": perimeter_px * np.sqrt(pixel_area_um2),
        "circularity": (
            4.0 * np.pi * contour_area_px / perimeter_px**2
            if perimeter_px
            else 0.0
        ),
        "solidity": contour_area_px / hull_area_px if hull_area_px else 0.0,
        "aspect_ratio": max(width / height, height / width),
        "eccentricity": float(eccentricity),
        "major_axis_um": float(major_axis_um),
        "minor_axis_um": float(minor_axis_um),
        "boundary_irregularity": (
            perimeter_px**2 / (4.0 * np.pi * area_px) if area_px else 0.0
        ),
        "bbox_x_px": x,
        "bbox_y_px": y,
        "bbox_width_px": width,
        "bbox_height_px": height,
    }


def _radial_features(mask: np.ndarray, dab: np.ndarray) -> dict[str, float]:
    ys, xs = np.nonzero(mask)
    center_x = float(xs.mean())
    center_y = float(ys.mean())
    radius = np.sqrt((xs - center_x) ** 2 + (ys - center_y) ** 2)
    scale = float(np.quantile(radius, 0.9))
    if scale == 0:
        scale = 1.0
    normalized = radius / scale
    values = dab[ys, xs]
    inner = values[normalized <= 0.35]
    outer = values[(normalized >= 0.65) & (normalized <= 1.0)]
    return {
        "centroid_x_px": center_x,
        "centroid_y_px": center_y,
        "inner_mean_dab_od": float(inner.mean()) if inner.size else float(values.mean()),
        "outer_mean_dab_od": float(outer.mean()) if outer.size else float(values.mean()),
        "radial_dab_contrast": (
            float(inner.mean() - outer.mean())
            if inner.size and outer.size
            else 0.0
        ),
    }


def segment_amyloid_candidates(
    rgb: np.ndarray,
    dab_threshold: float,
    pixel_width_um: float,
    pixel_height_um: float,
    exclusion_mask: np.ndarray | None = None,
    split_touching: bool = False,
    split_prominence: float = 1.0,
) -> tuple[
    dict[str, float | int],
    list[dict[str, float | int | bool | str]],
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    tissue = field_tissue_mask(rgb)
    excluded = np.zeros(tissue.shape, dtype=bool)
    if exclusion_mask is not None:
        if exclusion_mask.shape != tissue.shape:
            raise ValueError("6E10 exclusion mask dimensions do not match the image")
        excluded |= exclusion_mask
    valid = tissue & ~excluded
    if valid.sum() < 100:
        raise ValueError("6E10 candidate detection requires at least 100 tissue pixels")

    dab = np.maximum(rgb_to_hed(rgb)[..., 2], 0.0).astype(np.float32)
    background_median = float(np.median(dab[valid]))
    background_mad = float(np.median(np.abs(dab[valid] - background_median)))
    background_sigma = max(1.4826 * background_mad, 0.001)
    seed_threshold = max(
        0.035,
        float(dab_threshold),
        background_median + 6.0 * background_sigma,
    )
    extent_threshold = max(
        0.010,
        0.30 * seed_threshold,
        background_median + 2.5 * background_sigma,
    )
    local_dab = cv2.GaussianBlur(dab, (0, 0), 1.2)
    context_dab = cv2.GaussianBlur(dab, (0, 0), 5.0)
    pixel_area_um2 = pixel_width_um * pixel_height_um
    minimum_seed_pixels = max(1, round(3.0 / pixel_area_um2))
    seed = _remove_small_components(
        valid & (local_dab >= seed_threshold), minimum_seed_pixels
    )
    extent = valid & (context_dab >= extent_threshold)
    extent = cv2.morphologyEx(
        extent.astype(np.uint8),
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
    ).astype(bool)
    count, extent_labels, stats, _ = cv2.connectedComponentsWithStats(
        extent.astype(np.uint8), 8
    )
    if split_touching:
        pixel_size_um = float(np.sqrt(pixel_width_um * pixel_height_um))
        body = cv2.GaussianBlur(dab, (0, 0), 3.0 / pixel_size_um)
        seeds = h_maxima(body, split_prominence * seed_threshold).astype(bool) & (body >= seed_threshold) & extent
        peak_count, markers = cv2.connectedComponents(seeds.astype(np.uint8), connectivity=8)
        peaks = np.zeros((peak_count - 1, 2))
        split = watershed(-body, markers, mask=extent)
        unseeded = extent & (split == 0)
        _, unseeded_labels = cv2.connectedComponents(unseeded.astype(np.uint8), connectivity=8)
        split[unseeded] = unseeded_labels[unseeded] + len(peaks)
        extent_labels = split.astype(np.int32)
        count = int(extent_labels.max()) + 1

    plaque_labels = np.zeros(tissue.shape, dtype=np.int32)
    core_labels = np.zeros(tissue.shape, dtype=np.int32)
    objects: list[dict[str, float | int | bool | str]] = []
    image_height, image_width = tissue.shape
    for component in range(1, count):
        component_mask = extent_labels == component
        seed_area_px = int((component_mask & seed).sum())
        if seed_area_px < minimum_seed_pixels:
            continue
        area_um2 = int(component_mask.sum()) * pixel_area_um2
        if not 20.0 <= area_um2 <= 50_000.0:
            continue
        shape = _shape_features(component_mask, pixel_area_um2)
        x = int(shape["bbox_x_px"])
        y = int(shape["bbox_y_px"])
        width = int(shape["bbox_width_px"])
        height = int(shape["bbox_height_px"])
        touches_edge = bool(
            x == 0
            or y == 0
            or x + width == image_width
            or y + height == image_height
        )
        values = dab[component_mask]
        core_threshold = max(
            0.10,
            1.20 * seed_threshold,
            float(np.quantile(values, 0.85)),
        )
        core = component_mask & (local_dab >= core_threshold)
        core = _remove_small_components(
            core, max(1, round(4.0 / pixel_area_um2))
        )
        core_count, candidate_core_labels, core_stats, _ = (
            cv2.connectedComponentsWithStats(core.astype(np.uint8), 8)
        )
        largest_core = np.zeros(core.shape, dtype=bool)
        if core_count > 1:
            largest = 1 + int(
                np.argmax(core_stats[1:, cv2.CC_STAT_AREA])
            )
            largest_core = candidate_core_labels == largest
        core_area_um2 = int(largest_core.sum()) * pixel_area_um2
        core_fraction = core_area_um2 / area_um2
        radial = _radial_features(component_mask, dab)
        if largest_core.any():
            core_y, core_x = np.nonzero(largest_core)
            centroid_distance_um = np.hypot(
                float(core_x.mean()) - radial["centroid_x_px"],
                float(core_y.mean()) - radial["centroid_y_px"],
            ) * np.sqrt(pixel_area_um2)
        else:
            centroid_distance_um = float("nan")
        equivalent_radius_um = np.sqrt(area_um2 / np.pi)
        normalized_core_offset = (
            centroid_distance_um / equivalent_radius_um
            if largest_core.any() and equivalent_radius_um
            else float("nan")
        )
        stained_area_um2 = (
            int((component_mask & (dab >= extent_threshold)).sum())
            * pixel_area_um2
        )
        stain_fill_fraction = stained_area_um2 / area_um2
        artifact = bool(
            (shape["aspect_ratio"] >= 5.0 and shape["major_axis_um"] >= 80.0)
            or (shape["major_axis_um"] >= 180.0 and shape["minor_axis_um"] <= 20.0)
            or (
                shape["major_axis_um"] >= 150.0
                and (
                    shape["minor_axis_um"] <= 35.0
                    or shape["solidity"] <= 0.20
                    or shape["boundary_irregularity"] >= 12.0
                )
            )
            or (
                shape["major_axis_um"] >= 45.0
                and shape["minor_axis_um"] <= 10.0
                and shape["eccentricity"] >= 0.99
            )
            or (
                shape["major_axis_um"] >= 180.0
                and shape["boundary_irregularity"] >= 8.0
                and shape["solidity"] <= 0.45
            )
            or (area_um2 >= 30_000.0 and touches_edge)
        )
        compact = bool(
            not artifact
            and not touches_edge
            and 8.0 <= core_area_um2 <= 0.45 * area_um2
            and normalized_core_offset <= 0.60
            and radial["radial_dab_contrast"] >= 0.010
        )
        diffuse = bool(
            not artifact
            and not touches_edge
            and not compact
            and 60.0 <= area_um2 <= 12_000.0
            and stain_fill_fraction >= 0.05
            and shape["aspect_ratio"] < 3.0
            and shape["boundary_irregularity"] < 8.0
        )
        if artifact:
            candidate_class = "artifact"
            decision_basis = "elongated_or_large_edge_dab_structure"
        elif compact:
            candidate_class = "compact"
            decision_basis = "central_compact_core_with_radial_dab_contrast"
        elif diffuse:
            candidate_class = "diffuse"
            decision_basis = "plaque_extent_without_qualifying_compact_core"
        elif touches_edge:
            candidate_class = "review"
            decision_basis = "edge_truncated_amyloid_deposit"
        else:
            candidate_class = "review"
            decision_basis = "small_large_or_merged_amyloid_deposit"

        label = len(objects) + 1
        plaque_labels[component_mask] = label
        if largest_core.any():
            core_labels[largest_core] = label
        objects.append(
            {
                "label": label,
                "candidate_class": candidate_class,
                "decision_basis": decision_basis,
                "centroid_x_px": radial["centroid_x_px"],
                "centroid_y_px": radial["centroid_y_px"],
                "deposit_area_um2": area_um2,
                "equivalent_diameter_um": 2.0 * equivalent_radius_um,
                "stained_area_um2": stained_area_um2,
                "stain_fill_fraction": stain_fill_fraction,
                "seed_area_um2": seed_area_px * pixel_area_um2,
                "core_area_um2": core_area_um2,
                "core_fraction": core_fraction,
                "core_threshold_od": core_threshold,
                "normalized_core_offset": normalized_core_offset,
                "mean_dab_od": float(values.mean()),
                "median_dab_od": float(np.median(values)),
                "p90_dab_od": float(np.quantile(values, 0.9)),
                "max_dab_od": float(values.max()),
                "inner_mean_dab_od": radial["inner_mean_dab_od"],
                "outer_mean_dab_od": radial["outer_mean_dab_od"],
                "radial_dab_contrast": radial["radial_dab_contrast"],
                "perimeter_um": shape["perimeter_um"],
                "circularity": shape["circularity"],
                "solidity": shape["solidity"],
                "aspect_ratio": shape["aspect_ratio"],
                "eccentricity": shape["eccentricity"],
                "major_axis_um": shape["major_axis_um"],
                "minor_axis_um": shape["minor_axis_um"],
                "boundary_irregularity": shape["boundary_irregularity"],
                "touches_edge": touches_edge,
            }
        )

    counts = {
        name: sum(row["candidate_class"] == name for row in objects)
        for name in ("diffuse", "compact", "review", "artifact")
    }
    tissue_area_mm2 = valid.sum() * pixel_area_um2 / 1_000_000.0
    accepted = [
        row for row in objects
        if row["candidate_class"] in {"diffuse", "compact"}
    ]
    review = [row for row in objects if row["candidate_class"] == "review"]
    summary = {
        "tissue_area_mm2": tissue_area_mm2,
        "background_median_dab_od": background_median,
        "background_mad_dab_od": background_mad,
        "seed_threshold_dab_od": seed_threshold,
        "extent_threshold_dab_od": extent_threshold,
        **{f"{name}_candidate_count": value for name, value in counts.items()},
        "accepted_plaque_candidate_count": len(accepted),
        "ambiguous_review_candidate_count": len(review),
        "accepted_plaque_candidate_density_mm2": (
            len(accepted) / tissue_area_mm2 if tissue_area_mm2 else float("nan")
        ),
        "accepted_plaque_deposit_area_fraction": (
            sum(float(row["deposit_area_um2"]) for row in accepted)
            / (valid.sum() * pixel_area_um2)
            if valid.sum()
            else float("nan")
        ),
        "manual_exclusion_fraction": float(excluded.mean()),
    }
    return summary, objects, plaque_labels, core_labels, excluded


def _metadata(
    row: dict[str, float | int | bool | str],
    candidate_id: str,
    proposal_source: str,
) -> dict[str, str]:
    return {
        "proposal_source": proposal_source,
        "candidate_id": candidate_id,
        "candidate_class": str(row["candidate_class"]),
        "decision_basis": str(row["decision_basis"]),
        "review_status": str(
            row.get("review_status", "ai_proposed_development")
        ),
        **{
            key: str(format_csv_value(value))
            for key, value in row.items()
            if key not in {"label", "candidate_class", "decision_basis"}
        },
    }


def write_amyloid_geojson(
    plaque_labels: np.ndarray,
    core_labels: np.ndarray,
    objects: list[dict[str, float | int | bool | str]],
    exclusion_mask: np.ndarray,
    output_path: Path,
    image_id: str,
    proposal_source: str,
) -> Path:
    features = []
    for index, row in enumerate(objects, start=1):
        candidate_id = f"{image_id}_AI_A{index:05d}"
        candidate_class = str(row["candidate_class"])
        class_name, color = CLASS_PROPERTIES[candidate_class]
        feature = mask_feature(
            plaque_labels == int(row["label"]),
            candidate_id,
            class_name,
            color,
            _metadata(row, candidate_id, proposal_source),
            convex_hull=False,
        )
        if feature is not None:
            features.append(feature)
        core_mask = core_labels == int(row["label"])
        if candidate_class == "compact" and core_mask.any():
            core_id = f"{candidate_id}_CORE"
            core_name, core_color = CLASS_PROPERTIES["compact_core"]
            core_feature = mask_feature(
                core_mask,
                core_id,
                core_name,
                core_color,
                {
                    "proposal_source": proposal_source,
                    "candidate_id": core_id,
                    "parent_candidate_id": candidate_id,
                    "candidate_class": "compact_core",
                    "geometry_role": "compact_center_boundary",
                    "review_status": "ai_proposed_development",
                },
                convex_hull=False,
            )
            if core_feature is not None:
                features.append(core_feature)
    count, labels = cv2.connectedComponents(
        exclusion_mask.astype(np.uint8), connectivity=8
    )
    for index in range(1, count):
        candidate_id = f"{image_id}_AI_X{index:03d}"
        class_name, color = CLASS_PROPERTIES["artifact"]
        feature = mask_feature(
            labels == index,
            candidate_id,
            class_name,
            color,
            {
                "proposal_source": proposal_source,
                "candidate_id": candidate_id,
                "candidate_class": "artifact_exclusion",
                "geometry_role": "manual_exclusion_region",
                "review_status": "ai_proposed_development",
            },
            convex_hull=False,
        )
        if feature is not None:
            features.append(feature)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump({"type": "FeatureCollection", "features": features}, handle, indent=2)
        handle.write("\n")
    return output_path


def write_amyloid_overlay(
    rgb: np.ndarray,
    plaque_labels: np.ndarray,
    core_labels: np.ndarray,
    objects: list[dict[str, float | int | bool | str]],
    exclusion_mask: np.ndarray,
    output_path: Path,
) -> Path:
    overlay = rgb.copy()
    for row in objects:
        color = CLASS_PROPERTIES[str(row["candidate_class"])][1]
        mask = (plaque_labels == int(row["label"])).astype(np.uint8)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(overlay, contours, -1, color, 3)
        core = (core_labels == int(row["label"])).astype(np.uint8)
        core_contours, _ = cv2.findContours(
            core, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if str(row["candidate_class"]) == "compact":
            cv2.drawContours(
                overlay, core_contours, -1, CLASS_PROPERTIES["compact_core"][1], 2
            )
    exclusion_contours, _ = cv2.findContours(
        exclusion_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(
        overlay, exclusion_contours, -1, CLASS_PROPERTIES["artifact"][1], 4
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(
        str(output_path),
        cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_JPEG_QUALITY, 95],
    ):
        raise OSError(f"Could not write 6E10 review overlay: {output_path}")
    return output_path


def write_amyloid_tables(
    objects: list[dict[str, float | int | bool | str]],
    summary: dict[str, float | int],
    object_path: Path,
    summary_path: Path,
    image_id: str,
) -> tuple[Path, Path]:
    object_fields = [
        "image_id",
        "candidate_id",
        "candidate_class",
        "decision_basis",
        "centroid_x_px",
        "centroid_y_px",
        "deposit_area_um2",
        "equivalent_diameter_um",
        "stained_area_um2",
        "stain_fill_fraction",
        "seed_area_um2",
        "core_area_um2",
        "core_fraction",
        "core_threshold_od",
        "normalized_core_offset",
        "mean_dab_od",
        "median_dab_od",
        "p90_dab_od",
        "max_dab_od",
        "inner_mean_dab_od",
        "outer_mean_dab_od",
        "radial_dab_contrast",
        "perimeter_um",
        "circularity",
        "solidity",
        "aspect_ratio",
        "eccentricity",
        "major_axis_um",
        "minor_axis_um",
        "boundary_irregularity",
        "touches_edge",
        "review_status",
    ]
    rows = []
    for index, row in enumerate(objects, start=1):
        rows.append(
            {
                "image_id": image_id,
                "candidate_id": f"{image_id}_AI_A{index:05d}",
                **{
                    key: format_csv_value(value)
                    for key, value in row.items()
                    if key != "label"
                },
                "review_status": row.get(
                    "review_status", "ai_proposed_development"
                ),
            }
        )
    write_csv(object_path, rows, object_fields)
    write_csv(
        summary_path,
        [
            {
                "image_id": image_id,
                **{key: format_csv_value(value) for key, value in summary.items()},
                "review_status": summary.get(
                    "review_status", "ai_proposed_development"
                ),
            }
        ],
    )
    return object_path, summary_path


__all__ = [
    "apply_amyloid_class_overrides",
    "segment_amyloid_candidates",
    "write_amyloid_geojson",
    "write_amyloid_overlay",
    "write_amyloid_tables",
]
