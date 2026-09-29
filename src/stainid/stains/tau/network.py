from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
from skimage.morphology import skeletonize

from stainid.imaging.color import rgb_to_hed
from stainid.imaging.tissue import field_tissue_mask
from stainid.stains.neun.candidates import mask_feature
from stainid.tables import format_csv_value, write_csv


def _remove_small_components(mask: np.ndarray, minimum_pixels: int) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), 8
    )
    output = np.zeros(mask.shape, dtype=np.uint8)
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] >= minimum_pixels:
            output[labels == label] = 1
    return output.astype(bool)


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
    covariance = np.cov(np.column_stack((xs, ys)).astype(np.float64), rowvar=False)
    eigenvalues = np.sort(np.linalg.eigvalsh(covariance))[::-1]
    major_axis_um = 4.0 * np.sqrt(max(eigenvalues[0], 0.0) * pixel_area_um2)
    minor_axis_um = 4.0 * np.sqrt(max(eigenvalues[-1], 0.0) * pixel_area_um2)
    return {
        "centroid_x_px": float(xs.mean()),
        "centroid_y_px": float(ys.mean()),
        "perimeter_um": perimeter_px * np.sqrt(pixel_area_um2),
        "circularity": 4.0 * np.pi * contour_area_px / perimeter_px**2 if perimeter_px else 0.0,
        "solidity": contour_area_px / hull_area_px if hull_area_px else 0.0,
        "aspect_ratio": max(width / height, height / width),
        "major_axis_um": float(major_axis_um),
        "minor_axis_um": float(minor_axis_um),
    }


def _orientation_entropy(dab: np.ndarray, mask: np.ndarray) -> float:
    if not mask.any():
        return float("nan")
    smooth = cv2.GaussianBlur(dab, (0, 0), 1.0)
    gradient_x = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.hypot(gradient_x, gradient_y)[mask]
    orientation = np.mod(np.arctan2(gradient_y[mask], gradient_x[mask]), np.pi)
    histogram, _ = np.histogram(
        orientation, bins=18, range=(0.0, np.pi), weights=magnitude
    )
    probabilities = histogram / histogram.sum() if histogram.sum() else histogram
    probabilities = probabilities[probabilities > 0]
    return float(-(probabilities * np.log(probabilities)).sum() / np.log(18))


def _patch_cv(
    positive: np.ndarray,
    valid: np.ndarray,
    block_um: float,
    pixel_size_um: float,
) -> float:
    block = max(1, round(block_um / pixel_size_um))
    fractions = []
    for y in range(0, valid.shape[0], block):
        for x in range(0, valid.shape[1], block):
            local_valid = valid[y : y + block, x : x + block]
            if local_valid.sum() < 0.5 * local_valid.size:
                continue
            local_positive = positive[y : y + block, x : x + block]
            fractions.append(float(local_positive[local_valid].mean()))
    if not fractions or np.mean(fractions) == 0:
        return float("nan")
    return float(np.std(fractions) / np.mean(fractions))


def _local_density(mask: np.ndarray, valid: np.ndarray, sigma_px: float) -> np.ndarray:
    height, width = mask.shape
    scale = max(1, int(sigma_px // 8))
    small = (max(1, width // scale), max(1, height // scale))

    def blur(image: np.ndarray) -> np.ndarray:
        reduced = cv2.resize(image.astype(np.float32), small, interpolation=cv2.INTER_AREA)
        return cv2.resize(cv2.GaussianBlur(reduced, (0, 0), sigma_px / scale), (width, height), interpolation=cv2.INTER_LINEAR)

    numerator = blur(mask)
    denominator = blur(valid)
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator >= 0.25,
    )


def _external_tissue_edge_guard(tissue: np.ndarray, radius_px: float) -> np.ndarray:
    _, labels = cv2.connectedComponents((~tissue).astype(np.uint8), 8)
    border_labels = np.unique(
        np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1]))
    )
    border_labels = border_labels[border_labels > 0]
    if border_labels.size == 0:
        return np.zeros(tissue.shape, dtype=bool)
    external_background = np.isin(labels, border_labels)
    distance = cv2.distanceTransform(
        (~external_background).astype(np.uint8), cv2.DIST_L2, 5
    )
    return tissue & (distance <= radius_px)


def segment_at8_network(
    rgb: np.ndarray,
    dab_threshold: float,
    pixel_width_um: float,
    pixel_height_um: float,
    exclusion_mask: np.ndarray | None = None,
    tangle_boxes: list[dict[str, float]] | None = None,
    edge_guard_um: float = 25.0,
    hole_rim_um: float = 5.0,
) -> tuple[
    dict[str, float | int],
    list[dict[str, float | int | bool | str]],
    dict[str, np.ndarray],
]:
    tissue = field_tissue_mask(rgb)
    pixel_area_um2 = pixel_width_um * pixel_height_um
    pixel_size_um = np.sqrt(pixel_area_um2)
    manual_excluded = np.zeros(tissue.shape, dtype=bool)
    if exclusion_mask is not None:
        if exclusion_mask.shape != tissue.shape:
            raise ValueError("AT8 exclusion mask dimensions do not match the image")
        manual_excluded |= exclusion_mask
    base_valid = tissue & ~manual_excluded
    if base_valid.sum() < 100:
        raise ValueError("AT8 network analysis requires at least 100 tissue pixels")

    dab = np.maximum(rgb_to_hed(rgb)[..., 2], 0.0).astype(np.float32)
    background_median = float(np.median(dab[base_valid]))
    background_mad = float(
        np.median(np.abs(dab[base_valid] - background_median))
    )
    background_sigma = max(1.4826 * background_mad, 0.001)
    threshold = max(
        0.012,
        float(dab_threshold),
        background_median + 5.0 * background_sigma,
    )
    smooth_dab = cv2.GaussianBlur(dab, (0, 0), 0.8)
    preliminary_positive = base_valid & (smooth_dab >= threshold)
    preliminary_positive = cv2.morphologyEx(
        preliminary_positive.astype(np.uint8),
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    ).astype(bool)
    preliminary_positive = _remove_small_components(
        preliminary_positive, max(2, round(1.0 / pixel_area_um2))
    )
    edge_probe = _external_tissue_edge_guard(tissue, 15.0 / pixel_size_um)
    edge_positive_fraction = float(
        preliminary_positive[edge_probe].sum()
        / max(1, preliminary_positive.sum())
    )
    edge_density = float(preliminary_positive[edge_probe].mean()) if edge_probe.any() else 0.0
    interior = base_valid & ~edge_probe
    interior_density = float(preliminary_positive[interior].mean()) if interior.any() else 0.0
    edge_positive_enrichment = edge_density / max(interior_density, 1e-9)
    edge_guard_triggered = bool(
        preliminary_positive.sum() * pixel_area_um2 >= 20.0
        and edge_positive_fraction >= 0.25
        and edge_positive_enrichment >= 4.0
    )
    edge_excluded = (
        _external_tissue_edge_guard(tissue, edge_guard_um / pixel_size_um)
        if edge_guard_triggered
        else np.zeros(tissue.shape, dtype=bool)
    )
    edge_excluded_display = _remove_small_components(edge_excluded, 1_000)
    excluded = manual_excluded | edge_excluded
    valid = tissue & ~excluded
    if valid.sum() < 100:
        raise ValueError("AT8 network analysis requires at least 100 valid tissue pixels")
    positive = preliminary_positive & valid
    rim_radius = max(1, round(hole_rim_um / pixel_size_um))
    hole_band = tissue & cv2.dilate(
        (~tissue).astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rim_radius + 1, 2 * rim_radius + 1)),
    ).astype(bool)
    positive_count, positive_labels, positive_stats, _ = cv2.connectedComponentsWithStats(
        positive.astype(np.uint8), 8
    )
    band_pixels = np.bincount(positive_labels[hole_band], minlength=positive_count)
    rim_component = band_pixels / np.maximum(positive_stats[:, cv2.CC_STAT_AREA], 1) >= 0.6
    rim_component[0] = False
    hole_rim_removed = rim_component[positive_labels]
    positive &= ~hole_rim_removed

    distance_um = (
        cv2.distanceTransform(positive.astype(np.uint8), cv2.DIST_L2, 5)
        * pixel_size_um
    )
    compact_core = _remove_small_components(
        distance_um >= 1.8, max(2, round(3.0 / pixel_area_um2))
    )
    compact_radius = max(1, round(2.5 / pixel_size_um))
    compact_support = positive & cv2.dilate(
        compact_core.astype(np.uint8),
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * compact_radius + 1, 2 * compact_radius + 1)
        ),
    ).astype(bool)
    compact_count, compact_labels, compact_stats, _ = cv2.connectedComponentsWithStats(
        compact_support.astype(np.uint8), 8
    )

    boxes = tangle_boxes or []
    objects: list[dict[str, float | int | bool | str]] = []
    kept_compact_labels = np.zeros(compact_labels.shape, dtype=np.int32)
    next_label = 1
    for label in range(1, compact_count):
        mask = compact_labels == label
        area_um2 = int(mask.sum()) * pixel_area_um2
        if not 12.0 <= area_um2 <= 1_000.0:
            continue
        shape = _shape_features(mask, pixel_area_um2)
        if shape["aspect_ratio"] > 4.0 or shape["minor_axis_um"] < 3.0:
            continue
        overlapping = [
            row
            for row in boxes
            if float(row["x1_px"]) <= shape["centroid_x_px"] <= float(row["x2_px"])
            and float(row["y1_px"]) <= shape["centroid_y_px"] <= float(row["y2_px"])
        ]
        tangle_confidence = max(
            [float(row["confidence"]) for row in overlapping], default=float("nan")
        )
        kept_compact_labels[mask] = next_label
        objects.append(
            {
                "label": next_label,
                "source_kind": "compact_profile",
                "candidate_class": "consensus_soma" if overlapping else "soma_review",
                "suggested_class": "AT8-positive neuronal soma or NFT",
                "area_um2": area_um2,
                **shape,
                "mean_dab_od": float(dab[mask].mean()),
                "max_dab_od": float(dab[mask].max()),
                "tangle_tracer_overlap": bool(overlapping),
                "tangle_tracer_max_confidence": tangle_confidence,
                "review_status": "ai_proposed_development",
            }
        )
        next_label += 1

    compact_dilation = max(1, round(1.0 / pixel_size_um))
    compact_exclusion = cv2.dilate(
        (kept_compact_labels > 0).astype(np.uint8),
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * compact_dilation + 1, 2 * compact_dilation + 1),
        ),
    ).astype(bool)
    thread = positive & ~compact_exclusion
    thread_skeleton = skeletonize(thread)
    neighbor_count = cv2.filter2D(
        thread_skeleton.astype(np.uint8),
        cv2.CV_16S,
        np.ones((3, 3), dtype=np.uint8),
        borderType=cv2.BORDER_CONSTANT,
    ) - thread_skeleton.astype(np.int16)
    branchpoints = thread_skeleton & (neighbor_count >= 3)
    endpoints = thread_skeleton & (neighbor_count == 1)

    local_density = _local_density(thread, valid, 12.0 / pixel_size_um)
    broad_density = _local_density(thread, valid, 40.0 / pixel_size_um)
    cluster_seed = (
        valid
        & (local_density >= 0.035)
        & ((local_density - broad_density) >= 0.010)
    )
    cluster_seed = cv2.morphologyEx(
        cluster_seed.astype(np.uint8),
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    ).astype(bool)
    cluster_count, raw_cluster_labels, cluster_stats, _ = cv2.connectedComponentsWithStats(
        cluster_seed.astype(np.uint8), 8
    )
    cluster_labels = np.zeros(cluster_seed.shape, dtype=np.int32)
    next_cluster = 1
    for label in range(1, cluster_count):
        mask = raw_cluster_labels == label
        area_um2 = int(mask.sum()) * pixel_area_um2
        if not 150.0 <= area_um2 <= 20_000.0:
            continue
        shape = _shape_features(mask, pixel_area_um2)
        cluster_labels[mask] = next_cluster
        objects.append(
            {
                "label": next_cluster,
                "source_kind": "thread_cluster",
                "candidate_class": "thread_cluster",
                "suggested_class": "AT8 thread-rich region",
                "area_um2": area_um2,
                **shape,
                "mean_dab_od": float(dab[mask].mean()),
                "max_dab_od": float(dab[mask].max()),
                "mean_local_thread_fraction": float(local_density[mask].mean()),
                "tangle_tracer_overlap": False,
                "tangle_tracer_max_confidence": float("nan"),
                "review_status": "ai_proposed_development",
            }
        )
        next_cluster += 1

    tissue_area_mm2 = valid.sum() * pixel_area_um2 / 1_000_000.0
    skeleton_length_mm = thread_skeleton.sum() * pixel_size_um / 1_000.0
    compact_objects = [row for row in objects if row["source_kind"] == "compact_profile"]
    cluster_objects = [row for row in objects if row["source_kind"] == "thread_cluster"]
    summary = {
        "tissue_area_mm2": tissue_area_mm2,
        "background_median_dab_od": background_median,
        "background_mad_dab_od": background_mad,
        "positive_threshold_dab_od": threshold,
        "at8_positive_area_fraction": float(positive[valid].mean()),
        "compact_profile_count": len(compact_objects),
        "compact_profile_density_mm2": len(compact_objects) / tissue_area_mm2,
        "tangle_tracer_consensus_count": sum(
            row["candidate_class"] == "consensus_soma" for row in compact_objects
        ),
        "noncompact_at8_area_fraction": float(thread[valid].mean()),
        "thread_area_fraction": float(thread[valid].mean()),
        "thread_skeleton_length_mm_per_mm2": skeleton_length_mm / tissue_area_mm2,
        "thread_branchpoint_density_mm2": int(branchpoints.sum()) / tissue_area_mm2,
        "thread_endpoint_density_mm2": int(endpoints.sum()) / tissue_area_mm2,
        "thread_orientation_entropy": _orientation_entropy(dab, thread_skeleton),
        "at8_patch_cv_50um": _patch_cv(positive, valid, 50.0, pixel_size_um),
        "at8_patch_cv_100um": _patch_cv(positive, valid, 100.0, pixel_size_um),
        "thread_cluster_count": len(cluster_objects),
        "manual_exclusion_fraction": float(manual_excluded.mean()),
        "edge_guard_triggered": edge_guard_triggered,
        "edge_positive_fraction": edge_positive_fraction,
        "edge_positive_enrichment": edge_positive_enrichment,
        "edge_exclusion_fraction": float(edge_excluded.mean()),
        "hole_rim_removed_area_fraction": float(hole_rim_removed[valid].mean()),
        "total_exclusion_fraction": float(excluded.mean()),
        "field_qc_status": (
            "edge_stain_artifact_review"
            if edge_guard_triggered
            else "usable_development"
        ),
        "biological_training_eligible": not edge_guard_triggered,
        "review_status": "ai_proposed_development",
    }
    maps = {
        "valid": valid,
        "excluded": excluded,
        "manual_excluded": manual_excluded,
        "edge_excluded": edge_excluded,
        "edge_excluded_display": edge_excluded_display,
        "positive": positive,
        "hole_rim_removed": hole_rim_removed,
        "compact_labels": kept_compact_labels,
        "thread": thread,
        "thread_skeleton": thread_skeleton,
        "branchpoints": branchpoints,
        "endpoints": endpoints,
        "cluster_labels": cluster_labels,
        "local_density": local_density,
    }
    return summary, objects, maps


def write_at8_overlay(
    rgb: np.ndarray,
    objects: list[dict[str, float | int | bool | str]],
    maps: dict[str, np.ndarray],
    output_path: Path,
) -> Path:
    overlay = rgb.copy()
    skeleton = maps["thread_skeleton"]
    overlay[skeleton] = (
        0.35 * overlay[skeleton] + 0.65 * np.asarray([40, 175, 205])
    ).astype(np.uint8)
    for row in objects:
        labels = (
            maps["compact_labels"]
            if row["source_kind"] == "compact_profile"
            else maps["cluster_labels"]
        )
        mask = (labels == int(row["label"])).astype(np.uint8)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        color = {
            "consensus_soma": (220, 35, 35),
            "soma_review": (245, 145, 25),
            "thread_cluster": (20, 170, 120),
        }[str(row["candidate_class"])]
        cv2.drawContours(overlay, contours, -1, color, 3)
    display_excluded = maps["manual_excluded"] | maps["edge_excluded_display"]
    exclusion_contours, _ = cv2.findContours(
        display_excluded.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(overlay, exclusion_contours, -1, (180, 50, 190), 4)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(
        str(output_path),
        cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_JPEG_QUALITY, 95],
    ):
        raise OSError(f"Could not write AT8 network overlay: {output_path}")
    return output_path


def write_at8_geojson(
    objects: list[dict[str, float | int | bool | str]],
    maps: dict[str, np.ndarray],
    output_path: Path,
    image_id: str,
    proposal_source: str,
) -> Path:
    features = []
    counters = {"compact_profile": 0, "thread_cluster": 0}
    for row in objects:
        source_kind = str(row["source_kind"])
        counters[source_kind] += 1
        prefix = "S" if source_kind == "compact_profile" else "R"
        candidate_id = f"{image_id}_AI_{prefix}{counters[source_kind]:05d}"
        labels = maps["compact_labels"] if source_kind == "compact_profile" else maps["cluster_labels"]
        class_name = "Ambiguous AT8 object" if source_kind == "compact_profile" else "AT8 thread-rich region"
        color = (245, 145, 25) if source_kind == "compact_profile" else (20, 170, 120)
        feature = mask_feature(
            labels == int(row["label"]),
            candidate_id,
            class_name,
            color,
            {
                "proposal_source": proposal_source,
                "candidate_id": candidate_id,
                "candidate_class": str(row["candidate_class"]),
                "suggested_class": str(row["suggested_class"]),
                "source_kind": source_kind,
                "review_status": "ai_proposed_development",
                "tangle_tracer_overlap": str(row["tangle_tracer_overlap"]).lower(),
                "tangle_tracer_max_confidence": str(
                    format_csv_value(row["tangle_tracer_max_confidence"])
                ),
            },
            convex_hull=False,
        )
        if feature is not None:
            features.append(feature)
    exclusion_layers = (
        ("X", "manual_exclusion_region", maps["manual_excluded"]),
        ("E", "automatic_edge_guard", maps["edge_excluded_display"]),
    )
    for prefix, geometry_role, exclusion_mask in exclusion_layers:
        count, labels = cv2.connectedComponents(
            exclusion_mask.astype(np.uint8), connectivity=8
        )
        for index in range(1, count):
            candidate_id = f"{image_id}_AI_{prefix}{index:03d}"
            feature = mask_feature(
                labels == index,
                candidate_id,
                "AT8 artifact",
                (180, 50, 190),
                {
                    "proposal_source": proposal_source,
                    "candidate_id": candidate_id,
                    "candidate_class": "artifact_exclusion",
                    "geometry_role": geometry_role,
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


def write_at8_tables(
    objects: list[dict[str, float | int | bool | str]],
    summary: dict[str, float | int | str],
    object_path: Path,
    summary_path: Path,
    image_id: str,
) -> tuple[Path, Path]:
    rows = []
    counters = {"compact_profile": 0, "thread_cluster": 0}
    for row in objects:
        source_kind = str(row["source_kind"])
        counters[source_kind] += 1
        prefix = "S" if source_kind == "compact_profile" else "R"
        rows.append(
            {
                "image_id": image_id,
                "candidate_id": f"{image_id}_AI_{prefix}{counters[source_kind]:05d}",
                **{
                    key: format_csv_value(value)
                    for key, value in row.items()
                    if key != "label"
                },
            }
        )
    write_csv(
        object_path,
        rows,
        [
            "image_id",
            "candidate_id",
            "source_kind",
            "candidate_class",
            "suggested_class",
            "area_um2",
            "centroid_x_px",
            "centroid_y_px",
            "perimeter_um",
            "circularity",
            "solidity",
            "aspect_ratio",
            "major_axis_um",
            "minor_axis_um",
            "mean_dab_od",
            "max_dab_od",
            "mean_local_thread_fraction",
            "tangle_tracer_overlap",
            "tangle_tracer_max_confidence",
            "review_status",
        ],
    )
    write_csv(
        summary_path,
        [{"image_id": image_id, **{key: format_csv_value(value) for key, value in summary.items()}}],
    )
    return object_path, summary_path


def read_tangle_boxes(path: Path) -> list[dict[str, float]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return [
            {
                key: float(row[key])
                for key in ("x1_px", "y1_px", "x2_px", "y2_px", "confidence")
            }
            for row in csv.DictReader(handle)
            if row["inside_tissue"].lower() == "true"
        ]


__all__ = [
    "read_tangle_boxes",
    "segment_at8_network",
    "write_at8_geojson",
    "write_at8_overlay",
    "write_at8_tables",
]
