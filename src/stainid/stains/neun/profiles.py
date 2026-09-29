from __future__ import annotations

import json
import uuid
from pathlib import Path

import cv2
import numpy as np

from stainid.imaging.color import rgb_to_hed
from stainid.imaging.tissue import field_tissue_mask, linear_artifact_mask
from stainid.stains.neun.candidates import mask_feature
from stainid.tables import format_csv_value, write_csv

CLASS_PROPERTIES = {
    "positive_profile": ("NeuN-positive perikaryon", (220, 35, 35)),
    "positive_center": ("NeuN-positive profile center", (220, 35, 35)),
    "review": ("Ambiguous or truncated neuronal profile", (245, 145, 25)),
    "negative": ("NeuN-negative nucleus", (30, 140, 230)),
    "artifact": ("NeuN artifact", (180, 50, 190)),
}


def segment_dab_profiles(
    rgb: np.ndarray,
    dab_threshold: float,
    pixel_width_um: float,
    pixel_height_um: float,
    exclusion_mask: np.ndarray | None = None,
) -> tuple[list[dict[str, float | int | bool | str]], np.ndarray]:
    tissue = field_tissue_mask(rgb)
    artifact = linear_artifact_mask(rgb)
    if exclusion_mask is not None:
        if exclusion_mask.shape != artifact.shape:
            raise ValueError("NeuN exclusion mask dimensions do not match the image")
        artifact |= exclusion_mask
    hed = rgb_to_hed(rgb)
    hematoxylin = np.maximum(hed[..., 0], 0.0).astype(np.float32)
    dab = np.maximum(hed[..., 2], 0.0).astype(np.float32)
    chromogen_fraction = dab / (hematoxylin + dab + 1e-6)
    minimum_dab = max(0.012, 0.35 * dab_threshold)
    stained = (
        tissue
        & ~artifact
        & (dab >= minimum_dab)
        & (chromogen_fraction >= 0.50)
    )
    closed = cv2.morphologyEx(
        stained.astype(np.uint8),
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )
    count, component_labels, stats, _ = cv2.connectedComponentsWithStats(closed, 8)
    profile_labels = np.zeros(stained.shape, dtype=np.int32)
    pixel_area_um2 = pixel_width_um * pixel_height_um
    objects: list[dict[str, float | int | bool | str]] = []
    image_height, image_width = stained.shape
    for component in range(1, count):
        x = int(stats[component, cv2.CC_STAT_LEFT])
        y = int(stats[component, cv2.CC_STAT_TOP])
        width = int(stats[component, cv2.CC_STAT_WIDTH])
        height = int(stats[component, cv2.CC_STAT_HEIGHT])
        local = (
            component_labels[y : y + height, x : x + width] == component
        ).astype(np.uint8)
        contours, _ = cv2.findContours(
            local, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contour = max(contours, key=cv2.contourArea)
        stained_area_um2 = (
            int(stained[y : y + height, x : x + width][local.astype(bool)].sum())
            * pixel_area_um2
        )
        profile_area_um2 = float(cv2.contourArea(contour)) * pixel_area_um2
        aspect_ratio = max(width / height, height / width)
        if (
            stained_area_um2 < 4.0
            or not 15.0 <= profile_area_um2 <= 650.0
            or aspect_ratio > 3.5
        ):
            continue
        shifted = contour + np.array([[[x, y]]], dtype=contour.dtype)
        profile_mask = np.zeros(stained.shape, dtype=np.uint8)
        cv2.drawContours(profile_mask, [shifted], -1, 1, -1)
        profile_mask = profile_mask.astype(bool) & tissue & ~artifact
        ys, xs = np.nonzero(profile_mask)
        perimeter = float(cv2.arcLength(contour, True))
        contour_area = float(cv2.contourArea(contour))
        hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
        solidity = contour_area / hull_area if hull_area else 0.0
        circularity = 4.0 * np.pi * contour_area / perimeter**2 if perimeter else 0.0
        label = len(objects) + 1
        profile_labels[profile_mask] = label
        touches_edge = bool(
            x == 0
            or y == 0
            or x + width == image_width
            or y + height == image_height
        )
        accepted = bool(
            not touches_edge
            and 25.0 <= profile_area_um2 <= 350.0
            and aspect_ratio <= 2.5
            and solidity >= 0.55
            and circularity >= 0.15
        )
        if accepted:
            decision_basis = "stain_visible_compact_profile"
        elif touches_edge:
            decision_basis = "stain_visible_edge_truncated_profile"
        elif profile_area_um2 > 350.0:
            decision_basis = "stain_visible_large_or_merged_profile"
        elif profile_area_um2 < 25.0:
            decision_basis = "stain_visible_small_profile"
        else:
            decision_basis = "stain_visible_irregular_profile"
        objects.append(
            {
                "label": label,
                "centroid_x_px": float(xs.mean()),
                "centroid_y_px": float(ys.mean()),
                "stained_area_um2": stained_area_um2,
                "profile_area_um2": profile_area_um2,
                "aspect_ratio": aspect_ratio,
                "solidity": solidity,
                "circularity": circularity,
                "mean_dab_od": float(dab[profile_mask].mean()),
                "mean_chromogen_fraction": float(chromogen_fraction[profile_mask].mean()),
                "candidate_class": "positive_profile" if accepted else "review",
                "touches_edge": touches_edge,
                "geometry_role": "visible_perikaryon_boundary",
                "decision_basis": decision_basis,
            }
        )
    return objects, profile_labels


def combine_profile_and_cellpose_reviews(
    dab_threshold: float,
    profile_objects: list[dict[str, float | int | bool | str]],
    profile_labels: np.ndarray,
    cellpose_labels: np.ndarray,
    cellpose_objects: list[dict[str, float | int | bool | str]],
    exclusion_mask: np.ndarray | None = None,
) -> list[dict[str, float | int | bool | str]]:
    combined = [
        dict(row, source_kind="dab_profile", source_candidate_class="")
        for row in profile_objects
    ]
    supported = cv2.dilate(
        (profile_labels > 0).astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (49, 49)),
    ).astype(bool)
    review_dab = max(0.010, 0.25 * dab_threshold)
    height, width = profile_labels.shape
    for row in cellpose_objects:
        x = float(row["centroid_x_px"])
        y = float(row["centroid_y_px"])
        xi = min(width - 1, max(0, round(x)))
        yi = min(height - 1, max(0, round(y)))
        if exclusion_mask is not None:
            cellpose_mask = cellpose_labels == int(row["label"])
            if exclusion_mask[yi, xi] or exclusion_mask[cellpose_mask].mean() >= 0.25:
                continue
        if supported[yi, xi]:
            continue
        cellpose_mask = cellpose_labels == int(row["label"])
        contours, _ = cv2.findContours(
            cellpose_mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        contour = max(contours, key=cv2.contourArea)
        contour_area = float(cv2.contourArea(contour))
        perimeter = float(cv2.arcLength(contour, True))
        hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
        _, _, width_bound, height_bound = cv2.boundingRect(contour)
        aspect_ratio = max(
            width_bound / max(height_bound, 1),
            height_bound / max(width_bound, 1),
        )
        hematoxylin = float(row["mean_hematoxylin_od"])
        dab = float(row["mean_dab_od"])
        chromogen_fraction = dab / (hematoxylin + dab + 1e-6)
        source_candidate_class = str(row["candidate_class"])
        if source_candidate_class == "negative" and hematoxylin < 0.025:
            continue
        if (
            (dab >= review_dab and chromogen_fraction >= 0.24)
            or source_candidate_class in {"positive", "review"}
        ):
            candidate_class = "review"
            decision_basis = "unmatched_cellpose_uncertain_neun_signal"
        else:
            candidate_class = "negative"
            decision_basis = "unmatched_cellpose_neun_negative"
        combined.append(
            {
                **row,
                "candidate_class": candidate_class,
                "source_candidate_class": source_candidate_class,
                "chromogen_fraction": chromogen_fraction,
                "equivalent_diameter_um": 2.0
                * np.sqrt(float(row["area_um2"]) / np.pi),
                "aspect_ratio": aspect_ratio,
                "solidity": contour_area / hull_area if hull_area else 0.0,
                "circularity": 4.0 * np.pi * contour_area / perimeter**2
                if perimeter
                else 0.0,
                "geometry_role": "cellpose_boundary",
                "source_kind": "cellpose",
                "decision_basis": decision_basis,
            }
        )
    return combined


def point_feature(
    x: float,
    y: float,
    name: str,
    class_name: str,
    color: tuple[int, int, int],
    metadata: dict[str, str],
) -> dict[str, object]:
    return {
        "type": "Feature",
        "id": str(uuid.uuid5(uuid.NAMESPACE_URL, name)),
        "geometry": {"type": "Point", "coordinates": [x, y]},
        "properties": {
            "objectType": "detection",
            "name": name,
            "classification": {"name": class_name, "color": list(color)},
            "metadata": metadata,
        },
    }


def write_review_geojson(
    rgb: np.ndarray,
    profile_labels: np.ndarray,
    cellpose_labels: np.ndarray,
    objects: list[dict[str, float | int | bool | str]],
    output_path: Path,
    image_id: str,
    proposal_source: str,
    exclusion_mask: np.ndarray | None = None,
) -> Path:
    features = []
    counters = {"dab_profile": 0, "cellpose": 0}
    for row in objects:
        source_kind = str(row["source_kind"])
        counters[source_kind] += 1
        prefix = "P" if source_kind == "dab_profile" else "C"
        candidate_id = f"{image_id}_AI_{prefix}{counters[source_kind]:05d}"
        candidate_class = str(row["candidate_class"])
        class_name, color = CLASS_PROPERTIES[candidate_class]
        metadata = {
            "proposal_source": proposal_source,
            "candidate_id": candidate_id,
            "candidate_class": candidate_class,
            "source_kind": source_kind,
            "geometry_role": str(row["geometry_role"]),
            "decision_basis": str(row["decision_basis"]),
            "review_status": "ai_reviewed_development",
            "touches_edge": str(row["touches_edge"]).lower(),
        }
        for key in (
            "source_candidate_class",
            "stained_area_um2",
            "profile_area_um2",
            "area_um2",
            "aspect_ratio",
            "solidity",
            "circularity",
            "mean_dab_od",
            "mean_hematoxylin_od",
            "neighborhood_dab_p80",
            "chromogen_fraction",
            "mean_chromogen_fraction",
        ):
            if key in row:
                metadata[key] = str(format_csv_value(row[key]))
        if candidate_class == "positive_center":
            feature = point_feature(
                float(row["centroid_x_px"]),
                float(row["centroid_y_px"]),
                candidate_id,
                class_name,
                color,
                metadata,
            )
        else:
            labels = profile_labels if source_kind == "dab_profile" else cellpose_labels
            feature = mask_feature(
                labels == int(row["label"]),
                candidate_id,
                class_name,
                color,
                metadata,
                convex_hull=source_kind != "dab_profile",
            )
        if feature is not None:
            features.append(feature)
    artifact = linear_artifact_mask(rgb)
    if exclusion_mask is not None:
        artifact |= exclusion_mask
    count, artifact_labels = cv2.connectedComponents(
        artifact.astype(np.uint8), connectivity=8
    )
    for index in range(1, count):
        candidate_id = f"{image_id}_AI_A{index:03d}"
        class_name, color = CLASS_PROPERTIES["artifact"]
        feature = mask_feature(
            artifact_labels == index,
            candidate_id,
            class_name,
            color,
            {
                "proposal_source": proposal_source,
                "candidate_id": candidate_id,
                "candidate_class": "artifact",
                "source_kind": "artifact",
                "geometry_role": "exclusion_region",
                "review_status": "ai_reviewed_development",
                "touches_edge": "false",
            },
        )
        if feature is not None:
            features.append(feature)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump({"type": "FeatureCollection", "features": features}, handle, indent=2)
        handle.write("\n")
    return output_path


def write_review_overlay(
    rgb: np.ndarray,
    profile_labels: np.ndarray,
    cellpose_labels: np.ndarray,
    objects: list[dict[str, float | int | bool | str]],
    output_path: Path,
    exclusion_mask: np.ndarray | None = None,
) -> Path:
    overlay = rgb.copy()
    for row in objects:
        candidate_class = str(row["candidate_class"])
        color = CLASS_PROPERTIES[candidate_class][1]
        if candidate_class == "positive_center":
            center = (
                round(float(row["centroid_x_px"])),
                round(float(row["centroid_y_px"])),
            )
            cv2.drawMarker(overlay, center, color, cv2.MARKER_CROSS, 18, 2)
            continue
        labels = (
            profile_labels
            if str(row["source_kind"]) == "dab_profile"
            else cellpose_labels
        )
        mask = (labels == int(row["label"])).astype(np.uint8)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(overlay, contours, -1, color, 2)
    if exclusion_mask is not None:
        contours, _ = cv2.findContours(
            exclusion_mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(overlay, contours, -1, CLASS_PROPERTIES["artifact"][1], 4)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(
        str(output_path),
        cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_JPEG_QUALITY, 95],
    ):
        raise OSError(f"Could not write NeuN review overlay: {output_path}")
    return output_path


def review_table_rows(
    objects: list[dict[str, float | int | bool | str]],
    image_id: str,
) -> list[dict[str, str]]:
    measurement_fields = [
        "stained_area_um2",
        "profile_area_um2",
        "area_um2",
        "equivalent_diameter_um",
        "perimeter_um",
        "aspect_ratio",
        "solidity",
        "circularity",
        "mean_dab_od",
        "max_dab_od",
        "mean_hematoxylin_od",
        "neighborhood_dab_p80",
        "chromogen_fraction",
        "mean_chromogen_fraction",
        "tissue_fraction",
    ]
    rows = []
    counters = {"dab_profile": 0, "cellpose": 0}
    for row in objects:
        source_kind = str(row["source_kind"])
        counters[source_kind] += 1
        prefix = "P" if source_kind == "dab_profile" else "C"
        rows.append(
            {
                "image_id": image_id,
                "candidate_id": f"{image_id}_AI_{prefix}{counters[source_kind]:05d}",
                "candidate_class": row["candidate_class"],
                "source_candidate_class": row["source_candidate_class"],
                "source_kind": source_kind,
                "geometry_role": row["geometry_role"],
                "decision_basis": row["decision_basis"],
                "centroid_x_px": format_csv_value(row["centroid_x_px"]),
                "centroid_y_px": format_csv_value(row["centroid_y_px"]),
                "source_label": row["label"],
                **{
                    field: format_csv_value(row.get(field, ""))
                    for field in measurement_fields
                },
                "touches_edge": format_csv_value(row["touches_edge"]),
            }
        )
    return [{key: str(value) for key, value in row.items()} for row in rows]


def write_review_tables(
    objects: list[dict[str, float | int | bool | str]],
    object_path: Path,
    summary_path: Path,
    image_id: str,
    manual_exclusion_fraction: float = 0.0,
) -> tuple[Path, Path]:
    write_csv(object_path, review_table_rows(objects, image_id))
    counts = {
        name: sum(row["candidate_class"] == name for row in objects)
        for name in ("positive_profile", "positive_center", "review", "negative")
    }
    write_csv(
        summary_path,
        [
            {
                "image_id": image_id,
                **{f"{name}_count": value for name, value in counts.items()},
                "profile_count": counts["positive_profile"] + counts["positive_center"],
                "manual_exclusion_fraction": format_csv_value(
                    manual_exclusion_fraction
                ),
                "review_status": "ai_reviewed_development",
            }
        ],
    )
    return object_path, summary_path


__all__ = [
    "combine_profile_and_cellpose_reviews",
    "review_table_rows",
    "segment_dab_profiles",
    "write_review_geojson",
    "write_review_overlay",
    "write_review_tables",
]
