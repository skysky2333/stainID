from __future__ import annotations

import json
import uuid
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage

from stainid.imaging.color import rgb_to_hed
from stainid.imaging.tissue import field_tissue_mask, linear_artifact_mask
from stainid.tables import format_csv_value, write_csv


def contour_is_simple(contour: np.ndarray) -> bool:
    points = [tuple(int(value) for value in point[0]) for point in contour]
    if len(points) < 3 or len(set(points)) != len(points):
        return False

    def orientation(a: tuple[int, int], b: tuple[int, int], c: tuple[int, int]) -> int:
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    def on_segment(a: tuple[int, int], b: tuple[int, int], c: tuple[int, int]) -> bool:
        return (
            min(a[0], b[0]) <= c[0] <= max(a[0], b[0])
            and min(a[1], b[1]) <= c[1] <= max(a[1], b[1])
        )

    def intersects(
        a: tuple[int, int],
        b: tuple[int, int],
        c: tuple[int, int],
        d: tuple[int, int],
    ) -> bool:
        values = (
            orientation(a, b, c),
            orientation(a, b, d),
            orientation(c, d, a),
            orientation(c, d, b),
        )
        if values[0] * values[1] < 0 and values[2] * values[3] < 0:
            return True
        return (
            (values[0] == 0 and on_segment(a, b, c))
            or (values[1] == 0 and on_segment(a, b, d))
            or (values[2] == 0 and on_segment(c, d, a))
            or (values[3] == 0 and on_segment(c, d, b))
        )

    count = len(points)
    for first in range(count):
        for second in range(first + 1, count):
            if second == (first + 1) % count or first == (second + 1) % count:
                continue
            if intersects(
                points[first],
                points[(first + 1) % count],
                points[second],
                points[(second + 1) % count],
            ):
                return False
    return True


def clean_seed_mask(mask: np.ndarray, minimum_pixels: int = 20) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), 8
    )
    clean = np.zeros(mask.shape, dtype=np.uint8)
    for index in range(1, count):
        if stats[index, cv2.CC_STAT_AREA] >= minimum_pixels:
            clean[labels == index] = 1
    return cv2.morphologyEx(
        clean,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    ).astype(bool)


def watershed_profiles(mask: np.ndarray, rgb: np.ndarray) -> np.ndarray:
    distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    peaks = (
        (distance == cv2.dilate(distance, np.ones((19, 19), dtype=np.uint8)))
        & (distance >= 3.0)
    )
    _, peak_labels = cv2.connectedComponents(peaks.astype(np.uint8))
    markers = np.zeros(mask.shape, dtype=np.int32)
    markers[~mask] = 1
    markers[peaks] = peak_labels[peaks] + 1
    return cv2.watershed(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), markers)


def segment_neun_candidates(
    rgb: np.ndarray,
    dab_threshold: float,
    pixel_width_um: float,
    pixel_height_um: float,
    artifact_mask: np.ndarray | None = None,
) -> tuple[
    dict[str, float | int],
    list[dict[str, float | int | bool | str]],
    np.ndarray,
]:
    initial_tissue = field_tissue_mask(rgb)
    artifact = linear_artifact_mask(rgb) if artifact_mask is None else artifact_mask
    if artifact.shape != initial_tissue.shape:
        raise ValueError("NeuN artifact mask dimensions do not match the image")
    tissue = initial_tissue & ~artifact
    if tissue.sum() < 100:
        raise ValueError("NeuN candidate detection requires at least 100 tissue pixels")
    hed = rgb_to_hed(rgb)
    hematoxylin = np.maximum(hed[..., 0], 0.0).astype(np.float32)
    dab = np.maximum(hed[..., 2], 0.0).astype(np.float32)
    signal = np.maximum(hematoxylin, 0.75 * dab)
    seed_threshold = max(0.035, float(np.quantile(signal[tissue], 0.78)))
    seed_mask = cv2.morphologyEx(
        (tissue & (signal >= seed_threshold)).astype(np.uint8),
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    ).astype(bool)
    seed_mask = clean_seed_mask(seed_mask)
    labels = watershed_profiles(seed_mask, rgb)
    pixel_area_um2 = pixel_width_um * pixel_height_um
    image_height, image_width = labels.shape
    objects: list[dict[str, float | int | bool | str]] = []
    for label, bounds in enumerate(ndimage.find_objects(labels), start=1):
        if label < 2 or bounds is None:
            continue
        y_slice, x_slice = bounds
        local = labels[y_slice, x_slice] == label
        area_um2 = int(local.sum()) * pixel_area_um2
        if not 8.0 <= area_um2 <= 650.0:
            continue
        y0 = max(0, y_slice.start - 5)
        y1 = min(image_height, y_slice.stop + 5)
        x0 = max(0, x_slice.start - 5)
        x1 = min(image_width, x_slice.stop + 5)
        expanded = labels[y0:y1, x0:x1] == label
        neighborhood = cv2.dilate(
            expanded.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
        ).astype(bool)
        values = dab[y_slice, x_slice][local]
        neighborhood_values = dab[y0:y1, x0:x1][neighborhood]
        ys, xs = np.nonzero(local)
        mean_dab = float(values.mean())
        p80_dab = float(np.quantile(neighborhood_values, 0.8))
        positive = mean_dab >= 0.5 * dab_threshold or p80_dab >= dab_threshold
        review = not positive and (
            mean_dab >= 0.4 * dab_threshold or p80_dab >= 0.75 * dab_threshold
        )
        objects.append(
            {
                "label": label,
                "centroid_x_px": float(x_slice.start + xs.mean()),
                "centroid_y_px": float(y_slice.start + ys.mean()),
                "area_um2": area_um2,
                "mean_hematoxylin_od": float(
                    hematoxylin[y_slice, x_slice][local].mean()
                ),
                "mean_dab_od": mean_dab,
                "neighborhood_dab_p80": p80_dab,
                "candidate_class": (
                    "positive" if positive else "review" if review else "negative"
                ),
                "neun_positive": bool(positive),
                "touches_edge": bool(
                    x_slice.start == 0
                    or y_slice.start == 0
                    or x_slice.stop == image_width
                    or y_slice.stop == image_height
                ),
            }
        )
    complete = [row for row in objects if not row["touches_edge"]]
    positive = [row for row in complete if row["neun_positive"]]
    review = [row for row in complete if row["candidate_class"] == "review"]
    tissue_area_mm2 = tissue.sum() * pixel_area_um2 / 1_000_000.0
    summary = {
        "seed_threshold_od": seed_threshold,
        "tissue_area_mm2": tissue_area_mm2,
        "candidate_profile_count": len(complete),
        "neun_positive_profile_count": len(positive),
        "neun_review_profile_count": len(review),
        "neun_positive_profile_density_mm2": (
            len(positive) / tissue_area_mm2 if tissue_area_mm2 else float("nan")
        ),
        "edge_profile_count": len(objects) - len(complete),
        "linear_artifact_area_fraction": float(
            artifact.sum() / initial_tissue.sum() if initial_tissue.sum() else 0.0
        ),
    }
    return summary, objects, labels


def write_candidate_overlay(
    rgb: np.ndarray,
    labels: np.ndarray,
    objects: list[dict[str, float | int | bool | str]],
    output_path: Path,
    artifact_mask: np.ndarray | None = None,
) -> Path:
    overlay = rgb.copy()
    for row in objects:
        label = int(row["label"])
        mask = (labels == label).astype(np.uint8)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        color = {
            "positive": (220, 35, 35),
            "review": (245, 145, 25),
            "negative": (30, 140, 230),
        }[str(row["candidate_class"])]
        cv2.drawContours(overlay, contours, -1, color, 2)
    if artifact_mask is not None:
        contours, _ = cv2.findContours(
            artifact_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(overlay, contours, -1, (180, 50, 190), 4)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(
        str(output_path),
        cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_JPEG_QUALITY, 95],
    ):
        raise OSError(f"Could not write NeuN candidate overlay: {output_path}")
    return output_path


def mask_feature(
    mask: np.ndarray,
    name: str,
    class_name: str,
    color: tuple[int, int, int],
    metadata: dict[str, str],
    convex_hull: bool = True,
) -> dict[str, object] | None:
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if convex_hull:
        contour = cv2.approxPolyDP(cv2.convexHull(contour), 0.5, True)
    else:
        original = contour
        for epsilon in (1.0, 2.0, 3.0, 4.0, 5.0):
            contour = cv2.approxPolyDP(original, epsilon, True)
            if contour_is_simple(contour):
                break
        else:
            contour = cv2.approxPolyDP(cv2.convexHull(original), 0.5, True)
    coordinates = [[float(point[0][0]), float(point[0][1])] for point in contour]
    if len(coordinates) < 3:
        return None
    coordinates.append(coordinates[0])
    return {
        "type": "Feature",
        "id": str(uuid.uuid5(uuid.NAMESPACE_URL, name)),
        "geometry": {"type": "Polygon", "coordinates": [coordinates]},
        "properties": {
            "objectType": "detection",
            "name": name,
            "classification": {"name": class_name, "color": list(color)},
            "metadata": metadata,
        },
    }


def write_candidate_geojson(
    rgb: np.ndarray,
    labels: np.ndarray,
    objects: list[dict[str, float | int | bool | str]],
    output_path: Path,
    identifier: str,
    proposal_source: str = "neun_candidates",
) -> Path:
    class_properties = {
        "positive": ("NeuN-positive neuronal nucleus", (220, 35, 35)),
        "review": ("Ambiguous or truncated neuronal profile", (245, 145, 25)),
        "negative": ("NeuN-negative nucleus", (30, 140, 230)),
    }
    features = []
    for index, row in enumerate(objects, start=1):
        candidate_id = f"{identifier}_C{index:05d}"
        candidate_class = str(row["candidate_class"])
        class_name, color = class_properties[candidate_class]
        feature = mask_feature(
            labels == int(row["label"]),
            candidate_id,
            class_name,
            color,
            {
                "proposal_source": proposal_source,
                "candidate_id": candidate_id,
                "candidate_class": candidate_class,
                "touches_edge": str(row["touches_edge"]).lower(),
            },
        )
        if feature is not None:
            features.append(feature)
    artifact = linear_artifact_mask(rgb)
    count, artifact_labels = cv2.connectedComponents(
        artifact.astype(np.uint8), connectivity=8
    )
    for index in range(1, count):
        artifact_id = f"{identifier}_A{index:03d}"
        feature = mask_feature(
            artifact_labels == index,
            artifact_id,
            "NeuN artifact",
            (180, 50, 190),
            {
                "proposal_source": proposal_source,
                "candidate_id": artifact_id,
                "candidate_class": "artifact",
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


def write_candidate_results(
    rgb: np.ndarray,
    labels: np.ndarray,
    objects: list[dict[str, float | int | bool | str]],
    summary: dict[str, float | int],
    output_stem: Path,
    identifier_name: str,
    identifier: str,
    tma: str,
    dab_threshold: float,
) -> tuple[Path, Path, Path, Path]:
    summary_path = output_stem.with_name(f"{output_stem.name}_summary.csv")
    object_path = output_stem.with_suffix(".csv")
    overlay_path = output_stem.with_suffix(".jpg")
    geojson_path = output_stem.with_suffix(".geojson")
    write_csv(
        summary_path,
        [
            {
                identifier_name: identifier,
                "tma": tma,
                "dab_threshold_od": format_csv_value(dab_threshold),
                **{key: format_csv_value(value) for key, value in summary.items()},
            }
        ],
    )
    object_rows = [
        {
            identifier_name: identifier,
            "candidate_id": f"{identifier}_C{index:05d}",
            **{
                key: format_csv_value(value)
                for key, value in values.items()
                if key != "label"
            },
        }
        for index, values in enumerate(objects, start=1)
    ]
    write_csv(
        object_path,
        object_rows,
        [
            identifier_name,
            "candidate_id",
            "centroid_x_px",
            "centroid_y_px",
            "area_um2",
            "mean_hematoxylin_od",
            "mean_dab_od",
            "neighborhood_dab_p80",
            "candidate_class",
            "neun_positive",
            "touches_edge",
        ],
    )
    write_candidate_overlay(rgb, labels, objects, overlay_path)
    write_candidate_geojson(rgb, labels, objects, geojson_path, identifier)
    return summary_path, object_path, overlay_path, geojson_path


__all__ = [
    "segment_neun_candidates",
    "write_candidate_overlay",
    "write_candidate_geojson",
    "write_candidate_results",
]
