from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from stainid.imaging.calibration import calibrate_threshold
from stainid.imaging.color import rgb_to_hed
from stainid.imaging.tissue import field_tissue_mask
from stainid.tables import format_csv_value, write_csv

MINIMUM_OBJECT_AREA_UM2 = {"6E10": 5.0, "AT8": 2.0, "NeuN": 5.0}


def component_features(
    positive: np.ndarray,
    dab: np.ndarray,
    pixel_width_um: float,
    pixel_height_um: float,
    minimum_area_um2: float,
) -> list[dict[str, float | int | bool]]:
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        positive.astype(np.uint8), 8
    )
    pixel_area_um2 = pixel_width_um * pixel_height_um
    minimum_pixels = max(1, round(minimum_area_um2 / pixel_area_um2))
    objects = []
    image_height, image_width = positive.shape
    for index in range(1, count):
        area_px = int(stats[index, cv2.CC_STAT_AREA])
        if area_px < minimum_pixels:
            continue
        x = int(stats[index, cv2.CC_STAT_LEFT])
        y = int(stats[index, cv2.CC_STAT_TOP])
        width = int(stats[index, cv2.CC_STAT_WIDTH])
        height = int(stats[index, cv2.CC_STAT_HEIGHT])
        local = (labels[y : y + height, x : x + width] == index).astype(np.uint8)
        contours, _ = cv2.findContours(local, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=cv2.contourArea)
        perimeter_px = float(cv2.arcLength(contour, True))
        hull_area_px = float(cv2.contourArea(cv2.convexHull(contour)))
        values = dab[labels == index]
        area_um2 = area_px * pixel_area_um2
        objects.append(
            {
                "centroid_x_px": float(centroids[index, 0]),
                "centroid_y_px": float(centroids[index, 1]),
                "area_um2": area_um2,
                "equivalent_diameter_um": 2.0 * np.sqrt(area_um2 / np.pi),
                "perimeter_um": perimeter_px * np.sqrt(pixel_area_um2),
                "circularity": 4.0 * np.pi * area_px / perimeter_px**2 if perimeter_px else 0.0,
                "solidity": area_px / hull_area_px if hull_area_px else 0.0,
                "aspect_ratio": max(width / height, height / width),
                "mean_dab_od": float(np.mean(values)),
                "max_dab_od": float(np.max(values)),
                "touches_edge": x == 0 or y == 0 or x + width == image_width or y + height == image_height,
            }
        )
    return objects


def summarize_field(
    rgb: np.ndarray,
    stain: str,
    threshold: float,
    pixel_width_um: float,
    pixel_height_um: float,
) -> tuple[dict[str, float | int], list[dict[str, float | int | bool]], np.ndarray, np.ndarray]:
    if stain not in MINIMUM_OBJECT_AREA_UM2:
        raise ValueError(f"Unsupported stain: {stain}")
    tissue = field_tissue_mask(rgb)
    dab = rgb_to_hed(rgb)[..., 2]
    positive = tissue & (dab >= threshold)
    objects = component_features(
        positive,
        dab,
        pixel_width_um,
        pixel_height_um,
        MINIMUM_OBJECT_AREA_UM2[stain],
    )
    complete = [row for row in objects if not row["touches_edge"]]
    tissue_pixels = int(tissue.sum())
    pixel_area_um2 = pixel_width_um * pixel_height_um
    areas = np.array([float(row["area_um2"]) for row in complete], dtype=float)
    summary = {
        "tissue_area_mm2": tissue_pixels * pixel_area_um2 / 1_000_000.0,
        "positive_area_fraction": float(positive.sum() / tissue_pixels) if tissue_pixels else float("nan"),
        "object_count": len(complete),
        "object_density_mm2": len(complete) / (tissue_pixels * pixel_area_um2 / 1_000_000.0) if tissue_pixels else float("nan"),
        "median_object_area_um2": float(np.median(areas)) if areas.size else float("nan"),
        "p90_object_area_um2": float(np.quantile(areas, 0.9)) if areas.size else float("nan"),
        "edge_object_count": len(objects) - len(complete),
    }
    return summary, objects, tissue, positive


def write_overlay(rgb: np.ndarray, tissue: np.ndarray, positive: np.ndarray, path: Path) -> None:
    scale = min(1.0, 768.0 / max(rgb.shape[:2]))
    size = (round(rgb.shape[1] * scale), round(rgb.shape[0] * scale))
    image = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
    tissue_small = cv2.resize(tissue.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST).astype(bool)
    positive_small = cv2.resize(positive.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST).astype(bool)
    overlay = image.astype(np.float32)
    overlay[~tissue_small] = 0.6 * overlay[~tissue_small] + 0.4 * np.array([255, 255, 255])
    overlay[positive_small] = 0.45 * overlay[positive_small] + 0.55 * np.array([220, 35, 35])
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), cv2.cvtColor(np.rint(overlay).astype(np.uint8), cv2.COLOR_RGB2BGR)):
        raise OSError(f"Could not write overlay: {path}")


def analyze_annotation_fields(
    annotation_manifest: Path,
    selection_key: Path,
    fields_dir: Path,
    summary_output: Path,
    object_output: Path,
    calibration_output: Path,
    overlay_dir: Path | None = None,
) -> tuple[Path, Path, Path]:
    with annotation_manifest.open(newline="", encoding="utf-8") as handle:
        images = list(csv.DictReader(handle))
    with selection_key.open(newline="", encoding="utf-8") as handle:
        keys = {row["annotation_id"]: row for row in csv.DictReader(handle)}
    groups: dict[tuple[int, str], list[dict[str, str]]] = defaultdict(list)
    for row in images:
        groups[(int(keys[row["annotation_id"]]["tma"]), row["stain"])].append(row)

    calibration_rows = []
    summary_rows = []
    object_rows = []
    for (tma, stain), group in sorted(groups.items()):
        loaded = []
        pooled = []
        for row in group:
            path = fields_dir / f"{row['image_id']}.png"
            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError(f"Could not read field: {path}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            tissue = field_tissue_mask(rgb)
            dab = rgb_to_hed(rgb)[..., 2]
            pooled.append(dab[tissue][::4])
            loaded.append((row, rgb))
        values = np.concatenate(pooled)
        threshold, quantiles = calibrate_threshold(values)
        calibration_rows.append(
            {
                "tma": tma,
                "stain": stain,
                "threshold_dab_od": f"{threshold:.8f}",
                "sampled_tissue_pixels": values.size,
                **{
                    name: f"{value:.8f}"
                    for name, value in zip(
                        ["dab_p50", "dab_p75", "dab_p90", "dab_p95", "dab_p99", "dab_p999"],
                        quantiles,
                    )
                },
            }
        )
        for row, rgb in loaded:
            summary, objects, tissue, positive = summarize_field(
                rgb,
                stain,
                threshold,
                float(row["pixel_width_um"]),
                float(row["pixel_height_um"]),
            )
            summary_rows.append(
                {
                    "annotation_id": row["annotation_id"],
                    "image_id": row["image_id"],
                    "tma": tma,
                    "stain": stain,
                    "threshold_dab_od": f"{threshold:.8f}",
                    **{key: format_csv_value(value) for key, value in summary.items()},
                }
            )
            for index, obj in enumerate(objects, start=1):
                object_rows.append(
                    {
                        "annotation_id": row["annotation_id"],
                        "image_id": row["image_id"],
                        "tma": tma,
                        "stain": stain,
                        "object_id": f"{row['image_id']}_O{index:05d}",
                        **{key: format_csv_value(value) for key, value in obj.items()},
                    }
                )
            if overlay_dir is not None:
                write_overlay(rgb, tissue, positive, overlay_dir / f"{row['image_id']}.jpg")

    write_csv(summary_output, summary_rows)
    write_csv(
        object_output,
        object_rows,
        [
            "annotation_id",
            "image_id",
            "tma",
            "stain",
            "object_id",
            "centroid_x_px",
            "centroid_y_px",
            "area_um2",
            "equivalent_diameter_um",
            "perimeter_um",
            "circularity",
            "solidity",
            "aspect_ratio",
            "mean_dab_od",
            "max_dab_od",
            "touches_edge",
        ],
    )
    write_csv(calibration_output, calibration_rows)
    return summary_output, object_output, calibration_output


__all__ = [
    "analyze_annotation_fields",
    "component_features",
    "summarize_field",
]


