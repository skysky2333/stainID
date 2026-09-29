from __future__ import annotations

from collections import defaultdict

import numpy as np

from stainid.imaging.color import rgb_to_hed
from stainid.imaging.tissue import field_tissue_mask


def summarize_preview_burden(
    rgb: np.ndarray,
    threshold: float,
    native_width_px: int,
    native_height_px: int,
    pixel_width_um: float,
    pixel_height_um: float,
) -> dict[str, float | int]:
    tissue = field_tissue_mask(rgb)
    if tissue.sum() < 100:
        raise ValueError("Burden screening requires at least 100 tissue pixels")
    dab = np.maximum(rgb_to_hed(rgb)[..., 2], 0.0)
    positive = tissue & (dab >= threshold)
    screen_pixel_width_um = pixel_width_um * native_width_px / rgb.shape[1]
    screen_pixel_height_um = pixel_height_um * native_height_px / rgb.shape[0]
    screen_pixel_area_mm2 = (
        screen_pixel_width_um * screen_pixel_height_um / 1_000_000.0
    )
    tissue_pixels = int(tissue.sum())
    positive_pixels = int(positive.sum())
    return {
        "screen_width_px": rgb.shape[1],
        "screen_height_px": rgb.shape[0],
        "screen_pixel_width_um": screen_pixel_width_um,
        "screen_pixel_height_um": screen_pixel_height_um,
        "tissue_area_mm2": tissue_pixels * screen_pixel_area_mm2,
        "positive_area_mm2": positive_pixels * screen_pixel_area_mm2,
        "positive_area_fraction": positive_pixels / tissue_pixels,
        "mean_tissue_dab_od": float(dab[tissue].mean()),
        "p95_tissue_dab_od": float(np.quantile(dab[tissue], 0.95)),
    }


def aggregate_donor_regions(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["sample_region_id"]), str(row["stain"]))].append(row)
    aggregated = []
    for (_, stain), group in sorted(grouped.items()):
        tissue_area = sum(float(row["tissue_area_mm2"]) for row in group)
        positive_area = sum(float(row["positive_area_mm2"]) for row in group)
        reference = group[0]
        aggregated.append(
            {
                "sample_region_id": reference["sample_region_id"],
                "donor_id": reference["donor_id"],
                "region": reference["region"],
                "disease_group": reference["disease_group"],
                "cerad": reference["cerad"],
                "braak": reference["braak"],
                "stain": stain,
                "technical_core_count": len(group),
                "tissue_area_mm2": tissue_area,
                "positive_area_mm2": positive_area,
                "positive_area_fraction": positive_area / tissue_area,
            }
        )
    return aggregated


def common_range(first: np.ndarray, second: np.ndarray) -> tuple[float, float]:
    lower = max(float(np.min(first)), float(np.min(second)))
    upper = min(float(np.max(first)), float(np.max(second)))
    return lower, upper


def icc_oneway(pairs: np.ndarray) -> float:
    values = np.asarray(pairs, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2 or values.shape[0] < 2:
        raise ValueError("ICC requires at least two complete replicate pairs")
    subject_means = values.mean(axis=1)
    grand_mean = float(values.mean())
    mean_between = 2.0 * float(np.sum((subject_means - grand_mean) ** 2)) / (
        values.shape[0] - 1
    )
    mean_within = float(np.sum((values - subject_means[:, None]) ** 2)) / values.shape[0]
    denominator = mean_between + mean_within
    return (mean_between - mean_within) / denominator if denominator else float("nan")


__all__ = [
    "aggregate_donor_regions",
    "common_range",
    "icc_oneway",
    "summarize_preview_burden",
]
