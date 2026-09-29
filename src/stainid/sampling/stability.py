from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from stainid.tables import format_csv_value, write_csv

PRIMARY_ENDPOINTS = {
    "6E10": "compact_plaque_fraction",
    "AT8": "tau_noncompact_area_fraction_of_at8",
    "NeuN": "neun_profile_density_mm2",
}

STABILITY_ENDPOINTS = (
    ("6E10", "compact_plaque_fraction"),
    ("AT8", "tau_noncompact_area_fraction_of_at8"),
    ("NeuN", "neun_profile_density_mm2"),
    ("NeuN", "neun_candidate_profile_density_mm2"),
    ("NeuN", "neun_positive_fraction_of_candidates"),
    ("NeuN", "neun_positive_profile_area_fraction"),
    ("NeuN", "neun_local_density_cv_100um"),
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _correlation(first: np.ndarray, second: np.ndarray) -> float:
    if first.size < 2 or np.std(first) == 0 or np.std(second) == 0:
        return float("nan")
    return float(np.corrcoef(first, second)[0, 1])


def compare_sampling_extents(
    rows: list[dict[str, str]],
    endpoints: dict[str, str] | tuple[tuple[str, str], ...] = STABILITY_ENDPOINTS,
) -> list[dict[str, object]]:
    indexed = {
        (row["core_id"], row["stain"], row["sampling_extent"]): row
        for row in rows
    }
    output = []
    endpoint_pairs = endpoints.items() if isinstance(endpoints, dict) else endpoints
    for stain, endpoint in endpoint_pairs:
        pairs = []
        core_ids = sorted(
            {
                row["core_id"]
                for row in rows
                if row["stain"] == stain
                and row["sampling_extent"] == "primary_four"
            }
        )
        for core_id in core_ids:
            primary = indexed[(core_id, stain, "primary_four")].get(endpoint, "")
            extended = indexed[(core_id, stain, "extended_eight")].get(endpoint, "")
            if primary and extended and np.isfinite(float(primary)) and np.isfinite(float(extended)):
                pairs.append((float(primary), float(extended)))
        values = np.asarray(pairs, dtype=float)
        first = values[:, 0] if values.size else np.asarray([], dtype=float)
        second = values[:, 1] if values.size else np.asarray([], dtype=float)
        difference = second - first
        scale = (np.abs(first) + np.abs(second)) / 2.0
        relative = np.abs(difference[scale > 0]) / scale[scale > 0]
        output.append(
            {
                "stain": stain,
                "endpoint": endpoint,
                "paired_core_count": len(pairs),
                "pearson_r": _correlation(first, second),
                "spearman_r": _correlation(rankdata(first), rankdata(second)),
                "median_absolute_difference": (
                    float(np.median(np.abs(difference)))
                    if difference.size
                    else float("nan")
                ),
                "median_relative_absolute_difference": (
                    float(np.median(relative)) if relative.size else float("nan")
                ),
                "mean_four_fields": (
                    float(np.mean(first)) if first.size else float("nan")
                ),
                "mean_eight_fields": (
                    float(np.mean(second)) if second.size else float("nan")
                ),
            }
        )
    return output


def write_sampling_stability(core_features: Path, output: Path) -> Path:
    rows = compare_sampling_extents(read_csv(core_features))
    write_csv(
        output,
        [{key: format_csv_value(value) for key, value in row.items()} for row in rows],
        list(rows[0]),
    )
    return output


__all__ = [
    "PRIMARY_ENDPOINTS",
    "STABILITY_ENDPOINTS",
    "compare_sampling_extents",
    "write_sampling_stability",
]
