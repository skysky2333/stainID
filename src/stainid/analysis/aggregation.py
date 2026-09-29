from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from stainid.tables import format_csv_value, write_csv

COMMON_COLUMNS = (
    "tma",
    "donor_id",
    "sample_region_id",
    "region",
    "disease_group",
)

PLAQUE_COMPOSITION_MINIMUM_COUNT = 10
TAU_COMPOSITION_MINIMUM_AREA_MM2 = 0.005


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _number(row: dict[str, str], key: str) -> float:
    return float(row[key])


def _sum(rows: list[dict[str, str]], key: str) -> float:
    return sum(_number(row, key) for row in rows)


def _median(rows: list[dict[str, str]], key: str) -> float:
    values = np.asarray([_number(row, key) for row in rows], dtype=float)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if values.size else float("nan")


def _weighted_mean(rows: list[dict[str, str]], value: str, weight: str) -> float:
    pairs = np.asarray(
        [
            (_number(row, value), _number(row, weight))
            for row in rows
            if np.isfinite(_number(row, value)) and _number(row, weight) > 0
        ],
        dtype=float,
    )
    return float(np.average(pairs[:, 0], weights=pairs[:, 1])) if pairs.size else float("nan")


def _first_available(row: dict[str, str], *keys: str) -> float:
    for key in keys:
        if row.get(key, ""):
            return _number(row, key)
    raise KeyError(f"None of the requested fields are available: {keys}")


def summarize_stain_group(
    rows: list[dict[str, str]],
    objects: list[dict[str, str]],
) -> dict[str, object]:
    stain = rows[0]["stain"]
    tissue_area_mm2 = _sum(rows, "tissue_area_mm2")
    summary: dict[str, object] = {
        "stain": stain,
        "analyzed_tile_count": len(rows),
        "analyzed_core_count": len({row["core_id"] for row in rows}),
        "tissue_area_mm2": tissue_area_mm2,
        "mean_artifact_exclusion_area_fraction": float(
            np.mean(
                [
                    _number(row, "total_artifact_exclusion_area_fraction")
                    for row in rows
                ]
            )
        ),
    }
    if stain == "6E10":
        plaque_count = round(_sum(rows, "accepted_plaque_candidate_count"))
        eligible_count = round(_sum(rows, "morphotype_eligible_plaque_count"))
        compact_count = round(_sum(rows, "compact_candidate_count"))
        eligible = [
            row for row in objects if row.get("candidate_class") in {"compact", "diffuse"}
        ]
        plaque_area_mm2 = sum(
            _number(row, "accepted_plaque_deposit_area_fraction")
            * _number(row, "tissue_area_mm2")
            for row in rows
        )
        positive_area_mm2 = sum(
            _number(row, "amyloid_positive_area_fraction")
            * _number(row, "tissue_area_mm2")
            for row in rows
        )
        summary.update(
            {
                "accepted_plaque_count": plaque_count,
                "morphotype_eligible_plaque_count": eligible_count,
                "compact_plaque_count": compact_count,
                "diffuse_plaque_count": eligible_count - compact_count,
                "small_plaque_count": plaque_count - eligible_count,
                "accepted_plaque_area_mm2": plaque_area_mm2,
                "accepted_plaque_area_fraction": (
                    plaque_area_mm2 / tissue_area_mm2
                    if tissue_area_mm2
                    else float("nan")
                ),
                "amyloid_positive_area_fraction": (
                    positive_area_mm2 / tissue_area_mm2
                    if tissue_area_mm2
                    else float("nan")
                ),
                "plaque_density_mm2": (
                    plaque_count / tissue_area_mm2
                    if tissue_area_mm2
                    else float("nan")
                ),
                "compact_plaque_fraction": (
                    compact_count / eligible_count if eligible_count else float("nan")
                ),
                "plaque_composition_minimum_count": PLAQUE_COMPOSITION_MINIMUM_COUNT,
                "plaque_composition_eligible": (
                    eligible_count >= PLAQUE_COMPOSITION_MINIMUM_COUNT
                ),
                "plaque_median_area_um2": _median(eligible, "deposit_area_um2"),
                "plaque_boundary_irregularity_median": _median(
                    eligible, "boundary_irregularity"
                ),
            }
        )
    elif stain == "AT8":
        positive_area_mm2 = sum(
            _number(row, "at8_positive_area_fraction")
            * _number(row, "tissue_area_mm2")
            for row in rows
        )
        thread_area_mm2 = sum(
            _first_available(
                row, "noncompact_at8_area_fraction", "thread_area_fraction"
            )
            * _number(row, "tissue_area_mm2")
            for row in rows
        )
        compact_count = round(_sum(rows, "compact_profile_count"))
        compact = [row for row in objects if row.get("source_kind") == "compact_profile"]
        thread_length_mm = sum(
            _number(row, "thread_skeleton_length_mm_per_mm2")
            * _number(row, "tissue_area_mm2")
            for row in rows
        )
        summary.update(
            {
                "at8_positive_area_mm2": positive_area_mm2,
                "at8_noncompact_area_mm2": thread_area_mm2,
                "tau_positive_area_fraction": (
                    positive_area_mm2 / tissue_area_mm2
                    if tissue_area_mm2
                    else float("nan")
                ),
                "tau_noncompact_area_fraction_of_at8": (
                    thread_area_mm2 / positive_area_mm2
                    if positive_area_mm2
                    else float("nan")
                ),
                "tau_composition_minimum_positive_area_mm2": TAU_COMPOSITION_MINIMUM_AREA_MM2,
                "tau_composition_eligible": (
                    positive_area_mm2 >= TAU_COMPOSITION_MINIMUM_AREA_MM2
                ),
                "at8_compact_profile_count": compact_count,
                "at8_compact_profile_density_mm2": (
                    compact_count / tissue_area_mm2
                    if tissue_area_mm2
                    else float("nan")
                ),
                "at8_compact_profile_median_area_um2": _median(compact, "area_um2"),
                "at8_thread_length_mm": thread_length_mm,
                "at8_thread_length_density_mm_per_mm2": (
                    thread_length_mm / tissue_area_mm2
                    if tissue_area_mm2
                    else float("nan")
                ),
                "edge_guard_field_count": sum(
                    row["edge_guard_triggered"].lower() == "true" for row in rows
                ),
            }
        )
    elif stain == "NeuN":
        profile_count = round(_sum(rows, "neun_positive_profile_count"))
        candidate_count = round(_sum(rows, "candidate_profile_count"))
        positive_area_mm2 = sum(
            _number(row, "neun_positive_profile_area_fraction")
            * _number(row, "tissue_area_mm2")
            for row in rows
        )
        positive = [
            row
            for row in objects
            if row.get("candidate_class") == "positive"
            and row.get("touches_edge", "false").lower() == "false"
        ]
        summary.update(
            {
                "neun_candidate_profile_count": candidate_count,
                "neun_positive_profile_count": profile_count,
                "neun_review_profile_count": round(
                    _sum(rows, "neun_review_profile_count")
                ),
                "neun_union_profile_density_mm2": (
                    _sum(rows, "neun_union_positive_profile_count") / tissue_area_mm2
                    if tissue_area_mm2
                    else float("nan")
                ),
                "neun_profile_density_mm2": (
                    profile_count / tissue_area_mm2
                    if tissue_area_mm2
                    else float("nan")
                ),
                "neun_candidate_profile_density_mm2": (
                    candidate_count / tissue_area_mm2
                    if tissue_area_mm2
                    else float("nan")
                ),
                "neun_positive_fraction_of_candidates": (
                    profile_count / candidate_count
                    if candidate_count
                    else float("nan")
                ),
                "neun_positive_profile_area_fraction": (
                    positive_area_mm2 / tissue_area_mm2
                    if tissue_area_mm2
                    else float("nan")
                ),
                "neun_local_density_cv_100um": _weighted_mean(
                    rows,
                    "neun_local_density_cv_100um",
                    "neun_local_density_window_count",
                ),
                "neun_local_density_window_count": round(
                    _sum(rows, "neun_local_density_window_count")
                ),
                "neun_median_profile_area_um2": _median(positive, "area_um2"),
                "neun_median_profile_dab_od": _median(positive, "mean_dab_od"),
            }
        )
    else:
        raise ValueError(f"Unsupported stain: {stain}")
    return summary


def aggregate_level(
    tile_rows: list[dict[str, str]],
    object_rows: list[dict[str, str]],
    level: str,
) -> list[dict[str, object]]:
    if level not in {"core", "donor_region"}:
        raise ValueError(f"Unsupported aggregation level: {level}")
    key_columns = ("core_id", "stain") if level == "core" else ("sample_region_id", "stain")
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in tile_rows:
        grouped[tuple(row[column] for column in key_columns)].append(row)
    objects_by_tile: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in object_rows:
        objects_by_tile[row["tile_id"]].append(row)
    output: list[dict[str, object]] = []
    for _, group in sorted(grouped.items()):
        first = group[0]
        for extent, maximum_order in (("primary_four", 4), ("extended_eight", 8)):
            selected = [
                row for row in group if int(row["selection_order"]) <= maximum_order
            ]
            tile_ids = {row["tile_id"] for row in selected}
            selected_objects = [
                obj for tile_id in tile_ids for obj in objects_by_tile.get(tile_id, [])
            ]
            common: dict[str, object] = {
                column: first[column] for column in COMMON_COLUMNS
            }
            common.update(
                {
                    "analysis_level": level,
                    "sampling_extent": extent,
                    "technical_replicate": (
                        first["technical_replicate"] if level == "core" else "pooled"
                    ),
                }
            )
            if level == "core":
                common["core_id"] = first["core_id"]
            output.append(
                {**common, **summarize_stain_group(selected, selected_objects)}
            )
    return output


def pivot_donor_region(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["sample_region_id"]), str(row["sampling_extent"]))].append(row)
    prefixes = {"6E10": "amyloid", "AT8": "tau", "NeuN": "neun"}
    stain_specific = {
        "analyzed_tile_count",
        "analyzed_core_count",
        "tissue_area_mm2",
        "mean_artifact_exclusion_area_fraction",
    }
    output: list[dict[str, object]] = []
    shared = set(COMMON_COLUMNS) | {
        "analysis_level",
        "sampling_extent",
        "technical_replicate",
        "sample_region_id",
        "stain",
    }
    for _, group in sorted(grouped.items()):
        first = group[0]
        row: dict[str, object] = {
            column: first[column] for column in COMMON_COLUMNS
        }
        row["sampling_extent"] = first["sampling_extent"]
        row["available_stain_count"] = len(group)
        for stain_row in group:
            prefix = prefixes[str(stain_row["stain"])]
            for key, value in stain_row.items():
                if key not in shared:
                    output_key = f"{prefix}_{key}" if key in stain_specific else key
                    if output_key in row:
                        raise ValueError(f"Duplicate wide morphology feature: {output_key}")
                    row[output_key] = value
        output.append(row)
    return output


def write_aggregated_tables(
    tile_feature_path: Path,
    object_path: Path,
    core_output: Path,
    donor_region_output: Path,
    wide_output: Path,
) -> tuple[Path, Path, Path]:
    tiles = read_csv(tile_feature_path)
    objects = read_csv(object_path)
    core = aggregate_level(tiles, objects, "core")
    donor_region = aggregate_level(tiles, objects, "donor_region")
    wide = pivot_donor_region(donor_region)
    for path, rows in (
        (core_output, core),
        (donor_region_output, donor_region),
        (wide_output, wide),
    ):
        columns = list(dict.fromkeys(key for row in rows for key in row))
        write_csv(
            path,
            [{key: format_csv_value(value) for key, value in row.items()} for row in rows],
            columns,
        )
    return core_output, donor_region_output, wide_output


__all__ = [
    "aggregate_level",
    "pivot_donor_region",
    "summarize_stain_group",
    "write_aggregated_tables",
]
