"""Why a results cell is missing: `results_missing_core.csv` and `results_missing_donor_region.csv` (stage 5).

Long format: the id columns of the results table, then `measure` (the column), `reason` and `n_objects` (the objects the
measure rests on). Every empty cell of `results_<level>.csv` has exactly one row, also the `_whole_core` columns.

| reason | meaning |
|---|---|
| `no_image` | the core has no image of the measure's stain (a donor-region: none of its cores has) |
| `no_grey_matter` | the stain's core images have too little grey matter to measure (`v2.white_matter.min_grey_mm2`) |
| `too_few_objects` | fewer objects than the measure's minimum (`MEASURES`; `n_objects` says how many) |
| `not_measured` | anything else: e.g. the white-matter share of a core whose white matter is undetermined |

A count, a density or an area fraction is never missing because nothing was found: with the tissue measured it is 0, and
a fraction or a shape measure under its minimum stays missing (no pseudo-counts).

`n_objects` is what the minimum counts: outlines for the outline measures, scored objects for the type scores, plaques of
at least 15 µm for the compact fraction, density windows for the patchiness, thread pieces for the thread measures (their
minimum is 0.5 mm of thread), mature tangles for the tangle spacing; 0 for `no_image` and `no_grey_matter`, where nothing
was counted. A measure inside the white matter is `not_measured` where the core has under the minimum of white matter.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from stainid.tables import read_csv
from stainid.v2.params import STAINS

REASONS = ("too_few_objects", "no_grey_matter", "no_image", "not_measured")
IDS = ("core_id", "sample_region_id", "donor_id", "tma", "region", "disease_group")
WHOLE = "_whole_core"
COUNTS = "results_counts_{}.csv"  # per row of the results table: its tiles per stain and what the minimums count


@dataclass(frozen=True)
class Needs:
    """What a measure needs: `count` the objects it rests on (`n_objects`), and at least `minimum` of `tested` (default
    `count`) for a value."""
    count: str
    minimum: float
    tested: str = ""


def _family(names: tuple[str, ...], needs: Needs) -> dict[str, Needs]:
    return dict.fromkeys(names, needs)


_MASK_SHAPE = ("area_median_um2", "area_p90_um2", "area_cv", "solidity_median", "circularity_median", "elongation_median", "normalized_dab_median")
MEASURES: dict[str, Needs] = {
    **_family(tuple(f"neun_soma_{s}" for s in _MASK_SHAPE), Needs("neun_masks", 20)),
    **_family(tuple(f"plaque_{s}" for s in _MASK_SHAPE) + ("plaque_dense_core_fraction_median", "plaque_cored_fraction"), Needs("plaque_masks", 10)),
    **_family(tuple(f"tau_neuron_{s}" for s in _MASK_SHAPE) + ("tau_neuron_mask_mature_fraction",), Needs("tau_masks", 5)),
    **_family(("neun_median_profile_area_um2", "neun_median_profile_dab_od"), Needs("neun_neurons", 1)),
    "neun_positive_fraction_of_candidates": Needs("neun_candidates", 1),
    "neun_local_density_cv_100um": Needs("neun_windows", 4),
    "v2_plaque_mean_area_um2": Needs("plaques", 1),
    "plaque_dense_core_area_fraction": Needs("plaques", 10),
    **_family(("compact_fraction_v2", "plaque_median_dense_core_fraction", "plaque_circularity_median"), Needs("plaques_ge15um", 10)),
    **_family(("periplaque_nucleus_enrichment", "nuclei_per_plaque"), Needs("plaques_with_nuclei", 10)),
    **_family(("tau_neuron_mature_fraction", "tau_neuron_median_soma_um2", "thread_to_neuron_ratio_mm"), Needs("tau_neurons", 5)),
    **_family(("thread_mean_width_um", "thread_branchpoints_per_mm", "thread_attached_share"), Needs("thread_pieces", 0.5, "thread_mm")),
    "tangle_nn_median_um": Needs("tangles_with_neighbour", 10),
    "tangle_l200_excess_um": Needs("tangles", 10, "tangles_largest_core"),
    **_family(("tau_processes_per_neuron", "tau_attached_thread_um_per_neuron", "tau_neurons_with_process_fraction"), Needs("owned_neurons", 5)),
    **_family(("tau_processes_per_neuron_mature", "tau_attached_thread_um_per_neuron_mature"), Needs("owned_mature", 5)),
    **_family(("tau_processes_per_neuron_pretangle", "tau_attached_thread_um_per_neuron_pretangle"), Needs("owned_pretangle", 5)),
    **_family(("neun_edge_touch_fraction", "neun_edge_mostly_fraction"), Needs("neun_neurons", 20)),
    **_family(("plaque_edge_touch_fraction", "plaque_edge_mostly_fraction"), Needs("plaques", 10)),
    **_family(("tau_neuron_edge_touch_fraction", "tau_neuron_edge_mostly_fraction", "tau_neuron_boundary_fraction"), Needs("tau_neurons", 5)),
}
WHITE_MATTER_MEASURES = ("plaque_density_white_mm2", "plaque_area_fraction_white", "amyloid_positive_area_fraction_white", "tau_neuron_density_white_mm2",
                         "thread_density_white_mm_per_mm2", "at8_positive_area_fraction_white", "plaque_white_area_mm2", "tau_white_area_mm2")
for _prefix, _count, _minimum in (("tau_maturity", "tau_scored", 5), ("plaque_compactness", "plaque_scored", 10)):
    _bins = {"tau_maturity": ("pretangle", "intermediate", "mature"), "plaque_compactness": ("diffuse", "intermediate", "compact")}[_prefix]
    MEASURES |= _family(tuple(f"{_prefix}_score_{s}" for s in ("median", "iqr", "mean")) + tuple(f"{_prefix}_{b}_fraction" for b in _bins),
                        Needs(_count, _minimum))
STAIN_PREFIX = (("neun_", "NeuN"), ("white_matter_", "NeuN"), ("tau_", "AT8"), ("tangle_", "AT8"), ("thread_", "AT8"), ("at8_", "AT8"), ("plaque_", "6E10"),
                ("amyloid_", "6E10"), ("v2_plaque", "6E10"), ("compact_fraction", "6E10"), ("periplaque_", "6E10"), ("nuclei_per_plaque", "6E10"),
                ("vascular_or_edge_", "6E10"))


def stain_of(measure: str) -> str | None:
    """The stain a results column is measured on (None: not one stain's)."""
    for prefix, stain in STAIN_PREFIX:
        if measure.startswith(prefix):
            return stain
    return None


def _number(row: dict, key: str) -> float:
    value = row.get(key, 0)
    return 0.0 if value in ("", None) else float(value)


def group_counts(tiles: dict[str, list[dict]], objects: dict[str, list[dict]], masks: dict[str, list[dict]], scored: dict[str, list[dict]]) -> dict:
    """One row of the results table's counts, from its rows per stain (stage 5's text rows)."""
    plaques, taus, neurons = tiles.get("6E10", []), tiles.get("AT8", []), tiles.get("NeuN", [])
    out: dict[str, float] = {f"tiles_{stain}": len(tiles.get(stain, [])) for stain in STAINS}
    out |= {"neun_masks": len(masks.get("NeuN", [])), "plaque_masks": len(masks.get("6E10", [])), "tau_masks": len(masks.get("AT8", [])),
            "neun_neurons": len(objects.get("NeuN", [])),
            "neun_candidates": round(sum(_number(r, "candidate_profile_count") for r in neurons)),
            "neun_windows": round(sum(_number(r, "neun_local_density_window_count") for r in neurons)),
            "plaques": round(sum(_number(r, "v2_plaque_count") for r in plaques)),
            "plaques_ge15um": sum(_number(r, "equivalent_diameter_um") >= 15.0 for r in objects.get("6E10", [])),
            "plaques_with_nuclei": round(sum(_number(r, "v2_plaque_count") for r in plaques if r.get("nuclei_available", "true") == "true")),
            "tau_neurons": round(sum(_number(r, "tau_neuron_count") for r in taus)),
            "thread_pieces": round(sum(_number(r, "thread_piece_count") for r in taus)),
            "thread_mm": sum(_number(r, "thread_ridge_length_mm") for r in taus),
            "tau_scored": sum(bool(r.get("tau_maturity_bin")) for r in scored.get("AT8", [])),
            "plaque_scored": sum(bool(r.get("plaque_compactness_bin")) for r in scored.get("6E10", []))}
    return out


def _reason(measure: str, row_id: str, measured: dict, whole: dict | None) -> tuple[str, int]:
    """The reason of one empty cell. `measured`: the counts of the table the column comes from, `whole`: the whole-core
    table's when the column is grey matter (an image with no grey matter to measure has tiles there and none here)."""
    name = measure[: -len(WHOLE)] if measure.endswith(WHOLE) else measure
    stain = stain_of(name)
    counts = measured.get(row_id, {})
    if stain is not None:
        known = whole if whole is not None else measured
        if not known.get(row_id, {}).get(f"tiles_{stain}", 0):
            return "no_image", 0
        if name in WHITE_MATTER_MEASURES:
            return "not_measured", 0
        if not counts.get(f"tiles_{stain}", 0):
            return "no_grey_matter", 0
    needs = MEASURES.get(name)
    if needs is None:
        return "not_measured", 0
    n_objects = round(counts.get(needs.count, 0))
    if counts.get(needs.tested or needs.count, 0) < needs.minimum:
        return "too_few_objects", n_objects
    return "not_measured", n_objects


def _counts(folder: Path, name: str, key: str) -> dict:
    path = folder / COUNTS.format(name)
    if not path.exists():
        return {}
    return {row[key]: {k: _number(row, k) for k in row if k not in IDS} for row in read_csv(path)}


def write(folder: Path, whole_folder: Path | None = None) -> list[str]:
    """`results_missing_<level>.csv` for the results tables in `folder`; `whole_folder`: where the whole-core tables are
    when `folder` is grey matter only."""
    lines = []
    for level, key in (("core", "core_id"), ("donor_region", "sample_region_id")):
        path = folder / f"results_{level}.csv"
        if not path.exists():
            continue
        table = pd.read_csv(path, low_memory=False, dtype={"donor_id": str})
        ids = [c for c in IDS if c in table.columns]
        main, whole = _counts(folder, level, key), _counts(whole_folder, level, key) if whole_folder else None
        rows = []
        for column in [c for c in table.columns if c not in ids]:
            empty = table[column].isna().to_numpy()
            if not empty.any():
                continue
            is_whole = column.endswith(WHOLE)
            measured, known = (whole, None) if is_whole and whole is not None else (main, whole)
            for index in np.flatnonzero(empty):
                record = table.iloc[index]
                reason, n_objects = _reason(column, str(record[key]), measured, known)
                rows.append({**{c: record[c] for c in ids}, "measure": column, "reason": reason, "n_objects": n_objects})
        frame = pd.DataFrame(rows, columns=[*ids, "measure", "reason", "n_objects"])
        frame.to_csv(folder / f"results_missing_{level}.csv", index=False)
        tally = frame.reason.value_counts().to_dict() if len(frame) else {}
        lines.append(f"results_missing_{level}: {len(frame)} empty cells ({', '.join(f'{k} {v}' for k, v in sorted(tally.items()))})")
    return lines
