"""Stage 5, measures: per-tile, per-core and per-donor-region tables from the stored outputs of stages 2–4 (no image is
read), in the column names the results pages and the R analyses already use.

Two families of tables, both under `<outputs.v2>/measures/`:
- `results_core.csv`, `results_donor_region.csv` (+ the `neun_`, `fields_`, `masks_` parts): the stainID results tables
  (`stainid.analysis.field_aggregation`, `mask_aggregation`, `aggregation.summarize_stain_group`), fed with tile and
  object rows made from the candidate store;
- `cohort_morphology_core_features.csv`, `_donor_region_features.csv`, `_donor_region_wide.csv`: the tables the frozen v1
  pipeline wrote for the R analyses (`stainid.analysis.aggregation.write_aggregated_tables`), now from v2: amyloid burden,
  compact-plaque fraction, tau burden etc. keep their names where the meaning is the same (docs/pipeline_v2.md maps them).

Thread length leaves out the skeleton inside accepted tau+ neurons (`thread_*`; the length with them is
`thread_length_with_somata_mm`). Per-tile tables: `tiles_<stain>.csv` (legacy-shaped inputs of both families).

The continuous type scores (`stainid.v2.scores`: AT8 tau maturity, 6E10 plaque compactness) are written per accepted
object (`type_scores_<stain>.csv`) and summarised as new columns of `results_<level>.csv` (`scores.readout_names`);
they add columns only, every other column is as without them.

The tangle spacing, the thread ownership and the share of objects at an edge (`stainid.v2.organisation`) and the
pathology inside the white matter (`_attach_white`) are new columns of `results_<level>.csv` too. So are the share of
accepted tau+ neurons flagged at a tissue boundary and their density without the flagged ones
(`organisation.mark_boundary`); the flag itself is per object in `objects_AT8.csv`, and nothing leaves any other count.

The spatial interactions (`stainid.v2.interactions`: what surrounds plaques and tau+ neurons, depth from the white matter,
microcolumns, multi-stain microdomains, the amyloid-tau coupling, the NeuN section's nuclear graph, plaque marks) are
new columns of `results_<level>.csv` (grey matter only with `exclude`, no `_whole_core` version), with their counts in
`interactions_<level>.csv`. `measure_reliability.csv` gives every results column's replicate-core ICC and its variance not
explained by the primary densities (`stainid.v2.reliability`).

`results_missing_<level>.csv` says why each empty cell of the results tables is empty (`stainid.v2.missing`). A count, a
density or an area fraction is 0 when nothing was found in measured tissue; only fractions and shape measures under their
minimum count, and tissue not measured at all, are missing.

White matter (`stainid.v2.white_matter`, docs/white_matter.md). With `v2.white_matter.analysis: exclude` every table is
grey matter only: objects whose centre lies in the core image's white matter (and AT8 thread pieces whose middle does)
are dropped, its counted cells leave every tile's area, and the 6E10 / AT8 stained areas are the grey-matter pass's
(`v2.grey_matter`, which must be up to date). The whole-core tables are written as before under `whole_core/`, and the
results and R tables also carry each whole-core number as `<column>_whole_core`. Whenever the white-matter step has run,
the results and R tables carry `white_matter_status`, `white_matter_share` (of the NeuN core image's analysed tissue;
donor-region: of the NeuN cores' together) and the NeuN-pale flag `neun_pale` (the core's NeuN stain too pale to read;
donor-region: any of its cores, `neun_pale_cores` how many), whatever the setting: pale cores are flagged, never left out.

A core image left with less grey matter than `v2.white_matter.min_grey_mm2` (default 1 mm²: a core that is white matter
as a whole keeps a sliver, and a density over a sliver is noise) is not measured as grey matter: its grey-matter row
keeps its identity and whole-core columns with every measure missing (NA), `grey_too_small` 1; a donor-region whose
cores are all like that is NA too (`grey_too_small` 1), otherwise it is measured from its other cores
(`grey_too_small_cores`: how many were left out).
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from stainid import candidates as store
from stainid.analysis import aggregation, field_aggregation, mask_aggregation
from stainid.analysis.aggregation import summarize_stain_group
from stainid.tables import read_csv, write_records
from stainid.v2 import biology_measures, cells, grey_matter, interactions, manifest, missing, organisation, reliability, scores, white_matter
from stainid.v2.classify import load_model
from stainid.v2.params import STAINS

REGION_KEYS = ("core_id", "sample_region_id", "donor_id", "tma", "region", "disease_group")
LEVELS = {"core_id": "core", "sample_region_id": "donor_region"}
CANDIDATE_COLUMNS = ["candidate_id", "tile_id", "x_core", "y_core", "area_um2", "decision", "reason", "class", "subtype", "probability",
                     "equivalent_diameter_um", "circularity", "solidity", "boundary_irregularity", "normalized_inner_dab", "core_dab_q",
                     "nuclei_inside", "nuclei_ring15", "ring15_area_um2", "nuclei_ring45", "ring45_area_um2", "touches_crop_edge",
                     "mean_dab_od", "m_selection_area_um2", "m_proposal_is_positive_profile", "m_soma_area_um2", "m_soma_dab_p90",
                     "m_soma_solidity", "m_soma_circularity", "m_soma_elongation", "m_soma_dab_mean", "m_normalized_contrast",
                     "m_deposit_area_um2", "m_boundary_irregularity", "legacy_class", "source",
                     "outline_default", "sam_status", "sam_area_um2", "sam_solidity", "sam_circularity", "sam_elongation", "sam_normalized_mean_dab",
                     "sam_core_dab_q", "edge_band_fraction", "excluded_fraction"]
CANDIDATE_COLUMNS += [c for c in biology_measures.CANDIDATE_COLUMNS if c not in CANDIDATE_COLUMNS]  # stage 2's biology (`biology`)
CANDIDATE_COLUMNS += [c for stain in scores.COLUMNS for c in scores.COLUMNS[stain] if c not in CANDIDATE_COLUMNS]  # the type scores' inputs


def fraction_at_least(quantiles: str, level: float) -> float:
    """The share of a region's pixels at or above `level` from its stored quantiles (0 %, 5 %, …, 100 %)."""
    values = np.asarray([float(v) for v in str(quantiles).split()]) if isinstance(quantiles, str) and quantiles.strip() else np.zeros(0)
    if not len(values) or not np.isfinite(level):
        return float("nan")
    if level <= values[0]:
        return 1.0
    if level > values[-1]:
        return 0.0
    probabilities = np.linspace(0, 1, len(values))
    return float(1.0 - np.interp(level, values, probabilities))


def core_images(project) -> dict[str, list[str]]:
    """The core images per stain that have calls (stage 4 done)."""
    return {stain: sorted(manifest.read_all(project, "calls", stain)) for stain in STAINS}


def by_tile(frame: pd.DataFrame) -> dict:
    """The rows of each tile (in their order), so a core's tiles are not each filtered out of the whole table (a whole
    slide has thousands of tiles); `.get(tile_id, empty)` for a tile without rows."""
    if frame.empty or "tile_id" not in frame:
        return {}
    return {tile_id: rows for tile_id, rows in frame.groupby("tile_id", sort=False)}


def _tiles(project, stain: str, core_id: str) -> pd.DataFrame:
    return store.read_table(store.path(project, stain, core_id, "tiles"))


def _local_cv(windows: str, size: str, points: np.ndarray, pixel_width_um: float, pixel_height_um: float) -> tuple[int, float]:
    values = [int(v) for v in str(windows).split()] if isinstance(windows, str) else []
    if not values:
        return 0, float("nan")
    width, height = (int(v) for v in str(size).split())
    area_mm2 = pixel_width_um * pixel_height_um / 1e6
    densities = []
    for x0, y0, tissue in zip(values[0::3], values[1::3], values[2::3]):
        inside = ((points[:, 0] >= x0) & (points[:, 0] < x0 + width) & (points[:, 1] >= y0) & (points[:, 1] < y0 + height)).sum() if len(points) else 0
        densities.append(inside / (tissue * area_mm2))
    mean = float(np.mean(densities))
    return len(densities), float(np.std(densities, ddof=1) / mean) if len(densities) >= 4 and mean > 0 else float("nan")


def neun_rows(tiles: pd.DataFrame, found: pd.DataFrame, addition_threshold: float) -> tuple[list[dict], list[dict]]:
    tile_rows, objects = [], []
    grouped, none = by_tile(found), found.iloc[0:0]
    for tile in tiles.to_dict("records"):
        mine = grouped.get(tile["tile_id"], none)
        scored = mine[~mine.reason.astype(str).str.startswith("rule:")]
        accepted = mine[mine.decision == "accepted"]
        probability = pd.to_numeric(scored.probability, errors="coerce").to_numpy(float)
        union = np.maximum(probability, pd.to_numeric(scored.m_proposal_is_positive_profile, errors="coerce").fillna(0).to_numpy(float))
        area = float(tile["analysis_area_mm2"])
        windows, cv = _local_cv(tile.get("density_windows"), tile.get("density_window_px"), accepted[["x_core", "y_core"]].to_numpy(float),
                                float(tile.get("pixel_width_um", 0.2738) or 0.2738), float(tile.get("pixel_height_um", 0.2738) or 0.2738))
        tile_rows.append({**tile, "tissue_area_mm2": area, "candidate_profile_count": len(scored), "v2_candidate_count": len(mine),
                          "neun_positive_profile_count": len(accepted),
                          "neun_union_positive_profile_count": int((union >= addition_threshold).sum()),
                          "neun_review_profile_count": int(((probability > 0.2) & (probability < 0.8)).sum()),
                          "neun_positive_profile_area_fraction": float(accepted.area_um2.sum() / 1e6 / area) if area else np.nan,
                          "neun_local_density_window_count": windows, "neun_local_density_cv_100um": cv,
                          "total_artifact_exclusion_area_fraction": tile["exclusion_fraction"]})
        for row in accepted.to_dict("records"):
            objects.append({**{k: tile[k] for k in REGION_KEYS if k in tile}, "tile_id": tile["tile_id"], "stain": "NeuN", "candidate_class": "positive",
                            "touches_edge": "false", "area_um2": row["area_um2"], "mean_dab_od": row["mean_dab_od"]})
    return tile_rows, objects


def amyloid_rows(tiles: pd.DataFrame, found: pd.DataFrame, morphotype: float, threshold_of: dict) -> tuple[list[dict], list[dict], list[dict]]:
    """6E10 tile rows for both families, the v1-shaped plaque rows and the stainID-shaped plaque rows."""
    tile_rows, v1_objects, objects = [], [], []
    grouped, none = by_tile(found), found.iloc[0:0]
    for tile in tiles.to_dict("records"):
        mine = grouped.get(tile["tile_id"], none)
        plaques = mine[mine.decision == "accepted"]
        core_fraction = np.array([fraction_at_least(q, morphotype) for q in plaques.core_dab_q]) if len(plaques) else np.zeros(0)
        eligible = plaques[plaques.subtype.isin(["compact", "diffuse"])]
        area = float(tile["analysis_area_mm2"])
        plaque_area = float(plaques.area_um2.sum())
        nuclei = float(tile.get("nucleus_count", 0) or 0)
        far_nuclei = max(0.0, nuclei - float(plaques.nuclei_inside.sum()) - float(plaques.nuclei_ring45.sum()))
        far_area = max(0.0, float(tile["valid_area_mm2"]) - (plaque_area + float(plaques.ring45_area_um2.sum())) / 1e6)
        tile_rows.append({
            **tile, "tissue_area_mm2": area, "v2_tissue_area_mm2": area,
            "accepted_plaque_candidate_count": len(plaques), "morphotype_eligible_plaque_count": len(eligible),
            "compact_candidate_count": int((eligible.subtype == "compact").sum()),
            "accepted_plaque_deposit_area_fraction": plaque_area / 1e6 / area if area else np.nan,
            "amyloid_positive_area_fraction": float(tile["amyloid_extent_area_mm2"]) / area if area else np.nan,
            "total_artifact_exclusion_area_fraction": tile["exclusion_fraction"],
            "v2_plaque_count": len(plaques), "v2_plaque_area_sum_um2": plaque_area,
            "v2_dense_core_area_sum_um2": float(np.nansum(core_fraction * plaques.area_um2.to_numpy(float))) if len(plaques) else 0.0,
            "v2_speck_count": int((mine.reason == "rule:speck").sum()), "v2_vascular_or_edge_count": int((mine.reason == "rule:vascular_or_edge").sum()),
            "v2_plaque_nuclei": int(plaques.nuclei_inside.sum()), "v2_ring_nuclei": int(plaques.nuclei_ring15.sum()),
            "v2_ring_area_um2": float(plaques.ring15_area_um2.sum()), "v2_far_nuclei": far_nuclei, "v2_far_area_mm2": far_area,
            "nuclei_available": str(bool(tile.get("nuclei_available", True))).lower(),
        })
        meta = {k: tile[k] for k in REGION_KEYS if k in tile} | {"tile_id": tile["tile_id"], "stain": "6E10"}
        for row, fraction in zip(plaques.to_dict("records"), core_fraction):
            v1_objects.append({**meta, "candidate_class": row["subtype"] if row["subtype"] in ("compact", "diffuse") else "small_plaque",
                               "deposit_area_um2": row["area_um2"], "boundary_irregularity": row["boundary_irregularity"]})
            objects.append({**meta, "plaque_class": "small_plaque" if row["subtype"] == "small" else row["subtype"],
                            "equivalent_diameter_um": row["equivalent_diameter_um"], "dense_core_fraction": fraction, "circularity": row["circularity"]})
    return tile_rows, v1_objects, objects


def tau_rows(project, tiles: pd.DataFrame, found: pd.DataFrame, threads: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    tile_rows, objects = [], []
    accepted_ids = set(found.candidate_id[found.decision == "accepted"])
    ps = project.pixel_size_um
    grouped, none = by_tile(found), found.iloc[0:0]
    pieces = by_tile(threads)
    for tile in tiles.to_dict("records"):
        mine = grouped.get(tile["tile_id"], none)
        neurons = mine[mine.decision == "accepted"]
        segments = pieces.get(tile["tile_id"], threads.iloc[0:0]) if not threads.empty else threads
        free = segments[~segments.candidate_id.isin(accepted_ids)] if not segments.empty else segments
        skeleton_px = int(free.n_px.sum()) if not free.empty else 0
        area = float(tile["analysis_area_mm2"])
        length_mm = skeleton_px * ps / 1000
        legacy_length = float(tile["at8_legacy_skeleton_px"]) * ps / 1000
        tile_rows.append({
            **tile, "tissue_area_mm2": area, "v2_tissue_area_mm2": area,
            "at8_positive_area_fraction": float(tile["at8_positive_area_mm2"]) / area if area else np.nan,
            "noncompact_at8_area_fraction": float(tile["at8_noncompact_area_mm2"]) / area if area else np.nan,
            "compact_profile_count": tile["at8_compact_profile_count"],
            "thread_skeleton_length_mm_per_mm2": legacy_length / area if area else np.nan,
            "edge_guard_triggered": str(bool(tile["edge_guard_triggered"])).lower(),
            "total_artifact_exclusion_area_fraction": tile["exclusion_fraction"],
            "tau_neuron_count": len(neurons), "tau_neuron_mature_count": int((neurons.subtype == "mature tangle").sum()),
            "thread_ridge_length_mm": length_mm, "thread_ridge_skeleton_px": skeleton_px,
            "thread_piece_count": int((free.kind == "piece").sum()) if not free.empty and "kind" in free else 0,
            "thread_ridge_width_sum_um": float(free.width_sum_um.sum()) if not free.empty else 0.0,
            "thread_length_with_somata_mm": float(tile["thread_skeleton_px"]) * ps / 1000,
            "thread_branchpoint_count": tile["thread_branchpoint_count"], "thread_endpoint_count": tile["thread_endpoint_count"],
            "nuclei_available": str(bool(tile.get("nuclei_available", True))).lower(),
        })
        meta = {k: tile[k] for k in REGION_KEYS if k in tile} | {"tile_id": tile["tile_id"], "stain": "AT8"}
        for row in neurons.to_dict("records"):
            flag = row.get(organisation.BOUNDARY_FLAG, np.nan)
            objects.append({**meta, "soma_area_um2": row["area_um2"], "candidate_id": row["candidate_id"],
                            organisation.BOUNDARY_FLAG: "" if pd.isna(flag) else int(flag),
                            organisation.BOUNDARY_AREA: "" if pd.isna(flag) else float(row[organisation.BOUNDARY_AREA])})
    return tile_rows, objects


def mask_rows(found: pd.DataFrame, tiles: pd.DataFrame, stain: str, morphotype: float | None) -> list[dict]:
    """Accepted objects with a kept Segment Anything outline, in the column names of the legacy outline step."""
    meta = {tile["tile_id"]: {k: tile[k] for k in REGION_KEYS if k in tile} for tile in tiles.to_dict("records")}
    rows = []
    for row in found[(found.decision == "accepted") & (found.sam_status == "ok")].to_dict("records"):
        out = {**meta[row["tile_id"]], "tile_id": row["tile_id"], "mask_status": "ok",
               "area_um2": row["sam_area_um2"], "solidity": row["sam_solidity"], "circularity": row["sam_circularity"],
               "elongation": row["sam_elongation"], "normalized_mean_dab": row["sam_normalized_mean_dab"]}
        if stain == "6E10":
            out["dense_core_fraction"] = fraction_at_least(row["sam_core_dab_q"], morphotype)
        if stain == "AT8":
            out["seed_mature"] = str(row["subtype"] == "mature tangle").lower()
        rows.append(out)
    return rows


def _write(path: Path, rows: list[dict]) -> Path:
    write_records(path, rows, ["tile_id"])
    return path


def _group(rows: list[dict], level: str) -> dict[str, list[dict]]:
    out = defaultdict(list)
    for row in rows:
        out[row[level]].append(row)
    return out


CELL_COLUMNS = ("neun_cell_density_mm2", "neun_nonneuronal_cell_density_mm2", "neun_neuron_fraction_of_cells")
MIN_CELLS = 20  # the neuron share of cells needs at least this many cells


def cell_columns(group: list[dict]) -> dict:
    """The NeuN section's cells (`cells`) over its tiles with nuclei: all cells (accepted neurons and the counterstain
    nuclei in no accepted neuron) per mm², the other cells per mm², and the neurons' share of all cells. A neuron is
    counted once whatever the nuclei inside it, so the share does not depend on whether Cellpose found its nucleus.
    Missing where no tile has nuclei (and the share under `MIN_CELLS` cells)."""
    def number(row, key):
        value = row.get(key, "")
        return float(value) if value not in ("", None) and value == value else np.nan

    have = [r for r in group if np.isfinite(number(r, "nonneuronal_cell_count"))]
    area = sum(number(r, "tissue_area_mm2") for r in have)
    if not have or area <= 0:
        return dict.fromkeys(CELL_COLUMNS, np.nan)
    neurons = sum(number(r, "neun_positive_profile_count") for r in have)
    other = sum(number(r, "nonneuronal_cell_count") for r in have)
    total = neurons + other
    return {"neun_cell_density_mm2": total / area, "neun_nonneuronal_cell_density_mm2": other / area,
            "neun_neuron_fraction_of_cells": neurons / total if total >= MIN_CELLS else np.nan}


def results_tables(folder: Path, tile_rows: dict[str, list[dict]], objects: dict[str, list[dict]], masks: dict[str, list[dict]],
                   scored: dict[str, list[dict]] | None = None, organised: dict | None = None) -> list[str]:
    """The stainID results tables (`results_<level>.csv` and their parts) from the legacy-shaped rows; `scored`: the
    type scores' object rows per stain (`scores.object_rows`), summarised as their own columns; `organised`:
    `_Rows.organised`, the columns of `stainid.v2.organisation`."""
    organised = organised or {"tallies": {s: [] for s in STAINS}, "cores": [], "neurons": []}
    organised.setdefault("bio", {s: [] for s in STAINS})
    lines = []
    for level, name in LEVELS.items():
        merged: dict[str, dict] = {}
        neun_tiles, neun_objects = _group(tile_rows["NeuN"], level), _group(objects["NeuN"], level)
        for key, group in neun_tiles.items():
            summary = summarize_stain_group(group, neun_objects.get(key, []))
            del summary["stain"]
            merged.setdefault(key, {k: group[0][k] for k in REGION_KEYS}).update({k if k.startswith("neun_") else f"neun_{k}": v for k, v in summary.items()})
            merged[key].update(cell_columns(group))
        for stain in ("6E10", "AT8"):
            tiles_, objects_ = _group(tile_rows[stain], level), _group(objects[stain], level)
            for key, group in tiles_.items():
                merged.setdefault(key, {k: group[0][k] for k in REGION_KEYS}).update(field_aggregation.summarize(stain, group, objects_.get(key, [])))
        area = {(row[level], stain): 0.0 for stain in STAINS for row in tile_rows[stain]}
        for stain in STAINS:
            for row in tile_rows[stain]:
                area[(row[level], stain)] += float(row["tissue_area_mm2"])
            by_key = _group(masks[stain], level)
            for key in {row[level] for row in tile_rows[stain]}:  # no outline in measured tissue is 0, not missing
                merged[key].update(mask_aggregation.summarize(stain, by_key.get(key, []), area[(key, stain)]))
        for stain, rows_ in (scored or {}).items():  # the area of the stain's density columns (v2_tissue_area_mm2)
            measured = defaultdict(float)
            for row in tile_rows[stain]:
                measured[row[level]] += float(row["v2_tissue_area_mm2"])
            by_key = _group(rows_, level)
            for key in measured:
                merged[key].update(scores.readouts(stain, by_key.get(key, []), measured[key]))
        for stain in STAINS:
            tallies = _group(organised["tallies"][stain], level)
            for key in {row[level] for row in tile_rows[stain]}:
                merged[key].update(organisation.edge_columns(stain, tallies.get(key, [])))
        tau_cores, tau_neurons, tau_tiles = _group(organised["cores"], level), _group(organised["neurons"], level), _group(tile_rows["AT8"], level)
        tau_tallies = _group(organised["tallies"]["AT8"], level)
        extra_counts = {key: organisation.counts(tau_cores.get(key, []), tau_neurons.get(key, [])) for key in tau_tiles}
        bio = {stain: _group(organised["bio"][stain], level) for stain in STAINS}
        for stain in STAINS:  # stage 2's biology measurements (`biology_measures`)
            for key in {row[level] for row in tile_rows[stain]}:
                merged[key].update(biology_measures.columns(stain, bio[stain].get(key, [])))
        for key in merged:
            extra_counts[key] = extra_counts.get(key, {}) | biology_measures.counts({s: bio[s].get(key, []) for s in STAINS})
        for key, group in tau_tiles.items():
            merged[key].update(organisation.summarize(tau_cores.get(key, []), tau_neurons.get(key, []),
                                                      sum(float(r["thread_ridge_length_mm"]) for r in group),
                                                      sum(float(r["v2_tissue_area_mm2"]) for r in group)))
            merged[key].update(organisation.boundary_columns(tau_tallies.get(key, []), sum(float(r["v2_tissue_area_mm2"]) for r in group)))
        rows = [merged[k] for k in sorted(merged)]
        if level != "core_id":
            for row in rows:
                row.pop("core_id", None)
        target = folder / f"results_{name}.csv"
        write_records(target, rows)
        tiled, objected, masked, scored_ = ({s: _group(rows_[s], level) for s in rows_} for rows_ in (tile_rows, objects, masks, scored or {}))
        write_records(folder / missing.COUNTS.format(name),
                      [{level: k, **missing.group_counts(*({s: by[s].get(k, []) for s in by} for by in (tiled, objected, masked, scored_))),
                        **extra_counts.get(k, {})} for k in sorted(merged)])
        lines.append(f"{len(rows)} {name} rows -> {target}")
    return lines


def grey_inputs(project, stain: str, core_id: str, tiles: pd.DataFrame, found: pd.DataFrame, threads: pd.DataFrame | None):
    """A core image's tiles, candidates and threads with its white matter left out (`v2.white_matter.read`): objects
    whose centre lies in it dropped, its counted cells out of each tile's area (and the NeuN density windows), the 6E10 /
    AT8 stained areas the grey-matter pass's. None when the core image has no white matter (nothing changes)."""
    from stainid import exclusions  # noqa: PLC0415

    white = white_matter.read(project, core_id, stain)
    if white is None or not white.any():
        return None
    layer = exclusions.automatic(project, core_id, stain)
    if layer is None:
        return None
    counted = layer.codes == exclusions.COUNTED
    h, w = min(white.shape[0], counted.shape[0]), min(white.shape[1], counted.shape[1])
    gone = np.zeros(counted.shape, bool)
    gone[:h, :w] = white[:h, :w] & counted[:h, :w]
    size = grey_matter.SCALE

    def in_white(x, y) -> np.ndarray:
        x, y = np.asarray(x, float), np.asarray(y, float)
        ok = np.isfinite(x) & np.isfinite(y)
        cx, cy = np.where(ok, x // size, -1).astype(int), np.where(ok, y // size, -1).astype(int)
        inside = (cx >= 0) & (cy >= 0) & (cx < white.shape[1]) & (cy < white.shape[0])
        out = np.zeros(len(cx), bool)
        out[inside] = white[cy[inside], cx[inside]]
        return out

    kept = found[~in_white(found.x_core, found.y_core)] if len(found) else found
    redone = grey_matter.current(project, stain, core_id) if stain in grey_matter.STAINS else None
    rows = []
    for tile in tiles.to_dict("records"):
        tile = dict(tile)
        x0, y0 = int(tile["x_px"]) // size, int(tile["y_px"]) // size
        tw, th = max(1, int(tile["width_px"]) // size), max(1, int(tile["height_px"]) // size)
        cells = int(gone[y0: y0 + th, x0: x0 + tw].sum())
        cell_mm2 = size * size * float(tile.get("pixel_width_um") or 0.2738) * float(tile.get("pixel_height_um") or 0.2738) / 1e6
        tile["analysis_area_mm2"] = max(0.0, float(tile["analysis_area_mm2"]) - cells * cell_mm2)
        tile["valid_area_mm2"] = tile["analysis_area_mm2"]
        if stain == "NeuN" and isinstance(tile.get("density_windows"), str) and tile["density_windows"].strip():
            ww, wh = (int(v) for v in str(tile["density_window_px"]).split())
            values = [int(v) for v in tile["density_windows"].split()]
            windows = []
            for wx, wy, count in zip(values[0::3], values[1::3], values[2::3]):
                lost = int(gone[wy // size: (wy + wh) // size, wx // size: (wx + ww) // size].sum()) * size * size
                if count - lost >= 0.5 * ww * wh:
                    windows.append(f"{wx} {wy} {count - lost}")
            tile["density_windows"] = " ".join(windows)
        rows.append(tile)
    frame = pd.DataFrame(rows, columns=tiles.columns)
    if redone is not None and len(redone):
        index = frame.set_index("tile_id")
        values = redone.set_index("tile_id")
        for column in grey_matter.COLUMNS[stain]:
            if column in index and column in values:
                index.loc[values.index, column] = values[column]
        frame = index.reset_index()[tiles.columns]
    if threads is not None and not threads.empty and "points" in threads:
        middles = [_middle(p) for p in threads.points]
        xs, ys = np.array([m[0] for m in middles]), np.array([m[1] for m in middles])
        threads = threads[~in_white(xs, ys)]
    return frame, kept, threads


def _middle(points) -> tuple[float, float]:
    """The middle of a thread piece's points ("x y x y …", core px); far away when it has none."""
    values = [float(v) for v in str(points).split()] if isinstance(points, str) else []
    if len(values) < 2:
        return -1e9, -1e9
    return float(np.mean(values[0::2])), float(np.mean(values[1::2]))


class _Rows:
    """Every row stage 5 collects, before the tables are written."""

    def __init__(self):
        self.tiles = {s: [] for s in STAINS}
        self.objects = {s: [] for s in STAINS}
        self.v1_objects = {s: [] for s in STAINS}
        self.masks = {s: [] for s in STAINS}
        self.scored = {s: [] for s in scores.SCORES}
        self.organised = {"tallies": {s: [] for s in STAINS}, "cores": [], "neurons": [], "bio": {s: [] for s in STAINS}}

    def add(self, stain: str, organised: dict) -> None:
        """What `_organised` measured on one core image."""
        self.organised["tallies"][stain].append(organised["tally"])
        if "bio" in organised:
            self.organised["bio"][stain].append(organised["bio"])
        if "core" in organised:
            self.organised["cores"].append(organised["core"])
            self.organised["neurons"] += organised["neurons"]


def _core_rows(project, stain: str, core_id: str, tiles: pd.DataFrame, found: pd.DataFrame, threads, models: dict, morphotype: float,
               addition: float, between: dict, into: _Rows) -> None:
    if stain == "NeuN":
        t, o = neun_rows(tiles, found, addition)
        v1 = o
    elif stain == "6E10":
        t, v1, o = amyloid_rows(tiles, found, morphotype, {})
    else:
        t, o = tau_rows(project, tiles, found, threads)
        v1 = []
    into.tiles[stain] += t
    into.objects[stain] += o
    into.v1_objects[stain] += v1
    into.masks[stain] += mask_rows(found, tiles, stain, morphotype if stain == "6E10" else None)
    if stain in into.scored:
        meta = {tile["tile_id"]: {k: tile[k] for k in REGION_KEYS if k in tile} for tile in tiles.to_dict("records")}
        into.scored[stain] += scores.object_rows(stain, found, models[stain].bundle, between[stain], meta)


def _organised(project, stain: str, core_id: str, tiles: pd.DataFrame, found: pd.DataFrame, threads, bodies: pd.DataFrame | None,
               white: np.ndarray | None, neun_cells: tuple | None = None) -> dict:
    """One core image's `stainid.v2.organisation` measures on these tiles, objects and threads; `white`: left out of the
    tissue the tangles are placed in."""
    first = tiles.iloc[0]
    meta = {k: str(first[k]) for k in LEVELS if k in tiles}
    out = {"tally": {**meta, **organisation.edge_tally(found)}}
    threshold = float(pd.to_numeric(tiles.threshold_dab_od, errors="coerce").median()) if "threshold_dab_od" in tiles else None
    out["bio"] = {**meta, **biology_measures.core_tally(project, stain, core_id, found, white, threads, neun_cells, threshold)}
    if stain == "AT8":
        out["tally"] |= organisation.boundary_tally(found)
        pixel_um = project.pixel_size_um
        kept = found.candidate_id[found.decision == "accepted"] if len(found) else []
        mine = bodies[bodies.candidate_id.isin(kept)] if bodies is not None and len(bodies) else pd.DataFrame(columns=organisation.OUTLINE_COLUMNS)
        neurons, attached_mm = organisation.ownership(mine, threads, pixel_um)
        out["core"] = {**meta, **organisation.spacing(found, organisation.valid_tissue(project, core_id, stain, white), pixel_um, core_id),
                       "attached_mm": attached_mm}
        out["neurons"] = [{**meta, **row} for row in neurons]
    return out


def _bodies(project, stain: str, core_id: str, found: pd.DataFrame) -> pd.DataFrame | None:
    """The accepted tau+ neurons' outlines, read only here (they are long and only thread ownership needs them)."""
    if stain != "AT8" or found.empty or not (found.decision == "accepted").any():
        return None
    frame = store.read_core(project, stain, core_id, organisation.OUTLINE_COLUMNS)
    frame = frame[frame.decision == "accepted"].copy()
    for column in organisation.OUTLINE_COLUMNS:
        if column not in frame:
            frame[column] = "" if column in store.TEXT_COLUMNS else np.nan
    for column in store.TEXT_COLUMNS:
        if column in frame:
            frame[column] = frame[column].fillna("")
    return frame


def _in_white(white: np.ndarray, x, y) -> np.ndarray:
    """Whether each point (core px) lies in the core image's white matter (`white_matter.read`, `grey_matter.SCALE`)."""
    size = grey_matter.SCALE
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    cx, cy = np.where(ok, x // size, -1).astype(int), np.where(ok, y // size, -1).astype(int)
    inside = (cx >= 0) & (cy >= 0) & (cx < white.shape[1]) & (cy < white.shape[0])
    out = np.zeros(len(cx), bool)
    out[inside] = white[cy[inside], cx[inside]]
    return out


def _neun_cells(project, core_id: str) -> tuple[pd.DataFrame | None, np.ndarray]:
    """The NeuN core image's cells (stage 2's per-nucleus table) and whether each lies inside an accepted neuron's own
    outline (the one the views draw, `candidates.outline_rings`)."""
    table = cells.read(project, "NeuN", core_id)
    if table is None or table.empty:
        return table, np.zeros(0, bool)
    frame = store.read_core(project, "NeuN", core_id, organisation.OUTLINE_COLUMNS)
    frame = frame[frame.decision == "accepted"] if "decision" in frame else frame.iloc[0:0]
    for column in organisation.OUTLINE_COLUMNS:
        if column not in frame:
            frame[column] = "" if column in store.TEXT_COLUMNS else np.nan
    for column in store.TEXT_COLUMNS:
        if column in frame:
            frame[column] = frame[column].fillna("")
    return table, cells.in_outlines(table, [store.outline_rings(r) for r in frame.to_dict("records")])


def _with_cells(tiles: pd.DataFrame, table: pd.DataFrame, neuronal: np.ndarray, white: np.ndarray | None = None) -> pd.DataFrame:
    """NeuN tile rows with their cell counts (`cells.tile_counts`); `white`: the cells in it left out (grey matter)."""
    if white is not None and table is not None and len(table):
        keep = ~_in_white(white, table.x_core, table.y_core)
        table, neuronal = table[keep], neuronal[keep]
    return cells.tile_counts(tiles, table, neuronal)


def _totals(stain: str, tile_rows: list[dict]) -> dict:
    """What the white-matter columns need of a core image's tile rows, summed."""
    def total(column: str) -> float:
        return float(sum(float(row[column]) for row in tile_rows))
    if stain == "6E10":
        return {"area": total("tissue_area_mm2"), "plaques": total("v2_plaque_count"), "plaque_um2": total("v2_plaque_area_sum_um2"),
                "amyloid_mm2": total("amyloid_extent_area_mm2")}
    return {"area": total("tissue_area_mm2"), "tau_neurons": total("tau_neuron_count"), "thread_mm": total("thread_ridge_length_mm"),
            "at8_mm2": total("at8_positive_area_mm2")}


def _white_row(project, stain: str, tiles: pd.DataFrame, whole_rows: list[dict], grey_tiles: pd.DataFrame, grey_found: pd.DataFrame, threads,
               morphotype: float) -> dict:
    """One core image's white matter: its whole-core tile rows less its grey-matter ones, each never below 0."""
    if stain == "6E10":
        grey_rows = amyloid_rows(grey_tiles, grey_found, morphotype, {})[0]
    else:
        grey_rows = tau_rows(project, grey_tiles, grey_found, threads)[0]
    whole_total, grey_total = _totals(stain, whole_rows), _totals(stain, grey_rows)
    first = tiles.iloc[0]
    return {**{k: str(first[k]) for k in LEVELS if k in tiles}, "stain": stain,
            **{k: max(0.0, whole_total[k] - grey_total[k]) for k in whole_total}}


WHITE_COLUMNS = {"6E10": {"plaque_white_area_mm2": None, "plaque_density_white_mm2": ("plaques", 1.0), "plaque_area_fraction_white": ("plaque_um2", 1e-6),
                          "amyloid_positive_area_fraction_white": ("amyloid_mm2", 1.0)},
                 "AT8": {"tau_white_area_mm2": None, "tau_neuron_density_white_mm2": ("tau_neurons", 1.0),
                         "thread_density_white_mm_per_mm2": ("thread_mm", 1.0), "at8_positive_area_fraction_white": ("at8_mm2", 1.0)}}


def _measured_rows(whole: _Rows) -> dict[str, dict[str, set]]:
    """Per stain, the core images and donor-regions with tiles in the whole-core tables."""
    return {stain: {key: {str(row[key]) for row in whole.tiles[stain]} for key in LEVELS} for stain in WHITE_COLUMNS}


def _attach_white(folder: Path, white_rows: list[dict], measured: dict[str, set], floor: float) -> list[str]:
    """Pathology inside the white matter, next to the grey-matter columns: plaques, tau+ neurons, thread and stained area
    of each row's core images in their white matter over its area. Missing without at least `floor` mm² of white matter;
    0 when nothing was found in it. `<stain>_white_area_mm2`: that area, 0 for a core image without white matter."""
    lines = []
    for level, key in (("core", "core_id"), ("donor_region", "sample_region_id")):
        path = folder / f"results_{level}.csv"
        if not path.exists():
            continue
        table = pd.read_csv(path, low_memory=False, dtype={"donor_id": str})
        ids = table[key].astype(str)
        for stain, columns in WHITE_COLUMNS.items():
            rows = pd.DataFrame([r for r in white_rows if r["stain"] == stain])
            summed = rows.groupby(key).sum(numeric_only=True) if len(rows) else pd.DataFrame(columns=["area"])
            area = ids.map(summed["area"]).astype(float)
            seen = ids.isin(measured[stain][key])
            for column, part in columns.items():
                if part is None:
                    table[column] = area.fillna(0.0).where(seen)
                else:
                    table[column] = (ids.map(summed[part[0]]).astype(float) * part[1] / area).where(area >= floor) if len(rows) else np.nan
        table.to_csv(path, index=False)
        lines.append(f"results_{level}: pathology inside the white matter ({sum(len(c) for c in WHITE_COLUMNS.values())} columns)")
    return lines


def write_measures(project, folder: Path | None = None) -> list[str]:
    """Stage 5: every table, from the core images whose calls exist (into `folder`, default `<outputs.v2>/measures`);
    grey matter only with `v2.white_matter.analysis: exclude` (the whole-core tables under `whole_core/`)."""
    folder = Path(folder) if folder is not None else store.root(project) / "measures"
    folder.mkdir(parents=True, exist_ok=True)
    exclude = white_matter.excluded(project)
    models = {stain: load_model(project, stain) for stain in STAINS}
    morphotype = float(models["6E10"].bundle["morphotype_threshold"])
    addition = float(models["NeuN"].bundle.get("addition_threshold", 0.6))
    whole, grey = _Rows(), (_Rows() if exclude else None)
    between = {s: scores.cuts(project, s) for s in scores.SCORES}
    images = core_images(project)
    lines = [f"stage 5 (measures): {', '.join(f'{len(v)} {s}' for s, v in images.items())} core images with calls"]
    left_out = 0
    floor = float(white_matter.settings(project)["min_grey_mm2"]) if exclude else 0.0
    too_small: set[tuple[str, str]] = set()
    white_rows: list[dict] = []
    for stain, cores in images.items():
        for core_id in cores:
            tiles = _tiles(project, stain, core_id)
            found = store.read_core(project, stain, core_id, CANDIDATE_COLUMNS)
            # a core image without a single candidate (an AT8 control core) is measured all the same: its tiles count
            # with no objects (zero burden), else the core and its donor-region drop out of every table
            if tiles.empty or ("decision" not in found and not found.empty):
                continue
            for column in CANDIDATE_COLUMNS:
                if column not in found:
                    found[column] = np.nan
            found["area_um2"] = store.shown_area_um2(found)  # each object as the views draw it (the stain's default outline)
            if stain == "AT8":
                found = organisation.mark_boundary(project, core_id, found)
            threads = store.threads(project, core_id) if stain == "AT8" else None
            if stain == "NeuN":  # the NeuN section's cells, neuronal or not (`cells`)
                cell_table, neuronal = _neun_cells(project, core_id)
                tiles = _with_cells(tiles, cell_table, neuronal)
            before = len(whole.tiles[stain])
            _core_rows(project, stain, core_id, tiles, found, threads, models, morphotype, addition, between, whole)
            bodies = _bodies(project, stain, core_id, found)
            neun_cells = (cell_table, neuronal) if stain == "NeuN" else None
            as_whole = _organised(project, stain, core_id, tiles, found, threads, bodies, None, neun_cells)
            whole.add(stain, as_whole)
            if grey is not None:
                changed = grey_inputs(project, stain, core_id, tiles, found, threads)
                if changed is not None and stain == "NeuN":
                    changed = (_with_cells(changed[0], cell_table, neuronal, white_matter.read(project, core_id, stain)), *changed[1:])
                if changed is None:
                    _core_rows(project, stain, core_id, tiles, found, threads, models, morphotype, addition, between, grey)
                    grey.add(stain, as_whole)
                else:
                    left_out += 1
                    if stain in WHITE_COLUMNS:
                        white_rows.append(_white_row(project, stain, tiles, whole.tiles[stain][before:], changed[0], changed[1], changed[2], morphotype))
                    if float(pd.to_numeric(changed[0]["analysis_area_mm2"], errors="coerce").sum()) < floor:
                        too_small.add((str(core_id), stain))  # a sliver of grey matter: not measured (NA), see the docstring
                    else:
                        _core_rows(project, stain, core_id, *changed, models, morphotype, addition, between, grey)
                        grey.add(stain, _organised(project, stain, core_id, *changed, bodies, white_matter.read(project, core_id, stain), neun_cells))
    if grey is None:
        lines += _write_tables(folder, whole, between)
    else:
        lines.append(f"white matter left out (v2.white_matter.analysis: exclude) of {left_out} core images; whole-core tables -> "
                     f"{folder / 'whole_core'}")
        lines += _write_tables(folder / "whole_core", whole, between)
        lines += _write_tables(folder, grey, between)
        lines += _attach_whole(folder, folder / "whole_core")
        lines += _attach_too_small(project, folder, too_small, floor)
        lines += _attach_white(folder, white_rows, _measured_rows(whole), floor)
    if white_matter.summary_path(project).exists():
        lines += _attach_flags(project, folder)
    lines += interactions.attach(project, folder, exclude, too_small)  # the spatial interactions (`stainid.v2.interactions`)
    lines += missing.write(folder, folder / "whole_core" if grey is not None else None)
    if grey is not None:
        lines += missing.write(folder / "whole_core")
    lines += reliability.write(folder)
    return lines


RESULT_TABLES = ("results_core", "results_donor_region", "cohort_morphology_core_features", "cohort_morphology_donor_region_features",
                 "cohort_morphology_donor_region_wide")
TABLE_KEYS = ("core_id", "sample_region_id", "stain", "sampling_extent", "analysis_level")
NOT_MEASURES = ("tma", "donor_id", "technical_replicate", "white_matter_share", "neun_pale", "neun_pale_cores")
WHOLE = "_whole_core"


def _attach_whole(folder: Path, whole: Path) -> list[str]:
    """Each whole-core number next to its grey-matter one, as `<column>_whole_core`, in the results and R tables."""
    lines = []
    for name in RESULT_TABLES:
        main_path, whole_path = folder / f"{name}.csv", whole / f"{name}.csv"
        if not main_path.exists() or not whole_path.exists():
            continue
        main, other = _read_or_empty(main_path), pd.read_csv(whole_path, low_memory=False)
        if main.empty and len(other):  # no core image had grey matter enough to measure: the whole-core rows, every measure NA
            main = other.copy()
            for column in main.columns:
                if column not in TABLE_KEYS and column not in NOT_MEASURES and pd.api.types.is_numeric_dtype(main[column]):
                    main[column] = np.nan
        keys = [k for k in TABLE_KEYS if k in main and k in other and not (k == "core_id" and "results_donor" in name)]
        numbers = [c for c in other.columns if c not in keys and c not in NOT_MEASURES and pd.api.types.is_numeric_dtype(other[c])]
        joined = main.merge(other[keys + numbers].rename(columns={c: c + WHOLE for c in numbers}), on=keys, how="left")
        joined.to_csv(main_path, index=False)
        lines.append(f"{name}: {len(numbers)} whole-core columns ({WHOLE})")
    return lines


def _read_or_empty(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, low_memory=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _attach_too_small(project, folder: Path, too_small: set[tuple[str, str]], floor: float) -> list[str]:
    """The core images left with too little grey matter (`too_small`: (core_id, stain)): their rows (and those of
    donor-regions left with none) put back from the whole-core tables with every measure missing, and the flags
    `grey_too_small` (core: in any stain; donor-region: all its cores) and `grey_too_small_cores` (donor-region)."""
    cores = {c for c, _ in too_small}
    region_of = _regions(project, folder / "whole_core")
    small_in_region: dict[str, int] = defaultdict(int)
    for core_id in cores:
        if core_id in region_of:
            small_in_region[region_of[core_id]] += 1
    all_in_region: dict[str, int] = defaultdict(int)
    for core_id, region in region_of.items():
        all_in_region[region] += 1
    lines = [f"grey matter under {floor:g} mm² (v2.white_matter.min_grey_mm2): {len(too_small)} core images of {len(cores)} cores "
             f"not measured (NA)"]
    for name in RESULT_TABLES:
        main_path, whole_path = folder / f"{name}.csv", folder / "whole_core" / f"{name}.csv"
        if not main_path.exists() or not whole_path.exists():
            continue
        main, whole = pd.read_csv(main_path, low_memory=False), pd.read_csv(whole_path, low_memory=False)
        keys = [k for k in TABLE_KEYS if k in main and k in whole and not (k == "core_id" and "results_donor" in name)]
        have = set(map(tuple, main[keys].astype(str).to_numpy()))
        missing = whole[[tuple(r) not in have for r in whole[keys].astype(str).to_numpy()]].copy()
        if len(missing):
            numbers = [c for c in whole.columns if c not in keys and c not in NOT_MEASURES and pd.api.types.is_numeric_dtype(whole[c])]
            for column in numbers:
                if column + WHOLE in main:
                    missing[column + WHOLE] = missing[column]
                missing[column] = np.nan
            main = pd.concat([main, missing[[c for c in missing.columns if c in main.columns]]], ignore_index=True)
        per_core = "core_id" in main and "results_donor" not in name and "donor_region" not in name
        if per_core:
            if "stain" in main:
                flagged = [(str(c), str(st)) in too_small for c, st in zip(main.core_id, main.stain)]
            else:
                flagged = [str(c) in cores for c in main.core_id]
            main["grey_too_small"] = np.array(flagged, int)
        elif "sample_region_id" in main:
            ids = main.sample_region_id.astype(str)
            main["grey_too_small_cores"] = ids.map(lambda r: small_in_region.get(r, 0)).to_numpy()
            main["grey_too_small"] = ids.map(lambda r: int(small_in_region.get(r, 0) > 0 and small_in_region[r] >= all_in_region.get(r, 0))).to_numpy()
        main.to_csv(main_path, index=False)
        lines.append(f"{name}: {len(missing)} rows with too little grey matter put back as NA")
    return lines


def _attach_flags(project, folder: Path) -> list[str]:
    """The white-matter status and share and the NeuN-pale flag per core and donor-region, in the results and R tables."""
    table = white_matter.summary(project)
    neun = table[table.stain == white_matter.SOURCE].set_index("core_id")
    pale = white_matter.neun_pale(project)
    core = pd.DataFrame({"white_matter_status": neun.status, "white_matter_share": neun.white_share,
                         "neun_pale": [int(c in pale) for c in neun.index]}, index=neun.index)
    core.index = core.index.astype(str)
    lines = []
    for name in RESULT_TABLES:
        path = folder / f"{name}.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path, low_memory=False).drop(columns=["white_matter_status", "white_matter_share", "neun_pale", "neun_pale_cores"],
                                                         errors="ignore")
        if "core_id" in frame and "results_donor" not in name and "donor_region" not in name:
            flags = core.reindex(frame.core_id.astype(str))
            frame["white_matter_status"] = flags.white_matter_status.to_numpy()
            frame["white_matter_share"] = flags.white_matter_share.to_numpy()
            frame["neun_pale"] = flags.neun_pale.fillna(0).astype(int).to_numpy()
        elif "sample_region_id" in frame:
            region_of = _regions(project, folder)
            per = neun.assign(sample_region_id=[region_of.get(str(c)) for c in neun.index], pale=[int(str(c) in pale) for c in neun.index])
            per = per.dropna(subset=["sample_region_id"])
            sums = per.groupby("sample_region_id").agg(white=("white_mm2", "sum"), tissue=("tissue_mm2", "sum"), pale=("pale", "sum"),
                                                       white_cores=("status", lambda s: int((s == "white").sum())))
            share = (sums.white / sums.tissue.where(sums.tissue > 0)).round(4)
            ids = frame.sample_region_id.astype(str)
            frame["white_matter_status"] = ids.map(lambda r: "white" if sums.white_cores.get(r, 0) else ("grey" if r in sums.index else "")).to_numpy()
            frame["white_matter_share"] = ids.map(share).to_numpy()
            frame["neun_pale"] = ids.map(lambda r: int(sums.pale.get(r, 0) > 0)).to_numpy()
            frame["neun_pale_cores"] = ids.map(lambda r: int(sums.pale.get(r, 0))).to_numpy()
        else:
            continue
        frame.to_csv(path, index=False)
        lines.append(f"{name}: white-matter and NeuN-pale flags")
    return lines


def _regions(project, folder: Path) -> dict[str, str]:
    """Each core's donor-region: from the core results table just written, else the core image table."""
    made = folder / "results_core.csv"
    table = made if made.exists() else project.input("core_images")
    return {str(r["core_id"]): str(r["sample_region_id"]) for r in read_csv(table)} if table.exists() else {}


def _write_neurons(folder: Path, neurons: list[dict]) -> None:
    """Thread ownership per accepted tau+ neuron (`stainid.v2.organisation.ownership`)."""
    columns = ["core_id", "sample_region_id", "candidate_id", "subtype", "processes", "attached_thread_um"]
    pd.DataFrame(neurons, columns=columns).to_csv(folder / "tau_neuron_threads.csv", index=False)


def _write_tables(folder: Path, collected: _Rows, between: dict) -> list[str]:
    """Every table of stage 5 from the collected rows, into `folder`."""
    folder.mkdir(parents=True, exist_ok=True)
    tile_rows, objects, v1_objects, masks, scored = (collected.tiles, collected.objects, collected.v1_objects, collected.masks,
                                                     collected.scored)
    lines = []
    for stain in STAINS:
        _write(folder / f"tiles_{stain}.csv", tile_rows[stain])
        _write(folder / f"objects_{stain}.csv", objects[stain])
    for stain, rows in scored.items():
        _write(folder / f"type_scores_{stain}.csv", rows)
    # read back as text, as the aggregators expect (empty cell = missing)
    text_tiles = {s: read_csv(folder / f"tiles_{s}.csv") if tile_rows[s] else [] for s in STAINS}
    text_objects = {s: read_csv(folder / f"objects_{s}.csv") if objects[s] else [] for s in STAINS}
    _write(folder / "masks_objects.csv", [r | {"stain": s} for s in STAINS for r in masks[s]])
    text_masks = {s: [r for r in read_csv(folder / "masks_objects.csv") if r["stain"] == s] if any(masks.values()) else [] for s in STAINS}
    lines += results_tables(folder, text_tiles, text_objects, text_masks, scored, collected.organised)
    _write_neurons(folder, collected.organised["neurons"])
    for stain, rows in scored.items():
        named = scores.SCORES[stain]
        counts = {b: sum(r[f"{named.column}_bin"] == b for r in rows) for b in named.bins}
        lines.append(f"{named.column}: {len(rows)} accepted {stain} objects scored, {sum(counts.values())} binned (cuts {between[stain][0]:.2f}, "
                     f"{between[stain][1]:.2f}): " + ", ".join(f"{b} {n}" for b, n in counts.items()))
    v1_tiles = [row for s in STAINS for row in text_tiles[s]]
    _write(folder / "cohort_morphology_v2_objects.csv", [r for s in STAINS for r in v1_objects[s]])
    _write(folder / "cohort_morphology_v2_tile_features.csv", v1_tiles)
    core, region, wide = aggregation.write_aggregated_tables(folder / "cohort_morphology_v2_tile_features.csv", folder / "cohort_morphology_v2_objects.csv",
                                                             folder / "cohort_morphology_core_features.csv",
                                                             folder / "cohort_morphology_donor_region_features.csv",
                                                             folder / "cohort_morphology_donor_region_wide.csv")
    lines += [f"R tables -> {core.name}, {region.name}, {wide.name}"]
    return lines
