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
from stainid.v2 import manifest
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
                     "sam_core_dab_q"]


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
            "thread_ridge_width_sum_um": float(free.width_sum_um.sum()) if not free.empty else 0.0,
            "thread_length_with_somata_mm": float(tile["thread_skeleton_px"]) * ps / 1000,
            "thread_branchpoint_count": tile["thread_branchpoint_count"], "thread_endpoint_count": tile["thread_endpoint_count"],
            "nuclei_available": str(bool(tile.get("nuclei_available", True))).lower(),
        })
        meta = {k: tile[k] for k in REGION_KEYS if k in tile} | {"tile_id": tile["tile_id"], "stain": "AT8"}
        for row in neurons.to_dict("records"):
            objects.append({**meta, "soma_area_um2": row["area_um2"]})
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


def results_tables(folder: Path, tile_rows: dict[str, list[dict]], objects: dict[str, list[dict]], masks: dict[str, list[dict]]) -> list[str]:
    """The stainID results tables (`results_<level>.csv` and their parts) from the legacy-shaped rows."""
    lines = []
    for level, name in LEVELS.items():
        merged: dict[str, dict] = {}
        neun_tiles, neun_objects = _group(tile_rows["NeuN"], level), _group(objects["NeuN"], level)
        for key, group in neun_tiles.items():
            summary = summarize_stain_group(group, neun_objects.get(key, []))
            del summary["stain"]
            merged.setdefault(key, {k: group[0][k] for k in REGION_KEYS}).update({k if k.startswith("neun_") else f"neun_{k}": v for k, v in summary.items()})
        for stain in ("6E10", "AT8"):
            tiles_, objects_ = _group(tile_rows[stain], level), _group(objects[stain], level)
            for key, group in tiles_.items():
                merged.setdefault(key, {k: group[0][k] for k in REGION_KEYS}).update(field_aggregation.summarize(stain, group, objects_.get(key, [])))
        area = {(row[level], stain): 0.0 for stain in STAINS for row in tile_rows[stain]}
        for stain in STAINS:
            for row in tile_rows[stain]:
                area[(row[level], stain)] += float(row["tissue_area_mm2"])
            by_key = _group(masks[stain], level)
            for key in {row[level] for row in tile_rows[stain]}:
                if by_key.get(key):
                    merged[key].update(mask_aggregation.summarize(stain, by_key[key], area[(key, stain)]))
        rows = [merged[k] for k in sorted(merged)]
        if level != "core_id":
            for row in rows:
                row.pop("core_id", None)
        target = folder / f"results_{name}.csv"
        write_records(target, rows)
        lines.append(f"{len(rows)} {name} rows -> {target}")
    return lines


def write_measures(project) -> list[str]:
    """Stage 5: every table, from the core images whose calls exist."""
    folder = store.root(project) / "measures"
    folder.mkdir(parents=True, exist_ok=True)
    models = {stain: load_model(project, stain) for stain in STAINS}
    morphotype = float(models["6E10"].bundle["morphotype_threshold"])
    addition = float(models["NeuN"].bundle.get("addition_threshold", 0.6))
    tile_rows = {s: [] for s in STAINS}
    objects = {s: [] for s in STAINS}
    v1_objects = {s: [] for s in STAINS}
    masks = {s: [] for s in STAINS}
    images = core_images(project)
    lines = [f"stage 5 (measures): {', '.join(f'{len(v)} {s}' for s, v in images.items())} core images with calls"]
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
            if stain == "NeuN":
                t, o = neun_rows(tiles, found, addition)
                v1 = o
            elif stain == "6E10":
                t, v1, o = amyloid_rows(tiles, found, morphotype, {})
            else:
                t, o = tau_rows(project, tiles, found, store.threads(project, core_id))
                v1 = []
            tile_rows[stain] += t
            objects[stain] += o
            v1_objects[stain] += v1
            masks[stain] += mask_rows(found, tiles, stain, morphotype if stain == "6E10" else None)
    for stain in STAINS:
        _write(folder / f"tiles_{stain}.csv", tile_rows[stain])
        _write(folder / f"objects_{stain}.csv", objects[stain])
    # read back as text, as the aggregators expect (empty cell = missing)
    text_tiles = {s: read_csv(folder / f"tiles_{s}.csv") if tile_rows[s] else [] for s in STAINS}
    text_objects = {s: read_csv(folder / f"objects_{s}.csv") if objects[s] else [] for s in STAINS}
    _write(folder / "masks_objects.csv", [r | {"stain": s} for s in STAINS for r in masks[s]])
    text_masks = {s: [r for r in read_csv(folder / "masks_objects.csv") if r["stain"] == s] if any(masks.values()) else [] for s in STAINS}
    lines += results_tables(folder, text_tiles, text_objects, text_masks)
    v1_tiles = [row for s in STAINS for row in text_tiles[s]]
    _write(folder / "cohort_morphology_v2_objects.csv", [r for s in STAINS for r in v1_objects[s]])
    _write(folder / "cohort_morphology_v2_tile_features.csv", v1_tiles)
    core, region, wide = aggregation.write_aggregated_tables(folder / "cohort_morphology_v2_tile_features.csv", folder / "cohort_morphology_v2_objects.csv",
                                                             folder / "cohort_morphology_core_features.csv",
                                                             folder / "cohort_morphology_donor_region_features.csv",
                                                             folder / "cohort_morphology_donor_region_wide.csv")
    lines += [f"R tables -> {core.name}, {region.name}, {wide.name}"]
    return lines
