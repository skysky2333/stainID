"""Stage 5 readouts of stage 2's biology measurements (`stainid.v2.biology`), per core and donor-region; no image read.

Per core image a tally (`core_tally`), summed per row of the results table (`columns`). Each needs its minimum count,
else it is missing (`stainid.v2.missing`); a density is 0 when nothing was found in measured tissue. Every column is
missing for core images whose stage 2 ran before the measurement existed (`not_measured`).

- Tangle nucleus (AT8, accepted tau+ neurons): `tau_neuron_nucleus_visible_fraction` (and `_mature`, `_pretangle`: of
  that subtype), `tau_neuron_nucleus_offset_median` (of those with a nucleus; in soma radii). `MIN_NEURONS` each.
- Plaque profile (6E10, accepted plaques of at least `PLAQUE_UM`): medians of `profile_core_contrast`,
  `profile_core_to_halo`, `profile_halo_width_um`, and the shares
  `plaque_bare_core_fraction` (a core of at least `CORE_LEVEL` slide thresholds with a halo under `BARE_HALO_UM`: a
  "burnt-out" core) and `plaque_haloed_core_fraction` (such a core with a halo of at least `HALO_UM`). `MIN_PLAQUES` each.
- Vessels (every stain, the stain's own section): `<prefix>_vessel_density_mm2` and `_vessel_area_fraction` over the
  measured tissue; on 6E10 `caa_vessel_fraction` (vessels whose wall band is positive in at least `CAA_COVERAGE` of its
  sectors; `MIN_VESSELS`) and `caa_vessel_density_mm2`, `plaque_vessel_touch_fraction` and `plaque_vessel_near_fraction`
  (plaques whose edge lies within `TOUCH_UM` / `NEAR_UM` of a vessel lumen: centre distance to the lumen less the
  plaque's equivalent radius, on the masks' grid); on AT8 the density of tau+ neurons and of free thread within
  `AROUND_UM` of a vessel lumen and elsewhere in the valid tissue (`tau_neuron_vessel_near_density_mm2`, `_far_`,
  `thread_vessel_near_density_mm_per_mm2`, `_far_`) and their ratios `tau_neuron_vessel_enrichment`,
  `thread_vessel_enrichment` (need `MIN_NEURONS` neurons / 0.5 mm of thread and 0.01 mm² near vessels).
- NeuN stain against loss (NeuN, the per-nucleus table `cells`): `neun_neuron_nucleus_area_median_um2` (the nucleus of
  accepted neurons, `MIN_NEUN` neurons), and of the neuron-sized nuclei (at least `NEURON_NUCLEUS_UM2`) the density
  `neun_large_nucleus_density_mm2`, the shares inside an accepted neuron (`neun_large_nucleus_in_neuron_fraction`), in
  none and stained weakly (DAB 0.5-1 slide threshold, `neun_large_nucleus_weak_fraction`) or not at all (under 0.5,
  `neun_large_nucleus_unstained_fraction`), `MIN_LARGE` nuclei. Antigen loss: neuron-sized nuclei are there but
  unstained; neuron loss: they are not there.
"""
from __future__ import annotations

import cv2
import numpy as np
import pandas as pd

from stainid.v2 import biology

SCALE = 8
MIN_NEURONS, MIN_PLAQUES, MIN_VESSELS, MIN_NEUN, MIN_LARGE = 5, 10, 5, 20, 20
PLAQUE_UM = 15.0
CORE_LEVEL, BARE_HALO_UM, HALO_UM = 2.0, 2.0, 4.0
CAA_COVERAGE = 0.5
TOUCH_UM, NEAR_UM, AROUND_UM = 2.0, 10.0, 20.0
NEURON_NUCLEUS_UM2 = 50.0
MATURE, PRETANGLE = "mature tangle", "pretangle"
PREFIX = {"NeuN": "neun", "6E10": "amyloid", "AT8": "at8"}
CANDIDATE_COLUMNS = ["soma_nucleus_visible", "soma_nucleus_offset", "soma_nucleus_area_um2", "profile_core_dab", "profile_core_contrast",
                     "profile_core_to_halo", "profile_halo_width_um"]
PROFILE_MEDIANS = ("profile_core_contrast", "profile_core_to_halo", "profile_halo_width_um")


def _numbers(frame: pd.DataFrame, column: str) -> np.ndarray:
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(float) if column in frame else np.full(len(frame), np.nan)


def _distance_to_vessels(vessels: pd.DataFrame | None, shape: tuple[int, int], pixel_um: float) -> np.ndarray | None:
    """µm from each mask cell to the nearest vessel lumen (0 inside one); None without a vessel table or vessel."""
    if vessels is None or not len(vessels) or not vessels.vessel.astype(bool).any():
        return None
    lumen = (biology.vessel_mask(vessels[vessels.vessel.astype(bool)], shape, SCALE, pixel_um) & biology.VESSEL_BITS["vessel"]) > 0
    return cv2.distanceTransform((~lumen).astype(np.uint8), cv2.DIST_L2, 5) * SCALE * pixel_um


def _at(grid: np.ndarray, x, y) -> np.ndarray:
    cx = np.clip((np.asarray(x, float) // SCALE).astype(int), 0, grid.shape[1] - 1)
    cy = np.clip((np.asarray(y, float) // SCALE).astype(int), 0, grid.shape[0] - 1)
    return grid[cy, cx]


def core_tally(project, stain: str, core_id: str, found: pd.DataFrame, white: np.ndarray | None = None, threads: pd.DataFrame | None = None,
               neun_cells: tuple | None = None, thresholds: float | None = None) -> dict:
    """What one core image adds to its rows' readouts (`columns`); `white`: the white matter, left out of the tissue
    and of the vessels; `neun_cells`: (cells table, inside an accepted neuron) of a NeuN image."""
    from stainid.v2 import organisation  # noqa: PLC0415

    pixel_um = project.pixel_size_um
    accepted = found[found.decision == "accepted"] if len(found) and "decision" in found else found.iloc[0:0]
    measured = any(c in found for c in biology.NUCLEUS_COLUMNS + biology.PROFILE_COLUMNS) and (
        len(accepted) == 0 or np.isfinite(_numbers(accepted, "soma_nucleus_visible" if stain != "6E10" else "profile_core_dab")).any()
        or stain == "NeuN")
    vessels = biology.read_vessels(project, stain, core_id)
    out: dict = {"stain": stain, "measured": bool(measured), "vessels_known": vessels is not None}
    valid = organisation.valid_tissue(project, core_id, stain, white)
    cell_mm2 = (SCALE * pixel_um) ** 2 / 1e6
    out["area_mm2"] = float(valid.sum() * cell_mm2) if valid is not None else 0.0
    distance = _distance_to_vessels(vessels, valid.shape, pixel_um) if vessels is not None and valid is not None else None
    if vessels is not None:
        kept = vessels[vessels.vessel.astype(bool)] if len(vessels) else vessels
        if len(kept) and valid is not None:
            kept = kept[_at(valid, kept.x_core, kept.y_core)]
        out["vessels"] = len(kept)
        out["vessel_area_mm2"] = float(pd.to_numeric(kept.area_um2, errors="coerce").sum() / 1e6) if len(kept) else 0.0
        coverage = _numbers(kept, "wall_coverage")
        out["caa_vessels"] = int((coverage >= CAA_COVERAGE).sum())
    if stain == "AT8":
        out["nucleus_visible"] = _numbers(accepted, "soma_nucleus_visible")
        out["nucleus_offset"] = _numbers(accepted, "soma_nucleus_offset")
        out["subtype"] = accepted.subtype.astype(str).to_numpy() if "subtype" in accepted else np.full(len(accepted), "")
        if distance is not None and valid is not None:
            near_cells = valid & (distance <= AROUND_UM)
            out["near_area_mm2"], out["far_area_mm2"] = float(near_cells.sum() * cell_mm2), float((valid & ~near_cells).sum() * cell_mm2)
            xy = accepted[["x_core", "y_core"]].to_numpy(float)
            inside = _at(valid, xy[:, 0], xy[:, 1]) if len(xy) else np.zeros(0, bool)
            near = _at(distance, xy[:, 0], xy[:, 1]) <= AROUND_UM if len(xy) else np.zeros(0, bool)
            out["near_neurons"], out["far_neurons"] = int((inside & near).sum()), int((inside & ~near).sum())
            out["near_thread_mm"] = out["far_thread_mm"] = 0.0
            if threads is not None and len(threads):
                free = threads[~threads.candidate_id.isin(set(accepted.candidate_id))]
                middles = np.array([_middle(p) for p in free.points]) if len(free) else np.zeros((0, 2))
                if len(middles):
                    length = free.n_px.to_numpy(float) * pixel_um / 1000
                    ok = _at(valid, middles[:, 0], middles[:, 1])
                    close = _at(distance, middles[:, 0], middles[:, 1]) <= AROUND_UM
                    out["near_thread_mm"], out["far_thread_mm"] = float(length[ok & close].sum()), float(length[ok & ~close].sum())
    if stain == "6E10":
        big = accepted[_numbers(accepted, "equivalent_diameter_um") >= PLAQUE_UM]
        out["profiles"] = {c: _numbers(big, c) for c in PROFILE_MEDIANS + ("profile_core_dab",)}
        if distance is not None and len(accepted):
            edge = _at(distance, accepted.x_core, accepted.y_core) - _numbers(accepted, "equivalent_diameter_um") / 2
            out["plaque_vessel_distance"] = edge
        elif vessels is not None:
            out["plaque_vessel_distance"] = np.full(len(accepted), np.inf)
    if stain == "NeuN":
        out["neuron_nucleus_area"] = _numbers(accepted, "soma_nucleus_area_um2")
        if neun_cells is not None and neun_cells[0] is not None and len(neun_cells[0]) and thresholds:
            table, neuronal = neun_cells
            keep = np.ones(len(table), bool)
            if white is not None:
                keep = ~_at(white, table.x_core, table.y_core)
            area = _numbers(table, "area_um2")
            dab = _numbers(table, "dab_mean") / thresholds
            large = keep & (area >= NEURON_NUCLEUS_UM2)
            inside = np.asarray(neuronal, bool)
            out["large"] = int(large.sum())
            out["large_in_neuron"] = int((large & inside).sum())
            out["large_weak"] = int((large & ~inside & (dab >= 0.5) & (dab < 1.0)).sum())
            out["large_unstained"] = int((large & ~inside & (dab < 0.5)).sum())
    return out


def _middle(points) -> tuple[float, float]:
    values = [float(v) for v in str(points).split()] if isinstance(points, str) else []
    return (float(np.mean(values[0::2])), float(np.mean(values[1::2]))) if len(values) >= 2 else (-1e9, -1e9)


def _cat(tallies: list[dict], key: str) -> np.ndarray:
    parts = [np.asarray(t[key]) for t in tallies if key in t]
    return np.concatenate(parts) if parts else np.zeros(0)


def _share(hits: np.ndarray, known: np.ndarray, minimum: int) -> float:
    return float(hits[known].mean()) if known.sum() >= minimum else float("nan")


def columns(stain: str, tallies: list[dict]) -> dict:
    """The readouts of one row of the results table from its core images' tallies (see the module's docstring)."""
    names = names_of(stain)
    measured = [t for t in tallies if t["measured"]]
    seen = [t for t in tallies if t["vessels_known"]]
    out = dict.fromkeys(names, float("nan"))
    prefix = PREFIX[stain]
    area = sum(t["area_mm2"] for t in seen)
    if seen and area > 0:
        out[f"{prefix}_vessel_density_mm2"] = sum(t["vessels"] for t in seen) / area
        out[f"{prefix}_vessel_area_fraction"] = sum(t["vessel_area_mm2"] for t in seen) / area
    if stain == "AT8" and measured:
        visible, offset, subtype = _cat(measured, "nucleus_visible"), _cat(measured, "nucleus_offset"), _cat(measured, "subtype")
        known = np.isfinite(visible)
        out["tau_neuron_nucleus_visible_fraction"] = _share(visible > 0, known, MIN_NEURONS)
        out["tau_neuron_nucleus_visible_fraction_mature"] = _share(visible > 0, known & (subtype == MATURE), MIN_NEURONS)
        out["tau_neuron_nucleus_visible_fraction_pretangle"] = _share(visible > 0, known & (subtype == PRETANGLE), MIN_NEURONS)
        have = offset[np.isfinite(offset)]
        out["tau_neuron_nucleus_offset_median"] = float(np.median(have)) if len(have) >= MIN_NEURONS else float("nan")
    if stain == "AT8" and seen:
        with_vessels = [t for t in seen if "near_area_mm2" in t]
        near_area = sum(t["near_area_mm2"] for t in with_vessels)
        far_area = sum(t["far_area_mm2"] for t in with_vessels)
        near_n, far_n = sum(t["near_neurons"] for t in with_vessels), sum(t["far_neurons"] for t in with_vessels)
        near_mm, far_mm = sum(t["near_thread_mm"] for t in with_vessels), sum(t["far_thread_mm"] for t in with_vessels)
        if near_area >= 0.01 and far_area > 0:
            out["tau_neuron_vessel_near_density_mm2"], out["tau_neuron_vessel_far_density_mm2"] = near_n / near_area, far_n / far_area
            out["thread_vessel_near_density_mm_per_mm2"], out["thread_vessel_far_density_mm_per_mm2"] = near_mm / near_area, far_mm / far_area
            if near_n + far_n >= MIN_NEURONS and far_n:
                out["tau_neuron_vessel_enrichment"] = (near_n / near_area) / (far_n / far_area)
            if near_mm + far_mm >= 0.5 and far_mm:
                out["thread_vessel_enrichment"] = (near_mm / near_area) / (far_mm / far_area)
    if stain == "6E10" and measured:
        profiles = {c: np.concatenate([t["profiles"][c] for t in measured if "profiles" in t] or [np.zeros(0)])
                    for c in PROFILE_MEDIANS + ("profile_core_dab",)}
        core, width = profiles["profile_core_dab"], profiles["profile_halo_width_um"]
        known = np.isfinite(core) & np.isfinite(width)
        if known.sum() >= MIN_PLAQUES:
            for c in PROFILE_MEDIANS:
                values = profiles[c][np.isfinite(profiles[c])]
                out[f"plaque_{c}_median"] = float(np.median(values)) if len(values) >= MIN_PLAQUES else float("nan")
            cored = known & (core >= CORE_LEVEL)
            out["plaque_bare_core_fraction"] = float((cored & (width < BARE_HALO_UM)).sum() / known.sum())
            out["plaque_haloed_core_fraction"] = float((cored & (width >= HALO_UM)).sum() / known.sum())
    if stain == "6E10" and seen:
        vessels = sum(t["vessels"] for t in seen)
        out["caa_vessel_density_mm2"] = sum(t["caa_vessels"] for t in seen) / area if area > 0 else float("nan")
        out["caa_vessel_fraction"] = sum(t["caa_vessels"] for t in seen) / vessels if vessels >= MIN_VESSELS else float("nan")
        edge = _cat(seen, "plaque_vessel_distance")
        if len(edge) >= MIN_PLAQUES:
            out["plaque_vessel_touch_fraction"] = float((edge <= TOUCH_UM).mean())
            out["plaque_vessel_near_fraction"] = float((edge <= NEAR_UM).mean())
    if stain == "NeuN" and measured:
        nucleus = _cat(measured, "neuron_nucleus_area")
        nucleus = nucleus[np.isfinite(nucleus)]
        out["neun_neuron_nucleus_area_median_um2"] = float(np.median(nucleus)) if len(nucleus) >= MIN_NEUN else float("nan")
        with_cells = [t for t in measured if "large" in t]
        large = sum(t["large"] for t in with_cells)
        cell_area = sum(t["area_mm2"] for t in with_cells)
        if with_cells and cell_area > 0:
            out["neun_large_nucleus_density_mm2"] = large / cell_area
            if large >= MIN_LARGE:
                for key, name in (("large_in_neuron", "in_neuron"), ("large_weak", "weak"), ("large_unstained", "unstained")):
                    out[f"neun_large_nucleus_{name}_fraction"] = sum(t[key] for t in with_cells) / large
    return out


def names_of(stain: str) -> list[str]:
    prefix = PREFIX[stain]
    base = [f"{prefix}_vessel_density_mm2", f"{prefix}_vessel_area_fraction"]
    if stain == "AT8":
        return base + ["tau_neuron_nucleus_visible_fraction", "tau_neuron_nucleus_visible_fraction_mature", "tau_neuron_nucleus_visible_fraction_pretangle",
                       "tau_neuron_nucleus_offset_median", "tau_neuron_vessel_near_density_mm2", "tau_neuron_vessel_far_density_mm2",
                       "tau_neuron_vessel_enrichment", "thread_vessel_near_density_mm_per_mm2", "thread_vessel_far_density_mm_per_mm2",
                       "thread_vessel_enrichment"]
    if stain == "6E10":
        return base + [f"plaque_{c}_median" for c in PROFILE_MEDIANS] + ["plaque_bare_core_fraction", "plaque_haloed_core_fraction",
                                                                         "caa_vessel_density_mm2", "caa_vessel_fraction",
                                                                         "plaque_vessel_touch_fraction", "plaque_vessel_near_fraction"]
    return base + ["neun_neuron_nucleus_area_median_um2", "neun_large_nucleus_density_mm2", "neun_large_nucleus_in_neuron_fraction",
                   "neun_large_nucleus_weak_fraction", "neun_large_nucleus_unstained_fraction"]


def counts(tallies: dict[str, list[dict]]) -> dict:
    """What the minimums count, for `stainid.v2.missing`."""
    at8, amyloid, neun = tallies.get("AT8", []), tallies.get("6E10", []), tallies.get("NeuN", [])
    visible, subtype = _cat(at8, "nucleus_visible"), _cat(at8, "subtype")
    known = np.isfinite(visible)
    core = np.concatenate([t["profiles"]["profile_core_dab"] for t in amyloid if "profiles" in t] or [np.zeros(0)])
    return {"nucleus_known": int(known.sum()), "nucleus_known_mature": int((known & (subtype == MATURE)).sum()),
            "nucleus_known_pretangle": int((known & (subtype == PRETANGLE)).sum()),
            "nucleus_found": int(np.isfinite(_cat(at8, "nucleus_offset")).sum()),
            "profiled_plaques": int(np.isfinite(core).sum()),
            "vessels_6e10": sum(t.get("vessels", 0) for t in amyloid),
            "plaques_vessel_known": int(len(_cat(amyloid, "plaque_vessel_distance"))),
            "near_vessel_neurons": sum(t.get("near_neurons", 0) + t.get("far_neurons", 0) for t in at8),
            "neun_nuclei_known": int(np.isfinite(_cat(neun, "neuron_nucleus_area")).sum()),
            "neun_large": sum(t.get("large", 0) for t in neun),
            **{f"bio_measured_{s}": sum(bool(t["measured"]) for t in tallies.get(s, [])) for s in PREFIX},
            **{f"vessels_measured_{s}": sum(bool(t["vessels_known"]) for t in tallies.get(s, [])) for s in PREFIX}}
