"""Spatial interactions (stage 5; no image is read): what surrounds plaques and tau+ neurons on their own section, where
lesions sit by depth from the white matter, the order of neurons in microcolumns, multi-stain microdomains on a 100 /
200 µm grid, the amyloid-tau coupling across distances, the NeuN section's nuclear graph and the plaques' size and
containment. The literature review behind each feature: `alzh/reports/Spatial interaction features for AD TMAs.md`;
definitions, minimums and the study's numbers: docs/pipeline_v2.md, "Spatial interactions".

Inputs, all stored: the accepted calls and their shown outlines (`candidates/`), stage 2's masks, the threads, the white
matter (`white_matter/`), the type scores stage 5 just wrote, the registration (the table, else the pair registered and
kept in the project's `registration` cache; a core without either has the registered features missing,
`no_registration`), and the counterstain nuclei: stage 2's per-nucleus table (`cells/`, its `counted` nuclei) where it
exists, else stage 1's Cellpose nuclei (`tile_nuclei/`, a cell's size and centre in its tile's own area; no hematoxylin
rule, so on 6E10 / AT8 dense DAB objects outlined by Cellpose are among them until the stage-2 re-run).

Everything is grey matter only with `v2.white_matter.analysis: exclude` (the counted tissue of the analysed tiles without
the white matter), all counted tissue otherwise. Random placements are seeded by the core id. A feature under its
minimum is missing (`REQUIRE`; `results_missing_<level>.csv` says why), never a pseudo-value.

Neuron-like nuclei on 6E10 / AT8: a logistic model on nuclear shape only (log area, eccentricity, major and minor axis,
their ratio), trained on the NeuN section, where a nucleus whose centre lies in an accepted NeuN+ neuron's shown outline
is a neuron's; fitted on the cohort's NeuN nuclei at every run (`MODEL_SAMPLE` per core image, seeded), checked
leave-one-TMA-out (`interactions_nucleus_model.csv`). Counts are soft: the sum of the probabilities.
"""
from __future__ import annotations

import json
import os
import zlib
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage, stats
from scipy.spatial import cKDTree

from stainid import candidates as store
from stainid.v2 import cells, organisation, white_matter

SCALE = 8  # core px per mask px (the stage-2 masks' grid)
RING_NULLS = 99  # random placements of the plaques / tau+ neurons for the ring features
NULL_TISSUE_SAMPLE = 200_000  # tissue cells whose rings the random placements count (scaled to the tissue)
CSR_NULLS = 19  # random placements of the points for the column features
SHIFT_NULLS = 99  # random shifts of the thread field for the coupling
MIN_FILLED_BINS = 3  # squares holding plaque (thread, tau+ neurons) for a coupling; else empty, too_few_objects
MODEL_SAMPLE = 3000  # NeuN nuclei per core image for the neuron-like model
MIN_PLAQUE_UM = 20.0  # plaques that get rings
FAR_UM = 100.0  # the reference: nuclei at least this far from any plaque / tau+ neuron
PLAQUE_RINGS = (("contact", 0.0, 10.0), ("ring10_30", 10.0, 30.0), ("ring30_60", 30.0, 60.0))
BODY_RINGS = (("ring15_60", 15.0, 60.0),)
DEPTH_MAX_UM, DEPTH_SPLIT_UM, DEPTH_BIN_UM = 2000.0, 1000.0, 250.0
LARGE_NEURON_UM2 = 150.0
COLUMN = {"radius_um": 150.0, "band_um": 10.0, "near_um": 30.0}  # Buldyrev's band: within 10 µm of the axis, 30-150 µm along it
ALIGN_RADIUS_UM, ALIGN_DEGREES = 300.0, 30.0
MIX_RADII = (50.0, 100.0)
MARK_RADII = (50.0, 100.0)
CONTAIN_UM, NEIGHBOURHOOD_UM = 20.0, 100.0
BIN_SIZES = (100.0, 200.0)  # the registered grid, µm (NeuN frame)
MIN_JOINT = 0.8  # a bin's share of tissue on all three sections
SHIFT_MIN_BINS, SHIFT_MAX_BINS = 3, 10
LAGS = 8  # correlogram lags (bins)
HIGH, LOW = 2 / 3, 1 / 3  # niche cut points on the within-TMA percentile ranks
NICHES = ("amyloid_tau", "amyloid_only", "tau_only", "quiet")
GAP_BIN_UM, GAP_MAX_UM = 2.0, 600.0
GAP_QUANTILE = 0.95
SATELLITE_UM = 2.0
NEURON, MATURE, PRETANGLE = "neuron", organisation.MATURE, organisation.PRETANGLE
SHAPE = ("log_area", "eccentricity", "major_um", "minor_um", "aspect")
OUTLINE_COLUMNS = [*organisation.OUTLINE_COLUMNS, "x_core", "y_core", "equivalent_diameter_um", "area_um2", "sam_area_um2"]
NUCLEUS_COLUMNS = ["x_core", "y_core", "area_um2", "eccentricity", "major_um", "minor_um", "counted"]
NUCLEI_CACHE_ENV = "STAINID_NUCLEI_TABLE_CACHE"  # a folder for stage 1's nuclei as tables (test runs only)


# ---------------------------------------------------------------- what each column needs

def _needs(*items):
    return tuple(items)


def _n(key, minimum, reason="too_few_objects"):
    return (key, float(minimum), reason)


REG = ("registered", 1, "no_registration")
REQUIRE: dict[str, tuple] = {}
for _ring in ("contact", "ring10_30", "ring30_60"):
    REQUIRE[f"plaque_{_ring}_neuron_share_ratio"] = _needs(_n("plaques_ge20um", 10), _n(f"pl_all_{_ring}_cells", 20), _n("pl_far_cells", 50))
REQUIRE["plaque_ring_neuron_share_slope"] = REQUIRE["plaque_ring10_30_neuron_share_ratio"] + (_n("pl_all_ring30_60_cells", 20),)
REQUIRE["plaque_ring10_30_neuronlike_density_ratio"] = REQUIRE["plaque_ring10_30_neuron_share_ratio"]
REQUIRE["plaque_ring10_30_other_density_ratio"] = REQUIRE["plaque_ring10_30_neuron_share_ratio"]
for _kind in ("compact", "diffuse"):
    REQUIRE[f"plaque_{_kind}_ring10_30_neuron_share_ratio"] = _needs(_n(f"plaques_{_kind}_ge20um", 10), _n(f"pl_{_kind}_ring10_30_cells", 20),
                                                                    _n("pl_far_cells", 50))
for _name, _group in (("tangle", "mature"), ("tau_pretangle", "pretangle")):
    REQUIRE[f"{_name}_ring15_60_neuron_share_ratio"] = _needs(_n(f"tb_{_group}_bodies", 10), _n(f"tb_{_group}_ring15_60_cells", 20), _n("tb_far_cells", 50))
REQUIRE["tangle_ring15_60_other_density_ratio"] = REQUIRE["tangle_ring15_60_neuron_share_ratio"]
REQUIRE["tau_mature_no_nucleus_fraction"] = _needs(_n("tau_mature_outlined", 10))
REQUIRE["tau_pretangle_no_nucleus_fraction"] = _needs(_n("tau_pretangle_outlined", 10))
REQUIRE["neun_no_nucleus_fraction"] = _needs(_n("neun_outlined", 20))
REQUIRE["tangle_ghost_excess"] = _needs(_n("tau_mature_outlined", 10), _n("neun_outlined", 20))
REQUIRE["tau_satellites_per_neuron"] = _needs(_n("tau_outlined", 10))
REQUIRE["tau_satellite_enrichment"] = _needs(_n("tau_outlined", 10), _n("tb_far_cells", 50))
for _section, _stain in (("neun", "NeuN"), ("amyloid_section", "6E10"), ("tau_section", "AT8")):
    REQUIRE[f"{_section}_microcolumn_strength"] = _needs((f"wm_{_stain}", 1, "no_white_matter_boundary"), _n(f"col_{_stain}_points", 200),
                                                         _n(f"col_{_stain}_pairs", 100))
REQUIRE["amyloid_section_neuronlike_fraction"] = _needs(_n("nl_6E10_cells", 100))
REQUIRE["tau_section_neuronlike_fraction"] = _needs(_n("nl_AT8_cells", 100))
for _r in MIX_RADII:
    REQUIRE[f"tangle_pretangle_mature_mix_{_r:.0f}um"] = _needs(_n("mix_mature", 10), _n("mix_pretangle", 10), _n(f"mix_{_r:.0f}_pairs", 20))
REQUIRE["tau_neuron_column_alignment"] = _needs(("wm_AT8", 1, "no_white_matter_boundary"), _n("align_bodies", 10), _n("align_pairs", 20))
_DEPTH = {"AT8": (("tau_neuron", "tau"), ("tangle", "mature"), ("tau_pretangle", "pretangle"), ("thread", "thread")),
          "6E10": (("plaque", "plaque"), ("plaque_compact", "compact"), ("plaque_diffuse", "diffuse")),
          "NeuN": (("neun", "neuron"), ("neun_large_neuron", "large"))}
for _stain, _kinds in _DEPTH.items():
    for _name, _kind in _kinds:
        _minimum = 0.5 if _kind == "thread" else 10
        REQUIRE[f"{_name}_depth_excess_um"] = _needs((f"wm_{_stain}", 1, "no_white_matter_boundary"), _n(f"dep_{_stain}_{_kind}_n", _minimum))
REQUIRE["tau_neuron_deep_upper_ratio"] = _needs(("wm_AT8", 1, "no_white_matter_boundary"), _n("dep_AT8_tau_n", 10),
                                                _n("dep_AT8_deep_mm2", 0.25), _n("dep_AT8_upper_mm2", 0.25))
REQUIRE["neun_deep_upper_ratio"] = _needs(("wm_NeuN", 1, "no_white_matter_boundary"), _n("dep_NeuN_neuron_n", 20),
                                          _n("dep_NeuN_deep_mm2", 0.25), _n("dep_NeuN_upper_mm2", 0.25))
REQUIRE["neun_large_neuron_peak_depth_um"] = _needs(("wm_NeuN", 1, "no_white_matter_boundary"), _n("dep_NeuN_large_n", 20))
REQUIRE["tangle_plaque_depth_difference_um"] = REQUIRE["tangle_depth_excess_um"] + REQUIRE["plaque_compact_depth_excess_um"]
# a rank correlation needs squares holding each side: plaque area / thread / tau+ neurons in at least MIN_FILLED_BINS
_FILLED = {"plaque": _n("cp_plaque_bins", MIN_FILLED_BINS), "thread": _n("cp_thread_bins", MIN_FILLED_BINS),
           "tau": _n("cp_tau_bins", MIN_FILLED_BINS), "compact": _n("cp_compact_bins", MIN_FILLED_BINS),
           "diffuse": _n("cp_diffuse_bins", MIN_FILLED_BINS)}
for _name, _sides in (("plaque_thread_coupling_rho", ("plaque", "thread")), ("plaque_thread_coupling_excess", ("plaque", "thread")),
                      ("plaque_compact_thread_coupling", ("compact", "thread")), ("plaque_diffuse_thread_coupling", ("diffuse", "thread")),
                      ("plaque_tangle_coupling_excess", ("plaque", "tau"))):
    REQUIRE[_name] = _needs(REG, _n("cp_bins", 30), *(_FILLED[k] for k in _sides))
REQUIRE["plaque_thread_coupling_depth_adjusted"] = _needs(REG, ("wm_NeuN", 1, "no_white_matter_boundary"), _n("cp_depth_bins", 30),
                                                          _FILLED["plaque"], _FILLED["thread"])
REQUIRE["plaque_thread_correlation_length_um"] = _needs(REG, _n("cp_bins", 30), _FILLED["plaque"], _FILLED["thread"],
                                                        _n("cp_length_n", 1, "not_measured"))
for _size in BIN_SIZES:
    _p = f"niche{_size:.0f}"
    for _niche in (*NICHES, "amyloid_tau_neuron_poor"):
        REQUIRE[f"{_p}_{_niche}_fraction"] = _needs(REG, _n(f"{_p}_bins", 20))
    REQUIRE[f"{_p}_entropy"] = _needs(REG, _n(f"{_p}_bins", 20))
    REQUIRE[f"{_p}_damage_quiet_contact_ratio"] = _needs(REG, _n(f"{_p}_bins", 20), _n(f"{_p}_dq_expected", 2))
    REQUIRE[f"{_p}_amyloid_tau_cohesion"] = _needs(REG, _n(f"{_p}_bins", 20), _n(f"{_p}_aa_expected", 2))
REQUIRE["neun_other_to_neuron_clq"] = _needs(_n("clq_neuron", 20), _n("clq_other", 20))
REQUIRE["neun_neuron_to_other_clq"] = REQUIRE["neun_other_to_neuron_clq"]
REQUIRE["neun_satellite_fraction"] = _needs(_n("sat_neurons", 20))
REQUIRE["neun_large_satellite_fraction"] = _needs(_n("sat_large", 10))
REQUIRE["neun_satellite_enrichment"] = _needs(_n("sat_neurons", 20), _n("sat_expected", 1))
REQUIRE["neun_gap_area_fraction"] = _needs(_n("gap_neurons", 20), _n("gap_interior_mm2", 0.5))
REQUIRE["neun_neuron_share_cv_100um"] = _needs(_n("share_windows", 10))
for _r in MARK_RADII:
    REQUIRE[f"plaque_size_mark_correlation_{_r:.0f}um"] = _needs(_n("mk_plaques", 20), _n(f"mk_{_r:.0f}_pairs", 20))
REQUIRE["plaque_compact_in_diffuse_fraction"] = _needs(_n("ct_compact", 10), _n("ct_diffuse", 5))
REQUIRE["plaque_compact_in_diffuse_enrichment"] = REQUIRE["plaque_compact_in_diffuse_fraction"]
REQUIRE["plaque_compactness_diffuse_neighbourhood_rho"] = _needs(_n("cs_plaques", 20))
COLUMNS = tuple(REQUIRE)
REASONS = ("no_white_matter_boundary", "no_registration")  # added to `missing.REASONS`


def needs(column: str) -> tuple:
    """What a column needs: (count key, minimum, reason when under it), in the order they are checked."""
    return REQUIRE[column]


# ---------------------------------------------------------------- inputs

def _grid_um(project) -> float:
    return SCALE * float(project.pixel_size_um)


def _shape_to(mask: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray | None:
    if mask is None or not mask.size:
        return None
    out = np.zeros(shape, bool)
    h, w = min(shape[0], mask.shape[0]), min(shape[1], mask.shape[1])
    out[:h, :w] = mask[:h, :w]
    return out


def tissue(project, core_id: str, stain: str, exclude: bool) -> tuple[np.ndarray | None, np.ndarray | None]:
    """The tissue the features count in (grey matter only with `exclude`) and the white matter, on the masks' grid."""
    valid = organisation.valid_tissue(project, core_id, stain)
    if valid is None:
        return None, None
    white = _shape_to(white_matter.read(project, core_id, stain), valid.shape)
    if white is not None and not white.any():
        white = None
    return (valid & ~white if exclude and white is not None else valid), white


def depth_field(white: np.ndarray | None, counted: np.ndarray, grid_um: float):
    """Depth from the white matter (µm, on the grid) and the radial axis (unit vectors along the depth's gradient), or
    None without white matter. Only tissue outside the white matter within `DEPTH_MAX_UM` with a clear axis is used."""
    if white is None or not white.any():
        return None
    depth = cv2.distanceTransform((~white).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE).astype(np.float32) * grid_um
    smooth = ndimage.gaussian_filter(depth, 50.0 / grid_um)
    gy, gx = np.gradient(smooth)
    norm = np.hypot(gx, gy)
    usable = counted & ~white & (depth <= DEPTH_MAX_UM) & (norm > 0.5 * grid_um)
    with np.errstate(invalid="ignore", divide="ignore"):
        axis = np.stack([gx / norm, gy / norm], axis=-1)
    return {"depth": depth, "axis": axis, "usable": usable}


def cells_of(xy_core: np.ndarray, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Grid cells (row, column) of points in core px and whether each lies on the grid."""
    xy_core = np.asarray(xy_core, float).reshape(-1, 2)
    ok = np.isfinite(xy_core).all(axis=1)
    cx = np.where(ok, xy_core[:, 0] // SCALE, -1).astype(int)
    cy = np.where(ok, xy_core[:, 1] // SCALE, -1).astype(int)
    ok &= (cx >= 0) & (cy >= 0) & (cx < shape[1]) & (cy < shape[0])
    return cy.clip(0, shape[0] - 1), cx.clip(0, shape[1] - 1), ok


def in_mask(mask: np.ndarray, xy_core: np.ndarray) -> np.ndarray:
    cy, cx, ok = cells_of(xy_core, mask.shape)
    return ok & mask[cy, cx]


def _npz_nuclei(project, stain: str, core_id: str) -> pd.DataFrame:
    """Stage 1's Cellpose nuclei of a core image's tiles as a table: centre (mean of the pixels), area and the shape of
    the pixels' second moments (as `skimage.measure.regionprops`), 10-200 µm², centred in the tile's own area."""
    from stainid.v2.measures import _tiles  # noqa: PLC0415

    tiles = _tiles(project, stain, core_id)
    folder, context, ps = Path(project.output("nuclei")), int(project.context_px), float(project.pixel_size_um)
    parts = []
    for t in tiles.to_dict("records") if len(tiles) else []:
        try:
            with np.load(folder / f"{t['tile_id']}.npz") as loaded:
                labels = loaded["masks"]
        except (OSError, KeyError, ValueError):
            continue
        top = int(labels.max()) if labels.size else 0
        if not top:
            continue
        flat = labels.ravel().astype(np.int64)
        n = np.bincount(flat, minlength=top + 1).astype(np.float64)
        yy, xx = np.divmod(np.arange(flat.size, dtype=np.int64), labels.shape[1])
        sx, sy = np.bincount(flat, xx, top + 1), np.bincount(flat, yy, top + 1)
        sxx, syy, sxy = np.bincount(flat, xx * xx, top + 1), np.bincount(flat, yy * yy, top + 1), np.bincount(flat, xx * yy, top + 1)
        del yy, xx, flat
        index = np.flatnonzero(n > 0)
        index = index[index > 0]
        m = n[index]
        mx, my = sx[index] / m, sy[index] / m
        cxx, cyy, cxy = sxx[index] / m - mx ** 2, syy[index] / m - my ** 2, sxy[index] / m - mx * my
        half, root = (cxx + cyy) / 2, np.sqrt(np.maximum(((cxx - cyy) / 2) ** 2 + cxy ** 2, 0))
        big, small = np.maximum(half + root, 0), np.maximum(half - root, 0)
        x0, y0 = max(0, int(t["x_px"]) - context), max(0, int(t["y_px"]) - context)
        x, y, area = mx + x0, my + y0, m * ps * ps
        own = (x >= t["x_px"]) & (x < t["x_px"] + t["width_px"]) & (y >= t["y_px"]) & (y < t["y_px"] + t["height_px"])
        own &= (area >= cells.MIN_UM2) & (area <= cells.MAX_UM2)
        with np.errstate(invalid="ignore", divide="ignore"):
            eccentricity = np.where(big > 0, np.sqrt(np.clip(1 - small / big, 0, 1)), 0.0)
        parts.append(pd.DataFrame({"x_core": x[own], "y_core": y[own], "area_um2": area[own], "eccentricity": eccentricity[own],
                                   "major_um": 4 * np.sqrt(big[own]) * ps, "minor_um": 4 * np.sqrt(small[own]) * ps, "counted": True}))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=NUCLEUS_COLUMNS)


def nuclei(project, stain: str, core_id: str) -> pd.DataFrame:
    """A core image's counted nuclei (`NUCLEUS_COLUMNS`): stage 2's per-nucleus table where it exists (its `counted`
    ones), else stage 1's Cellpose nuclei (`_npz_nuclei`)."""
    table = cells.read(project, stain, core_id)
    if table is not None:
        table = table[table.counted.astype(bool)] if len(table) else table
        return pd.DataFrame({c: table[c].to_numpy() if c in table else np.zeros(len(table)) for c in NUCLEUS_COLUMNS})
    cache = os.environ.get(NUCLEI_CACHE_ENV)
    target = Path(cache) / stain / f"{core_id}.parquet" if cache else None
    if target is not None and target.exists():
        return pd.read_parquet(target)
    frame = _npz_nuclei(project, stain, core_id)
    if target is not None:
        target.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(target.with_suffix(".tmp"))
        os.replace(target.with_suffix(".tmp"), target)
    return frame


def accepted(project, stain: str, core_id: str) -> pd.DataFrame:
    """A core image's accepted calls with their shown outline (`ring`: the first ring, core px, float32 k x 2) and
    shown area; rows without an outline keep `ring` None."""
    frame = store.read_core(project, stain, core_id, OUTLINE_COLUMNS)
    if frame.empty or "decision" not in frame:
        return pd.DataFrame(columns=[*OUTLINE_COLUMNS, "ring", "shown_um2"])
    frame = frame[frame.decision == "accepted"].copy()
    for column in OUTLINE_COLUMNS:
        if column not in frame:
            frame[column] = "" if column in store.TEXT_COLUMNS else np.nan
    for column in store.TEXT_COLUMNS:
        if column in frame:
            frame[column] = frame[column].fillna("")
    rings = []
    for row in frame.to_dict("records"):
        found = store.outline_rings(row)
        ring = np.asarray(found[0], np.float32).reshape(-1, 2) if found and len(found[0]) >= 6 else None
        rings.append(ring)
    frame["ring"] = rings
    frame["shown_um2"] = store.shown_area_um2(frame).to_numpy(float)
    return frame.reset_index(drop=True)


def to_stain(project, core_id: str, stain: str) -> np.ndarray | None:
    """The matrix from the NeuN image's core px to a stain image's: the registration table, else the pair kept in the
    project's `registration` cache (`stainid v2 register` keeps every pair there; read only here, and marked as used so
    the cache's cleanup keeps it); None without one or when it failed."""
    from stainid import disk_cache  # noqa: PLC0415
    from stainid.registration import cores as registration  # noqa: PLC0415

    found = registration._from_table(project, core_id, "NeuN", stain)
    if found is not None:
        return None if found.status == "fail" else found.matrix
    path = registration.kept_path(project.root, core_id, "NeuN", stain)
    if not path.exists():
        return None
    try:
        kept = json.loads(path.read_text())
        matrix = np.vstack([np.asarray(kept["matrix"], np.float64).reshape(2, 3), [0.0, 0.0, 1.0]])
    except (ValueError, KeyError):
        return None
    disk_cache.used(path)
    return None if kept.get("status") == "fail" else matrix


def _seed(core_id: str, what: str) -> np.random.Generator:
    return np.random.default_rng(zlib.crc32(f"{core_id}:{what}".encode()))


# ---------------------------------------------------------------- the neuron-like nucleus model

def shape_matrix(frame: pd.DataFrame) -> np.ndarray:
    area = np.maximum(frame.area_um2.to_numpy(float), 1e-3)
    major, minor = frame.major_um.to_numpy(float), frame.minor_um.to_numpy(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        aspect = np.where(major > 0, minor / major, 1.0)
    return np.nan_to_num(np.c_[np.log(area), frame.eccentricity.to_numpy(float), major, minor, aspect], nan=0.0)


def _model():
    from sklearn.linear_model import LogisticRegression  # noqa: PLC0415
    from sklearn.pipeline import make_pipeline  # noqa: PLC0415
    from sklearn.preprocessing import StandardScaler  # noqa: PLC0415

    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))


def fit_model(samples: pd.DataFrame):
    """The neuron-like model on the NeuN nuclei sample (`SHAPE` + `neuron` + `tma`) and its leave-one-TMA-out check."""
    from sklearn.metrics import roc_auc_score  # noqa: PLC0415

    rows = []
    if samples.empty or samples.neuron.nunique() < 2:
        return None, pd.DataFrame(columns=["held_out_tma", "nuclei", "neuron_share", "auc"])
    x, y = samples[list(SHAPE)].to_numpy(float), samples.neuron.to_numpy(bool)
    for tma in sorted(samples.tma.astype(str).unique()):
        test = samples.tma.astype(str).to_numpy() == tma
        if test.all() or len(np.unique(y[test])) < 2 or len(np.unique(y[~test])) < 2:
            continue
        model = _model().fit(x[~test], y[~test])
        rows.append({"held_out_tma": tma, "nuclei": int(test.sum()), "neuron_share": float(y[test].mean()),
                     "auc": float(roc_auc_score(y[test], model.predict_proba(x[test])[:, 1]))})
    return _model().fit(x, y), pd.DataFrame(rows)


def neuron_like(model, frame: pd.DataFrame) -> np.ndarray:
    if model is None or frame.empty:
        return np.full(len(frame), np.nan)
    table = pd.DataFrame(shape_matrix(frame), columns=list(SHAPE))
    return model.predict_proba(table.to_numpy(float))[:, 1]


# ---------------------------------------------------------------- rings around objects (plaques, tau+ neurons)

def _contours(frame: pd.DataFrame) -> list[np.ndarray]:
    return [np.rint(r / SCALE).astype(np.int32).reshape(-1, 1, 2) for r in frame.ring]


def _distance(shape, contours: list[np.ndarray]) -> np.ndarray:
    """µm-free grid distance from every cell to the nearest cell of these filled outlines (inf without any)."""
    if not contours:
        return np.full(shape, np.inf, np.float32)
    canvas = np.full(shape, 255, np.uint8)
    cv2.fillPoly(canvas, contours, 0)
    if canvas.all():
        return np.full(shape, np.inf, np.float32)
    return cv2.distanceTransform(canvas, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)


def _moved(contours: list[np.ndarray], centres: np.ndarray, places: np.ndarray) -> list[np.ndarray]:
    shift = np.rint(places - centres).astype(np.int32)
    return [c + s.reshape(1, 1, 2) for c, s in zip(contours, shift)]


def ring_stats(prefix: str, shape, counted: np.ndarray, groups: dict[str, pd.DataFrame], others: pd.DataFrame, cell_yx, p: np.ndarray,
               grid_um: float, rings, core_id: str, nulls: int = RING_NULLS) -> dict:
    """Counts behind the ring features: per group (compact / diffuse plaques; mature / pretangle tau+ neurons) and
    ring, the neuron-like (soft) and all nuclei and the tissue area in the ring around the group's nearest object
    (each nucleus and tissue cell goes to its nearest object of the groups, a Voronoi split), and the same far from
    every object (`FAR_UM`, `others` included); observed (`_o`) and the mean of `nulls` random placements of every
    object's outline at a random cell of the tissue (`_n`). `cell_yx`: the nuclei's grid cells (in the tissue)."""
    names = list(groups)
    ys, xs = np.nonzero(counted)
    if not len(ys):
        return {}
    y0, x0 = int(ys.min()), int(xs.min())  # work on the tissue's bounding box
    counted = counted[y0: int(ys.max()) + 1, x0: int(xs.max()) + 1]
    shape = counted.shape
    offset = np.array([x0, y0], np.int32).reshape(1, 1, 2)
    contours = {g: [c - offset for c in _contours(groups[g])] for g in names}
    rest = [c - offset for c in _contours(others)] if len(others) else []
    centres = {g: (groups[g][["x_core", "y_core"]].to_numpy(float) / SCALE - [x0, y0]) for g in names}
    rest_centres = others[["x_core", "y_core"]].to_numpy(float) / SCALE - [x0, y0] if len(others) else np.zeros((0, 2))
    ty, tx = np.nonzero(counted)
    cy, cx = np.asarray(cell_yx[0]) - y0, np.asarray(cell_yx[1]) - x0
    p = np.nan_to_num(np.asarray(p, float), nan=0.0)
    cell_area = grid_um ** 2 / 1e6
    edges = [(name, lo / grid_um, hi / grid_um) for name, lo, hi in rings]
    far = FAR_UM / grid_um
    groups_n = max(1, len(names))
    slots = len(edges) * groups_n
    # the random placements' ring areas from a fixed sample of the tissue cells (scaled to the tissue)
    sample = _seed(core_id, f"{prefix} tissue sample").permutation(len(ty))[:NULL_TISSUE_SAMPLE]
    scale_up = len(ty) / max(1, len(sample))

    def ring_code(at: np.ndarray, union_at: np.ndarray, owner_at: np.ndarray) -> np.ndarray:
        """Each point's ring x group slot (`slots`: none), from its distance to the nearest grouped object."""
        code = np.full(len(at), slots, np.int64)
        for k, (_, lo, hi) in enumerate(edges):
            inside = (at >= lo) & (at < hi) & (union_at > 0) if lo <= 0 else (at >= lo) & (at < hi)
            code[inside] = k * groups_n + owner_at[inside]
        return code

    def tally(placed: dict[str, list], placed_rest: list, cells: tuple[np.ndarray, np.ndarray, float]) -> dict:
        out = {}
        dist = [_distance(shape, placed[g]) for g in names]
        nearest = dist[0]
        owner = np.zeros(shape, np.int64)
        for k in range(1, len(dist)):
            closer = dist[k] < nearest
            owner[closer] = k
            nearest = np.where(closer, dist[k], nearest)
        union = nearest if not placed_rest else np.minimum(nearest, _distance(shape, placed_rest))
        sy, sx, weight = cells
        union_t = union[sy, sx]
        code_n = ring_code(nearest[cy, cx], union[cy, cx], owner[cy, cx])
        code_t = ring_code(nearest[sy, sx], union_t, owner[sy, sx])
        cell_area_ = cell_area * weight
        far_n, far_t = union[cy, cx] >= far, union_t >= far
        out["far_nl"], out["far_cells"], out["far_mm2"] = float(p[far_n].sum()), float(far_n.sum()), float(far_t.sum() * cell_area_)
        nl = np.bincount(code_n, weights=p, minlength=slots + 1)
        count = np.bincount(code_n, minlength=slots + 1)
        area = np.bincount(code_t, minlength=slots + 1) * cell_area_
        for k, (name, _, _) in enumerate(edges):
            part = slice(k * groups_n, (k + 1) * groups_n)
            out[f"all_{name}_nl"], out[f"all_{name}_cells"], out[f"all_{name}_mm2"] = float(nl[part].sum()), float(count[part].sum()), float(area[part].sum())
            for g, group in enumerate(names):
                j = k * groups_n + g
                out[f"{group}_{name}_nl"], out[f"{group}_{name}_cells"], out[f"{group}_{name}_mm2"] = float(nl[j]), float(count[j]), float(area[j])
        return out

    observed = tally(contours, rest, (ty, tx, 1.0))
    result = {f"{prefix}_{k}": v for k, v in observed.items()}
    result |= {f"{prefix}_{k}_o": v for k, v in observed.items()}
    if len(ty) and any(len(contours[g]) for g in names):
        rng = _seed(core_id, prefix)
        total = {k: 0.0 for k in observed}
        for _ in range(nulls):
            placed = {}
            for g in names:
                pick = rng.integers(0, len(ty), len(contours[g]))
                placed[g] = _moved(contours[g], centres[g], np.c_[tx[pick], ty[pick]].astype(float))
            pick = rng.integers(0, len(ty), len(rest))
            drawn = tally(placed, _moved(rest, rest_centres, np.c_[tx[pick], ty[pick]].astype(float)) if len(rest) else [],
                          (ty[sample], tx[sample], scale_up))
            for k, v in drawn.items():
                total[k] += v
        result |= {f"{prefix}_{k}_n": v / nulls for k, v in total.items()}
    return result


def share_ratio(s: dict, prefix: str, group: str, ring: str, kind: str = "share") -> float:
    """Observed over null: the neuron-like share (`share`), the neuron-like density (`neuronlike`) or the other nuclei's
    density (`other`) in a ring relative to far tissue."""
    def one(tag):
        nl, n, a = s.get(f"{prefix}_{group}_{ring}_nl_{tag}", 0), s.get(f"{prefix}_{group}_{ring}_cells_{tag}", 0), s.get(f"{prefix}_{group}_{ring}_mm2_{tag}", 0)
        fnl, fn, fa = s.get(f"{prefix}_far_nl_{tag}", 0), s.get(f"{prefix}_far_cells_{tag}", 0), s.get(f"{prefix}_far_mm2_{tag}", 0)
        with np.errstate(invalid="ignore", divide="ignore"):
            if kind == "share":
                return (nl / n) / (fnl / fn) if n and fn and fnl else np.nan
            if kind == "neuronlike":
                return (nl / a) / (fnl / fa) if a and fa and fnl else np.nan
            return ((n - nl) / a) / ((fn - fnl) / fa) if a and fa and fn - fnl > 0 else np.nan
    obs, null = one("o"), one("n")
    return float(obs / null) if np.isfinite(obs) and np.isfinite(null) and null > 0 else np.nan


# ---------------------------------------------------------------- columns along the radial axis

def _axis_at(field: dict, xy_core: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cy, cx, ok = cells_of(xy_core, field["usable"].shape)
    ok &= field["usable"][cy, cx]
    return field["axis"][cy, cx], ok


def column_counts(xy_um: np.ndarray, axis: np.ndarray) -> tuple[float, float, float]:
    """Buldyrev's microcolumn band: pairs within `band_um` of the radial axis and `near_um`-`radius_um` along it (C),
    the same band turned by 90° (R), and all pairs within `radius_um`."""
    if len(xy_um) < 2:
        return 0.0, 0.0, 0.0
    pairs = cKDTree(xy_um).query_pairs(COLUMN["radius_um"], output_type="ndarray")
    if not len(pairs):
        return 0.0, 0.0, 0.0
    d = xy_um[pairs[:, 1]] - xy_um[pairs[:, 0]]
    u = axis[pairs[:, 0]]
    along = np.abs(d[:, 0] * u[:, 0] + d[:, 1] * u[:, 1])
    across = np.abs(d[:, 0] * u[:, 1] - d[:, 1] * u[:, 0])
    band, near, far = COLUMN["band_um"], COLUMN["near_um"], COLUMN["radius_um"]
    radial = (across <= band) & (along >= near) & (along <= far)
    lateral = (along <= band) & (across >= near) & (across <= far)
    return float(radial.sum()), float(lateral.sum()), float(len(pairs))


def alignment_counts(xy_um: np.ndarray, axis: np.ndarray) -> tuple[float, float]:
    """Pairs within `ALIGN_RADIUS_UM` and those within `ALIGN_DEGREES` of the radial axis."""
    if len(xy_um) < 2:
        return 0.0, 0.0
    pairs = cKDTree(xy_um).query_pairs(ALIGN_RADIUS_UM, output_type="ndarray")
    if not len(pairs):
        return 0.0, 0.0
    d = xy_um[pairs[:, 1]] - xy_um[pairs[:, 0]]
    u = (axis[pairs[:, 0]] + axis[pairs[:, 1]] * np.sign(np.sum(axis[pairs[:, 0]] * axis[pairs[:, 1]], axis=1, keepdims=True)))
    u /= np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-9)
    cos = np.abs(np.sum(d * u, axis=1)) / np.maximum(np.linalg.norm(d, axis=1), 1e-9)
    return float((cos >= np.cos(np.radians(ALIGN_DEGREES))).sum()), float(len(pairs))


def _random_points(field: dict, n: int, rng) -> tuple[np.ndarray, np.ndarray]:
    ys, xs = np.nonzero(field["usable"])
    pick = rng.integers(0, len(ys), n)
    xy = np.c_[(xs[pick] + rng.random(n)) * SCALE, (ys[pick] + rng.random(n)) * SCALE]
    return xy, field["axis"][ys[pick], xs[pick]]


def column_stats(key: str, xy_core: np.ndarray, field: dict | None, ps: float, core_id: str) -> dict:
    """`col_<key>_*`: the microcolumn counts of points (core px) and of `CSR_NULLS` random placements of as many in the
    same tissue (with its axis)."""
    out = {f"wm_{key}": 0.0, f"col_{key}_points": 0.0, f"col_{key}_pairs": 0.0}
    if field is None or not field["usable"].any():
        return out
    axis, ok = _axis_at(field, xy_core)
    xy = np.asarray(xy_core, float)[ok]
    out[f"wm_{key}"] = 1.0
    radial, lateral, pairs = column_counts(xy * ps, axis[ok])
    out |= {f"col_{key}_points": float(len(xy)), f"col_{key}_pairs": pairs, f"col_{key}_radial_o": radial, f"col_{key}_lateral_o": lateral}
    if len(xy) >= 2:
        rng = _seed(core_id, f"column {key}")
        nr = nl = 0.0
        for _ in range(CSR_NULLS):
            points, axes = _random_points(field, len(xy), rng)
            r, la, _ = column_counts(points * ps, axes)
            nr, nl = nr + r, nl + la
        out |= {f"col_{key}_radial_n": nr / CSR_NULLS, f"col_{key}_lateral_n": nl / CSR_NULLS}
    return out


def _ratio_of_ratios(s: dict, a: str, b: str) -> float:
    o, n = (s.get(f"{a}_o", 0), s.get(f"{b}_o", 0)), (s.get(f"{a}_n", 0), s.get(f"{b}_n", 0))
    if not (o[1] and n[0] and n[1]):
        return np.nan
    return float((o[0] / o[1]) / (n[0] / n[1]))


# ---------------------------------------------------------------- depth from the white matter

def depth_stats(stain: str, field: dict | None, kinds: dict[str, tuple[np.ndarray, np.ndarray]], grid_um: float) -> dict:
    """`dep_<stain>_*`: the tissue's and each kind of object's counts and summed depth within `DEPTH_MAX_UM` of the white
    matter, deep (< `DEPTH_SPLIT_UM`) and upper; `kinds`: name -> (points core px, weights)."""
    out = {f"wm_{stain}": 0.0}
    if field is None or not field["usable"].any():
        return out
    depth, usable = field["depth"], field["usable"]
    cell = grid_um ** 2 / 1e6
    values = depth[usable]
    out |= {f"wm_{stain}": 1.0, f"dep_{stain}_tissue_n": float(len(values)), f"dep_{stain}_tissue_sum": float(values.sum()),
            f"dep_{stain}_deep_mm2": float((values < DEPTH_SPLIT_UM).sum() * cell), f"dep_{stain}_upper_mm2": float((values >= DEPTH_SPLIT_UM).sum() * cell)}
    bins = np.arange(0, DEPTH_MAX_UM + DEPTH_BIN_UM, DEPTH_BIN_UM)
    tissue_bins = np.histogram(values, bins)[0] * cell
    for k, area in enumerate(tissue_bins):
        out[f"dep_{stain}_bin{k}_mm2"] = float(area)
    for name, (xy, weight) in kinds.items():
        cy, cx, ok = cells_of(xy, usable.shape)
        ok &= usable[cy, cx]
        d, w = depth[cy[ok], cx[ok]], np.asarray(weight, float)[ok]
        out |= {f"dep_{stain}_{name}_n": float(w.sum()), f"dep_{stain}_{name}_sum": float((d * w).sum()),
                f"dep_{stain}_{name}_deep": float(w[d < DEPTH_SPLIT_UM].sum()), f"dep_{stain}_{name}_upper": float(w[d >= DEPTH_SPLIT_UM].sum())}
        if name == "large":
            for k, count in enumerate(np.histogram(d, bins, weights=w)[0]):
                out[f"dep_{stain}_large_bin{k}"] = float(count)
    return out


def _excess(s: dict, stain: str, kind: str) -> float:
    n, tn = s.get(f"dep_{stain}_{kind}_n", 0), s.get(f"dep_{stain}_tissue_n", 0)
    return float(s[f"dep_{stain}_{kind}_sum"] / n - s[f"dep_{stain}_tissue_sum"] / tn) if n and tn else np.nan


def _deep_upper(s: dict, stain: str, kind: str) -> float:
    deep, upper = s.get(f"dep_{stain}_{kind}_deep", 0), s.get(f"dep_{stain}_{kind}_upper", 0)
    a, b = s.get(f"dep_{stain}_deep_mm2", 0), s.get(f"dep_{stain}_upper_mm2", 0)
    return float((deep / a) / (upper / b)) if a and b and upper else np.nan


def _peak_depth(s: dict, stain: str) -> float:
    density = []
    for k in range(int(DEPTH_MAX_UM / DEPTH_BIN_UM)):
        area = s.get(f"dep_{stain}_bin{k}_mm2", 0)
        density.append(s.get(f"dep_{stain}_large_bin{k}", 0) / area if area >= 0.05 else -1)
    if max(density) <= 0:
        return np.nan
    return float((int(np.argmax(density)) + 0.5) * DEPTH_BIN_UM)


# ---------------------------------------------------------------- tau: maturity mixing, nuclei in somata, satellites

def mix_stats(xy_um: np.ndarray, mature: np.ndarray) -> dict:
    """Pretangle-mature pairs within each radius against random labelling of maturity among the same bodies (its exact
    expectation)."""
    n, nm = len(xy_um), int(mature.sum())
    out = {"mix_mature": float(nm), "mix_pretangle": float(n - nm)}
    tree = cKDTree(xy_um) if n >= 2 else None
    for r in MIX_RADII:
        pairs = tree.query_pairs(r, output_type="ndarray") if tree is not None else np.zeros((0, 2), int)
        mixed = float((mature[pairs[:, 0]] != mature[pairs[:, 1]]).sum()) if len(pairs) else 0.0
        expected = len(pairs) * 2 * nm * (n - nm) / (n * (n - 1)) if n >= 2 else 0.0
        out |= {f"mix_{r:.0f}_pairs": float(len(pairs)), f"mix_{r:.0f}_mixed": mixed, f"mix_{r:.0f}_expected": float(expected)}
    return out


def nucleus_inside(bodies: pd.DataFrame, nuclei_frame: pd.DataFrame, max_share: float = 0.5) -> np.ndarray:
    """Whether each body's shown outline holds a nucleus centre; a "nucleus" covering more than `max_share` of the body
    is the body itself outlined by Cellpose, not its nucleus."""
    found = np.zeros(len(bodies), bool)
    if not len(bodies) or nuclei_frame.empty:
        return found
    xy = nuclei_frame[["x_core", "y_core"]].to_numpy(float)
    area = nuclei_frame.area_um2.to_numpy(float)
    tree = cKDTree(xy)
    for i, row in enumerate(bodies.itertuples()):
        ring = row.ring
        if ring is None:
            continue
        lo, hi = ring.min(axis=0), ring.max(axis=0)
        near = tree.query_ball_point((lo + hi) / 2, float(np.hypot(*(hi - lo)) / 2 + 1))
        contour = ring.reshape(-1, 1, 2)
        for j in near:
            if area[j] <= max_share * row.shown_um2 and cv2.pointPolygonTest(contour, (float(xy[j, 0]), float(xy[j, 1])), False) >= 0:
                found[i] = True
                break
    return found


def satellite_counts(bodies: pd.DataFrame, nuclei_frame: pd.DataFrame, others: np.ndarray, ps: float) -> tuple[np.ndarray, float]:
    """Per body (with an outline) the `others` nuclei just outside it (centre within the nucleus's radius +
    `SATELLITE_UM` of the outline, not inside it), and the summed band area (µm²: perimeter x the median band width)."""
    counts = np.zeros(len(bodies))
    xy = nuclei_frame[["x_core", "y_core"]].to_numpy(float)[others]
    radius_px = np.sqrt(nuclei_frame.area_um2.to_numpy(float)[others] / np.pi) / ps
    reach = radius_px + SATELLITE_UM / ps
    band_um = float(np.median(reach) * ps) if len(reach) else 0.0
    area = 0.0
    tree = cKDTree(xy) if len(xy) else None
    for i, row in enumerate(bodies.itertuples()):
        ring = row.ring
        if ring is None:
            continue
        contour = ring.reshape(-1, 1, 2)
        area += cv2.arcLength(contour, True) * ps * band_um
        if tree is None:
            continue
        lo, hi = ring.min(axis=0), ring.max(axis=0)
        near = tree.query_ball_point((lo + hi) / 2, float(np.hypot(*(hi - lo)) / 2 + reach.max() + 1))
        for j in near:
            d = cv2.pointPolygonTest(contour, (float(xy[j, 0]), float(xy[j, 1])), True)
            if -reach[j] <= d < 0:
                counts[i] += 1
    return counts, area


# ---------------------------------------------------------------- the NeuN section's nuclear graph

def clq_stats(xy: np.ndarray, neuron: np.ndarray) -> dict:
    """Co-location quotient counts: nuclei of each kind and those whose nearest other nucleus is of the other kind."""
    out = {"clq_neuron": float(neuron.sum()), "clq_other": float((~neuron).sum()), "clq_all": float(len(xy))}
    if len(xy) < 3:
        return out | {"clq_on": 0.0, "clq_no": 0.0}
    nearest = cKDTree(xy).query(xy, k=2)[1][:, 1]
    out["clq_on"] = float((~neuron & neuron[nearest]).sum())  # other whose nearest is a neuron's
    out["clq_no"] = float((neuron & ~neuron[nearest]).sum())
    return out


def _clq(s: dict, a: str) -> float:
    n, nn, no, total = s.get("clq_all", 0), s.get("clq_neuron", 0), s.get("clq_other", 0), s.get("clq_all", 0)
    if a == "other":
        return float((s["clq_on"] / no) / (nn / (total - 1))) if no and nn and n > 1 else np.nan
    return float((s["clq_no"] / nn) / (no / (total - 1))) if no and nn and n > 1 else np.nan


def gap_stats(neuron_xy: np.ndarray, counted: np.ndarray, grid_um: float) -> dict:
    """The histogram (`GAP_BIN_UM` bins) of the distance from each tissue cell to the nearest neuron, over the cells whose
    empty circle lies inside the tissue (the distance to the tissue's edge is larger)."""
    shape = counted.shape
    cy, cx, ok = cells_of(neuron_xy, shape)
    canvas = np.full(shape, 255, np.uint8)
    canvas[cy[ok], cx[ok]] = 0
    out = {"gap_neurons": float(ok.sum())}
    if not ok.any():
        return out
    to_neuron = cv2.distanceTransform(canvas, cv2.DIST_L2, cv2.DIST_MASK_PRECISE) * grid_um
    to_edge = cv2.distanceTransform(counted.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE) * grid_um
    interior = counted & (to_edge > to_neuron)
    values = to_neuron[interior]
    out["gap_interior_mm2"] = float(interior.sum() * grid_um ** 2 / 1e6)
    hist = np.histogram(np.minimum(values, GAP_MAX_UM - 1e-3), np.arange(0, GAP_MAX_UM + GAP_BIN_UM, GAP_BIN_UM))[0]
    for k in np.flatnonzero(hist):
        out[f"gaphist_{k}"] = float(hist[k])
    return out


def _gap_bins() -> int:
    return int(GAP_MAX_UM / GAP_BIN_UM)


def gap_threshold(control: list[dict]) -> float:
    """The `GAP_QUANTILE` of the empty-circle distance over the control cores' interior tissue."""
    hist = np.zeros(_gap_bins())
    for s in control:
        for k in range(len(hist)):
            hist[k] += s.get(f"gaphist_{k}", 0)
    if not hist.sum():
        return np.nan
    k = int(np.searchsorted(np.cumsum(hist) / hist.sum(), GAP_QUANTILE))
    return float((k + 1) * GAP_BIN_UM)


def share_windows(nuclei_xy: np.ndarray, neuron: np.ndarray, counted: np.ndarray, grid_um: float, ps: float) -> list[float]:
    """The neuron share of the nuclei in each 100 µm window that is at least 80 % tissue and holds 10 nuclei."""
    size = 100.0
    gy, gx = np.indices(counted.shape)
    bx, by = (((gx + 0.5) * grid_um) // size).astype(int), (((gy + 0.5) * grid_um) // size).astype(int)
    width = int(bx.max()) + 1
    cells_in = np.bincount((by * width + bx).ravel(), minlength=(int(by.max()) + 1) * width)
    tissue_in = np.bincount((by * width + bx).ravel(), weights=counted.ravel(), minlength=len(cells_in))
    if not len(nuclei_xy):
        return []
    um = np.asarray(nuclei_xy, float) * ps
    ix, iy = (um[:, 0] // size).astype(int), (um[:, 1] // size).astype(int)
    keep = (ix >= 0) & (iy >= 0) & (ix < width) & (iy <= int(by.max()))
    index = iy[keep] * width + ix[keep]
    total = np.bincount(index, minlength=len(cells_in))
    neurons = np.bincount(index, weights=neuron[keep].astype(float), minlength=len(cells_in))
    good = (cells_in > 0) & (tissue_in >= 0.8 * cells_in) & (total >= 10)
    return list((neurons[good] / total[good]).astype(float))


# ---------------------------------------------------------------- plaques: size marks and containment

def mark_stats(xy_um: np.ndarray, marks: np.ndarray) -> dict:
    out = {"mk_plaques": float(len(xy_um)), "mk_sum": float(marks.sum()), "mk_sum2": float((marks ** 2).sum())}
    tree = cKDTree(xy_um) if len(xy_um) >= 2 else None
    for r in MARK_RADII:
        pairs = tree.query_pairs(r, output_type="ndarray") if tree is not None else np.zeros((0, 2), int)
        out[f"mk_{r:.0f}_pairs"] = float(len(pairs))
        out[f"mk_{r:.0f}_product"] = float((marks[pairs[:, 0]] * marks[pairs[:, 1]]).sum()) if len(pairs) else 0.0
    return out


def _mark(s: dict, r: float) -> float:
    pairs, n = s.get(f"mk_{r:.0f}_pairs", 0), s.get("mk_plaques", 0)
    if not pairs or not n:
        return np.nan
    return float((s[f"mk_{r:.0f}_product"] / pairs) / (s["mk_sum"] / n) ** 2)


# ---------------------------------------------------------------- the registered grid (NeuN frame)

def _grid_matrix(matrix: np.ndarray) -> np.ndarray:
    """A core-px matrix (NeuN -> stain) on the masks' grid, cell centres matched (as `white_matter._look`)."""
    m = np.array(matrix, np.float64)[:2].copy()
    m[:, 2] /= SCALE
    m[:, 2] += (0.5 * (m[:, 0] + m[:, 1]) - 0.5)
    return m


def onto_neun(mask: np.ndarray, matrix: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    return cv2.warpAffine(mask.astype(np.uint8), _grid_matrix(matrix), (shape[1], shape[0]), flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0


def to_neun_points(xy: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    inverse = np.linalg.inv(matrix)
    xy = np.asarray(xy, float).reshape(-1, 2)
    return xy @ inverse[:2, :2].T + inverse[:2, 2]


def bin_table(joint: np.ndarray, grid_um: float, ps: float, size_um: float, layers: dict[str, tuple[np.ndarray, np.ndarray]],
              depth: np.ndarray | None) -> pd.DataFrame:
    """The bins of `size_um` (NeuN core px frame) that are at least `MIN_JOINT` tissue on all three sections, with each
    layer's summed weight (`layers`: name -> (points NeuN core px, weights)), the tissue area and the mean depth."""
    gy, gx = np.indices(joint.shape)
    bx, by = (((gx + 0.5) * grid_um) // size_um).astype(int), (((gy + 0.5) * grid_um) // size_um).astype(int)
    width, height = int(bx.max()) + 1, int(by.max()) + 1
    index = (by * width + bx).ravel()
    cells_in = np.bincount(index, minlength=width * height).astype(float)
    tissue_in = np.bincount(index, weights=joint.ravel(), minlength=width * height)
    good = (cells_in > 0) & (tissue_in >= MIN_JOINT * cells_in) & (cells_in >= 0.5 * (size_um / grid_um) ** 2)
    out = pd.DataFrame({"ix": np.flatnonzero(good) % width, "iy": np.flatnonzero(good) // width,
                        "tissue_mm2": tissue_in[good] * grid_um ** 2 / 1e6})
    for name, (xy, weight) in layers.items():
        um = np.asarray(xy, float).reshape(-1, 2) * ps
        ix, iy = np.floor(um[:, 0] / size_um).astype(int), np.floor(um[:, 1] / size_um).astype(int)
        keep = (ix >= 0) & (iy >= 0) & (ix < width) & (iy < height)
        if len(um):
            gx_, gy_ = np.clip((um[:, 0] / grid_um).astype(int), 0, joint.shape[1] - 1), np.clip((um[:, 1] / grid_um).astype(int), 0, joint.shape[0] - 1)
            keep &= joint[gy_, gx_]  # only what lies in the joint tissue
        summed = np.bincount(iy[keep] * width + ix[keep], weights=np.asarray(weight, float)[keep], minlength=width * height)
        out[name] = summed[good]
    if depth is not None:
        d = np.bincount(index, weights=np.where(joint, np.nan_to_num(depth, nan=0.0), 0).ravel(), minlength=width * height)
        out["depth_um"] = np.where(tissue_in[good] > 0, d[good] / np.maximum(tissue_in[good], 1), np.nan)
    return out


def _spearman(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return np.nan
    return float(stats.spearmanr(a, b).statistic)


def coupling_stats(bins: pd.DataFrame, core_id: str) -> dict:
    """The plaque-thread coupling on the 100 µm bins: Spearman rho of plaque area and thread length; less its mean over
    `SHIFT_NULLS` shifts of the thread field by `SHIFT_MIN_BINS`-`SHIFT_MAX_BINS` bins (overlapping bins only); with
    depth partialled out; the correlation length (the lag where rho of the plaque bin with the mean thread `k` bins away
    falls to half its lag-0 value); compact and diffuse plaque area separately; tangles against plaque area."""
    out = {"cp_bins": float(len(bins))}
    for name, key in (("plaque_um2", "cp_plaque_bins"), ("thread_mm", "cp_thread_bins"), ("tau", "cp_tau_bins"),
                      ("compact_um2", "cp_compact_bins"), ("diffuse_um2", "cp_diffuse_bins")):
        out[key] = float((bins[name].to_numpy(float) > 0).sum()) if len(bins) else 0.0
    if len(bins) < 3:
        return out
    density = lambda name: bins[name].to_numpy(float) / bins.tissue_mm2.to_numpy(float)  # noqa: E731
    plaque, thread, tangles = density("plaque_um2"), density("thread_mm"), density("tau")
    rho = _spearman(plaque, thread)
    out |= {"cp_rho__sum": rho * len(bins), "cp_rho__w": float(len(bins)) if np.isfinite(rho) else 0.0}
    for name, key in (("compact_um2", "cp_compact"), ("diffuse_um2", "cp_diffuse")):
        r = _spearman(density(name), thread)
        out |= {f"{key}__sum": r * len(bins) if np.isfinite(r) else 0.0, f"{key}__w": float(len(bins)) if np.isfinite(r) else 0.0}
    lookup = {(int(x), int(y)): i for i, (x, y) in enumerate(zip(bins.ix, bins.iy))}
    rng = _seed(core_id, "coupling")
    shifts = [(dx, dy) for dx in range(-SHIFT_MAX_BINS, SHIFT_MAX_BINS + 1) for dy in range(-SHIFT_MAX_BINS, SHIFT_MAX_BINS + 1)
              if max(abs(dx), abs(dy)) >= SHIFT_MIN_BINS]

    def shifted(values, target):
        nulls = []
        for k in rng.permutation(len(shifts)):
            dx, dy = shifts[k]
            pairs = [(i, lookup[(x + dx, y + dy)]) for (x, y), i in lookup.items() if (x + dx, y + dy) in lookup]
            if len(pairs) < 30:
                continue
            a, b = np.array(pairs).T
            r = _spearman(target[a], values[b])
            if np.isfinite(r):
                nulls.append(r)
            if len(nulls) >= SHIFT_NULLS:
                break
        return float(np.mean(nulls)) if nulls else np.nan

    null = shifted(thread, plaque)
    if np.isfinite(rho) and np.isfinite(null):
        out |= {"cp_excess__sum": (rho - null) * len(bins), "cp_excess__w": float(len(bins))}
    tangle_rho = _spearman(plaque, tangles)
    tangle_null = shifted(tangles, plaque)
    if np.isfinite(tangle_rho) and np.isfinite(tangle_null):
        out |= {"cp_tangle__sum": (tangle_rho - tangle_null) * len(bins), "cp_tangle__w": float(len(bins))}
    if "depth_um" in bins and bins.depth_um.notna().sum() >= 3:
        known = bins.depth_um.notna().to_numpy()
        layer = (bins.depth_um[known].to_numpy(float) // DEPTH_BIN_UM).astype(int)
        ra, rb = stats.rankdata(plaque[known]), stats.rankdata(thread[known])
        for k in np.unique(layer):
            ra[layer == k] -= ra[layer == k].mean()
            rb[layer == k] -= rb[layer == k].mean()
        r = float(np.corrcoef(ra, rb)[0, 1]) if np.std(ra) > 0 and np.std(rb) > 0 else np.nan
        out["cp_depth_bins"] = float(known.sum())
        if np.isfinite(r):
            out |= {"cp_depth__sum": r * known.sum(), "cp_depth__w": float(known.sum())}
    if np.isfinite(rho) and rho >= 0.1:
        xs, ys = bins.ix.to_numpy(int), bins.iy.to_numpy(int)
        curve = [rho]
        for lag in range(1, LAGS + 1):
            ring_mean = np.full(len(bins), np.nan)
            for i, (x, y) in enumerate(zip(xs, ys)):
                near = [lookup[(x + dx, y + dy)] for dx in range(-lag, lag + 1) for dy in range(-lag, lag + 1)
                        if max(abs(dx), abs(dy)) == lag and (x + dx, y + dy) in lookup]
                if near:
                    ring_mean[i] = thread[near].mean()
            ok = np.isfinite(ring_mean)
            curve.append(_spearman(plaque[ok], ring_mean[ok]) if ok.sum() >= 30 else np.nan)
        curve = np.asarray(curve)
        below = [k for k in range(1, len(curve)) if np.isfinite(curve[k]) and curve[k] <= rho / 2]
        if below:
            k = below[0]
            before = curve[k - 1] if np.isfinite(curve[k - 1]) else rho
            frac = (before - rho / 2) / (before - curve[k]) if before != curve[k] else 1.0
            out |= {"cp_length__sum": (k - 1 + frac) * BIN_SIZES[0], "cp_length__w": 1.0, "cp_length_n": 1.0}
    return out


def _mean(s: dict, key: str) -> float:
    w = s.get(f"{key}__w", 0)
    return float(s[f"{key}__sum"] / w) if w else np.nan


def niche_labels(bins: pd.DataFrame) -> np.ndarray:
    """Each bin's niche from its within-TMA percentile ranks (`a`, `t`, `n` columns: amyloid, tau, neurons): amyloid and
    tau both in the top third, one of them, or neither; `neuron_poor` when the neurons are in the bottom third."""
    a, t = bins.a.to_numpy(float) >= HIGH, bins.t.to_numpy(float) >= HIGH
    base = np.where(a & t, 0, np.where(a, 1, np.where(t, 2, 3)))
    return base * 2 + (bins.n.to_numpy(float) <= LOW).astype(int)


def niche_stats(prefix: str, bins: pd.DataFrame, labels: np.ndarray) -> dict:
    """`<prefix>_*`: bins per niche, and the four-neighbour contacts of damaged (amyloid and tau) with quiet bins and of
    damaged with damaged bins, observed and expected under random relabelling of the core's bins."""
    out = {f"{prefix}_bins": float(len(bins))}
    for k in range(8):
        out[f"{prefix}_label{k}"] = float((labels == k).sum())
    lookup = {(int(x), int(y)): i for i, (x, y) in enumerate(zip(bins.ix, bins.iy))}
    edges = [(i, lookup[(x + 1, y)]) for (x, y), i in lookup.items() if (x + 1, y) in lookup]
    edges += [(i, lookup[(x, y + 1)]) for (x, y), i in lookup.items() if (x, y + 1) in lookup]
    base = labels // 2
    n = len(bins)
    nd, nq = float((base == 0).sum()), float((base == 3).sum())
    e = np.array(edges).reshape(-1, 2)
    if len(e):
        a, b = base[e[:, 0]], base[e[:, 1]]
        out[f"{prefix}_dq_observed"] = float((((a == 0) & (b == 3)) | ((a == 3) & (b == 0))).sum())
        out[f"{prefix}_aa_observed"] = float(((a == 0) & (b == 0)).sum())
    else:
        out[f"{prefix}_dq_observed"] = out[f"{prefix}_aa_observed"] = 0.0
    out[f"{prefix}_dq_expected"] = len(e) * 2 * nd * nq / (n * (n - 1)) if n > 1 else 0.0
    out[f"{prefix}_aa_expected"] = len(e) * nd * (nd - 1) / (n * (n - 1)) if n > 1 else 0.0
    return out


def _niche_columns(s: dict, prefix: str) -> dict:
    n = s.get(f"{prefix}_bins", 0)
    counts = np.array([s.get(f"{prefix}_label{k}", 0) for k in range(8)], float)
    out = {}
    for k, name in enumerate(NICHES):
        out[f"{prefix}_{name}_fraction"] = float(counts[2 * k: 2 * k + 2].sum() / n) if n else np.nan
    out[f"{prefix}_amyloid_tau_neuron_poor_fraction"] = float(counts[1] / n) if n else np.nan
    share = counts[counts > 0] / n if n else np.zeros(0)
    out[f"{prefix}_entropy"] = float(-(share * np.log(share)).sum() / np.log(8)) if n else np.nan
    for key, column in (("dq", "damage_quiet_contact_ratio"), ("aa", "amyloid_tau_cohesion")):
        expected = s.get(f"{prefix}_{key}_expected", 0)
        out[f"{prefix}_{column}"] = float(s.get(f"{prefix}_{key}_observed", 0) / expected) if expected else np.nan
    return out


# ---------------------------------------------------------------- per core

def _thread_points(threads: pd.DataFrame | None, accepted_ids: set, ps: float) -> tuple[np.ndarray, np.ndarray]:
    """Free thread (outside accepted tau+ neurons) as its vertices (core px) and each vertex's share of its piece's
    length (mm)."""
    if threads is None or threads.empty or "points" not in threads:
        return np.zeros((0, 2)), np.zeros(0)
    free = threads[~threads.candidate_id.isin(accepted_ids)] if "candidate_id" in threads else threads
    xy, w = [], []
    for points, n_px in zip(free.points, free.n_px):
        p = organisation._points(points)
        if len(p):
            xy.append(p)
            w.append(np.full(len(p), float(n_px) * ps / 1000 / len(p)))
    return (np.vstack(xy), np.concatenate(w)) if xy else (np.zeros((0, 2)), np.zeros(0))


def _cell_yx(frame: pd.DataFrame, shape) -> tuple[np.ndarray, np.ndarray]:
    cy, cx, _ = cells_of(frame[["x_core", "y_core"]].to_numpy(float), shape)
    return cy, cx


def neun_core(project, core_id: str, exclude: bool) -> dict:
    """Pass 1, the NeuN section: the nuclear graph, the microcolumns, the depth of neurons, the neurons without a nucleus,
    the nuclei sample for the neuron-like model and what the registered grid needs."""
    ps, grid_um = float(project.pixel_size_um), _grid_um(project)
    counted, white = tissue(project, core_id, "NeuN", exclude)
    out = {"stats": {}, "sample": None, "grid": None}
    if counted is None:
        return out
    s = out["stats"]
    neurons = accepted(project, "NeuN", core_id)
    neurons = neurons[in_mask(counted, neurons[["x_core", "y_core"]].to_numpy(float))] if len(neurons) else neurons
    outlined = neurons[neurons.ring.notna()] if len(neurons) else neurons
    nuc = nuclei(project, "NeuN", core_id)
    nuc = nuc[in_mask(counted, nuc[["x_core", "y_core"]].to_numpy(float))].reset_index(drop=True) if len(nuc) else nuc
    rings = [[r] for r in outlined.ring] if len(outlined) else []
    is_neuron = cells.in_outlines(nuc, rings) if len(nuc) else np.zeros(0, bool)
    xy = nuc[["x_core", "y_core"]].to_numpy(float) if len(nuc) else np.zeros((0, 2))
    s |= clq_stats(xy * ps, is_neuron)
    # satellites: other nuclei just outside an accepted neuron's outline
    sat, band_um2 = satellite_counts(outlined, nuc, ~is_neuron, ps) if len(outlined) and len(nuc) else (np.zeros(len(outlined)), 0.0)
    large = outlined.shown_um2.to_numpy(float) >= LARGE_NEURON_UM2 if len(outlined) else np.zeros(0, bool)
    area_mm2 = float(counted.sum() * grid_um ** 2 / 1e6)
    other_density = float((~is_neuron).sum() / (area_mm2 * 1e6)) if area_mm2 else 0.0  # per µm²
    s |= {"sat_neurons": float(len(outlined)), "sat_with": float((sat > 0).sum()), "sat_large": float(large.sum()),
          "sat_large_with": float((sat[large] > 0).sum()), "sat_observed": float(sat.sum()), "sat_expected": other_density * band_um2}
    s |= gap_stats(neurons[["x_core", "y_core"]].to_numpy(float) if len(neurons) else np.zeros((0, 2)), counted, grid_um)
    shares = share_windows(xy, is_neuron, counted, grid_um, ps)
    if len(shares) >= 2 and np.mean(shares) > 0:
        s |= {"share_windows": float(len(shares)), "share_cv__sum": float(np.std(shares, ddof=1) / np.mean(shares)) * len(shares),
              "share_cv__w": float(len(shares))}
    else:
        s["share_windows"] = float(len(shares))
    has = nucleus_inside(outlined, nuc, max_share=np.inf) if len(outlined) else np.zeros(0, bool)  # NeuN outlines the nucleus too
    s |= {"neun_outlined": float(len(outlined)), "neun_no_nucleus": float((~has).sum())}
    field = depth_field(white, counted if exclude or white is None else counted & ~white, grid_um)
    nxy = neurons[["x_core", "y_core"]].to_numpy(float) if len(neurons) else np.zeros((0, 2))
    s |= column_stats("NeuN", nxy, field, ps, core_id)
    big = neurons.shown_um2.to_numpy(float) >= LARGE_NEURON_UM2 if len(neurons) else np.zeros(0, bool)
    s |= depth_stats("NeuN", field, {"neuron": (nxy, np.ones(len(nxy))), "large": (nxy[big], np.ones(int(big.sum())))}, grid_um)
    if len(nuc):
        rng = _seed(core_id, "model sample")
        pick = rng.permutation(len(nuc))[:MODEL_SAMPLE]
        sample = pd.DataFrame(shape_matrix(nuc.iloc[pick]), columns=list(SHAPE))
        sample["neuron"] = is_neuron[pick]
        out["sample"] = sample
    # for the registered grid: the neurons, their sizes and the other nuclei in the NeuN frame
    out["grid"] = {"counted": counted, "white": white, "depth": None if field is None else field["depth"],
                   "neurons": nxy, "other": xy[~is_neuron] if len(xy) else np.zeros((0, 2))}
    return out


def plaque_core(project, core_id: str, exclude: bool, model, scores: pd.DataFrame | None) -> tuple[dict, dict | None]:
    """Pass 2, the 6E10 section: rings of neuron-like nuclei, the microcolumns of its neuron-like nuclei, the plaques'
    depth, size marks, containment and compactness against diffuse amyloid near by."""
    ps, grid_um = float(project.pixel_size_um), _grid_um(project)
    counted, white = tissue(project, core_id, "6E10", exclude)
    if counted is None:
        return {}, None
    s: dict = {}
    plaques = accepted(project, "6E10", core_id)
    if len(plaques):
        plaques = plaques[in_mask(counted, plaques[["x_core", "y_core"]].to_numpy(float))].reset_index(drop=True)
    outlined = plaques[plaques.ring.notna()] if len(plaques) else plaques
    diameter = pd.to_numeric(outlined.equivalent_diameter_um, errors="coerce").to_numpy(float) if len(outlined) else np.zeros(0)
    big = diameter >= MIN_PLAQUE_UM
    subtype = outlined.subtype.astype(str).to_numpy() if len(outlined) else np.zeros(0, str)
    groups = {"compact": outlined[big & (subtype == "compact")], "diffuse": outlined[big & (subtype == "diffuse")]}
    others = outlined[~(big & np.isin(subtype, ["compact", "diffuse"]))]
    nuc = nuclei(project, "6E10", core_id)
    nuc = nuc[in_mask(counted, nuc[["x_core", "y_core"]].to_numpy(float))].reset_index(drop=True) if len(nuc) else nuc
    p = neuron_like(model, nuc)
    s |= {"plaques_ge20um": float(sum(len(g) for g in groups.values())), "plaques_compact_ge20um": float(len(groups["compact"])),
          "plaques_diffuse_ge20um": float(len(groups["diffuse"])), "nl_6E10_cells": float(len(nuc)), "nl_6E10_sum": float(np.nansum(p))}
    if len(nuc) and np.isfinite(p).all():
        s |= ring_stats("pl", counted.shape, counted, groups, others, _cell_yx(nuc, counted.shape), p, grid_um, PLAQUE_RINGS, core_id)
    field = depth_field(white, counted & ~white if white is not None else counted, grid_um)
    like = nuc[p >= 0.5][["x_core", "y_core"]].to_numpy(float) if len(nuc) and np.isfinite(p).all() else np.zeros((0, 2))
    s |= column_stats("6E10", like, field, ps, core_id)
    pxy = plaques[["x_core", "y_core"]].to_numpy(float) if len(plaques) else np.zeros((0, 2))
    ptype = plaques.subtype.astype(str).to_numpy() if len(plaques) else np.zeros(0, str)
    s |= depth_stats("6E10", field, {"plaque": (pxy, np.ones(len(pxy))), "compact": (pxy[ptype == "compact"], np.ones(int((ptype == "compact").sum()))),
                                     "diffuse": (pxy[ptype == "diffuse"], np.ones(int((ptype == "diffuse").sum())))}, grid_um)
    if len(plaques):
        marks = pd.to_numeric(plaques.equivalent_diameter_um, errors="coerce").to_numpy(float)
        ok = np.isfinite(marks)
        s |= mark_stats(pxy[ok] * ps, marks[ok])
    diffuse = outlined[subtype == "diffuse"] if len(outlined) else outlined
    compact = plaques[ptype == "compact"] if len(plaques) else plaques
    s |= {"ct_compact": float(len(compact)), "ct_diffuse": float(len(diffuse))}
    if len(compact) and len(diffuse):
        near = _distance(counted.shape, _contours(diffuse)) * grid_um <= CONTAIN_UM
        cy, cx, ok = cells_of(compact[["x_core", "y_core"]].to_numpy(float), counted.shape)
        s |= {"ct_inside": float((ok & near[cy, cx]).sum()), "ct_expected": float(len(compact) * (near & counted).sum() / max(counted.sum(), 1))}
    if scores is not None and len(plaques):
        score = plaques.candidate_id.map(scores).to_numpy(float)
        typed = np.isfinite(score) & (np.nan_to_num(pd.to_numeric(plaques.equivalent_diameter_um, errors="coerce").to_numpy(float)) >= 15)
        if typed.sum() >= 3:
            area = plaques.shown_um2.to_numpy(float)
            is_diffuse = ptype == "diffuse"
            tree = cKDTree(pxy * ps)
            local = np.zeros(len(plaques))
            for i in np.flatnonzero(typed):
                near = [j for j in tree.query_ball_point(pxy[i] * ps, NEIGHBOURHOOD_UM) if j != i and is_diffuse[j]]
                local[i] = area[near].sum() if near else 0.0
            r = _spearman(score[typed], local[typed])
            s |= {"cs_plaques": float(typed.sum())}
            if np.isfinite(r):
                s |= {"cs_rho__sum": r * typed.sum(), "cs_rho__w": float(typed.sum())}
    layers = None
    if len(plaques):
        area = plaques.shown_um2.to_numpy(float)
        layers = {"plaque_um2": (pxy, area), "compact_um2": (pxy[ptype == "compact"], area[ptype == "compact"]),
                  "diffuse_um2": (pxy[ptype == "diffuse"], area[ptype == "diffuse"])}
    return s, {"counted": counted, "layers": layers or {"plaque_um2": (np.zeros((0, 2)), np.zeros(0)), "compact_um2": (np.zeros((0, 2)), np.zeros(0)),
                                                        "diffuse_um2": (np.zeros((0, 2)), np.zeros(0))}}


def tau_core(project, core_id: str, exclude: bool, model) -> tuple[dict, dict | None]:
    """Pass 2, the AT8 section: neuron-like nuclei around mature tangles and pretangles, nuclei in somata, satellites,
    the maturity mixing, the column alignment, the depth of tau, and what the registered grid needs."""
    ps, grid_um = float(project.pixel_size_um), _grid_um(project)
    counted, white = tissue(project, core_id, "AT8", exclude)
    if counted is None:
        return {}, None
    s: dict = {}
    bodies = accepted(project, "AT8", core_id)
    accepted_ids = set(bodies.candidate_id) if len(bodies) else set()
    if len(bodies):
        bodies = bodies[in_mask(counted, bodies[["x_core", "y_core"]].to_numpy(float))].reset_index(drop=True)
    outlined = bodies[bodies.ring.notna()].reset_index(drop=True) if len(bodies) else bodies
    subtype = outlined.subtype.astype(str).to_numpy() if len(outlined) else np.zeros(0, str)
    nuc = nuclei(project, "AT8", core_id)
    nuc = nuc[in_mask(counted, nuc[["x_core", "y_core"]].to_numpy(float))].reset_index(drop=True) if len(nuc) else nuc
    p = neuron_like(model, nuc)
    groups = {"mature": outlined[subtype == MATURE], "pretangle": outlined[subtype == PRETANGLE]}
    s |= {"tb_mature_bodies": float(len(groups["mature"])), "tb_pretangle_bodies": float(len(groups["pretangle"])),
          "nl_AT8_cells": float(len(nuc)), "nl_AT8_sum": float(np.nansum(p))}
    if len(nuc) and np.isfinite(p).all():
        s |= ring_stats("tb", counted.shape, counted, groups, outlined[~np.isin(subtype, [MATURE, PRETANGLE])], _cell_yx(nuc, counted.shape), p,
                        grid_um, BODY_RINGS, core_id)
    has = nucleus_inside(outlined, nuc) if len(outlined) else np.zeros(0, bool)
    s |= {"tau_outlined": float(len(outlined)), "tau_mature_outlined": float((subtype == MATURE).sum()),
          "tau_pretangle_outlined": float((subtype == PRETANGLE).sum()), "tau_mature_no_nucleus": float(((subtype == MATURE) & ~has).sum()),
          "tau_pretangle_no_nucleus": float(((subtype == PRETANGLE) & ~has).sum())}
    if len(outlined) and len(nuc):
        inside = cells.in_outlines(nuc, [[r] for r in outlined.ring])
        sat, band_um2 = satellite_counts(outlined, nuc, ~inside, ps)
        area_mm2 = float(counted.sum() * grid_um ** 2 / 1e6)
        s |= {"tau_satellites": float(sat.sum()), "tau_sat_expected": float((~inside).sum() / (area_mm2 * 1e6) * band_um2) if area_mm2 else 0.0}
    bxy = bodies[["x_core", "y_core"]].to_numpy(float) if len(bodies) else np.zeros((0, 2))
    btype = bodies.subtype.astype(str).to_numpy() if len(bodies) else np.zeros(0, str)
    typed = np.isin(btype, [MATURE, PRETANGLE])
    s |= mix_stats(bxy[typed] * ps, btype[typed] == MATURE)
    field = depth_field(white, counted & ~white if white is not None else counted, grid_um)
    like = nuc[p >= 0.5][["x_core", "y_core"]].to_numpy(float) if len(nuc) and np.isfinite(p).all() else np.zeros((0, 2))
    s |= column_stats("AT8", like, field, ps, core_id)
    s["align_bodies"] = 0.0
    if field is not None and field["usable"].any() and len(bxy):
        axis, ok = _axis_at(field, bxy)
        aligned, pairs = alignment_counts(bxy[ok] * ps, axis[ok])
        rng = _seed(core_id, "alignment")
        na = npairs = 0.0
        for _ in range(CSR_NULLS):
            points, axes = _random_points(field, int(ok.sum()), rng)
            a, b = alignment_counts(points * ps, axes)
            na, npairs = na + a, npairs + b
        s |= {"align_bodies": float(ok.sum()), "align_pairs": pairs, "align_o": aligned, "align_pairs_o": pairs,
              "align_n": na / CSR_NULLS, "align_pairs_n": npairs / CSR_NULLS}
    threads = store.threads(project, core_id)
    txy, tw = _thread_points(threads, accepted_ids, ps)
    keep = in_mask(counted, txy) if len(txy) else np.zeros(0, bool)
    txy, tw = txy[keep], tw[keep]
    s |= depth_stats("AT8", field, {"tau": (bxy, np.ones(len(bxy))), "mature": (bxy[btype == MATURE], np.ones(int((btype == MATURE).sum()))),
                                    "pretangle": (bxy[btype == PRETANGLE], np.ones(int((btype == PRETANGLE).sum()))), "thread": (txy, tw)}, grid_um)
    return s, {"counted": counted, "layers": {"thread_mm": (txy, tw), "tau": (bxy, np.ones(len(bxy)))}}


def registered_bins(project, core_id: str, neun: dict | None, plaque: dict | None, tau: dict | None) -> dict[float, pd.DataFrame] | None:
    """The bins of each size in the NeuN frame (`bin_table`) with every section's layers, or None without the three
    sections or a registration of both pairs."""
    if not neun or plaque is None or tau is None:
        return None
    matrices = {stain: to_stain(project, core_id, stain) for stain in ("6E10", "AT8")}
    if any(m is None for m in matrices.values()):
        return None
    ps, grid_um = float(project.pixel_size_um), _grid_um(project)
    shape = neun["counted"].shape
    joint = neun["counted"] & onto_neun(plaque["counted"], matrices["6E10"], shape) & onto_neun(tau["counted"], matrices["AT8"], shape)
    layers = {"neurons": (neun["neurons"], np.ones(len(neun["neurons"]))), "other": (neun["other"], np.ones(len(neun["other"])))}
    for name, (xy, w) in plaque["layers"].items():
        layers[name] = (to_neun_points(xy, matrices["6E10"]), w)
    for name, (xy, w) in tau["layers"].items():
        layers[name] = (to_neun_points(xy, matrices["AT8"]), w)
    return {size: bin_table(joint, grid_um, ps, size, layers, neun["depth"]) for size in BIN_SIZES}


def core_pass2(project, core_id: str, exclude: bool, model, scores, neun_points: dict | None, skip: set) -> dict:
    """Pass 2 of one core: its 6E10 and AT8 sections and, with all three registered, the grid features."""
    plaque_stats, plaque_grid = ({}, None)
    if _has(project, "6E10", core_id) and (core_id, "6E10") not in skip:
        plaque_stats, plaque_grid = plaque_core(project, core_id, exclude, model, scores)
    tau_stats, tau_grid = ({}, None)
    if _has(project, "AT8", core_id) and (core_id, "AT8") not in skip:
        tau_stats, tau_grid = tau_core(project, core_id, exclude, model)
    neun_grid = None
    if neun_points is not None:
        counted, white = tissue(project, core_id, "NeuN", exclude)
        if counted is not None:
            field = depth_field(white, counted if white is None else counted & ~white, _grid_um(project))
            neun_grid = {"counted": counted, "depth": None if field is None else field["depth"], **neun_points}
    bins = registered_bins(project, core_id, neun_grid, plaque_grid, tau_grid)
    out = {"stats": {**plaque_stats, **tau_stats, "registered": 0.0}, "bins": None}
    if bins is not None:
        out["stats"]["registered"] = 1.0
        out["stats"] |= coupling_stats(bins[BIN_SIZES[0]], core_id)
        out["bins"] = bins
    return out


def _has(project, stain: str, core_id: str) -> bool:
    from stainid.v2 import manifest  # noqa: PLC0415

    return core_id in manifest.read_all(project, "calls", stain)


# ---------------------------------------------------------------- columns from counts

def _check(s: dict, column: str) -> bool:
    return all(s.get(key, 0) >= minimum for key, minimum, _ in REQUIRE[column])


def columns(s: dict, gap_um: float) -> dict:
    """Every column (`COLUMNS`) of one core's or donor-region's summed counts; missing under its minimum."""
    out = {}
    for ring in ("contact", "ring10_30", "ring30_60"):
        out[f"plaque_{ring}_neuron_share_ratio"] = share_ratio(s, "pl", "all", ring)
    out["plaque_ring_neuron_share_slope"] = out["plaque_ring30_60_neuron_share_ratio"] - out["plaque_ring10_30_neuron_share_ratio"]
    out["plaque_ring10_30_neuronlike_density_ratio"] = share_ratio(s, "pl", "all", "ring10_30", "neuronlike")
    out["plaque_ring10_30_other_density_ratio"] = share_ratio(s, "pl", "all", "ring10_30", "other")
    for kind in ("compact", "diffuse"):
        out[f"plaque_{kind}_ring10_30_neuron_share_ratio"] = share_ratio(s, "pl", kind, "ring10_30")
    out["tangle_ring15_60_neuron_share_ratio"] = share_ratio(s, "tb", "mature", "ring15_60")
    out["tau_pretangle_ring15_60_neuron_share_ratio"] = share_ratio(s, "tb", "pretangle", "ring15_60")
    out["tangle_ring15_60_other_density_ratio"] = share_ratio(s, "tb", "mature", "ring15_60", "other")

    def frac(a, b):
        return float(s.get(a, 0) / s[b]) if s.get(b, 0) else np.nan

    out["tau_mature_no_nucleus_fraction"] = frac("tau_mature_no_nucleus", "tau_mature_outlined")
    out["tau_pretangle_no_nucleus_fraction"] = frac("tau_pretangle_no_nucleus", "tau_pretangle_outlined")
    out["neun_no_nucleus_fraction"] = frac("neun_no_nucleus", "neun_outlined")
    out["tangle_ghost_excess"] = out["tau_mature_no_nucleus_fraction"] - out["neun_no_nucleus_fraction"]
    out["tau_satellites_per_neuron"] = frac("tau_satellites", "tau_outlined")
    out["tau_satellite_enrichment"] = frac("tau_satellites", "tau_sat_expected")
    for section, stain in (("neun", "NeuN"), ("amyloid_section", "6E10"), ("tau_section", "AT8")):
        out[f"{section}_microcolumn_strength"] = _ratio_of_ratios(s, f"col_{stain}_radial", f"col_{stain}_lateral")
    out["amyloid_section_neuronlike_fraction"] = frac("nl_6E10_sum", "nl_6E10_cells")
    out["tau_section_neuronlike_fraction"] = frac("nl_AT8_sum", "nl_AT8_cells")
    for r in MIX_RADII:
        out[f"tangle_pretangle_mature_mix_{r:.0f}um"] = frac(f"mix_{r:.0f}_mixed", f"mix_{r:.0f}_expected")
    out["tau_neuron_column_alignment"] = _ratio_of_ratios(s, "align", "align_pairs")
    for stain, kinds in _DEPTH.items():
        for name, kind in kinds:
            out[f"{name}_depth_excess_um"] = _excess(s, stain, kind)
    out["tau_neuron_deep_upper_ratio"] = _deep_upper(s, "AT8", "tau")
    out["neun_deep_upper_ratio"] = _deep_upper(s, "NeuN", "neuron")
    out["neun_large_neuron_peak_depth_um"] = _peak_depth(s, "NeuN")
    out["tangle_plaque_depth_difference_um"] = out["tangle_depth_excess_um"] - out["plaque_compact_depth_excess_um"]
    out["plaque_thread_coupling_rho"] = _mean(s, "cp_rho")
    out["plaque_thread_coupling_excess"] = _mean(s, "cp_excess")
    out["plaque_compact_thread_coupling"] = _mean(s, "cp_compact")
    out["plaque_diffuse_thread_coupling"] = _mean(s, "cp_diffuse")
    out["plaque_tangle_coupling_excess"] = _mean(s, "cp_tangle")
    out["plaque_thread_coupling_depth_adjusted"] = _mean(s, "cp_depth")
    out["plaque_thread_correlation_length_um"] = _mean(s, "cp_length")
    for size in BIN_SIZES:
        out |= _niche_columns(s, f"niche{size:.0f}")
    out["neun_other_to_neuron_clq"] = _clq(s, "other")
    out["neun_neuron_to_other_clq"] = _clq(s, "neuron")
    out["neun_satellite_fraction"] = frac("sat_with", "sat_neurons")
    out["neun_large_satellite_fraction"] = frac("sat_large_with", "sat_large")
    out["neun_satellite_enrichment"] = frac("sat_observed", "sat_expected")
    hist = np.array([s.get(f"gaphist_{k}", 0) for k in range(_gap_bins())], float)
    out["neun_gap_area_fraction"] = float(hist[int(gap_um / GAP_BIN_UM):].sum() / hist.sum()) if hist.sum() and np.isfinite(gap_um) else np.nan
    out["neun_neuron_share_cv_100um"] = _mean(s, "share_cv")
    for r in MARK_RADII:
        out[f"plaque_size_mark_correlation_{r:.0f}um"] = _mark(s, r)
    out["plaque_compact_in_diffuse_fraction"] = frac("ct_inside", "ct_compact")
    out["plaque_compact_in_diffuse_enrichment"] = frac("ct_inside", "ct_expected")
    out["plaque_compactness_diffuse_neighbourhood_rho"] = _mean(s, "cs_rho")
    return {c: (out[c] if _check(s, c) and np.isfinite(out[c]) else np.nan) for c in COLUMNS}


def _sum(rows: list[dict]) -> dict:
    out: dict[str, float] = {}
    for row in rows:
        for k, v in row.items():
            if np.isfinite(v):
                out[k] = out.get(k, 0.0) + float(v)
    for key in ("registered",) + tuple(f"wm_{s}" for s in ("NeuN", "6E10", "AT8")):
        if key in out:  # a flag: any core
            out[key] = float(out[key] > 0)
    return out


# ---------------------------------------------------------------- the cohort

def _processes() -> int:
    """Worker processes: `STAINID_INTERACTION_PROCESSES`, else as stage 2 sizes its pool (at most 8)."""
    from stainid.v2.run import default_processes  # noqa: PLC0415

    given = int(os.environ.get("STAINID_INTERACTION_PROCESSES") or 0)
    return given or max(1, min(8, default_processes(0)))


_STATE: dict = {}


def _init(root: str, exclude: bool, model, scores, skip: set) -> None:
    from stainid.project import load_project  # noqa: PLC0415

    cv2.setNumThreads(1)
    _STATE.update(project=load_project(root), exclude=exclude, model=model, scores=scores, skip=skip)


def _pass1(core_id: str) -> tuple[str, dict]:
    project = _STATE["project"]
    if not _has(project, "NeuN", core_id) or (core_id, "NeuN") in _STATE["skip"]:
        return core_id, {"stats": {}, "sample": None, "grid": None}
    found = neun_core(project, core_id, _STATE["exclude"])
    grid = found.pop("grid")
    return core_id, {**found, "grid": None if grid is None else {k: grid[k] for k in ("neurons", "other")}}


def _pass2(task: tuple[str, dict | None]) -> tuple[str, dict]:
    core_id, points = task
    return core_id, core_pass2(_STATE["project"], core_id, _STATE["exclude"], _STATE["model"], _STATE["scores"], points, _STATE["skip"])


def _map(function, items: list, processes: int, setup: tuple) -> list:
    """`function` over `items`, in `processes` worker processes (each set up with `_init(*setup)`), else here."""
    if processes <= 1 or len(items) <= 1:
        _init(*setup)
        return [function(i) for i in items]
    import multiprocessing as mp  # noqa: PLC0415

    with mp.get_context("forkserver").Pool(processes, initializer=_init, initargs=setup) as pool:
        return list(pool.imap_unordered(function, items, chunksize=1))


def _ranks_within(bins: pd.DataFrame, column: str) -> np.ndarray:
    return bins.groupby("tma")[column].rank(pct=True, method="average").to_numpy(float)


def niche_rows(bins: dict[str, dict[float, pd.DataFrame]], tma_of: dict[str, str]) -> dict[str, dict]:
    """The niche counts of every core from the cohort's bins (percentile ranks within each TMA)."""
    out: dict[str, dict] = {c: {} for c in bins}
    for size in BIN_SIZES:
        frames = [b[size].assign(core_id=c, tma=tma_of.get(c, "")) for c, b in bins.items() if b is not None and len(b[size])]
        if not frames:
            continue
        table = pd.concat(frames, ignore_index=True)
        area = table.tissue_mm2.to_numpy(float)
        table["amyloid"] = table.plaque_um2 / (area * 1e6)
        table["thread_density"], table["tau_density"] = table.thread_mm / area, table.tau / area
        table["neuron_density"] = table.neurons / area
        table["a"] = _ranks_within(table, "amyloid")
        both = pd.Series((_ranks_within(table, "thread_density") + _ranks_within(table, "tau_density")) / 2, index=table.index)
        table["t"] = both.groupby(table.tma).rank(pct=True, method="average").to_numpy(float)
        table["n"] = _ranks_within(table, "neuron_density")
        labels = niche_labels(table)
        for core_id, index in table.groupby("core_id").indices.items():
            out[core_id] |= niche_stats(f"niche{size:.0f}", table.iloc[index].reset_index(drop=True), labels[index])
    return out


def compute(project, folder: Path, exclude: bool, skip: set[tuple[str, str]] | None = None) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Every core's and donor-region's interaction columns (and the counts behind them), from the core images with
    calls; `skip`: (core_id, stain) images not measured (too little grey matter)."""
    from stainid.v2.measures import REGION_KEYS, core_images  # noqa: PLC0415

    skip = set(skip or ())
    images = core_images(project)
    meta = {}
    for stain in images:
        for core_id in images[stain]:
            if core_id not in meta:
                tiles = store.read_table(store.path(project, stain, core_id, "tiles"), list(REGION_KEYS))
                if len(tiles):
                    meta[core_id] = {k: str(tiles.iloc[0][k]) for k in REGION_KEYS if k in tiles}
    cores = sorted(meta)
    processes = _processes()
    first = dict(_map(_pass1, cores, processes, (str(project.root), exclude, None, None, skip)))
    samples = [first[c]["sample"].assign(tma=meta[c].get("tma", "")) for c in cores if first[c].get("sample") is not None]
    model, check = fit_model(pd.concat(samples, ignore_index=True) if samples else pd.DataFrame(columns=[*SHAPE, "neuron", "tma"]))
    check.to_csv(folder / "interactions_nucleus_model.csv", index=False)
    scores = None
    scores_path = folder / "type_scores_6E10.csv"
    if scores_path.exists():
        table = pd.read_csv(scores_path, usecols=lambda c: c in ("candidate_id", "plaque_compactness_score"))
        if "plaque_compactness_score" in table:
            scores = table.set_index("candidate_id").plaque_compactness_score
    tasks = [(c, first[c].get("grid")) for c in cores]
    second = dict(_map(_pass2, tasks, processes, (str(project.root), exclude, model, scores, skip)))
    niches = niche_rows({c: second[c]["bins"] for c in cores}, {c: meta[c].get("tma", "") for c in cores})
    per_core = {c: {**first[c]["stats"], **second[c]["stats"], **niches.get(c, {})} for c in cores}
    gap_um = gap_threshold([per_core[c] for c in cores if meta[c].get("disease_group") == "CT"])
    core_rows = [{**meta[c], **columns(per_core[c], gap_um), **{f"ix_{k}": v for k, v in _counts(per_core[c]).items()}} for c in cores]
    by_region: dict[str, list[str]] = {}
    for c in cores:
        by_region.setdefault(meta[c].get("sample_region_id", ""), []).append(c)
    region_rows = []
    for region, members in sorted(by_region.items()):
        summed = _sum([per_core[c] for c in members])
        ids = {k: v for k, v in meta[members[0]].items() if k != "core_id"}
        region_rows.append({**ids, **columns(summed, gap_um), **{f"ix_{k}": v for k, v in _counts(summed).items()}})
    auc = f"leave-one-TMA-out AUC {check.auc.median():.3f} (median of {len(check)} TMAs)" if len(check) else "no leave-one-TMA-out check"
    lines = [f"interactions: {len(cores)} cores ({processes} processes); neuron-like nucleus model {auc}",
             f"interactions: control neuron-gap distance ({GAP_QUANTILE:.0%} of the control cores' interior tissue) {gap_um:.0f} µm"]
    unlined = [c for c in cores if not second[c]["stats"].get("registered") and all(_has(project, s, c) for s in ("NeuN", "6E10", "AT8"))]
    if unlined:
        lines.append(f"interactions: {len(unlined)} cores with all three sections not lined up (no registration, or it failed; "
                     f"`stainid v2 register` registers the pairs the table has no row for), e.g. {', '.join(unlined[:5])}")
    return pd.DataFrame(core_rows), pd.DataFrame(region_rows), lines


def _counts(s: dict) -> dict:
    """The counts the minimums test (`REQUIRE`), for `results_counts_<level>.csv`."""
    keys = {key for need in REQUIRE.values() for key, _, _ in need}
    return {k: float(s.get(k, 0.0)) for k in sorted(keys)}


def attach(project, folder: Path, exclude: bool, skip: set[tuple[str, str]] | None = None) -> list[str]:
    """Stage 5's hook: `interactions_<level>.csv` (every column and its counts), the columns into `results_<level>.csv`
    and the counts into `results_counts_<level>.csv` (for `results_missing_<level>.csv`)."""
    core, region, lines = compute(project, folder, exclude, skip)
    return lines + attach_tables(folder, core, region)


def attach_tables(folder: Path, core: pd.DataFrame, region: pd.DataFrame) -> list[str]:
    """Write the interaction tables and add their columns to the results and counts tables. The tables are read and
    written back as text, so every cell already there stays exactly as it was."""
    lines = []
    for level, key, table in (("core", "core_id", core), ("donor_region", "sample_region_id", region)):
        table.to_csv(folder / f"interactions_{level}.csv", index=False)
        path = folder / f"results_{level}.csv"
        if not path.exists() or table.empty:
            continue
        indexed = table.assign(**{key: table[key].astype(str)}).set_index(key)
        results = pd.read_csv(path, dtype=str, keep_default_na=False)
        ids = results[key].astype(str)
        added = pd.DataFrame({c: ids.map(indexed[c]).astype(float).to_numpy() for c in COLUMNS}, index=results.index)
        pd.concat([results.drop(columns=[c for c in COLUMNS if c in results]), added], axis=1).to_csv(path, index=False)
        counts_path = folder / f"results_counts_{level}.csv"
        counts = pd.read_csv(counts_path, dtype=str, keep_default_na=False) if counts_path.exists() else pd.DataFrame()
        if key in counts:
            extra = [c for c in table.columns if c.startswith("ix_")]
            ids = counts[key].astype(str)
            added = pd.DataFrame({c: ids.map(indexed[c]).astype(float).fillna(0.0).to_numpy() for c in extra}, index=counts.index)
            pd.concat([counts.drop(columns=[c for c in extra if c in counts]), added], axis=1).to_csv(counts_path, index=False)
        lines.append(f"results_{level}: {len(COLUMNS)} spatial-interaction columns")
    return lines
