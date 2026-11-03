"""How tau pathology is organised, from stored results (stage 5; no image is read): tangle spacing, thread ownership, and
the QC share of accepted objects touching an edge or an excluded area.

Tangle spacing, on the mature tangles whose centre lies in the core image's valid tissue (the counted tissue of the
analysed tiles, in the grey-matter tables without the white matter): `tangle_nn_median_um`, the median distance to the
nearest other mature tangle of the same core image, and `tangle_l200_excess_um`, Ripley's L at 200 µm minus its mean over
`NULLS` placements of as many points at random in the same tissue, so the core's shape and holes cancel. Above 0 the
tangles gather, below 0 they keep apart. Both need `MIN_TANGLES`. A donor-region pools its cores' distances and takes the
count-weighted mean of the L excess of the cores that reach the minimum. The random placements are seeded by the core id.

Thread ownership: a neuron's *process* is a connected stretch of free thread (thread outside accepted tau+ neurons) within
`WINDOW_UM` of the outline that comes within `TOUCH_UM` of it. Each thread piece of a process is attached; a piece reached
by k neurons gives each 1/k of its length, so attached + unattached thread is exactly the thread length of the table
(`thread_ridge_length_mm`). A piece counts whole also when it runs on past the window. Means per neuron need
`MIN_NEURONS` neurons (of that subtype for the `_mature` and `_pretangle` versions), the attached share `MIN_THREAD_MM` of
thread; the two densities are 0 when nothing was found.

`EDGE_QC`: the share of accepted objects whose outline touches the 5 µm band along a tissue edge or hole or an excluded
area, and of those mostly in the band (at least 0.5, the cut of `rule:edge`). A quality number only: docs/pipeline_v2.md
has what the band does and does not mean.
"""
from __future__ import annotations

import zlib

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.spatial import cKDTree

from stainid import candidates as store

SCALE = 8  # core-image px per mask px (stainid.exclusions.SCALE)
RADIUS_UM = 200.0
NULLS = 39
MIN_TANGLES = 10
TOUCH_UM = 2.0
WINDOW_UM = 50.0
MIN_NEURONS = 5
MIN_THREAD_MM = 0.5
MATURE, PRETANGLE = "mature tangle", "pretangle"
OUTLINE_COLUMNS = ["candidate_id", "decision", "subtype", "outline", "outline_default", "sam_outline", "sam_status", "sam_default", "sam_masks"]
EDGE_QC = {"NeuN": ("neun_edge_touch_fraction", "neun_edge_mostly_fraction", 20),
           "6E10": ("plaque_edge_touch_fraction", "plaque_edge_mostly_fraction", 10),
           "AT8": ("tau_neuron_edge_touch_fraction", "tau_neuron_edge_mostly_fraction", MIN_NEURONS)}


def valid_tissue(project, core_id: str, stain: str, white: np.ndarray | None = None) -> np.ndarray | None:
    """The counted tissue of the analysed tiles on the masks' grid, without `white`; None without a stage-2 mask."""
    from stainid import exclusions  # noqa: PLC0415

    layer = exclusions.automatic(project, core_id, stain)
    if layer is None:
        return None
    valid = layer.codes == exclusions.COUNTED
    tiled = (layer.bits & exclusions.TILE) > 0
    if tiled.any():
        valid = valid & tiled
    if white is not None:
        h, w = min(white.shape[0], valid.shape[0]), min(white.shape[1], valid.shape[1])
        valid = valid.copy()
        valid[:h, :w] &= ~white[:h, :w]
    return valid


def _in_mask(mask: np.ndarray, xy: np.ndarray) -> np.ndarray:
    cells = np.floor(xy / SCALE).astype(int)
    ok = (cells[:, 0] >= 0) & (cells[:, 1] >= 0) & (cells[:, 0] < mask.shape[1]) & (cells[:, 1] < mask.shape[0])
    out = np.zeros(len(xy), bool)
    out[ok] = mask[cells[ok, 1], cells[ok, 0]]
    return out


def _pairs(xy: np.ndarray, radius_px: float) -> float:
    tree = cKDTree(xy)
    return float(tree.count_neighbors(tree, radius_px) - len(xy))


def l_excess(xy: np.ndarray, mask: np.ndarray, pixel_um: float, seed: str, radius_um: float = RADIUS_UM, nulls: int = NULLS) -> float:
    """L(radius) of the points (core px) minus its mean over `nulls` random placements in `mask`, in µm."""
    n = len(xy)
    ys, xs = np.nonzero(mask)
    if n < 2 or not len(xs):
        return float("nan")
    area_um2 = len(xs) * (SCALE * pixel_um) ** 2
    scale = area_um2 / (n * (n - 1)) / np.pi
    rng = np.random.default_rng(zlib.crc32(str(seed).encode()))
    radius_px = radius_um / pixel_um
    random_l = []
    for _ in range(nulls):
        pick = rng.integers(0, len(xs), n)
        points = np.c_[(xs[pick] + rng.random(n)) * SCALE, (ys[pick] + rng.random(n)) * SCALE]
        random_l.append(np.sqrt(_pairs(points, radius_px) * scale))
    return float(np.sqrt(_pairs(xy, radius_px) * scale) - np.mean(random_l))


def spacing(found: pd.DataFrame, mask: np.ndarray | None, pixel_um: float, core_id: str) -> dict:
    """One core image's mature tangles in `mask`: how many, their nearest-neighbour distances and the L excess."""
    empty = {"tangles": 0, "nn_um": np.zeros(0), "l_excess": float("nan")}
    if mask is None or found.empty or "decision" not in found:
        return empty
    mature = found[(found.decision == "accepted") & (found.subtype == MATURE)]
    xy = mature[["x_core", "y_core"]].to_numpy(float)
    xy = xy[np.isfinite(xy).all(axis=1)]
    if len(xy):
        xy = xy[_in_mask(mask, xy)]
    if len(xy) < 2:
        return empty | {"tangles": len(xy)}
    distances = cKDTree(xy).query(xy, k=2)[0][:, 1] * pixel_um
    return {"tangles": len(xy), "nn_um": distances,
            "l_excess": l_excess(xy, mask, pixel_um, core_id) if len(xy) >= MIN_TANGLES else float("nan")}


def _points(cell) -> np.ndarray:
    return np.asarray(str(cell).split(), float).reshape(-1, 2) if isinstance(cell, str) and cell.strip() else np.zeros((0, 2))


def _kernel(radius_px: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius_px + 1, 2 * radius_px + 1))


def ownership(bodies: pd.DataFrame, threads: pd.DataFrame | None, pixel_um: float) -> tuple[list[dict], float]:
    """Per accepted tau+ neuron with an outline (`bodies`: `OUTLINE_COLUMNS` of the accepted ones) its processes and its
    attached thread (µm), and the core image's attached thread (mm). See the module's docstring."""
    columns = bodies.to_dict("records")
    rings = [[np.asarray(r, float).reshape(-1, 2) for r in store.outline_rings(row)] for row in columns]
    known = [i for i, r in enumerate(rings) if r]
    ids = set(bodies.candidate_id)
    free = threads[~threads.candidate_id.isin(ids)] if threads is not None and not threads.empty else None
    if not known or free is None or free.empty:
        return [{"candidate_id": columns[i]["candidate_id"], "subtype": columns[i]["subtype"], "processes": 0, "attached_thread_um": 0.0}
                for i in known], 0.0
    pieces = [_points(p) for p in free.points]
    lengths = free.n_px.to_numpy(float) * pixel_um
    piece_of = np.concatenate([np.full(len(p), i) for i, p in enumerate(pieces)]) if pieces else np.zeros(0, int)
    vertices = np.vstack(pieces) if pieces else np.zeros((0, 2))
    tree = cKDTree(vertices) if len(vertices) else None
    window, band = int(round(WINDOW_UM / pixel_um)), int(round(TOUCH_UM / pixel_um))
    kernel = _kernel(band)
    reached: list[set[int]] = []
    processes = []
    for i in known:
        corners = np.vstack(rings[i])
        x0, y0 = np.floor(corners.min(axis=0)).astype(int) - window
        x1, y1 = np.ceil(corners.max(axis=0)).astype(int) + window
        mine: set[int] = set()
        count = 0
        near = np.unique(piece_of[tree.query_ball_point([(x0 + x1) / 2, (y0 + y1) / 2], np.hypot(x1 - x0, y1 - y0) / 2)]) if tree is not None else []
        if len(near):
            canvas = np.zeros((y1 - y0 + 1, x1 - x0 + 1), np.int32)
            for index in near:
                line = np.rint(pieces[index] - [x0, y0]).astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(canvas, [line], False, int(index) + 1, 1)
            body = np.zeros(canvas.shape, np.uint8)
            cv2.fillPoly(body, [np.rint(r - [x0, y0]).astype(np.int32) for r in rings[i]], 1)
            touching = cv2.dilate(body, kernel) > 0
            labels, _ = ndimage.label(canvas > 0, np.ones((3, 3)))
            hit = np.unique(labels[touching & (labels > 0)])
            count = len(hit)
            if count:
                chosen = canvas[np.isin(labels, hit)]
                mine = {int(v) - 1 for v in np.unique(chosen[chosen > 0])}
        reached.append(mine)
        processes.append(count)
    owners = np.zeros(len(pieces), int)
    for mine in reached:
        for index in mine:
            owners[index] += 1
    rows = []
    for i, count, mine in zip(known, processes, reached):
        rows.append({"candidate_id": columns[i]["candidate_id"], "subtype": columns[i]["subtype"], "processes": count,
                     "attached_thread_um": float(sum(lengths[k] / owners[k] for k in mine))})
    return rows, float(lengths[owners > 0].sum() / 1000)


def edge_tally(found: pd.DataFrame) -> dict:
    """Accepted objects, those touching the edge band or an excluded area, and those mostly in the band."""
    if found.empty or "decision" not in found:
        return {"accepted": 0, "touching": 0, "mostly": 0}
    accepted = found[found.decision == "accepted"]
    band = pd.to_numeric(accepted.edge_band_fraction, errors="coerce").fillna(0).to_numpy(float)
    excluded = pd.to_numeric(accepted.excluded_fraction, errors="coerce").fillna(0).to_numpy(float)
    return {"accepted": len(accepted), "touching": int(((band > 0) | (excluded > 0)).sum()), "mostly": int((band >= 0.5).sum())}


def edge_columns(stain: str, tallies: list[dict]) -> dict:
    touch, mostly, minimum = EDGE_QC[stain]
    accepted = sum(t["accepted"] for t in tallies)
    if accepted < minimum:
        return {touch: float("nan"), mostly: float("nan")}
    return {touch: sum(t["touching"] for t in tallies) / accepted, mostly: sum(t["mostly"] for t in tallies) / accepted}


def _per_neuron(rows: list[dict], suffix: str = "") -> dict:
    enough = len(rows) >= MIN_NEURONS
    return {f"tau_processes_per_neuron{suffix}": float(np.mean([r["processes"] for r in rows])) if enough else float("nan"),
            f"tau_attached_thread_um_per_neuron{suffix}": float(np.mean([r["attached_thread_um"] for r in rows])) if enough else float("nan")}


def summarize(cores: list[dict], neurons: list[dict], thread_mm: float, area_mm2: float) -> dict:
    """The columns of one row of the results table: its core images' `spacing` + `attached_mm` and neurons' `ownership`,
    its thread length and the AT8 area measured."""
    distances = np.concatenate([c["nn_um"] for c in cores]) if cores else np.zeros(0)
    usable = [c for c in cores if c["tangles"] >= MIN_TANGLES and np.isfinite(c["l_excess"])]
    weight = sum(c["tangles"] for c in usable)
    attached = sum(c["attached_mm"] for c in cores)
    out = {"tangle_nn_median_um": float(np.median(distances)) if len(distances) >= MIN_TANGLES else float("nan"),
           "tangle_l200_excess_um": sum(c["tangles"] * c["l_excess"] for c in usable) / weight if weight else float("nan")}
    out |= _per_neuron(neurons)
    out |= _per_neuron([r for r in neurons if r["subtype"] == MATURE], "_mature")
    out |= _per_neuron([r for r in neurons if r["subtype"] == PRETANGLE], "_pretangle")
    out["tau_neurons_with_process_fraction"] = float(np.mean([r["processes"] > 0 for r in neurons])) if len(neurons) >= MIN_NEURONS else float("nan")
    out["thread_attached_density_mm_per_mm2"] = attached / area_mm2 if area_mm2 else float("nan")
    out["thread_unattached_density_mm_per_mm2"] = max(0.0, thread_mm - attached) / area_mm2 if area_mm2 else float("nan")
    out["thread_attached_share"] = attached / thread_mm if thread_mm >= MIN_THREAD_MM else float("nan")
    return out


def counts(cores: list[dict], neurons: list[dict]) -> dict:
    """What the minimums count, for `stainid.v2.missing`."""
    return {"tangles_with_neighbour": sum(len(c["nn_um"]) for c in cores), "tangles_largest_core": max([c["tangles"] for c in cores], default=0),
            "tangles": sum(c["tangles"] for c in cores), "owned_neurons": len(neurons),
            "owned_mature": sum(r["subtype"] == MATURE for r in neurons), "owned_pretangle": sum(r["subtype"] == PRETANGLE for r in neurons)}
