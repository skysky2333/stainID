"""Counterstain cells: stage 2's per-nucleus table (`cells/<stain>/<core_id>.parquet`) and which nuclei every nucleus count takes.

A cell is a Cellpose nucleus of Find nuclei (`Field.nuclei`: enough hematoxylin, vacuoles dropped) of 10-200 µm² whose
centre (the mean of its pixels, as `stainid.nuclei.nucleus_centroids` takes it) lies in the measured tissue. On 6E10 and
AT8 a nucleus must also hold at least as much hematoxylin as DAB: Cellpose outlines dense DAB objects too (plaque cores,
tangles, thread swellings), about 7 % of its "nuclei" there, and they are no cells (`counted`). NeuN stains neuronal nuclei
brown, so on NeuN only the hematoxylin and size rules apply. Every nucleus count of stage 2 counts the `counted` cells:
the tile's `nucleus_count` and the 6E10 nuclei inside and around each plaque. The AT8 model's own inputs (the nuclei a
candidate overlaps) and the Cellpose rings AT8 proposes from are unchanged.

The table keeps every cell of the tile's own area (counted or not) with its shape and stain: position (core px), area,
eccentricity, solidity, major and minor axis, mean hematoxylin and DAB (optical density). Stage 5 reads the NeuN one for the
neuron share of all cells on the NeuN section and the density of the other cells (`neun_cells`).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage

MIN_UM2, MAX_UM2 = 10.0, 200.0  # the size of a cell's nucleus (`nucleus_centroids`' rule)
DENSE_DAB_STAINS = ("6E10", "AT8")  # where a nucleus with more DAB than hematoxylin is a stained object, not a cell
COLUMNS = ["tile_id", "x_core", "y_core", "area_um2", "eccentricity", "solidity", "major_um", "minor_um", "hematoxylin_mean", "dab_mean",
           "counted"]


def _all(field) -> pd.DataFrame:
    """Every nucleus of the field of a cell's size: label, centre (field px, integer as `nucleus_centroids`), area (px) and
    mean hematoxylin and DAB; made once per field."""
    if "cells_all" in field.memo:
        return field.memo["cells_all"]
    labels = field.nuclei
    um2 = field.pixel_size_um ** 2
    found = []
    for label, bounds in enumerate(ndimage.find_objects(labels), start=1):  # as `nucleus_centroids`: one look per nucleus
        if bounds is None:
            continue
        local = labels[bounds] == label
        area = int(local.sum())
        if not MIN_UM2 <= area * um2 <= MAX_UM2:
            continue
        ys, xs = np.nonzero(local)
        found.append((label, int(bounds[1].start + xs.mean()), int(bounds[0].start + ys.mean()), area,
                      float(field.hematoxylin[bounds][local].mean()), float(field.dab[bounds][local].mean())))
    frame = pd.DataFrame(found, columns=["label", "x", "y", "area_px", "hematoxylin_mean", "dab_mean"])
    field.memo["cells_all"] = frame
    return frame


def _counted(frame: pd.DataFrame, stain: str) -> np.ndarray:
    if stain not in DENSE_DAB_STAINS:
        return np.ones(len(frame), bool)
    return (frame.hematoxylin_mean >= frame.dab_mean).to_numpy(bool)


def centroids(field, valid: np.ndarray, stain: str) -> np.ndarray:
    """The counted cells' centres (field px, N x 2 integers) whose centre lies in `valid`: what a nucleus count counts."""
    frame = _all(field)
    if frame.empty:
        return np.zeros((0, 2), int)
    keep = valid[frame.y.to_numpy(), frame.x.to_numpy()] & _counted(frame, stain)
    return frame.loc[keep, ["x", "y"]].to_numpy(int).reshape(-1, 2)


def _shape(local: np.ndarray) -> tuple[float, float, float, float]:
    """Eccentricity, solidity and major / minor axis (px) of one nucleus mask, as skimage's `regionprops` defines them
    (the ellipse of the same second moments; solidity over the filled convex hull of the pixel centres, within about
    0.02 of skimage's)."""
    import cv2  # noqa: PLC0415

    ys, xs = np.nonzero(local)
    if len(xs) < 2:
        return 0.0, 1.0, 0.0, 0.0
    cov = np.cov(np.vstack([xs, ys]).astype(np.float64), bias=True)
    low, high = np.linalg.eigvalsh(cov)
    low, high = max(low, 0.0), max(high, 0.0)
    eccentricity = float(np.sqrt(1.0 - low / high)) if high > 0 else 0.0
    hull = cv2.convexHull(np.c_[xs, ys].astype(np.int32))
    filled = np.zeros(local.shape, np.uint8)
    cv2.fillConvexPoly(filled, hull, 1)
    return eccentricity, float(len(xs) / max(1, int(filled.sum()))), float(4 * np.sqrt(high)), float(4 * np.sqrt(low))


def tile_cells(field, valid: np.ndarray, stain: str) -> pd.DataFrame:
    """The cells of the tile's own area whose centre lies in `valid` (counted or not), with shape and stain, in core px."""
    frame = _all(field)
    if not frame.empty:
        xs, ys = frame.x.to_numpy(), frame.y.to_numpy()
        inner = field.inner
        own = (xs >= inner[1].start) & (xs < inner[1].stop) & (ys >= inner[0].start) & (ys < inner[0].stop)
        own &= valid[ys, xs]
        frame = frame[own]
    if frame.empty:
        return pd.DataFrame(columns=COLUMNS)
    boxes = ndimage.find_objects(field.nuclei)
    shape = np.array([_shape(field.nuclei[boxes[label - 1]] == label) for label in frame.label], np.float64).reshape(-1, 4)
    ps = field.pixel_size_um
    ox, oy = field.origin
    return pd.DataFrame({
        "tile_id": field.tile["tile_id"], "x_core": frame.x.to_numpy() + ox, "y_core": frame.y.to_numpy() + oy,
        "area_um2": (frame.area_px.to_numpy() * ps * ps).astype(np.float32), "eccentricity": shape[:, 0].astype(np.float32),
        "solidity": shape[:, 1].astype(np.float32), "major_um": (shape[:, 2] * ps).astype(np.float32),
        "minor_um": (shape[:, 3] * ps).astype(np.float32), "hematoxylin_mean": frame.hematoxylin_mean.to_numpy(np.float32),
        "dab_mean": frame.dab_mean.to_numpy(np.float32), "counted": _counted(frame, stain)}, columns=COLUMNS)


# ---------------------------------------------------------------- stage 5: the NeuN section's cells

def read(project, stain: str, core_id: str) -> pd.DataFrame | None:
    """A core image's cells (stage 2); None when stage 2 ran before the table existed (the cells are not known)."""
    from stainid import candidates as store  # noqa: PLC0415

    target = store.stored(project, stain, core_id, "cells")
    if target is None:
        return None
    frame = store.read_table(target)
    return frame if len(frame.columns) else pd.DataFrame(columns=COLUMNS)


def in_outlines(cells: pd.DataFrame, rings: list[list[np.ndarray]]) -> np.ndarray:
    """Whether each cell's centre lies inside any of these outlines (each a list of rings, core px; the first ring is
    the outline, holes are not cut out)."""
    import cv2  # noqa: PLC0415

    inside = np.zeros(len(cells), bool)
    if not len(cells) or not rings:
        return inside
    xs, ys = cells.x_core.to_numpy(float), cells.y_core.to_numpy(float)
    order = np.argsort(xs)
    sorted_x = xs[order]
    for outline in rings:
        if not outline or len(outline[0]) < 3:
            continue
        ring = np.asarray(outline[0], np.float32).reshape(-1, 2)
        lo, hi = ring.min(axis=0), ring.max(axis=0)
        start, stop = np.searchsorted(sorted_x, lo[0], "left"), np.searchsorted(sorted_x, hi[0], "right")
        if start >= stop:
            continue
        near = order[start:stop]
        near = near[(ys[near] >= lo[1]) & (ys[near] <= hi[1]) & ~inside[near]]
        contour = ring.reshape(-1, 1, 2)
        for i in near:
            if cv2.pointPolygonTest(contour, (float(xs[i]), float(ys[i])), False) >= 0:
                inside[i] = True
    return inside


def tile_counts(tiles: pd.DataFrame, cells: pd.DataFrame | None, neuronal: np.ndarray) -> pd.DataFrame:
    """Per tile: the counted cells (`cell_count`) and those in no accepted neuron (`nonneuronal_cell_count`); 0 for a tile
    without cells, NaN for a tile without nuclei (Find nuclei has not run on it) and for every tile when `cells` is None."""
    if cells is None:
        return tiles.assign(cell_count=np.nan, nonneuronal_cell_count=np.nan)
    counted = cells.counted.astype(bool).to_numpy() if len(cells) else np.zeros(0, bool)
    by_tile = pd.DataFrame({"tile_id": cells.tile_id.to_numpy() if len(cells) else [], "all": counted,
                            "other": counted & ~neuronal}).groupby("tile_id").sum() if len(cells) else pd.DataFrame(columns=["all", "other"])
    out = tiles.copy()
    available = out.get("nuclei_available", pd.Series(True, index=out.index)).astype(str).str.lower().isin(("true", "1", "1.0"))
    out["cell_count"] = out.tile_id.map(by_tile["all"]).fillna(0).astype(float).where(available)
    out["nonneuronal_cell_count"] = out.tile_id.map(by_tile["other"]).fillna(0).astype(float).where(available)
    return out
