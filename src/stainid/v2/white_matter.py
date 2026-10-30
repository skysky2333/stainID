"""White matter, found automatically: the "Grey / white matter" algorithm, a step after detection (`stainid v2
white-matter`) that reads detection's stored outputs and never re-runs it.

What it looks at: the NeuN+ neurons (stage 4's accepted NeuN calls) on the NeuN section, and the cells (hematoxylin
nuclei, on every section; neurons are only the NeuN+ ones) of the NeuN image to confirm. Grey matter is where the
neurons are; white matter is tissue with almost none and more cells than the grey matter, and the cortex ends in it at
a sharp drop in neuron density (`docs/white_matter.md`).

Per NeuN core image (`detect`), on a grid of `CELL` mask px (64 core px, 17.5 µm), one rule set for every core:
1. the NeuN+ neuron density over the analysed tissue (stage 2's mask: tissue in a tile), smoothed over `sigma_um`;
2. the core's own grey-matter density, `reference`: the `reference_quantile` of that density over its tissue;
3. candidates: tissue far below the reference (seeds under `seed` x, opened with a disc of `open_um` so that a thin
   sparse band such as layer I is no seed, grown through tissue under `grow` x), and the low side of every sharp drop:
   tissue under `step` x the densest tissue within `near_um` (a drop to a fifth within 400 µm is a step, not a fade),
   reaching the drop's steepest part (a watershed of the density's gradient). The drop is relative to the grey matter
   next to it, so dense grey matter does not hide white matter beside it. A core whose densest tissue is under
   `min_reference_per_mm2` has no grey matter to drop from: it is a candidate as a whole;
4. each part checked: thick enough, a real step at its edge (`max_step`), deep in the tissue unless its step is clear
   and its cells are white matter's (white matter at a core's edge is common; layer I, the pia and a frayed rim are
   not), and its cells those of white matter: few neuron-sized nuclei (stage 1's nuclei of the NeuN section) against
   the grey matter, or more cells (hematoxylin nuclei of the NeuN section and of the 6E10 / AT8 sections carried through
   the registration) or less NeuN stain on an unchanged background. Fewer neurons on a paler background with as many
   neuron-sized nuclei is a pale stain, not white matter. A core with no grey matter of its own is compared with the
   cohort's grey matter (`cohort_large_share`) and must show no grey-matter pathology on its other sections; a pale core
   that is not white stays "undetermined" (its NeuN stain too pale to read). Holes under `fill_holes_mm2` are filled;
5. the area on the mask grid (1/8 of the core image), its edge smoothed, inside the analysed tissue.

Which cue carries it: the neurons and the sharp drop in their density; the cells and their sizes confirm. Cells
alone separate white from grey matter within a core with a median AUC of 0.9 but fail on a quarter of the cores.

The NeuN area is carried to the core's 6E10 and AT8 images through the registration (`registration.cores`, the same
matrices the "Grey / white matter" round carries people's areas with), inside each image's own analysed tissue.

Outputs, under `<outputs.v2>/white_matter/`: `<stain>/<core_id>.png` (1/8 of the core image, the masks' grid: 1 white
matter, 0 not), `manifests/<core_id>.json` (what each core's areas were made from: a core is redone only when its
NeuN calls, a stage-2 mask, the registration or these settings change) and `white_matter.csv` (per core image: status
white / grey / undetermined, why, the white-matter share of the analysed tissue, the reference density). A white
matter area with a "white" status is the algorithm's say; "grey" means it found none.

Like detection's other areas (folds, linear artifacts, no tissue, the AT8 edge guard) it is shown in the views
(`stainid.exclusions.view`, "white matter"), and people correct it in the "Grey / white matter" round
(`labels.matter`): their areas only train and evaluate this algorithm (`agreement`). Whether the analysis leaves it
out is the project's setting `v2: white_matter: analysis` ("off": counted as before; see docs/white_matter.md for
what "exclude" would change). This step changes no analysis output while it is off.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from stainid.tables import write_text_atomic

STAGE = "white_matter"
VERSION = 2  # bump when the code changes what it writes: every core is redone
SCALE = 8  # core px per mask px (the stage-2 masks' grid, `exclusions.SCALE`)
CELL = 8  # mask px per grid cell of the density (64 core px)
WHITE = 1  # the mask's value for white matter
TISSUE_BIT, TILE_BIT = 1, 32  # `v2.proposals.BITS`
STAINS = ("NeuN", "6E10", "AT8")
SOURCE = "NeuN"  # the stain the neurons are counted on
# `v2.white_matter.analysis`: what the analysis does with the area: "off" counts it as before; "exclude" leaves it out
# of stage 5's measures (objects in it dropped, its area out of every denominator, the grey-matter burdens of
# `v2.grey_matter`), with the whole-core values kept as `<column>_whole_core` columns (docs/white_matter.md)
ANALYSIS_CHOICES = ("off", "exclude")
PALE_REASON = "too few NeuN+ neurons"  # the start of the reason of a core whose NeuN stain is too pale to read
STATUSES = ("white", "grey", "undetermined")
COLUMNS = ["core_id", "stain", "status", "reason", "white_share", "white_mm2", "tissue_mm2", "reference_per_mm2", "neurons",
           "registration", "version"]

DEFAULTS = {
    "sigma_um": 100.0,  # the neuron density is smoothed over this (Gaussian sigma)
    "reference_quantile": 90.0,  # the core's grey-matter density: this quantile of its density over its tissue
    "min_reference_per_mm2": 80.0,  # below it the core's NeuN cannot be read: undetermined
    "min_tissue_mm2": 1.0,  # less analysed tissue than this: undetermined
    "seed": 0.10,  # seeds: below this share of the reference
    "grow": 0.25,  # grown through tissue below this share of the reference
    "open_um": 150.0,  # seeds are opened with a disc of this radius: a thin sparse band (layer I) is no seed
    # the sharp drop: tissue below `step` x the densest tissue within `near_um` (a drop to a fifth within 400 um: a step,
    # not a fade) seeds too, where that grey side has at least `min_grey_share` x the reference (and 80/mm²); the
    # region of each seed reaches the steepest part of the drop (a watershed of the density's gradient between the seeds
    # and the grey matter: tissue at `grey_marker` x the reference or more)
    "step": 0.2, "near_um": 400.0, "min_grey_share": 0.25, "grey_marker": 0.4,
    "min_area_mm2": 0.25,  # smaller parts are dropped
    "min_thick_um": 180.0,  # thinner parts (a band along an edge or a tear, layer I under the pia) are dropped
    # a part must reach this far from the tissue's outer edge (holes and tears inside the core are no edge) unless its
    # step is clear (`clear_step`) and its cells are those of white matter (`edge_cell_ratio`): white matter at a core's
    # edge is common; layer I, the pia and a frayed rim have few cells
    "min_depth_um": 450.0, "clear_step": 0.35, "edge_cell_ratio": 0.9,
    "max_step": 0.7,  # the part's density over the density just outside it, at most: a gradual fade is no white matter
    # the check on the cells (hematoxylin nuclei of the NeuN section and of the 6E10 / AT8 sections carried through the
    # registration, averaged): white matter has at least `min_cell_ratio` x the grey matter's cells (oligodendrocytes in
    # rows), or at most `max_dab_ratio` x its NeuN stain where the background stain is the grey matter's (no stained
    # neuropil, not a paler stain: the part's median DAB at least `min_background_ratio` x the grey matter's around it).
    # A drop in neurons together with a drop in the background stain is a pale stain, unless the cells say white matter
    "min_cell_ratio": 1.0, "max_dab_ratio": 0.65, "min_background_ratio": 0.65, "ring_um": 200.0,
    # the cells' sizes (stage 1's nuclei on the NeuN section): neurons have large nuclei (at least `large_nucleus_um2`),
    # glia small ones. A part with at most `max_large_ratio` x the grey matter's share of large nuclei is confirmed by
    # that alone (the share also varies between slides, so more large nuclei do not veto a part)
    # On a paler background (a NeuN stain gradient) fewer neurons need clearly glial nuclei: at most
    # `pale_large_ratio` x, or more cells
    "large_nucleus_um2": 45.0, "max_large_ratio": 0.3, "pale_large_ratio": 0.3,
    # a core with no grey matter of its own to compare with (its densest tissue under `min_reference_per_mm2`, or less
    # than `min_grey_mm2` of grey matter outside the candidates) is compared with the cohort's grey matter instead:
    # `cohort_large_share` large nuclei (the grey matter of the study's cores: 0.26-0.31; white matter 0.04), and its
    # 6E10 / AT8 sections must not show grey-matter pathology (at most `max_tau_per_mm2` AT8+ neurons and
    # `max_plaques_per_mm2` plaques per mm² of their analysed tissue: an AD core with neurons lost is grey matter)
    # `max_cohort_large_ratio` x that (a core of white matter has hardly any: 0.03-0.07; the cores whose neurons are only
    # unstained keep theirs: 0.10-0.32)
    "min_grey_mm2": 1.0, "cohort_large_share": 0.28, "max_cohort_large_ratio": 0.3, "max_tau_per_mm2": 8.0, "max_plaques_per_mm2": 50.0,
    "fill_holes_mm2": 0.1,  # holes smaller than this inside white matter are filled
    "grey_share": 0.5,  # the core's grey matter for the cell check: tissue at least this share of the reference density
    "analysis": "off",  # what the analysis does with it: "off" (counted as before) or "exclude" (ANALYSIS_CHOICES)
}


# ---------------------------------------------------------------- settings and places

def settings(project) -> dict:
    from stainid.v2.params import params  # noqa: PLC0415

    found = {**DEFAULTS, **(params(project).get("white_matter") or {})}
    if found["analysis"] not in ANALYSIS_CHOICES:
        raise ValueError(f"v2.white_matter.analysis must be one of {', '.join(ANALYSIS_CHOICES)}, not {found['analysis']!r} "
                         "(docs/white_matter.md)")
    return found


def excluded(project) -> bool:
    """Whether the analysis leaves the white matter out (`v2.white_matter.analysis: exclude`)."""
    return settings(project)["analysis"] == "exclude"


def neun_pale(project) -> set[str]:
    """The cores whose NeuN stain is too pale to read (the step's NeuN status undetermined for that reason): flagged in
    the results, never left out; empty before the step ran."""
    table = summary(project)
    if table.empty:
        return set()
    mine = table[(table.stain == SOURCE) & (table.status == "undetermined") & table.reason.fillna("").str.startswith(PALE_REASON)]
    return set(mine.core_id.astype(str))


def detection_settings(project) -> dict:
    """The settings the areas depend on (not `analysis`, which changes nothing here)."""
    return {k: v for k, v in settings(project).items() if k != "analysis"}


def root(project) -> Path:
    from stainid import candidates  # noqa: PLC0415

    return candidates.root(project) / STAGE


def mask_path(project, core_id: str, stain: str) -> Path:
    return root(project) / stain / f"{core_id}.png"


def manifest_path(project, core_id: str) -> Path:
    return root(project) / "manifests" / f"{core_id}.json"


def summary_path(project) -> Path:
    return root(project) / "white_matter.csv"


# ---------------------------------------------------------------- the algorithm

def _disc(radius: float) -> np.ndarray:
    r = max(1, int(round(radius)))
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return x * x + y * y <= r * r


def _cells(plane: np.ndarray, factor: int) -> np.ndarray:
    """Sums over blocks of `factor` x `factor` (the last, partial block too)."""
    h, w = -(-plane.shape[0] // factor), -(-plane.shape[1] // factor)
    padded = np.zeros((h * factor, w * factor), plane.dtype)
    padded[: plane.shape[0], : plane.shape[1]] = plane
    return padded.reshape(h, factor, w, factor).sum(axis=(1, 3))


def density(neurons: np.ndarray, tissue: np.ndarray, cell_um: float, sigma_um: float) -> np.ndarray:
    """Neurons per mm² of tissue, smoothed over `sigma_um` (normalised by the smoothed tissue, so the core edge and
    holes do not lower it); NaN where there is almost no tissue around."""
    sigma = sigma_um / cell_um
    num = cv2.GaussianBlur(neurons.astype(np.float32), (0, 0), sigma)
    den = cv2.GaussianBlur(tissue.astype(np.float32), (0, 0), sigma)
    cell_mm2 = (cell_um / 1000.0) ** 2
    return np.where(den > 0.15, num / np.maximum(den, 1e-6) / cell_mm2, np.nan)


def detect(tissue: np.ndarray, neurons: np.ndarray, cell_um: float, s: dict | None = None, look=None) -> tuple[np.ndarray, dict]:
    """The white matter of one NeuN core image on the density grid: `tissue` (share of each cell that is analysed
    tissue), `neurons` (NeuN+ neurons per cell), `cell_um` (a cell's side); `look()`: what the parts are checked with
    (`Look`, or a (cells, DAB) pair of the NeuN image), asked for only when there is a candidate part (None: parts are
    not checked). Returns the grid's white matter and {status, reason, reference_per_mm2, white_share, parts}: each
    candidate part's area, step, depth, thickness, its cells, nuclei, stains and pathology, and whether it was kept."""
    s = {**DEFAULTS, **(s or {})}
    cell_mm2 = (cell_um / 1000.0) ** 2
    present = tissue > 0.5
    found = density(neurons, tissue, cell_um, s["sigma_um"])
    values = found[present & np.isfinite(found)]
    none = np.zeros(tissue.shape, bool)
    if values.size * cell_mm2 < s["min_tissue_mm2"]:
        return none, {"status": "undetermined", "reason": f"less than {s['min_tissue_mm2']:g} mm² of analysed tissue", "reference_per_mm2": np.nan, "white_share": np.nan}
    reference = float(np.percentile(values, s["reference_quantile"]))
    level = np.where(present, np.nan_to_num(found, nan=0.0), 0.0).astype(np.float32)
    pale = reference < s["min_reference_per_mm2"]
    # a core whose densest tissue has hardly any NeuN+ neuron is a candidate as a whole: white matter, or grey matter
    # whose NeuN stain is pale or failed, which its cells and its other sections tell apart
    candidates = _drop_small(present, s["min_area_mm2"] / cell_mm2) if pale else _candidates(level, present, reference, cell_um, s)
    checked = []
    white = none
    if candidates.any():
        grids = look() if look is not None else None
        white, checked = _judge(candidates, level, present, tissue, reference, cell_um, grids, s)
    holes = ndi.binary_fill_holes(white) & ~white
    white |= holes & ~_drop_small(holes, s["fill_holes_mm2"] / cell_mm2) & present
    white = _drop_small(white & present, s["min_area_mm2"] / cell_mm2)  # nor any piece a tissue gap cut off
    if pale and not white.any():
        return none, {"status": "undetermined", "reason": f"{PALE_REASON} to tell grey from white matter (the densest part has "
                      f"{reference:.0f}/mm²: the NeuN stain is pale or failed)", "reference_per_mm2": reference, "white_share": np.nan,
                      "parts": checked}
    share = float(white.sum() / max(1, present.sum()))
    return white, {"status": "white" if white.any() else "grey", "reason": "", "reference_per_mm2": reference, "white_share": share,
                   "parts": checked}


def _candidates(level: np.ndarray, present: np.ndarray, reference: float, cell_um: float, s: dict) -> np.ndarray:
    """The candidate white matter: tissue grown from seeds far below the core's grey matter (`seed`, `grow`), and the
    low side of every sharp drop in density (`step`), reaching the drop's steepest part."""
    from skimage.segmentation import watershed  # noqa: PLC0415

    seeds = present & (level < s["seed"] * reference)
    room = present & (level < s["grow"] * reference)
    seeds = ndi.binary_opening(seeds, _disc(s["open_um"] / cell_um))
    grown = ndi.binary_propagation(seeds, mask=room) if seeds.any() else np.zeros(present.shape, bool)
    r = max(1, int(round(s["near_um"] / cell_um)))
    grey_near = cv2.dilate(level, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    floor = max(s["min_reference_per_mm2"], s["min_grey_share"] * reference)
    steps = present & (grey_near >= floor) & (level < s["step"] * grey_near)
    steps = _drop_small(ndi.binary_opening(steps, _disc(1.5)), 0.01 / (cell_um / 1000.0) ** 2)
    marks = steps | seeds
    if marks.any():
        grey = present & (level >= s["grey_marker"] * reference) & ~ndi.binary_dilation(marks, iterations=2)
        smooth = cv2.GaussianBlur(np.log(level + 20.0).astype(np.float32), (0, 0), 1.0)
        gy, gx = np.gradient(smooth)
        markers = np.zeros(present.shape, np.int32)
        markers[grey] = 1
        markers[marks] = 2
        grown |= (watershed(np.hypot(gx, gy), markers, mask=present) == 2) & present
    return _drop_small(grown, s["min_area_mm2"] / (cell_um / 1000.0) ** 2)


class Look:
    """What the parts are checked with, on the density grid: the NeuN image's cells (hematoxylin blobs), mean DAB and
    background DAB (the median of its tissue px); the cells of the other sections carried through the registration
    ((cells, tissue) pairs); `nuclei()`: stage 1's nuclei of the NeuN section ((all, large) counts per grid cell);
    `pathology()`: {"tau": AT8+ neurons per mm², "plaques": plaques per mm²} of the core's other sections. The two
    callables are asked only when a part needs them."""

    def __init__(self, cells, dab, background=None, others=(), nuclei=None, pathology=None):
        self.cells, self.dab, self.background, self.others = cells, dab, background, list(others)
        self._nuclei, self._pathology = nuclei, pathology
        self._got = {}

    def get(self, what):
        if what not in self._got:
            source = self._nuclei if what == "nuclei" else self._pathology
            self._got[what] = source() if source is not None else None
        return self._got[what]


def _judge(candidates: np.ndarray, level: np.ndarray, present: np.ndarray, tissue: np.ndarray, reference: float, cell_um: float,
           grids, s: dict) -> tuple[np.ndarray, list[dict]]:
    """Each candidate part kept or not (`DEFAULTS`): thick enough, a step at its edge, deep enough unless its step is
    clear and its cells are white matter's, and its cells those of white matter: few neuron-sized nuclei against the
    grey matter (the core's own, or the cohort's when the core has none, with no grey-matter pathology on its other
    sections), or more cells / less NeuN stain on an unchanged background."""
    if grids is not None and not isinstance(grids, Look):
        grids = Look(*grids)
    cell_mm2 = (cell_um / 1000.0) ** 2
    weight = tissue.astype(np.float64)
    solid = ndi.binary_fill_holes(present)
    depth = ndi.distance_transform_edt(np.pad(solid, 1))[1:-1, 1:-1] * cell_um
    grey = present & (level >= s["grey_share"] * max(reference, s["min_reference_per_mm2"])) & ~candidates
    own_grey = grey.sum() * cell_mm2 >= s["min_grey_mm2"] and reference >= s["min_reference_per_mm2"]
    ring_px = max(1, int(round(s["ring_um"] / cell_um)))

    def per_tissue(values, where, w=weight):
        return float(values[where].sum() / max(1e-9, w[where].sum()))

    def ratio(values, part, w=weight):
        if not own_grey:
            return np.nan
        g = per_tissue(values, grey, w)
        return per_tissue(values, part, w) / g if g > 0 else np.nan

    labels, n = ndi.label(candidates)
    out = np.zeros(present.shape, bool)
    parts = []
    for k in range(1, n + 1):
        part = labels == k
        thick = float(ndi.distance_transform_edt(np.pad(part, 1))[1:-1, 1:-1].max()) * cell_um
        around = ndi.binary_dilation(part, iterations=ring_px) & ~part & present
        inner = part & ~ndi.binary_erosion(part, iterations=ring_px)
        inner = inner if inner.any() else part
        step = float(np.median(level[inner]) / max(1e-6, float(np.median(level[around])))) if around.any() and own_grey else np.nan
        deep = float(depth[part].max())
        row = {"cells": int(part.sum()), "mm2": round(float(part.sum() * cell_mm2), 3), "step": _round(step), "depth_um": round(deep),
               "thick_um": round(thick), "own_grey": bool(own_grey)}
        shaped = thick >= s["min_thick_um"] and not step > s["max_step"]
        if grids is None:
            keep = shaped and deep >= s["min_depth_um"] and own_grey
        else:
            sections = [ratio(grids.cells, part)] + [ratio(c, part, t.astype(np.float64)) for c, t in grids.others]
            sections = [v for v in sections if np.isfinite(v)]
            cells = float(np.mean(sections)) if sections else np.nan
            dab = ratio(grids.dab, part)
            if grids.background is not None and around.any() and own_grey:
                background = float(np.nanmedian(grids.background[part]) / max(1e-6, float(np.nanmedian(grids.background[around]))))
            else:
                background = np.nan
            large = np.nan
            nuclei = grids.get("nuclei") if shaped else None
            if nuclei is not None:
                every, big = nuclei
                share = float(big[part].sum() / max(1.0, every[part].sum()))
                if own_grey:
                    g = float(big[grey].sum() / max(1.0, every[grey].sum()))
                    large = share / g if g > 0 else np.nan
                else:
                    large = share / s["cohort_large_share"]
            sick = False
            if not own_grey:
                ill = grids.get("pathology") if shaped else None
                ill = ill or {}
                sick = bool(ill.get("tau", 0.0) > s["max_tau_per_mm2"] or ill.get("plaques", 0.0) > s["max_plaques_per_mm2"])
                row |= {"tau_per_mm2": _round(ill.get("tau", np.nan)), "plaques_per_mm2": _round(ill.get("plaques", np.nan))}
            stain_ok = not background < s["min_background_ratio"]  # a paler stain, not missing neurons
            few_neurons = large <= s["max_large_ratio"]
            if own_grey:
                if stain_ok:
                    confirmed = few_neurons or cells >= s["min_cell_ratio"] or dab <= s["max_dab_ratio"]
                else:  # the drop in neurons comes with a drop in the background stain: a pale stain unless the cells say not
                    confirmed = cells >= s["min_cell_ratio"] or large <= s["pale_large_ratio"]
            else:
                confirmed = bool(large <= s["max_cohort_large_ratio"] and not sick)
            placed = deep >= s["min_depth_um"] or (step <= s["clear_step"] and (cells >= s["edge_cell_ratio"] or few_neurons))
            keep = bool(shaped and placed and confirmed)
            row |= {"cell_ratio": _round(cells), "dab_ratio": _round(dab), "background_ratio": _round(background), "large_ratio": _round(large)}
        row["kept"] = bool(keep)
        parts.append(row)
        if keep:
            out |= part
    return out, parts


def _round(value: float) -> float | None:
    return None if not np.isfinite(value) else round(float(value), 3)


HED_FROM_RGB = np.array([[1.87798274, -1.00767869, -0.55611582], [-0.06590806, 1.13473037, -0.13552180],
                         [-0.60190736, -0.48041419, 1.57358807]], np.float32)  # Ruifrok & Johnston (skimage's)
_OD = (-np.log10(np.maximum(np.arange(256, dtype=np.float32), 1.0) / 255.0)).astype(np.float32)
NUCLEUS_SIGMA_UM = 2.5  # the scale of the cell (nucleus) blobs counted: nuclei ~5-10 µm across
NUCLEUS_MIN_LOG, NUCLEUS_MIN_OD = 0.06, 0.15  # a blob's contrast (scale-normalised LoG) and hematoxylin optical density


def look_at(bgr: np.ndarray, um_per_px: float, shape: tuple[int, int], cell_px: int) -> tuple[np.ndarray, np.ndarray]:
    """Cells and NeuN stain of a core image read at `um_per_px` (BGR), on the density grid (`shape` cells of `cell_px`
    px of this picture): small dark hematoxylin blobs (cells: every nucleus, neurons' and glia's) per grid cell, and the
    DAB optical density summed over the grid cell's px / its px (glass adds nothing: divided by the tissue share it is the
    tissue's mean)."""
    od = _OD[bgr[..., ::-1]] @ HED_FROM_RGB
    hema, dab = np.maximum(od[..., 0], 0), np.maximum(od[..., 2], 0)
    smooth = cv2.GaussianBlur(hema, (0, 0), 1.0 / um_per_px)
    sigma = NUCLEUS_SIGMA_UM / um_per_px / 1.1
    log = -ndi.gaussian_laplace(smooth, sigma) * sigma ** 2
    peaks = (log == ndi.maximum_filter(log, max(3, int(round(2 * sigma)) | 1))) & (log > NUCLEUS_MIN_LOG) & (smooth > NUCLEUS_MIN_OD)
    cells, mean_dab = np.zeros(shape, np.float32), np.zeros(shape, np.float32)
    counted, stained = _cells(peaks.astype(np.float32), cell_px), _cells(dab, cell_px) / (cell_px * cell_px)
    h, w = min(shape[0], counted.shape[0]), min(shape[1], counted.shape[1])
    cells[:h, :w], mean_dab[:h, :w] = counted[:h, :w], stained[:h, :w]
    return cells, mean_dab


def background_at(bgr: np.ndarray, shape: tuple[int, int], cell_px: int) -> np.ndarray:
    """The background NeuN stain on the density grid: the median DAB optical density of each grid cell's tissue px (the
    neuropil; the few neurons do not move a median), NaN without tissue."""
    od = _OD[bgr[..., ::-1]]
    dab = np.maximum(od @ HED_FROM_RGB[:, 2], 0)
    tissue = od.sum(axis=2) > 0.15
    h, w = min(shape[0], dab.shape[0] // cell_px), min(shape[1], dab.shape[1] // cell_px)
    blocks = np.where(tissue, dab, np.nan)[: h * cell_px, : w * cell_px].reshape(h, cell_px, w, cell_px).transpose(0, 2, 1, 3)
    out = np.full(shape, np.nan, np.float32)
    with np.errstate(all="ignore"):
        import warnings  # noqa: PLC0415

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            out[:h, :w] = np.nanmedian(blocks.reshape(h, w, -1), axis=2)
    return out


def nuclei_grid(project, core_id: str, shape: tuple[int, int], large_um2: float) -> tuple[np.ndarray, np.ndarray] | None:
    """Stage 1's nuclei (Cellpose, on the hematoxylin) of the NeuN core image on the density grid: all nuclei and the
    large ones (at least `large_um2`: neurons' nuclei) per grid cell, each nucleus once (its tile's own px); None
    without them."""
    from stainid import candidates  # noqa: PLC0415

    try:
        tiles = candidates.read_table(candidates.path(project, SOURCE, core_id, "tiles"))
    except (FileNotFoundError, OSError, ValueError):
        return None
    if tiles is None or len(tiles) == 0:
        return None
    folder, ctx, ps = project.output("nuclei"), int(project.context_px), float(project.pixel_size_um)
    size = SCALE * CELL
    every, big = np.zeros(shape, np.float32), np.zeros(shape, np.float32)
    seen = False
    for t in tiles.to_dict("records"):
        try:
            with np.load(folder / f"{t['tile_id']}.npz") as z:
                lab = z["masks"]
        except (OSError, KeyError, ValueError):
            continue
        seen = True
        n = int(lab.max())
        if not n:
            continue
        flat = lab.ravel().astype(np.int64)
        area = np.bincount(flat, minlength=n + 1).astype(np.float64)
        yy, xx = np.divmod(np.arange(flat.size), lab.shape[1])
        sx, sy = np.bincount(flat, xx, n + 1), np.bincount(flat, yy, n + 1)
        ok = area > 0
        ok[0] = False
        x0, y0 = max(0, int(t["x_px"]) - ctx), max(0, int(t["y_px"]) - ctx)
        x, y, a = sx[ok] / area[ok] + x0, sy[ok] / area[ok] + y0, area[ok] * ps * ps
        own = (x >= t["x_px"]) & (x < t["x_px"] + t["width_px"]) & (y >= t["y_px"]) & (y < t["y_px"] + t["height_px"])
        gx, gy = (x[own] // size).astype(int), (y[own] // size).astype(int)
        inside = (gx < shape[1]) & (gy < shape[0])
        np.add.at(every, (gy[inside], gx[inside]), 1)
        np.add.at(big, (gy[inside], gx[inside]), (a[own][inside] >= large_um2).astype(np.float32))
    return (every, big) if seen else None


def pathology_of(project, core_id: str, stains: list[str]) -> dict:
    """The core's grey-matter pathology on its other sections: accepted AT8 calls (tau+ neurons) and 6E10 calls
    (plaques) per mm² of each image's analysed tissue."""
    from stainid import candidates  # noqa: PLC0415

    out = {}
    mm2 = (float(project.pixel_size_um) * SCALE / 1000.0) ** 2
    for stain, key in (("AT8", "tau"), ("6E10", "plaques")):
        if stain not in stains:
            continue
        bits = _bits(project, core_id, stain)
        if bits is None:
            continue
        analysed = analysed_of(bits)
        frame = candidates.read_core(project, stain, core_id, ["x_core", "y_core", "decision"])
        if not len(frame) or "decision" not in frame or not analysed.any():
            continue
        frame = frame[frame.decision.astype(str) == "accepted"]
        x = np.clip((frame.x_core.to_numpy(float) // SCALE).astype(int), 0, analysed.shape[1] - 1)
        y = np.clip((frame.y_core.to_numpy(float) // SCALE).astype(int), 0, analysed.shape[0] - 1)
        out[key] = float(analysed[y, x].sum() / (analysed.sum() * mm2))
    return out


def _drop_small(area: np.ndarray, min_cells: float) -> np.ndarray:
    labels, n = ndi.label(area)
    if not n:
        return area
    sizes = np.bincount(labels.ravel(), minlength=n + 1)
    keep = sizes >= min_cells
    keep[0] = False
    return keep[labels]


def on_mask_grid(white: np.ndarray, analysed: np.ndarray) -> np.ndarray:
    """The grid's white matter on the mask grid (`CELL` mask px per cell), its edge smoothed (bilinear, at one half),
    inside the analysed tissue."""
    if not white.any():
        return np.zeros(analysed.shape, bool)
    h, w = analysed.shape
    up = cv2.resize(white.astype(np.float32), (white.shape[1] * CELL, white.shape[0] * CELL), interpolation=cv2.INTER_LINEAR)
    out = np.zeros((h, w), bool)
    hh, ww = min(h, up.shape[0]), min(w, up.shape[1])
    out[:hh, :ww] = up[:hh, :ww] >= 0.5
    return out & analysed


def carry(white: np.ndarray, matrix: np.ndarray, analysed: np.ndarray) -> np.ndarray:
    """A white-matter area on one image's mask grid moved to another's (`matrix`: from the first image's core px to the
    other's), inside the other's analysed tissue."""
    if not white.any():
        return np.zeros(analysed.shape, bool)
    on_grid = np.array(matrix, np.float64)[:2].copy()
    on_grid[:, 2] /= SCALE  # the same rotation and scale on the 1/SCALE grid, the shift in grid px (cell centres at +0.5)
    on_grid[:, 2] += 0.5 * (on_grid[:, 0] + on_grid[:, 1]) - 0.5
    moved = cv2.warpAffine(white.astype(np.uint8), on_grid, (analysed.shape[1], analysed.shape[0]), flags=cv2.INTER_NEAREST,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return (moved > 0) & analysed


# ---------------------------------------------------------------- reading detection's outputs

def _bits(project, core_id: str, stain: str) -> np.ndarray | None:
    from stainid import exclusions  # noqa: PLC0415

    path = exclusions.mask_path(project, core_id, stain)
    return cv2.imread(str(path), cv2.IMREAD_UNCHANGED) if path.exists() else None


def analysed_of(bits: np.ndarray) -> np.ndarray:
    """Analysed tissue on the mask grid: tissue in a tile (folds are tissue: NeuN counts folded tissue)."""
    return ((bits & TISSUE_BIT) > 0) & ((bits & TILE_BIT) > 0)


def neuron_grid(project, core_id: str, analysed: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """The density grid of one NeuN core image: the share of each cell that is analysed tissue, the NeuN+ neurons
    (accepted NeuN calls) per cell, and how many there are."""
    from stainid import candidates  # noqa: PLC0415

    tissue = _cells(analysed.astype(np.float32), CELL) / (CELL * CELL)
    neurons = np.zeros(tissue.shape, np.float32)
    frame = candidates.read_core(project, SOURCE, core_id, ["x_core", "y_core", "decision"])
    if len(frame) and "decision" in frame:
        accepted = frame[frame.decision.astype(str) == "accepted"]
        size = SCALE * CELL
        x, y = accepted.x_core.to_numpy(float), accepted.y_core.to_numpy(float)
        inside = (x >= 0) & (y >= 0) & (x < tissue.shape[1] * size) & (y < tissue.shape[0] * size)
        np.add.at(neurons, ((y[inside] // size).astype(int), (x[inside] // size).astype(int)), 1)
        return tissue, neurons, int(inside.sum())
    return tissue, neurons, 0


# ---------------------------------------------------------------- the step

def _inputs(project, core_id: str, stains: list[str]) -> dict:
    from stainid import exclusions  # noqa: PLC0415
    from stainid.registration import cores as registration  # noqa: PLC0415
    from stainid.v2 import manifest  # noqa: PLC0415

    calls = manifest.read(project, "candidates", SOURCE, core_id) or manifest.read(project, "calls", SOURCE, core_id) or {}
    rows = registration.table(project)
    return {"version": VERSION, "settings": detection_settings(project), "neun_calls": calls.get("digest"),
            "masks": {s: exclusions.digest(project, core_id, s) for s in stains},
            "registration": {s: {k: v for k, v in (rows.get((core_id, s)) or {}).items() if k.startswith("native_m") or k in ("status", "reference_stain")}
                             for s in stains if s != SOURCE}}


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _write_png(path: Path, white: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", white.astype(np.uint8) * WHITE)
    if not ok:
        raise RuntimeError(f"could not encode {path}")
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(encoded.tobytes())
    tmp.replace(path)


def core_stains(project) -> dict[str, list[str]]:
    """The stain images of each core that have a stage-2 mask, NeuN first."""
    from stainid import exclusions  # noqa: PLC0415

    out: dict[str, list[str]] = {}
    for stain in STAINS:
        folder = exclusions.masks_folder(project, stain)
        if folder.is_dir():
            for path in folder.glob("*.png"):
                out.setdefault(path.stem, []).append(stain)
    return {core: [s for s in STAINS if s in found] for core, found in sorted(out.items())}


def run_core(project, core_id: str, stains: list[str], picture=None) -> list[dict]:
    """One core: the NeuN white matter and its carry to the core's other stain images; writes the masks and the
    manifest, returns the summary rows. `picture(core_id, stain)`: a whole-core RGB picture for registering a pair
    the registration table has no row for (`registration.cores.carry_project`)."""
    from stainid.registration import cores as registration  # noqa: PLC0415

    s = detection_settings(project)
    ps = float(project.pixel_size_um)
    cell_um = ps * SCALE * CELL
    rows = []
    bits = _bits(project, core_id, SOURCE) if SOURCE in stains else None
    if bits is None:
        info = {"status": "undetermined", "reason": "no NeuN image: white matter is found from the NeuN+ neurons", "reference_per_mm2": np.nan,
                "white_share": np.nan}
        source_white, neurons = None, 0
    else:
        analysed = analysed_of(bits)
        tissue, grid, neurons = neuron_grid(project, core_id, analysed)
        cells, info = detect(tissue, grid, cell_um, s, look=lambda: _look(project, core_id, tissue.shape, stains, s, picture))
        source_white = on_mask_grid(cells, analysed)
    for stain in stains:
        target_bits = bits if stain == SOURCE else _bits(project, core_id, stain)
        if target_bits is None:
            continue
        analysed = analysed_of(target_bits)
        status, reason, how = info["status"], info["reason"], ""
        if source_white is None:
            white = np.zeros(analysed.shape, bool)
        elif stain == SOURCE:
            white = source_white
        elif not source_white.any():
            white = np.zeros(analysed.shape, bool)
        else:
            try:
                moved = registration.carry_project(project, core_id, SOURCE, stain, picture)
            except (KeyError, FileNotFoundError, ValueError, RuntimeError) as error:
                moved = None
                reason = f"the NeuN and {stain} images could not be lined up: {error}"
            how = "" if moved is None else f"{moved.how}:{moved.status}"
            if moved is None or moved.status == "fail":
                white = np.zeros(analysed.shape, bool)
                status = "undetermined"
                reason = reason or f"the registration of the NeuN and {stain} images failed"
            else:
                white = carry(source_white, moved.matrix, analysed)
        _write_png(mask_path(project, core_id, stain), white)
        mm2 = (ps * SCALE / 1000.0) ** 2
        tissue_cells = int(analysed.sum())
        rows.append({"core_id": core_id, "stain": stain, "status": status, "reason": reason,
                     "white_share": round(float(white.sum() / tissue_cells), 4) if tissue_cells and status != "undetermined" else np.nan,
                     "white_mm2": round(float(white.sum() * mm2), 4), "tissue_mm2": round(tissue_cells * mm2, 4),
                     "reference_per_mm2": None if not np.isfinite(info.get("reference_per_mm2", np.nan)) else round(float(info["reference_per_mm2"]), 2),
                     "neurons": neurons, "registration": how, "version": VERSION})
    return rows


def _look(project, core_id: str, shape: tuple[int, int], stains: list[str], s: dict, picture=None) -> Look:
    """What a NeuN core's candidate parts are checked with (`Look`): the NeuN image read at 1/4 (`look_at`,
    `background_at`), the cells of its 6E10 and AT8 images carried onto the NeuN grid through the registration (a pair
    that cannot be lined up is left out), stage 1's nuclei of the NeuN section and the other sections' pathology."""
    from stainid.imaging.core_images import read_bgr  # noqa: PLC0415
    from stainid.registration import cores as registration  # noqa: PLC0415
    from stainid.registration.cores import _project_native  # noqa: PLC0415

    _, rows = _project_native(project)
    um = float(project.pixel_size_um) * 4
    px = SCALE * CELL // 4

    def read(stain):
        bgr = read_bgr(project.root / rows[(core_id, stain)]["image_path"], cv2.IMREAD_REDUCED_COLOR_4)
        if bgr is None:
            raise FileNotFoundError(f"cannot read the {stain} image of {core_id}")
        return bgr

    bgr = read(SOURCE)
    cells, dab = look_at(bgr, um, shape, px)
    background = background_at(bgr, shape, px)
    others = []
    for stain in stains:
        if stain == SOURCE or (core_id, stain) not in rows:
            continue
        try:
            moved = registration.carry_project(project, core_id, SOURCE, stain, picture)
            bits = _bits(project, core_id, stain)
            if moved is None or moved.status == "fail" or bits is None:
                continue
            analysed = analysed_of(bits)
            own = (-(-analysed.shape[0] // CELL), -(-analysed.shape[1] // CELL))
            their_cells, _ = look_at(read(stain), um, own, px)
            their_tissue = (_cells(analysed.astype(np.float32), CELL) / (CELL * CELL))[: own[0], : own[1]]
        except (KeyError, FileNotFoundError, ValueError, RuntimeError):
            continue
        m = np.array(moved.matrix, np.float64)[:2].copy()  # NeuN core px -> the other image's core px, on the grid
        m[:, 2] /= SCALE * CELL
        m[:, 2] += 0.5 * (m[:, 0] + m[:, 1]) - 0.5

        def onto(values):
            return cv2.warpAffine(values.astype(np.float32), m, (shape[1], shape[0]), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        others.append((onto(their_cells), onto(their_tissue)))
    return Look(cells, dab, background, others, nuclei=lambda: nuclei_grid(project, core_id, shape, s["large_nucleus_um2"]),
                pathology=lambda: pathology_of(project, core_id, stains))


def todo(project, redo: bool = False, cores: set[str] | None = None) -> list[tuple[str, list[str], dict]]:
    """(core, its stain images, the inputs) of every core whose areas are missing or outdated."""
    out = []
    for core_id, stains in core_stains(project).items():
        if cores is not None and core_id not in cores:
            continue
        inputs = _inputs(project, core_id, stains)
        if not redo:
            try:
                recorded = json.loads(manifest_path(project, core_id).read_text())
            except (OSError, ValueError):
                recorded = None
            if recorded and recorded.get("digest") == _digest(inputs) and all(mask_path(project, core_id, s).exists() for s in stains):
                continue
        out.append((core_id, stains, inputs))
    return out


def run(project, redo: bool = False, cores: set[str] | None = None, picture=None, log=print) -> pd.DataFrame:
    """The step: every core whose areas are missing or outdated, then the summary table of all of them."""
    jobs = todo(project, redo, cores)
    log(f"white matter: {len(jobs)} core(s) to do")
    kept = {}
    for n, (core_id, stains, inputs) in enumerate(jobs, 1):
        rows = run_core(project, core_id, stains, picture)
        target = manifest_path(project, core_id)
        target.parent.mkdir(parents=True, exist_ok=True)
        write_text_atomic(target, json.dumps({"stage": STAGE, "inputs": inputs, "digest": _digest(inputs), "rows": rows,
                                              "written": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=1, default=str))
        kept[core_id] = rows
        found = next((r for r in rows if r["stain"] == SOURCE), rows[0] if rows else None)
        if found:
            log(f"{n}/{len(jobs)} {core_id}: {found['status']}" + (f" {100 * found['white_share']:.0f} %" if found["status"] == "white" else ""))
    return write_summary(project)


def write_summary(project) -> pd.DataFrame:
    """`white_matter.csv` from every core's manifest."""
    rows = []
    folder = root(project) / "manifests"
    for path in sorted(folder.glob("*.json")) if folder.is_dir() else []:
        try:
            rows += json.loads(path.read_text()).get("rows", [])
        except (OSError, ValueError):
            continue
    frame = pd.DataFrame(rows, columns=COLUMNS)
    target = summary_path(project)
    target.parent.mkdir(parents=True, exist_ok=True)
    write_text_atomic(target, frame.to_csv(index=False))
    return frame


# ---------------------------------------------------------------- reading the areas

def read(project, core_id: str, stain: str) -> np.ndarray | None:
    """The algorithm's white matter of one core image on the masks' grid (bool); None before the step made it."""
    from stainid import cache  # noqa: PLC0415

    path = mask_path(project, core_id, stain)
    if cache.signature(path) is None:
        return None

    def make() -> np.ndarray:
        found = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        return np.zeros((0, 0), bool) if found is None else found == WHITE

    out = cache.memo(("white matter", str(path)), [path], make)
    return out if out.size else None


def digest(project, core_id: str, stain: str) -> str:
    """Changes whenever the step rewrites the core image's area ("" before it made one)."""
    from stainid import cache  # noqa: PLC0415

    found = cache.signature(mask_path(project, core_id, stain))
    return "" if found is None else hashlib.sha1(repr(found).encode()).hexdigest()[:12]


def status_of(project, core_id: str, stain: str) -> dict | None:
    """The step's say about one core image (its row of `white_matter.csv`, from the core's manifest): status, reason,
    white_share, …, and `version` (`digest`); None before the step ran on the core."""
    from stainid import cache  # noqa: PLC0415

    path = manifest_path(project, core_id)
    if cache.signature(path) is None:
        return None

    def make() -> list:
        try:
            return json.loads(path.read_text()).get("rows", [])
        except (OSError, ValueError):
            return []

    row = next((r for r in cache.memo(("white matter rows", str(path)), [path], make) if r.get("stain") == stain), None)
    if row is None:
        return None
    clean = {k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in row.items()}
    return {**clean, "version": digest(project, core_id, stain)}


def png(white: np.ndarray, shape: tuple[int, int]) -> str:
    """An area on the masks' grid as a white-on-transparent PNG data URL of `shape` (the brush's form, `labels.matter.png`)."""
    import base64  # noqa: PLC0415

    cells = np.zeros(shape, bool)
    h, w = min(shape[0], white.shape[0]), min(shape[1], white.shape[1])
    cells[:h, :w] = white[:h, :w]
    rgba = np.zeros((*shape, 4), np.uint8)
    rgba[cells] = 255
    ok, encoded = cv2.imencode(".png", rgba)
    if not ok:
        raise RuntimeError("could not encode the white matter")
    return "data:image/png;base64," + base64.b64encode(encoded.tobytes()).decode()


def summary(project) -> pd.DataFrame:
    path = summary_path(project)
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(path, dtype={"core_id": str, "stain": str})


def agreement(algorithm: np.ndarray, people: np.ndarray, analysed: np.ndarray) -> dict:
    """The algorithm's white matter against people's on one core image (cells of the masks' grid inside the analysed
    tissue): IoU, Dice, each one's share and their difference (people's white matter is the truth)."""
    a, p = algorithm & analysed, people & analysed
    both, either = int((a & p).sum()), int((a | p).sum())
    tissue = max(1, int(analysed.sum()))
    return {"iou": both / either if either else 1.0, "dice": 2 * both / (int(a.sum()) + int(p.sum())) if either else 1.0,
            "algorithm_share": round(float(a.sum() / tissue), 4), "people_share": round(float(p.sum() / tissue), 4),
            "share_difference": round(float((a.sum() - p.sum()) / tissue), 4)}
