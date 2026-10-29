"""White matter, found automatically: the "Grey / white matter" algorithm, a step after detection (`stainid v2
white-matter`) that reads detection's stored outputs and never re-runs it.

What it looks at: the NeuN+ neurons (stage 4's accepted NeuN calls) on the NeuN section, and the cells (hematoxylin
nuclei, on every section; neurons are only the NeuN+ ones) of the NeuN image to confirm. Grey matter is where the
neurons are; white matter is tissue with almost none and more cells than the grey matter, and the cortex ends in it at
a sharp drop in neuron density (`docs/white_matter.md`).

Per NeuN core image (`detect`), on a grid of `CELL` mask px (64 core px, 17.5 µm):
1. the NeuN+ neuron density over the analysed tissue (stage 2's mask: tissue in a tile), smoothed over `sigma_um`;
2. the core's own grey-matter density, `reference`: the `reference_quantile` of that density over its tissue. A core
   whose reference is under `min_reference_per_mm2` cannot be read (its NeuN stain is too pale or failed: the core
   has hardly any NeuN+ neuron even where 6E10 and AT8 show grey-matter pathology) and is "undetermined";
3. seeds: tissue below `seed` × the reference, opened with a disc of `open_um` (a thin sparse band, such as layer I
   under the pia, goes); grown (region growing) through tissue below `grow` × the reference; parts under
   `min_area_mm2` dropped;
4. each part checked on the NeuN image itself (`look_at`, read at 1/4): kept when it has at least `min_cell_ratio` x
   the cells per mm² of the core's grey matter (hematoxylin nuclei: white matter's rows of oligodendrocytes) or at most
   `max_dab_ratio` x its NeuN stain (no stained neuropil). Grey matter that is only poor in NeuN+ neurons (a pale or
   uneven NeuN stain, deep layer VI) has neither, and goes. Holes under `fill_holes_mm2` are filled;
5. the area on the mask grid (1/8 of the core image), its edge smoothed, inside the analysed tissue.

Which cue carries it: the neurons. Cells alone separate white from grey matter within a core with a median AUC of 0.9
(NeuN, 6E10 and AT8 sections alike) but fail on a quarter of the cores; here they only confirm the parts the neurons
propose.

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
VERSION = 1  # bump when the code changes what it writes: every core is redone
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
    "min_area_mm2": 0.25,  # smaller parts are dropped
    # a part must reach this far from the tissue's edge (glass, tears, holes): a band along an edge (layer I under the
    # pia, the leptomeninges, a sulcus, a frayed core edge) is no white matter
    "min_depth_um": 450.0,
    "fill_holes_mm2": 0.1,  # holes smaller than this inside white matter are filled
    # the check on the NeuN image itself: a part is kept when it has at least `min_cell_ratio` x the cells (hematoxylin
    # nuclei) per mm² of the core's grey matter, or at most `max_dab_ratio` x its NeuN stain (mean DAB). White matter has
    # more cells (oligodendrocytes in rows) and no NeuN-stained neuropil; grey matter that is only sparse in NeuN+
    # neurons (a pale NeuN stain, deep layer VI) has neither
    "min_cell_ratio": 1.0, "max_dab_ratio": 0.65,
    "grey_share": 0.5,  # the core's grey matter for that check: tissue at least this share of the reference density
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
    tissue), `neurons` (NeuN+ neurons per cell), `cell_um` (a cell's side); `look()`: (cells, DAB) grids of the NeuN image
    (`look_at`), asked for only when there is a candidate part to check (None: parts are not checked). Returns the
    grid's white matter and {status, reason, reference_per_mm2, white_share, parts}: each candidate part's area, cell
    and DAB ratio and whether it was kept."""
    s = {**DEFAULTS, **(s or {})}
    cell_mm2 = (cell_um / 1000.0) ** 2
    present = tissue > 0.5
    found = density(neurons, tissue, cell_um, s["sigma_um"])
    values = found[present & np.isfinite(found)]
    none = np.zeros(tissue.shape, bool)
    if values.size * cell_mm2 < s["min_tissue_mm2"]:
        return none, {"status": "undetermined", "reason": f"less than {s['min_tissue_mm2']:g} mm² of analysed tissue", "reference_per_mm2": np.nan, "white_share": np.nan}
    reference = float(np.percentile(values, s["reference_quantile"]))
    if reference < s["min_reference_per_mm2"]:
        return none, {"status": "undetermined", "reason": f"too few NeuN+ neurons to tell grey from white matter (the densest part "
                      f"has {reference:.0f}/mm²: the NeuN stain is pale or failed)", "reference_per_mm2": reference, "white_share": np.nan}
    level = np.nan_to_num(found, nan=np.inf)
    seeds = present & (level < s["seed"] * reference)
    room = present & (level < s["grow"] * reference)
    seeds = ndi.binary_opening(seeds, _disc(s["open_um"] / cell_um))
    white = ndi.binary_propagation(seeds, mask=room) if seeds.any() else none
    white = _drop_small(white, s["min_area_mm2"] / cell_mm2)
    white = _deep_enough(white, present, s["min_depth_um"] / cell_um)
    checked = []
    if white.any() and look is not None:
        grey = present & (level >= s["grey_share"] * reference) & ~white
        white, checked = _confirm(white, grey, tissue, look(), s)
    white &= present
    holes = ndi.binary_fill_holes(white) & ~white
    white |= holes & ~_drop_small(holes, s["fill_holes_mm2"] / cell_mm2) & present
    white = _drop_small(white, s["min_area_mm2"] / cell_mm2)  # nor any piece a tissue gap cut off
    share = float(white.sum() / max(1, present.sum()))
    return white, {"status": "white" if white.any() else "grey", "reason": "", "reference_per_mm2": reference, "white_share": share,
                   "parts": checked}


def _confirm(white: np.ndarray, grey: np.ndarray, tissue: np.ndarray, grids: tuple[np.ndarray, np.ndarray], s: dict) -> tuple[np.ndarray, list[dict]]:
    """Keep the candidate parts the NeuN image confirms (`min_cell_ratio`, `max_dab_ratio`), each against the core's
    grey matter (`grey`)."""
    cells, dab = grids
    weight = tissue.astype(np.float64)

    def per_tissue(values, where):
        return float(values[where].sum() / max(1e-9, weight[where].sum()))

    if weight[grey].sum() < 1:  # no grey matter to compare with: nothing to say against the parts
        return white, []
    grey_cells, grey_dab = per_tissue(cells, grey), per_tissue(dab, grey)
    labels, n = ndi.label(white)
    out = np.zeros(white.shape, bool)
    parts = []
    for k in range(1, n + 1):
        part = labels == k
        cell_ratio = per_tissue(cells, part) / max(grey_cells, 1e-9)
        dab_ratio = per_tissue(dab, part) / max(grey_dab, 1e-9)
        keep = cell_ratio >= s["min_cell_ratio"] or dab_ratio <= s["max_dab_ratio"]
        parts.append({"cells": int(part.sum()), "cell_ratio": round(cell_ratio, 3), "dab_ratio": round(dab_ratio, 3), "kept": bool(keep)})
        if keep:
            out |= part
    return out, parts


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


def _deep_enough(area: np.ndarray, present: np.ndarray, cells: float) -> np.ndarray:
    """The parts of `area` reaching at least `cells` from the edge of the tissue (`present`; outside the grid is no
    tissue)."""
    labels, n = ndi.label(area)
    if not n:
        return area
    depth = ndi.distance_transform_edt(np.pad(present, 1))[1:-1, 1:-1]
    reach = ndi.maximum(depth, labels, index=np.arange(1, n + 1))
    keep = np.r_[False, np.asarray(reach) >= cells]
    return keep[labels]


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
        cells, info = detect(tissue, grid, cell_um, s, look=lambda: _look(project, core_id, tissue.shape))
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


def _look(project, core_id: str, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """`look_at` the NeuN core image, read at 1/4."""
    from stainid.imaging.core_images import read_bgr  # noqa: PLC0415
    from stainid.registration.cores import _project_native  # noqa: PLC0415

    _, rows = _project_native(project)
    bgr = read_bgr(project.root / rows[(core_id, SOURCE)]["image_path"], cv2.IMREAD_REDUCED_COLOR_4)
    if bgr is None:
        raise FileNotFoundError(f"cannot read the {SOURCE} image of {core_id}")
    return look_at(bgr, float(project.pixel_size_um) * 4, shape, SCALE * CELL // 4)


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
