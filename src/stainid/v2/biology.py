"""Stage 2 measurements of biology rather than detection: the nucleus inside a soma, a plaque's radial DAB profile, and
vessel lumens. Additions only: no model input, rule or existing column uses them (models and scores may later).

Nucleus in the soma (`soma_nucleus`, AT8 and NeuN candidates). The cells of the field (`stainid.v2.cells`: Cellpose
nuclei with enough hematoxylin, 10-200 µm²) with at least half of their area inside the candidate; the one most inside is
the soma's nucleus. `soma_nucleus_visible` 1 when it holds at least as much hematoxylin as DAB (the rule by which 6E10 and
AT8 count a nucleus as a cell: a blue nucleus seen through the stain), 0 when there is none or the stain hides it (or
Cellpose took a dark tangle for a nucleus); `soma_nucleus_offset` its centre's distance from the soma's centroid over the
soma's equivalent radius, `soma_nucleus_area_um2`, `soma_nucleus_hematoxylin_od`, `soma_nucleus_dab_od` (NaN without one);
`soma_hematoxylin_share` the share of the soma's own pixels with at least as much hematoxylin as DAB (no Cellpose needed).
NaN everywhere when the field has no nuclei. On NeuN the nucleus is the neuron's own (stained brown): its area and DAB.

Radial profile (`radial_profile`, 6E10 candidates at least `PROFILE_MIN_UM` across; NaN for specks). From the deposit's peak (the maximum of its DAB smoothed over 1 µm),
the mean DAB of the valid tissue in rings of `RING_UM` out to `PROFILE_UM`: `radial_dab_um` (OD per ring, inner first), and
in slide thresholds over fixed bins `BINS_UM`: `profile_core_dab` (0-4 µm), `profile_inner_halo_dab` (4-10 µm),
`profile_outer_halo_dab` (10-20 µm), `profile_background_dab` (25-40 µm). Derived: `profile_core_contrast` (core minus
inner halo), `profile_core_to_halo` (core over the mean of both halo bins), `profile_core_radius_um` (the first ring below
halfway between the core and one slide threshold; 0 when the core is below the threshold),
`profile_extent_radius_um` (the first ring below one slide threshold) and `profile_halo_width_um` (extent minus core
radius). A cored plaque with a halo: a strong core, a halo above the threshold, a halo width of several µm; a bare
("burnt-out") core: a strong core, a halo near the background, a width near 0; a diffuse plaque: little core contrast.

Vessels (`vessels`, every stain, on the counterstain). Lumen candidates are the field's enclosed empty spaces
(lighter than the core's glass-level tissue cut, holes not filled, not touching the crop) of `LUMEN_UM2`; per lumen its
shape and its lining: cells (Cellpose nuclei with enough hematoxylin) centred within `LINING_UM` outside the lumen, how
many are elongated (`ELONGATED`) and lie along the wall (major axis within 30° of its direction), and the nuclei of
tissue the space surrounds (`island_nuclei`). A lumen is a vessel when it is at least `MIN_VESSEL_UM` across, no tear (solidity at
least `MIN_SOLIDITY`) and has two elongated nuclei along its wall (one per `LINING_PER_UM` of perimeter for a large one),
or one and an elongated shape (`ELONGATED_LUMEN`, a capillary cut along its length); or it is a large smooth round
space (`ROUND_VESSEL_UM`, solidity `ROUND_SOLIDITY`); or it surrounds tissue of `ISLAND_UM2` (a vessel in its
perivascular space).
On 6E10 the wall band (`WALL_UM` outside the lumen) gives the CAA-like ring: `wall_positive_fraction` (band pixels at or
above the slide threshold) and `wall_coverage` (of 16 sectors around the lumen, the share with a tenth of their band
positive).
"""
from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage

RING_UM = 2.0
PROFILE_UM = 46.0
PROFILE_MIN_UM = 10.0  # the profile is measured for deposits at least this wide (equivalent diameter): no specks
BINS_UM = {"core": (0.0, 4.0), "inner_halo": (4.0, 10.0), "outer_halo": (10.0, 20.0), "background": (25.0, 40.0)}
LUMEN_UM2 = (50.0, 20000.0)  # 8 µm across and more: smaller spaces are no vessel by the rules below
LINING_UM = 4.0
ELONGATED = 1.8
MIN_LINING = 1
LINING_PER_UM = 60.0
MIN_SOLIDITY = 0.55
ELONGATED_LUMEN = 2.5  # a capillary cut along its length
MIN_VESSEL_UM = 8.0  # smaller lined spaces are mostly the halo of a cell with a satellite nucleus (pilot, by eye)
ROUND_VESSEL_UM, ROUND_SOLIDITY = 15.0, 0.85  # a large smooth round space: a vessel's lumen
ISLAND_UM2 = 30.0
WALL_UM = 4.0
SECTORS = 16

NUCLEUS_COLUMNS = ("soma_nucleus_visible", "soma_nucleus_offset", "soma_nucleus_area_um2", "soma_nucleus_hematoxylin_od", "soma_nucleus_dab_od",
                   "soma_hematoxylin_share")
PROFILE_COLUMNS = ("radial_dab_um", "profile_core_dab", "profile_inner_halo_dab", "profile_outer_halo_dab", "profile_background_dab",
                   "profile_core_contrast", "profile_core_to_halo", "profile_core_radius_um", "profile_extent_radius_um", "profile_halo_width_um")
VESSEL_COLUMNS = ("tile_id", "x_core", "y_core", "area_um2", "equivalent_diameter_um", "elongation", "solidity", "perimeter_um", "lining_nuclei",
                  "lining_along_wall", "island_nuclei", "rim_hematoxylin_od", "lumen_dab_od", "wall_positive_fraction", "wall_coverage", "vessel", "outline")


def soma_nucleus(field, region) -> dict:
    """The nucleus inside one candidate (see the module's docstring)."""
    from stainid.v2 import cells  # noqa: PLC0415

    if field.raw_nuclei is None:
        return dict.fromkeys(NUCLEUS_COLUMNS, float("nan"))
    ps = field.pixel_size_um
    box, mask = region.box, region.mask
    h, d = field.hematoxylin[box][mask], field.dab[box][mask]
    out = {"soma_hematoxylin_share": float((h >= d).mean()) if mask.any() else float("nan")}
    table = cells._all(field)
    inside = field.nuclei[box][mask]
    labels, counts = np.unique(inside[inside > 0], return_counts=True)
    found = table[table.label.isin(labels)].set_index("label") if len(labels) and len(table) else table.iloc[0:0]
    best, best_fraction = None, 0.5
    for label, count in zip(labels, counts):
        if label in found.index and count / found.at[label, "area_px"] >= best_fraction:
            best, best_fraction = label, count / found.at[label, "area_px"]
    if best is None:
        return out | {"soma_nucleus_visible": 0.0, "soma_nucleus_offset": float("nan"), "soma_nucleus_area_um2": float("nan"),
                      "soma_nucleus_hematoxylin_od": float("nan"), "soma_nucleus_dab_od": float("nan")}
    row = found.loc[best]
    my, mx = np.nonzero(mask)
    radius = np.sqrt(mask.sum() / np.pi)
    offset = np.hypot(row.x - (region.x0 + mx.mean()), row.y - (region.y0 + my.mean())) / radius if radius else float("nan")
    return out | {"soma_nucleus_visible": float(row.hematoxylin_mean >= row.dab_mean), "soma_nucleus_offset": float(offset),
                  "soma_nucleus_area_um2": float(row.area_px * ps * ps), "soma_nucleus_hematoxylin_od": float(row.hematoxylin_mean),
                  "soma_nucleus_dab_od": float(row.dab_mean)}


def _smoothed_dab(field) -> np.ndarray:
    if "dab_1um" not in field.memo:
        field.memo["dab_1um"] = cv2.GaussianBlur(field.dab, (0, 0), max(0.5, 1.0 / field.pixel_size_um))
    return field.memo["dab_1um"]


def profile_numbers(rings: np.ndarray, threshold: float) -> dict:
    """The derived profile numbers from the ring means (OD, `RING_UM` steps from the peak)."""
    centres = (np.arange(len(rings)) + 0.5) * RING_UM

    def mean_in(lo: float, hi: float) -> float:
        picked = rings[(centres >= lo) & (centres < hi)]
        picked = picked[np.isfinite(picked)]
        return float(picked.mean() / threshold) if len(picked) else float("nan")

    out = {f"profile_{name}_dab": mean_in(*span) for name, span in BINS_UM.items()}
    core, inner, outer = out["profile_core_dab"], out["profile_inner_halo_dab"], out["profile_outer_halo_dab"]
    halo = np.nanmean([inner, outer]) if np.isfinite([inner, outer]).any() else float("nan")
    out["profile_core_contrast"] = core - inner
    out["profile_core_to_halo"] = core / halo if halo and np.isfinite(halo) and halo > 0 else float("nan")
    scaled = rings / threshold

    def first_below(level: float) -> float:
        below = np.flatnonzero(np.isfinite(scaled) & (scaled < level))
        return float(centres[below[0]] - RING_UM / 2) if len(below) else float(centres[-1] + RING_UM / 2)

    core_radius = first_below((core + 1.0) / 2) if np.isfinite(core) and core > 1.0 else 0.0
    extent = first_below(1.0) if np.isfinite(scaled[0]) and scaled[0] >= 1.0 else 0.0
    out |= {"profile_core_radius_um": core_radius, "profile_extent_radius_um": extent, "profile_halo_width_um": max(0.0, extent - core_radius)}
    return out


def _template(ps: float) -> np.ndarray:
    """The ring index of every pixel of a (2 reach + 1)² window around its centre, made once."""
    reach = int(np.ceil(PROFILE_UM / ps))
    key = (round(ps, 6), reach)
    if key not in _TEMPLATES:
        yy, xx = np.mgrid[-reach: reach + 1, -reach: reach + 1]
        _TEMPLATES[key] = (np.hypot(yy, xx) * ps / RING_UM).astype(np.int16)
    return _TEMPLATES[key]


_TEMPLATES: dict = {}


def radial_profile(field, region) -> dict:
    """One deposit's radial DAB profile (see the module's docstring); missing for deposits under `PROFILE_MIN_UM`."""
    ps, threshold = field.pixel_size_um, field.threshold
    if region.area_px * ps * ps < np.pi * (PROFILE_MIN_UM / 2) ** 2:
        return dict.fromkeys(PROFILE_COLUMNS, float("nan")) | {"radial_dab_um": ""}
    smooth = _smoothed_dab(field)
    local = np.where(region.mask, smooth[region.box], -np.inf)
    py, px = np.unravel_index(int(np.argmax(local)), local.shape)
    cy, cx = py + region.y0, px + region.x0
    reach = int(np.ceil(PROFILE_UM / ps))
    y0, y1, x0, x1 = max(0, cy - reach), min(field.shape[0], cy + reach + 1), max(0, cx - reach), min(field.shape[1], cx + reach + 1)
    rings_of = _template(ps)
    window = (slice(y0 - cy + reach, y1 - cy + reach), slice(x0 - cx + reach, x1 - cx + reach))
    ring = rings_of[window]
    keep = field.valid[y0:y1, x0:x1] & (ring < int(PROFILE_UM / RING_UM))
    n = int(PROFILE_UM / RING_UM)
    sums = np.bincount(ring[keep], weights=field.dab[y0:y1, x0:x1][keep], minlength=n)[:n]
    counts = np.bincount(ring[keep], minlength=n)[:n]
    with np.errstate(invalid="ignore", divide="ignore"):
        rings = np.where(counts >= 3, sums / np.maximum(counts, 1), np.nan)
    if not threshold or not np.isfinite(threshold):
        return dict.fromkeys(PROFILE_COLUMNS, float("nan")) | {"radial_dab_um": " ".join(f"{v:.4f}" for v in rings)}
    return {"radial_dab_um": " ".join(f"{v:.4f}" for v in rings), **profile_numbers(rings, threshold)}


def _nucleus_shape(labels: np.ndarray, bounds: tuple[slice, slice], label: int) -> tuple[float, float, float, float] | None:
    """(x, y, elongation, major-axis angle) of one labelled nucleus (field px); None under 5 px."""
    ys, xs = np.nonzero(labels[bounds] == label)
    if len(xs) < 5:
        return None
    values, vectors = np.linalg.eigh(np.cov(np.vstack([xs, ys]).astype(float)))
    elongation = float(np.sqrt(max(values[1], 1e-9) / max(values[0], 1e-9)))
    angle = float(np.arctan2(vectors[1, 1], vectors[0, 1]))
    return float(bounds[1].start + xs.mean()), float(bounds[0].start + ys.mean()), elongation, angle


def _nuclei_near(field, box: tuple[int, int, int, int], cache: dict) -> list[tuple[float, float, float, float]]:
    """The shapes of the nuclei with a pixel in the box (x0, y0, x1, y1), each measured once per field."""
    labels = field.nuclei
    x0, y0, x1, y1 = box
    if "nucleus_bounds" not in cache:
        cache["nucleus_bounds"] = ndimage.find_objects(labels)
    bounds = cache["nucleus_bounds"]
    out = []
    for label in np.unique(labels[y0:y1, x0:x1]):
        if label == 0 or label > len(bounds) or bounds[label - 1] is None:
            continue
        if label not in cache:
            cache[label] = _nucleus_shape(labels, bounds[label - 1], int(label))
        if cache[label] is not None:
            out.append(cache[label])
    return out


def _empty(field) -> np.ndarray:
    """Empty space at the core's glass level, holes kept (the tissue masks fill holes up to ~10,000 µm², lumens among
    them): pixels whose darkest channel is lighter than the tissue cut, opened by one pixel."""
    from stainid.imaging.tissue import GLASS_MARGIN, TISSUE_DARKNESS  # noqa: PLC0415

    cut = TISSUE_DARKNESS if field.glass is None else max(TISSUE_DARKNESS, field.glass + GLASS_MARGIN)
    lightest_dark = 255.0 * np.exp(-cut)  # -ln(channel / 255) < cut on the 8-bit values (so pure black is never empty)
    rgb = field.rgb
    darkest = np.minimum(np.minimum(rgb[..., 0], rgb[..., 1]), rgb[..., 2])  # twice as fast as min(axis=2)
    empty = (darkest > lightest_dark).astype(np.uint8)
    return cv2.morphologyEx(empty, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def vessels(field, stain: str) -> list[dict]:
    """The lumen candidates of the tile's own area (centre in it), each with its vessel call (see the module's docstring)."""
    ps = field.pixel_size_um
    empty = _empty(field)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(empty, connectivity=8)
    if count <= 1:
        return []
    h, w = field.shape
    pixel_area = ps * ps
    lo, hi = LUMEN_UM2[0] / pixel_area, LUMEN_UM2[1] / pixel_area
    sizes = stats[1:, cv2.CC_STAT_AREA]
    if not ((sizes >= lo) & (sizes <= hi)).any():
        return []
    cache: dict = {}
    lining_px, wall_px = max(1, round(LINING_UM / ps)), max(1, round(WALL_UM / ps))
    rows = []
    for index in range(1, count):
        x, y, bw, bh, area = stats[index]
        if not lo <= area <= hi or x == 0 or y == 0 or x + bw >= w or y + bh >= h:
            continue
        cx, cy = centroids[index]
        if not field.inside(cx, cy):
            continue
        pad = lining_px + wall_px + 2
        bx0, by0, bx1, by1 = max(0, x - pad), max(0, y - pad), min(w, x + bw + pad), min(h, y + bh + pad)
        lumen = labels[by0:by1, bx0:bx1] == index
        outside = cv2.distanceTransform((~lumen).astype(np.uint8), cv2.DIST_L2, 5)
        contours, hierarchy = cv2.findContours(lumen.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        outer = [i for i in range(len(contours)) if hierarchy[0][i][3] < 0]
        contour = contours[max(outer, key=lambda i: cv2.contourArea(contours[i]))]
        island = np.zeros(lumen.shape, np.uint8)  # tissue the space surrounds: a vessel in its perivascular space (or a cell in its halo)
        for i in range(len(contours)):
            if hierarchy[0][i][3] >= 0 and cv2.contourArea(contours[i]) * pixel_area >= 30:
                cv2.drawContours(island, contours, i, 1, -1)
        perimeter = float(cv2.arcLength(contour, True)) * ps
        hull = float(cv2.contourArea(cv2.convexHull(contour)))
        solidity = float(cv2.contourArea(contour) / hull) if hull else 0.0
        ys, xs = np.nonzero(lumen)
        values = np.linalg.eigvalsh(np.cov(np.vstack([xs, ys]).astype(float))) if len(xs) >= 3 else np.array([1.0, 1.0])
        elongation = float(np.sqrt(max(values[1], 1e-9) / max(values[0], 1e-9)))
        lining = along = in_island = 0
        nuclei = _nuclei_near(field, (bx0, by0, bx1, by1), cache) if field.raw_nuclei is not None else []
        if nuclei:
            gy, gx = np.gradient(outside)
            for nx, ny, nucleus_elongation, angle in nuclei:
                ix, iy = int(nx) - bx0, int(ny) - by0
                if not (0 <= ix < bx1 - bx0 and 0 <= iy < by1 - by0):
                    continue
                if island[iy, ix]:
                    in_island += 1
                    continue
                distance = outside[iy, ix]
                if not 0 < distance <= lining_px:
                    continue
                lining += 1
                normal = np.arctan2(gy[iy, ix], gx[iy, ix])
                across = abs(np.sin(angle - normal))  # 1 when the nucleus lies along the wall (across the normal)
                if nucleus_elongation >= ELONGATED and across >= np.cos(np.deg2rad(30)):
                    along += 1
        rim = (outside > 0) & (outside <= wall_px) & field.tissue[by0:by1, bx0:bx1]
        hematoxylin, dab = field.hematoxylin[by0:by1, bx0:bx1], field.dab[by0:by1, bx0:bx1]
        positive = rim & (dab >= field.threshold) if np.isfinite(field.threshold) else np.zeros(rim.shape, bool)
        coverage = float("nan")
        if rim.any():
            ry, rx = np.nonzero(rim)
            sector = ((np.arctan2(ry + by0 - cy, rx + bx0 - cx) + np.pi) / (2 * np.pi) * SECTORS).astype(int) % SECTORS
            totals = np.bincount(sector, minlength=SECTORS)
            hits = np.bincount(sector, weights=positive[ry, rx].astype(float), minlength=SECTORS)
            seen = totals > 0
            coverage = float(((hits[seen] / totals[seen]) >= 0.1).mean()) if seen.any() else float("nan")
        needed = max(MIN_LINING, int(perimeter // LINING_PER_UM))
        diameter = 2 * np.sqrt(area * pixel_area / np.pi)
        lined = solidity >= MIN_SOLIDITY and (along >= max(2, needed) or (along >= needed and elongation >= ELONGATED_LUMEN))
        is_vessel = (lined and diameter >= MIN_VESSEL_UM) or (diameter >= ROUND_VESSEL_UM and solidity >= ROUND_SOLIDITY)
        is_vessel = is_vessel or island.sum() * pixel_area >= ISLAND_UM2  # a vessel in its perivascular space
        ox, oy = field.origin
        points = (contour[:, 0, :] + [bx0 + ox, by0 + oy]).reshape(-1)
        rows.append({"tile_id": field.tile["tile_id"], "x_core": float(cx + ox), "y_core": float(cy + oy), "area_um2": float(area * pixel_area),
                     "equivalent_diameter_um": float(2 * np.sqrt(area * pixel_area / np.pi)), "elongation": elongation, "solidity": solidity,
                     "perimeter_um": perimeter, "lining_nuclei": lining, "lining_along_wall": along,
                     "rim_hematoxylin_od": float(hematoxylin[rim].mean()) if rim.any() else float("nan"),
                     "lumen_dab_od": float(dab[lumen].mean()),
                     "wall_positive_fraction": float(positive[rim].mean()) if rim.any() and stain == "6E10" else float("nan"),
                     "wall_coverage": coverage if stain == "6E10" else float("nan"),
                     "island_nuclei": in_island, "vessel": bool(is_vessel),
                     "outline": " ".join(str(int(v)) for v in points)})
    return rows


VESSEL_BITS = {"vessel": 1, "wall": 2, "lumen": 4}  # vessels/<stain>/<core_id>.png: a vessel's lumen, its wall band, any other lumen


def vessel_mask(table: pd.DataFrame, shape: tuple[int, int], scale: int, pixel_size_um: float | None = None) -> np.ndarray:
    """The vessel picture on the masks' grid (1/`scale` of the core image) from a vessel table: `VESSEL_BITS`."""
    out = np.zeros(shape, np.uint8)
    lumen, vessel = np.zeros(shape, np.uint8), np.zeros(shape, np.uint8)
    for row in table.itertuples():
        values = np.asarray(str(row.outline).split(), float)
        if len(values) < 6:
            continue
        points = np.rint(values.reshape(-1, 1, 2) / scale).astype(np.int32)
        cv2.fillPoly(vessel if bool(row.vessel) else lumen, [points], 1)
    wall_px = max(1, round(WALL_UM / ((pixel_size_um or 0.2738) * scale)))
    wall = cv2.dilate(vessel, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * wall_px + 1, 2 * wall_px + 1))) & (1 - vessel)
    out |= vessel * VESSEL_BITS["vessel"]
    out |= wall * VESSEL_BITS["wall"]
    out |= (lumen & (1 - vessel)) * VESSEL_BITS["lumen"]
    return out


def write_vessels(root, stain: str, core_id: str, rows: list[dict], shape: tuple[int, int], scale: int) -> None:
    """`vessels/<stain>/<core_id>` (table, the store) and the picture drawn from it (`vessel_mask`)."""
    from stainid import candidates as store  # noqa: PLC0415

    table = pd.DataFrame(rows, columns=list(VESSEL_COLUMNS))
    table.insert(0, "vessel_id", [f"{stain}:{core_id}:v{i + 1}" for i in range(len(table))])
    folder = root / "vessels" / stain
    folder.mkdir(parents=True, exist_ok=True)
    store.write_table(folder / f"{core_id}.csv.gz", table)
    target = folder / f"{core_id}.png"
    temporary = target.with_name(f".{target.stem}.tmp.png")
    cv2.imwrite(str(temporary), vessel_mask(table, shape, scale))
    temporary.replace(target)


def read_vessels(project, stain: str, core_id: str) -> pd.DataFrame | None:
    """A core image's lumen candidates (stage 2); None when stage 2 ran before vessels were looked for."""
    from stainid import candidates as store  # noqa: PLC0415

    folder = store.root(project) / "vessels" / stain
    for name in (f"{core_id}.parquet", f"{core_id}.csv.gz"):
        if (folder / name).exists():
            return store.read_table(folder / name)
    return None


def candidate_columns(stain: str, field, regions: list) -> list[dict]:
    """The columns of this module for each kept candidate of a tile."""
    if stain == "6E10":
        return [radial_profile(field, region) for region in regions]
    return [soma_nucleus(field, region) for region in regions]
