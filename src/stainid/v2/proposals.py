"""Stage 2, proposals: every candidate of one core image and stain, with its outline, sources and every feature, plus the
per-tile summaries, the AT8 thread skeleton, the stained regions no candidate covers and a low-resolution mask picture.

The core image is read once; each tile is cut out with its context margin (objects at a tile edge are segmented whole)
and keeps the candidates whose centroid lies in it or within half the margin of it; an object segmented by two or more
tiles is then kept once (`owned_once`, the most complete copy) and belongs to the tile holding its centroid. Cellpose
nuclei come from Find nuclei (never re-run here). Nothing here applies a model: stage 4 decides from the stored features.
"""
from __future__ import annotations

import gc
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial import cKDTree

from stainid import candidates as store
from stainid.imaging.core_images import open_image
from stainid.imaging.shared import shared_pictures
from stainid.imaging.tissue import core_glass_darkness
from stainid.nuclei import nucleus_centroids
from stainid.pipelines.cohort_v1 import context_crop
from stainid.v2 import amyloid, manifest, neun, tau
from stainid.v2.context import context_features_many
from stainid.v2.field import Field
from stainid.v2.geometry import Region, components, disk, outline
from stainid.v2.measure import generic_features
from stainid.v2.params import digest, params, stage_params
from stainid.v2.threads import segments
from stainid.v2.tree import stained_level

Image.MAX_IMAGE_PIXELS = None
TILE_KEYS = ("tile_id", "core_id", "tma", "donor_id", "sample_region_id", "region", "disease_group", "technical_replicate", "stain",
             "selection_order", "x_px", "y_px", "width_px", "height_px", "pixel_width_um", "pixel_height_um", "preview_tissue_fraction")
MASK_SCALE = 8
BITS = {"tissue": 1, "linear": 2, "fold": 4, "manual": 8, "edge_guard": 16, "tile": 32}
LEVELS = (0.5, 0.75, 1.0, 1.5, 2.0)  # DAB-positive areas at these multiples of the threshold, per tile


# ---------------------------------------------------------------- one tile

def _centroid(region: Region) -> tuple[float, float]:
    row = region.legacy or {}
    if "centroid_x_px" in row:
        return float(row["centroid_x_px"]), float(row["centroid_y_px"])
    return region.centroid()


def _model_inputs(stain: str, field: Field, regions: list[Region]) -> list[dict]:
    if not regions:
        return []
    if stain == "NeuN":
        contexts = context_features_many(field, [_centroid(r) for r in regions], neun.context_diameters(regions), 55.0)
        return neun.model_inputs(field, regions, contexts)
    if stain == "AT8":
        contexts = context_features_many(field, [_centroid(r) for r in regions], tau.context_diameters(regions), 55.0)
        return tau.model_inputs(regions, contexts, field.threshold)
    return amyloid.model_inputs(field, regions)


def _extras(stain: str, field: Field, regions: list[Region]) -> list[dict]:
    if stain == "6E10":
        return amyloid.extras(field, regions)
    if stain == "AT8":
        return tau.extras(field, regions)
    return [{} for _ in regions]


def _legacy_columns(stain: str, region: Region) -> dict:
    row = region.legacy or {}
    flags = {f"flag_{k}": v for k, v in region.flags.items()}
    if stain == "NeuN":
        return {"legacy_class": row.get("candidate_class", ""), "legacy_source_class": row.get("source_candidate_class", ""), **flags}
    if stain == "6E10":
        return {"legacy_class": row.get("candidate_class", ""), **flags}
    return {"legacy_class": row.get("candidate_source", ""), **flags}


UNPROPOSED = {"smooth_um": 0.0, "level": 1.0, "contrast": 0.0, "min_um2": 20.0, "min_cover": 0.0}


def unproposed_regions(field: Field, stain: str, owner: np.ndarray, rule: dict | float, thread_line: np.ndarray | None,
                       exclusion: np.ndarray) -> list[dict]:
    """Stained components no candidate proposes: the DAB smoothed over `smooth_um` (0: the field's own smoothing) at or
    above `stained_level`, in tissue, at least `min_um2`, centroid in the tile, that no candidate touches (`min_cover` 0)
    or that no single candidate covers `min_cover` of (6E10: a pale diffuse plaque that only small pieces touch counts).
    `owner` labels each pixel with the candidate (1, 2, …) painted there first; `rule` may be just the minimum area."""
    rule = {**UNPROPOSED, **({"min_um2": rule} if isinstance(rule, (int, float)) else rule)}
    ps = field.pixel_size_um
    smooth = field.smooth if not float(rule["smooth_um"]) else cv2.GaussianBlur(field.dab, (0, 0), float(rule["smooth_um"]) / ps)
    level = stained_level(field, smooth, float(rule["level"]), float(rule["contrast"]), field.tissue & ~exclusion)
    positive = cv2.morphologyEx(((smooth >= level) & field.tissue & ~exclusion).astype(np.uint8), cv2.MORPH_CLOSE, disk(1))
    min_cover = float(rule["min_cover"])
    covered = cv2.dilate((owner > 0).astype(np.uint8), disk(max(1, round(1.0 / ps)))).astype(bool) if not min_cover else None
    out = []
    minimum_px = max(1, round(float(rule["min_um2"]) / ps ** 2))
    for box, local, area in components(positive.astype(bool)):
        if area < minimum_px:
            continue
        if covered is not None:
            if covered[box][local].any():
                continue
        else:
            held = owner[box][local]
            held = held[held > 0]
            if held.size and np.bincount(held).max() >= min_cover * area:
                continue
        ys, xs = np.nonzero(local)
        cx, cy = box[1].start + xs.mean(), box[0].start + ys.mean()
        if not field.inside(cx, cy):
            continue
        values = field.dab[box][local]
        region = Region(box[0].start, box[1].start, local, "unproposed")
        out.append({"x_field": cx, "y_field": cy, "area_um2": float(area * ps * ps), "mean_dab_od": float(values.mean()),
                    "p90_dab_od": float(np.quantile(values, 0.9)), "dab_over_threshold": float(values.mean() / field.threshold),
                    "excluded_fraction": float(exclusion[box][local].mean()),
                    "thread_fraction": float(thread_line[box][local].mean()) if thread_line is not None else np.nan,
                    "outline": store.encode_points(outline(region, field.origin))})
    return out


def tile_summary(field: Field, stain: str, kept: list[dict], unproposed: int) -> dict:
    """The tile's measures. Areas are counts of the tile's cells on the excluded-areas grid (`counted_cells`), and every
    stain or nucleus count is taken inside the counted cells (and outside the excluded pixels)."""
    from stainid import exclusions  # noqa: PLC0415

    ps = field.pixel_size_um
    mm2 = ps * ps / 1e6
    cell_mm2 = MASK_SCALE * MASK_SCALE * mm2
    inner = field.inner
    exclusion = field.neun_exclusion if stain == "NeuN" else field.exclusion
    codes, counted = counted_cells(field, stain)
    valid = field.tissue & ~exclusion
    valid[inner] &= counted
    valid[: inner[0].start] = False
    valid[inner[0].stop:] = False
    valid[:, : inner[1].start] = False
    valid[:, inner[1].stop:] = False
    area = float((codes == exclusions.COUNTED).sum() * cell_mm2)
    out = {k: field.tile.get(k, "") for k in TILE_KEYS}
    out |= {
        "threshold_dab_od": field.threshold, "glass_darkness": field.glass, "nuclei_available": field.raw_nuclei is not None,
        "tissue_area_mm2": float(((codes != 3) & (codes != exclusions.OUTSIDE)).sum() * cell_mm2),
        "analysis_area_mm2": area,
        "valid_area_mm2": area,  # the same counted cells (the far-area measure's denominator)
        "linear_artifact_fraction": float(field.linear[inner].mean()), "fold_fraction": float(field.fold[inner].mean()),
        "manual_exclusion_fraction": float(field.manual[inner].mean()),
        "exclusion_fraction": float(np.isin(codes, (1, 2, 4, 5)).mean()),  # of the tile's cells
        "candidate_count": len(kept), "unproposed_count": unproposed,
    }
    for level in LEVELS:
        out[f"dab_area_mm2_at_{level:g}"] = float(((field.smooth >= level * field.threshold) & valid)[inner].sum() * mm2)
    if stain in ("6E10", "AT8"):
        points = nucleus_centroids(field.nuclei, valid, ps)
        inside = [(x, y) for x, y in points if field.inside(x, y)]
        out["nucleus_count"] = len(inside)
    if stain == "NeuN":
        tissue = np.zeros(field.shape, bool)
        tissue[inner] = counted  # the counted cells, as every area
        size = max(1, round(100.0 / field.pixel_width_um)), max(1, round(100.0 / field.pixel_height_um))
        windows = []
        for y0 in range(inner[0].start, inner[0].stop - size[1] + 1, size[1]):
            for x0 in range(inner[1].start, inner[1].stop - size[0] + 1, size[0]):
                count = int(tissue[y0: y0 + size[1], x0: x0 + size[0]].sum())
                if count >= 0.5 * size[0] * size[1]:
                    windows.append((x0 + field.origin[0], y0 + field.origin[1], count))
        out["density_window_px"] = f"{size[0]} {size[1]}"
        out["density_windows"] = " ".join(f"{x} {y} {c}" for x, y, c in windows)
    if stain == "6E10":
        levels = amyloid._levels(field)
        out |= {"seed_threshold_dab_od": levels["seed_threshold"], "extent_threshold_dab_od": levels["extent_threshold"],
                "background_median_dab_od": levels["background_median"]}
        extent = field.memo.get("amyloid_extent")
        out["amyloid_extent_area_mm2"] = float((extent & valid)[inner].sum() * mm2) if extent is not None else np.nan
    if stain == "AT8":
        found = tau.maps(field)
        maps_, threads = found["maps"], found["threads"]
        inner_valid = maps_["valid"][inner] & counted
        skeleton = threads["skeleton"][inner] & inner_valid
        summary = found["summary"]
        out |= {
            "at8_valid_area_mm2": float(inner_valid.sum() * mm2),
            "at8_positive_area_mm2": float((maps_["positive"][inner] & inner_valid).sum() * mm2),
            "at8_noncompact_area_mm2": float((maps_["thread"][inner] & inner_valid).sum() * mm2),
            "at8_compact_profile_count": sum(o["source_kind"] == "compact_profile" and field.inside(o["centroid_x_px"], o["centroid_y_px"])
                                             for o in found["objects"]),
            "at8_legacy_skeleton_px": int(maps_["thread_skeleton"][inner].sum()),
            "thread_line_area_mm2": float((threads["line"][inner] & inner_valid).sum() * mm2),
            "thread_skeleton_px": int(skeleton.sum()),
            "thread_width_sum_um": float(threads["width_um"][inner][skeleton].sum()),
            "thread_branchpoint_count": int((threads["branchpoints"][inner] & inner_valid).sum()),
            "thread_endpoint_count": int((threads["endpoints"][inner] & inner_valid).sum()),
            "edge_guard_triggered": bool(summary["edge_guard_triggered"]),
            "edge_exclusion_fraction": float(maps_["edge_excluded"][inner].mean()),
            "hole_rim_removed_fraction": float(maps_["hole_rim_removed"][inner][inner_valid].mean()) if inner_valid.any() else np.nan,
        }
    return out


def process_tile(field: Field, stain: str, settings: dict, shared: dict) -> dict:
    """Everything stage 2 keeps of one tile: candidate rows (field coordinates), summary, unproposed regions, threads."""
    module = {"NeuN": neun, "6E10": amyloid, "AT8": tau}[stain]
    regions = module.propose(field, settings, shared)
    if stain == "6E10":
        extent = np.zeros(field.shape, bool)
        for region in regions:
            if region.source == "legacy":
                extent[region.box] |= region.mask
        field.memo["amyloid_extent"] = extent
    # an object across the tile's edge is segmented from each side's crop, with slightly different centroids: keep it
    # here when its centroid lies within half the context margin of the tile (`owned_once` keeps one copy per core)
    margin = own_margin(field)
    kept = [r for r in regions if field.inside(*_centroid(r), margin)]
    exclusion = field.neun_exclusion if stain == "NeuN" else field.exclusion
    rows = []
    inputs = _model_inputs(stain, field, kept)
    extras = _extras(stain, field, kept)
    ps = field.pixel_size_um
    ox, oy = field.origin
    tx, ty = int(field.tile["x_px"]), int(field.tile["y_px"])
    for region, model_values, extra in zip(kept, inputs, extras):
        cx, cy = _centroid(region)
        generic = generic_features(field, region)
        rows.append({
            "tile_id": field.tile["tile_id"], "x_core": cx + ox, "y_core": cy + oy, "x_tile": cx + ox - tx, "y_tile": cy + oy - ty,
            "x_field": cx + ox - max(0, tx - int(field.tile.get("context_px", 128))), "y_field": cy + oy - max(0, ty - int(field.tile.get("context_px", 128))),
            "source": region.source, "sources": "|".join(region.sources), "n_sources": len(region.sources),
            "proposal_threshold": region.level if region.level is not None else np.nan,
            "radius_px": float(np.sqrt(region.area_px / np.pi)),
            "bbox": f"{region.x0 + ox} {region.y0 + oy} {region.x0 + region.mask.shape[1] + ox} {region.y0 + region.mask.shape[0] + oy}",
            "outline": store.encode_points(outline(region, field.origin)),
            "excluded_fraction": float(exclusion[region.box][region.mask].mean()),
            **generic, **_legacy_columns(stain, region), **extra, **model_values,
            "_own": field.inside(cx, cy), "_area_px": region.area_px,
        })
    owner = np.zeros(field.shape, np.int32)
    for index, region in enumerate(regions, start=1):
        view = owner[region.box]
        view[region.mask & (view == 0)] = index
    line = tau.maps(field)["threads"]["line"] if stain == "AT8" else None
    unproposed = unproposed_regions(field, stain, owner, unproposed_rule(settings, shared, stain), line, exclusion)
    for row in unproposed:
        row |= {"tile_id": field.tile["tile_id"], "x_core": row.pop("x_field") + ox, "y_core": row.pop("y_field") + oy}
    thread_rows = []
    if stain == "AT8":
        found = tau.maps(field)
        inner_mask = np.zeros(field.shape, bool)
        inner_mask[field.inner] = True
        skeleton = found["threads"]["skeleton"] & found["maps"]["valid"] & inner_mask
        owner = np.zeros(field.shape, np.int32)
        owners = []
        for index, region in enumerate(regions, start=1):
            view = owner[region.box]
            view[region.mask & (view == 0)] = index
            owners.append(_centroid(region))
        for segment in segments(skeleton, found["threads"]["width_um"], owner, ps):
            at = owners[segment["owner"] - 1] if segment["owner"] else None
            thread_rows.append({"tile_id": field.tile["tile_id"], "kind": segment["kind"], "n_px": segment["n_px"],
                                "length_um": segment["length_um"], "width_um": segment["width_um"], "width_sum_um": segment["width_sum_um"],
                                "owner_x": at[0] + ox if at else np.nan, "owner_y": at[1] + oy if at else np.nan,
                                "points": store.encode_points(segment["points"] + np.array([ox, oy]))})
    return {"rows": rows, "summary": tile_summary(field, stain, rows, len(unproposed)), "unproposed": unproposed, "threads": thread_rows}


def unproposed_rule(settings: dict, shared: dict, stain: str) -> dict:
    """The stain's unproposed search (`unproposed` in its proposal settings; an older `unproposed_min_um2` sets the area)."""
    rule = dict(settings.get("unproposed") or {})
    if "min_um2" not in rule and "unproposed_min_um2" in shared:
        rule["min_um2"] = float(shared["unproposed_min_um2"][stain])
    return {**UNPROPOSED, **rule}


def own_margin(field: Field) -> int:
    """How far beyond its tile (pixels) a candidate's centroid may lie and still be measured in that tile: half the
    context margin, so the object reaches at least that far into the crop on the far side."""
    return int(field.tile.get("context_px", 128)) // 2


# ---------------------------------------------------------------- the detection window

def detection_context(project, settings: dict) -> int:
    """How far (core px) beyond its tile a tile's detection window reaches (`proposals.<stain>.context_px`, at least the
    project's context margin): as far as the largest object reaches from its centre, so that the tile owning an object's
    centre sees all of it."""
    return max(int(project.context_px), int(settings.get("context_px") or project.context_px))


def _crop_box(tile: dict, context: int, size: tuple[int, int]) -> tuple[int, int, int, int]:
    """The crop `context_crop` cuts for a tile with this context margin, as (x0, y0, x1, y1) core px."""
    x, y, w, h = int(tile["x_px"]), int(tile["y_px"]), int(tile["width_px"]), int(tile["height_px"])
    return max(0, x - context), max(0, y - context), min(size[1], x + w + context), min(size[0], y + h + context)


def _overlap(a: tuple, b: tuple) -> tuple[int, int, int, int] | None:
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None


def automatic_masks(image, tiles: list[dict], context: int, pixel_size: dict, size: tuple[int, int], workers: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Detection's linear-artifact and fold masks of the whole core image, each tile's own area from its standard crop
    (the tile and the project's context margin; the masks the calibration and the tile measures use): a wider detection
    window cuts them from here, so they do not depend on the window."""
    linear, fold = np.zeros(size, bool), np.zeros(size, bool)
    cutting = threading.Lock()

    def one(tile: dict) -> None:
        x, y = int(tile["x_px"]), int(tile["y_px"])
        with cutting:
            rgb, inner = context_crop(image, x, y, int(tile["width_px"]), int(tile["height_px"]), context)
        field = Field({**tile, "context_px": context}, rgb, inner, (x - inner[1].start, y - inner[0].start), float("nan"),
                      pixel_size[tile["tile_id"]], np.zeros(rgb.shape[:2], bool), None)
        with shared_pictures(rgb):
            h, w = inner[0].stop - inner[0].start, inner[1].stop - inner[1].start
            linear[y: y + h, x: x + w] = field.linear[inner]
            fold[y: y + h, x: x + w] = field.fold[inner]
        field.release()

    if workers > 1 and len(tiles) > 1:
        with ThreadPoolExecutor(min(workers, len(tiles)), thread_name_prefix="stainid-masks") as pool:
            list(pool.map(one, tiles))
    else:
        for tile in tiles:
            one(tile)
    return linear, fold


def window_nuclei(tile: dict, tiles: list[dict], window: tuple[int, int, int, int], context: int, size: tuple[int, int],
                  load) -> np.ndarray | None:
    """Cellpose labels over a detection window wider than the tile's nuclei crop: the tile's own nuclei (its standard
    crop, as Find nuclei made them) and beyond that the other tiles' nuclei of their own areas (relabelled); None when the
    tile has none. A nucleus across the seam of the tile's crop keeps the part on each side."""
    own = load(tile)
    if own is None:
        return None
    x0, y0, x1, y1 = window
    out = np.zeros((y1 - y0, x1 - x0), np.uint32)
    standard = _crop_box(tile, context, size)
    next_label = int(own.max()) + 1
    for other in tiles:
        if other["tile_id"] == tile["tile_id"]:
            continue
        mine = (int(other["x_px"]), int(other["y_px"]), int(other["x_px"]) + int(other["width_px"]), int(other["y_px"]) + int(other["height_px"]))
        part = _overlap(mine, window)
        if part is None:
            continue
        labels = load(other)
        if labels is None:
            continue
        ox0, oy0 = _crop_box(other, context, size)[:2]
        piece = labels[part[1] - oy0: part[3] - oy0, part[0] - ox0: part[2] - ox0].astype(np.uint32)
        piece = np.where(piece > 0, piece + next_label, 0).astype(np.uint32)
        next_label += int(labels.max()) + 1
        out[part[1] - y0: part[3] - y0, part[0] - x0: part[2] - x0] = piece
    sx0, sy0, sx1, sy1 = standard
    out[sy0 - y0: sy1 - y0, sx0 - x0: sx1 - x0] = own[: sy1 - sy0, : sx1 - sx0]
    return out



# ---------------------------------------------------------------- one copy per object across tiles

def _raster(row: dict, box: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = box
    mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
    flat = store.points(row["outline"])
    if len(flat) >= 6:
        cv2.fillPoly(mask, [np.asarray(flat, np.int32).reshape(-1, 2) - np.array([x0, y0])], 1)
    return mask.astype(bool)


def _box(row: dict) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = (int(v) for v in str(row["bbox"]).split())
    return x0, y0, x1, y1


def same_copy(a: dict, b: dict, same_iou: float = 0.3, same_inside: float = 0.5) -> bool:
    """Whether two candidate rows (from different tiles) are one object: their outlines overlap with IoU ≥ `same_iou`, or
    one lies `same_inside` inside the other."""
    ax0, ay0, ax1, ay1 = _box(a)
    bx0, by0, bx1, by1 = _box(b)
    if ax1 <= bx0 or bx1 <= ax0 or ay1 <= by0 or by1 <= ay0:
        return False
    box = (min(ax0, bx0), min(ay0, by0), max(ax1, bx1) + 1, max(ay1, by1) + 1)
    ma, mb = _raster(a, box), _raster(b, box)
    inter = int((ma & mb).sum())
    if not inter:
        return False
    sa, sb = int(ma.sum()), int(mb.sum())
    return inter / (sa + sb - inter) >= same_iou or inter >= same_inside * min(sa, sb)


def owned_once(rows: list[dict], tiles: list[dict], context: int = 128, cell: int = 512) -> list[dict]:
    """Each object of the core image once. A tile keeps the candidates whose centroid lies in it or within its margin
    (`own_margin`), so an object across a tile edge is kept by every tile that segmented it, each copy cut at that tile's
    crop. Of the copies (rows of different tiles whose outlines are the same object, `same_copy`) the most complete stays:
    not cut by its crop, then centroid in its own tile, then the largest, then the first tile, then position. Copies are
    found the same way when both are whole with their centroids in their own tiles: two tiles' windows can segment one
    object slightly differently, its centroid then lying on either side of their border (objects that only touch are
    never copies). A copy whose centroid lies in no tile of the image is dropped. The kept row moves to the tile holding its centroid (`tile_id`, `x_tile`, `x_field`…)."""
    order = {t["tile_id"]: i for i, t in enumerate(tiles)}

    def home(x: float, y: float) -> dict | None:
        for t in tiles:
            if int(t["x_px"]) <= x < int(t["x_px"]) + int(t["width_px"]) and int(t["y_px"]) <= y < int(t["y_px"]) + int(t["height_px"]):
                return t
        return None

    rank = sorted(range(len(rows)), key=lambda i: (bool(rows[i].get("touches_crop_edge")), not rows[i]["_own"], -rows[i]["_area_px"],
                                                   order.get(rows[i]["tile_id"], len(order)), rows[i]["y_core"], rows[i]["x_core"]))
    grid: dict[tuple[int, int], list[int]] = {}
    keep: list[int] = []
    for i in rank:
        row = rows[i]
        x0, y0, x1, y1 = _box(row)
        cells = [(cx, cy) for cx in range(x0 // cell, x1 // cell + 1) for cy in range(y0 // cell, y1 // cell + 1)]
        near = {j for c in cells for j in grid.get(c, ())}
        copy = False
        for j in near:
            other = rows[j]
            if other["tile_id"] == row["tile_id"]:
                continue
            if same_copy(row, other):
                copy = True
                break
        if copy or (not row["_own"] and home(row["x_core"], row["y_core"]) is None):
            continue
        keep.append(i)
        for c in cells:
            grid.setdefault(c, []).append(i)
    out = []
    for i in sorted(keep):
        row = rows[i]
        if not row["_own"]:
            tile = home(row["x_core"], row["y_core"])
            tx, ty = int(tile["x_px"]), int(tile["y_px"])
            row = {**row, "tile_id": tile["tile_id"], "x_tile": row["x_core"] - tx, "y_tile": row["y_core"] - ty,
                   "x_field": row["x_core"] - max(0, tx - context), "y_field": row["y_core"] - max(0, ty - context)}
        out.append({k: v for k, v in row.items() if k not in ("_own", "_area_px")})
    return out


def mask_piece(field: Field, stain: str) -> np.ndarray:
    """The tile's own area at 1/MASK_SCALE as bit flags (BITS), each cell flagged by the majority of its pixels: the grid
    the excluded-areas layer, the views and every area measure share (`counted_cells`)."""
    if "mask_piece" in field.memo:
        return field.memo["mask_piece"]
    inner = field.inner
    planes = [(field.tissue_area[inner], BITS["tissue"]), (field.linear[inner], BITS["linear"]), (field.fold[inner], BITS["fold"]),
              (field.manual[inner], BITS["manual"])]
    if stain == "AT8":
        planes.append((tau.maps(field)["maps"]["edge_excluded"][inner], BITS["edge_guard"]))
    height, width = planes[0][0].shape
    size = (max(1, width // MASK_SCALE), max(1, height // MASK_SCALE))
    bits = np.full((size[1], size[0]), BITS["tile"], np.uint8)
    for plane, bit in planes:
        if plane.any():
            share = cv2.resize(plane[: size[1] * MASK_SCALE, : size[0] * MASK_SCALE].astype(np.float32), size, interpolation=cv2.INTER_AREA) \
                if height >= MASK_SCALE and width >= MASK_SCALE else np.full((size[1], size[0]), float(plane.mean()), np.float32)
            bits[share >= 0.5] |= bit
    field.memo["mask_piece"] = bits
    return bits


def counted_cells(field: Field, stain: str) -> tuple[np.ndarray, np.ndarray]:
    """The tile's counted cells (`stainid.exclusions.codes_of` of its mask piece: tissue, not excluded) and the same at
    the pixels of the tile's own area (cells past the last whole one are not counted, as the mask has none there). Every
    area of the tile is a count of these cells, and every stain or nucleus count is taken inside them, so each density
    and burden divides by exactly the area the views show as counted."""
    from stainid import exclusions  # noqa: PLC0415

    codes = exclusions.codes_of(mask_piece(field, stain), stain)
    cells = codes == exclusions.COUNTED
    height, width = field.inner[0].stop - field.inner[0].start, field.inner[1].stop - field.inner[1].start
    pixels = np.zeros((height, width), bool)
    up = np.repeat(np.repeat(cells, MASK_SCALE, axis=0), MASK_SCALE, axis=1)
    pixels[: up.shape[0], : up.shape[1]] = up[:height, :width]
    return codes, pixels


# ---------------------------------------------------------------- one core image

def inputs_of(project, stain: str, core_id: str, tiles: list[dict], threshold: float, nuclei_names: frozenset) -> dict:
    # no exclusion input: the analysis leaves out detection's own areas only (folds, linear artifacts, no tissue, the AT8
    # edge guard), found here; people's marked areas only train and evaluate models (`stainid.exclusions.drop_excluded`)
    image = project.root / tiles[0]["image_path"]
    return {
        "image": manifest.file_signature(image),
        "threshold": round(float(threshold), 10),
        "tiles": digest([(t["tile_id"], t["x_px"], t["y_px"], t["width_px"], t["height_px"]) for t in tiles]),
        "nuclei_tiles": sum(f"{t['tile_id']}.npz" in nuclei_names for t in tiles),
    }


def expected(project, stain: str, core_id: str, tiles: list[dict], threshold: float, nuclei_names: frozenset,
             settings: dict | None = None) -> dict:
    """What stage 2's manifest of this core would be now (`settings`: the stain's `stage_params`, when known already)."""
    return manifest.make("proposals", stage_params(project, "proposals", stain) if settings is None else settings,
                         inputs_of(project, stain, core_id, tiles, threshold, nuclei_names))


def propose_core(project, stain: str, core_id: str, tiles: list[dict], threshold: float, nuclei_names: frozenset,
                 tiles_at_once: int = 1) -> dict:
    """Run stage 2 on one core image and write its outputs; returns counts and timing. `tiles_at_once` tiles are worked on
    in threads at once (they share the decoded core image; each needs about 0.7 GB more); the outputs keep the tile
    order and do not depend on it."""
    started = time.time()
    p = params(project)["proposals"]
    settings, shared = p[stain], {k: v for k, v in p.items() if k not in ("NeuN", "6E10", "AT8")}
    nuclei_dir = project.output("nuclei")
    context = project.context_px  # the tiles' nuclei crops and field pictures
    window = detection_context(project, settings)  # how far each tile's detection reaches (the largest object)
    pixel_size = {t["tile_id"]: float(np.sqrt(float(t["pixel_width_um"]) * float(t["pixel_height_um"]))) for t in tiles}
    rows, summaries, unproposed, thread_rows = [], [], [], []

    def load_nuclei(tile: dict) -> np.ndarray | None:
        nuclei_path = nuclei_dir / f"{tile['tile_id']}.npz"
        if f"{tile['tile_id']}.npz" in nuclei_names or nuclei_path.exists():
            with np.load(nuclei_path) as loaded:
                return loaded["masks"]
        return None

    with open_image(project.root / tiles[0]["image_path"]) as image:
        image.load()
        glass = core_glass_darkness(image)
        size = (image.height, image.width)
        mask = np.zeros(((size[0] + MASK_SCALE - 1) // MASK_SCALE, (size[1] + MASK_SCALE - 1) // MASK_SCALE), np.uint8)
        wide = window > context
        automatic = automatic_masks(image, tiles, context, pixel_size, size, tiles_at_once) if wide else None
        cutting = threading.Lock()

        def one(tile: dict) -> tuple[dict, np.ndarray]:
            x, y = int(tile["x_px"]), int(tile["y_px"])
            with cutting:
                rgb, inner = context_crop(image, x, y, int(tile["width_px"]), int(tile["height_px"]), window)
            origin = (x - inner[1].start, y - inner[0].start)
            box = (origin[0], origin[1], origin[0] + rgb.shape[1], origin[1] + rgb.shape[0])
            raw = window_nuclei(tile, tiles, box, context, size, load_nuclei) if wide else load_nuclei(tile)
            field = Field({**tile, "context_px": context}, rgb, inner, origin, threshold, pixel_size[tile["tile_id"]],
                          np.zeros(rgb.shape[:2], bool), raw, glass,
                          smooth_px=float(p["smooth_px"]), background_um=float(p["local_background_um"]))
            if automatic is not None:  # detection's masks as the standard crops make them (calibration, tile measures)
                field.__dict__["linear"] = automatic[0][box[1]: box[3], box[0]: box[2]]
                field.__dict__["fold"] = automatic[1][box[1]: box[3], box[0]: box[2]]
            with shared_pictures(rgb):  # the stain separation and tissue mask of the field, made once for every step
                result = process_tile(field, stain, settings, shared)
                piece = mask_piece(field, stain)
            field.release()
            del field, rgb, raw
            gc.collect()
            return result, piece

        if tiles_at_once > 1 and len(tiles) > 1:
            with ThreadPoolExecutor(min(tiles_at_once, len(tiles)), thread_name_prefix="stainid-tile") as pool:
                done = list(pool.map(one, tiles))
        else:
            done = map(one, tiles)
        for tile, (result, piece) in zip(tiles, done):
            rows += result["rows"]
            summaries.append(result["summary"])
            unproposed += result["unproposed"]
            thread_rows += result["threads"]
            my, mx = int(tile["y_px"]) // MASK_SCALE, int(tile["x_px"]) // MASK_SCALE
            h, w = min(piece.shape[0], mask.shape[0] - my), min(piece.shape[1], mask.shape[1] - mx)
            mask[my: my + h, mx: mx + w] = piece[:h, :w]
    rows = owned_once(rows, tiles, context)
    counts = pd.Series([r["tile_id"] for r in rows], dtype=object).value_counts()
    for summary in summaries:
        summary["candidate_count"] = int(counts.get(summary["tile_id"], 0))
    write_core(project, stain, core_id, tiles, rows, summaries, unproposed, thread_rows, mask)
    manifest.write(project, "proposals", stain, core_id, expected(project, stain, core_id, tiles, threshold, nuclei_names),
                   candidates=len(rows), unproposed=len(unproposed), seconds=round(time.time() - started, 1), tiles=len(tiles))
    return {"candidates": len(rows), "unproposed": len(unproposed), "seconds": time.time() - started}


def write_core(project, stain: str, core_id: str, tiles: list[dict], rows: list[dict], summaries: list[dict], unproposed: list[dict],
               thread_rows: list[dict], mask: np.ndarray) -> None:
    root = store.root(project)
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values(["tile_id", "y_core", "x_core"], kind="stable").reset_index(drop=True)
        frame.insert(0, "candidate_id", store.unique_ids(stain, core_id, frame.x_core, frame.y_core))
        frame.insert(1, "stain", stain)
        frame.insert(2, "core_id", core_id)
    else:
        frame = pd.DataFrame(columns=["candidate_id", "stain", "core_id", "tile_id", "x_core", "y_core", "outline"])
    store.write_table(root / "tiles" / stain / f"{core_id}.csv.gz", pd.DataFrame(summaries))
    regions = pd.DataFrame(unproposed, columns=["tile_id", "x_core", "y_core", "area_um2", "mean_dab_od", "p90_dab_od", "dab_over_threshold",
                                                "excluded_fraction", "thread_fraction", "outline"])
    regions.insert(0, "region_id", [f"{stain}:{core_id}:u{i + 1}" for i in range(len(regions))])
    store.write_table(root / "unproposed" / stain / f"{core_id}.csv.gz", regions)
    if stain == "AT8":
        threads = pd.DataFrame(thread_rows, columns=["tile_id", "kind", "n_px", "length_um", "width_um", "width_sum_um", "owner_x", "owner_y", "points"])
        threads.insert(1, "segment", np.arange(1, len(threads) + 1))
        ids = [""] * len(threads)
        if len(threads) and not frame.empty:
            owned = threads.owner_x.notna().to_numpy()
            if owned.any():
                tree = cKDTree(frame[["x_core", "y_core"]].to_numpy(float))
                distance, nearest = tree.query(threads.loc[owned, ["owner_x", "owner_y"]].to_numpy(float),
                                               distance_upper_bound=store.MATCH_RADIUS_UM["AT8"] / project.pixel_size_um)
                for position, d, n in zip(np.flatnonzero(owned), distance, nearest):
                    if np.isfinite(d):
                        ids[position] = frame.candidate_id.iloc[n]
        threads.insert(threads.columns.get_loc("width_sum_um") + 1, "candidate_id", ids)
        store.write_table(root / "threads" / f"{core_id}.csv.gz", threads.drop(columns=["owner_x", "owner_y"]))
    (root / "masks" / stain).mkdir(parents=True, exist_ok=True)
    target = root / "masks" / stain / f"{core_id}.png"
    temporary = target.with_name(f".{target.stem}.tmp.png")
    cv2.imwrite(str(temporary), mask)
    temporary.replace(target)
    store.write_table(root / "proposals" / stain / f"{core_id}.csv.gz", frame)
