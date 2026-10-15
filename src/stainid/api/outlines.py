"""Segment Anything outlines of one whole core image, for the core viewer.

The "Outline objects" step (`stainid masks`) writes, per core image, `<masks>/<stain>/parts/<core>_<stain>_objects.csv`
(one row per detection it was prompted with: the prompt point `x`, `y` and, when the outline was kept, its `label`) and,
per tile, `<masks>/<stain>/labels/<tile>.npz` (a label image of the tile's context crop). Both are in field-crop pixels,
whose origin in the core image is `render.crop_origin` of the tile, as for the detections. The csv is written last, so
it marks a finished core. Outlines are turned into simplified polygons in core-image pixels once and cached in
`data/cache/core_outlines/`, rebuilt when the csv changes.
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage

from stainid import disk_cache
from stainid.api.render import crop_origin
from stainid.api.state import ProjectState
from stainid.tables import read_csv

CACHE_VERSION = 2
EPSILON_PX = 1.0  # polygon simplification at full resolution
MIN_PIECE_PX = 9.0  # specks of an outline smaller than ~4 x 4 px are left out (the largest piece always stays)


def parts_path(state: ProjectState, core_id: str, stain: str) -> Path:
    return state.project.output("masks") / stain / "parts" / f"{core_id}_{stain}_objects.csv"


def label_rings(labels: np.ndarray, epsilon: float = EPSILON_PX) -> dict[int, list[list[int]]]:
    """Every label's outline as simplified polygons (one per separate piece, specks left out), each a flat
    [x0, y0, x1, y1, …] of pixel indices (the polygon runs through the centres of the object's edge pixels)."""
    rings: dict[int, list[list[int]]] = {}
    for index, box in enumerate(ndimage.find_objects(labels)):
        if box is None:
            continue
        piece = np.pad((labels[box] == index + 1).astype(np.uint8), 1)
        contours, _ = cv2.findContours(piece, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE, offset=(box[1].start - 1, box[0].start - 1))
        areas = [cv2.contourArea(c) for c in contours]
        largest = int(np.argmax(areas))
        kept = [cv2.approxPolyDP(c, epsilon, True)[:, 0, :].ravel().tolist()
                for k, (c, a) in enumerate(zip(contours, areas)) if k == largest or a >= MIN_PIECE_PX]
        kept = [ring for ring in kept if len(ring) >= 6]  # a polygon needs three corners
        if kept:
            rings[index + 1] = kept
    return rings


def _build(state: ProjectState, parts: Path, tiles: pd.DataFrame) -> list[dict]:
    origins = {t["tile_id"]: crop_origin(t, state.project.context_px) for t in tiles.to_dict("records")}
    kept: dict[str, list[dict]] = {}
    for row in read_csv(parts):
        if row.get("mask_status") == "ok" and row.get("label") and row.get("tile_id") in origins:
            kept.setdefault(row["tile_id"], []).append(row)
    out = []
    for tile_id, rows in kept.items():
        try:
            labels = np.load(parts.parent.parent / "labels" / f"{tile_id}.npz")["labels"]
        except (FileNotFoundError, OSError, KeyError, ValueError, zipfile.BadZipFile):
            continue  # a tile's label image missing or being replaced: its objects are left out
        rings = label_rings(labels)
        x0, y0 = origins[tile_id]
        for row in rows:
            pieces = rings.get(int(float(row["label"])))
            if not pieces:
                continue
            area = float(row["area_um2"]) if row.get("area_um2") else None
            out.append({"x": float(row["x"]) + x0, "y": float(row["y"]) + y0, "area_um2": None if area is None else round(area, 1),
                        "rings": [[v + (x0 if i % 2 == 0 else y0) for i, v in enumerate(ring)] for ring in pieces]})
    return out


def core_outlines(state: ProjectState, core_id: str, stain: str, tiles: pd.DataFrame) -> list[dict] | None:
    """The outlines of one core image: per object its prompt point `x`, `y` (core pixels, the detection it belongs to),
    its outlined `area_um2` and `rings` (polygons in core pixels). None when the step has not outlined this core."""
    parts = parts_path(state, core_id, stain)
    try:
        stat = parts.stat()
    except FileNotFoundError:
        return None
    key = [CACHE_VERSION, stat.st_mtime_ns, stat.st_size]
    target = state.cache_dir("core_outlines") / f"{core_id}_{stain}.json"
    try:
        cached = json.loads(target.read_text())
        if cached.get("source") == key:
            disk_cache.used(target)
            return cached["objects"]
    except (FileNotFoundError, ValueError, KeyError, AttributeError):
        pass
    objects = _build(state, parts, tiles)
    disk_cache.write(target, json.dumps({"source": key, "objects": objects}, separators=(",", ":")).encode())
    return objects
