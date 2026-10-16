"""Server-side image products for the viewer: field crops, raster overlays and object outlines (core pictures: `pyramid`)."""
from __future__ import annotations

import io
import json
import threading
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from stainid import disk_cache
from stainid.api import pyramid
from stainid.api.state import ProjectState
from stainid.imaging.color import rgb_to_hed
from stainid.imaging.core_images import open_image
from stainid.imaging.tissue import field_tissue_mask, fold_mask, linear_artifact_mask
from stainid.qc.exclusions import rasterize_core_exclusions, read_manual_exclusions
from stainid.tables import read_csv

Image.MAX_IMAGE_PIXELS = None
FIELD_JPEG_QUALITY = 88
_IMAGE_LOCKS: dict[str, threading.Lock] = {}


def crop_origin(tile: dict, context_px: int) -> tuple[int, int]:
    return max(0, int(tile["x_px"]) - context_px), max(0, int(tile["y_px"]) - context_px)


def core_thumbnail(state: ProjectState, core_id: str, stain: str, size: int = 512) -> Path:
    """A JPEG of the whole core, at most `size` px across (see `stainid.api.pyramid`)."""
    return pyramid.thumbnail(state, core_id, stain, size)


def field_path(state: ProjectState, tile_id: str) -> Path:
    return state.cache_dir("fields") / f"{tile_id}.jpg"


def field_rgb(state: ProjectState, tile_id: str) -> np.ndarray:
    """Field crop including the context halo, cached as JPEG (coordinates match all pipeline outputs)."""
    target = field_path(state, tile_id)
    for attempt in (1, 2):  # a second time when it was cleaned up between the look and the read
        if not target.exists():
            tile = state.tile(tile_id)
            with _IMAGE_LOCKS.setdefault(tile["image_path"], threading.Lock()):
                if not target.exists():
                    _crop_field(state, tile, target)
        try:
            with Image.open(target) as image:
                rgb = np.asarray(image.convert("RGB"))
        except FileNotFoundError:
            if attempt == 2:
                raise
            continue
        disk_cache.used(target)
        return rgb


def _crop_field(state: ProjectState, tile: dict, target: Path) -> None:
    """Cut one tile (with its context margin) out of its core image."""
    context = state.project.context_px
    with pyramid.DECODE, open_image(state.root / tile["image_path"]) as image:
        x0, y0 = crop_origin(tile, context)
        x1 = min(int(tile["x_px"]) + int(tile["width_px"]) + context, image.width)
        y1 = min(int(tile["y_px"]) + int(tile["height_px"]) + context, image.height)
        part = image.crop((x0, y0, x1, y1)).convert("RGB")
    out = io.BytesIO()
    part.save(out, format="JPEG", quality=FIELD_JPEG_QUALITY)
    disk_cache.write(target, out.getvalue())


def field_jpeg(state: ProjectState, tile_id: str) -> Path:
    field_rgb(state, tile_id)
    return state.cache_dir("fields") / f"{tile_id}.jpg"


def _rgba_png(mask: np.ndarray, color: tuple[int, int, int], alpha: int, target: Path) -> Path:
    rgba = np.zeros((*mask.shape, 4), np.uint8)
    rgba[mask] = (*color, alpha)
    out = io.BytesIO()
    Image.fromarray(rgba).save(out, format="PNG")
    disk_cache.write(target, out.getvalue())
    return target


def exclusion_overlay(state: ProjectState, tile_id: str) -> Path:
    target = state.cache_dir("overlays") / f"{tile_id}_exclusions.png"
    if not target.exists():
        rgb = field_rgb(state, tile_id)
        tile = state.tile(tile_id)
        mask = ~field_tissue_mask(rgb) | linear_artifact_mask(rgb)
        if tile["stain"] != "NeuN":
            mask |= fold_mask(rgb, state.project.pixel_size_um)
        manual = read_manual_exclusions(state.project.input("manual_exclusions")).get(tile["image_path"], [])
        origin = [max(0, int(tile[k]) - state.project.context_px) for k in ("x_px", "y_px")]
        mask |= rasterize_core_exclusions(manual, origin[0], origin[1], mask.shape)
        _rgba_png(mask, (90, 110, 255), 110, target)
    disk_cache.used(target)
    return target


def dab_overlay(state: ProjectState, tile_id: str) -> Path:
    """Pixels above the slide's calibrated DAB threshold."""
    target = state.cache_dir("overlays") / f"{tile_id}_dab.png"
    if not target.exists():
        tile = state.tile(tile_id)
        cal = state.calibration()
        threshold = float(cal[(cal.tma == str(tile["tma"])) & (cal.stain == tile["stain"])].threshold_dab_od.iloc[0])
        rgb = field_rgb(state, tile_id)
        _rgba_png(np.maximum(rgb_to_hed(rgb)[..., 2], 0) >= threshold, (255, 40, 40), 120, target)
    disk_cache.used(target)
    return target


def thread_overlay(state: ProjectState, tile_id: str) -> Path:
    target = state.cache_dir("overlays") / f"{tile_id}_threads.png"
    if not target.exists():
        from stainid.nuclei import clean_nuclei
        from stainid.stains.tau.threads import thread_network

        tile = state.tile(tile_id)
        cal = state.calibration()
        threshold = float(cal[(cal.tma == str(tile["tma"])) & (cal.stain == "AT8")].threshold_dab_od.iloc[0])
        rgb = field_rgb(state, tile_id)
        nuclei_path = state.project.output("nuclei") / f"{tile_id}.npz"
        nuclei = clean_nuclei(np.load(nuclei_path)["masks"], rgb) if nuclei_path.exists() else None
        valid = field_tissue_mask(rgb) & ~linear_artifact_mask(rgb) & ~fold_mask(rgb, state.project.pixel_size_um)
        skeleton = thread_network(rgb, valid, threshold, state.project.pixel_size_um, nuclei=nuclei)["skeleton"]
        _rgba_png(cv2.dilate(skeleton.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool), (40, 110, 255), 235, target)
    disk_cache.used(target)
    return target


def mask_outlines(state: ProjectState, tile_id: str) -> list[dict]:
    """SAM object outlines as simplified polygons in field-crop coordinates."""
    tile = state.tile(tile_id)
    target = state.cache_dir("outlines") / f"{tile_id}.json"
    try:
        out = json.loads(target.read_text())
        disk_cache.used(target)
        return out
    except FileNotFoundError:
        pass
    labels_path = state.project.output("masks") / tile["stain"] / "labels" / f"{tile_id}.npz"
    if not labels_path.exists():
        return []
    labels = np.load(labels_path)["labels"]
    parts = state.project.output("masks") / tile["stain"] / "parts" / f"{tile['core_id']}_{tile['stain']}_objects.csv"
    meta = {int(r["label"]): r for r in read_csv(parts) if r.get("tile_id") == tile_id and r.get("mask_status") == "ok" and r.get("label")} if parts.exists() else {}
    out = []
    for label in np.unique(labels[labels > 0]):
        contours, _ = cv2.findContours((labels == label).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = cv2.approxPolyDP(max(contours, key=cv2.contourArea), 1.0, True)[:, 0, :]
        row = meta.get(int(label), {})
        out.append({"label": int(label), "points": contour.tolist(), "area_um2": float(row.get("area_um2") or "nan"),
                    "circularity": float(row.get("circularity") or "nan"), "seed_source": row.get("seed_source", "")})
    disk_cache.write(target, json.dumps(out).encode())
    return out
