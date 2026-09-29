"""Server-side image products for the viewer: core thumbnails, field crops, raster overlays and object outlines."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from stainid.api.state import ProjectState
from stainid.imaging.color import rgb_to_hed
from stainid.imaging.tissue import field_tissue_mask, fold_mask, linear_artifact_mask
from stainid.tables import read_csv

Image.MAX_IMAGE_PIXELS = None
FIELD_JPEG_QUALITY = 88


def crop_origin(tile: dict, context_px: int) -> tuple[int, int]:
    return max(0, int(tile["x_px"]) - context_px), max(0, int(tile["y_px"]) - context_px)


def core_thumbnail(state: ProjectState, core_id: str, stain: str, size: int = 512) -> Path:
    target = state.cache_dir("thumbs") / f"{core_id}_{stain}_{size}.jpg"
    if not target.exists():
        image_path = state.root / state.core_image(core_id, stain)["image_path"]
        bgr = cv2.imread(str(image_path), cv2.IMREAD_REDUCED_COLOR_8)
        if bgr is None:
            raise FileNotFoundError(image_path)
        scale = size / max(bgr.shape[:2])
        cv2.imwrite(str(target), cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return target


def field_rgb(state: ProjectState, tile_id: str) -> np.ndarray:
    """Field crop including the context halo, cached as JPEG (coordinates match all pipeline outputs)."""
    target = state.cache_dir("fields") / f"{tile_id}.jpg"
    if not target.exists():
        tile = state.tile(tile_id)
        x0, y0 = crop_origin(tile, state.project.context_px)
        x1 = int(tile["x_px"]) + int(tile["width_px"]) + state.project.context_px
        y1 = int(tile["y_px"]) + int(tile["height_px"]) + state.project.context_px
        with Image.open(state.root / tile["image_path"]) as image:
            crop = image.crop((x0, y0, min(x1, image.width), min(y1, image.height))).convert("RGB")
        crop.save(target, quality=FIELD_JPEG_QUALITY)
    return np.asarray(Image.open(target).convert("RGB"))


def field_jpeg(state: ProjectState, tile_id: str) -> Path:
    field_rgb(state, tile_id)
    return state.cache_dir("fields") / f"{tile_id}.jpg"


def _rgba_png(mask: np.ndarray, color: tuple[int, int, int], alpha: int, target: Path) -> Path:
    rgba = np.zeros((*mask.shape, 4), np.uint8)
    rgba[mask] = (*color, alpha)
    Image.fromarray(rgba).save(target)
    return target


def exclusion_overlay(state: ProjectState, tile_id: str) -> Path:
    target = state.cache_dir("overlays") / f"{tile_id}_exclusions.png"
    if not target.exists():
        rgb = field_rgb(state, tile_id)
        tile = state.tile(tile_id)
        mask = ~field_tissue_mask(rgb) | linear_artifact_mask(rgb)
        if tile["stain"] != "NeuN":
            mask |= fold_mask(rgb, state.project.pixel_size_um)
        _rgba_png(mask, (90, 110, 255), 110, target)
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
    return target


def mask_outlines(state: ProjectState, tile_id: str) -> list[dict]:
    """SAM object outlines as simplified polygons in field-crop coordinates."""
    tile = state.tile(tile_id)
    target = state.cache_dir("outlines") / f"{tile_id}.json"
    if target.exists():
        return json.loads(target.read_text())
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
    target.write_text(json.dumps(out))
    return out
