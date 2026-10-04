"""Server-side image products for the viewer: core thumbnails, field crops, raster overlays and object outlines."""
from __future__ import annotations

import json
import threading
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from stainid.api.state import ProjectState
from stainid.imaging.color import rgb_to_hed
from stainid.imaging.tissue import field_tissue_mask, fold_mask, linear_artifact_mask
from stainid.qc.exclusions import rasterize_core_exclusions, read_manual_exclusions
from stainid.stains.neun.audit import crop_with_padding
from stainid.tables import read_csv

Image.MAX_IMAGE_PIXELS = None
FIELD_JPEG_QUALITY = 88
_IMAGE_LOCKS: dict[str, threading.Lock] = {}


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
        image_path = state.tile(tile_id)["image_path"]
        with _IMAGE_LOCKS.setdefault(image_path, threading.Lock()):
            if not target.exists():
                _crop_all_fields(state, image_path)
    return np.asarray(Image.open(target).convert("RGB"))


def _crop_all_fields(state: ProjectState, image_path: str) -> None:
    """Decoding a core image is the slow part, so cut out every field of that image in one pass."""
    context = state.project.context_px
    with Image.open(state.root / image_path) as image:
        image.load()
        for tile in state.tiles[state.tiles.image_path == image_path].to_dict("records"):
            x0, y0 = crop_origin(tile, context)
            x1 = min(int(tile["x_px"]) + int(tile["width_px"]) + context, image.width)
            y1 = min(int(tile["y_px"]) + int(tile["height_px"]) + context, image.height)
            image.crop((x0, y0, x1, y1)).convert("RGB").save(state.cache_dir("fields") / f"{tile['tile_id']}.jpg", quality=FIELD_JPEG_QUALITY)


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
        manual = read_manual_exclusions(state.project.input("manual_exclusions")).get(tile["image_path"], [])
        origin = [max(0, int(tile[k]) - state.project.context_px) for k in ("x_px", "y_px")]
        mask |= rasterize_core_exclusions(manual, origin[0], origin[1], mask.shape)
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


def object_crop(state: ProjectState, stain: str, tile_id: str, x: float, y: float, half: int) -> np.ndarray:
    """A square crop around a labelled object, wherever its image lives: an analysed field, an earlier study's pilot
    field (annotations/fields/<id>.png) or a whole core image (labels made on core coordinates). Cached on disk."""
    target = state.cache_dir("object_crops") / f"{tile_id}_{round(x)}_{round(y)}_{half}.jpg"
    if target.exists():
        return np.asarray(Image.open(target).convert("RGB"))
    pilot = state.project.output("reviews") / "fields" / f"{tile_id}.png"
    if (state.tiles.tile_id == tile_id).any():
        crop = crop_with_padding(field_rgb(state, tile_id), x, y, 2 * half)
    elif pilot.exists():
        crop = crop_with_padding(np.asarray(Image.open(pilot).convert("RGB")), x, y, 2 * half)
    else:
        with Image.open(state.root / state.core_image(tile_id, stain)["image_path"]) as image:
            crop = np.asarray(image.crop((int(x) - half, int(y) - half, int(x) + half, int(y) + half)).convert("RGB"))
    Image.fromarray(crop).save(target, quality=92)
    return crop


CORE_TILE = 1024


def _core_tiles(state: ProjectState, core_id: str, stain: str) -> Path:
    """The core image cut once into 1024 px JPEG tiles, so any region can be served without decoding the whole scan."""
    folder = state.cache_dir("core_tiles") / f"{core_id}_{stain}"
    if (folder / "size.json").exists():
        return folder
    image_path = state.core_image(core_id, stain)["image_path"]
    with _IMAGE_LOCKS.setdefault(image_path, threading.Lock()):
        if not (folder / "size.json").exists():
            folder.mkdir(parents=True, exist_ok=True)
            with Image.open(state.root / image_path) as image:
                image = image.convert("RGB")
                for ty in range(0, image.height, CORE_TILE):
                    for tx in range(0, image.width, CORE_TILE):
                        image.crop((tx, ty, min(tx + CORE_TILE, image.width), min(ty + CORE_TILE, image.height))).save(
                            folder / f"{ty // CORE_TILE}_{tx // CORE_TILE}.jpg", quality=FIELD_JPEG_QUALITY)
                (folder / "size.json").write_text(json.dumps({"width": image.width, "height": image.height}))
    return folder


def region_box(state: ProjectState, core_id: str, stain: str, x: int, y: int, size: int) -> tuple[int, int, int]:
    """A size × size box centred near (x, y) in core-image pixels, kept inside the image: (x0, y0, size)."""
    full = json.loads((_core_tiles(state, core_id, stain) / "size.json").read_text())
    size = min(size, full["width"], full["height"])
    return int(min(max(0, x - size // 2), full["width"] - size)), int(min(max(0, y - size // 2), full["height"] - size)), size


def core_region(state: ProjectState, core_id: str, stain: str, x0: int, y0: int, size: int) -> np.ndarray:
    """The full-resolution region at (x0, y0) of a core image, assembled from its tiles."""
    folder = _core_tiles(state, core_id, stain)
    out = np.full((size, size, 3), 255, np.uint8)
    for ty in range(y0 // CORE_TILE, (y0 + size - 1) // CORE_TILE + 1):
        for tx in range(x0 // CORE_TILE, (x0 + size - 1) // CORE_TILE + 1):
            tile = np.asarray(Image.open(folder / f"{ty}_{tx}.jpg").convert("RGB"))
            ox, oy = tx * CORE_TILE, ty * CORE_TILE
            sx0, sy0 = max(x0, ox), max(y0, oy)
            sx1, sy1 = min(x0 + size, ox + tile.shape[1]), min(y0 + size, oy + tile.shape[0])
            out[sy0 - y0: sy1 - y0, sx0 - x0: sx1 - x0] = tile[sy0 - oy: sy1 - oy, sx0 - ox: sx1 - ox]
    return out


def region_dab(state: ProjectState, core_id: str, stain: str, rgb: np.ndarray) -> np.ndarray:
    """RGBA overlay of the pixels above the slide's calibrated DAB threshold."""
    tma = str(state.core_image(core_id, stain)["tma"])
    cal = state.calibration()
    threshold = float(cal[(cal.tma == tma) & (cal.stain == stain)].threshold_dab_od.iloc[0])
    rgba = np.zeros((*rgb.shape[:2], 4), np.uint8)
    rgba[np.maximum(rgb_to_hed(rgb)[..., 2], 0) >= threshold] = (255, 40, 40, 120)
    return rgba
