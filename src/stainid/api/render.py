"""Server-side image products for the viewer: field crops, raster overlays and object outlines (core pictures: `pyramid`)."""
from __future__ import annotations

import io
import json
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from stainid import cache, disk_cache, palette
from stainid.api import pyramid
from stainid.api.state import ProjectState
from stainid.imaging.color import rgb_to_hed
from stainid.imaging.core_images import open_image
from stainid.imaging.tissue import core_tissue_darkness, field_tissue_mask, fold_mask, linear_artifact_mask
from stainid.tables import read_csv
from stainid.v2.params import linear_contrast

Image.MAX_IMAGE_PIXELS = None
FIELD_JPEG_QUALITY = 88
_IMAGE_LOCKS: dict[str, threading.Lock] = {}
URGENT_DECODE = threading.Semaphore(1)  # a field someone waits for: decoded besides the background's (`pyramid.DECODE`)
_URGENT = [0]  # pictures someone waits for, being made now: work made ahead waits for them (`yield_to_people`)
_URGENT_LOCK = threading.Lock()


@contextmanager
def someone_waits():
    """Around the making of a picture someone waits for: what is made ahead (pictures, thread ladders, Segment
    Anything features, the idle-time preparation) does not start a core decode meanwhile, so the person's has the
    disk and the processor."""
    with _URGENT_LOCK:
        _URGENT[0] += 1
    try:
        yield
    finally:
        with _URGENT_LOCK:
            _URGENT[0] -= 1


def people_waiting() -> bool:
    return _URGENT[0] > 0


def yield_to_people(most: float = 60.0) -> None:
    """Work made ahead calls this before a heavy step: it waits (at most `most` seconds) while someone waits for a
    picture."""
    end = time.monotonic() + most
    while _URGENT[0] > 0 and time.monotonic() < end:
        time.sleep(0.05)


def wait_for_picture(state: ProjectState, tile_id: str, most: float = 20.0) -> bool:
    """Work made ahead for a field someone has open waits (at most `most` seconds) for the field's picture, which the
    page asks for at once, rather than decoding the core beside it."""
    target = field_path(state, tile_id)
    end = time.monotonic() + most
    while not target.exists() and time.monotonic() < end:
        time.sleep(0.1)
    return target.exists()


def open_rows(path: Path | str, rows: int):
    """A core image with only its first `rows` rows read, where its format allows (a PNG is stored row by row: a
    field near the top is cut after a fraction of the file is read and inflated; everything else: the whole image)."""
    image = open_image(path)
    try:
        tiles = getattr(image, "tile", None) or []
        if image.format == "PNG" and len(tiles) == 1 and tiles[0][0] == "zip" and 0 < rows < image.height:
            width = image.width
            image._size = (width, rows)
            image.tile = [(tiles[0][0], (0, 0, width, rows), tiles[0][2], tiles[0][3])]
    except Exception:  # noqa: BLE001 - another reader: read whole
        pass
    return image


def crop_origin(tile: dict, context_px: int) -> tuple[int, int]:
    return max(0, int(tile["x_px"]) - context_px), max(0, int(tile["y_px"]) - context_px)


def crop_box(state: ProjectState, tile: dict) -> tuple[int, int, int, int]:
    """Where the field's picture (the tile with its context margin) lies in its core image: x0, y0, x1, y1 (core px),
    as `_cut` cuts it, known without reading the image."""
    context = state.project.context_px
    x0, y0 = crop_origin(tile, context)
    x1, y1 = int(tile["x_px"]) + int(tile["width_px"]) + context, int(tile["y_px"]) + int(tile["height_px"]) + context
    try:
        core = state.core_image(str(tile["core_id"]), str(tile["stain"]))
        x1, y1 = min(x1, int(core["native_width_px"])), min(y1, int(core["native_height_px"]))
    except (KeyError, IndexError, ValueError, TypeError):
        pass
    return x0, y0, x1, y1


def picture_size(state: ProjectState, tile: dict) -> list[int]:
    """The field picture's size in px, before it is made: the page lays the field out (and draws its objects) at once."""
    x0, y0, x1, y1 = crop_box(state, tile)
    return [x1 - x0, y1 - y0]


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


def _bottom(state: ProjectState, tile: dict) -> int:
    return int(tile["y_px"]) + int(tile["height_px"]) + state.project.context_px


def _crop_field(state: ProjectState, tile: dict, target: Path, urgent: bool = False) -> None:
    """Cut one tile (with its context margin) out of its core image: only the rows down to it are read. `urgent`
    (someone waits for it): on the slot of its own; else on the background's (the callers made ahead look at
    `yield_to_people` before: never here, under the core's lock, which the person's picture may wait for)."""
    with URGENT_DECODE if urgent else pyramid.DECODE, open_rows(state.root / tile["image_path"], _bottom(state, tile)) as image:
        _cut(state, image, tile, target)


def _cut(state: ProjectState, image, tile: dict, target: Path) -> None:
    context = state.project.context_px
    x0, y0 = crop_origin(tile, context)
    x1 = min(int(tile["x_px"]) + int(tile["width_px"]) + context, image.width)
    y1 = min(int(tile["y_px"]) + int(tile["height_px"]) + context, image.height)
    part = image.crop((x0, y0, x1, y1)).convert("RGB")
    out = io.BytesIO()
    part.save(out, format="JPEG", quality=FIELD_JPEG_QUALITY)
    disk_cache.write(target, out.getvalue())


def fields_made(state: ProjectState) -> frozenset[str]:
    """The tiles whose field picture is made (one listing of the folder, kept a few seconds)."""
    return frozenset(name[:-4] for name in cache.names(state.cache_dir("fields")) if name.endswith(".jpg"))


def field_jpegs(state: ProjectState, tile_ids, step=None, urgent: bool = False) -> int:
    """Make the field pictures of these tiles that are not made yet, decoding each core image once for all its fields
    (a whole core is decoded to cut even one field). `step(done, total)` hears the progress in core images. Returns
    how many pictures were made. `urgent` (someone waits for the field): decoded on its own slot (`URGENT_DECODE`),
    never behind pictures made ahead or a core pyramid."""
    wanted: dict[str, list[dict]] = {}
    for tile_id in dict.fromkeys(tile_ids):
        if not field_path(state, tile_id).exists():
            tile = state.tile(tile_id)
            wanted.setdefault(tile["image_path"], []).append(tile)
    made = 0
    for done, (image_path, tiles) in enumerate(wanted.items()):
        if step:
            step(done, len(wanted))
        if not urgent:
            yield_to_people()
        with _IMAGE_LOCKS.setdefault(image_path, threading.Lock()):
            missing = [t for t in tiles if not field_path(state, t["tile_id"]).exists()]
            if missing:
                rows = max(_bottom(state, t) for t in missing)  # only down to the lowest field wanted
                with URGENT_DECODE if urgent else pyramid.DECODE, open_rows(state.root / image_path, rows) as image:
                    for tile in missing:
                        _cut(state, image, tile, field_path(state, tile["tile_id"]))
                        made += 1
    if step:
        step(len(wanted), len(wanted))
    return made


def field_jpeg(state: ProjectState, tile_id: str) -> Path:
    """The field's picture, made now if need be (someone waits for it: `someone_waits`): only it, from the rows down to
    it; the core's other fields are cut afterwards, ahead (`field_jpegs`)."""
    target = field_path(state, tile_id)
    if not target.exists():
        with someone_waits():
            tile = state.tile(tile_id)
            with _IMAGE_LOCKS.setdefault(tile["image_path"], threading.Lock()):
                if not target.exists():
                    _crop_field(state, tile, target, urgent=True)
    disk_cache.used(target)
    return target


def _rgba_png(mask: np.ndarray, color: tuple[int, int, int], alpha: int, target: Path) -> Path:
    rgba = np.zeros((*mask.shape, 4), np.uint8)
    rgba[mask] = (*color, alpha)
    out = io.BytesIO()
    Image.fromarray(rgba).save(out, format="PNG")
    disk_cache.write(target, out.getvalue())
    return target


def linear_mask(state: ProjectState, tile: dict, rgb: np.ndarray) -> np.ndarray:
    """Detection's linear-artifact mask of a tile's field, as stage 2 makes it (`v2.proposals.linear_artifact_contrast`,
    relative to the core image's tissue darkness, worked out once per core image)."""
    contrast = linear_contrast(state.project)
    if contrast is None:
        return linear_artifact_mask(rgb)
    path = state.root / tile["image_path"]

    def darkness() -> float | None:
        with open_image(path) as image:
            return core_tissue_darkness(np.asarray(image.convert("RGB")))

    return linear_artifact_mask(rgb, contrast, cache.memo(("core tissue darkness", str(path)), [path], darkness))


def exclusion_overlay(state: ProjectState, tile_id: str) -> Path:
    target = state.cache_dir("overlays") / f"{tile_id}_exclusions.png"
    if not target.exists():
        rgb = field_rgb(state, tile_id)
        tile = state.tile(tile_id)
        mask = ~field_tissue_mask(rgb) | linear_mask(state, tile, rgb)
        if tile["stain"] != "NeuN":
            mask |= fold_mask(rgb, state.project.pixel_size_um)
        _rgba_png(mask, palette.overlay("excluded"), 110, target)
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
        _rgba_png(np.maximum(rgb_to_hed(rgb)[..., 2], 0) >= threshold, palette.overlay("dab"), 120, target)
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
        valid = field_tissue_mask(rgb) & ~linear_mask(state, tile, rgb) & ~fold_mask(rgb, state.project.pixel_size_um)
        skeleton = thread_network(rgb, valid, threshold, state.project.pixel_size_um, nuclei=nuclei)["skeleton"]
        _rgba_png(cv2.dilate(skeleton.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool), palette.overlay("threads"), 235, target)
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
