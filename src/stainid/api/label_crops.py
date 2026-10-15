"""Pictures of labelled objects as the model sees them: a square of `CROP_UM[stain]` around the object (the crop of the
labelling queue), cut once and kept as a small JPEG in `data/cache/label_crops`.

A crop is cut from the cheapest picture that holds it: the field picture if the viewer already cut it, a pilot field's
PNG, or the core's deep-zoom pyramid (`pyramid.region`), all made on request in a moment. Otherwise the whole core
image has to be decoded (seconds, on a busy drive more): that goes to the background picture queue (one decode at a
time) and cuts every labelled object of that core image, of every session, in the same pass, so the next objects of
the core are ready at once. Pages ask `status` and show a placeholder until the crop is ready.

Positions follow `stainid.labels.frames`: field-picture pixels on a project tile, pilot-picture pixels on a pilot field,
core-image pixels when a label names a core.
"""
from __future__ import annotations

import json
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from stainid import cache
from stainid.api import pyramid
from stainid.api.state import ProjectState
from stainid.imaging.thumbnails import jpeg, write_atomic
from stainid.labels import frames, store
from stainid.labels.tasks import CROP_UM
from stainid.stains.neun.audit import crop_with_padding

QUALITY = 90
_LOCKS: dict[str, threading.Lock] = {}
_GUARD = threading.Lock()
_KNOWN: list = [None, frozenset()]  # the tile table last seen and its ids
_MASKS: OrderedDict[str, np.ndarray] = OrderedDict()  # a few fields' object masks, for outlines


class NotReady(LookupError):
    """The crop needs its core image decoded first (`status` queues that)."""


@dataclass(frozen=True)
class Spot:
    """One labelled object: its stain, the tile (or pilot field, or core) its label names, and the position there."""
    stain: str
    tile_id: str
    x: float
    y: float


def _lock(key: str) -> threading.Lock:
    with _GUARD:
        return _LOCKS.setdefault(key, threading.Lock())


def size_px(state: ProjectState, stain: str) -> int:
    return max(32, round(CROP_UM[stain] / state.project.pixel_size_um))


def _known(state: ProjectState) -> frozenset[str]:
    tiles = state.tiles
    if _KNOWN[0] is not tiles:
        _KNOWN[:] = [tiles, frozenset(tiles.tile_id) if not tiles.empty else frozenset()]
    return _KNOWN[1]


def frame(state: ProjectState, tile_id: str, stain: str) -> str | None:
    """The frame of a label naming `tile_id` (see `stainid.labels.frames`), None when no picture of it exists."""
    if tile_id in _known(state):
        return frames.FIELD
    if _pilot(state, tile_id).exists():
        return frames.PILOT
    images = state.core_images
    if not images.empty and ((images.core_id == tile_id) & (images.stain == stain)).any():
        return frames.CORE
    return None


def core_of(state: ProjectState, spot: Spot) -> tuple[str, float, float] | None:
    """The core image a spot lies on and its position there (None for a pilot field or an unknown tile)."""
    kind = frame(state, spot.tile_id, spot.stain)
    if kind == frames.FIELD:
        tile = state.tile(spot.tile_id)
        return (str(tile["core_id"]), *frames.core_position(tile, spot.x, spot.y, state.project.context_px))
    if kind == frames.CORE:
        return spot.tile_id, spot.x, spot.y
    return None


def path(state: ProjectState, spot: Spot) -> Path:
    return state.cache_dir("label_crops") / f"{spot.tile_id}_{round(spot.x)}_{round(spot.y)}_{size_px(state, spot.stain)}.jpg"


def _pilot(state: ProjectState, tile_id: str) -> Path:
    return state.project.output("reviews") / "fields" / f"{tile_id}.png"


def _write(state: ProjectState, spot: Spot, bgr: np.ndarray) -> Path:
    target = path(state, spot)
    write_atomic(target, jpeg(bgr, QUALITY))
    return target


def _box(state: ProjectState, spot: Spot, x: float, y: float) -> tuple[int, int, int]:
    """Top-left corner and side of the crop around image position (x, y), as `crop_with_padding` cuts it."""
    size = size_px(state, spot.stain)
    return round(x) - size // 2, round(y) - size // 2, size


def _field_picture(state: ProjectState, spot: Spot) -> Path | None:
    """The field picture the viewer cut earlier, when the whole crop lies inside it."""
    cached = state.cache_dir("fields") / f"{spot.tile_id}.jpg"
    if not cached.exists():
        return None
    try:
        with Image.open(cached) as picture:
            width, height = picture.size
    except OSError:
        return None
    x0, y0, size = _box(state, spot, spot.x, spot.y)
    return cached if x0 >= 0 and y0 >= 0 and x0 + size <= width and y0 + size <= height else None


def _from_field_picture(state: ProjectState, spot: Spot) -> np.ndarray | None:
    cached = _field_picture(state, spot)
    bgr = None if cached is None else cv2.imread(str(cached), cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    x0, y0, size = _box(state, spot, spot.x, spot.y)
    return bgr[y0:y0 + size, x0:x0 + size].copy()


def _quick(state: ProjectState, spot: Spot) -> np.ndarray | None:
    """The crop from a picture that is quick to read, or None when only the whole core image holds it."""
    kind = frame(state, spot.tile_id, spot.stain)
    if kind == frames.PILOT:
        bgr = cv2.imread(str(_pilot(state, spot.tile_id)), cv2.IMREAD_COLOR)
        return None if bgr is None else crop_with_padding(bgr, spot.x, spot.y, size_px(state, spot.stain))
    if kind == frames.FIELD and (crop := _from_field_picture(state, spot)) is not None:
        return crop
    place = core_of(state, spot)
    if place is None:
        return None
    core_id, cx, cy = place
    return pyramid.region(state, core_id, spot.stain, *_box(state, spot, cx, cy)[:2], size_px(state, spot.stain), size_px(state, spot.stain))


def quick_source(state: ProjectState, spot: Spot) -> bool:
    """Whether `get` can cut this crop in a moment (without decoding a core image)."""
    kind = frame(state, spot.tile_id, spot.stain)
    if kind == frames.PILOT or (kind == frames.FIELD and _field_picture(state, spot) is not None):
        return True
    place = core_of(state, spot)
    try:
        return place is not None and pyramid._index(state, place[0], spot.stain) is not None
    except (KeyError, FileNotFoundError):
        return False


def get(state: ProjectState, spot: Spot) -> Path:
    """The crop's JPEG: kept, or cut now from a quick picture. NotReady when the core image must be decoded first,
    FileNotFoundError when there is no picture of the spot at all."""
    target = path(state, spot)
    if target.exists():
        return target
    if frame(state, spot.tile_id, spot.stain) is None:
        raise FileNotFoundError(f"there is no picture of {spot.tile_id}")
    with _lock(str(target)):
        if target.exists():
            return target
        crop = _quick(state, spot)
        if crop is None:
            raise NotReady(f"the core image of {spot.tile_id} is being read")
        return _write(state, spot, crop)


def status(state: ProjectState, spot: Spot) -> dict:
    """{"state": "ready"} when `get` answers at once; else the decode of its core image is queued (or found queued)
    and this says where it stands: "waiting" (with how many core images are `ahead`), "preparing" or "failed"."""
    if path(state, spot).exists() or quick_source(state, spot):
        return {"state": "ready"}
    place = core_of(state, spot)
    if place is None:
        return {"state": "missing"}
    core_id = place[0]
    return pyramid._ask(f"label crops {state.root} {core_id} {spot.stain}", 0, lambda: cut_core(state, core_id, spot.stain))


def labelled_spots(state: ProjectState, stain: str) -> pd.DataFrame:
    """Every position of this stain any label names (tile_id, x, y in the label's frame), kept until a label is saved."""
    root = state.project.output("reviews")

    def make() -> pd.DataFrame:
        labels = store.current(root)
        labels = labels[(labels.stain == stain) & labels.x.notna() & labels.task.isin(["class", "outline"])]
        labels = frames.in_field_frame(labels, state.tiles, state.project.context_px)
        return labels.assign(rx=labels.x.astype(float).round(), ry=labels.y.astype(float).round()).drop_duplicates(
            ["tile_id", "rx", "ry"])[["tile_id", "x", "y"]].reset_index(drop=True)

    return cache.memo(("labelled spots", str(state.root), stain), [store.log_path(root)], make)


def cut_core(state: ProjectState, core_id: str, stain: str) -> int:
    """Decode one core image and cut every labelled object on it that has no crop yet. Returns how many were cut."""
    spots = labelled_spots(state, stain)
    tiles = state.tiles
    mine = set(tiles.tile_id[(tiles.core_id == core_id) & (tiles.stain == stain)]) | {core_id} if not tiles.empty else {core_id}
    todo = [Spot(stain, r.tile_id, float(r.x), float(r.y)) for r in spots[spots.tile_id.isin(mine)].itertuples()]
    todo = [s for s in todo if not path(state, s).exists() and core_of(state, s) is not None]
    if not todo:
        return 0
    bgr = pyramid._read(pyramid._source(state, core_id, stain))
    for spot in todo:
        _, cx, cy = core_of(state, spot)
        _write(state, spot, crop_with_padding(bgr, cx, cy, size_px(state, stain)))
    return len(todo)


def outline(state: ProjectState, spot: Spot) -> list[list[float]] | None:
    """The AI's outline of the object (its Segment Anything mask), in crop pixels, when the field has masks. Kept next
    to the crop until the masks change."""
    if frame(state, spot.tile_id, spot.stain) != frames.FIELD:
        return None
    masks = state.project.output("masks") / spot.stain / "labels" / f"{spot.tile_id}.npz"
    kept = path(state, spot).with_suffix(".outline.json")
    try:
        if kept.stat().st_mtime >= masks.stat().st_mtime:
            return json.loads(kept.read_text())
    except FileNotFoundError:
        if not masks.exists():
            return None
    points = _mask_outline(masks, *_box(state, spot, spot.x, spot.y))
    write_atomic(kept, json.dumps(points).encode())
    return points


def _mask_outline(masks: Path, x0: int, y0: int, size: int) -> list[list[float]] | None:
    """The outline of the mask under the middle of a crop box, in crop pixels."""
    with _GUARD:
        labels = _MASKS.get(str(masks))
    if labels is None:
        with np.load(masks) as loaded:
            labels = loaded["labels"]
        with _GUARD:
            _MASKS[str(masks)] = labels
            while len(_MASKS) > 4:
                _MASKS.popitem(last=False)
    window = np.zeros((size, size), labels.dtype)
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(labels.shape[1], x0 + size), min(labels.shape[0], y0 + size)
    if sx1 <= sx0 or sy1 <= sy0:
        return None
    window[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = labels[sy0:sy1, sx0:sx1]
    c = size // 2
    near = window[max(0, c - 3):c + 4, max(0, c - 3):c + 4]
    found = near[near > 0]
    if not found.size:
        return None
    label = np.bincount(found.astype(np.int64)).argmax()
    contours, _ = cv2.findContours((window == label).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = cv2.approxPolyDP(max(contours, key=cv2.contourArea), 1.0, True)[:, 0, :]
    return contour.astype(float).tolist() if len(contour) >= 3 else None
