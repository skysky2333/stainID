"""Core pictures for the viewer, without ever sending (or re-decoding) a whole core image.

A core image is a ~16 000 px PNG (about 750 MB once decoded) that may sit on a slow drive. The first time someone opens
a core, its image is decoded once and cut into a deep-zoom pyramid: 512 px JPEG tiles at full resolution, half, quarter …
down to a single tile, packed into one file per core image (`data/cache/core_pyramids/<core>_<stain>.tiles`, with a
`.json` index). From then on the browser asks only for the few tiles it shows, at the resolution of the screen. The core
grid uses small thumbnails (`data/cache/thumbs`, see `stainid.imaging.thumbnails`): written by core export, else made
from the pyramid or from the ~2000 px picture Check core quality kept (a fraction of a second), else from a reduced
decode of the whole image. Both are rebuilt when the image file changes (the pyramid stores its modification time and
size, a thumbnail must be newer than it); the source images are only ever read.

Decoding is the memory-hungry part, so it runs on one background thread, one image at a time (`DECODE` also guards
the few decodes made while a request waits). Pages ask for a picture, get "waiting"/"preparing", and ask again; work
that nobody has asked about for a while (the page was closed) is dropped from the queue.
"""
from __future__ import annotations

import json
import math
import os
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from stainid import disk_cache
from stainid.api.state import ProjectState
from stainid.imaging.color import rgb_to_hed
from stainid.imaging.core_images import read_bgr
from stainid.imaging.thumbnails import THUMB_BASE, base_path, jpeg, shrink, without_outlines, write_atomic, write_base
from stainid.qc.core_qc import overlay_path

TILE = 512
JPEG_QUALITY = 85
STALE_SECONDS = 45  # queued work nobody asked about for this long is dropped
DECODE = threading.Semaphore(1)  # whole core images decoded at once, across all projects (one is ~750 MB)


# ---------------------------------------------------------------- the background queue

@dataclass
class _Task:
    key: str
    priority: int  # 0 = someone is looking at this core, 1 = a thumbnail for the grid, 2 = pictures made ahead
    run: Callable[[], None]
    keep: bool = False  # work made ahead: kept in the queue although nobody is asking about it
    asked: float = field(default_factory=time.monotonic)
    queued: float = field(default_factory=time.monotonic)
    running: bool = False


_TASKS: dict[str, _Task] = {}
_FAILED: dict[str, tuple[str, float]] = {}  # key -> (message, when): reported, and retried after a minute
_COND = threading.Condition()
_WORKER: threading.Thread | None = None


def _next_task() -> _Task:
    with _COND:
        while True:
            now = time.monotonic()
            for key in [k for k, t in _TASKS.items() if not t.running and not t.keep and now - t.asked > STALE_SECONDS]:
                del _TASKS[key]
            waiting = [t for t in _TASKS.values() if not t.running]
            if waiting:
                task = min(waiting, key=lambda t: (t.priority, t.queued))
                task.running = True
                return task
            _COND.wait(timeout=10)


def _work() -> None:
    while True:
        task = _next_task()
        try:
            task.run()
            _FAILED.pop(task.key, None)
        except Exception as error:  # noqa: BLE001 - reported to the page that asked
            _FAILED[task.key] = (f"{type(error).__name__}: {error}", time.monotonic())
        finally:
            with _COND:
                _TASKS.pop(task.key, None)


def _ask(key: str, priority: int, run: Callable[[], None], keep: bool = False) -> dict:
    """Queue `run` under `key` (or note that someone still wants it). Returns where it stands. `keep`: work made ahead,
    which stays queued until it runs (other work is dropped once nobody has asked about it for `STALE_SECONDS`)."""
    global _WORKER
    with _COND:
        failed = _FAILED.get(key)
        if failed and time.monotonic() - failed[1] < 60:
            return {"state": "failed", "error": failed[0]}
        task = _TASKS.get(key)
        if task is None:
            task = _TASKS[key] = _Task(key, priority, run, keep=keep)
        task.asked = time.monotonic()
        task.priority = min(task.priority, priority)
        task.keep = task.keep or keep
        if _WORKER is None or not _WORKER.is_alive():
            _WORKER = threading.Thread(target=_work, name="stainid-pictures", daemon=True)
            _WORKER.start()
        _COND.notify_all()
        if task.running:
            return {"state": "preparing", "ahead": 0}
        ahead = sum(1 for t in _TASKS.values() if t is not task and (t.running or (t.priority, t.queued) < (task.priority, task.queued)))
        return {"state": "waiting" if ahead else "preparing", "ahead": ahead}


# ---------------------------------------------------------------- files

def _source(state: ProjectState, core_id: str, stain: str) -> Path:
    return state.root / state.core_image(core_id, stain)["image_path"]


def _stamp(path: Path) -> dict:
    stat = path.stat()
    return {"source_mtime_ns": stat.st_mtime_ns, "source_size": stat.st_size}


def _fresh(path: Path, source: Path) -> bool:
    """A cached picture made from the current version of its source image."""
    try:
        return path.stat().st_mtime >= source.stat().st_mtime
    except FileNotFoundError:
        return False


def _read(source: Path, flag: int = cv2.IMREAD_COLOR) -> np.ndarray:
    with DECODE:
        bgr = read_bgr(source, flag)
    if bgr is None:
        raise FileNotFoundError(f"could not read {source}")
    return bgr


# ---------------------------------------------------------------- the pyramid

def _pyramid_paths(state: ProjectState, core_id: str, stain: str) -> tuple[Path, Path]:
    folder = state.cache_dir("core_pyramids")
    return folder / f"{core_id}_{stain}.tiles", folder / f"{core_id}_{stain}.json"


_INDEXES: OrderedDict[str, tuple[int, dict]] = OrderedDict()  # index path -> (its mtime, contents); a few dozen kept


def _load_index(path: Path) -> dict | None:
    try:
        mtime = path.stat().st_mtime_ns
    except FileNotFoundError:
        return None
    cached = _INDEXES.get(str(path))
    if cached and cached[0] == mtime:
        _INDEXES.move_to_end(str(path))
        disk_cache.used(path)
        return cached[1]
    index = json.loads(path.read_text())
    disk_cache.used(path)
    _INDEXES[str(path)] = (mtime, index)
    while len(_INDEXES) > 48:
        _INDEXES.popitem(last=False)
    return index


def _index(state: ProjectState, core_id: str, stain: str, check_source: bool = True) -> dict | None:
    """The pyramid of this core image if it is ready and made from the current image file, else None (`check_source`
    False: any pyramid of it, without looking at the image file or the pack)."""
    pack, index_path = _pyramid_paths(state, core_id, stain)
    index = _load_index(index_path)
    if index is None or (check_source and not pack.exists()):  # the pack may have been cleaned up: made again
        return None
    if check_source and {k: index.get(k) for k in ("source_mtime_ns", "source_size")} != _stamp(_source(state, core_id, stain)):
        return None
    return index


def build_pyramid(source: Path, pack: Path, index_path: Path, thumb: Path | None = None) -> dict:
    """Decode `source` once and write every level of its tile pyramid into `pack` (+ `index_path`), and the grid
    thumbnail base to `thumb` on the way."""
    stamp = _stamp(source)
    level = _read(source)
    levels = []
    temporary = pack.with_name(pack.name + ".tmp")
    with temporary.open("wb") as out:
        while True:
            height, width = level.shape[:2]
            cols, rows = math.ceil(width / TILE), math.ceil(height / TILE)
            tiles = []
            for row in range(rows):
                for col in range(cols):
                    data = jpeg(level[row * TILE:(row + 1) * TILE, col * TILE:(col + 1) * TILE], JPEG_QUALITY)
                    tiles.append([out.tell(), len(data)])
                    out.write(data)
            levels.append({"width": width, "height": height, "cols": cols, "rows": rows, "tiles": tiles})
            if thumb is not None and max(width, height) <= THUMB_BASE and not _fresh(thumb, source):
                write_base(thumb, level)
            if max(width, height) <= TILE:
                break
            level = cv2.resize(level, ((width + 1) // 2, (height + 1) // 2), interpolation=cv2.INTER_AREA)
    os.replace(temporary, pack)
    disk_cache.wrote(pack)
    index = {"width": levels[0]["width"], "height": levels[0]["height"], "tile": TILE, "levels": levels, **stamp}
    write_atomic(index_path, json.dumps(index).encode())
    return index


def picture(state: ProjectState, core_id: str, stain: str, prepare: bool = True) -> dict:
    """What the viewer needs to show a core image: its pyramid (levels from full resolution down), or how far its
    preparation has got (and it is queued unless `prepare` is False)."""
    index = _index(state, core_id, stain)
    if index is not None:
        return {"ready": True, "width": index["width"], "height": index["height"], "tile": index["tile"],
                "version": str(index["source_mtime_ns"]),
                "levels": [{k: level[k] for k in ("width", "height", "cols", "rows")} for level in index["levels"]]}
    source = _source(state, core_id, stain)
    if not source.exists():
        raise FileNotFoundError(f"the image of {core_id} {stain} is missing: {state.project.relative(source)}")
    pack, index_path = _pyramid_paths(state, core_id, stain)
    thumb = _thumb_base(state, core_id, stain)
    status = _ask(str(pack), 0, lambda: build_pyramid(source, pack, index_path, thumb)) if prepare else {"state": "not_started"}
    # a picture to show meanwhile, of a size that is made or quick to make (any other would decode the image again)
    thumb_size = 1024 if thumbnail_state(state, core_id, stain, 1024) != "slow" else next(
        (n for n in (512, 256) if _fresh(_thumb_sized(state, core_id, stain, n), source)), None)
    return {"ready": False, **status, "thumb_size": thumb_size}


def tile(state: ProjectState, core_id: str, stain: str, level: int, col: int, row: int) -> bytes:
    """One JPEG tile of the pyramid (KeyError when there is no such tile or the pyramid is not ready)."""
    index = _index(state, core_id, stain, check_source=False)
    if index is None or not 0 <= level < len(index["levels"]):
        raise KeyError(f"no level {level} for {core_id} {stain}")
    info = index["levels"][level]
    if not (0 <= col < info["cols"] and 0 <= row < info["rows"]):
        raise KeyError(f"no tile {col},{row} on level {level}")
    offset, length = info["tiles"][row * info["cols"] + col]
    pack, _ = _pyramid_paths(state, core_id, stain)
    try:
        with pack.open("rb") as handle:
            handle.seek(offset)
            return handle.read(length)
    except FileNotFoundError:  # cleaned up: the viewer asks for the picture again, which makes it again
        raise KeyError(f"the pyramid of {core_id} {stain} is being made again") from None


def region(state: ProjectState, core_id: str, stain: str, x0: int, y0: int, width: int, height: int) -> np.ndarray | None:
    """Full-resolution pixels (BGR) of a box of a core image, read from the few pyramid tiles it touches (white beyond
    the image); None when the core has no up-to-date pyramid."""
    index = _index(state, core_id, stain)
    if index is None:
        return None
    level = index["levels"][0]
    out = np.full((height, width, 3), 255, np.uint8)
    for row in range(max(0, y0 // TILE), min(level["rows"], (y0 + height - 1) // TILE + 1)):
        for col in range(max(0, x0 // TILE), min(level["cols"], (x0 + width - 1) // TILE + 1)):
            part = cv2.imdecode(np.frombuffer(tile(state, core_id, stain, 0, col, row), np.uint8), cv2.IMREAD_COLOR)
            tx, ty = col * TILE, row * TILE
            ax0, ay0 = max(x0, tx), max(y0, ty)
            ax1, ay1 = min(x0 + width, tx + part.shape[1]), min(y0 + height, ty + part.shape[0])
            if ax1 > ax0 and ay1 > ay0:
                out[ay0 - y0:ay1 - y0, ax0 - x0:ax1 - x0] = part[ay0 - ty:ay1 - ty, ax0 - tx:ax1 - tx]
    return out


_DAB_TILES: OrderedDict[tuple, bytes] = OrderedDict()
_DAB_BYTES = 24 * 2**20  # memory kept for rendered DAB tiles


def dab_tile(state: ProjectState, core_id: str, stain: str, level: int, col: int, row: int) -> bytes:
    """A transparent PNG over one tile: the pixels darker in DAB than the slide's calibrated threshold."""
    index = _index(state, core_id, stain, check_source=False) or {}
    key = (str(state.root), core_id, stain, index.get("source_mtime_ns"), level, col, row)
    if key in _DAB_TILES:
        _DAB_TILES.move_to_end(key)
        return _DAB_TILES[key]
    tma = str(state.core_image(core_id, stain)["tma"])
    cal = state.calibration()
    match = cal[(cal.tma == tma) & (cal.stain == stain)] if not cal.empty else cal
    if match.empty:
        raise KeyError(f"no stain threshold for {tma} {stain} yet")
    threshold = float(match.threshold_dab_od.iloc[0])
    bgr = cv2.imdecode(np.frombuffer(tile(state, core_id, stain, level, col, row), np.uint8), cv2.IMREAD_COLOR)
    mask = np.maximum(rgb_to_hed(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))[..., 2], 0) >= threshold
    bgra = np.zeros((*mask.shape, 4), np.uint8)
    bgra[mask] = (40, 40, 255, 130)
    ok, encoded = cv2.imencode(".png", bgra)
    data = encoded.tobytes()
    _DAB_TILES[key] = data
    while sum(map(len, _DAB_TILES.values())) > _DAB_BYTES:
        _DAB_TILES.popitem(last=False)
    return data


# ---------------------------------------------------------------- thumbnails

QUALITY_DECODES = threading.Semaphore(4)  # quality pictures (~2000 px, ~12 MB decoded) made into thumbnails at once
_MAKERS: dict[str, threading.Lock] = {}  # one maker per thumbnail file; parallel requests for it wait for that one
_MAKERS_GUARD = threading.Lock()


def _maker(path: Path) -> threading.Lock:
    with _MAKERS_GUARD:
        return _MAKERS.setdefault(str(path), threading.Lock())


def _thumb_base(state: ProjectState, core_id: str, stain: str) -> Path:
    return base_path(state.cache_dir("thumbs"), core_id, stain)


def _thumb_sized(state: ProjectState, core_id: str, stain: str, size: int) -> Path:
    return state.cache_dir("thumbs") / f"{core_id}_{stain}_{size}.jpg"


def _quality_picture(state: ProjectState, core_id: str, stain: str) -> Path | None:
    """The ~2000 px picture Check core quality kept of this core image (outlines drawn on), if made from this version
    of the image."""
    source, manifest = _source(state, core_id, stain), state.project.input("core_manifest")
    if not source.is_relative_to(manifest.parent):
        return None
    picture = overlay_path(manifest, {"output_path": source.relative_to(manifest.parent).as_posix()})
    return picture if _fresh(picture, source) else None


def _pyramid_level(state: ProjectState, core_id: str, stain: str) -> int | None:
    """The largest level of the core's pyramid (if it has one) that is no larger than a thumbnail base."""
    index = _index(state, core_id, stain)
    return next((i for i, level in enumerate(index["levels"]) if max(level["width"], level["height"]) <= THUMB_BASE), None) if index else None


def _from_quality_picture(picture: Path) -> np.ndarray | None:
    with QUALITY_DECODES:
        bgr = cv2.imread(str(picture), cv2.IMREAD_COLOR)
        return without_outlines(bgr) if bgr is not None else None


def _make_base(state: ProjectState, core_id: str, stain: str) -> Path:
    """The thumbnail base: from the pyramid when there is one, else from the core's quality picture (a fraction of a
    second), else from a reduced decode of the whole image (seconds; one at a time)."""
    source, base = _source(state, core_id, stain), _thumb_base(state, core_id, stain)
    with _maker(base):
        if _fresh(base, source):
            disk_cache.used(base)
            return base
        bgr = None
        if (small := _pyramid_level(state, core_id, stain)) is not None:
            info = _index(state, core_id, stain)["levels"][small]
            bgr = np.zeros((info["height"], info["width"], 3), np.uint8)
            for row in range(info["rows"]):
                for col in range(info["cols"]):
                    part = cv2.imdecode(np.frombuffer(tile(state, core_id, stain, small, col, row), np.uint8), cv2.IMREAD_COLOR)
                    bgr[row * TILE:row * TILE + part.shape[0], col * TILE:col * TILE + part.shape[1]] = part
        elif (picture := _quality_picture(state, core_id, stain)) is not None:
            bgr = _from_quality_picture(picture)
        if bgr is None:
            bgr = _read(source, cv2.IMREAD_REDUCED_COLOR_4)
        return write_base(base, bgr)


def thumbnail_state(state: ProjectState, core_id: str, stain: str, size: int) -> str:
    """'ready' (made), 'quick' (made on request in a fraction of a second, from the pyramid or the quality picture) or
    'slow' (the whole image has to be decoded: queue it with `prepare_thumbnail`)."""
    source = _source(state, core_id, stain)
    if _fresh(_thumb_sized(state, core_id, stain, size), source) or _fresh(_thumb_base(state, core_id, stain), source):
        return "ready"
    if _pyramid_level(state, core_id, stain) is not None or _quality_picture(state, core_id, stain) is not None:
        return "quick"
    return "slow"


def prepare_thumbnail(state: ProjectState, core_id: str, stain: str) -> dict:
    """Queue the thumbnail of a core image that needs its whole image decoded (one at a time, in the background)."""
    return _ask(str(_thumb_base(state, core_id, stain)), 1, lambda: _make_base(state, core_id, stain))


def thumbnail(state: ProjectState, core_id: str, stain: str, size: int) -> Path:
    """A JPEG of the whole core at most `size` px across (made now if need be: this can take a few seconds when
    `thumbnail_state` is 'slow')."""
    source = _source(state, core_id, stain)
    target = _thumb_sized(state, core_id, stain, size)
    with _maker(target):
        if not _fresh(target, source):
            base = cv2.imread(str(_make_base(state, core_id, stain)), cv2.IMREAD_COLOR)
            write_atomic(target, jpeg(shrink(base, size), 85))
    disk_cache.used(target)
    return target


def field_preview(state: ProjectState, tile: dict, box: tuple[int, int, int, int]) -> Path | None:
    """The field's picture (`box`: its place in the core, core px) cut from the core's thumbnail base: blurred, but
    there in a fraction of a second while the sharp one is made. None when the base needs the whole image decoded."""
    core_id, stain = str(tile["core_id"]), str(tile["stain"])
    if thumbnail_state(state, core_id, stain, THUMB_BASE) == "slow":
        return None
    core = state.core_image(core_id, stain)
    source = _source(state, core_id, stain)
    target = state.cache_dir("thumbs") / f"{tile['tile_id']}_preview.jpg"
    with _maker(target):
        if not _fresh(target, source):
            base = cv2.imread(str(_make_base(state, core_id, stain)), cv2.IMREAD_COLOR)
            scale = base.shape[1] / float(core["native_width_px"])
            x0, y0, x1, y1 = (int(round(v * scale)) for v in box)
            part = base[y0:max(y0 + 1, y1), x0:max(x0 + 1, x1)]
            write_atomic(target, jpeg(part, 80))
    disk_cache.used(target)
    return target


def field_thumbnail(state: ProjectState, tile: dict, size: int = 192) -> Path:
    """A small JPEG of one tile (a labelling field), cut from its core's thumbnail: cheap, so pickers can show many."""
    core = state.core_image(str(tile["core_id"]), str(tile["stain"]))
    source = _source(state, str(tile["core_id"]), str(tile["stain"]))
    target = state.cache_dir("thumbs") / f"{tile['tile_id']}_field_{size}.jpg"
    if not _fresh(target, source):
        base = cv2.imread(str(_make_base(state, str(tile["core_id"]), str(tile["stain"]))), cv2.IMREAD_COLOR)
        scale = base.shape[1] / float(core["native_width_px"])
        x, y = int(int(tile["x_px"]) * scale), int(int(tile["y_px"]) * scale)
        w, h = max(1, round(int(tile["width_px"]) * scale)), max(1, round(int(tile["height_px"]) * scale))
        part = base[y:y + h, x:x + w]
        write_atomic(target, jpeg(cv2.resize(part, (size, max(1, round(size * part.shape[0] / part.shape[1]))), interpolation=cv2.INTER_AREA), 85))
    disk_cache.used(target)
    return target
