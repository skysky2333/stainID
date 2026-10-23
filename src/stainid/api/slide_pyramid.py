"""The picture of a whole slide, put together from its regions (`stainid.imageset.slides`), as a deep-zoom pyramid in
the same format as a core's (`stainid.api.pyramid`: 512 px JPEG tiles, full resolution down to one tile, one pack
file and a `.json` index), so the Cores view and the annotation pages zoom into a whole section as they do into a core.

It is built once per slide and kept (`data/cache/slide_pyramids/<slide>.tiles`), made again only when a region's
picture changes (`signature`: every region's image file). Building reads each region's picture at the analysis
scale once (`read_bgr`, the picture *Prepare images* keeps) and works cell by cell of the regions' grid (`CELL` =
6144 px, 12 tiles across: a region never crosses a cell's edge): the cell's tiles at full, half and quarter
resolution, and its eighth into one small picture of the whole slide (on disk, `np.memmap`), from which the coarser
levels are cut. Glass is one shared white tile. A build stops between cells when asked (`stop()`) and goes on from
there next time (`<slide>.partial.json` and its picture so far). It runs in the idle-time preparation
(`stainid.api.ahead`, the lowest priority of all) or, when someone opens a slide whose picture is not made, in the
pictures thread (`pyramid._ask`), one build at a time.

The regions and every pipeline output stay as they are: this is a picture for viewing only.
"""
from __future__ import annotations

import hashlib
import json
import math
import threading
import time
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np

from stainid import disk_cache
from stainid.api import pyramid
from stainid.api.state import ProjectState
from stainid.imageset.slides import CELL, Slide, layouts
from stainid.imaging.color import rgb_to_hed
from stainid.imaging.thumbnails import jpeg, shrink, write_atomic

TILE = pyramid.TILE
KIND = "slide_pyramids"
CELL_LEVELS = 3  # levels made cell by cell (CELL / 2**2 is still a whole number of tiles); the rest from the small picture
SAVE_EVERY = 8  # cells between two saves of a build's progress
THUMBS = (512, 2048)  # the whole-slide pictures the annotation pages ask for (the field strip's "Whole slide" and its middle zoom)
_BUILD = threading.Lock()  # one slide built at a time (each holds a cell, a region and the small picture)
_LAYOUTS: dict[str, tuple[object, dict[str, Slide]]] = {}  # project root -> (the core image table, its slides)


class NotSlides(KeyError):
    """The project is not a whole-slide project, or has no such slide."""


def slides(state: ProjectState) -> dict[str, Slide]:
    """The project's slides (empty unless it is a whole-slide project), kept until the core image table changes."""
    if state.project.kind != "slides":
        return {}
    table = state.core_images
    key = str(state.root)
    kept = _LAYOUTS.get(key)
    if kept is None or kept[0] is not table:  # the table is read again when its file changes
        kept = _LAYOUTS[key] = (table, layouts(table))
    return kept[1]


def slide(state: ProjectState, slide_id: str) -> Slide:
    found = slides(state).get(slide_id)
    if found is None:
        raise NotSlides(f"no slide {slide_id}")
    return found


def slide_of(state: ProjectState, core_id: str) -> Slide | None:
    """The slide a region belongs to (None in other projects)."""
    for s in slides(state).values():
        if s.find(core_id) is not None:
            return s
    return None


# ---------------------------------------------------------------- files

def _paths(state: ProjectState, slide_id: str) -> tuple[Path, Path, Path, Path]:
    folder = state.cache_dir(KIND)
    return folder / f"{slide_id}.tiles", folder / f"{slide_id}.json", folder / f"{slide_id}.partial.json", folder / f"{slide_id}.small.raw"


def _image(state: ProjectState, core_id: str, stain: str) -> Path:
    return state.root / state.core_image(core_id, stain)["image_path"]


def signature(state: ProjectState, s: Slide) -> str:
    """What the picture is made from: each region's place and its image file's version."""
    parts = []
    for r in s.regions:
        try:
            stat = _image(state, r.core_id, s.stain).stat()
            stamp = f"{stat.st_mtime_ns}:{stat.st_size}"
        except (FileNotFoundError, KeyError):
            stamp = "missing"
        parts.append(f"{r.core_id}:{r.x}:{r.y}:{r.width}:{r.height}:{stamp}")
    return hashlib.sha1("|".join([s.slide_id, str(s.width), str(s.height), *parts]).encode()).hexdigest()[:16]


_INDEXES: OrderedDict[str, tuple[int, dict]] = OrderedDict()


def _load(path: Path) -> dict | None:
    try:
        mtime = path.stat().st_mtime_ns
    except FileNotFoundError:
        return None
    kept = _INDEXES.get(str(path))
    if kept and kept[0] == mtime:
        _INDEXES.move_to_end(str(path))
        return kept[1]
    try:
        index = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    _INDEXES[str(path)] = (mtime, index)
    while len(_INDEXES) > 24:
        _INDEXES.popitem(last=False)
    disk_cache.used(path)
    return index


def index(state: ProjectState, slide_id: str, check: bool = True) -> dict | None:
    """The slide's pyramid when it is made (and, with `check`, made from the regions' current pictures)."""
    pack, index_path, _, _ = _paths(state, slide_id)
    found = _load(index_path)
    if found is None or (check and not pack.exists()):
        return None
    if check and found.get("signature") != signature(state, slide(state, slide_id)):
        return None
    return found


def ready(state: ProjectState, slide_id: str) -> bool:
    return index(state, slide_id) is not None


# ---------------------------------------------------------------- building

def _level_sizes(width: int, height: int) -> list[tuple[int, int]]:
    sizes = [(width, height)]
    while max(sizes[-1]) > TILE:
        w, h = sizes[-1]
        sizes.append(((w + 1) // 2, (h + 1) // 2))
    return sizes


def _cell_canvas(state: ProjectState, s: Slide, cx: int, cy: int) -> np.ndarray | None:
    """One cell of the slide at full resolution (white glass), from the regions in it; None when it holds none."""
    x0, y0 = cx * CELL, cy * CELL
    w, h = min(CELL, s.width - x0), min(CELL, s.height - y0)
    inside = [r for r in s.regions if x0 <= r.x < x0 + CELL and y0 <= r.y < y0 + CELL]
    if not inside:
        return None
    canvas = np.full((h, w, 3), 255, np.uint8)
    for r in inside:
        bgr = pyramid._read(_image(state, r.core_id, s.stain))
        px, py = r.x - x0, r.y - y0
        part = bgr[:max(0, min(bgr.shape[0], h - py)), :max(0, min(bgr.shape[1], w - px))]
        canvas[py:py + part.shape[0], px:px + part.shape[1]] = part
    return canvas


class _Pack:
    """The pack being written: tiles appended, glass written once per size."""

    def __init__(self, path: Path, resume: int):
        self.handle = path.open("r+b" if resume else "wb")
        self.handle.truncate(resume)
        self.handle.seek(resume)
        self.blank: dict[str, list[int]] = {}

    def add(self, data: bytes) -> list[int]:
        at = self.handle.tell()
        self.handle.write(data)
        return [at, len(data)]

    def glass(self, w: int, h: int, known: dict) -> list[int]:
        key = f"{w}x{h}"
        if key not in known:
            known[key] = self.add(jpeg(np.full((h, w, 3), 255, np.uint8), pyramid.JPEG_QUALITY))
        return known[key]


def build(state: ProjectState, slide_id: str, stop=lambda: False) -> dict | None:
    """Make the slide's pyramid (or go on with a build that stopped); returns its index, or None when `stop()` asked
    it to stop first (what is made so far is kept for next time)."""
    s = slide(state, slide_id)
    with _BUILD:
        made = index(state, slide_id)
        if made is not None:
            return made
        return _build(state, s, stop)


def _build(state: ProjectState, s: Slide, stop) -> dict | None:
    pack_path, index_path, partial_path, small_path = _paths(state, s.slide_id)
    sig = signature(state, s)
    sizes = _level_sizes(s.width, s.height)
    grid = [(math.ceil(w / TILE), math.ceil(h / TILE)) for w, h in sizes]
    small_level = min(CELL_LEVELS, len(sizes) - 1)
    sw, sh = sizes[small_level]
    partial = None
    try:
        partial = json.loads(partial_path.read_text())
    except (OSError, ValueError):
        partial = None
    if not partial or partial.get("signature") != sig or not pack_path.exists() or not small_path.exists():
        partial = {"signature": sig, "done": [], "size": 0, "glass": {}, "tiles": {str(k): {} for k in range(small_level)}}
        small = np.memmap(small_path, np.uint8, "w+", shape=(sh, sw, 3))
        small[:] = 255
    else:
        small = np.memmap(small_path, np.uint8, "r+", shape=(sh, sw, 3))
    out = _Pack(pack_path, int(partial["size"]))
    done = {tuple(c) for c in partial["done"]}
    cells = [(cx, cy) for cy in range(math.ceil(s.height / CELL)) for cx in range(math.ceil(s.width / CELL))]
    try:
        for n, (cx, cy) in enumerate(cells):
            if (cx, cy) in done:
                continue
            if stop():
                small.flush()
                out.handle.flush()
                partial.update(done=sorted(done), size=out.handle.tell())
                write_atomic(partial_path, json.dumps(partial).encode())
                return None
            canvas = _cell_canvas(state, s, cx, cy)
            if canvas is not None:
                level = canvas
                for k in range(small_level):
                    per = CELL >> k  # the cell's side at this level
                    c0, r0 = cx * per // TILE, cy * per // TILE
                    for row in range(math.ceil(level.shape[0] / TILE)):
                        for col in range(math.ceil(level.shape[1] / TILE)):
                            part = level[row * TILE:(row + 1) * TILE, col * TILE:(col + 1) * TILE]
                            partial["tiles"][str(k)][str((r0 + row) * grid[k][0] + c0 + col)] = out.add(jpeg(part, pyramid.JPEG_QUALITY))
                    level = cv2.resize(level, ((level.shape[1] + 1) // 2, (level.shape[0] + 1) // 2), interpolation=cv2.INTER_AREA)
                per = CELL >> small_level
                x0, y0 = cx * per, cy * per
                part = level[:max(0, sh - y0), :max(0, sw - x0)]
                small[y0:y0 + part.shape[0], x0:x0 + part.shape[1]] = part
            done.add((cx, cy))
            if len(done) % SAVE_EVERY == 0 or n == len(cells) - 1:
                small.flush()
                out.handle.flush()
                partial.update(done=sorted(done), size=out.handle.tell())
                write_atomic(partial_path, json.dumps(partial).encode())
        levels = []
        for k in range(small_level):  # the cell levels: glass where no region is
            cols, rows = grid[k]
            known = partial["tiles"][str(k)]
            tiles = []
            for row in range(rows):
                for col in range(cols):
                    ref = known.get(str(row * cols + col))
                    if ref is None:
                        w = min(TILE, sizes[k][0] - col * TILE)
                        h = min(TILE, sizes[k][1] - row * TILE)
                        ref = out.glass(w, h, partial["glass"])
                    tiles.append(ref)
            levels.append({"width": sizes[k][0], "height": sizes[k][1], "cols": cols, "rows": rows, "tiles": tiles})
        level = np.asarray(small)
        for k in range(small_level, len(sizes)):
            if k > small_level:
                level = cv2.resize(level, sizes[k], interpolation=cv2.INTER_AREA)
            cols, rows = grid[k]
            tiles = [out.add(jpeg(level[row * TILE:(row + 1) * TILE, col * TILE:(col + 1) * TILE], pyramid.JPEG_QUALITY))
                     for row in range(rows) for col in range(cols)]
            levels.append({"width": sizes[k][0], "height": sizes[k][1], "cols": cols, "rows": rows, "tiles": tiles})
    finally:
        out.handle.close()
        del small
    found = {"width": s.width, "height": s.height, "tile": TILE, "levels": levels, "signature": sig, "version": sig,
             "slide_id": s.slide_id, "stain": s.stain}
    write_atomic(index_path, json.dumps(found).encode())
    disk_cache.wrote(pack_path, index_path)
    for leftover in (partial_path, small_path):
        leftover.unlink(missing_ok=True)
    return found


# ---------------------------------------------------------------- serving

def picture(state: ProjectState, slide_id: str, prepare: bool = True) -> dict:
    """What the viewer needs to show a slide (as `pyramid.picture` for a core): its levels, or how far it is."""
    s = slide(state, slide_id)
    found = index(state, slide_id)
    if found is not None:
        return {"ready": True, "width": found["width"], "height": found["height"], "tile": found["tile"], "version": found["version"],
                "levels": [{k: level[k] for k in ("width", "height", "cols", "rows")} for level in found["levels"]]}
    status = ask(state, slide_id) if prepare else {"state": "not_started"}
    return {"ready": False, **status, "thumb_size": None, "width": s.width, "height": s.height}


# The slides someone is waiting for, built one at a time in a thread of their own (not the core pictures' thread, which
# a slide's few minutes would hold up), most recently asked for first; dropped when nobody asked for `STALE` seconds.
_ASKED: dict[tuple[str, str], tuple[ProjectState, float]] = {}
_RUNNING: set[tuple[str, str]] = set()
_FAILED: dict[tuple[str, str], tuple[str, float]] = {}
_COND = threading.Condition()
_WORKER: list[threading.Thread] = []
STALE = 120.0


def ask(state: ProjectState, slide_id: str) -> dict:
    key = (str(state.root), slide_id)
    with _COND:
        failed = _FAILED.get(key)
        if failed and time.monotonic() - failed[1] < 60:
            return {"state": "failed", "error": failed[0]}
        if key in _RUNNING:
            return {"state": "preparing", "ahead": 0}
        _ASKED[key] = (state, time.monotonic())
        if not _WORKER or not _WORKER[0].is_alive():
            _WORKER[:] = [threading.Thread(target=_work, name="stainid-slide-pictures", daemon=True)]
            _WORKER[0].start()
        _COND.notify_all()
        ahead = len(_RUNNING) + sum(1 for k, (_, t) in _ASKED.items() if k != key and t > _ASKED[key][1])
        return {"state": "waiting" if ahead else "preparing", "ahead": ahead}


def _work() -> None:
    while True:
        with _COND:
            now = time.monotonic()
            for k in [k for k, (_, t) in _ASKED.items() if now - t > STALE]:
                del _ASKED[k]
            if not _ASKED:
                _COND.wait(timeout=10)
                continue
            key = max(_ASKED, key=lambda k: _ASKED[k][1])
            state, _ = _ASKED.pop(key)
            _RUNNING.add(key)
        try:
            build(state, key[1], stop=lambda k=key: k not in _RUNNING)
            _FAILED.pop(key, None)
        except Exception as error:  # noqa: BLE001 - reported to the page that asked
            _FAILED[key] = (f"{type(error).__name__}: {error}", time.monotonic())
        finally:
            with _COND:
                _RUNNING.discard(key)


def tile(state: ProjectState, slide_id: str, level: int, col: int, row: int) -> bytes:
    found = index(state, slide_id, check=False)
    if found is None or not 0 <= level < len(found["levels"]):
        raise KeyError(f"no level {level} for {slide_id}")
    info = found["levels"][level]
    if not (0 <= col < info["cols"] and 0 <= row < info["rows"]):
        raise KeyError(f"no tile {col},{row} on level {level}")
    offset, length = info["tiles"][row * info["cols"] + col]
    pack, _, _, _ = _paths(state, slide_id)
    try:
        with pack.open("rb") as handle:
            handle.seek(offset)
            return handle.read(length)
    except FileNotFoundError:
        raise KeyError(f"the picture of {slide_id} is being made again") from None


def pixels(state: ProjectState, slide_id: str, level: int, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    """A box of one level (that level's px), from the tiles it touches (white beyond the slide)."""
    found = index(state, slide_id, check=False)
    if found is None:
        raise KeyError(f"the picture of {slide_id} is not made yet")
    info = found["levels"][level]
    out = np.full((max(1, y1 - y0), max(1, x1 - x0), 3), 255, np.uint8)
    for row in range(max(0, y0 // TILE), min(info["rows"], (y1 - 1) // TILE + 1)):
        for col in range(max(0, x0 // TILE), min(info["cols"], (x1 - 1) // TILE + 1)):
            part = cv2.imdecode(np.frombuffer(tile(state, slide_id, level, col, row), np.uint8), cv2.IMREAD_COLOR)
            tx, ty = col * TILE, row * TILE
            ax0, ay0, ax1, ay1 = max(x0, tx), max(y0, ty), min(x1, tx + part.shape[1]), min(y1, ty + part.shape[0])
            if ax1 > ax0 and ay1 > ay0:
                out[ay0 - y0:ay1 - y0, ax0 - x0:ax1 - x0] = part[ay0 - ty:ay1 - ty, ax0 - tx:ax1 - tx]
    return out


def crop(state: ProjectState, slide_id: str, box: tuple[float, float, float, float], size: int) -> np.ndarray:
    """A box of the slide (slide px: x0, y0, x1, y1) at most `size` px across, from the coarsest level fine enough."""
    found = index(state, slide_id, check=False)
    if found is None:
        raise KeyError(f"the picture of {slide_id} is not made yet")
    x0, y0, x1, y1 = box
    across = max(x1 - x0, y1 - y0, 1)
    level = max(0, min(len(found["levels"]) - 1, int(math.floor(math.log2(max(1.0, across / size))))))
    f = 2 ** level
    part = pixels(state, slide_id, level, int(x0 // f), int(y0 // f), int(math.ceil(x1 / f)), int(math.ceil(y1 / f)))
    return shrink(part, size)


def thumbnail(state: ProjectState, slide_id: str, size: int) -> Path:
    """A JPEG of the whole slide at most `size` px across (from its pyramid; KeyError until that is made)."""
    found = index(state, slide_id)
    if found is None:
        raise KeyError(f"the picture of {slide_id} is not made yet")
    size = max(32, min(int(size), 4096))
    target = state.cache_dir(KIND) / f"{slide_id}_{found['version']}_{size}.jpg"
    if not target.exists():
        write_atomic(target, jpeg(crop(state, slide_id, (0, 0, found["width"], found["height"]), size), 85))
        disk_cache.wrote(target)
    else:
        disk_cache.used(target)
    return target


def crop_jpeg(state: ProjectState, slide_id: str, box: tuple[float, float, float, float], size: int) -> bytes:
    return jpeg(crop(state, slide_id, box, max(32, min(int(size), 4096))), 85)


_DAB: OrderedDict[tuple, bytes] = OrderedDict()


def dab_tile(state: ProjectState, slide_id: str, level: int, col: int, row: int) -> bytes:
    """The DAB-above-threshold overlay of one tile (as `pyramid.dab_tile`), with the slide's stain threshold."""
    s = slide(state, slide_id)
    found = index(state, slide_id, check=False) or {}
    key = (str(state.root), slide_id, found.get("version"), level, col, row)
    if key in _DAB:
        _DAB.move_to_end(key)
        return _DAB[key]
    cal = state.calibration()
    match = cal[(cal.tma == s.tma) & (cal.stain == s.stain)] if not cal.empty else cal
    if match.empty:
        raise KeyError(f"no stain threshold for {s.tma} {s.stain} yet")
    bgr = cv2.imdecode(np.frombuffer(tile(state, slide_id, level, col, row), np.uint8), cv2.IMREAD_COLOR)
    mask = np.maximum(rgb_to_hed(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))[..., 2], 0) >= float(match.threshold_dab_od.iloc[0])
    bgra = np.zeros((*mask.shape, 4), np.uint8)
    bgra[mask] = (40, 40, 255, 130)
    data = cv2.imencode(".png", bgra)[1].tobytes()
    _DAB[key] = data
    while sum(map(len, _DAB.values())) > 16 * 2**20:
        _DAB.popitem(last=False)
    return data


def made(state: ProjectState, slide_id: str, sizes=THUMBS) -> bool:
    """Whether a slide's picture and its thumbnails are made (looked at without making anything)."""
    found = index(state, slide_id)
    if found is None:
        return False
    return all((state.cache_dir(KIND) / f"{slide_id}_{found['version']}_{size}.jpg").exists() for size in sizes)


def pending(state: ProjectState) -> list[str]:
    """The slides whose picture is not made (for the idle-time preparation)."""
    return [slide_id for slide_id in slides(state) if not ready(state, slide_id)]


def bytes_estimate(s: Slide) -> int:
    """About what a slide's picture takes on disk (~70 kB a full-resolution tile of tissue, a third more for the rest)."""
    tissue = sum(r.width * r.height for r in s.regions)
    return int(tissue / TILE**2 * 70_000 * 4 / 3)


def remove_partial(state: ProjectState, slide_id: str) -> None:
    for p in _paths(state, slide_id)[2:]:
        p.unlink(missing_ok=True)


