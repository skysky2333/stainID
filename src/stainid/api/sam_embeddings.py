"""Segment Anything features of outline-session fields, made ahead so that even the first click outlines at once.

A click is outlined from the encoder's features of the 1024-px window around it (`stainid.masks.clicks`); making them
is the slow part. So the features of every window of a field are made on a background thread and kept on disk
(`data/cache/sam_embeddings/`, about 2 MB a window, nine windows for a 2,304-px field), plus a small `.json` saying the
field is complete. They are named after what they are made from: the core image (its modification time and size), the
field's square and margin, the window and the weights, so a changed image or model makes them again.

What is made, most urgent first: the field someone has open (`NOW`, asked by `open_field` and by the page while it shows
"Preparing outlines…"); the next fields that person will see (`NEXT`); and every field of the open outline sessions
(`AHEAD`, when a session is made, opened or looked at). A field yields to more urgent ones between two windows. The
first two kinds are made on the CPU too when there is no GPU; every field ahead only on a GPU (a window takes 40 ms
there, 5 s on the CPU). A click whose window is not made yet makes it at once (and keeps it).
"""
from __future__ import annotations

import hashlib
import io
import json
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from stainid import cache, disk_cache
from stainid.api import render
from stainid.api.state import ProjectState
from stainid.masks import clicks
from stainid.masks.sam import STRIDE, WINDOW, window_origin, window_origins

log = logging.getLogger(__name__)

VERSION = 1  # of the file format and the encoder set-up: a new one makes every embedding again
NOW, NEXT, AHEAD = 0, 1, 2
NEXT_FIELDS = 2  # fields made ahead for a person, after the one they are on
KEEP = 24  # windows kept in memory (2 MB each)
KEEP_FIELDS = 2  # field pictures kept in memory (16 MB each)
EVERY = 60.0  # seconds between two looks at a session's fields
RETRY_SECONDS = 30.0  # a field whose features failed is tried again after this long
PLAIN_FAILED = ("One-click outlines are not working on this field right now (Segment Anything could not start). Draw the outline by hand "
                "for now; it is tried again in a moment. The details are in the server's log.")

_MEMORY: OrderedDict[tuple[str, tuple[int, int]], clicks.Embedding] = OrderedDict()
_FIELDS: OrderedDict[tuple[str, str], tuple[tuple | None, np.ndarray]] = OrderedDict()
_MEMORY_LOCK = threading.Lock()
_LOOKED: dict[tuple[str, str], float] = {}
_WAITING = [0]  # clicks waiting to make a window: the background lets them go first


def checkpoint(state: ProjectState) -> Path:
    return state.project.model("sam")


# ---------------------------------------------------------------- the files

def field_stem(state: ProjectState, tile_id: str) -> str:
    """`<tile>.<digest of what the embeddings are made from>`: the start of the names of a field's files (deleted
    together)."""
    tile = state.tile(tile_id)
    weights = checkpoint(state)
    made_from = (VERSION, weights.name, weights.stat().st_size, str(tile["image_path"]), cache.signature(state.root / tile["image_path"]),
                 int(tile["x_px"]), int(tile["y_px"]), int(tile["width_px"]), int(tile["height_px"]), state.project.context_px,
                 render.FIELD_JPEG_QUALITY, WINDOW, STRIDE)
    return f"{tile_id}.{hashlib.sha1(repr(made_from).encode()).hexdigest()[:16]}"


def window_path(state: ProjectState, stem: str, origin: tuple[int, int]) -> Path:
    return state.cache_dir("sam_embeddings") / f"{stem}.{origin[0]}-{origin[1]}.npz"


def marker_path(state: ProjectState, stem: str) -> Path:
    """Written once every window of the field is made."""
    return state.cache_dir("sam_embeddings") / f"{stem}.json"


def _save(target: Path, embedding: clicks.Embedding) -> None:
    out = io.BytesIO()
    np.savez(out, features=embedding.features, original_size=np.array(embedding.original_size), input_size=np.array(embedding.input_size))
    disk_cache.write(target, out.getvalue())


def _load(target: Path) -> clicks.Embedding | None:
    try:
        with np.load(target) as saved:
            found = clicks.Embedding(saved["features"], tuple(int(v) for v in saved["original_size"]), tuple(int(v) for v in saved["input_size"]))
    except (OSError, ValueError, KeyError):  # not made, cleaned up, or half a file of a crash
        return None
    disk_cache.used(target)
    return found


def embedding(state: ProjectState, rgb: np.ndarray, stem: str, origin: tuple[int, int], model: clicks.ClickOutliner,
              remember: bool = True, click: bool = False) -> clicks.Embedding:
    """The features of one window of a field: from memory, else from disk, else made now (and kept on disk). A `click`
    goes before the background's next window."""
    key = (f"{state.root}/{stem}", origin)
    with _MEMORY_LOCK:
        if key in _MEMORY:
            _MEMORY.move_to_end(key)
            return _MEMORY[key]
    target = window_path(state, stem, origin)
    found = _load(target)
    if found is None:
        with _MEMORY_LOCK:
            _WAITING[0] += click
        try:
            with model.encoding:  # someone else may have been making this one
                found = _load(target)
                if found is None:
                    x0, y0 = origin
                    found = model.encode(np.ascontiguousarray(rgb[y0: y0 + WINDOW, x0: x0 + WINDOW]))
                    _save(target, found)
        finally:
            with _MEMORY_LOCK:
                _WAITING[0] -= click
    if remember:
        with _MEMORY_LOCK:
            _MEMORY[key] = found
            while len(_MEMORY) > KEEP:
                _MEMORY.popitem(last=False)
    return found


def field_picture(state: ProjectState, tile_id: str) -> np.ndarray:
    """`render.field_rgb`, with the last few fields kept in memory: every click needs its field's picture."""
    path, key = render.field_path(state, tile_id), (str(state.root), tile_id)
    with _MEMORY_LOCK:
        kept = _FIELDS.get(key)
        if kept is not None and kept[0] is not None and kept[0] == cache.signature(path):
            _FIELDS.move_to_end(key)
            disk_cache.used(path)
            return kept[1]
    rgb = render.field_rgb(state, tile_id)
    with _MEMORY_LOCK:
        _FIELDS[key] = (cache.signature(path), rgb)
        while len(_FIELDS) > KEEP_FIELDS:
            _FIELDS.popitem(last=False)
    return rgb


def outline(state: ProjectState, tile_id: str, stain: str, points: list[tuple[float, float, bool]]) -> dict:
    """`ClickOutliner.outline` for clicks on a field, from the window around the first click."""
    model = clicks.outliner(checkpoint(state))
    rgb = field_picture(state, tile_id)
    origin = window_origin(rgb.shape, points[0][0], points[0][1])
    found = embedding(state, rgb, field_stem(state, tile_id), origin, model, click=True)
    return model.outline(found, origin, rgb, stain, points)


def _make_field(state: ProjectState, tile_id: str, remember: bool, more_urgent) -> bool:
    """Make every missing window of a field; False when it stopped for something more urgent."""
    stem = field_stem(state, tile_id)
    marker = marker_path(state, stem)
    if marker.exists():
        disk_cache.used(marker)
        return True
    model = clicks.outliner(checkpoint(state))
    rgb = render.field_rgb(state, tile_id)
    origins = window_origins(rgb.shape)
    for origin in origins:
        while _WAITING[0]:  # someone clicked where nothing is made yet
            time.sleep(0.01)
        if more_urgent():
            return False
        embedding(state, rgb, stem, origin, model, remember)
    disk_cache.write(marker, json.dumps({"tile_id": tile_id, "shape": list(rgb.shape[:2]), "windows": [list(o) for o in origins]}).encode())
    return True


# ---------------------------------------------------------------- the background queue

@dataclass
class _Task:
    state: ProjectState
    tile_id: str
    priority: int
    queued: float = field(default_factory=time.monotonic)
    running: bool = False


_TASKS: dict[tuple[str, str], _Task] = {}
_FAILED: dict[tuple[str, str], tuple[str, float]] = {}
_COND = threading.Condition()
_WORKER: threading.Thread | None = None


def _ask(state: ProjectState, tile_id: str, priority: int) -> None:
    global _WORKER
    key = (str(state.root), tile_id)
    with _COND:
        failed = _FAILED.get(key)
        if failed and time.monotonic() - failed[1] < RETRY_SECONDS:
            return
        task = _TASKS.get(key)
        if task is None:
            _TASKS[key] = _Task(state, tile_id, priority)
        else:
            task.priority = min(task.priority, priority)
        if _WORKER is None or not _WORKER.is_alive():
            _WORKER = threading.Thread(target=_work, name="stainid-sam-embeddings", daemon=True)
            _WORKER.start()
        _COND.notify_all()


def _next_task() -> _Task:
    with _COND:
        while True:
            waiting = [t for t in _TASKS.values() if not t.running]
            if waiting:
                task = min(waiting, key=lambda t: (t.priority, t.queued))
                task.running = True
                return task
            _COND.wait(timeout=30)


def _more_urgent(task: _Task):
    def check() -> bool:
        with _COND:
            return any(not t.running and t.priority < task.priority for t in _TASKS.values())
    return check


def _run(task: _Task) -> bool:
    """Make one field's embeddings; True when the task is finished (made, not wanted, or failed)."""
    weights = checkpoint(task.state)
    if not weights.exists():
        return True
    if task.priority == AHEAD and not clicks.encodes_on_gpu(weights):
        return True  # on the CPU only the fields people are on or about to see
    return _make_field(task.state, task.tile_id, task.priority == NOW, _more_urgent(task))


def _work() -> None:
    while True:
        task = _next_task()
        key = (str(task.state.root), task.tile_id)
        finished = True
        try:
            finished = _run(task)
            _FAILED.pop(key, None)
        except Exception as error:  # noqa: BLE001 - the page says it is not ready; a click makes the window itself
            log.warning("Segment Anything features of %s not made: %s", task.tile_id, error)
            _FAILED[key] = (f"{type(error).__name__}: {error}", time.monotonic())
        finally:
            with _COND:
                task.running = False
                if finished:
                    _TASKS.pop(key, None)


# ---------------------------------------------------------------- what the routes call

def load(state: ProjectState) -> None:
    """Load the Segment Anything model now (server start, a background thread), so no request waits for torch."""
    try:
        clicks.outliner(checkpoint(state))
    except Exception as error:  # noqa: BLE001 - a click loads it then, and says what is wrong
        log.warning("Segment Anything not loaded ahead: %s", error)


def busy() -> bool:
    """Whether the encoder is making some field's features now (a click on a field not made yet waits for a window)."""
    with _COND:
        return any(t.running for t in _TASKS.values())


def status(state: ProjectState, tile_id: str) -> dict:
    """Whether a field's clicks will be outlined at once ("ready"), or its features are being made ("preparing";
    asks for them first), or cannot be ("unavailable": no weights; "failed")."""
    if not checkpoint(state).exists():
        return {"ready": False, "state": "unavailable"}
    try:
        marker = marker_path(state, field_stem(state, tile_id))
    except (KeyError, OSError):
        return {"ready": False, "state": "unavailable"}
    if marker.exists():
        return {"ready": True, "state": "ready"}
    failed = _FAILED.get((str(state.root), tile_id))
    if failed and time.monotonic() - failed[1] < RETRY_SECONDS:
        return {"ready": False, "state": "failed", "error": PLAIN_FAILED}  # the detail (failed[0]) is in the log
    _ask(state, tile_id, NOW)
    return {"ready": False, "state": "preparing"}


def prepare_field(state: ProjectState, meta: dict, tile_id: str, finished: set[str]) -> dict:
    """Someone opened a field: make it first, then the next `NEXT_FIELDS` of the session they have not `finished`.
    Returns the field's `status`."""
    now = status(state, tile_id)
    if now["state"] == "unavailable":
        return now
    fields = meta["fields"]
    at = fields.index(tile_id) if tile_id in fields else -1
    after = [t for t in fields[at + 1:] + fields[:max(0, at)] if t != tile_id and t not in finished]
    for tile in after[:NEXT_FIELDS]:
        _ask(state, tile, NEXT)
    return now


def prepare_session(state: ProjectState, meta: dict, force: bool = False) -> None:
    """Make every field of an open outline session ahead (at most every `EVERY` seconds unless `force`)."""
    look = (str(state.root), meta["name"])
    if not meta.get("open") or not checkpoint(state).exists():
        return
    if not force and time.monotonic() - _LOOKED.get(look, -EVERY) < EVERY:
        return
    _LOOKED[look] = time.monotonic()
    for tile_id in meta["fields"]:
        _ask(state, tile_id, AHEAD)
