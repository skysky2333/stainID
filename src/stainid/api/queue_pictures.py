"""Pictures of quick-check queues, made ahead so that the next object shows at once.

A queue session (`stainid.labels.sessions`) shows one object at a time, as the crop the model sees (`label_crops`).
Cutting a crop can need its whole core image decoded (seconds), so the crop of every object a queue may serve
(`sessions.may_serve`) is cut ahead on the background picture queue (`pyramid`): one task per core image, which reads
the image once (its pyramid when it has one, else one decode) for all its objects, the labelled objects of every
session on that core included. `prepare` queues what is missing; it runs when a queue session is made and whenever
the queue is asked for objects or its progress, so fields prepared later follow. `ready` says which objects have their
picture, so the queue hands those out first and the page shows "pictures ready: 1,240 of 1,500" meanwhile.
"""
from __future__ import annotations

import time

import pandas as pd

from stainid import cache
from stainid.api import label_crops, pyramid
from stainid.api.state import ProjectState
from stainid.labels import frames, pool, rounds, sessions, store

AHEAD, NOW = 2, 0  # priorities on the picture queue: made ahead; someone is about to see it
EVERY = 10.0  # seconds between two looks for missing pictures of the same session
_LOOKED: dict[tuple[str, str], float] = {}


def _reviews(state: ProjectState):
    return state.project.output("reviews")


def ready(state: ProjectState, stain: str, objects: pd.DataFrame) -> pd.Series:
    """Whether each object's crop is cut (a boolean per row)."""
    if objects.empty:
        return pd.Series(False, index=objects.index)
    return label_crops.file_names(state, stain, objects).isin(cache.names(state.cache_dir("label_crops")))


def _cores(state: ProjectState, stain: str, objects: pd.DataFrame) -> pd.Series:
    """The core image each object lies on ("" for a pilot field of an earlier study, which is quick to read)."""
    tiles = state.tiles
    core_of_tile = dict(zip(tiles.tile_id, tiles.core_id.astype(str))) if not tiles.empty else {}
    place = {}
    for tile_id in objects.tile_id.unique():
        kind = label_crops.frame(state, tile_id, stain)
        place[tile_id] = core_of_tile.get(tile_id, "") if kind == frames.FIELD else tile_id if kind == frames.CORE else ""
    return objects.tile_id.map(place)


def _wanted(state: ProjectState, stain: str) -> pd.DataFrame:
    """Every object any queue session of this stain may serve."""
    root = _reviews(state)
    queues = [m for m in (sessions.meta(root, n) for n in sessions.names(root))
              if m["kind"] == "queue" and m["stain"] == stain and not rounds.is_round(m)]  # old round-1 sessions take no work
    if not queues:
        return pd.DataFrame(columns=["tile_id", "x", "y"])
    objects, labels = pool.all_objects(state.project, stain), store.current(root)
    return pd.concat([sessions.may_serve(m, objects, labels) for m in queues]).drop_duplicates("key")


def _spots(stain: str, objects: pd.DataFrame) -> list[label_crops.Spot]:
    return [label_crops.Spot(stain, r.tile_id, float(r.x), float(r.y)) for r in objects.itertuples()]


def cut(state: ProjectState, stain: str, core_id: str) -> int:
    """Cut the missing crops of every queue object on one core image ("": on the pilot fields). Returns how many."""
    wanted = _wanted(state, stain)
    mine = wanted[_cores(state, stain, wanted) == core_id] if len(wanted) else wanted
    if core_id:
        return label_crops.cut_core(state, core_id, stain, _spots(stain, mine))
    made = 0
    for spot in _spots(stain, mine):
        if not label_crops.path(state, spot).exists():
            label_crops.get(state, spot)
            made += 1
    return made


def _queue(state: ProjectState, stain: str, core_id: str, priority: int) -> None:
    pyramid._ask(f"queue pictures {state.root} {stain} {core_id}", priority, lambda: cut(state, stain, core_id), keep=True)


def prepare(state: ProjectState, meta: dict, objects: pd.DataFrame, labels: pd.DataFrame, first: pd.DataFrame | None = None,
            force: bool = False) -> dict:
    """Queue the cutting of every picture this queue session may need and has not got (at most every `EVERY` seconds
    unless `force`); the cores of `first` (objects someone is about to see) go to the front. Returns how many of the
    pictures are `ready` of the `total`."""
    return prepare_wanted(state, meta, sessions.may_serve(meta, objects, labels), first, force)


def prepare_wanted(state: ProjectState, meta: dict, wanted: pd.DataFrame, first: pd.DataFrame | None = None, force: bool = False) -> dict:
    """`prepare` for these objects (tile_id, x, y) the session may serve."""
    made = ready(state, meta["stain"], wanted)
    look = (str(state.root), meta["name"])
    if force or time.monotonic() - _LOOKED.get(look, 0.0) >= EVERY:
        _LOOKED[look] = time.monotonic()
        missing = wanted[~made]
        for core_id in _cores(state, meta["stain"], missing).unique():
            _queue(state, meta["stain"], core_id, AHEAD)
    if first is not None and len(first):
        urgent = first[~ready(state, meta["stain"], first)]
        for core_id in _cores(state, meta["stain"], urgent).unique():
            _queue(state, meta["stain"], core_id, NOW)
    return {"ready": int(made.sum()), "total": int(len(wanted))}


def picture(state: ProjectState, stain: str, tile_id: str, x: float, y: float):
    """The crop of one queue object (a path), cut now when it was not made ahead: then the whole core image is read
    once and every queue object on it is cut on the way. FileNotFoundError when there is no picture of it."""
    spot = label_crops.Spot(stain, tile_id, x, y)
    try:
        return label_crops.get(state, spot)
    except label_crops.NotReady:
        core_id = label_crops.core_of(state, spot)[0]
        wanted = _wanted(state, stain)
        on_core = wanted[_cores(state, stain, wanted) == core_id] if len(wanted) else wanted
        label_crops.cut_core(state, core_id, stain, [spot, *_spots(stain, on_core)])
        return label_crops.get(state, spot)
