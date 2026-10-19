"""The annotation rounds over the v2 candidates (`stainid.labels.rounds`): making a round session, and for the people
labelling the next tile (field) with its candidates, live updates (who else is on it, progress in tiles), the field page
of round 2 with every candidate, one-click Segment Anything outlines (`api.sam_embeddings`), and saving a tile.

A tile is what people claim, in both rounds: the tile handed to someone (or opened by them) is held for them while their
page is open (`_TILES_HELD`, renewed by the page's live calls), so the others are handed other tiles; a tile saved as
finished ("Save and continue") is done for that round and, once `labels_needed` people finished it, handed to nobody
else. Pages keep every change on the page until the person saves: `save_tile` writes all of a tile's labels at once
(and, when it finishes the tile, the calls left as the AI made them, confirmed, and its `done` label), or nothing.

The generic session routes (`routes.annotate`) hand round sessions to the functions here; the paths only round sessions
have are registered on this router.

Practice sessions (`practice-2-<stain>`) are made up on the fly for the tutorial a new annotator goes through: one
real field of the store, shown exactly like a round session, but nothing anyone does
there is saved: answers are echoed back and never reach the label store, decisions of others are not shown, nothing is
held. Who finished the tutorial is kept in `<reviews>/tutorials.json` (`/api/tutorial`).
"""
from __future__ import annotations

import json
import logging
import re
import secrets
import threading
import time
from collections import OrderedDict

import cv2
import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel

from stainid import candidates as candidate_store_ids
from stainid.api import render, sam_embeddings, work
from stainid.api.routes.annotate import (
    LAYERS,
    Mark,
    NewSession,
    asks_subtypes,
    busy,
    class_of,
    field_tile,
    free_name,
    measure_added_soon,
    root,
    session_meta,
    vocabulary,
    writable,
)
from stainid.api.session import PREVIEW, annotator, require_admin
from stainid.api.state import get_state
from stainid.api.util import fast_json
from stainid.labels import added, candidate_store, decisions, exclusions, objectedit, readiness, rounds, sessions, starting, status, store
from stainid.labels import vocabulary as vocab
from stainid.labels.tasks import STAINS
from stainid.tables import write_text_atomic

router = APIRouter(tags=["annotation rounds"])
log = logging.getLogger(__name__)
admin = [Depends(require_admin)]
_TILES_HELD = sessions.Reservations(rounds.TILE_HOLD_SECONDS)  # tiles on someone's screen, per round and stain
OUTLINE_SIMPLIFY_PX = 1.5
SHAPE_POINTS = 96  # at most this many corners per outline sent to the field page


def tile_hold(meta: dict) -> tuple:
    """Holds on tiles are shared by every session of a round and stain: a tile is the same work whichever session shows it."""
    return get_state().id, "tiles", rounds.round_of(meta), meta["stain"]


def tiles_held(meta: dict) -> dict[str, set[str]]:
    return {} if meta.get("practice") else _TILES_HELD.held(tile_hold(meta))


def finished_tiles(meta: dict, labels: pd.DataFrame | None = None) -> dict[str, set[str]]:
    """Who finished each tile in this round (in any session of the stain)."""
    if meta.get("practice"):
        return {}
    return store.reviewed_fields(store.current(root()) if labels is None else labels, meta["stain"], rounds.done_kind(meta))


def round_items(meta: dict) -> pd.DataFrame:
    """An old "Check detection's calls" (round 1) session's candidates in its order, for its progress (no such session is
    made any more; its labels stay valid object labels)."""
    state = get_state()
    return rounds.items(state.project, meta, state.tiles)


def session_tiles(meta: dict) -> list[str]:
    """The tiles of a round session in the order they are handed out: round 1 those of its candidates, round 2 its fields."""
    return rounds.tiles_of(round_items(meta)) if rounds.round_of(meta) == 1 else list(meta["fields"])


def tile_of(meta: dict, tile_id: str) -> dict:
    """One tile of a round session (404 when it is not one of the session's)."""
    if rounds.round_of(meta) == 1 and tile_id not in set(session_tiles(meta)):
        raise HTTPException(404, f"{tile_id} is not part of this session")
    return field_tile(meta, tile_id)


def decided_now(stain: str, stale_ok: bool = False) -> decisions.State:
    state = get_state()
    return decisions.state(state.project, stain, state.tiles, stale_ok)


def store_ready(stain: str, tiles: pd.DataFrame) -> pd.DataFrame:
    """The tiles of this stain whose core has candidates (and a picture)."""
    tiles = tiles[tiles.stain == stain] if not tiles.empty else tiles
    return tiles[[b is None for b in readiness.store_blockers(get_state().project, tiles)]] if not tiles.empty else tiles


# ---------- making a round session (admin) ----------

def _scope(body: NewSession) -> list[str]:
    """The fields a new round session draws from: the ones chosen, else every ready field (of the chosen TMAs)."""
    state = get_state()
    if body.tile_ids:
        tiles = state.tiles[state.tiles.stain == body.stain]
        wanted = list(dict.fromkeys(body.tile_ids))
        unknown = [t for t in wanted if t not in set(tiles.tile_id)]
        if unknown:
            raise HTTPException(400, f"unknown {body.stain} fields: {', '.join(unknown[:5])}")
        chosen = tiles[tiles.tile_id.isin(set(wanted))].drop_duplicates("tile_id")
        blocked = [b for b in readiness.store_blockers(state.project, chosen) if b]
        if blocked:
            raise HTTPException(400, f"{len(blocked)} of these fields cannot be labelled yet. " + " ".join(sorted({b["reason"] for b in blocked})))
        return wanted
    tiles = state.labelling_tiles
    tiles = tiles[tiles.stain == body.stain] if not tiles.empty else tiles
    if body.tmas and not tiles.empty:
        tiles = tiles[tiles.tma.astype(str).isin({str(t) for t in body.tmas})]
    ready = store_ready(body.stain, tiles)
    if ready.empty:
        raise HTTPException(400, f"No {body.stain} core has detection's candidates yet" + (" on these TMAs" if body.tmas else "")
                            + ": run the v2 detection (“Find candidates”) first.")
    return list(ready.tile_id)


def create_round_session(body: NewSession) -> dict:
    if body.stain not in STAINS and body.stain not in set(get_state().tiles.stain):
        raise HTTPException(400, f"unknown stain {body.stain}")
    if body.round == 1:
        raise HTTPException(400, "“Check detection’s calls” sessions are no longer made: make a “Fix the field” or a “Mark bad areas” session")
    if body.round not in rounds.KIND:
        raise HTTPException(400, "round must be 2 (Fix the field)")
    if body.level is not None and body.level not in set(rounds.LEVELS.values()):
        raise HTTPException(400, f"level must be one of {[v for v in rounds.LEVELS.values() if v]} (or empty for every candidate)")
    name = body.name or free_name(body.title or f"{body.stain} round {body.round}", body.stain, rounds.KIND[body.round])
    if not re.fullmatch(r"[A-Za-z0-9_\-]+", name) or name in sessions.names(root()) or name == sessions.REFERENCE:
        raise HTTPException(409, "a session with this name exists, or the name has other characters than letters, digits, - and _")
    needs = max(1, body.labels_needed)
    scope = _scope(body)
    common = {"title": body.title or name, "stain": body.stain, "instructions": body.instructions, "subtypes": body.subtypes, "ai_view": "full",
              "source": rounds.SOURCE, "round": body.round, "level": body.level, "labels_needed": needs, "per_field": needs}
    asked = rounds.ALL_FIELDS if body.n_fields is None else body.n_fields
    seed = body.seed if body.seed is not None else secrets.randbelow(2**31)
    if body.tile_ids:
        fields, left = scope, len(scope)
    else:
        fields, left = _pick_fields(body.stain, scope, asked, seed, body.spread)
    draw = {} if body.tile_ids else {"field_seed": seed, "fields_asked": asked, "fields_remaining": left, "fields_spread": body.spread}
    draw["fields_from_checked_areas"] = len(checked_areas_fields(body.stain, fields))
    meta = sessions.create(root(), name, kind="fields", tools=["class", "point", "outline", "tag"], rule="random", fields=fields, scope=fields,
                           **common, **draw)
    if body.open:
        meta = sessions.update(root(), name, open=True)
    warm(meta, force=True)
    return meta


def assigned_fields(but: str = "") -> set[str]:
    """The fields of the open labelling sessions (but this one)."""
    out: set[str] = set()
    for name in sessions.names(root()):
        meta = sessions.meta(root(), name)
        if meta.get("open") and name != but:
            out |= set(meta.get("fields", []))
    return out


def remaining(stain: str, scope: list[str], but: str = "") -> list[str]:
    """The fields of the scope that are not annotated yet and not assigned to another open session (`rounds.remaining_fields`)."""
    return rounds.remaining_fields(scope, set(store.reviewed_fields(store.current(root()), stain)), assigned_fields(but))


def checked_areas_fields(stain: str, fields: list[str]) -> set[str]:
    """The fields whose core image's bad areas a person checked (`labels.status.areas_checked`, as the minimap reads)."""
    return rounds.from_checked(fields, get_state().tiles[get_state().tiles.stain == stain], status.areas_checked(store.current(root()), stain))


def _pick_fields(stain: str, scope: list[str], n: int, seed: int, spread: bool = False, but: str = "") -> tuple[list[str], int]:
    """The fields of a new "Fix the field" session and how many remained: n drawn at random from the remaining ones
    (`rounds.draw_fields`; -1 or more than there are: every one), first from the fields whose core image's bad areas a
    person checked, then the rest; with `spread` spread over TMAs and groups (within each of the two pools)."""
    left = remaining(stain, scope, but)
    if not left:
        raise HTTPException(400, f"Every {stain} field there has been checked or belongs to an open session already.")
    first = checked_areas_fields(stain, left)
    if spread:
        tiles = get_state().labelling_tiles
        tiles = tiles[(tiles.stain == stain) & tiles.tile_id.isin(set(left))]
        wanted = len(left) if n < 0 else n
        picked = sessions.pick_fields(tiles[tiles.tile_id.isin(first)], set(), min(wanted, len(first)), seed) if first else []
        if len(picked) < wanted:
            picked += sessions.pick_fields(tiles[~tiles.tile_id.isin(first)], set(), wanted - len(picked), seed)
        return picked, len(left)
    return rounds.draw_fields(left, n, seed, first or None), len(left)


def add_fields(meta: dict, n: int, tile_ids: list[str] | None, seed: int | None = None) -> dict:
    """More fields for a "Fix the field" session: these (checked to have candidates) or n more drawn at random from the
    remaining ones (-1: every remaining one)."""
    if rounds.round_of(meta) != 2:
        raise HTTPException(400, "a “Check detection’s calls” session keeps the candidates it was made with; make a new session for more")
    scope = _scope(NewSession(stain=meta["stain"], round=2, tile_ids=tile_ids))
    left = [t for t in scope if t not in set(meta["fields"])]
    picked = left if tile_ids else _pick_fields(meta["stain"], left, n if n != 0 else 1, secrets.randbelow(2**31) if seed is None else seed,
                                                 bool(meta.get("fields_spread")), meta["name"])[0]
    if not picked:
        raise HTTPException(400, f"no more {meta['stain']} fields can be added")
    updated = sessions.add_fields(root(), meta["name"], picked)
    updated = sessions.update(root(), meta["name"], scope=updated["fields"],
                              fields_from_checked_areas=len(checked_areas_fields(meta["stain"], updated["fields"])))
    warm(updated, force=True)
    return updated


def warm(meta: dict, force: bool = False) -> None:
    """Make ahead what people will see: the Segment Anything features and the pictures of a "Fix the field" session's
    fields (when the session is open). An old round-1 session needs nothing: nobody works on it any more."""
    state = get_state()
    if meta.get("practice") or rounds.round_of(meta) != 2:
        return
    sam_embeddings.prepare_session(state, meta, force=force)
    prepare_pictures(meta, force)


def prepare_pictures(meta: dict, force: bool = False) -> work.Task | None:
    """Cut the pictures of an open round-2 session's fields in the background (one decode per core image), so that
    nobody waits for them; its progress is the session's `preparing`."""
    state = get_state()
    if not meta.get("open") or not meta.get("fields"):
        return None
    made = render.fields_made(state)  # one listing (a session may have hundreds of fields)
    if all(t in made for t in meta["fields"]):
        return None
    return work.submit("pictures", ("pictures", str(state.root), meta["name"]),
                       lambda task: render.field_jpegs(state, [t for t in meta["fields"] if _known(t)], task.step), work.AHEAD,
                       "Preparing the session's pictures", fresh=lambda task: not force)


def _known(tile_id: str) -> bool:
    try:
        get_state().tile(tile_id)
    except KeyError:
        return False
    return True


def warm_up(state) -> None:
    """When the server starts (a background thread): load Segment Anything (its import alone takes seconds, never in a
    request), make the decisions of the stains of the open round sessions, and queue the pictures and first fields of
    those sessions."""
    from stainid.api.state import use_project  # noqa: PLC0415

    with use_project(state.id):
        if state.project.model("sam").exists():
            sam_embeddings.load(state)
        for name in sessions.names(root()):
            meta = sessions.meta(root(), name)
            if not (rounds.is_round(meta) and meta.get("open")):
                continue
            decided_now(meta["stain"])
            if rounds.round_of(meta) == 2:
                finished = finished_tiles(meta)
                for tile in [t for t in meta["fields"] if len(finished.get(t, set())) < rounds.needed(meta)][:NEXT_FIELDS]:
                    work.submit("field", _prepare_key(meta, tile), lambda task, t=tile, m=meta: prepare_field(m, t, task), work.AHEAD,
                                "Preparing the first fields")
                prepare_pictures(meta)


@router.get("/labels/rounds/scope", dependencies=admin)
def scope_overview(stain: str) -> dict:
    """What a new round session can draw from, per TMA: how many cores and fields have the AI's candidates (are ready),
    and the choices of the advanced settings."""
    state = get_state()
    tiles = state.labelling_tiles
    tiles = tiles[tiles.stain == stain] if not tiles.empty else tiles
    ready = set(store_ready(stain, tiles).tile_id) if not tiles.empty else set()
    left = set(remaining(stain, sorted(ready))) if ready else set()
    out = []
    for tma, group in (tiles.groupby(tiles.tma.astype(str)) if not tiles.empty else []):
        ok = group[group.tile_id.isin(ready)]
        out.append({"tma": tma, "cores": int(group.core_id.nunique()), "cores_ready": int(ok.core_id.nunique()), "fields": int(len(group)),
                    "fields_ready": int(len(ok)), "fields_remaining": int(group.tile_id.isin(left).sum())})
    out.sort(key=lambda t: (len(t["tma"]), t["tma"]))
    return {"stain": stain, "tmas": out, "cores": sum(t["cores"] for t in out), "cores_ready": sum(t["cores_ready"] for t in out),
            "fields_ready": len(ready), "fields_remaining": len(left), "levels": rounds.LEVELS,
            "from_pool": (candidate_store.folder(state.project, stain) / ".from_pool.json").exists()}


# ---------- what the generic session routes ask ----------

NO_ROUND_ONE = ("“Check detection’s calls” is no longer used: its labels stay in the label log for training and evaluation. "
                "Ask for a “Fix the field” session instead.")

def detail(meta: dict, person: str | None) -> dict:
    """The extra facts of a round session for its pages."""
    words = vocabulary(meta["stain"])
    out = {"round": rounds.round_of(meta), "labels_needed": rounds.needed(meta), "level": rounds.level(meta),
           "remove_class": candidate_store.negative_class(words), "segment_anything": get_state().project.model("sam").exists()}
    return out | {"progress": progress(meta)}


def progress(meta: dict, finished: dict[str, set[str]] | None = None) -> dict:
    """How far a round session is (live), in tiles: done, in progress, left, and how many each person finished."""
    finished = finished_tiles(meta) if finished is None else finished
    return rounds.progress(session_tiles(meta), finished, tiles_held(meta), rounds.needed(meta))


def ready_fields(meta: dict) -> set[str]:
    state = get_state()
    tiles = state.tiles[state.tiles.tile_id.isin(set(meta["fields"]))]
    return set(store_ready(meta["stain"], tiles).tile_id)


def hand_out(meta: dict, person: str, skip: str, ready: set[str] | None = None, prefer: set[str] | None = None) -> tuple[str | None, int]:
    """The next tile for this person, held for them from now on (`skip`: the tile they put aside, released)."""
    if meta.get("practice"):
        return session_tiles(meta)[0] if session_tiles(meta) else None, 0
    if skip:
        _TILES_HELD.release(tile_hold(meta), person, [skip])
    tile, taken = rounds.next_tile(session_tiles(meta), finished_tiles(meta), person, tiles_held(meta), rounds.needed(meta), ready,
                                   {skip} if skip else None, prefer)
    if tile:
        _TILES_HELD.hold(tile_hold(meta), person, [tile])
        busy(meta["name"], tile, person)
    return tile, taken


def next_field(meta: dict, person: str, skip: str = "") -> dict:
    """Round 2: the next field for this person (held for them), how many are still being prepared and how many others are on."""
    ready = set(meta["fields"]) if meta.get("practice") else ready_fields(meta)
    tile, taken = hand_out(meta, person, skip, ready)
    return {"tile_id": tile, "waiting": len(set(meta["fields"]) - ready), "taken": taken}


def session_fields(meta: dict, person: str | None) -> list[dict]:
    """The fields of a round-2 session as its overview page lists them."""
    if meta.get("practice"):
        return [{"tile_id": t, "done_by_me": False, "annotators_done": 0, "open_by_others": 0, "open_by": [], "prepared": True, "done": False,
                 "done_by": []} for t in meta["fields"]]
    reviewed = finished_tiles(meta)
    ready, held = ready_fields(meta), tiles_held(meta)
    return [{"tile_id": t, "done_by_me": bool(person) and person in reviewed.get(t, set()), "annotators_done": len(reviewed.get(t, set())),
             "open_by_others": len(held.get(t, set()) - {person}), "open_by": sorted(held.get(t, set()) - {person}), "prepared": t in ready,
             "done": len(reviewed.get(t, set())) >= rounds.needed(meta), "done_by": sorted(reviewed.get(t, set()))} for t in meta["fields"]]


def admin_fields(meta: dict) -> list[dict]:
    labels = store.current(root())
    working = sessions.labelled_by(labels[labels.tile_id.isin(set(meta["fields"]))], meta["stain"])
    return rounds.field_states(meta, finished_tiles(meta, labels), working, tiles_held(meta), ready_fields(meta))


def _field_candidates(meta: dict, tile: dict) -> pd.DataFrame:
    state = get_state()
    return candidate_store.in_field(state.project, meta["stain"], tile, state.project.context_px, rounds.level(meta))


def _simplified(points) -> list[list[float]] | None:
    if not points:
        return None
    simple = cv2.approxPolyDP(np.asarray(points, np.float32).reshape(-1, 1, 2), OUTLINE_SIMPLIFY_PX, True)[:, 0, :]
    return simple.round(1).tolist() if len(simple) >= 3 else None


def checks(found: pd.DataFrame, decided: decisions.State) -> dict[str, dict]:
    """Who decided each candidate of a field, and the class most of them gave."""
    out = {}
    for c in found.candidate_id:
        if decided.count(c):
            identity, subtype = decided.calls[c]
            out[c] = {"by": decided.by(c), "identity": identity, "subtype": subtype, "time": decided.times.get(c)}
    return out


_OBJECTS: OrderedDict[tuple, tuple[vocab.Vocabulary, list[dict]]] = OrderedDict()
_OBJECTS_LIMIT = 96
_OBJECTS_LOCK = threading.Lock()
NEXT_FIELDS = 2  # fields made ready ahead for a person, after the one they open


def _objects_key(meta: dict, tile: dict) -> tuple:
    state = get_state()
    return candidate_store.field_key(state.project, meta["stain"], tile, state.project.context_px, rounds.level(meta))


def _shape(points) -> list[int] | None:
    """An outline as the field page draws it: simplified (Douglas–Peucker, `OUTLINE_SIMPLIFY_PX`, coarser until it has at
    most `SHAPE_POINTS` corners), whole pixels, flat [x0, y0, x1, y1, …]."""
    if not points:
        return None
    ring = np.asarray(points, np.float32).reshape(-1, 1, 2)
    tolerance = OUTLINE_SIMPLIFY_PX
    simple = cv2.approxPolyDP(ring, tolerance, True)
    while len(simple) > SHAPE_POINTS:
        tolerance *= 1.5
        simple = cv2.approxPolyDP(ring, tolerance, True)
    simple = np.rint(simple[:, 0, :]).astype(np.int64)
    return simple.reshape(-1).tolist() if len(simple) >= 3 else None


def _field_shapes(meta: dict, tile: dict) -> tuple[list[dict], dict[str, list]]:
    """The candidates of a field as the round-2 page shows them (the AI's call, the default outline) and every outline
    of each (`field_masks`), kept per field until the store file or the label classes change. Callers must not change
    them."""
    state = get_state()
    words = vocabulary(meta["stain"])
    key = _objects_key(meta, tile)
    with _OBJECTS_LOCK:
        kept = _OBJECTS.get(key)
        if kept is not None and kept[0] is words:
            _OBJECTS.move_to_end(key)
            return kept[1], kept[2]
    objects, masks = [], {}
    passing = starting.if_accepted(state.project, meta["stain"], str(tile["core_id"]))
    for r in _field_candidates(meta, tile).rename(columns={"class": "klass"}).itertuples():
        identity, subtype = candidate_store.ai_call(words, r.decision, r.klass, r.subtype, r.reason)
        would = passing.get(r.candidate_id)
        if_accepted = list(candidate_store.ai_call(words, *would)) if would is not None else None
        found_masks, default = candidate_store.choices(r)  # [the stain's, SAM small, medium, large]: what Tab cycles through
        shapes = [_shape(m) for m in found_masks]
        masks[r.candidate_id] = shapes
        objects.append({"candidate_id": r.candidate_id, "x": round(float(r.x), 1), "y": round(float(r.y), 1), "own": bool(r.own),
                        "radius_px": round(candidate_store.radius(r, state.project.pixel_size_um), 1), "ai_value": identity, "ai_subtype": subtype,
                        "ai_probability": None if pd.isna(r.probability) else round(float(r.probability), 3), "decision": r.decision, "reason": r.reason,
                        "masks": [m if i == default else None for i, m in enumerate(shapes)], "mask": default,
                        "more_masks": sum(m is not None for i, m in enumerate(shapes) if i != default),
                        "if_accepted": if_accepted, "rule": would[3] if would is not None else (r.reason if str(r.reason).startswith("rule:") else ""),
                        "threshold": None if pd.isna(getattr(r, "threshold", np.nan)) else round(float(r.threshold), 3)})
    with _OBJECTS_LOCK:
        _OBJECTS[key] = (words, objects, masks)
        _OBJECTS.move_to_end(key)
        while len(_OBJECTS) > _OBJECTS_LIMIT:
            _OBJECTS.popitem(last=False)
    return objects, masks


def excluded_ids(meta: dict, ids) -> set[str]:
    """Which of these candidates lie in an excluded area (`stainid.exclusions`: share at least `EXCLUDED_AT`): drawn
    faded on the round pages and not asked about."""
    from stainid import exclusions as excluded_areas  # noqa: PLC0415

    by_core: dict[str, list[str]] = {}
    for c in ids:
        try:
            by_core.setdefault(candidate_store_ids.parse_id(c)[1], []).append(c)
        except ValueError:
            continue
    out: set[str] = set()
    for core_id, group in by_core.items():
        shares = excluded_areas.shares(get_state().project, meta["stain"], core_id, group)
        out |= {c for c, v in shares.items() if v >= excluded_areas.EXCLUDED_AT}
    return out


def field_objects(meta: dict, tile: dict) -> list[dict]:
    """The field's candidates with only their default outline (flat whole pixels; the others: `field_masks`), each
    with whether it lies in an excluded area (`excluded`: drawn faded, not asked about)."""
    found = _field_shapes(meta, tile)[0]
    out = excluded_ids(meta, [o["candidate_id"] for o in found])
    return [{**o, "excluded": o["candidate_id"] in out} for o in found]


def field_masks(meta: dict, tile: dict) -> dict[str, list]:
    """Every outline of every candidate of the field (`candidate_store.CHOICES`: the stain outline, then Segment
    Anything's small, medium and large; None where it has none), flat whole pixels: what the page loads after it has
    drawn the field, for Tab."""
    return _field_shapes(meta, tile)[1]


def field_prepared(meta: dict, tile: dict) -> bool:
    """Whether a field opens at once: its picture made, its candidates read and the decisions up to date."""
    state = get_state()
    with _OBJECTS_LOCK:
        kept = _OBJECTS.get(_objects_key(meta, tile))
    return (kept is not None and kept[0] is vocabulary(meta["stain"]) and render.field_path(state, str(tile["tile_id"])).exists()
            and (bool(meta.get("practice")) or decisions.cached(state.project, meta["stain"])))


def _same_core_fields(meta: dict, tile: dict) -> list[str]:
    """The fields of this session on the same core image: cut from one decode of it."""
    state = get_state()
    tiles = state.tiles
    mine = tiles[tiles.tile_id.isin(set(meta.get("fields") or []))]
    return list(mine.tile_id[mine.image_path == tile["image_path"]])


def prepare_field(meta: dict, tile_id: str, task: work.Task | None = None) -> bool:
    """Make what opening a field needs: its picture (and those of the session's other fields on that core, from the same
    decode), its candidates and the decisions."""
    state = get_state()
    tile = state.tile(tile_id)
    step = task.step if task else (lambda *a: None)
    step(0, 3)
    render.field_jpegs(state, [tile_id, *_same_core_fields(meta, tile)], urgent=task is None or task.priority <= work.NOW)
    step(1, 3)
    field_objects(meta, tile)
    step(2, 3)
    if not meta.get("practice"):
        decided_now(meta["stain"])
    step(3, 3)
    return True


def _prepare_key(meta: dict, tile_id: str) -> tuple:
    return ("field", str(get_state().root), meta["name"], tile_id)


def prefetch(meta: dict, tile_id: str, person: str) -> None:
    """Make the next fields this person may be handed ready in the background, so "Save and continue" opens at once."""
    if meta.get("practice"):
        return
    fields = list(meta.get("fields") or [])
    finished, held = finished_tiles(meta), tiles_held(meta)
    at = fields.index(tile_id) if tile_id in fields else -1
    order = fields[at + 1:] + fields[:max(0, at)]
    nexts = [t for t in order if person not in finished.get(t, set()) and len(finished.get(t, set())) < rounds.needed(meta)
             and not (held.get(t, set()) - {person})][:NEXT_FIELDS]
    state = get_state()
    for tile in nexts:
        if not field_prepared(meta, state.tile(tile)):
            work.submit("field", _prepare_key(meta, tile), lambda task, t=tile: prepare_field(meta, t, task), work.AHEAD, "Preparing the next field")


def open_field(meta: dict, tile_id: str, person: str):
    """Round 2: a field with every candidate of the session's level (the field's own and, faded, its neighbours' in the
    margin), the AI's call and outline of each, who decided it already, this person's own saved labels there, and who
    else is on it or finished it. Opening a field holds it for this person. A field that is not ready yet (its core not
    read, its picture not cut) is made ready in the background: a 202 `pending` answer says how far that is."""
    state = get_state()
    tile = field_tile(meta, tile_id)
    practice = bool(meta.get("practice"))
    if not practice:
        _TILES_HELD.hold(tile_hold(meta), person, [tile_id])
        busy(meta["name"], tile_id, person)
    if not field_prepared(meta, tile):
        made = work.answer("field", _prepare_key(meta, tile_id), lambda task: prepare_field(meta, tile_id, task), work.NOW, "Opening the field",
                           fresh=work.within(10))
        if isinstance(made, Response):
            return made
    found = _field_candidates(meta, tile)
    objects = field_objects(meta, tile)
    decided = decisions.State() if practice else decided_now(meta["stain"])
    x0, y0 = render.crop_origin(tile, state.project.context_px)
    labels = store.current(root())
    mine = labels[(labels.source == person) & (labels.tile_id == tile_id) & (labels.stain == meta["stain"])]
    if practice:
        mine = mine.iloc[:0]
    box = {"x": int(tile["x_px"]) - x0, "y": int(tile["y_px"]) - y0, "width": int(tile["width_px"]), "height": int(tile["height_px"])}
    finished = finished_tiles(meta, labels)
    sam = sam_embeddings.prepare_field(state, meta, tile_id, {t for t, people in finished.items() if person in people})
    prefetch(meta, tile_id, person)
    thresholds = [o["threshold"] for o in objects if o.get("threshold") is not None]
    chosen = mine[mine.task == starting.TASK] if "task" in mine else mine.iloc[:0]
    return {"tile_id": tile_id, "box": box, "objects": objects, "checked": checks(found, decided), "layers": LAYERS.get(meta["stain"], []),
            "detection_threshold": float(np.median(thresholds)) if thresholds else None,
            "threshold": _number(chosen.value.iloc[-1]) if len(chosen) else None,
            "mine": mine.replace({np.nan: None}).to_dict(orient="records"), "added": added_states(meta, tile, mine, found),
            "segment_anything": sam["state"] != "unavailable", "segment_ready": bool(sam.get("ready")),
            "open_by": sorted(tiles_held(meta).get(tile_id, set()) - {person}), "done_by": sorted(finished.get(tile_id, set()) - {person}),
            "pixel_size_um": state.project.pixel_size_um}


def _number(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if number != number else number


def added_states(meta: dict, tile: dict, mine: pd.DataFrame, found: pd.DataFrame | None = None) -> list[dict]:
    """Where this person's added objects (class labels on no candidate) stand: being measured, measured (can train the
    model) or could not be measured."""
    project = get_state().project
    if meta.get("practice"):
        return []
    found = _field_candidates(meta, tile) if found is None else found
    own = found[found.own]
    extra = added.load_field(project, meta["stain"], str(tile["tile_id"]))
    radius = candidate_store.match_px(meta["stain"], project.pixel_size_um)
    out = []
    rows = mine[(mine.task == "class") & mine.x.notna()]
    if "candidate" in rows:
        rows = rows[rows.candidate.fillna("").astype(str) == ""]
    for row in rows.itertuples():
        x, y = float(row.x), float(row.y)
        if len(own) and np.hypot(own.x.to_numpy(float) - x, own.y.to_numpy(float) - y).min() <= radius:
            state = "proposed"
            hit = None
        else:
            hit = added.resolve(extra, x, y)
            state = "measuring" if hit is None else "measured" if added.is_true(hit.measured) else "could_not_be_measured"
        reason = "" if hit is None or pd.isna(hit.get("reason")) else str(hit.reason)
        out.append({"x": x, "y": y, "state": state, "reason": reason if state == "could_not_be_measured" else ""})
    return out


class Live(BaseModel):
    tile_id: str = ""  # the tile on the page (held for this person while the page keeps asking)


def live(meta: dict, body: Live, person: str) -> dict:
    """The heartbeat of a round page: the tile on it stays this person's; who else is on it or finished it, (round 2)
    who decided which of its candidates, and the session's progress."""
    if meta.get("practice"):
        return {"checked": {}, "open_by": [], "done_by": [], "progress": progress(meta), "time": time.time()}
    out: dict = {"time": time.time()}
    if body.tile_id:
        tile = tile_of(meta, body.tile_id)
        _TILES_HELD.hold(tile_hold(meta), person, [body.tile_id])
        busy(meta["name"], body.tile_id, person)
        finished = finished_tiles(meta)
        out |= {"open_by": sorted(tiles_held(meta).get(body.tile_id, set()) - {person}),
                "done_by": sorted(finished.get(body.tile_id, set()) - {person})}
        if rounds.round_of(meta) == 2:
            out["checked"] = checks(_field_candidates(meta, tile), decided_now(meta["stain"], stale_ok=True))
    return out | {"progress": progress(meta)}


@router.get("/annotate/{name}/fields/{tile_id}/masks")
def masks_route(name: str, tile_id: str, request: Request, person: str = Depends(annotator)):
    """Every outline of the field's candidates, by candidate id (the field itself sends only the default one)."""
    meta = _round_meta(name, request)
    tile = field_tile(meta, tile_id)
    return fast_json(request, field_masks(meta, tile))


@router.post("/annotate/{name}/live")
def live_route(name: str, body: Live, request: Request, person: str = Depends(annotator)) -> dict:
    meta = session_meta(name, request)
    if not rounds.is_round(meta):
        raise HTTPException(404, f"{name} is not an annotation round")
    return live(meta, body, person)


# ---------- saving ----------

def _near_candidate(state, stain: str, core_id: str, x: str, y: str, candidate: str) -> pd.DataFrame | None:
    """The store's candidates around a candidate id's position: those of the field it is on (kept, `field_rows`), else
    read around it. The whole core is never read for one candidate."""
    try:
        cx, cy = float(x), float(y)
    except ValueError:
        return None
    tiles = state.tiles
    on = tiles[(tiles.core_id.astype(str) == core_id) & (tiles.stain == stain) & (tiles.x_px <= cx) & (cx < tiles.x_px + tiles.width_px)
               & (tiles.y_px <= cy) & (cy < tiles.y_px + tiles.height_px)] if not tiles.empty else tiles
    for tile in on.drop_duplicates("tile_id").to_dict(orient="records"):
        found = candidate_store.field_rows(state.project, stain, tile, state.project.context_px)
        if found is not None and (found.candidate_id == candidate).any():
            return found
    return candidate_store.read_box(state.project, stain, core_id, cx - 1, cy - 1, cx + 1, cy + 1)


def candidate_row(stain: str, candidate: str) -> pd.Series:
    """One candidate of the store, with `x`, `y` in its own field's picture (404 when the store has no such candidate)."""
    state = get_state()
    parts = candidate.split(":")
    if len(parts) < 4 or parts[0] != stain:
        raise HTTPException(400, f"{candidate!r} is not a {stain} candidate")
    found = _near_candidate(state, stain, ":".join(parts[1:-2]), parts[-2], parts[-1].split("~")[0], candidate)
    hit = found[found.candidate_id == candidate] if found is not None else None
    if hit is None or hit.empty:
        raise HTTPException(404, f"no candidate {candidate}")
    row = hit.iloc[0].copy()
    try:
        tile = state.tile(str(row.tile_id))
    except KeyError:
        raise HTTPException(404, f"the field of candidate {candidate} is not part of this project")
    x0, y0 = render.crop_origin(tile, state.project.context_px)
    row["x"], row["y"] = float(row.x_core) - x0, float(row.y_core) - y0
    return row


def clean(meta: dict, mark, person: str) -> dict | None:
    """A label of a round session, checked: about a candidate (`candidate`: its field and position come from the store;
    `keep` records the AI's call as confirmed; an outline of it, e.g. a precise edit) or, in round 2, about the field, an
    object people added or an edit of the excluded areas (None when the brush painted nothing)."""
    stain = meta["stain"]
    second = rounds.round_of(meta) == 2
    allowed = {"class", "outline"} | ({"tag", "note", "done", exclusions.TASK, starting.TASK, objectedit.TASK} if second else set())
    if mark.task not in allowed:
        raise HTTPException(400, f"{mark.task} labels are not part of this session")
    record = {"source": person, "stain": stain, "task": mark.task, "session": meta["name"], "deleted": mark.deleted, "value": mark.value,
              "subtype": None, "points": mark.points, "tile_id": mark.tile_id, "x": mark.x, "y": mark.y, "candidate": mark.candidate or None,
              "meta": dict(mark.meta) if mark.meta else None}
    words = vocabulary(stain)
    if mark.task == exclusions.TASK:  # an edit of the excluded areas (the brush): what was painted and cleared this time
        field_tile(meta, mark.tile_id)
        pngs = dict(mark.pngs) or ({"1": mark.png} if mark.png else {})  # one painted mask per type (`exclusions.KINDS`)
        try:
            rings, holes, kinds = ([], [], []) if mark.deleted else exclusions.from_pngs(pngs, mark.scale)
        except ValueError as error:
            raise HTTPException(400, str(error))
        if not rings:
            return None
        return record | {"id": exclusions.label_id(mark.tile_id), "x": None, "y": None, "points": rings,
                         "value": exclusions.value_text(holes, kinds), "deleted": False}
    if mark.task == starting.TASK:  # the field's acceptance threshold, locked before the first edit
        field_tile(meta, mark.tile_id)
        value = _number(mark.value)
        if value is None or not 0 <= value <= 1:
            raise HTTPException(400, "the acceptance threshold is a number from 0 to 1")
        return record | {"id": starting.label_id(mark.tile_id), "value": f"{value:.3f}", "x": None, "y": None, "points": None, "candidate": None}
    if mark.task == objectedit.TASK and not mark.deleted and mark.value not in objectedit.EDITS:
        raise HTTPException(400, f"an object is replaced by a {' or a '.join(objectedit.EDITS)}")
    if mark.candidate:
        row = candidate_row(stain, mark.candidate)
        record |= {"tile_id": str(row.tile_id), "x": round(float(row.x), 1), "y": round(float(row.y), 1)}
        if second:
            field_tile(meta, record["tile_id"])
        if mark.task == "class" and mark.keep and not mark.deleted:
            identity, subtype = candidate_store.ai_call(words, row.decision, row["class"], row.subtype, row.get("reason", ""))
            record |= {"value": identity, "subtype": (subtype or None) if asks_subtypes(meta) else None, "confirmed": True}
            return record | {"id": store.object_id("class", record["tile_id"], record["x"], record["y"])}
        if mark.task == "outline" and not mark.deleted and candidate_store.choice_index(str(mark.value or ""), row) is not None:
            record |= _chosen_outline(row, mark)
    elif not second or mark.task in ("class", "outline", objectedit.TASK):
        if not second:
            raise HTTPException(400, "a label of this round names its candidate")
        if mark.x is None or mark.y is None:
            raise HTTPException(400, "object labels need x and y")
        field_tile(meta, mark.tile_id)
    else:
        field_tile(meta, mark.tile_id)
    if not mark.deleted:
        if mark.task == "class":
            record["value"], record["subtype"] = class_of(words, mark.value, mark.subtype)
        if mark.task == "outline" and (not mark.points or len(mark.points) < 3 or any(len(p) != 2 for p in mark.points)):
            raise HTTPException(400, "an outline needs at least three [x, y] points")
        from stainid.labels.tasks import FIELD_TAGS

        if mark.task == "tag" and mark.value not in FIELD_TAGS:
            raise HTTPException(400, f"unknown tag {mark.value!r}")
    ids = {"class": lambda: store.object_id("class", record["tile_id"], record["x"], record["y"]),
           "outline": lambda: store.object_id("outline", record["tile_id"], record["x"], record["y"]),
           "tag": lambda: f"tag:{mark.tile_id}:{mark.value}", "note": lambda: f"note:{mark.tile_id}", "done": lambda: f"done:{mark.tile_id}",
           objectedit.TASK: lambda: store.object_id(objectedit.TASK, record["tile_id"], record["x"], record["y"])}
    return record | {"id": ids[mark.task]()}


def _chosen_outline(row, mark) -> dict:
    """The outline label of one of a candidate's outlines (`candidate_store.CHOICES`, by name in `mark.value`; "sam" of
    earlier pages: the output stage 3 kept): the option's own points (in the field's picture), or, for Segment Anything
    where the store has none, the outline the page asked of it live (`mark.points`); confirmed when it is the stain's
    default (the one shown first), so only a changed option counts as an outline fix."""
    found, default = candidate_store.choices(row)
    index = candidate_store.choice_index(str(mark.value), row)
    dx, dy = float(row.x_core) - float(row.x), float(row.y_core) - float(row.y)
    stored = found[index]
    points = _simplified([[px - dx, py - dy] for px, py in stored]) if stored else mark.points
    if not points or len(points) < 3:
        raise HTTPException(400, f"candidate {mark.candidate} has no {mark.value} outline: send the one Segment Anything drew")
    return {"points": points, "confirmed": index == default, "value": candidate_store.CHOICES[index]}


def mask_outline(meta: dict, mark, record: dict) -> dict | None:
    """The outline label of the outline someone chose with a decision (`mark.mask`: an index into the candidate's
    `candidate_store.choices`): None when none was chosen or the call says there is no object there ("not an object",
    "unsure")."""
    outline_id = store.object_id("outline", record["tile_id"], record["x"], record["y"])
    if mark.deleted:  # a decision made with a mask taken back (undo): the mask goes with it
        return {**record, "id": outline_id, "task": "outline", "value": "", "subtype": None, "points": None} if mark.mask is not None else None
    words = vocabulary(meta["stain"])
    if mark.mask is None or record["value"] == candidate_store.negative_class(words) or words.skipped(record["value"]):
        return None
    row = candidate_row(meta["stain"], mark.candidate)
    found, default = candidate_store.choices(row)
    if not 0 <= mark.mask < len(found) or not found[mark.mask]:
        raise HTTPException(400, f"candidate {mark.candidate} has no outline {mark.mask}")
    dx, dy = float(row.x_core) - float(row.x), float(row.y_core) - float(row.y)
    points = _simplified([[px - dx, py - dy] for px, py in found[mark.mask]])
    return {**record, "id": outline_id, "task": "outline", "value": candidate_store.CHOICES[mark.mask],
            "subtype": None, "points": points, "confirmed": mark.mask == default}


class TileSave(BaseModel):
    records: list[Mark] = []  # every change on the page since the last save
    submit: bool = False  # "Save and continue": the tile is finished for this round


def _mine_after(meta: dict, tile_id: str, person: str, batch: list[dict]) -> pd.DataFrame:
    """This person's labels on a tile as they will stand once this batch is saved."""
    labels = store.current(root())
    mine = labels[(labels.source == person) & (labels.tile_id == tile_id) & (labels.stain == meta["stain"])]
    if not batch:
        return mine
    latest = {r["id"]: r for r in mine.to_dict(orient="records")} | {r["id"]: r for r in batch}
    return pd.DataFrame([r for r in latest.values() if not r.get("deleted")], columns=mine.columns)


def _confirmations(meta: dict, tile: dict, person: str, mine: pd.DataFrame) -> list[dict]:
    """Round 2, finishing a field: the whole field counts as checked by this person. Every candidate of it they left
    untouched gets detection's call as their answer (confirmed; whatever others decided), and so do detection's outlines
    of the objects that stay positive."""
    tile_id = str(tile["tile_id"])
    found = _field_candidates(meta, tile)
    own = found[found.own]
    own = own[~own.candidate_id.isin(excluded_ids(meta, list(own.candidate_id)))]  # in an excluded area: not asked, not confirmed
    words = vocabulary(meta["stain"])
    classes = mine[mine.task == "class"]
    said = {k: words.positive(v, s) for k, v, s in zip(classes.id, classes.value, classes.subtype)}
    outlined = set(mine.id[mine.task == "outline"])
    replaced = set(mine.candidate[(mine.task == objectedit.TASK)].dropna().astype(str)) if "candidate" in mine else set()
    chosen = mine[mine.task == starting.TASK]
    threshold = _number(chosen.value.iloc[-1]) if len(chosen) else None
    starts = {o["candidate_id"]: o for o in field_objects(meta, tile)} if threshold is not None else {}
    reject = (candidate_store.negative_class(words), "")
    records = []
    stamp = {"source": person, "stain": meta["stain"], "tile_id": tile_id, "session": meta["name"]}
    for r in own.rename(columns={"class": "klass"}).itertuples():
        if r.candidate_id in replaced:  # split or merged: its pieces are this person's answer
            continue
        x, y = round(float(r.x), 1), round(float(r.y), 1)
        key = store.object_id("class", tile_id, x, y)
        detected = candidate_store.ai_call(words, r.decision, r.klass, r.subtype, r.reason)
        identity, subtype = detected
        start = starts.get(r.candidate_id)
        if start is not None and start.get("threshold") is not None and abs(threshold - start["threshold"]) > 1e-9:
            identity, subtype = starting.start_call(start.get("ai_probability"), start.get("if_accepted"), threshold, detected, reject)
        if key in said:
            positive = said[key]
        else:  # left as it started (detection's call, at the threshold they set): that is this person's answer
            records.append({**stamp, "id": key, "task": "class", "x": x, "y": y, "value": identity, "confirmed": (identity, subtype) == detected,
                            "subtype": (subtype or None) if asks_subtypes(meta) else None, "candidate": r.candidate_id})
            positive = words.positive(identity, subtype)
        options, shown = candidate_store.choices(r)
        outline = _simplified(options[shown])  # the outline shown first (the stain's default), which the person left as it is
        outline_key = store.object_id("outline", tile_id, x, y)
        if positive and outline and outline_key not in outlined:
            records.append({**stamp, "id": outline_key, "task": "outline", "x": x, "y": y, "points": outline, "candidate": r.candidate_id, "confirmed": True,
                            "value": candidate_store.CHOICES[shown]})
    return records


def save_tile(meta: dict, tile_id: str, marks: list, submit: bool, person: str) -> dict:
    """Save everything this person changed on a tile, all at once (or nothing, when one label is wrong). With `submit`
    ("Save and continue") the field is finished: the calls and outlines left as the AI made them are confirmed. The field
    stays this person's until it is finished. An old round-1 session takes no more labels."""
    meta = writable(meta)
    if rounds.round_of(meta) != 2:
        raise HTTPException(410, NO_ROUND_ONE)
    tile = tile_of(meta, tile_id)
    cleaned = []
    for mark in marks:
        if not mark.candidate and not mark.tile_id:
            mark = mark.model_copy(update={"tile_id": tile_id})
        record = clean(meta, mark, person)
        if record is None:
            continue
        if record["tile_id"] != tile_id:
            raise HTTPException(400, f"{mark.candidate or mark.tile_id} is not on {tile_id}: save each field's changes with that field")
        cleaned.append(record)
        outline = mask_outline(meta, mark, record) if mark.candidate and mark.task == "class" else None
        if outline:
            cleaned.append(outline)
    practice = bool(meta.get("practice"))
    if submit:
        if not practice:
            cleaned += _confirmations(meta, tile, person, _mine_after(meta, tile_id, person, cleaned))
        cleaned.append({"id": rounds.done_id(meta, tile_id), "source": person, "stain": meta["stain"], "task": "done", "tile_id": tile_id,
                        "session": meta["name"], "value": rounds.done_kind(meta) or None})
    if practice:  # nothing of the tutorial is kept
        now = time.time()
        return {"saved": [{"confirmed": False, "deleted": False, "time": now, **r} for r in cleaned], "submitted": submit, "progress": progress(meta)}
    saved = store.append(root(), cleaned) if cleaned else []
    if submit:
        _TILES_HELD.release(tile_hold(meta), person, [tile_id])
    else:
        _TILES_HELD.hold(tile_hold(meta), person, [tile_id])
    if any(r["task"] == "class" and not r.get("candidate") and not r["deleted"] for r in saved):
        measure_added_soon(get_state().project, meta["stain"])
    return {"saved": saved, "submitted": submit, "progress": progress(meta)}


@router.post("/annotate/{name}/tiles/{tile_id}/save")
def save_route(name: str, tile_id: str, body: TileSave, request: Request, person: str = Depends(annotator)) -> dict:
    return save_tile(_round_meta(name, request), tile_id, body.records, body.submit, person)


# ---------- one-click outlines ----------

class Clicks(BaseModel):
    points: list[list[float]]  # [x, y, 1 on the object | 0 not on it], the first on it
    box: list[float] | None = None  # x0, y0, x1, y1: the object lies within (a split piece, padded)
    sizes: bool = False  # Segment Anything's three sizes whatever the clicks (a split piece: small and medium)
    avoid: list[list[list[float]]] | None = None  # outlines cut out of the result (a split piece's siblings)


def _round_meta(name: str, request: Request) -> dict:
    """A round session (round 1 refines the outline of the object on screen, round 2 outlines and refines in fields)."""
    meta = session_meta(name, request)
    if not rounds.is_round(meta):
        raise HTTPException(404, f"{name} is not an annotation round")
    return meta


@router.get("/annotate/{name}/fields/{tile_id}/excluded")
def field_excluded(name: str, tile_id: str, request: Request, person: str = Depends(annotator)) -> dict:
    """The areas detection excludes from the analysis over a field: its part of the core's automatic layer
    (`store_view.field_excluded`), the same pixels the Cores view draws and the analysis uses. People's areas are
    `…/annotated`; the page draws its unsaved strokes over both."""
    from stainid.api import store_view  # noqa: PLC0415

    meta = _round_meta(name, request)
    return store_view.field_excluded(get_state().project, field_tile(meta, tile_id))[0]


@router.get("/annotate/{name}/fields/{tile_id}/excluded.png")
def field_excluded_png(name: str, tile_id: str, request: Request, person: str = Depends(annotator)) -> Response:
    from stainid.api import store_view  # noqa: PLC0415

    meta = _round_meta(name, request)
    png = store_view.field_excluded(get_state().project, field_tile(meta, tile_id))[1]
    if png is None:
        raise HTTPException(404, f"the excluded areas of {tile_id} are not made yet (the v2 proposals make them)")
    return Response(png, media_type="image/png", headers={"Cache-Control": "private, max-age=86400"})


@router.get("/annotate/{name}/fields/{tile_id}/annotated")
def field_annotated(name: str, tile_id: str, request: Request, person: str = Depends(annotator)) -> dict:
    """People's areas over a field (`store_view.people_areas`, field-picture px): painted areas by type and the parts
    marked fine, with how they compare with detection's. They train and evaluate models; the analysis never reads them."""
    from stainid.api import store_view  # noqa: PLC0415

    meta = _round_meta(name, request)
    tile = field_tile(meta, tile_id)
    x0, y0 = render.crop_origin(tile, get_state().project.context_px)
    x1 = int(tile["x_px"]) + int(tile["width_px"]) + get_state().project.context_px
    y1 = int(tile["y_px"]) + int(tile["height_px"]) + get_state().project.context_px
    return store_view.people_areas(get_state().project, str(tile["core_id"]), str(tile["stain"]), (x0, y0), (x0, y0, x1, y1))


@router.get("/annotate/{name}/fields/{tile_id}/areas")
def field_areas(name: str, tile_id: str, request: Request, person: str = Depends(annotator)) -> dict:
    """Each brush type's current area over a field (`store_view.field_areas`: detection's area of the type with
    people's saved edits merged in), what the brush adds to and takes from."""
    from stainid.api import store_view  # noqa: PLC0415

    meta = _round_meta(name, request)
    return store_view.field_areas(get_state().project, field_tile(meta, tile_id))


class Split(BaseModel):
    points: list[list[float]] = []  # the object's outline (field px)
    seed: list[float] | None = None  # a click on the object: the split keeps it apart from the far side
    line: list[list[float]] | None = None  # a drawn line: the object is cut along it
    outlines: list[list[list[float]]] = []  # with `line`: every object to try (the page sends those it crosses)


def _field_picture(tile_id: str) -> np.ndarray | None:
    path = render.field_path(get_state(), tile_id)
    picture = cv2.imread(str(path), cv2.IMREAD_COLOR) if path.exists() else None
    return None if picture is None else cv2.cvtColor(picture, cv2.COLOR_BGR2RGB)


@router.post("/annotate/{name}/fields/{tile_id}/split")
def split_route(name: str, tile_id: str, body: Split, request: Request, person: str = Depends(annotator)) -> dict:
    """Split mode (S): an object split in two, as a preview the page shows until Enter (`labels.objectedit`): a click,
    a watershed of it (`seed`: a second click moves the split); a line, every object it crosses right through cut along
    it (`pieces`: per outline sent, its two pieces or null). Nothing is stored here: the page saves the pieces."""
    meta = _round_meta(name, request)
    field_tile(meta, tile_id)
    if body.line is not None:
        return {"pieces": [objectedit.split_line(o, body.line) if len(o) >= 3 else None for o in body.outlines]}
    if len(body.points) < 3:
        raise HTTPException(400, "an object to split needs its outline")
    pieces = objectedit.split_auto(body.points, _field_picture(tile_id), body.seed)
    if pieces is None:
        raise HTTPException(400, "this object does not fall in two here: click elsewhere on it, or draw a line across it")
    return {"pieces": pieces}


class Merge(BaseModel):
    outlines: list[list[list[float]]]


@router.post("/annotate/{name}/fields/{tile_id}/merge")
def merge_route(name: str, tile_id: str, body: Merge, request: Request, person: str = Depends(annotator)) -> dict:
    """Merge mode (M): one outline for the objects chosen (their union, bridged where they do not touch)."""
    meta = _round_meta(name, request)
    field_tile(meta, tile_id)
    shapes = [o for o in body.outlines if o is not None and len(o) >= 3]
    if len(shapes) < 2:
        raise HTTPException(400, "Choose two or more objects to merge.")
    gap_um = objectedit.merge_gap(shapes) * get_state().project.pixel_size_um
    limit_um = objectedit.MERGE_GAP_PX * get_state().project.pixel_size_um
    if gap_um > limit_um:
        raise HTTPException(400, f"These objects are {gap_um:.0f} µm apart: only objects that touch or nearly touch (within {limit_um:.0f} µm) can be merged.")
    outline = objectedit.merge(shapes)
    if outline is None:
        raise HTTPException(400, "These objects could not be joined into one outline: choose objects that touch or nearly touch.")
    return {"outline": outline}


@router.get("/annotate/{name}/fields/{tile_id}/segment")
def segment_status(name: str, tile_id: str, request: Request, person: str = Depends(annotator)) -> dict:
    """Whether clicks on this field outline at once (`ready`), or Segment Anything is still preparing it."""
    meta = _round_meta(name, request)
    field_tile(meta, tile_id)
    return sam_embeddings.status(get_state(), tile_id)


@router.post("/annotate/{name}/fields/{tile_id}/segment")
def segment(name: str, tile_id: str, body: Clicks, request: Request, person: str = Depends(annotator)) -> dict:
    """Outlines Segment Anything proposes for a clicked object: one click, its three sizes (smallest first, with the one
    it would choose); several (spots to include, 1, and to leave out, 0), the one outline that fits them all. A split
    piece asks for its sizes (`sizes`) with its siblings' centres left out, a `box` around it and the siblings' outlines
    cut out of the result (`avoid`)."""
    meta = _round_meta(name, request)
    field_tile(meta, tile_id)
    if not body.points or any(len(p) != 3 for p in body.points):
        raise HTTPException(400, "send at least one click as [x, y, 1]")
    state = get_state()
    if not state.project.model("sam").exists():
        raise HTTPException(409, "The Segment Anything weights are not downloaded: draw the outline by hand, or ask the person running stainID "
                                 "to download them on the Models page.")
    height, width = sam_embeddings.field_picture(state, tile_id).shape[:2]
    points = [(float(x), float(y), bool(on)) for x, y, on in body.points if 0 <= x < width and 0 <= y < height]
    if not points or (points[0][:2] != tuple(map(float, body.points[0][:2]))):
        raise HTTPException(400, "click inside the picture")
    if body.box is not None and (len(body.box) != 4 or body.box[2] <= body.box[0] or body.box[3] <= body.box[1]):
        raise HTTPException(400, "a box is x0, y0, x1, y1")
    box = None if body.box is None else tuple(float(v) for v in body.box)
    avoid = [[[float(x), float(y)] for x, y in ring] for ring in (body.avoid or []) if all(len(p) == 2 for p in ring)]
    def run(task: work.Task) -> dict:
        try:
            return sam_embeddings.outline(state, tile_id, meta["stain"], points, box=box, sizes=body.sizes, avoid=avoid)
        except IndexError:  # no click left inside the window around the first one
            return {"options": [], "chosen": 0}
        except Exception as error:  # noqa: BLE001 - the weights would not load, the model failed: a plain message
            log.exception("Segment Anything failed on %s", tile_id)
            raise HTTPException(503, SAM_FAILED) from error

    if sam_embeddings.status(state, tile_id)["ready"]:  # the features are made: the decoder answers in a tenth of a second
        return run(None)
    label = "Waiting for Segment Anything" if sam_embeddings.busy() else "Outlining"
    return work.answer("segment", ("segment", secrets.token_hex(8)), run, work.URGENT, label, then="result")


SAM_FAILED = ("One-click outlines are not working on this server right now (Segment Anything could not start). Draw the "
              "outline by hand for now (Draw an object by hand…), and tell the person running stainID: the details are in the server's log.")


# ---------- practice for the tutorial ----------

PRACTICE = "practice"
_TUTORIALS = threading.Lock()


def practice_name(round_: int, stain: str) -> str:
    return f"{PRACTICE}-{round_}-{stain}"


def practice_tile(stain: str) -> str | None:
    """The field the tutorial practises on: the one tutorial of the server (`stainid.tutorial`), the same whatever
    the project; None before it is built or for another stain."""
    from stainid import tutorial  # noqa: PLC0415

    found = tutorial.practice()
    return found["tile_id"] if found and found["stain"] == stain else None


def practice_meta(name: str) -> dict:
    """The made-up session of a practice name (404 when there is nothing to practise on)."""
    parts = name.split("-", 2)
    if len(parts) != 3 or parts[1] != "2":
        raise HTTPException(404, f"no labelling session {name}")
    round_, stain = int(parts[1]), parts[2]
    tile = practice_tile(stain)
    if tile is None:
        raise HTTPException(404, "The tutorial is not set up on this server yet (stainid tutorial build).")
    from stainid import tutorial  # noqa: PLC0415
    return {"name": name, "title": "Practice: " + rounds.TITLE[round_], "stain": stain, "kind": rounds.KIND[round_], "tools": ["class", "point", "outline", "tag"],
            "ai_view": "full", "per_field": 1, "rule": rounds.RULE, "fields": [tile] if round_ == 2 else [], "scope": [tile], "source": rounds.SOURCE,
            "round": round_, "level": None, "labels_needed": 1, "subtypes": True, "instructions": "", "open": True, "created": 0.0, "practice": True,
            "credit": (tutorial.info() or {}).get("credit", "")}


def _tutorials_path():
    return root() / "tutorials.json"


def _tutorials() -> dict:
    try:
        return json.loads(_tutorials_path().read_text())
    except (FileNotFoundError, ValueError):
        return {}


class Tutorial(BaseModel):
    done: bool = True


@router.get("/tutorial")
def tutorial(request: Request, person: str = Depends(annotator)) -> dict:
    """Whether this person has been through the tutorial, and where it practises: the one tutorial of the server
    (`stainid.tutorial`: a public dataset's field, the same for every project, a new one too), with its credit line
    and the hint of the "Change the type" step; `practice` None before `stainid tutorial build`. The "Mark bad areas"
    practice is `practice-areas`. Nothing of the project is read."""
    from stainid import tutorial as shared  # noqa: PLC0415

    return {"done": _done(request, person), "practice": shared.practice()}


def _done(request: Request, person: str) -> bool:
    """Whether the signed-in person went through "Fix the field": kept with their account (`/api/session/tutorials`);
    the record of earlier releases (`tutorials.json` of the project, by name) counts too."""
    from stainid.api import accounts  # noqa: PLC0415
    from stainid.api.session import current_account  # noqa: PLC0415

    account = current_account(request)
    if account is None or person == PREVIEW:
        return False
    return accounts.tutorials_done(account.id)["objects"] or person in _tutorials()


@router.post("/tutorial")
def tutorial_done(body: Tutorial, request: Request, person: str = Depends(annotator)) -> dict:
    """Remember that this person finished (or skipped) "Fix the field", or forget it (to see it again), with their
    account (`/api/session/tutorials`); someone trying it without an account (`PREVIEW`) is not remembered."""
    from stainid.api import accounts  # noqa: PLC0415
    from stainid.api.session import current_account  # noqa: PLC0415

    account = current_account(request)
    if account is None or person == PREVIEW:
        return {"done": body.done}
    accounts.set_tutorial(account.id, "objects", body.done)
    if not body.done and person in _tutorials():  # seen again: the earlier release's record goes too
        with _TUTORIALS:
            done = _tutorials()
            done.pop(person, None)
            write_text_atomic(_tutorials_path(), json.dumps(done, indent=1))
    return {"done": body.done}
