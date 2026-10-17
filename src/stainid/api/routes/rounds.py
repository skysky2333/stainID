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

Practice sessions (`practice-<round>-<stain>`) are made up on the fly for the tutorial a new annotator goes through: one
real field of the store (and its candidates for round 1), shown exactly like a round session, but nothing anyone does
there is saved: answers are echoed back and never reach the label store, decisions of others are not shown, nothing is
held. Who finished the tutorial is kept in `<reviews>/tutorials.json` (`/api/session/tutorial`).
"""
from __future__ import annotations

import json
import re
import threading
import time

import cv2
import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from stainid import cache
from stainid.api import label_crops, queue_pictures, render, sam_embeddings
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
from stainid.labels import added, candidate_store, decisions, readiness, rounds, sessions, store
from stainid.labels import vocabulary as vocab
from stainid.labels.tasks import STAINS
from stainid.tables import write_text_atomic

router = APIRouter(tags=["annotation rounds"])
admin = [Depends(require_admin)]
_TILES_HELD = sessions.Reservations(rounds.TILE_HOLD_SECONDS)  # tiles on someone's screen, per round and stain
OUTLINE_SIMPLIFY_PX = 1.5


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
    """A round-1 session's candidates in its order (the practice's few for the tutorial)."""
    state = get_state()
    return practice_items(meta) if meta.get("practice") else rounds.items(state.project, meta, state.tiles)


def session_tiles(meta: dict) -> list[str]:
    """The tiles of a round session in the order they are handed out: round 1 those of its candidates, round 2 its fields."""
    return rounds.tiles_of(round_items(meta)) if rounds.round_of(meta) == 1 else list(meta["fields"])


def tile_of(meta: dict, tile_id: str) -> dict:
    """One tile of a round session (404 when it is not one of the session's)."""
    if rounds.round_of(meta) == 1 and tile_id not in set(session_tiles(meta)):
        raise HTTPException(404, f"{tile_id} is not part of this session")
    return field_tile(meta, tile_id)


def decided_now(stain: str) -> decisions.State:
    state = get_state()
    return decisions.state(state.project, stain, state.tiles)


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
    if body.round not in rounds.KIND:
        raise HTTPException(400, "round must be 1 or 2")
    if body.level is not None and body.level not in set(rounds.LEVELS.values()):
        raise HTTPException(400, f"level must be one of {[v for v in rounds.LEVELS.values() if v]} (or empty for every candidate)")
    state = get_state()
    name = body.name or free_name(body.title or f"{body.stain} round {body.round}", body.stain, rounds.KIND[body.round])
    if not re.fullmatch(r"[A-Za-z0-9_\-]+", name) or name in sessions.names(root()) or name == sessions.REFERENCE:
        raise HTTPException(409, "a session with this name exists, or the name has other characters than letters, digits, - and _")
    needs = max(1, body.labels_needed)
    scope = _scope(body)
    common = {"title": body.title or name, "stain": body.stain, "instructions": body.instructions, "subtypes": body.subtypes, "ai_view": "full",
              "source": rounds.SOURCE, "round": body.round, "level": body.level, "labels_needed": needs, "per_field": needs}
    if body.round == 1:
        ids = rounds.choose(state.project, body.stain, state.tiles, scope, body.level, max(0, body.n_objects), decided_now(body.stain), needs)
        if not ids:
            raise HTTPException(400, f"Every {body.stain} candidate there has been checked already.")
        meta = sessions.create(root(), name, kind="queue", tools=["class"], rule=rounds.RULE, fields=[], scope=scope, objects=len(ids), **common)
        rounds.save_items(root(), name, ids)
    else:
        fields = scope if body.tile_ids else _pick_fields(body.stain, scope, max(1, body.n_fields), len(sessions.names(root())))
        meta = sessions.create(root(), name, kind="fields", tools=["class", "point", "outline", "tag"], rule="random", fields=fields, scope=fields,
                               **common)
    if body.open:
        meta = sessions.update(root(), name, open=True)
    warm(meta, force=True)
    return meta


def _pick_fields(stain: str, scope: list[str], n: int, seed: int) -> list[str]:
    """n fields of the scope nobody finished and no session has, spread over TMAs and groups."""
    state = get_state()
    tiles = state.labelling_tiles
    tiles = tiles[(tiles.stain == stain) & tiles.tile_id.isin(set(scope))] if not tiles.empty else tiles
    busy_fields = set(store.reviewed_fields(store.current(root()), stain)) | set(sessions.taken(root()))
    free = tiles[~tiles.tile_id.isin(busy_fields)] if not tiles.empty else tiles
    if free.empty:
        raise HTTPException(400, f"Every {stain} field there has been checked or belongs to a session already.")
    return sessions.pick_fields(free, set(), n, seed)


def add_fields(meta: dict, n: int, tile_ids: list[str] | None) -> dict:
    """More fields for a "Fix the field" session: these (checked to have candidates) or n more stainID picks."""
    if rounds.round_of(meta) != 2:
        raise HTTPException(400, "a “Check detection’s calls” session keeps the candidates it was made with; make a new session for more")
    scope = _scope(NewSession(stain=meta["stain"], round=2, tile_ids=tile_ids))
    left = [t for t in scope if t not in set(meta["fields"])]
    picked = left if tile_ids else _pick_fields(meta["stain"], left, max(1, n), len(meta["fields"]) + 1)
    if not picked:
        raise HTTPException(400, f"no more {meta['stain']} fields can be added")
    updated = sessions.add_fields(root(), meta["name"], picked)
    updated = sessions.update(root(), meta["name"], scope=updated["fields"])
    warm(updated, force=True)
    return updated


def warm(meta: dict, force: bool = False) -> None:
    """Make ahead what people will see: round 1 the pictures of its candidates, round 2 the Segment Anything features
    of its fields (when the session is open)."""
    state = get_state()
    if meta.get("practice"):
        return
    if rounds.round_of(meta) == 1:
        queue_pictures.prepare_wanted(state, meta, open_items(meta, round_items(meta)), force=force)
    else:
        sam_embeddings.prepare_session(state, meta, force=force)


def open_items(meta: dict, frame: pd.DataFrame, finished: dict[str, set[str]] | None = None) -> pd.DataFrame:
    """A round-1 session's candidates on tiles that still need people (whose pictures are worth making)."""
    return rounds.open_items(frame, finished_tiles(meta) if finished is None else finished, rounds.needed(meta))


@router.get("/labels/rounds/scope", dependencies=admin)
def scope_overview(stain: str) -> dict:
    """What a new round session can draw from, per TMA: how many cores and fields have the AI's candidates (are ready),
    and the choices of the advanced settings."""
    state = get_state()
    tiles = state.labelling_tiles
    tiles = tiles[tiles.stain == stain] if not tiles.empty else tiles
    ready = set(store_ready(stain, tiles).tile_id) if not tiles.empty else set()
    out = []
    for tma, group in (tiles.groupby(tiles.tma.astype(str)) if not tiles.empty else []):
        ok = group[group.tile_id.isin(ready)]
        out.append({"tma": tma, "cores": int(group.core_id.nunique()), "cores_ready": int(ok.core_id.nunique()), "fields": int(len(group)),
                    "fields_ready": int(len(ok))})
    out.sort(key=lambda t: (len(t["tma"]), t["tma"]))
    return {"stain": stain, "tmas": out, "cores": sum(t["cores"] for t in out), "cores_ready": sum(t["cores_ready"] for t in out),
            "fields_ready": len(ready), "levels": rounds.LEVELS, "default_objects": rounds.DEFAULT_OBJECTS,
            "from_pool": (candidate_store.folder(state.project, stain) / ".from_pool.json").exists()}


# ---------- what the generic session routes ask ----------

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


def my_answers(meta: dict, person: str, ids: list[str]) -> dict[str, dict]:
    """This person's saved answers on these candidates: the class (identity, subtype; `keep` when it confirmed the AI's
    call), the mask chosen with it and an outline they edited (field-picture pixels)."""
    if meta.get("practice") or not ids:
        return {}
    labels = store.current(root())
    if "candidate" not in labels:
        return {}
    mine = labels[(labels.source == person) & (labels.stain == meta["stain"]) & labels.candidate.isin(set(ids))]
    out: dict[str, dict] = {}
    for r in mine[mine.task == "class"].itertuples():
        out[r.candidate] = {"identity": vocab.text(r.value), "subtype": vocab.text(r.subtype), "keep": vocab.text(r.confirmed) == "True", "mask": None, "outline": None}
    for r in mine[mine.task == "outline"].itertuples():
        if r.candidate in out and isinstance(r.points, list):
            value = vocab.text(r.value)
            out[r.candidate] |= {"mask": int(value[5:])} if re.fullmatch(r"mask \d+", value) else {"outline": r.points}
    return out


def _item(row, size: int, pixel_um: float, words: vocab.Vocabulary, mine: dict | None) -> dict:
    identity, subtype = candidate_store.ai_call(words, row.decision, row.klass, row.subtype)
    found, default = candidate_store.masks(row)
    x0, y0 = round(float(row.x)) - size // 2, round(float(row.y)) - size // 2
    dx, dy = float(row.x_core) - float(row.x), float(row.y_core) - float(row.y)  # the field picture's corner on the core image
    in_crop = lambda poly: None if not poly else [[round(px - dx - x0, 1), round(py - dy - y0, 1)] for px, py in poly]  # noqa: E731
    if mine and mine.get("outline"):
        mine = {**mine, "outline": [[round(px - x0, 1), round(py - y0, 1)] for px, py in mine["outline"]]}  # field picture -> crop
    return {"key": row.candidate_id, "candidate_id": row.candidate_id, "tile_id": row.tile_id, "x": round(float(row.x), 1), "y": round(float(row.y), 1),
            "radius_px": round(candidate_store.radius(row, pixel_um), 1), "ai_value": identity, "ai_subtype": subtype,
            "ai_probability": None if pd.isna(row.probability) else round(float(row.probability), 3), "decision": row.decision,
            "reason": row.reason, "tier": int(row.tier) if hasattr(row, "tier") and pd.notna(row.tier) else None,
            "masks": [in_crop(m) for m in found], "mask": default, "crop_px": size, "mine": mine}


def next_tile_objects(meta: dict, person: str, skip: str = "") -> dict:
    """Round 1: the next tile for this person (their tile in progress first; held for them) with all its candidates to
    check, in the session's order, and their saved answers there; how many tiles are left for them, the pictures and
    the session's progress."""
    state = get_state()
    stain = meta["stain"]
    frame = round_items(meta)
    finished = finished_tiles(meta)
    made = queue_pictures.ready(state, stain, frame) if len(frame) else None
    complete = set() if made is None else {t for t, ok in frame.assign(made=made.to_numpy(bool)).groupby("tile_id").made.all().items() if ok}
    tile, taken = hand_out(meta, person, skip, prefer=complete)
    chosen = frame[frame.tile_id == tile] if tile else frame.iloc[:0]
    size, words = label_crops.size_px(state, stain), vocabulary(stain)
    answers = my_answers(meta, person, list(chosen.candidate_id))
    objects = [_item(r, size, state.project.pixel_size_um, words, answers.get(r.candidate_id)) for r in chosen.rename(columns={"class": "klass"}).itertuples()]
    pictures = queue_pictures.prepare_wanted(state, meta, open_items(meta, frame, finished), first=chosen[["tile_id", "x", "y"]] if len(chosen) else None)
    tiles, needs = session_tiles(meta), rounds.needed(meta)
    left = [t for t in tiles if person not in finished.get(t, set()) and len(finished.get(t, set())) < needs]
    return {"tile_id": tile, "objects": objects, "tiles_left": len(left), "taken": taken, "open_by": sorted(tiles_held(meta).get(tile, set()) - {person}) if tile else [],
            "pictures": pictures, "progress": progress(meta, finished)}


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


def open_field(meta: dict, tile_id: str, person: str) -> dict:
    """Round 2: a field with every candidate of the session's level (the field's own and, faded, its neighbours' in the
    margin), the AI's call and outline of each, who decided it already, this person's own saved labels there, and who
    else is on it or finished it. Opening a field holds it for this person."""
    state = get_state()
    tile = field_tile(meta, tile_id)
    practice = bool(meta.get("practice"))
    if not practice:
        _TILES_HELD.hold(tile_hold(meta), person, [tile_id])
        busy(meta["name"], tile_id, person)
    found = _field_candidates(meta, tile)
    words = vocabulary(meta["stain"])
    decided = decisions.State() if practice else decided_now(meta["stain"])
    x0, y0 = render.crop_origin(tile, state.project.context_px)
    objects = []
    for r in found.rename(columns={"class": "klass"}).itertuples():
        identity, subtype = candidate_store.ai_call(words, r.decision, r.klass, r.subtype)
        found_masks, default = candidate_store.masks(r)
        objects.append({"candidate_id": r.candidate_id, "x": round(float(r.x), 1), "y": round(float(r.y), 1), "own": bool(r.own),
                        "radius_px": round(candidate_store.radius(r, state.project.pixel_size_um), 1), "ai_value": identity, "ai_subtype": subtype,
                        "ai_probability": None if pd.isna(r.probability) else round(float(r.probability), 3), "decision": r.decision, "reason": r.reason,
                        "masks": [_simplified(m) for m in found_masks], "mask": default})
    labels = store.current(root())
    mine = labels[(labels.source == person) & (labels.tile_id == tile_id) & (labels.stain == meta["stain"])]
    if practice:
        mine = mine.iloc[:0]
    box = {"x": int(tile["x_px"]) - x0, "y": int(tile["y_px"]) - y0, "width": int(tile["width_px"]), "height": int(tile["height_px"])}
    finished = finished_tiles(meta, labels)
    sam = sam_embeddings.prepare_field(state, meta, tile_id, {t for t, people in finished.items() if person in people})
    return {"tile_id": tile_id, "box": box, "objects": objects, "checked": checks(found, decided), "layers": LAYERS.get(meta["stain"], []),
            "mine": mine.replace({np.nan: None}).to_dict(orient="records"), "added": added_states(meta, tile, mine, found),
            "segment_anything": sam["state"] != "unavailable", "segment_ready": bool(sam.get("ready")),
            "open_by": sorted(tiles_held(meta).get(tile_id, set()) - {person}), "done_by": sorted(finished.get(tile_id, set()) - {person}),
            "pixel_size_um": state.project.pixel_size_um}


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
            out["checked"] = checks(_field_candidates(meta, tile), decided_now(meta["stain"]))
    return out | {"progress": progress(meta)}


@router.post("/annotate/{name}/live")
def live_route(name: str, body: Live, request: Request, person: str = Depends(annotator)) -> dict:
    meta = session_meta(name, request)
    if not rounds.is_round(meta):
        raise HTTPException(404, f"{name} is not an annotation round")
    return live(meta, body, person)


# ---------- saving ----------

def candidate_row(stain: str, candidate: str) -> pd.Series:
    """One candidate of the store, with `x`, `y` in its own field's picture (404 when the store has no such candidate)."""
    state = get_state()
    parts = candidate.split(":")
    if len(parts) < 4 or parts[0] != stain:
        raise HTTPException(400, f"{candidate!r} is not a {stain} candidate")
    found = candidate_store.read_core(state.project, stain, ":".join(parts[1:-2]))
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


def clean(meta: dict, mark, person: str) -> dict:
    """A label of a round session, checked: about a candidate (`candidate`: its field and position come from the store;
    `keep` records the AI's call as confirmed; an outline of it, e.g. a precise edit) or, in round 2, about the field or an
    object people added."""
    stain = meta["stain"]
    second = rounds.round_of(meta) == 2
    allowed = {"class", "outline"} | ({"tag", "note", "done"} if second else set())
    if mark.task not in allowed:
        raise HTTPException(400, f"{mark.task} labels are not part of this session")
    record = {"source": person, "stain": stain, "task": mark.task, "session": meta["name"], "deleted": mark.deleted, "value": mark.value,
              "subtype": None, "points": mark.points, "tile_id": mark.tile_id, "x": mark.x, "y": mark.y, "candidate": mark.candidate or None}
    words = vocabulary(stain)
    if mark.candidate:
        row = candidate_row(stain, mark.candidate)
        record |= {"tile_id": str(row.tile_id), "x": round(float(row.x), 1), "y": round(float(row.y), 1)}
        if second:
            field_tile(meta, record["tile_id"])
        if mark.task == "class" and mark.keep and not mark.deleted:
            identity, subtype = candidate_store.ai_call(words, row.decision, row["class"], row.subtype)
            record |= {"value": identity, "subtype": (subtype or None) if asks_subtypes(meta) else None, "confirmed": True}
            return record | {"id": store.object_id("class", record["tile_id"], record["x"], record["y"])}
    elif not second or mark.task in ("class", "outline"):
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
           "tag": lambda: f"tag:{mark.tile_id}:{mark.value}", "note": lambda: f"note:{mark.tile_id}", "done": lambda: f"done:{mark.tile_id}"}
    return record | {"id": ids[mark.task]()}


def mask_outline(meta: dict, mark, record: dict) -> dict | None:
    """The outline label of the mask someone chose with a decision (`mark.mask`, small to large): None when no mask was
    chosen or the call says there is no object there ("not an object", "unsure")."""
    outline_id = store.object_id("outline", record["tile_id"], record["x"], record["y"])
    if mark.deleted:  # a decision made with a mask taken back (undo): the mask goes with it
        return {**record, "id": outline_id, "task": "outline", "value": "", "subtype": None, "points": None} if mark.mask is not None else None
    words = vocabulary(meta["stain"])
    if mark.mask is None or record["value"] == candidate_store.negative_class(words) or words.skipped(record["value"]):
        return None
    row = candidate_row(meta["stain"], mark.candidate)
    found, default = candidate_store.masks(row)
    if not 0 <= mark.mask < len(found) or not found[mark.mask]:
        raise HTTPException(400, f"candidate {mark.candidate} has no mask {mark.mask}")
    dx, dy = float(row.x_core) - float(row.x), float(row.y_core) - float(row.y)
    points = _simplified([[px - dx, py - dy] for px, py in found[mark.mask]])
    return {**record, "id": outline_id, "task": "outline", "value": f"mask {mark.mask}",
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


def _unanswered(meta: dict, tile_id: str, mine: pd.DataFrame) -> list[str]:
    """Round 1: the session's candidates on this tile this person has no answer on."""
    frame = round_items(meta)
    answered = set(mine.candidate[mine.task == "class"].dropna()) if "candidate" in mine else set()
    return [c for c in frame.candidate_id[frame.tile_id == tile_id] if c not in answered]


def _confirmations(meta: dict, tile: dict, person: str, mine: pd.DataFrame) -> list[dict]:
    """Round 2, finishing a field: the calls of its candidates nobody has decided and this person left untouched count as
    confirmed by them, and so do the AI's outlines of the objects that stay positive."""
    tile_id = str(tile["tile_id"])
    found = _field_candidates(meta, tile)
    own = found[found.own]
    words = vocabulary(meta["stain"])
    decided = decided_now(meta["stain"])
    classes = mine[mine.task == "class"]
    said = {k: words.positive(v, s) for k, v, s in zip(classes.id, classes.value, classes.subtype)}
    outlined = set(mine.id[mine.task == "outline"])
    records = []
    stamp = {"source": person, "stain": meta["stain"], "tile_id": tile_id, "session": meta["name"], "confirmed": True}
    for r in own.rename(columns={"class": "klass"}).itertuples():
        x, y = round(float(r.x), 1), round(float(r.y), 1)
        key = store.object_id("class", tile_id, x, y)
        identity, subtype = candidate_store.ai_call(words, r.decision, r.klass, r.subtype)
        if key in said:
            positive = said[key]
        elif decided.count(r.candidate_id):
            positive = words.positive(*decided.calls[r.candidate_id])
        else:
            records.append({**stamp, "id": key, "task": "class", "x": x, "y": y, "value": identity,
                            "subtype": (subtype or None) if asks_subtypes(meta) else None, "candidate": r.candidate_id})
            positive = words.positive(identity, subtype)
        outline = _simplified(candidate_store.ai_outline(r))  # the default mask
        outline_key = store.object_id("outline", tile_id, x, y)
        if positive and outline and outline_key not in outlined:
            records.append({**stamp, "id": outline_key, "task": "outline", "x": x, "y": y, "points": outline, "candidate": r.candidate_id})
    return records


def save_tile(meta: dict, tile_id: str, marks: list, submit: bool, person: str) -> dict:
    """Save everything this person changed on a tile, all at once (or nothing, when one label is wrong). With `submit`
    ("Save and continue") the tile is finished for this round: round 1 needs an answer on each of its candidates; round 2
    confirms the calls and outlines left as the AI made them. The tile stays this person's until it is finished."""
    meta = writable(meta)
    tile = tile_of(meta, tile_id)
    cleaned = []
    for mark in marks:
        if not mark.candidate and not mark.tile_id:
            mark = mark.model_copy(update={"tile_id": tile_id})
        record = clean(meta, mark, person)
        if record["tile_id"] != tile_id:
            raise HTTPException(400, f"{mark.candidate or mark.tile_id} is not on {tile_id}: save each field's changes with that field")
        cleaned.append(record)
        outline = mask_outline(meta, mark, record) if mark.candidate and mark.task == "class" else None
        if outline:
            cleaned.append(outline)
    practice = bool(meta.get("practice"))
    if submit:
        if rounds.round_of(meta) == 1 and not practice:
            missing = _unanswered(meta, tile_id, _mine_after(meta, tile_id, person, cleaned))
            if missing:
                raise HTTPException(400, f"{len(missing)} object{'s' if len(missing) > 1 else ''} on this field still need{'' if len(missing) > 1 else 's'} "
                                         "an answer before it can be finished")
        if rounds.round_of(meta) == 2 and not practice:
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


def _round_meta(name: str, request: Request) -> dict:
    """A round session (round 1 refines the outline of the object on screen, round 2 outlines and refines in fields)."""
    meta = session_meta(name, request)
    if not rounds.is_round(meta):
        raise HTTPException(404, f"{name} is not an annotation round")
    return meta


@router.get("/annotate/{name}/fields/{tile_id}/segment")
def segment_status(name: str, tile_id: str, request: Request, person: str = Depends(annotator)) -> dict:
    """Whether clicks on this field outline at once (`ready`), or Segment Anything is still preparing it."""
    meta = _round_meta(name, request)
    field_tile(meta, tile_id)
    return sam_embeddings.status(get_state(), tile_id)


@router.post("/annotate/{name}/fields/{tile_id}/segment")
def segment(name: str, tile_id: str, body: Clicks, request: Request, person: str = Depends(annotator)) -> dict:
    """Outlines Segment Anything proposes for a clicked object: one click, its three sizes (smallest first, with the one
    it would choose); several (spots to include, 1, and to leave out, 0), the one outline that fits them all."""
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
    try:
        return sam_embeddings.outline(state, tile_id, meta["stain"], points)
    except IndexError:  # no click left inside the window around the first one
        return {"options": [], "chosen": 0}


# ---------- practice for the tutorial ----------

PRACTICE = "practice"
PRACTICE_OBJECTS = 8
_TUTORIALS = threading.Lock()


def practice_name(round_: int, stain: str) -> str:
    return f"{PRACTICE}-{round_}-{stain}"


def practice_tile(stain: str) -> str | None:
    """The field the tutorial practises on: of the first cores with candidates, the ready field with most of them (kept
    until the store changes)."""
    state = get_state()

    def make() -> str | None:
        tiles = state.labelling_tiles
        if tiles.empty:
            return None
        ready = store_ready(stain, tiles[tiles.stain == stain])
        best, most = None, 0
        for core_id in sorted(set(ready.core_id.astype(str)))[:3]:
            found = candidate_store.read_core(state.project, stain, core_id)
            if found is None or found.empty:
                continue
            counts = found.tile_id.value_counts()
            for tile_id in ready.tile_id[ready.core_id.astype(str) == core_id]:
                if counts.get(tile_id, 0) > most:
                    best, most = tile_id, int(counts[tile_id])
        return best

    return cache.memo(("practice field", str(state.root), stain), [candidate_store.folder(state.project, stain)], make, folders=True)


def practice_meta(name: str) -> dict:
    """The made-up session of a practice name (404 when there is nothing to practise on)."""
    parts = name.split("-", 2)
    if len(parts) != 3 or parts[1] not in ("1", "2"):
        raise HTTPException(404, f"no labelling session {name}")
    round_, stain = int(parts[1]), parts[2]
    tile = practice_tile(stain)
    if tile is None:
        raise HTTPException(404, f"there is no {stain} field with detection's candidates to practise on yet")
    return {"name": name, "title": "Practice: " + rounds.TITLE[round_], "stain": stain, "kind": rounds.KIND[round_], "tools": ["class", "point", "outline", "tag"],
            "ai_view": "full", "per_field": 1, "rule": rounds.RULE, "fields": [tile] if round_ == 2 else [], "scope": [tile], "source": rounds.SOURCE,
            "round": round_, "level": None, "labels_needed": 1, "subtypes": True, "instructions": "", "open": True, "created": 0.0, "practice": True}


def practice_items(meta: dict) -> pd.DataFrame:
    """The candidates of the practice field for round 1: the most informative few."""
    state = get_state()
    found = rounds.candidates_in(state.project, meta["stain"], state.tiles, meta["scope"], None)
    ordered = rounds.rank(found).head(PRACTICE_OBJECTS) if len(found) else found
    return ordered.assign(order=range(len(ordered))).reset_index(drop=True)


def _tutorials_path():
    return root() / "tutorials.json"


def _tutorials() -> dict:
    try:
        return json.loads(_tutorials_path().read_text())
    except (FileNotFoundError, ValueError):
        return {}


class Tutorial(BaseModel):
    done: bool = True


@router.get("/session/tutorial")
def tutorial(request: Request, person: str = Depends(annotator)) -> dict:
    """Whether this person has been through the tutorial, and where it practises: the stain of the first open round
    session (else the first with candidates)."""
    state = get_state()
    metas = [sessions.meta(root(), n) for n in sessions.names(root())]
    stains = [m["stain"] for m in metas if rounds.is_round(m) and m.get("open")]
    stains += [s for s in STAINS if candidate_store.cores(state.project, s)]
    practice = None
    for stain in dict.fromkeys(stains):
        tile = practice_tile(stain)
        if tile:
            practice = {"stain": stain, "tile_id": tile, "round1": practice_name(1, stain), "round2": practice_name(2, stain)}
            break
    return {"done": person in _tutorials() and person != PREVIEW, "practice": practice}


@router.post("/session/tutorial")
def tutorial_done(body: Tutorial, person: str = Depends(annotator)) -> dict:
    """Remember that this person finished (or skipped) the tutorial, or forget it (to see it again)."""
    with _TUTORIALS:
        done = _tutorials()
        if body.done:
            done[person] = time.time()
        else:
            done.pop(person, None)
        _tutorials_path().parent.mkdir(parents=True, exist_ok=True)
        write_text_atomic(_tutorials_path(), json.dumps(done, indent=1))
    return {"done": body.done}
