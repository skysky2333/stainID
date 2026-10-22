"""Labels: the admin's overview and session management (/api/labels) and the public annotation pages (/api/annotate)."""
from __future__ import annotations

import contextlib
import os
import re
import socket
import threading
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

from stainid import cache, practice
from stainid.api import label_crops, pyramid, queue_pictures, render, sam_embeddings, work
from stainid.api.routes.images import OVERLAYS
from stainid.api.session import GUEST, annotator, annotator_name, is_admin, require_admin
from stainid.api.state import get_state
from stainid.api.util import fast_json
from stainid.labels import added, frames, outline, overview, pool, readiness, rounds, sessions, store
from stainid.labels import objects as label_objects
from stainid.labels import vocabulary as vocab
from stainid.labels.overview import SPLIT
from stainid.labels.tasks import CROP_UM, FIELD_TAGS, NEGATIVE, STAINS

router = APIRouter(tags=["labels"])
BUSY_SECONDS = 15 * 60
LAYERS = {"NeuN": ["dab", "exclusions"], "6E10": ["dab", "exclusions"], "AT8": ["dab", "exclusions", "threads"]}
_BUSY: dict[tuple[str, str], dict[str, float]] = {}
_BUSY_LOCK = threading.Lock()
_HELD = sessions.Reservations()  # objects of queue sessions on someone's screen, kept from the others for a while
admin = [Depends(require_admin)]
FIELD_FACTS = ("core_id", "core_label", "tma", "region", "disease_group")
BOX = ("x_px", "y_px", "width_px", "height_px")


def root() -> Path:
    return get_state().project.output("reviews")


def vocabulary(stain: str) -> vocab.Vocabulary:
    """The project's label classes of this stain."""
    return vocab.of(get_state().project, stain)


def busy(name: str, tile_id: str | None = None, person: str | None = None) -> dict[str, set[str]]:
    """Who has which field of this project's session open (seen in the last 15 minutes)."""
    now = time.time()
    session = (get_state().id, name)
    if tile_id and person:
        practice.guard("who is on which field")
    with _BUSY_LOCK:
        if tile_id and person:
            _BUSY.setdefault((session, tile_id), {})[person] = now
        return {tile: {p for p, t in people.items() if now - t < BUSY_SECONDS} for (s, tile), people in _BUSY.items() if s == session}


def viewer(request: Request) -> str | None:
    """A signed-in person; on the tutorial's practice sessions anyone (`GUEST`)."""
    name = annotator_name(request)
    if name is None and not is_admin(request):
        if practice.practising():
            return GUEST
        raise HTTPException(401, "sign in first")
    return name


def _areas():
    """The routes of "Mark bad areas" (imported late: they build on this module)."""
    from stainid.api.routes import areas as area_routes

    return area_routes


def session_meta(name: str, request: Request) -> dict:
    if name == "practice-areas":  # the tutorial's practice of "Mark bad areas" (`routes.areas`): nothing is saved
        return _areas().practice_meta()
    if name.startswith("practice-"):  # the tutorial's practice (`routes.rounds`): nothing is saved
        return _rounds().practice_meta(name)
    if not re.fullmatch(r"[A-Za-z0-9_\-]+", name) or name not in sessions.names(root()):
        raise HTTPException(404, f"no labelling session {name}")
    meta = sessions.meta(root(), name)
    if not meta["open"] and not is_admin(request):
        raise HTTPException(403, "this labelling session is not open at the moment")
    from stainid.api.routes import accounts as access  # noqa: PLC0415 - who may label it: admins, and the people who joined it

    access.may_label(request, name)
    return meta


def labelled_here(meta: dict) -> dict:
    """A session people label in: not a read-only label set (an AI review round, `sessions.label_set_only`)."""
    if sessions.label_set_only(meta):
        raise HTTPException(409, "This is a read-only set of AI reviewer labels: look at them on the Labels page (nobody labels in it)")
    return meta


def ai_review_summary(meta: dict, labels: pd.DataFrame) -> dict:
    """The Labels page's card of an AI review round (`review.ai_ingest`): the objects labelled and on how many fields,
    per stain; per reviewer the objects they labelled; and how often the reviewers agreed on the objects both saw (from
    the round's `ingest_report.json`, when there). No field progress: nobody labels in it."""
    import json  # noqa: PLC0415

    from stainid.review import ai_batch  # noqa: PLC0415

    mine = labels[(labels.session == meta["name"]) & labels.task.isin(("class", "object"))] if len(labels) else labels
    if len(mine):
        where = mine.candidate.where(mine.candidate.notna() & (mine.candidate.astype(str) != ""),
                                     mine.x.round().astype(str) + ":" + mine.y.round().astype(str)) if "candidate" in mine else mine.x.astype(str)
        objects = mine.assign(_object=mine.tile_id.astype(str) + "|" + where.astype(str)).drop_duplicates(["stain", "_object"])
        by_stain = {str(s): int(n) for s, n in objects.stain.value_counts().items()}
        reviewers = [{"person": str(p), "objects": int(n)} for p, n in mine.groupby("source")["tile_id"].count().sort_index().items()]
        fields = int(mine.tile_id.nunique())
    else:
        objects, by_stain, reviewers, fields = mine, {}, [], 0
    agreement = None
    round_ = (meta.get("ai_review") or {}).get("round")
    if round_ is not None:
        try:
            report = json.loads((ai_batch.stain_folder(get_state().project, meta["stain"], int(round_)) / "ingest_report.json").read_text())
            if report.get("overlap_objects"):
                agreement = {"objects": int(report["overlap_objects"]), "agreed": int(report.get("overlap_agreed") or 0)}
        except (OSError, ValueError, TypeError):
            pass
    return {"ai_summary": {"objects": int(len(objects)), "fields": fields, "stains": by_stain, "reviewers": reviewers, "agreement": agreement,
                           "round": round_}}


def writable(meta: dict) -> dict:
    """A session people may add labels to: not one of the read-only sessions imported from a dataset package."""
    if meta.get("read_only"):
        raise HTTPException(403, "These labels were imported from a dataset and stay as they were; to label these fields yourself, start a new "
                                 "labelling session on the Labels page")
    return meta


def field_tile(meta: dict, tile_id: str) -> dict:
    if meta["kind"] != "queue" and tile_id not in meta["fields"]:
        raise HTTPException(404, f"{tile_id} is not part of this session")
    return get_state().tile(tile_id)


def people_only(labels: pd.DataFrame) -> pd.DataFrame:
    """People's labels: the AI's (`store.is_ai_label`) make no field in progress or done, nor anyone working on it."""
    return labels[~labels.source.map(store.is_ai_label)] if len(labels) else labels


def session_labels(meta: dict) -> pd.DataFrame:
    """The labels of this session only (its own logs, `store.current(root, session)`): inside a session nothing another
    session or label set said counts, for who finished or is working on a field, the progress, the next field, what a
    field starts from. (The tutorial's practice sessions have none: they save nothing.)"""
    return store.current(root(), session=meta["name"])


def finished_fields(meta: dict, labels: pd.DataFrame) -> dict[str, set[str]]:
    """Who finished each field of a session: marked it complete (outline sessions) or pressed Finished (reviews)."""
    return outline.finished(labels, meta["name"]) if meta["kind"] == outline.KIND else store.reviewed_fields(people_only(labels), meta["stain"])


def ready_fields(meta: dict) -> set[str]:
    """The fields people can open: outline sessions need only the picture, reviews their candidates found (round sessions
    in the candidate store, older ones in the pool)."""
    if rounds.is_round(meta):
        return _rounds().ready_fields(meta)
    return set(meta["fields"]) if meta["kind"] == outline.KIND else pool.prepared(get_state().project, meta["stain"])


def _rounds():
    """The routes of the annotation rounds (imported late: they build on this module)."""
    from stainid.api.routes import rounds as round_routes

    return round_routes


def field_objects(meta: dict, tile_id: str) -> pd.DataFrame:
    objects = pool.load_field(get_state().project, meta["stain"], tile_id)
    if objects is None or not objects.complete.astype(str).eq("True").all():
        raise HTTPException(409, "this field is still being prepared; try again in a minute")
    return objects


def labellers(meta: dict, labels: pd.DataFrame) -> list[dict]:
    """Everyone who has labelled in this session: how far they are and when they last saved something."""
    mine = labels[labels.session == meta["name"]]
    reviewed = finished_fields(meta, labels)
    out = []
    for person, group in mine.groupby("source"):
        classes = group.task.isin(("class", "object"))
        confirmed = group.confirmed.astype(bool)
        last = pd.to_numeric(group.time, errors="coerce").max()
        out.append({"person": person, "fields_done": sum(person in reviewed.get(t, set()) for t in meta.get("fields", [])),
                    "labels": int(classes.sum()), "changed": int((classes & ~confirmed).sum()), "confirmed": int((classes & confirmed).sum()),
                    "outlines": int((group.task == "outline").sum()), "tags": int((group.task == "tag").sum()),
                    "last_active": float(last) if pd.notna(last) else 0.0})
    return sorted(out, key=lambda p: -p["last_active"])


def session_progress(meta: dict, labels: pd.DataFrame) -> dict:
    mine = labels[labels.session == meta["name"]]
    people = labellers(meta, labels)
    now = {p for open_by in busy(meta["name"]).values() for p in open_by}
    progress = {"people": len(people), "labels": int(mine.task.isin(("class", "object")).sum()), "labellers": people,
                "labelling_now": sum(time.time() - p["last_active"] < BUSY_SECONDS or p["person"] in now for p in people)}
    if meta["kind"] == "areas":
        return progress | {"round_progress": _areas().progress(meta, labels), "cores": len(meta.get("cores") or [])}
    if sessions.label_set_only(meta):  # an AI review round: its labels, no fields to prepare or finish
        return progress | ai_review_summary(meta, labels)
    if rounds.is_round(meta):
        progress |= {"round_progress": _rounds().progress(meta)}
    if rounds.is_round(meta) and meta["kind"] == "queue":
        pass  # an old "Check detection's calls" session: its progress in tiles only
    elif meta["kind"] != "queue":
        reviewed = finished_fields(meta, labels)
        covered = sum(len(reviewed.get(t, set())) >= meta["per_field"] for t in meta["fields"])
        prepared = len(ready_fields(meta) & set(meta["fields"]))
        progress |= {"fields": len(meta["fields"]), "fields_covered": covered, "fields_prepared": prepared}
    else:
        progress |= {"pictures": queue_pictures.prepare(get_state(), meta, pool.all_objects(get_state().project, meta["stain"]), labels)}
    return progress


def prepare_job(stain: str, fields: int = 0, session: str = "") -> None:
    get_state().jobs.submit("prepare_labels", {"stain": stain, "fields": fields, "session": session}, owner=session or stain)  # turns per session


_MEASURING: dict[tuple[str, str], bool] = {}  # (project, stain) being measured -> another round was asked for meanwhile
_MEASURING_LOCK = threading.Lock()


def measure_added_soon(project, stain: str) -> threading.Thread | None:
    """Measure the objects people added where the AI proposed nothing (`labels.added`), in the background and one round
    at a time per stain, so they can train the model. Returns the thread it started (None when a round is running: that
    round goes again when it ends)."""
    key = (str(project.root), stain)
    with _MEASURING_LOCK:
        if key in _MEASURING:
            _MEASURING[key] = True
            return None
        _MEASURING[key] = False

    def run() -> None:
        while True:
            try:
                added.measure(project, stain)
            except Exception as error:  # noqa: BLE001 - a failed round is retried with the next added object
                print(f"measuring the {stain} objects people added failed: {error}", flush=True)
            with _MEASURING_LOCK:
                if not _MEASURING[key]:
                    del _MEASURING[key]
                    return
                _MEASURING[key] = False

    thread = threading.Thread(target=run, daemon=True, name=f"measure-added-{stain}")
    thread.start()
    return thread


def added_states(project, stain: str, tile_id: str, labels: pd.DataFrame, objects: pd.DataFrame) -> list[dict]:
    """For each object these labels name: `proposed` (the AI proposed it: a candidate is there, perhaps hidden from the
    person), or, for an object added where the AI proposed nothing, `measured` (it can train the model),
    `could_not_be_measured` (with the reason) or `measuring` (still to do)."""
    extra = added.load_field(project, stain, tile_id)
    out = []
    for row in labels[(labels.task == "class") & labels.x.notna()].itertuples():
        x, y = float(row.x), float(row.y)
        hit = None if pool.nearest(objects, x, y) is not None else added.resolve(extra, x, y)
        if pool.nearest(objects, x, y) is not None or (hit is not None and pd.notna(hit.get("candidate_x"))):
            state = "proposed"
        else:
            state = "measuring" if hit is None else "measured" if added.is_true(hit.measured) else "could_not_be_measured"
        reason = "" if hit is None or pd.isna(hit.get("reason")) else str(hit.reason)
        out.append({"x": x, "y": y, "state": state, "reason": reason if state == "could_not_be_measured" else ""})
    return out


# ---------- admin ----------

class NewSession(BaseModel):
    name: str = ""
    title: str = ""
    stain: str
    kind: str = "fields"
    tools: list[str] = ["class", "point", "outline", "tag"]
    ai_view: str = "full"
    per_field: int = 1
    rule: str = "uncertain"
    n_fields: int | None = None  # how many fields: round sessions -1 (every remaining field) by default, the older kinds 20
    tile_ids: list[str] | None = None
    seed: int | None = None  # round sessions: the seed of the random draw of fields (a new one when empty; kept with the session)
    spread: bool = False  # round sessions: spread the drawn fields over TMAs and groups instead of a plain random draw
    instructions: str = ""
    open: bool = False
    subtypes: bool = True  # people are asked for the type (subtype) of the objects, where a class has types
    tmas: list[str] | None = None  # outline and round sessions: the TMAs fields are drawn from (outline default: `outline.default_tmas`)
    round: int | None = None  # an annotation round over the v2 candidates (`labels.rounds`): 1 check the AI's calls, 2 fix the field
    n_objects: int = 500  # round 1: how many candidates (0: every one)
    level: float | None = None  # rounds: how generous (lowest proposal level shown; None: every candidate)
    labels_needed: int = 1  # rounds: people who must decide a candidate (or finish a field) before it is done for everyone
    order: str = "problems"  # "Mark bad areas" (kind "areas"): "problems" (likely problems first, then at random) or "random"


class Opening(BaseModel):
    open: bool


class MoreFields(BaseModel):
    n: int = 10  # -1: every remaining field (round sessions)
    tile_ids: list[str] | None = None
    seed: int | None = None


class FieldList(BaseModel):
    tile_ids: list[str]


class Prepare(BaseModel):
    stain: str
    fields: int = 20


@router.get("/labels/defaults", dependencies=admin)
def defaults() -> dict:
    stains = get_state().project.stains
    return {"classes": {s: [c.name for c in vocabulary(s).visible] for s in stains}, "vocabularies": {s: vocabulary(s).to_api() for s in stains},
            "tags": FIELD_TAGS, "tools": list(sessions.FIELD_TOOLS), "ai_views": list(sessions.AI_VIEWS), "rules": sessions.RULES}


@router.get("/labels/overview", dependencies=admin)
def labels_overview():
    """Every stain's labels and every session, counted in the background when that takes a while (`work`)."""
    return work.answer("labels overview", ("labels overview", str(get_state().root)), lambda task: _labels_overview(), work.NOW,
                       "Counting the labels", admin=True, fresh=work.within(15))


def _labels_overview() -> dict:
    state = get_state()
    labels = store.current(root())
    stains = [overview.cached_stain_overview(state.project, s, root(), state.tiles) for s in STAINS if s in set(state.tiles.stain)]
    listed = []
    for name in sessions.names(root()):
        meta = sessions.meta(root(), name)
        listed.append({**{k: v for k, v in meta.items() if k != "fields"}, **session_progress(meta, session_labels(meta)), "link": annotate_path(name),
                       "kind_title": sessions.kind_title(meta), "label_set": sessions.label_set_only(meta)})
    reference = labels[sessions.session_of(labels) == sessions.REFERENCE]
    objects = reference[reference.task == "class"]
    summary = {"name": sessions.REFERENCE, "title": sessions.REFERENCE_TITLE, "labels": int(len(objects)),
               "stains": {str(k): int(v) for k, v in objects.stain.value_counts().items()}, "sets": sorted(set(reference.source)),
               "fields": int(objects.tile_id.nunique())} if len(reference) else None
    return {"stains": stains, "sessions": listed, "reference": summary}


def _titles() -> dict[str, str]:
    return {name: sessions.meta(root(), name).get("title", name) for name in sessions.names(root())}


@router.get("/labels/sets", dependencies=admin)
def label_sets(stain: str) -> list[dict]:
    """The sessions (and the reference labels) with object labels of this stain, to choose what a model learns from."""
    return sessions.label_sets(store.current(root()), stain, _titles(), vocabulary(stain))


OBJECTS_PAGE = 60


def _session_objects(name: str, labels: pd.DataFrame, stain: str) -> pd.DataFrame:
    """The labelled objects of one session and stain (`labels.objects.labelled`), kept until a label or pool changes."""
    state = get_state()
    mine = labels[sessions.session_of(labels) == name]
    return cache.memo(("session objects", str(state.root), name, stain, vocabulary(stain), len(mine)), [store.log_path(root()), pool.folder(state.project, stain)],
                      lambda: label_objects.labelled(state.project, state.tiles, mine, stain), folders=True)


def class_groups(vocabulary: vocab.Vocabulary, shown: set[str]) -> list[dict]:
    """The classes the labels view shows, grouped by identity in the order of their keys (`Vocabulary.by_key`): each
    identity's name, its colour slot (`Vocabulary.colour_slots`, so a class keeps its colour whatever is shown; -2 the
    reject, dotted grey, -3 "unsure", dotted amber: webapp classPalette.ts) and the classes under it
    (`Vocabulary.display` names: the identity itself, typed or not, and each of its subtypes). Classes the vocabulary
    no longer has, and objects people split on, come last."""
    groups, known = [], set()
    slot = {**vocabulary.colour_slots(), vocabulary.reject: -2, vocabulary.unsure: -3}  # colours never move
    for c in vocabulary.by_key():  # listed in the order of their keys, as on every page
        colour = slot.get(c.id, -1)
        keys = [*(vocabulary.display(c.id, s.id) for s in c.subtypes if not s.hidden or vocabulary.display(c.id, s.id) in shown), vocabulary.display(c.id)]
        keys += [vocabulary.disputed(c.id)] if c.subtypes and vocabulary.disputed(c.id) in shown else []
        known |= set(keys)
        if not c.hidden or shown & set(keys):
            groups.append({"id": c.id, "name": c.name, "role": c.role, "colour": colour, "keys": keys})
    rest = sorted(shown - known - {SPLIT})
    for key in rest:
        groups.append({"id": key, "name": key, "role": vocab.NEGATIVE, "colour": -1, "keys": [key]})
    return groups + [{"id": SPLIT, "name": SPLIT, "role": vocab.SKIP, "colour": None, "keys": [SPLIT]}]


def ai_verdict(vocabulary: vocab.Vocabulary, identity: str, subtype: str, ai: str | None) -> str | None:
    """How the AI's call compares with the label (its identity and subtype): "agrees", "type" (both positive, of
    another identity or type), "disagrees", or None (no AI call there, or the label is "unsure" or split)."""
    if not vocab.text(ai) or not identity or vocabulary.role(identity) == vocab.SKIP:  # no AI call (None, or NaN in pandas 3 text columns)
        return None
    ai_identity, ai_subtype = vocabulary.resolve(ai)
    mine = vocabulary.role(identity) == vocab.POSITIVE
    if mine != (vocabulary.role(ai_identity) == vocab.POSITIVE):
        return "disagrees"
    if not mine:
        return "agrees"
    return "type" if identity != ai_identity or (subtype and ai_subtype and subtype != ai_subtype) else "agrees"


def _place(state, tile_id: str, kind: str | None, facts: dict[str, dict]) -> dict:
    if kind == frames.FIELD:
        return {k: facts.get(tile_id, {}).get(k) for k in FIELD_FACTS}
    if kind == frames.CORE:  # from the core's own row, else from one of its tiles
        rows = [t[t.core_id == tile_id].iloc[0].to_dict() for t in (state.core_images, state.tiles) if not t.empty and (t.core_id == tile_id).any()]
        return {k: next((_text(r.get(k)) for r in rows if _text(r.get(k)) is not None), None) for k in FIELD_FACTS}
    return {}


def _outline(state, spot: label_crops.Spot, outlines: pd.DataFrame, size: int) -> tuple[list | None, str | None]:
    """The object's outline in crop pixels: one a person drew, else the AI's mask outline."""
    near = outlines[(outlines.tile_id == spot.tile_id) & (np.hypot(outlines.x.astype(float) - spot.x, outlines.y.astype(float) - spot.y) <= pool.MATCH_PX)]
    if len(near) and isinstance(near.points.iloc[-1], list):
        x0, y0 = round(spot.x) - size // 2, round(spot.y) - size // 2
        return [[float(px) - x0, float(py) - y0] for px, py in near.points.iloc[-1]], "person"
    try:
        found = label_crops.outline(state, spot)
    except (OSError, ValueError, KeyError):
        found = None
    return (found, "AI") if found else (None, None)


@router.get("/labels/sessions/{name}/objects", dependencies=admin)
def session_objects(name: str, request: Request, stain: str = "", values: str = "", tile: str = "", person: str = "", ai: str = "",
                    offset: int = 0, limit: int = OBJECTS_PAGE) -> dict:
    """The labelled objects of one session (or of the reference labels), each as the crop the model sees around it: what
    people said, the AI's call, the object's outline or size, where it is, and whether its crop is ready (crops that
    need their core image decoded are queued; ask again). Filtered by classes (comma-separated), field or core,
    person and how the AI compares; a page at a time. Positions are in each label's own frame (`labels.frames`)."""
    reference = name == sessions.REFERENCE
    meta = {"title": sessions.REFERENCE_TITLE, "stain": ""} if reference else session_meta(name, request)
    state = get_state()
    labels = store.current(root()) if reference else store.current(root(), session=name)
    classes_of = labels[(sessions.session_of(labels) == name) & (labels.task == "class")]
    stains = sorted(set(classes_of.stain))
    stain = stain or meta.get("stain") or (stains[0] if stains else "")
    if stain not in STAINS:
        return {"name": name, "title": meta.get("title", name), "reference": reference, "stain": stain, "stains": stains, "vocabulary": [],
                "groups": [], "classes": {}, "people": {}, "ai": {}, "objects": [], "total": 0, "all": 0, "crop": None, "preparing": None}
    words = vocabulary(stain)
    found = _session_objects(name, labels, stain)
    verdicts = [ai_verdict(words, i, s, a) for i, s, a in zip(found.identity, found.subtype, found.ai_value)] if len(found) else []
    found = found.assign(verdict=verdicts, shown=found.value.replace("", SPLIT)) if len(found) else found.assign(verdict=[], shown=[])
    classes = {str(k): int(v) for k, v in found.shown.value_counts().items()}
    groups = class_groups(words, set(classes))
    order = [key for g in groups for key in g["keys"]]
    people = {str(k): int(v) for k, v in pd.Series([s for ss in found.sources for s in ss]).value_counts().items()} if len(found) else {}
    ai_counts = {k: int(v) for k, v in found.verdict.fillna("unknown").value_counts().items()}
    chosen = found
    if values:
        chosen = chosen[chosen.shown.isin(set(values.split("\n")))]  # one class per line (names may hold commas)
    if tile:
        chosen = chosen[chosen.tile_id.str.contains(tile, case=False, regex=False)]
    if person:
        chosen = chosen[chosen.sources.apply(lambda ss: person in ss)]
    if ai:
        chosen = chosen[chosen.verdict.fillna("unknown").isin(set(ai.split(",")))]
    page = chosen.iloc[max(0, offset): max(0, offset) + max(1, min(limit, 120))]
    size = label_crops.size_px(state, stain)
    facts = _field_facts(sorted(set(page.tile_id)))
    outlines = labels[(labels.task == "outline") & (labels.stain == stain) & (sessions.session_of(labels) == name)]
    outlines = frames.in_field_frame(outlines, state.tiles, state.project.context_px)
    items, waiting = [], []
    for r in page.itertuples():
        spot = label_crops.Spot(stain, r.tile_id, float(r.x), float(r.y))
        kind = label_crops.frame(state, r.tile_id, stain)
        crop = label_crops.status(state, spot) if kind else {"state": "missing"}
        if crop["state"] in ("waiting", "preparing"):
            waiting.append((label_crops.core_of(state, spot) or ("",))[0])
        points, drawn_by = _outline(state, spot, outlines, size) if kind else (None, None)
        on_core = label_crops.core_of(state, spot) if kind else None
        items.append({"tile_id": r.tile_id, "x": float(r.x), "y": float(r.y), "frame": kind, "value": r.shown, "identity": r.identity,
                      "subtype": r.subtype, "labels": r.labels, "time": r.time, "ai_value": words.describe(r.ai_value) if r.ai_value else r.ai_value,
                      "ai_probability": r.ai_probability, "ai": r.verdict,
                      "radius_px": None if r.radius_px is None or pd.isna(r.radius_px) else round(float(r.radius_px), 1),
                      "outline": points, "outline_by": drawn_by, "crop": crop, **_place(state, r.tile_id, kind, facts),
                      "core_x": round(on_core[1]) if on_core else None, "core_y": round(on_core[2]) if on_core else None})
    preparing = {"objects": len(waiting), "cores": len(set(waiting)),
                 "ahead": max([i["crop"].get("ahead", 0) for i in items if i["crop"]["state"] == "waiting"], default=0)} if waiting else None
    return {"name": name, "title": meta.get("title", name), "reference": reference, "stain": stain, "stains": stains, "vocabulary": order,
            "groups": groups, "classes": classes, "people": people, "ai": ai_counts, "objects": items, "total": int(len(chosen)), "all": int(len(found)),
            "crop": {"size_px": size, "um": CROP_UM[stain], "pixel_um": state.project.pixel_size_um}, "preparing": preparing}


@router.get("/labels/crops/{stain}/{tile_id}.jpg", dependencies=admin)
def label_crop(stain: str, tile_id: str, x: float, y: float) -> FileResponse:
    """The crop around one labelled object (positions as `session_objects` gives them); 409 while its core image is
    still being read."""
    if stain not in STAINS:
        raise HTTPException(404, f"unknown stain {stain}")
    state = get_state()
    spot = label_crops.Spot(stain, tile_id, x, y)
    try:
        return FileResponse(label_crops.get(state, spot), headers={"Cache-Control": "private, max-age=86400"})
    except label_crops.NotReady as error:
        label_crops.status(state, spot)
        raise HTTPException(409, str(error)) from error
    except (KeyError, FileNotFoundError) as error:
        raise HTTPException(404, str(error).strip("'")) from error


@router.get("/labels/share", dependencies=admin)
def share(request: Request) -> dict:
    """Addresses at which other computers on the network reach this project's annotation pages."""
    port = (request.scope.get("server") or (None, request.url.port or 80))[1]  # the port this server listens on, also behind a proxy
    host = socket.gethostname()
    links = [f"http://{a}:{port}{annotate_path()}" for a in [*local_addresses(host), host]]
    opened = request.headers.get("host", "")  # the address this page was opened at, when other computers can use it too
    if opened and opened.split(":")[0] not in ("localhost", "") and not opened.startswith("127.") and not public_address(request):
        links = [f"http://{opened}{annotate_path()}"] + [link for link in links if not link.startswith(f"http://{opened}/")]
    public = public_address(request)
    return {"links": ([f"{public}{annotate_path()}"] if public else []) + links,
            "network": bool(public) or os.environ.get("STAINID_HOST", "0.0.0.0") != "127.0.0.1"}


def local_addresses(host: str) -> list[str]:
    """This computer's network addresses (not 127.x: 127.0.1.1 is what many Linux computers call themselves)."""
    found = set()
    with contextlib.suppress(OSError):
        found |= {info[4][0] for info in socket.getaddrinfo(host, None, socket.AF_INET)}
    with contextlib.suppress(OSError), socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        probe.connect(("192.0.2.1", 9))  # picks the address of the default route; no packet is sent
        found.add(probe.getsockname()[0])
    return sorted(a for a in found if not a.startswith("127."))


def public_address(request: Request) -> str:
    """The address this page was opened at when it is a name rather than this computer (e.g. through a tunnel or a
    reverse proxy: https://tma.example.org); empty for localhost and plain IP addresses."""
    host = request.headers.get("x-forwarded-host") or request.headers.get("host", "")
    name = host.rsplit(":", 1)[0] if not host.startswith("[") else ""
    if not name or name == "localhost" or re.fullmatch(r"[\d.]+", name) or "." not in name:
        return ""
    scheme = request.headers.get("x-forwarded-proto") or request.url.scheme
    return f"{scheme}://{host}"


def annotate_path(session: str = "") -> str:
    """The web app's address of the annotation pages of this project (and of one session): /p/<project>/annotate[/name]."""
    return f"/p/{get_state().id}/annotate" + (f"/{session}" if session else "")


def free_name(title: str, stain: str, kind: str) -> str:
    """A session name made from its title (or stain and kind) that no session uses yet."""
    base = re.sub(r"[^a-z0-9]+", "_", (title or f"{stain} {'review' if kind == 'fields' else 'queue'}").lower()).strip("_")[:40] or "session"
    taken = set(sessions.names(root())) | {p.name.rsplit("-", 2)[0] for p in (root() / sessions.FOLDER / ".deleted").glob("*")} | {sessions.REFERENCE}
    return next(name for name in (base if i == 1 else f"{base}_{i}" for i in range(1, 10_000)) if name not in taken)


def _known_fields(stain: str, tile_ids: list[str]) -> list[str]:
    """These fields, checked to be tiles of this stain that can be labelled now (400 with the reason otherwise)."""
    tiles = get_state().tiles
    tiles = tiles[tiles.stain == stain] if not tiles.empty else tiles
    wanted = list(dict.fromkeys(tile_ids))
    unknown = [t for t in wanted if t not in set(tiles.tile_id)]
    if unknown:
        raise HTTPException(400, f"unknown {stain} fields: {', '.join(unknown[:5])}")
    chosen = tiles[tiles.tile_id.isin(set(wanted))].drop_duplicates("tile_id")
    blocked = [b for b in readiness.blockers(get_state().project, chosen) if b]
    if blocked:
        raise HTTPException(400, f"{len(blocked)} of these fields cannot be labelled yet. " + " ".join(sorted({b["reason"] for b in blocked})))
    return wanted


def _pick(stain: str, n: int, seed: int) -> list[str]:
    """n fields that are ready, that nobody has checked and no session has, spread over TMAs and groups."""
    tiles = get_state().labelling_tiles
    tiles = tiles[tiles.stain == stain] if not tiles.empty else tiles
    reviewed = set(store.reviewed_fields(store.current(root()), stain))
    free = tiles[~tiles.tile_id.isin(reviewed | set(sessions.taken(root())))] if not tiles.empty else tiles
    if free.empty:
        raise HTTPException(400, f"Every {stain} field has been checked or belongs to a session already.")
    blocked = readiness.blockers(get_state().project, free)
    if all(blocked):
        first = pd.Series([b["reason"] for b in blocked]).value_counts().index[0]
        raise HTTPException(400, f"No {stain} field can be labelled yet. {first}")
    return sessions.pick_fields(free[[b is None for b in blocked]], set(), n, seed)


@router.post("/labels/sessions", dependencies=admin)
def create_session(body: NewSession):
    """A new labelling session, made in the background when that takes a while (choosing the candidates of a large
    scope): the page then follows its progress (`work`)."""
    return work.answer("new session", ("new session", uuid.uuid4().hex), lambda task: _create_session(body), work.NOW, "Making the session",
                       then="result", admin=True)


def _create_session(body: NewSession) -> dict:
    if not body.name:
        body.name = free_name(body.title, body.stain, body.kind)
    if not re.fullmatch(r"[A-Za-z0-9_\-]+", body.name):
        raise HTTPException(400, "name may contain letters, digits, - and _ only")
    if body.name == sessions.REFERENCE:
        raise HTTPException(400, f"{sessions.REFERENCE} is the name of the imported reference labels; choose another name")
    if body.kind == "areas":
        return _areas().create(body)
    if body.round:
        return _rounds().create_round_session(body)
    if body.kind == outline.KIND:
        from stainid.api.routes.outline import RETIRED

        raise HTTPException(400, RETIRED)
    fields: list[str] = []
    if body.kind == "fields":
        fields = _known_fields(body.stain, body.tile_ids) if body.tile_ids else _pick(body.stain, 20 if body.n_fields is None else body.n_fields, len(sessions.names(root())))
        if not fields:
            raise HTTPException(400, f"no {body.stain} fields left to review")
    try:
        meta = sessions.create(root(), body.name, title=body.title or body.name, stain=body.stain, kind=body.kind, tools=body.tools,
                               ai_view=body.ai_view, per_field=max(1, body.per_field), rule=body.rule, fields=fields, instructions=body.instructions,
                               subtypes=body.subtypes)
    except FileExistsError:
        raise HTTPException(409, "a session with this name exists")
    except ValueError as error:
        raise HTTPException(400, str(error))
    prepare_job(body.stain, 0 if fields else (20 if body.n_fields is None else body.n_fields), body.name if fields else "")
    if body.kind == "queue":
        queue_pictures.prepare(get_state(), meta, pool.all_objects(get_state().project, body.stain), store.current(root()), force=True)
    return sessions.update(root(), body.name, open=True) if body.open else meta


@router.post("/labels/sessions/{name}/open", dependencies=admin)
def set_open(name: str, body: Opening, request: Request) -> dict:
    session_meta(name, request)
    meta = sessions.update(root(), name, open=body.open)
    if meta["kind"] == outline.KIND:
        sam_embeddings.prepare_session(get_state(), meta, force=True)
    if rounds.is_round(meta) and meta["open"]:
        _rounds().warm(meta, force=True)
    return meta


@router.post("/labels/sessions/{name}/fields", dependencies=admin)
def add_fields(name: str, body: MoreFields, request: Request) -> dict:
    """More fields for a session: these exact ones (`tile_ids`), or `n` that stainID picks."""
    meta = labelled_here(session_meta(name, request))
    if meta["kind"] == outline.KIND:
        from stainid.api.routes.outline import RETIRED

        raise HTTPException(400, RETIRED)
    if rounds.is_round(meta):
        return _rounds().add_fields(meta, body.n, body.tile_ids, body.seed)
    if meta["kind"] != "fields":
        prepare_job(meta["stain"], body.n)
        return meta
    more = _known_fields(meta["stain"], body.tile_ids) if body.tile_ids else _pick(meta["stain"], body.n, len(meta["fields"]))
    if not more:
        raise HTTPException(400, f"no {meta['stain']} fields left that nobody has checked")
    updated = sessions.add_fields(root(), name, more)
    prepare_job(meta["stain"], 0, name)
    return updated


@router.post("/labels/sessions/{name}/fields/remove", dependencies=admin)
def remove_fields(name: str, body: FieldList, request: Request) -> dict:
    """Take fields out of a session (and its scope); labels already made on them are kept. A "Fix the field" session's
    work made ahead (pictures, Segment Anything) is queued again for the fields it keeps; the caches built from its
    fields (the confidence, the core maps) are keyed by them and follow by themselves."""
    meta = session_meta(name, request)
    updated = sessions.remove_fields(root(), name, body.tile_ids)
    if rounds.is_round(meta):
        _rounds().warm(updated, force=True)
    return updated


def _with_place(meta: dict, fields: list[dict], states: dict[str, dict], labels: pd.DataFrame) -> list[dict]:
    """Each field for the person labelling with its TMA and core (the fields page groups them so, each core drawn as its
    tiles), its square in the core image (`box`), and where it stands for everyone (`state`, who is working on it or has
    it open, who finished it and when, as `sessions.field_states`)."""
    facts = _field_facts([f["tile_id"] for f in fields])
    wanted = {f["tile_id"] for f in fields}
    here = labels[(labels.stain == meta["stain"]) & labels.tile_id.isin(wanted)]  # this session's own
    done = here[(here.task == "done") & ~here.source.map(store.is_ai_label)]
    everyone = store.current(root())
    everyone = everyone[(everyone.stain == meta["stain"]) & everyone.tile_id.isin(wanted)]
    ai_reviewed = set(everyone.tile_id[everyone.source.map(store.is_ai_label)])  # the AI's labels (its own sessions): never shown while labelling, only flagged
    bad: dict[str, dict[str, str]] = {}  # people's marks of a whole field as bad: who, and the type
    for r in here[(here.task == "exclude") & ~here.source.map(store.is_ai_label)].itertuples():
        if isinstance(r.meta, dict) and r.meta.get("whole_field"):
            bad.setdefault(str(r.tile_id), {})[str(r.source)] = str(r.meta.get("reason") or "")
    when = done.groupby(["tile_id", "source"]).time.max().to_dict() if len(done) else {}
    prefix = get_state().project.tma_prefix or ""
    out = []
    for f in fields:
        place, s = facts.get(f["tile_id"], {}), states.get(f["tile_id"], {})
        done_by = s.get("done_by", f.get("done_by", []))
        out.append({**f, "tma": place.get("tma"), "tma_name": f"{prefix}{place['tma']}" if place.get("tma") else None,
                    "core_id": place.get("core_id"), "core": place.get("core_label") or place.get("core_id"),
                    "box": [place[k] for k in BOX] if all(place.get(k) is not None for k in BOX) else None, "state": s.get("state"),
                    "working": s.get("working", []), "open_by": s.get("open_by", f.get("open_by", [])), "done_by": done_by,
                    "done_at": {p: float(when[(f["tile_id"], p)]) for p in done_by if (f["tile_id"], p) in when},
                    "ai_reviewed": f["tile_id"] in ai_reviewed, "marked_bad": bad.get(f["tile_id"], {})})
    return out


def _core_layouts(meta: dict, fields: list[dict]) -> dict[str, dict]:
    """Each core of the session drawn as its tiles: the frame around all its tiles of the stain (core px) and the tiles
    that are not in this session (drawn faint, so the core's shape reads)."""
    tiles = get_state().tiles
    cores = {f["core_id"] for f in fields if f.get("core_id")}
    if tiles.empty or not cores:
        return {}
    key = (id(tiles), meta["name"], meta["stain"], hash(tuple(f["tile_id"] for f in fields)))
    kept = _LAYOUTS.get(key)
    if kept is not None:  # the page asks every few seconds: the same tiles, the same drawing
        return kept
    mine = {f["tile_id"] for f in fields}
    rows = tiles[(tiles.stain == meta["stain"]) & tiles.core_id.astype(str).isin(cores)].drop_duplicates("tile_id")
    out = {}
    for core_id, group in rows.groupby(rows.core_id.astype(str)):
        x0, y0 = int(group.x_px.min()), int(group.y_px.min())
        x1, y1 = int((group.x_px + group.width_px).max()), int((group.y_px + group.height_px).max())
        others = group[~group.tile_id.isin(mine)]
        out[core_id] = {"x": x0, "y": y0, "width": x1 - x0, "height": y1 - y0,
                        "others": [[int(r.x_px), int(r.y_px), int(r.width_px), int(r.height_px), str(r.tile_id)] for r in others.itertuples()]}
    if len(_LAYOUTS) > 16:
        _LAYOUTS.clear()
    _LAYOUTS[key] = out
    return out


_LAYOUTS: dict[tuple, dict[str, dict]] = {}


NEAR_THRESHOLD = 0.2  # how far (in probability) from the acceptance threshold an object still counts as uncertain
_CONFIDENCE: dict[tuple, dict[str, dict]] = {}  # (project, session, stain, level, fields) -> the scores and the stores' signatures
_CONFIDENCE_LOCK = threading.Lock()
_CONFIDENCE_BUSY: set[tuple] = set()


def field_confidence(meta: dict) -> dict[str, dict]:
    """How sure detection is on each field of a session: each of the field's own objects weighs how close its probability
    sits to its acceptance threshold (1 at it, 0 from `NEAR_THRESHOLD` away); `mean` is the field's average of those
    (the score "By confidence" ranks by: higher, less confident; None without objects), `near` their sum, `objects`
    how many it has. Straight from the stage-4 candidate store, one read per core of only these columns, whether or not a field is
    prepared (nothing is prepared to score it); kept until a core's store file changes (its signature)."""
    from stainid import candidates as stored  # noqa: PLC0415

    state = get_state()
    tiles = state.tiles
    rows = tiles[tiles.tile_id.isin(set(meta["fields"]))].drop_duplicates("tile_id") if not tiles.empty else tiles
    cores = sorted({str(c) for c in rows.core_id}) if len(rows) else []
    files = {c: stored.stored(state.project, meta["stain"], c) for c in cores}
    signature = tuple((c, cache.signature(f) if f is not None else None) for c, f in files.items())
    key = (state.id, meta["name"], meta["stain"], rounds.level(meta), len(meta["fields"]), hash(tuple(meta["fields"])))
    with _CONFIDENCE_LOCK:
        kept = _CONFIDENCE.get(key)
        if kept is not None and kept["signature"] == signature:
            return kept["scores"]
    wanted = set(meta["fields"])
    level = rounds.level(meta)

    def core_scores(core: str) -> dict[str, dict]:
        if files[core] is None:
            return {}
        found = stored.read_core(state.project, meta["stain"], core, columns=["tile_id", "probability", "threshold", "proposal_threshold"])
        if found.empty:
            return {}
        found = found[found.tile_id.astype(str).isin(wanted)]
        if level is not None and "proposal_threshold" in found:  # the candidates a session at this level shows (`at_level`)
            found = found[found.proposal_threshold.isna() | (pd.to_numeric(found.proposal_threshold, errors="coerce") >= float(level) - 1e-9)]
        probability = pd.to_numeric(found.probability, errors="coerce")
        threshold = pd.to_numeric(found.threshold, errors="coerce").fillna(0.5) if "threshold" in found else 0.5  # none kept: the usual 0.5
        weight = (1 - (probability - threshold).abs() / NEAR_THRESHOLD).clip(lower=0).fillna(0)
        frame = pd.DataFrame({"tile_id": found.tile_id.astype(str), "near": weight})
        return {t: {"near": round(float(g.near.sum()), 2), "objects": int(len(g)), "mean": round(float(g.near.mean()), 3)}
                for t, g in frame.groupby("tile_id")}

    from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415

    scores: dict[str, dict] = {t: {"near": 0.0, "objects": 0, "mean": None} for t in meta["fields"]}
    with ThreadPoolExecutor(max_workers=8) as pool:
        for part in pool.map(core_scores, cores):
            scores.update(part)
    with _CONFIDENCE_LOCK:
        _CONFIDENCE[key] = {"signature": signature, "scores": scores}
    return scores


def warm_confidence(meta: dict) -> None:
    """Score a session's fields in the background (when its page opens), so "By confidence" answers at once."""
    if meta.get("practice") or meta["kind"] != "fields":
        return
    state_id = get_state().id
    key = (state_id, meta["name"])
    with _CONFIDENCE_LOCK:
        if key in _CONFIDENCE_BUSY:
            return
        _CONFIDENCE_BUSY.add(key)
    from stainid.api.state import use_project  # noqa: PLC0415

    def run() -> None:
        try:
            with use_project(state_id):
                field_confidence(meta)
        except Exception:  # noqa: BLE001 — a warm-up only: the request computes it again and says what failed
            pass
        finally:
            with _CONFIDENCE_LOCK:
                _CONFIDENCE_BUSY.discard(key)

    threading.Thread(target=run, name=f"confidence-{meta['name']}", daemon=True).start()


@router.get("/annotate/{name}/confidence")
def session_confidence(name: str, request: Request, person: str | None = Depends(viewer)) -> dict:
    """How sure detection is on each field of a session (the fields page's "By confidence" order), `field_confidence`.
    More weight near the threshold: less confident."""
    meta = labelled_here(session_meta(name, request))
    if meta.get("practice") or meta["kind"] != "fields":
        return {}
    return field_confidence(meta)


def _field_facts(tile_ids: list[str]) -> dict[str, dict]:
    """Where each field is (core, TMA, region, group) and its square in the core image."""
    tiles = get_state().tiles
    rows = tiles[tiles.tile_id.isin(set(tile_ids))] if not tiles.empty else tiles
    return {r["tile_id"]: {k: _text(r.get(k)) for k in FIELD_FACTS} | {k: int(r[k]) for k in BOX} for r in rows.to_dict("records")}


def _text(value) -> str | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return str(int(value)) if isinstance(value, float) and value.is_integer() else str(value)


@router.get("/labels/sessions/{name}/fields", dependencies=admin)
def session_fields(name: str, request: Request) -> list[dict]:
    """Every field of a session with where it stands (see `sessions.field_states`) and where it is."""
    meta = session_meta(name, request)
    facts = _field_facts(meta["fields"])
    return [{**facts.get(s["tile_id"], {}), **s} for s in _field_states(meta)]


def _field_states(meta: dict) -> list[dict]:
    """Where each field of a session stands, the same for the admin's page and the person labelling (`sessions.field_states`)."""
    if rounds.is_round(meta):
        return _rounds().admin_fields(meta) if meta["kind"] == "fields" else []
    labels = session_labels(meta)
    working = outline.started(labels, meta["name"]) if meta["kind"] == outline.KIND else sessions.labelled_by(people_only(labels[labels.session == meta["name"]]), meta["stain"])
    return sessions.field_states(meta, finished_fields(meta, labels), working, ready_fields(meta), busy(meta["name"]))


@router.get("/labels/picker", dependencies=admin)
def picker(stain: str, tma: str = "", region: str = "", group: str = "", core: str = "", free: bool = False, offset: int = 0,
           limit: int = 12, source: str = "") -> dict:
    """The fields people can be asked to label, core by core (for choosing them by hand): each field's square in its
    core image, whether it already has labels or belongs to a session, and, when it cannot be labelled yet, why and
    which step comes first (`blocked`, see `labels.readiness`; a core whose fields all wait for the same thing carries
    it too). `free` keeps only fields nobody has labelled that no session has."""
    state = get_state()
    tiles = state.labelling_tiles
    tiles = tiles[tiles.stain == stain] if not tiles.empty else tiles
    facets = {key: sorted({str(v) for v in tiles[column].dropna()}, key=lambda v: (len(v), v)) if column in tiles else []
              for key, column in (("tmas", "tma"), ("regions", "region"), ("groups", "disease_group"))}
    labelled = sessions.labelled_by(store.current(root()), stain)
    taken = sessions.taken(root())
    for column, value in (("tma", tma), ("region", region), ("disease_group", group)):
        if value and column in tiles:
            tiles = tiles[tiles[column].astype(str) == value]
    if core:
        tiles = tiles[tiles.core_id.str.contains(core, case=False, regex=False)]
    if free:
        tiles = tiles[~tiles.tile_id.isin(set(labelled) | set(taken))]
    check = readiness.store_blockers if source == rounds.SOURCE else readiness.blockers  # rounds read the v2 candidate store
    blocked = dict(zip(tiles.tile_id, check(state.project, tiles))) if not tiles.empty else {}
    first_rows = tiles.drop_duplicates("core_id") if not tiles.empty else tiles
    order = [r["core_id"] for r in sorted(first_rows.to_dict("records"), key=_core_order)]
    page = order[max(0, offset): max(0, offset) + max(1, min(limit, 48))]
    images = state.core_images
    cores = []
    for core_id, group_tiles in tiles[tiles.core_id.isin(set(page))].groupby("core_id"):
        first = group_tiles.iloc[0]
        image = images[(images.core_id == core_id) & (images.stain == stain)] if not images.empty else images
        fields = [{"tile_id": t["tile_id"], **{k: int(t[k]) for k in BOX}, "labelled_by": sorted(labelled.get(t["tile_id"], set())),
                   "sessions": taken.get(t["tile_id"], []), "blocked": blocked.get(t["tile_id"])}
                  for t in group_tiles.sort_values("selection_order").to_dict("records")]
        reasons = {f["blocked"]["reason"] if f["blocked"] else None for f in fields}
        cores.append({"core_id": core_id, **{k: _text(first.get(k)) for k in ("tma", "core_label", "region", "disease_group", "donor_id")},
                      "width": int(image.native_width_px.iloc[0]) if len(image) else None,
                      "height": int(image.native_height_px.iloc[0]) if len(image) else None, "fields": fields,
                      "blocked": fields[0]["blocked"] if len(reasons) == 1 and None not in reasons else None})
    cores.sort(key=lambda c: order.index(c["core_id"]))
    return {"cores": cores, "total_cores": len(order), "total_fields": int(len(tiles)), "ready_fields": sum(b is None for b in blocked.values()),
            "facets": facets}


def _core_order(row: dict) -> tuple:
    """TMA by number, then the core's position on it row by row ('B-12' is row 12, column B)."""
    tma, label = str(row.get("tma", "")), str(row.get("core_label", ""))
    column, _, number = label.partition("-")
    return (int(tma) if tma.isdigit() else 1 << 30, tma, int(number) if number.isdigit() else 0, column)


@router.get("/labels/fields/{tile_id}/thumb.jpg", dependencies=admin)
def field_thumb(tile_id: str) -> FileResponse:
    return _thumb(tile_id)


def _thumb(tile_id: str) -> FileResponse:
    state = get_state()
    try:
        return FileResponse(pyramid.field_thumbnail(state, state.tile(tile_id)), headers={"Cache-Control": "private, max-age=86400"})
    except (KeyError, FileNotFoundError) as error:
        raise HTTPException(404, str(error).strip("'"))



@router.get("/labels/sessions/{name}/labels-count", dependencies=admin)
def session_label_count(name: str, request: Request) -> dict:
    """How many labels deleting this session removes (the delete dialog says so)."""
    session_meta(name, request)
    return {"labels": sessions.labels_in(root(), name)}


@router.delete("/labels/sessions/{name}", dependencies=admin)
def delete_session(name: str, request: Request) -> dict:
    """Deletes a session with every label made in it (`sessions.remove`: folder and logs into sessions/.deleted)."""
    session_meta(name, request)
    moved, removed = sessions.remove(root(), name)
    return {"moved_to": str(moved), "labels_removed": removed}


@router.post("/labels/prepare", dependencies=admin)
def prepare(body: Prepare) -> dict:
    prepare_job(body.stain, body.fields)
    return {"ok": True}


@router.get("/labels/sessions/{name}/people", dependencies=admin)
def session_people(name: str, request: Request) -> list[dict]:
    meta = session_meta(name, request)
    return labellers(meta, session_labels(meta))


def with_classes(labels: pd.DataFrame) -> pd.DataFrame:
    """The labels with, for class labels, the class read in the project's vocabulary: `identity` and `subtype` (ids;
    labels made before subtypes existed mapped onto them), the class `role` (what the stain's model learns from it) and
    `class` as the page shows it. `value` stays as the label log has it."""
    out = labels.assign(identity="", subtype=labels.get("subtype"), role="", **{"class": ""})
    for stain in set(labels.stain[labels.task == "class"].dropna()):
        words = vocabulary(stain)
        resolved = store.classes(labels, stain, words)
        out.loc[resolved.index, "identity"] = resolved.identity
        out.loc[resolved.index, "subtype"] = resolved.subtype
        out.loc[resolved.index, "role"] = resolved.role
        out.loc[resolved.index, "class"] = [words.display(i, s) for i, s in zip(resolved.identity, resolved.subtype)]
    return out


@router.get("/labels/export.csv", dependencies=admin)
def export(session: str = "") -> Response:
    labels = store.current(root(), session=session) if session else store.current(root())
    if session:
        labels = labels[labels.session == session]
    tiles = get_state().tiles[["tile_id", "core_id", "tma", "disease_group", "region"]].drop_duplicates("tile_id")
    table = with_classes(labels.drop(columns=["deleted"])).merge(tiles, on="tile_id", how="left")
    table["time"] = pd.to_datetime(table.time, unit="s").dt.strftime("%Y-%m-%dT%H:%M:%S")
    return Response(table.to_csv(index=False), media_type="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="labels{"_" + session if session else ""}.csv"'})


# ---------- annotators ----------

class Mark(BaseModel):
    task: str
    tile_id: str = ""  # a label naming its `candidate` takes the candidate's field
    x: float | None = None
    y: float | None = None
    value: str = ""
    subtype: str = ""  # class labels: the subtype's id, under the identity `value`
    points: list[list[float]] | None = None
    deleted: bool = False
    candidate: str = ""  # annotation rounds: the v2 candidate the label is about (its field and position come from the store)
    keep: bool = False  # annotation rounds: the AI's call on `candidate` is right (recorded as confirmed)
    mask: int | None = None  # annotation rounds: which of the candidate's Segment Anything masks is right (small to large)
    png: str = ""  # round 2, task "exclude": the area marked as bad, painted (a PNG data URL at 1/`scale` of the field picture)
    pngs: dict[str, str] = {}  # … one painted mask per type (labels.exclusions.KINDS: "1"–"4", "include" for a cleared part)
    scale: float = 1.0
    meta: dict | None = None  # where a label comes from: a split or merge (`labels.objectedit`)


class Marks(BaseModel):
    records: list[Mark]


@router.get("/annotate")
def list_open(request: Request, person: str | None = Depends(viewer)) -> list[dict]:
    """The sessions of this project the person may label: every one for an admin, else the ones they joined."""
    from stainid.api import accounts  # noqa: PLC0415
    from stainid.api.session import current_account  # noqa: PLC0415

    account = current_account(request)
    joined = None if is_admin(request) else {n for p, n, _ in accounts.sets_of(account.id) if p == get_state().id} if account else set()
    return open_sessions(request, person, joined)


def open_sessions(request: Request, person: str | None, only: set[str] | None = None) -> list[dict]:
    """The open sessions (an admin: every one) of the request's project, or of them the ones in `only`, with progress."""
    out = []
    for name in sessions.names(root()):
        if only is not None and name not in only:
            continue
        meta = sessions.meta(root(), name)
        if sessions.label_set_only(meta) or (not meta["open"] and not is_admin(request)):  # nobody labels in an AI review round
            continue
        labels = session_labels(meta)
        reviewed = finished_fields(meta, labels)
        done = sum(person in reviewed.get(t, set()) for t in meta["fields"]) if person else 0
        extra = {"round": rounds.round_of(meta), "progress": _rounds().progress(meta)} if rounds.is_round(meta) else {"round": None}
        if meta["kind"] == "areas":
            extra = {"round": "areas", "progress": _areas().progress(meta, labels), "cores": len(meta.get("cores") or [])}
        out.append({**{k: meta.get(k) for k in ("name", "title", "stain", "kind", "open", "instructions")}, **extra, "kind_title": sessions.kind_title(meta), "fields": len(meta["fields"]), "done_by_me": done,
                    "my_labels": int(((labels.session == name) & (labels.source == person) & labels.task.isin(("class", "object"))).sum()),
                    "link": annotate_path(name)})
    return out


@router.get("/annotate/{name}")
def session_detail(name: str, request: Request, person: str | None = Depends(viewer)) -> dict:
    meta = labelled_here(session_meta(name, request))
    if meta["kind"] == "areas":
        return {"meta": {**meta, **_areas().detail(meta, person)}, "fields": [], "annotator": person}
    labels = session_labels(meta)
    reviewed = finished_fields(meta, labels)
    prepared = ready_fields(meta)
    open_now = busy(name)
    fields = [{"tile_id": t, "done_by_me": bool(person) and person in reviewed.get(t, set()), "annotators_done": len(reviewed.get(t, set())),
               "open_by_others": len(open_now.get(t, set()) - {person}), "prepared": t in prepared} for t in meta["fields"]]
    if rounds.is_round(meta):
        fields = _rounds().session_fields(meta, person)
        meta = {**meta, **_rounds().detail(meta, person)}
        if meta["open"]:
            _rounds().warm(meta)
    fields = _with_place(meta, fields, {} if meta.get("practice") else {s["tile_id"]: s for s in _field_states(meta)}, labels)
    layouts = _core_layouts(meta, fields)
    warm_confidence(meta)
    words = vocabulary(meta["stain"])
    outlines = {"outline_classes": outline.classes(words)} if meta["kind"] == outline.KIND else {}
    if meta["kind"] == outline.KIND:
        sam_embeddings.prepare_session(get_state(), meta)
    return {"meta": {**meta, "subtypes": asks_subtypes(meta), "vocabulary": words.to_api(), "keys": words.keymap(asks_subtypes(meta)), "tags": FIELD_TAGS,
                     **outlines},
            "fields": fields, "cores": layouts, "needed": rounds.needed(meta), "annotator": person}


def asks_subtypes(meta: dict) -> bool:
    """Whether people are asked for the type of an object (sessions made before this choice existed did ask)."""
    return bool(meta.get("subtypes", True))


@router.get("/annotate/{name}/next")
def next_item(name: str, request: Request, person: str = Depends(annotator), count: int = 1, skip: str = "") -> dict:
    """A fields session: the next field for this person. A queue session: the next `count` objects (up to 20, the
    page shows the first and loads the others' pictures meanwhile), held for this person for a while so others get
    other objects; with how many the rule leaves them and how many of the queue's pictures are made. An annotation
    round: the next field for this person; `skip` puts the field they are on aside (an old round-1 session: 410)."""
    meta = labelled_here(session_meta(name, request))
    if rounds.is_round(meta):
        if meta["kind"] == "queue":
            raise HTTPException(410, _rounds().NO_ROUND_ONE)
        return _rounds().next_field(meta, person, skip)
    labels = session_labels(meta)
    state = get_state()
    if meta["kind"] != "queue":
        prepared = ready_fields(meta)
        ready = {**meta, "fields": [t for t in meta["fields"] if t in prepared]}
        return {"tile_id": sessions.next_field(ready, finished_fields(meta, labels), person, busy(name)),
                "waiting": len(meta["fields"]) - len(ready["fields"])}
    objects = pool.all_objects(state.project, meta["stain"])
    held = (state.id, name)
    items, remaining = sessions.next_objects(meta, objects, labels, person, min(20, max(1, count)),
                                             queue_pictures.ready(state, meta["stain"], objects), _HELD.held(held), vocabulary(meta["stain"]),
                                             _HELD.order(held, person))
    _HELD.hold(held, person, [i["key"] for i in items])
    pictures = queue_pictures.prepare(state, meta, objects, labels, first=pd.DataFrame(items, columns=["tile_id", "x", "y"]))
    return {"object": items[0] if items else None, "objects": items, "remaining": remaining, "pictures": pictures}


@router.post("/annotate/{name}/fields/{tile_id}/leave")
def leave_field(name: str, tile_id: str, request: Request, person: str = Depends(annotator)) -> dict:
    """The person closed this field: the others may open it now (else its hold lapses by itself)."""
    meta = session_meta(name, request)
    if rounds.is_round(meta):
        _rounds().leave_field(meta, tile_id, person)
    return {"left": tile_id}


@router.get("/annotate/{name}/fields/{tile_id}")
def open_field(name: str, tile_id: str, request: Request, edit: bool = False, own: bool = False, person: str = Depends(annotator)) -> dict:
    """A field to label. An annotation round's field that is finished (by this person, or by as many as it needs) opens
    to look at (`read_only`); `edit` (Redo this field: the person who finished it, from their own labels) or `own` (Add
    an extra annotation: someone else, from detection's output and their own labels only) open it to work on."""
    meta = session_meta(name, request)
    if rounds.is_round(meta):
        out = _rounds().open_field(meta, tile_id, person, edit, own)
        return out if isinstance(out, Response) else fast_json(request, out)
    tile = field_tile(meta, tile_id)
    objects = field_objects(meta, tile_id)
    busy(name, tile_id, person)
    state = get_state()
    x0, y0 = render.crop_origin(tile, state.project.context_px)
    shown = [] if meta["ai_view"] == "none" else [
        {"x": float(o.x), "y": float(o.y), "radius_px": float(o.radius_px),
         **({"ai_value": o.ai_value, "ai_probability": float(o.ai_probability)} if meta["ai_view"] == "full" else {})}
        for o in objects.itertuples()]
    outlines = []
    if meta["ai_view"] == "full" and "outline" in meta["tools"]:
        for outline in render.mask_outlines(state, tile_id):
            points = np.asarray(outline["points"], float)
            i = pool.nearest(objects, *points.mean(axis=0), tolerance=max(12.0, float(np.ptp(points, axis=0).max()) / 2))
            if i is not None:
                simple = cv2.approxPolyDP(points.astype(np.int32).reshape(-1, 1, 2), 2.5, True)[:, 0, :]
                outlines.append({"x": float(objects.x.iloc[i]), "y": float(objects.y.iloc[i]), "points": simple.tolist()})
    labels = session_labels(meta)
    mine = labels[(labels.source == person) & (labels.tile_id == tile_id) & (labels.stain == meta["stain"])]
    box = {"x": int(tile["x_px"]) - x0, "y": int(tile["y_px"]) - y0, "width": int(tile["width_px"]), "height": int(tile["height_px"])}
    return {"tile_id": tile_id, "box": box, "objects": shown, "outlines": outlines, "layers": LAYERS[meta["stain"]] if meta["ai_view"] == "full" else [],
            "mine": mine.replace({np.nan: None}).to_dict(orient="records"), "added": added_states(state.project, meta["stain"], tile_id, mine, objects)}


def _clean(meta: dict, mark: Mark, person: str) -> dict:
    stain, tools = meta["stain"], meta["tools"]
    allowed = {"class": "class" in tools or "point" in tools or meta["kind"] == "queue", "outline": "outline" in tools, "tag": "tag" in tools,
               "note": True, "done": meta["kind"] == "fields"}
    if not allowed.get(mark.task):
        raise HTTPException(400, f"{mark.task} labels are not part of this session")
    subtype = None
    if not mark.deleted:
        if mark.task == "class":
            mark.value, subtype = class_of(vocabulary(stain), mark.value, mark.subtype)
        if mark.task == "tag" and mark.value not in FIELD_TAGS:
            raise HTTPException(400, f"unknown tag {mark.value!r}")
        if mark.task == "outline" and (not mark.points or len(mark.points) < 3 or any(len(p) != 2 for p in mark.points)):
            raise HTTPException(400, "an outline needs at least three [x, y] points")
    if mark.task in ("class", "outline") and (mark.x is None or mark.y is None):
        raise HTTPException(400, "object labels need x and y")
    ids = {"class": lambda: store.object_id("class", mark.tile_id, mark.x, mark.y), "outline": lambda: store.object_id("outline", mark.tile_id, mark.x, mark.y),
           "tag": lambda: f"tag:{mark.tile_id}:{mark.value}", "note": lambda: f"note:{mark.tile_id}", "done": lambda: f"done:{mark.tile_id}"}
    return {"id": ids[mark.task](), "source": person, "stain": stain, "task": mark.task, "tile_id": mark.tile_id, "x": mark.x, "y": mark.y,
            "value": mark.value, "subtype": subtype, "points": mark.points, "session": meta["name"], "deleted": mark.deleted}


def class_of(words: vocab.Vocabulary, value: str, subtype: str = "") -> tuple[str, str | None]:
    """The identity and subtype ids of a class label sent by a page (ids, names or an old value; 400 if unknown)."""
    identity, sub = words.resolve(value, subtype)
    found = words.identity(identity) if identity else None
    if found is None:
        raise HTTPException(400, f"unknown class {value!r}")
    if sub and found.subtype(sub) is None:
        raise HTTPException(400, f"{found.name} has no type {sub!r}")
    return found.id, sub or None


@router.get("/annotate/{name}/fields/{tile_id}/added")
def field_added(name: str, tile_id: str, request: Request, person: str = Depends(annotator)) -> list[dict]:
    """Where this person's added objects stand (`added_states`), for the page to follow their measuring."""
    meta = session_meta(name, request)
    tile = field_tile(meta, tile_id)
    labels = session_labels(meta)
    mine = labels[(labels.source == person) & (labels.tile_id == tile_id) & (labels.stain == meta["stain"])]
    if rounds.is_round(meta):
        return _rounds().added_states(meta, tile, mine)
    return added_states(get_state().project, meta["stain"], tile_id, mine, field_objects(meta, tile_id))


@router.post("/annotate/{name}/labels")
def save(name: str, body: Marks, request: Request, person: str = Depends(annotator)) -> list[dict]:
    meta = writable(session_meta(name, request))
    if rounds.is_round(meta):
        raise HTTPException(400, f"an annotation round saves a field at a time: POST /api/annotate/{name}/tiles/<tile_id>/save")
    for mark in body.records:
        if meta["kind"] != "queue":
            field_tile(meta, mark.tile_id)
            busy(name, mark.tile_id, person)
    saved = store.append(root(), [_clean(meta, mark, person) for mark in body.records])
    project = get_state().project
    if meta["kind"] == "queue":
        answered = pd.DataFrame([r for r in saved if r["task"] == "class" and not r["deleted"]], columns=["tile_id", "x", "y"])
        _HELD.release((get_state().id, name), person, list(sessions.object_keys(answered)))
    if any(r["task"] == "class" and not r["deleted"] and pool.nearest(pool.load_field(project, meta["stain"], r["tile_id"]), r["x"], r["y"]) is None
           for r in saved):
        measure_added_soon(project, meta["stain"])
    return saved


@router.post("/annotate/{name}/fields/{tile_id}/finish")
def finish(name: str, tile_id: str, request: Request, person: str = Depends(annotator)) -> list[dict]:
    """Done with this field: every AI call (and AI outline of a positive object) the person left untouched counts as confirmed."""
    meta = writable(session_meta(name, request))
    if rounds.is_round(meta):
        return _rounds().save_tile(meta, tile_id, [], True, person)["saved"]
    field_tile(meta, tile_id)
    objects = field_objects(meta, tile_id)
    labels = session_labels(meta)
    mine = labels[(labels.source == person) & (labels.tile_id == tile_id) & (labels.stain == meta["stain"])]
    records = []
    words = vocabulary(meta["stain"])
    if meta["ai_view"] == "full":
        classes = mine[mine.task == "class"]
        labelled = set(classes.id)
        final = {k: words.positive(v, s) for k, v, s in zip(classes.id, classes.value, classes.get("subtype", pd.Series(None, index=classes.index)))}
        for o in objects.itertuples():
            key = store.object_id("class", tile_id, o.x, o.y)
            if key not in labelled:  # the AI's call, confirmed: its identity, and its type where the session asks for types
                call = words.record(o.ai_value) | ({} if asks_subtypes(meta) else {"subtype": None})
                records.append({"id": key, "source": person, "stain": meta["stain"], "task": "class", "tile_id": tile_id, "x": float(o.x), "y": float(o.y),
                                **call, "session": name, "confirmed": True})
            final.setdefault(key, words.positive(o.ai_value))
        if "outline" in meta["tools"]:
            edited = set(mine.id[mine.task == "outline"])
            for outline in open_field(name, tile_id, request, person=person)["outlines"]:
                key = store.object_id("outline", tile_id, outline["x"], outline["y"])
                if key not in edited and final.get(store.object_id("class", tile_id, outline["x"], outline["y"]), words.positive(NEGATIVE[meta["stain"]])):
                    records.append({"id": key, "source": person, "stain": meta["stain"], "task": "outline", "tile_id": tile_id, "x": outline["x"],
                                    "y": outline["y"], "points": outline["points"], "session": name, "confirmed": True})
    records.append({"id": f"done:{tile_id}", "source": person, "stain": meta["stain"], "task": "done", "tile_id": tile_id, "session": name})
    return store.append(root(), records)


@router.get("/annotate/{name}/fields/{tile_id}/image.jpg")
def image(name: str, tile_id: str, request: Request, person: str | None = Depends(viewer)) -> FileResponse:
    field_tile(session_meta(name, request), tile_id)
    return FileResponse(render.field_jpeg(get_state(), tile_id), headers={"Cache-Control": "private, max-age=86400"})


@router.get("/annotate/{name}/fields/{tile_id}/thumb.jpg")
def annotate_thumb(name: str, tile_id: str, request: Request, person: str | None = Depends(viewer)) -> FileResponse:
    """A field's small picture; also of the other tiles of a core the session has fields of (the core maps draw them
    dimmed, so the whole core shows)."""
    meta = session_meta(name, request)
    if tile_id not in set(meta["fields"]):
        tiles = get_state().tiles
        row = tiles[tiles.tile_id == tile_id] if not tiles.empty else tiles
        if row.empty or str(row.stain.iloc[0]) != meta["stain"] or str(row.core_id.iloc[0]) not in _session_cores(meta):
            raise HTTPException(404, f"{tile_id} is not part of this session")
    return _thumb(tile_id)


def _session_cores(meta: dict) -> set[str]:
    key = ("cores", get_state().id, meta["name"], len(meta["fields"]), hash(tuple(meta["fields"])))
    kept = _LAYOUTS.get(key)
    if kept is None:
        tiles = get_state().tiles
        kept = {str(c) for c in tiles[tiles.tile_id.isin(set(meta["fields"]))].core_id} if not tiles.empty else set()
        _LAYOUTS[key] = kept
    return kept


@router.get("/annotate/{name}/fields/{tile_id}/layer/{layer}.png")
def layer(name: str, tile_id: str, layer: str, request: Request, person: str | None = Depends(viewer)) -> FileResponse:
    meta = session_meta(name, request)
    field_tile(meta, tile_id)
    if meta["ai_view"] != "full" or layer not in LAYERS[meta["stain"]]:
        raise HTTPException(404, f"layer {layer} is not shown in this session")
    return FileResponse(OVERLAYS[layer](get_state(), tile_id), headers={"Cache-Control": "private, max-age=86400"})


@router.get("/annotate/{name}/object.jpg")
def object_image(name: str, tile_id: str, x: float, y: float, request: Request, person: str | None = Depends(viewer)) -> FileResponse:
    """The picture of one queue object: made ahead (`queue_pictures`), else cut now (seconds when its core image has
    to be read)."""
    meta = session_meta(name, request)
    try:
        crop = queue_pictures.picture(get_state(), meta["stain"], tile_id, x, y)
    except FileNotFoundError as error:
        raise HTTPException(404, str(error))
    return FileResponse(crop, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=86400"})
