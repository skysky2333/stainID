"""The "Mark bad areas" round (`stainid.labels.areas`) and the Cores view's review layers.

Round pages (annotators): the next core for a person (held for them while their page is open), one core's stain
images with their deep-zoom pyramids, thumbnails, detection's automatic areas (`stainid.exclusions.automatic`, the
starting point) and people's areas so far (`annotated`), and saving a core: an edit of people's areas per stain image
painted or cleared, and every stain image checked (`done`, value "areas"), also with no edits. People's areas only train
and evaluate models (the bad-area detector, `GET /models/bad-areas`, and the object models); the analysis never reads
them.
The tutorial practises on a made-up session `practice-areas` whose saves are echoed and never kept.

Admin pages: where the areas round stands (images checked per stain, people's agreement on the double-checked cores,
the disagreements to settle), the bad-area detector against people (Models page), and for the Cores view the review
status of a core image's tiles (`labels.status`) and the "Annotations" layer (`labels.compare`, people's areas).
"""
from __future__ import annotations

import secrets
import threading
import time
from collections import OrderedDict
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

from stainid import exclusions
from stainid.api import pyramid, render, slide_pyramid, slide_units, store_view
from stainid.api.routes.annotate import NewSession, free_name, root, session_meta, writable
from stainid.api.session import annotator, require_admin
from stainid.api.state import get_state
from stainid.labels import areas, compare, sessions, status, store
from stainid.labels import exclusions as marks

router = APIRouter(tags=["areas"])
admin = [Depends(require_admin)]
_HELD = sessions.Reservations(areas.HOLD_SECONDS)  # cores on someone's screen
PRACTICE = "practice-areas"


def hold_key(meta: dict) -> tuple:
    """Holds on cores are per session (sessions are separate)."""
    return get_state().id, "areas", meta["name"]


def held(meta: dict) -> dict[str, set[str]]:
    return _HELD.held(hold_key(meta))


def checked(labels: pd.DataFrame | None = None) -> dict[tuple[str, str], set[str]]:
    """Who checked each core image's bad areas: in every session (the overview, a new session's choice of cores), or in
    the `labels` given."""
    return status.areas_checked(store.current(root()) if labels is None else labels)


def session_checked(meta: dict, labels: pd.DataFrame | None = None) -> dict[tuple[str, str], set[str]]:
    """Who checked each core image in this session: its own labels only (`store.current(root, session)`); in a
    whole-slide project each slide too (who checked every one of its regions)."""
    return slide_units.with_slides(checked(store.current(root(), session=meta["name"]) if labels is None else labels))


# ---------------------------------------------------------------- making a session (admin)

def create(body: NewSession) -> dict:
    """A "Mark bad areas" session: the cores (each with all its stain images) of the chosen TMAs, `n_fields` of them
    (-1 or empty: all), in the chosen order. Sessions are independent: cores checked or held in another session are
    drawn as any other."""
    state = get_state()
    tiles = state.tiles
    if body.tmas and not tiles.empty:
        tiles = tiles[tiles.tma.astype(str).isin({str(t) for t in body.tmas})]
    stains_of = areas.core_stains(tiles)
    if not stains_of:
        raise HTTPException(400, "No core has analysed tiles" + (" on these TMAs" if body.tmas else ""))
    slides = slide_units.unit_stains(stains_of)  # a whole-slide project: the slides are the units, each checked whole
    regions = {c: [r.core_id for r in slide_units.slide_unit(c).regions if r.core_id in stains_of] for c in slides}
    stains_of = slides or stains_of
    order = body.order if body.order in areas.ORDERS else "problems"
    left = list(stains_of)
    seed = body.seed if body.seed is not None else secrets.randbelow(2**31)
    scores = areas.priority({c: _signals(c, stains_of[c], regions.get(c)) for c in left}) if order == "problems" else {}
    ordered = areas.order(left, scores, order, seed)
    n = -1 if body.n_fields is None else body.n_fields
    chosen = ordered if n < 0 else ordered[:n]
    name = body.name or free_name(body.title or "bad areas", "all", areas.KIND)
    meta = sessions.create(root(), name, title=body.title or name, stain="all", kind=areas.KIND, tools=["area"], ai_view="full", per_field=1,
                           rule="random", fields=[], instructions=body.instructions, source=areas.SOURCE, cores=chosen,
                           stains_of={c: stains_of[c] for c in chosen}, second=areas.second_checks(chosen, seed), order=order, seed=seed,
                           cores_asked=n, cores_remaining=len(left), priority={c: round(scores.get(c, 0.0), 2) for c in chosen})
    return sessions.update(root(), name, open=True) if body.open else meta


def _signals(core_id: str, stains: list[str], regions: list[str] | None = None) -> dict | None:
    """A core's problem signals over its stain images (a slide's over its regions: the worst of them)."""
    found = [s for s in (areas.signals(get_state().project, core, stain) for core in (regions or [core_id]) for stain in stains) if s]
    if not found:
        return None
    return {k: max(s[k] for s in found) if k != "tissue_share" else min(s[k] for s in found) for k in found[0]}


# ---------------------------------------------------------------- the round pages (annotators)

def practice_meta() -> dict:
    """The made-up session of the "Mark bad areas" tutorial: the core of the one tutorial of the server
    (`stainid.tutorial`, the same for every project), with its stains and credit line."""
    from stainid import tutorial  # noqa: PLC0415

    found = tutorial.info()
    if not found:
        raise HTTPException(404, "The tutorial is not set up on this server yet (stainid tutorial build).")
    core, stains = found["areas"]["core_id"], list(found["areas"]["stains"])
    return {"name": PRACTICE, "title": "Practice: " + areas.TITLE, "stain": "all", "kind": areas.KIND, "tools": ["area"], "ai_view": "full",
            "per_field": 1, "rule": "random", "fields": [], "source": areas.SOURCE, "cores": [core], "stains_of": {core: stains},
            "second": [], "order": "random", "instructions": "", "open": True, "created": 0.0, "practice": True, "credit": found.get("credit", "")}


def areas_meta(name: str, request: Request) -> dict:
    meta = practice_meta() if name == PRACTICE else session_meta(name, request)
    if not areas.is_areas(meta):
        raise HTTPException(404, f"{name} is not a {areas.TITLE} session")
    return meta


def core_meta(name: str, request: Request) -> dict:
    """A session whose unit is a whole core with its stain images: "Mark bad areas" or "Grey / white matter"
    (`routes.matter`), for the core pictures, tiles, thumbnails and detection's objects."""
    from stainid.labels import matter  # noqa: PLC0415

    meta = practice_meta() if name == PRACTICE else session_meta(name, request)
    if not (areas.is_areas(meta) or matter.is_matter(meta)):
        raise HTTPException(404, f"{name} is not a session of whole cores")
    return meta


def progress(meta: dict, labels: pd.DataFrame | None = None) -> dict:
    if meta.get("practice"):
        return areas.progress(meta, {}, {})
    return areas.progress(meta, areas.finished(meta, session_checked(meta, labels)), held(meta))


def detail(meta: dict, person: str | None) -> dict:
    """What the session page shows: the round, how far it is, and the cores this person did."""
    done = {} if meta.get("practice") else areas.finished(meta, session_checked(meta))
    mine = [c for c in meta.get("cores") or [] if person and person in done.get(c, set())]
    return {"round": areas.KIND, "progress": progress(meta), "done_by_me": len(mine)}


def _core(meta: dict, core_id: str) -> list[str]:
    stains = (meta.get("stains_of") or {}).get(core_id)
    if stains is None:
        raise HTTPException(404, f"{core_id} is not part of this session")
    return stains


def core_payload(meta: dict, core_id: str, person: str) -> dict:
    """One core of the session: its stain images (size, picture and layer versions, who checked each) and whether a
    second check is wanted. Opening it holds it for this person."""
    stains = _core(meta, core_id)
    state = get_state()
    found = {} if meta.get("practice") else session_checked(meta)
    if not meta.get("practice"):
        _HELD.hold(hold_key(meta), person, [core_id])
    images = []
    slide = slide_units.slide_unit(core_id)
    for stain in stains:
        if slide is not None:  # a whole slide: its picture and layers put together from its regions
            picture = slide_pyramid.picture(state, slide.slide_id)
            layer = slide_units.excluded_info(state.project, slide)
            edits = sum(len(exclusions.edits(state.project, r.core_id, stain, ai=False, session=meta["name"])) for r in slide.regions)
        else:
            try:
                picture = pyramid.picture(state, core_id, stain)
            except (KeyError, FileNotFoundError):
                picture = None
            layer = store_view.excluded_info(state.project, core_id, stain)
            edits = len(exclusions.edits(state.project, core_id, stain, ai=False, session=meta["name"]))
        by = sorted(found.get((core_id, stain), set()))
        images.append({"stain": stain, "picture": picture, "excluded": layer, "checked_by": by, "checked_by_me": person in by, "edits": edits,
                       "mask_scale": slide_units.mask_scale(slide) if slide is not None else exclusions.SCALE})
    others = sorted(held(meta).get(core_id, set()) - {person})
    return {"core_id": core_id, "images": images, "second_check": core_id in set(meta.get("second") or []), "open_by": others,
            "progress": progress(meta), "pixel_size_um": state.project.pixel_size_um, "unit": "slide" if slide is not None else "core",
            **({"name": slide.name, "regions": len(slide.regions)} if slide is not None else {})}


@router.get("/annotate/{name}/areas/next")
def next_core(name: str, request: Request, skip: str = "", person: str = Depends(annotator)) -> dict:
    """The next core for this person (None when there is none left for them) and the one after it (to prefetch)."""
    meta = areas_meta(name, request)
    if meta.get("practice"):
        return {"core_id": meta["cores"][0], "after": None, "taken": 0, "progress": progress(meta)}
    done = areas.finished(meta, session_checked(meta))
    now = held(meta)
    if skip:
        _HELD.release(hold_key(meta), person, [skip])
        now = held(meta)
    core, taken = areas.next_core(meta, done, person, now, skip)
    after = None
    if core:
        _HELD.hold(hold_key(meta), person, [core])
        after, _ = areas.next_core(meta, done, person, {**now, core: {"(someone)"}}, core)
    return {"core_id": core, "after": after, "taken": taken, "progress": progress(meta)}


@router.get("/annotate/{name}/cores/{core_id}")
def open_core(name: str, core_id: str, request: Request, person: str = Depends(annotator)) -> dict:
    return core_payload(areas_meta(name, request), core_id, person)


@router.post("/annotate/{name}/cores/{core_id}/hold")
def hold_core(name: str, core_id: str, request: Request, person: str = Depends(annotator)) -> dict:
    meta = areas_meta(name, request)
    _core(meta, core_id)
    if not meta.get("practice"):
        _HELD.hold(hold_key(meta), person, [core_id])
    return {"open_by": sorted(held(meta).get(core_id, set()) - {person})}


@router.get("/annotate/{name}/cores/{core_id}/{stain}/picture")
def core_picture(name: str, core_id: str, stain: str, request: Request, person: str = Depends(annotator)) -> dict:
    meta = core_meta(name, request)
    if stain not in _core(meta, core_id):
        raise HTTPException(404, f"{core_id} has no {stain} image in this session")
    slide = slide_units.slide_unit(core_id)
    try:
        return slide_pyramid.picture(get_state(), slide.slide_id) if slide is not None else pyramid.picture(get_state(), core_id, stain)
    except (KeyError, FileNotFoundError) as error:
        raise HTTPException(404, str(error).strip("'"))


@router.get("/annotate/{name}/cores/{core_id}/{stain}/tiles/{level}/{col}_{row}.jpg")
def core_tile(name: str, core_id: str, stain: str, level: int, col: int, row: int, request: Request, v: str = "",
              person: str = Depends(annotator)) -> Response:
    meta = core_meta(name, request)
    if stain not in _core(meta, core_id):
        raise HTTPException(404, f"{core_id} has no {stain} image in this session")
    slide = slide_units.slide_unit(core_id)
    try:
        data = slide_pyramid.tile(get_state(), slide.slide_id, level, col, row) if slide is not None else pyramid.tile(get_state(), core_id, stain, level, col, row)
    except KeyError as error:
        raise HTTPException(404, str(error).strip("'"))
    return Response(data, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=31536000, immutable" if v else "private, max-age=86400"})


@router.get("/annotate/{name}/cores/{core_id}/{stain}/thumb.jpg")
def core_thumb(name: str, core_id: str, stain: str, request: Request, size: int = 256, person: str = Depends(annotator)) -> FileResponse:
    meta = core_meta(name, request)
    if stain not in _core(meta, core_id):
        raise HTTPException(404, f"{core_id} has no {stain} image in this session")
    slide = slide_units.slide_unit(core_id)
    try:
        if slide is not None:
            return FileResponse(slide_pyramid.thumbnail(get_state(), slide.slide_id, size), headers={"Cache-Control": "private, max-age=600"})
        return FileResponse(render.core_thumbnail(get_state(), core_id, stain, max(32, min(size, pyramid.THUMB_BASE))),
                            headers={"Cache-Control": "private, max-age=86400"})
    except (KeyError, FileNotFoundError) as error:
        raise HTTPException(404, str(error))


@router.get("/annotate/{name}/cores/{core_id}/{stain}/excluded")
def core_excluded(name: str, core_id: str, stain: str, request: Request, person: str = Depends(annotator)) -> dict:
    meta = areas_meta(name, request)
    if stain not in _core(meta, core_id):
        raise HTTPException(404, f"{core_id} has no {stain} image in this session")
    slide = slide_units.slide_unit(core_id)
    if slide is not None:
        return slide_units.excluded_info(get_state().project, slide)
    return store_view.excluded_info(get_state().project, core_id, stain)


@router.get("/annotate/{name}/cores/{core_id}/{stain}/excluded.png")
def core_excluded_png(name: str, core_id: str, stain: str, request: Request, person: str = Depends(annotator)) -> Response:
    meta = areas_meta(name, request)
    if stain not in _core(meta, core_id):
        raise HTTPException(404, f"{core_id} has no {stain} image in this session")
    slide = slide_units.slide_unit(core_id)
    png = slide_units.excluded_png(get_state().project, slide) if slide is not None else store_view.excluded_png(get_state().project, core_id, stain)
    if png is None:
        raise HTTPException(404, f"the excluded areas of {core_id} {stain} are not made yet (the v2 proposals make them)")
    return Response(png, media_type="image/png", headers={"Cache-Control": "private, max-age=86400"})


@router.get("/annotate/{name}/cores/{core_id}/{stain}/annotated")
def core_annotated(name: str, core_id: str, stain: str, request: Request, person: str = Depends(annotator)) -> dict:
    """People's areas of a core image so far (`store_view.people_areas`, core px): what the brush edits."""
    meta = areas_meta(name, request)
    if stain not in _core(meta, core_id):
        raise HTTPException(404, f"{core_id} has no {stain} image in this session")
    slide = slide_units.slide_unit(core_id)
    if slide is not None:
        return slide_units.people_areas(get_state().project, slide, ai=False, session=meta["name"])
    return store_view.people_areas(get_state().project, core_id, stain, ai=False,
                                   session=meta["name"])  # this session's people: never the AI reviewer's


@router.get("/annotate/{name}/cores/{core_id}/{stain}/areas")
def core_areas(name: str, core_id: str, stain: str, request: Request, person: str = Depends(annotator)) -> dict:
    """Each brush type's current area over the core image (`store_view.current_areas`, core px): what the brush starts
    from, as on a field."""
    meta = areas_meta(name, request)
    if stain not in _core(meta, core_id):
        raise HTTPException(404, f"{core_id} has no {stain} image in this session")
    slide = slide_units.slide_unit(core_id)
    if slide is not None:
        return slide_units.current_areas(get_state().project, slide, session=meta["name"])
    return store_view.current_areas(get_state().project, core_id, stain, ai=False,
                                    session=meta["name"])  # the automatic layer and this session's edits only


@router.get("/annotate/{name}/cores/{core_id}/{stain}/objects")
def core_objects(name: str, core_id: str, stain: str, request: Request, person: str = Depends(annotator)) -> dict:
    """Detection's objects of a core image for context in "Mark bad areas" (they cannot be changed there): rows of
    [x, y, accepted, rings] in core pixels (the stain's default outline, `candidates.outline_rings`); `accepted`: a class,
    counted or a look-alike detection names (CAA: `candidate_store.lookalike`), drawn solid; a reject, or a reject
    reason such as a thread-like stain (`Vocabulary.rejecting`), is dotted."""
    meta = core_meta(name, request)
    if stain not in _core(meta, core_id):
        raise HTTPException(404, f"{core_id} has no {stain} image in this session")
    project = get_state().project
    slide = slide_units.slide_unit(core_id)
    if slide is not None:  # a whole slide: every region's, in slide px
        rows = [[x + r.x, y + r.y, accepted, [slide_units.flat_moved(ring, r) for ring in rings] if rings else rings]
                for r in slide.regions for x, y, accepted, rings in object_rows(project, r.core_id, stain)]
    else:
        rows = object_rows(project, core_id, stain)
    return {"columns": ["x", "y", "accepted", "rings"], "rows": rows}


_OBJECT_ROWS: OrderedDict[tuple, list[list]] = OrderedDict()  # (root, core, stain, the store's signatures) -> rows; a few cores kept
_OBJECT_ROWS_LOCK = threading.Lock()
OBJECT_ROWS_KEPT = 24


def object_rows(project, core_id: str, stain: str) -> list[list]:
    """Detection's objects of a core image as [x, y, accepted, rings] (core px; see `core_objects`), kept for the last
    few cores until their candidate store or the project's settings change (the Grey / white matter guide, made ahead
    in idle time by `ahead`, answers at once)."""
    from stainid import cache, candidates  # noqa: PLC0415
    from stainid.project import CONFIG_NAME  # noqa: PLC0415

    key = (str(project.root), core_id, stain, cache.signature(candidates.path(project, stain, core_id, "candidates")),
           cache.signature(candidates.path(project, stain, core_id, "proposals")), cache.signature(Path(project.root) / CONFIG_NAME))
    with _OBJECT_ROWS_LOCK:
        if key in _OBJECT_ROWS:
            _OBJECT_ROWS.move_to_end(key)
            return _OBJECT_ROWS[key]
    rows = _object_rows(project, core_id, stain)
    with _OBJECT_ROWS_LOCK:
        _OBJECT_ROWS[key] = rows
        while len(_OBJECT_ROWS) > OBJECT_ROWS_KEPT:
            _OBJECT_ROWS.popitem(last=False)
    return rows


def _object_rows(project, core_id: str, stain: str) -> list[list]:
    from stainid import candidates  # noqa: PLC0415
    from stainid.labels import candidate_store  # noqa: PLC0415
    from stainid.labels import vocabulary as vocab  # noqa: PLC0415

    frame = candidates.read_core(project, stain, core_id,
                                 ["x_core", "y_core", "decision", "reason", "class", "outline", "sam_outline", "sam_status", "outline_default"])
    words = vocab.of(project, stain)
    pairs = {(str(r.get("class") or ""), str(r.get("reason") or "")) for r in frame.to_dict("records")} if len(frame) else set()
    named = {pair for pair in pairs if candidate_store.lookalike(words, *pair) and not words.rejecting(candidate_store.lookalike(words, *pair))}
    return [[round(float(r["x_core"]), 1), round(float(r["y_core"]), 1),
             r.get("decision") == "accepted" or (str(r.get("class") or ""), str(r.get("reason") or "")) in named, candidates.outline_rings(r)]
            for r in frame.to_dict("records")] if len(frame) else []


class StainEdit(BaseModel):
    pngs: dict[str, str] = {}  # one painted mask per type (labels.exclusions.KINDS), at 1/`scale` of the core image
    scale: float = exclusions.SCALE


class CoreSave(BaseModel):
    edits: dict[str, StainEdit] = {}  # stain -> what was painted and cleared on it
    checked: list[str] = []  # the stain images looked at (every one of the core's, to save)


def save_core(meta: dict, core_id: str, body: CoreSave, person: str) -> dict:
    """Save a core: an edit per stain image painted or cleared, and a `done` label for every stain image (each must have
    been looked at); also with no edits, which confirms detection's areas."""
    stains = _core(meta, core_id)
    missing = [s for s in stains if s not in set(body.checked)]
    if missing:
        raise HTTPException(400, f"look at every stain image of {core_id} first: {', '.join(missing)} not yet")
    unknown = [s for s in body.edits if s not in stains]
    if unknown:
        raise HTTPException(400, f"{core_id} has no {', '.join(unknown)} image in this session")
    slide = slide_units.slide_unit(core_id)
    if slide is not None:
        records = _slide_records(meta, slide, body, person)
        return _saved(meta, core_id, records, person)
    records = []
    stamp = {"source": person, "session": meta["name"], "tile_id": core_id, "x": None, "y": None}
    for stain, edit in body.edits.items():
        try:
            rings, holes, kinds = marks.from_pngs(edit.pngs, edit.scale)
        except ValueError as error:
            raise HTTPException(400, str(error))
        if rings:
            records.append({**stamp, "id": marks.label_id(core_id), "stain": stain, "task": marks.TASK, "points": rings,
                            "value": marks.value_text(holes, kinds)})
    for stain in stains:
        records.append({**stamp, "id": areas.done_id(core_id, stain), "stain": stain, "task": "done", "value": areas.AREAS})
    return _saved(meta, core_id, records, person)


def _slide_records(meta: dict, slide, body: CoreSave, person: str) -> list[dict]:
    """A whole slide's save as its regions' labels: what was painted over the slide cut into each region's part (its
    rings in region px, one edit per region painted on) and a `done` label for every region, as a core's save."""
    records = []
    for stain, edit in body.edits.items():
        try:
            parts = slide_units.split_pngs(slide, edit.pngs, edit.scale)
        except ValueError as error:
            raise HTTPException(400, str(error))
        for core, (rings, holes, kinds) in parts.items():
            stamp = {"source": person, "session": meta["name"], "tile_id": core, "x": None, "y": None}
            records.append({**stamp, "id": marks.label_id(core), "stain": stain, "task": marks.TASK, "points": rings,
                            "value": marks.value_text(holes, kinds)})
    for r in slide.regions:
        stamp = {"source": person, "session": meta["name"], "tile_id": r.core_id, "x": None, "y": None}
        records.append({**stamp, "id": areas.done_id(r.core_id, slide.stain), "stain": slide.stain, "task": "done", "value": areas.AREAS})
    return records


def _saved(meta: dict, core_id: str, records: list[dict], person: str) -> dict:
    if meta.get("practice"):  # the tutorial: nothing is kept
        now = time.time()
        return {"saved": [{"confirmed": False, "deleted": False, "time": now, **r} for r in records], "progress": progress(meta)}
    writable(meta)
    saved = store.append(root(), records)
    _HELD.release(hold_key(meta), person, [core_id])
    return {"saved": saved, "progress": progress(meta)}


@router.post("/annotate/{name}/cores/{core_id}/save")
def save_route(name: str, core_id: str, body: CoreSave, request: Request, person: str = Depends(annotator)) -> dict:
    return save_core(areas_meta(name, request), core_id, body, person)


# ---------------------------------------------------------------- admin: the round's results and the Cores view

@router.get("/labels/areas", dependencies=admin)
def areas_overview() -> dict:
    """Where "Mark bad areas" stands: core images checked per stain (of those with analysed tiles) and people's
    agreement on the double-checked ones, with the disagreements to settle in the Cores view."""
    state = get_state()
    found = checked()
    stains_of = areas.core_stains(state.tiles)
    per_stain: dict[str, dict] = {}
    for core, stains in stains_of.items():
        for stain in stains:
            entry = per_stain.setdefault(stain, {"stain": stain, "images": 0, "checked": 0, "double_checked": 0})
            entry["images"] += 1
            people = found.get((core, stain), set())
            entry["checked"] += bool(people)
            entry["double_checked"] += len(people) >= 2
    return {"stains": list(per_stain.values()), "cores": len(stains_of),
            "cores_checked": sum(all(found.get((c, s)) for s in stains) for c, stains in stains_of.items()),
            "agreement": areas.agreement(state.project, found)}


@router.get("/models/bad-areas", dependencies=admin)
def bad_area_detector() -> dict:
    """The bad-area detector against people: detection's automatic areas (today's rules) against people's final say on
    the core images checked in "Mark bad areas", per stain, type and TMA, and per donor; `model` is the trained bad-area
    model's evaluation once there is one (none yet: the rules are the detector)."""
    state = get_state()
    found = checked()
    return {"images": len(found), "by_tma": areas.evaluation(state.project, found, state.tiles, "tma"),
            "by_donor": areas.evaluation(state.project, found, state.tiles, "donor_id"), "detector": "rules", "model": None}


@router.get("/cores/{core_id}/{stain}/status", dependencies=admin)
def core_status(core_id: str, stain: str) -> dict:
    """The minimap of a core image: each analysed tile (x, y, width, height in core px) with its object-review state
    (`labels.status`: reviewed, in progress, not annotated), the grid squares in no analysed tile ("not analysed"),
    the counts, and who checked the core image's bad areas."""
    from stainid.api.routes import rounds as round_routes  # noqa: PLC0415

    state = get_state()
    labels = store.current(root())
    tiles = state.tiles[(state.tiles.core_id.astype(str) == core_id) & (state.tiles.stain == stain)].drop_duplicates("tile_id")
    claims: dict[str, set[str]] = {}
    for name in _round_sessions(stain):  # holds are per session: who is on each tile in any session of the stain
        for tile, people in round_routes._TILES_HELD.held(round_routes.hold_key(state.id, name)).items():
            claims.setdefault(tile, set()).update(people)
    states = status.tile_states(labels, stain, list(tiles.tile_id), claims)
    rows = [{"tile_id": t.tile_id, "x": int(t.x_px), "y": int(t.y_px), "width": int(t.width_px), "height": int(t.height_px), **states[t.tile_id]}
            for t in tiles.itertuples()]
    size = _image_size(core_id, stain)
    grid = _missing_squares(rows, size) if size else []
    counts = status.counts(states)
    counts[status.NOT_ANALYSED] = len(grid)
    return {"core_id": core_id, "stain": stain, "width": size[0] if size else None, "height": size[1] if size else None, "tiles": rows,
            "not_analysed": grid, "counts": counts, "areas_checked_by": sorted(status.areas_checked(labels, stain).get((core_id, stain), set())),
            "sessions": _sessions_covering(stain, set(tiles.tile_id))}


def _round_sessions(stain: str) -> list[str]:
    from stainid.labels import rounds  # noqa: PLC0415

    return [name for name in sessions.names(root()) if rounds.is_round(meta := sessions.meta(root(), name)) and meta.get("stain") == stain]


def _image_size(core_id: str, stain: str) -> tuple[int, int] | None:
    from stainid.tables import read_csv  # noqa: PLC0415

    table = get_state().project.input("core_images")
    for row in read_csv(table) if table.exists() else []:
        if str(row.get("core_id")) == core_id and row.get("stain") == stain and row.get("native_width_px"):
            return int(float(row["native_width_px"])), int(float(row["native_height_px"]))
    return None


def _missing_squares(rows: list[dict], size: tuple[int, int], field: int = 2048) -> list[dict]:
    """The squares of the whole-core grid with no analysed tile."""
    from stainid.sampling.whole_core import grid  # noqa: PLC0415

    have = {(r["x"], r["y"]) for r in rows}
    return [{"x": x, "y": y, "width": w, "height": h} for y, h in grid(size[1], field) for x, w in grid(size[0], field) if (x, y) not in have]


def _sessions_covering(stain: str, tile_ids: set[str]) -> list[dict]:
    """The open object-round sessions with these tiles (to open a tile in annotation)."""
    from stainid.labels import rounds  # noqa: PLC0415

    out = []
    for name in sessions.names(root()):
        meta = sessions.meta(root(), name)
        if not (rounds.is_round(meta) and meta.get("stain") == stain and meta.get("open")):
            continue
        covered = set(meta.get("fields") or []) | set(meta.get("scope") or [])
        tiles = sorted(covered & tile_ids)
        if tiles:
            out.append({"name": name, "title": meta.get("title", name), "round": rounds.round_of(meta), "tiles": tiles})
    return out


@router.get("/cores/{core_id}/{stain}/annotations", dependencies=admin)
def core_annotations(core_id: str, stain: str) -> dict:
    """The Cores view's "Annotations" layer (`labels.compare`, the comparison scoring and the Labels overview use):
    every candidate people decided with detection's call and each person's answer, the objects people added that
    detection missed, the outlines people fixed (drawn only on hover; never instead of detection's), and the counts."""
    state = get_state()
    found = compare.versus_detection(state.project, stain, state.tiles, core_id)
    fixed = compare.fixed_outlines(state.project, stain, state.tiles, core_id)
    added = compare.added_objects(state.project, stain, state.tiles, core_id)
    outlines = {}
    for r in fixed.itertuples():
        outlines.setdefault(r.candidate_id, []).append({"person": r.person, "time": r.time, "points": r.points, "mask": r.mask, "option": r.option or None})
    objects = [{**r, "outline_fixed": outlines.get(r["candidate_id"], [])} for r in found.to_dict("records")]
    known = set(found.candidate_id) if len(found) else set()
    objects += [{"candidate_id": c, "core_id": core_id, "kind": "outline", "outline_fixed": v} for c, v in outlines.items() if c not in known]
    edits = compare.object_edits(state.project, stain, state.tiles, core_id)
    from stainid.api.routes import matter as matter_routes  # noqa: PLC0415

    return {"objects": objects, "added": added.to_dict("records"), "areas": store_view.people_areas(state.project, core_id, stain), "edits": edits,
            "matter": matter_routes.core_matter(core_id, stain),
            "counts": {**{k: int((found.kind == k).sum()) if len(found) else 0 for k in compare.KINDS}, "added": int(len(added)),
                       "outline_fixed": len(outlines), "split": sum(e["kind"] == "split" for e in edits),
                       "merge": sum(e["kind"] == "merge" for e in edits)}}


@router.get("/cores/{core_id}/{stain}/threshold-preview", dependencies=admin)
def threshold_preview(core_id: str, stain: str) -> dict:
    """What the Cores view's acceptance-threshold slider needs (a preview only: never saved, the analysis keeps
    detection's threshold): detection's threshold and, per candidate, its score (0 when it has none: it passes where
    p >= threshold), its call once it passes (`labels.starting.if_accepted`, label-class ids, with the rule after the
    model that decides it: empty where that rule rejects it, a speck; CAA where it names that class) and below it
    (`below`, `below_subtype`: the look-alike detection names once the model does not count it, an AT8 thread; empty for
    a reject); `fixed` where detection's call stays at every threshold: a rule decided it or it lies in an excluded area
    (a CAA deposit stays CAA)."""
    from stainid import candidates  # noqa: PLC0415
    from stainid.labels import candidate_store, starting  # noqa: PLC0415
    from stainid.labels import vocabulary as vocab  # noqa: PLC0415

    project = get_state().project
    frame = candidates.read_core(project, stain, core_id, ["candidate_id", "probability", "threshold", "reason"])
    if frame.empty:
        return {"detection": None, "columns": [], "rows": []}
    words = vocab.of(project, stain)
    passing = starting.if_accepted(project, stain, core_id)
    failing = starting.if_rejected(project, stain, core_id)
    reject = candidate_store.negative_class(words)
    thresholds = pd.to_numeric(frame.threshold, errors="coerce").dropna() if "threshold" in frame else pd.Series(dtype=float)
    rows = []
    for r in frame.itertuples(index=False):
        would = passing.get(r.candidate_id)
        identity, subtype = candidate_store.ai_call(words, *would) if would is not None else ("", "")
        identity, subtype = ("", "") if identity == reject else (identity, subtype)
        below = failing.get(r.candidate_id)
        below, below_subtype = candidate_store.ai_call(words, *below) if below is not None else ("", "")
        below, below_subtype = ("", "") if below == reject else (below, below_subtype)
        rule = would[3] if would is not None else (str(r.reason) if str(r.reason).startswith("rule:") else "")
        rows.append([r.candidate_id, starting.score(r.probability), identity, subtype, rule, would is None, below, below_subtype])
    return {"detection": float(thresholds.median()) if len(thresholds) else None,
            "columns": ["candidate_id", "probability", "identity", "subtype", "reason", "fixed", "below", "below_subtype"], "rows": rows}
