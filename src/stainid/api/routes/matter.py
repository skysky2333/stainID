"""The "Grey / white matter" round (`stainid.labels.matter`): people paint the white matter of each core, first on its
NeuN image, then on its other stain images, which start from the NeuN area carried over through the registration
(`stainid.registration.cores`).

Round pages (annotators): the session's cores by TMA and core with where each stands (the picker), the next core for a
person, one core's stain images, a stain image's area to start from (this person's saved one; else, after the NeuN
image is saved, the NeuN area carried over), and saving a stain image (its whole white-matter area and a `done` label;
"accept as carried" saves the carried area exactly). One person on a core at a time: their page's heartbeat holds it
(`_HELD`) and anyone else opening it is refused (409) until they close it. A core this person finished opens to look
at (`edit` to work on it again). The pictures, tiles, thumbnails and detection's objects (the NeuN neurons as a guide)
are the core routes of `routes.areas`.

Each core is done by one person. The tutorial practises on `practice-matter` (`stainid.tutorial.matter`: a public
NeuN image and the same tissue on another stain): its saves are checked and echoed, never kept, and the page sends the
NeuN area it painted to be carried (`…/carry`), as nothing of it is on the server.

Admin: where the round stands (stain images saved per stain). Like every human annotation the areas only train and
evaluate models; the analysis never reads them.
"""
from __future__ import annotations

import secrets
import time

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from stainid.api.routes.annotate import NewSession, free_name, root, session_meta, writable
from stainid.api.session import annotator, require_admin
from stainid.api.state import get_state
from stainid.labels import areas, matter, sessions, store
from stainid.labels import exclusions as marks
from stainid.registration import cores as registration

router = APIRouter(tags=["matter"])
admin = [Depends(require_admin)]
_HELD = sessions.Reservations(matter.HOLD_SECONDS)


def hold_key(meta: dict) -> tuple:
    return get_state().id, "matter", meta["name"]


def held(meta: dict) -> dict[str, set[str]]:
    return _HELD.held(hold_key(meta))


def session_labels(meta: dict) -> pd.DataFrame:
    return store.current(root(), session=meta["name"])


def session_checked(meta: dict, labels: pd.DataFrame | None = None) -> dict[tuple[str, str], set[str]]:
    return matter.checked(session_labels(meta) if labels is None else labels)


def working_text(people: set[str]) -> str:
    names = sorted(people)
    return f"{', '.join(names[:-1])} and {names[-1]} are working on this core" if len(names) > 1 else f"{names[0]} is working on this core"


# ---------------------------------------------------------------- making a session (admin)

def create(body: NewSession) -> dict:
    """A "Grey / white matter" session: the cores (each with all its stain images, NeuN first) of the chosen TMAs,
    `n_fields` of them (-1 or empty: all), in random order. Sessions are independent: cores done or held in another
    session are drawn as any other."""
    tiles = get_state().tiles
    if body.tmas and not tiles.empty:
        tiles = tiles[tiles.tma.astype(str).isin({str(t) for t in body.tmas})]
    stains_of = {c: matter.ordered(s) for c, s in areas.core_stains(tiles).items()}
    if not stains_of:
        raise HTTPException(400, "No core has analysed tiles" + (" on these TMAs" if body.tmas else ""))
    left = list(stains_of)
    seed = body.seed if body.seed is not None else secrets.randbelow(2**31)
    ordered = areas.order(left, {}, "random", seed)
    n = -1 if body.n_fields is None else body.n_fields
    chosen = ordered if n < 0 else ordered[:n]
    name = body.name or free_name(body.title or "grey white matter", "all", matter.KIND)
    meta = sessions.create(root(), name, title=body.title or name, stain="all", kind=matter.KIND, tools=["area"], ai_view="full", per_field=1,
                           rule="random", fields=[], instructions=body.instructions, source=matter.SOURCE, cores=chosen,
                           stains_of={c: stains_of[c] for c in chosen}, second=[],
                           order="random", seed=seed, cores_asked=n, cores_remaining=len(left))
    return sessions.update(root(), name, open=True) if body.open else meta


# ---------------------------------------------------------------- the round pages (annotators)

def practice_meta() -> dict:
    """The made-up session of the "Grey / white matter" tutorial: the tutorial's NeuN core and its other stain."""
    from stainid import tutorial  # noqa: PLC0415

    found = tutorial.matter()
    if not found:
        raise HTTPException(404, "The Grey / white matter practice is not set up on this server yet (stainid tutorial matter).")
    core, stains = found["core_id"], matter.ordered(found["stains"])
    return {"name": matter.PRACTICE, "title": "Practice: " + matter.TITLE, "stain": "all", "kind": matter.KIND, "tools": ["area"], "ai_view": "full",
            "per_field": 1, "rule": "random", "fields": [], "source": matter.SOURCE, "cores": [core], "stains_of": {core: stains}, "second": [],
            "order": "random", "instructions": "", "open": True, "created": 0.0, "practice": True, "credit": found.get("credit", "")}


def matter_meta(name: str, request: Request) -> dict:
    meta = practice_meta() if name == matter.PRACTICE else session_meta(name, request)
    if not matter.is_matter(meta):
        raise HTTPException(404, f"{name} is not a {matter.TITLE} session")
    return meta


def progress(meta: dict, labels: pd.DataFrame | None = None) -> dict:
    if meta.get("practice"):
        return areas.progress(meta, {}, {})
    return areas.progress(meta, areas.finished(meta, session_checked(meta, labels)), held(meta))


def _places() -> dict[str, dict]:
    images = get_state().core_images
    if images.empty:
        return {}
    first = images.drop_duplicates("core_id")
    return {str(r["core_id"]): {"tma": str(r.get("tma", "") or ""), "core": str(r.get("core_label", "") or r["core_id"])} for r in first.to_dict("records")}


def core_rows(meta: dict, person: str | None, labels: pd.DataFrame | None = None) -> list[dict]:
    """The picker: each core of the session with its TMA and label, where it stands for everyone (to do, in progress,
    done), who finished it, who has it open and the stain images this person saved."""
    labels = session_labels(meta) if labels is None else labels
    found = {} if meta.get("practice") else session_checked(meta, labels)
    done = areas.finished(meta, found)
    now = {} if meta.get("practice") else held(meta)
    places = _places()
    out = []
    for core in meta.get("cores") or []:
        stains = (meta.get("stains_of") or {}).get(core, [])
        by = done.get(core, set())
        started = any(found.get((core, s)) for s in stains)
        state = "done" if len(by) >= areas.needed(meta, core) else "in_progress" if now.get(core) or started else "not_started"
        out.append({"core_id": core, **places.get(core, {"tma": "", "core": core}), "stains": stains, "state": state, "done_by": sorted(by),
                    "open_by": sorted(now.get(core, set())), "done_by_me": bool(person) and person in by,
                    "saved_by_me": [s for s in stains if person and person in found.get((core, s), set())]})
    return out


def detail(meta: dict, person: str | None) -> dict:
    labels = store.current(root()).iloc[:0] if meta.get("practice") else session_labels(meta)
    rows = core_rows(meta, person, labels)
    return {"round": matter.KIND, "progress": progress(meta, labels), "done_by_me": sum(r["done_by_me"] for r in rows), "core_rows": rows}


def _core(meta: dict, core_id: str) -> list[str]:
    stains = (meta.get("stains_of") or {}).get(core_id)
    if stains is None:
        raise HTTPException(404, f"{core_id} is not part of this session")
    return stains


def _stain(meta: dict, core_id: str, stain: str) -> list[str]:
    stains = _core(meta, core_id)
    if stain not in stains:
        raise HTTPException(404, f"{core_id} has no {stain} image in this session")
    return stains


def refuse_if_taken(meta: dict, core_id: str, person: str) -> None:
    if meta.get("practice"):
        return
    others = held(meta).get(core_id, set()) - {person}
    if others:
        raise HTTPException(409, working_text(others))


@router.get("/annotate/{name}/matter/next")
def next_core(name: str, request: Request, skip: str = "", person: str = Depends(annotator)) -> dict:
    """The next core for this person (None when none is left for them)."""
    meta = matter_meta(name, request)
    if meta.get("practice"):
        return {"core_id": meta["cores"][0], "taken": 0, "progress": progress(meta)}
    done = areas.finished(meta, session_checked(meta))
    now = held(meta)
    if skip:
        _HELD.release(hold_key(meta), person, [skip])
        now = held(meta)
    core, taken = areas.next_core(meta, done, person, now, skip)
    return {"core_id": core, "taken": taken, "progress": progress(meta)}


@router.get("/annotate/{name}/matter/cores/{core_id}")
def open_core(name: str, core_id: str, request: Request, edit: bool = False, person: str = Depends(annotator)) -> dict:
    """One core: its stain images (NeuN first; the one the others start from is `source`), which this person saved, and
    whether it opens to look at (`read_only`: finished by this person, or by the people it needs, unless `edit`).
    Opening it to work on holds it for this person; someone else on it → 409."""
    from stainid.api import pyramid  # noqa: PLC0415

    meta = matter_meta(name, request)
    stains = _core(meta, core_id)
    practice = bool(meta.get("practice"))
    labels = store.current(root()).iloc[:0] if practice else session_labels(meta)
    found = {} if practice else session_checked(meta, labels)
    by = areas.finished(meta, found).get(core_id, set())
    looking = (person in by or len(by) >= areas.needed(meta, core_id)) and not edit
    if not looking and not practice:
        refuse_if_taken(meta, core_id, person)
        _HELD.hold(hold_key(meta), person, [core_id])
    state = get_state()
    images = []
    for stain in stains:
        try:
            picture = pyramid.picture(state, core_id, stain)
        except (KeyError, FileNotFoundError):
            picture = None
        mine = matter.latest(labels, core_id, stain, person)
        images.append({"stain": stain, "picture": picture, "saved_by": sorted(found.get((core_id, stain), set())),
                       "saved_by_me": person in found.get((core_id, stain), set()), "mine": mine.to_api() if mine else None})
    place = _places().get(core_id, {"tma": "", "core": core_id})
    return {"core_id": core_id, **place, "images": images, "source": matter.source_of(stains), "read_only": looking, "done_by": sorted(by),
            "progress": progress(meta, labels), "pixel_size_um": state.project.pixel_size_um, "practice": practice}


@router.post("/annotate/{name}/matter/cores/{core_id}/hold")
def hold_core(name: str, core_id: str, request: Request, person: str = Depends(annotator)) -> dict:
    meta = matter_meta(name, request)
    _core(meta, core_id)
    if meta.get("practice"):
        return {"held": False}
    refuse_if_taken(meta, core_id, person)
    _HELD.hold(hold_key(meta), person, [core_id])
    return {"held": True}


@router.post("/annotate/{name}/matter/cores/{core_id}/leave")
def leave_core(name: str, core_id: str, request: Request, person: str = Depends(annotator)) -> dict:
    """The person closed the core: free for the others at once."""
    meta = matter_meta(name, request)
    if not meta.get("practice"):
        _HELD.release(hold_key(meta), person, [core_id])
    return {"left": True}


def _size(core_id: str, stain: str) -> tuple[int, int]:
    try:
        row = get_state().core_image(core_id, stain)
    except KeyError as error:
        raise HTTPException(404, f"no core image for {core_id} {stain}") from error
    return int(float(row["native_width_px"])), int(float(row["native_height_px"]))


def _carried(meta: dict, labels: pd.DataFrame, core_id: str, stain: str, source: str, person: str | None):
    """The source stain's area (this person's, else anyone's when `person` is None) in `stain`'s pixels, with the
    carry; None before the source is saved."""
    area = matter.latest(labels, core_id, source, person)
    if area is None:
        return None
    try:
        carry = registration.carry(get_state(), core_id, source, stain)
    except (KeyError, FileNotFoundError, ValueError) as error:
        raise HTTPException(409, f"the {source} and {stain} images of {core_id} could not be lined up: {error}") from error
    return area, carry.apply(area.rings), carry


@router.get("/annotate/{name}/matter/cores/{core_id}/{stain}/area")
def stain_area(name: str, core_id: str, stain: str, request: Request, look: bool = False, person: str = Depends(annotator)) -> dict:
    """What the brush starts from on one stain image (the brush's form: a white-on-transparent PNG at 1/`scale`):
    `from` "saved" (this person's saved area; `look`: else anyone's in the session), "carried" (the source stain's
    saved area carried over, with the registration's status), "new" (the source stain, nothing saved yet: no white
    matter) or "waiting" (the source stain is not saved yet)."""
    meta = matter_meta(name, request)
    stains = _stain(meta, core_id, stain)
    labels = store.current(root()).iloc[:0] if meta.get("practice") else session_labels(meta)
    width, height, base = _base(core_id, stain)
    mine = matter.latest(labels, core_id, stain, person) or (matter.latest(labels, core_id, stain) if look else None)
    if mine is not None:
        return {**base, "kinds": {matter.AREA: matter.png(mine.rings, mine.holes, width, height)}, "from": "saved", "saved": mine.to_api()}
    source = matter.source_of(stains)
    if stain == source:
        return {**base, "kinds": {}, "from": "new"}
    found = _carried(meta, labels, core_id, stain, source, None if look else person)
    if found is None:
        return {**base, "kinds": {}, "from": "waiting", "carried_from": source}
    area, rings, carry = found
    return {**base, "kinds": {matter.AREA: matter.png(rings, area.holes, width, height)} if rings else {}, "from": "carried",
            "carried_from": source, "registration": carry.to_api()}


def _base(core_id: str, stain: str) -> tuple[int, int, dict]:
    """A stain image's size and the brush's frame for it (one mask px per `matter.SCALE` core px)."""
    width, height = _size(core_id, stain)
    scale = matter.SCALE
    shape = (-(-height // scale), -(-width // scale))
    return width, height, {"status": "done", "scale": scale, "x": 0, "y": 0, "width": shape[1] * scale, "height": shape[0] * scale}


class Painted(BaseModel):
    png: str = ""  # the source stain's white matter as painted (a PNG at 1/`scale` of its core image); empty: none
    scale: float = matter.SCALE


@router.post("/annotate/{name}/matter/cores/{core_id}/{stain}/carry")
def carry_area(name: str, core_id: str, stain: str, body: Painted, request: Request, person: str = Depends(annotator)) -> dict:
    """A white-matter area painted on the source stain, carried to `stain` (the brush's form, `from` "carried"): what
    the tutorial's page asks for, as its NeuN area is kept nowhere. Nothing is written."""
    meta = matter_meta(name, request)
    stains = _stain(meta, core_id, stain)
    source = matter.source_of(stains)
    try:
        rings, holes = marks.from_png(body.png, body.scale) if body.png else ([], [])
    except ValueError as error:
        raise HTTPException(400, str(error)) from error
    try:
        carry = registration.carry(get_state(), core_id, source, stain)
    except (KeyError, FileNotFoundError, ValueError) as error:
        raise HTTPException(409, f"the {source} and {stain} images of {core_id} could not be lined up: {error}") from error
    width, height, base = _base(core_id, stain)
    moved = carry.apply(rings)
    return {**base, "kinds": {matter.AREA: matter.png(moved, holes, width, height)} if moved else {}, "from": "carried", "carried_from": source,
            "registration": carry.to_api()}


class MatterSave(BaseModel):
    png: str = ""  # the whole white-matter area as painted (a PNG at 1/`scale` of the core image); empty: none
    scale: float = matter.SCALE
    accepted: bool = False  # "accept as carried": the source stain's area carried over, exactly (`png` is not read)


@router.post("/annotate/{name}/matter/cores/{core_id}/{stain}/save")
def save_stain(name: str, core_id: str, stain: str, body: MatterSave, request: Request, person: str = Depends(annotator)) -> dict:
    """Save one stain image's white matter: its whole area (or, `accepted`, the source stain's carried over) and its
    `done` label. The source stain (NeuN) first: the others start from it. `finished`: every stain image of the core
    is saved by this person now (the core is let go)."""
    meta = matter_meta(name, request)
    stains = _stain(meta, core_id, stain)
    source = matter.source_of(stains)
    if meta.get("practice"):  # the tutorial: checked, echoed, nothing kept
        try:
            rings, holes = marks.from_png(body.png, body.scale) if body.png and not body.accepted else ([], [])
        except ValueError as error:
            raise HTTPException(400, str(error)) from error
        stamp = {"source": person, "session": meta["name"], "tile_id": core_id, "x": None, "y": None, "time": time.time(), "confirmed": False, "deleted": False}
        return {"saved": matter.records(core_id, stain, rings, holes, stamp, None if stain == source else source, body.accepted, None),
                "finished": False, "progress": progress(meta)}
    writable(meta)
    refuse_if_taken(meta, core_id, person)
    labels = session_labels(meta)
    carried_from, status = None, None
    if stain != source:
        found = _carried(meta, labels, core_id, stain, source, person)
        if found is None:
            raise HTTPException(400, f"save the {source} image first: the other stains start from its white matter")
        area, carried, carry = found
        carried_from, status = source, carry.status
    if body.accepted:
        if stain == source:
            raise HTTPException(400, f"{source} is where the white matter is painted first: there is nothing to accept")
        rings, holes = carried, list(area.holes)
    else:
        try:
            rings, holes = marks.from_png(body.png, body.scale) if body.png else ([], [])
        except ValueError as error:
            raise HTTPException(400, str(error)) from error
    stamp = {"source": person, "session": meta["name"], "tile_id": core_id, "x": None, "y": None}
    saved = store.append(root(), matter.records(core_id, stain, rings, holes, stamp, carried_from, body.accepted, status))
    now = session_checked(meta)
    finished = all(person in now.get((core_id, s), set()) for s in stains)
    if finished:
        _HELD.release(hold_key(meta), person, [core_id])
    else:
        _HELD.hold(hold_key(meta), person, [core_id])
    return {"saved": saved, "finished": finished, "progress": progress(meta)}


# ---------------------------------------------------------------- admin

@router.get("/labels/matter", dependencies=admin)
def matter_overview() -> dict:
    """Where "Grey / white matter" stands: stain images saved per stain (of those with analysed tiles) and the cores
    with every stain image saved."""
    state = get_state()
    labels = store.current(root())
    found = matter.checked(labels)
    stains_of = areas.core_stains(state.tiles)
    per_stain: dict[str, dict] = {}
    for core, stains in stains_of.items():
        for stain in stains:
            entry = per_stain.setdefault(stain, {"stain": stain, "images": 0, "saved": 0})
            entry["images"] += 1
            entry["saved"] += bool(found.get((core, stain)))
    return {"stains": [per_stain[s] for s in matter.ordered(per_stain)], "cores": len(stains_of),
            "cores_done": sum(all(found.get((c, s)) for s in stains) for c, stains in stains_of.items())}


def core_matter(core_id: str, stain: str) -> dict | None:
    """The Cores view: a core image's white matter as it stands (the latest anyone saved), as regions in core px."""
    from stainid import exclusions  # noqa: PLC0415

    project = get_state().project
    area = matter.final(store.current(root()), core_id, stain)
    if area is None:
        return None
    return {**area.to_api(), "regions": matter.regions(area, exclusions._grid_shape(project, core_id, stain))}
