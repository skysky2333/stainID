"""Labels: the admin's overview and session management (/api/labels) and the public annotation pages (/api/annotate)."""
from __future__ import annotations

import os
import re
import socket
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

from stainid.api import render
from stainid.api.routes.images import OVERLAYS
from stainid.api.session import annotator, annotator_name, is_admin, require_admin
from stainid.api.state import get_state
from stainid.labels import overview, pool, sessions, store
from stainid.labels.tasks import CLASSES, CROP_UM, FIELD_TAGS, INSTRUCTIONS, NEGATIVE, STAINS, positive

router = APIRouter(tags=["labels"])
BUSY_SECONDS = 15 * 60
LAYERS = {"NeuN": ["dab", "exclusions"], "6E10": ["dab", "exclusions"], "AT8": ["dab", "exclusions", "threads"]}
_BUSY: dict[tuple[str, str], dict[str, float]] = {}
_BUSY_LOCK = threading.Lock()
admin = [Depends(require_admin)]


def root() -> Path:
    return get_state().project.output("reviews")


def busy(name: str, tile_id: str | None = None, person: str | None = None) -> dict[str, set[str]]:
    """Who has which field of this project's session open (seen in the last 15 minutes)."""
    now = time.time()
    session = (get_state().id, name)
    with _BUSY_LOCK:
        if tile_id and person:
            _BUSY.setdefault((session, tile_id), {})[person] = now
        return {tile: {p for p, t in people.items() if now - t < BUSY_SECONDS} for (s, tile), people in _BUSY.items() if s == session}


def viewer(request: Request) -> str | None:
    """A registered annotator, or the admin (who may look without registering)."""
    name = annotator_name(request)
    if name is None and not is_admin(request):
        raise HTTPException(401, "register with your name first")
    return name


def session_meta(name: str, request: Request) -> dict:
    if not re.fullmatch(r"[A-Za-z0-9_\-]+", name) or name not in sessions.names(root()):
        raise HTTPException(404, f"no labelling session {name}")
    meta = sessions.meta(root(), name)
    if not meta["open"] and not is_admin(request):
        raise HTTPException(403, "this labelling session is not open at the moment")
    return meta


def field_tile(meta: dict, tile_id: str) -> dict:
    if meta["kind"] == "fields" and tile_id not in meta["fields"]:
        raise HTTPException(404, f"{tile_id} is not part of this session")
    return get_state().tile(tile_id)


def field_objects(meta: dict, tile_id: str) -> pd.DataFrame:
    objects = pool.load_field(get_state().project, meta["stain"], tile_id)
    if objects is None or not objects.complete.astype(str).eq("True").all():
        raise HTTPException(409, "this field is still being prepared; try again in a minute")
    return objects


def labellers(meta: dict, labels: pd.DataFrame) -> list[dict]:
    """Everyone who has labelled in this session: how far they are and when they last saved something."""
    mine = labels[labels.session == meta["name"]]
    reviewed = store.reviewed_fields(labels, meta["stain"])
    out = []
    for person, group in mine.groupby("source"):
        classes = group.task == "class"
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
    progress = {"people": len(people), "labels": int((mine.task == "class").sum()), "labellers": people,
                "labelling_now": sum(time.time() - p["last_active"] < BUSY_SECONDS or p["person"] in now for p in people)}
    if meta["kind"] == "fields":
        reviewed = store.reviewed_fields(labels, meta["stain"])
        covered = sum(len(reviewed.get(t, set())) >= meta["per_field"] for t in meta["fields"])
        prepared = len(pool.prepared(get_state().project, meta["stain"]) & set(meta["fields"]))
        progress |= {"fields": len(meta["fields"]), "fields_covered": covered, "fields_prepared": prepared}
    return progress


def prepare_job(stain: str, fields: int = 0, session: str = "") -> None:
    get_state().jobs.submit("prepare_labels", {"stain": stain, "fields": fields, "session": session})


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
    n_fields: int = 20
    tile_ids: list[str] | None = None
    instructions: str = ""
    open: bool = False


class Opening(BaseModel):
    open: bool


class MoreFields(BaseModel):
    n: int = 10


class Prepare(BaseModel):
    stain: str
    fields: int = 20


@router.get("/labels/defaults", dependencies=admin)
def defaults() -> dict:
    return {"classes": CLASSES, "tags": FIELD_TAGS, "tools": list(sessions.FIELD_TOOLS), "ai_views": list(sessions.AI_VIEWS),
            "rules": sessions.RULES, "instructions": INSTRUCTIONS}


@router.get("/labels/overview", dependencies=admin)
def labels_overview() -> dict:
    state = get_state()
    labels = store.current(root())
    stains = [overview.cached_stain_overview(state.project, s, root(), state.tiles) for s in STAINS if s in set(state.tiles.stain)]
    listed = []
    for name in sessions.names(root()):
        meta = sessions.meta(root(), name)
        listed.append({**{k: v for k, v in meta.items() if k != "fields"}, **session_progress(meta, labels), "link": annotate_path(name)})
    return {"stains": stains, "sessions": listed}


@router.get("/labels/share", dependencies=admin)
def share(request: Request) -> dict:
    """Addresses at which other computers on the network reach this project's annotation pages."""
    port = request.url.port or 80
    host = socket.gethostname()
    addresses = sorted({a[4][0] for a in socket.getaddrinfo(host, None, socket.AF_INET)} - {"127.0.0.1"})
    links = [f"http://{a}:{port}{annotate_path()}" for a in [*addresses, host]]
    public = public_address(request)
    return {"links": ([f"{public}{annotate_path()}"] if public else []) + links,
            "network": bool(public) or os.environ.get("STAINID_HOST", "0.0.0.0") != "127.0.0.1"}


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
    taken = set(sessions.names(root())) | {p.name.rsplit("-", 2)[0] for p in (root() / sessions.FOLDER / ".deleted").glob("*")}
    return next(name for name in (base if i == 1 else f"{base}_{i}" for i in range(1, 10_000)) if name not in taken)


@router.post("/labels/sessions", dependencies=admin)
def create_session(body: NewSession) -> dict:
    if not body.name:
        body.name = free_name(body.title, body.stain, body.kind)
    if not re.fullmatch(r"[A-Za-z0-9_\-]+", body.name):
        raise HTTPException(400, "name may contain letters, digits, - and _ only")
    state = get_state()
    tiles = state.tiles[state.tiles.stain == body.stain]
    fields: list[str] = []
    if body.kind == "fields":
        if body.tile_ids:
            unknown = set(body.tile_ids) - set(tiles.tile_id)
            if unknown:
                raise HTTPException(400, f"unknown {body.stain} fields: {', '.join(sorted(unknown)[:5])}")
            fields = list(body.tile_ids)
        else:
            reviewed = set(store.reviewed_fields(store.current(root()), body.stain))
            taken = {t for n in sessions.names(root()) for t in sessions.meta(root(), n).get("fields", [])}
            fields = sessions.pick_fields(tiles, reviewed | taken, body.n_fields, len(sessions.names(root())))
        if not fields:
            raise HTTPException(400, f"no {body.stain} fields left to review")
    try:
        meta = sessions.create(root(), body.name, title=body.title or body.name, stain=body.stain, kind=body.kind, tools=body.tools,
                               ai_view=body.ai_view, per_field=max(1, body.per_field), rule=body.rule, fields=fields, instructions=body.instructions)
    except FileExistsError:
        raise HTTPException(409, "a session with this name exists")
    except ValueError as error:
        raise HTTPException(400, str(error))
    prepare_job(body.stain, 0 if fields else body.n_fields, body.name if fields else "")
    return sessions.update(root(), body.name, open=True) if body.open else meta


@router.post("/labels/sessions/{name}/open", dependencies=admin)
def set_open(name: str, body: Opening, request: Request) -> dict:
    session_meta(name, request)
    return sessions.update(root(), name, open=body.open)


@router.post("/labels/sessions/{name}/fields", dependencies=admin)
def add_fields(name: str, body: MoreFields, request: Request) -> dict:
    meta = session_meta(name, request)
    state = get_state()
    if meta["kind"] != "fields":
        prepare_job(meta["stain"], body.n)
        return meta
    reviewed = set(store.reviewed_fields(store.current(root()), meta["stain"]))
    taken = {t for n in sessions.names(root()) for t in sessions.meta(root(), n).get("fields", [])}
    more = sessions.pick_fields(state.tiles[state.tiles.stain == meta["stain"]], reviewed | taken, body.n, len(meta["fields"]))
    updated = sessions.update(root(), name, fields=meta["fields"] + more)
    prepare_job(meta["stain"], 0, name)
    return updated


@router.delete("/labels/sessions/{name}", dependencies=admin)
def delete_session(name: str, request: Request) -> dict:
    session_meta(name, request)
    return {"moved_to": str(sessions.remove(root(), name))}


@router.post("/labels/prepare", dependencies=admin)
def prepare(body: Prepare) -> dict:
    prepare_job(body.stain, body.fields)
    return {"ok": True}


@router.get("/labels/sessions/{name}/people", dependencies=admin)
def session_people(name: str, request: Request) -> list[dict]:
    return labellers(session_meta(name, request), store.current(root()))


@router.get("/labels/export.csv", dependencies=admin)
def export(session: str = "") -> Response:
    labels = store.current(root())
    if session:
        labels = labels[labels.session == session]
    tiles = get_state().tiles[["tile_id", "core_id", "tma", "disease_group", "region"]].drop_duplicates("tile_id")
    table = labels.drop(columns=["deleted"]).merge(tiles, on="tile_id", how="left")
    table["time"] = pd.to_datetime(table.time, unit="s").dt.strftime("%Y-%m-%dT%H:%M:%S")
    return Response(table.to_csv(index=False), media_type="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="labels{"_" + session if session else ""}.csv"'})


# ---------- annotators ----------

class Mark(BaseModel):
    task: str
    tile_id: str
    x: float | None = None
    y: float | None = None
    value: str = ""
    points: list[list[float]] | None = None
    deleted: bool = False


class Marks(BaseModel):
    records: list[Mark]


@router.get("/annotate")
def list_open(request: Request, person: str | None = Depends(viewer)) -> list[dict]:
    labels = store.current(root())
    out = []
    for name in sessions.names(root()):
        meta = sessions.meta(root(), name)
        if not meta["open"] and not is_admin(request):
            continue
        reviewed = store.reviewed_fields(labels, meta["stain"])
        done = sum(person in reviewed.get(t, set()) for t in meta["fields"]) if person else 0
        out.append({**{k: meta.get(k) for k in ("name", "title", "stain", "kind", "open", "instructions")}, "fields": len(meta["fields"]), "done_by_me": done,
                    "my_labels": int(((labels.session == name) & (labels.source == person) & (labels.task == "class")).sum()),
                    "link": annotate_path(name)})
    return out


@router.get("/annotate/{name}")
def session_detail(name: str, request: Request, person: str | None = Depends(viewer)) -> dict:
    meta = session_meta(name, request)
    labels = store.current(root())
    reviewed = store.reviewed_fields(labels, meta["stain"])
    prepared = pool.prepared(get_state().project, meta["stain"])
    open_now = busy(name)
    fields = [{"tile_id": t, "done_by_me": bool(person) and person in reviewed.get(t, set()), "annotators_done": len(reviewed.get(t, set())),
               "open_by_others": len(open_now.get(t, set()) - {person}), "prepared": t in prepared} for t in meta["fields"]]
    return {"meta": {**meta, "classes": CLASSES[meta["stain"]], "tags": FIELD_TAGS, "class_instructions": INSTRUCTIONS[meta["stain"]]},
            "fields": fields, "annotator": person}


@router.get("/annotate/{name}/next")
def next_item(name: str, request: Request, person: str = Depends(annotator)) -> dict:
    meta = session_meta(name, request)
    labels = store.current(root())
    if meta["kind"] == "fields":
        prepared = pool.prepared(get_state().project, meta["stain"])
        ready = {**meta, "fields": [t for t in meta["fields"] if t in prepared]}
        return {"tile_id": sessions.next_field(ready, store.reviewed_fields(labels, meta["stain"]), person, busy(name)),
                "waiting": len(meta["fields"]) - len(ready["fields"])}
    objects = pool.all_objects(get_state().project, meta["stain"])
    return {"object": sessions.next_object(meta, objects, labels, person)}


@router.get("/annotate/{name}/fields/{tile_id}")
def open_field(name: str, tile_id: str, request: Request, person: str = Depends(annotator)) -> dict:
    meta = session_meta(name, request)
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
    labels = store.current(root())
    mine = labels[(labels.source == person) & (labels.tile_id == tile_id) & (labels.stain == meta["stain"])]
    box = {"x": int(tile["x_px"]) - x0, "y": int(tile["y_px"]) - y0, "width": int(tile["width_px"]), "height": int(tile["height_px"])}
    return {"tile_id": tile_id, "box": box, "objects": shown, "outlines": outlines, "layers": LAYERS[meta["stain"]] if meta["ai_view"] == "full" else [],
            "mine": mine.replace({np.nan: None}).to_dict(orient="records")}


def _clean(meta: dict, mark: Mark, person: str) -> dict:
    stain, tools = meta["stain"], meta["tools"]
    allowed = {"class": "class" in tools or "point" in tools or meta["kind"] == "queue", "outline": "outline" in tools, "tag": "tag" in tools,
               "note": True, "done": meta["kind"] == "fields"}
    if not allowed.get(mark.task):
        raise HTTPException(400, f"{mark.task} labels are not part of this session")
    if not mark.deleted:
        if mark.task == "class" and mark.value not in CLASSES[stain]:
            raise HTTPException(400, f"unknown class {mark.value!r}")
        if mark.task == "tag" and mark.value not in FIELD_TAGS:
            raise HTTPException(400, f"unknown tag {mark.value!r}")
        if mark.task == "outline" and (not mark.points or len(mark.points) < 3 or any(len(p) != 2 for p in mark.points)):
            raise HTTPException(400, "an outline needs at least three [x, y] points")
    if mark.task in ("class", "outline") and (mark.x is None or mark.y is None):
        raise HTTPException(400, "object labels need x and y")
    ids = {"class": lambda: store.object_id("class", mark.tile_id, mark.x, mark.y), "outline": lambda: store.object_id("outline", mark.tile_id, mark.x, mark.y),
           "tag": lambda: f"tag:{mark.tile_id}:{mark.value}", "note": lambda: f"note:{mark.tile_id}", "done": lambda: f"done:{mark.tile_id}"}
    return {"id": ids[mark.task](), "source": person, "stain": stain, "task": mark.task, "tile_id": mark.tile_id, "x": mark.x, "y": mark.y,
            "value": mark.value, "points": mark.points, "session": meta["name"], "deleted": mark.deleted}


@router.post("/annotate/{name}/labels")
def save(name: str, body: Marks, request: Request, person: str = Depends(annotator)) -> list[dict]:
    meta = session_meta(name, request)
    for mark in body.records:
        if meta["kind"] == "fields":
            field_tile(meta, mark.tile_id)
            busy(name, mark.tile_id, person)
    return store.append(root(), [_clean(meta, mark, person) for mark in body.records])


@router.post("/annotate/{name}/fields/{tile_id}/finish")
def finish(name: str, tile_id: str, request: Request, person: str = Depends(annotator)) -> list[dict]:
    """Done with this field: every AI call (and AI outline of a positive object) the person left untouched counts as confirmed."""
    meta = session_meta(name, request)
    field_tile(meta, tile_id)
    objects = field_objects(meta, tile_id)
    labels = store.current(root())
    mine = labels[(labels.source == person) & (labels.tile_id == tile_id) & (labels.stain == meta["stain"])]
    records = []
    if meta["ai_view"] == "full":
        labelled = set(mine.id[mine.task == "class"])
        final = dict(zip(mine.id[mine.task == "class"], mine.value[mine.task == "class"]))
        for o in objects.itertuples():
            key = store.object_id("class", tile_id, o.x, o.y)
            if key not in labelled:
                records.append({"id": key, "source": person, "stain": meta["stain"], "task": "class", "tile_id": tile_id, "x": float(o.x), "y": float(o.y),
                                "value": o.ai_value, "session": name, "confirmed": True})
            final.setdefault(key, o.ai_value)
        if "outline" in meta["tools"]:
            edited = set(mine.id[mine.task == "outline"])
            for outline in open_field(name, tile_id, request, person)["outlines"]:
                key = store.object_id("outline", tile_id, outline["x"], outline["y"])
                if key not in edited and positive(meta["stain"], final.get(store.object_id("class", tile_id, outline["x"], outline["y"]), NEGATIVE[meta["stain"]])):
                    records.append({"id": key, "source": person, "stain": meta["stain"], "task": "outline", "tile_id": tile_id, "x": outline["x"],
                                    "y": outline["y"], "points": outline["points"], "session": name, "confirmed": True})
    records.append({"id": f"done:{tile_id}", "source": person, "stain": meta["stain"], "task": "done", "tile_id": tile_id, "session": name})
    return store.append(root(), records)


@router.get("/annotate/{name}/fields/{tile_id}/image.jpg")
def image(name: str, tile_id: str, request: Request, person: str | None = Depends(viewer)) -> FileResponse:
    field_tile(session_meta(name, request), tile_id)
    return FileResponse(render.field_jpeg(get_state(), tile_id), headers={"Cache-Control": "private, max-age=86400"})


@router.get("/annotate/{name}/fields/{tile_id}/layer/{layer}.png")
def layer(name: str, tile_id: str, layer: str, request: Request, person: str | None = Depends(viewer)) -> FileResponse:
    meta = session_meta(name, request)
    field_tile(meta, tile_id)
    if meta["ai_view"] != "full" or layer not in LAYERS[meta["stain"]]:
        raise HTTPException(404, f"layer {layer} is not shown in this session")
    return FileResponse(OVERLAYS[layer](get_state(), tile_id), headers={"Cache-Control": "private, max-age=86400"})


@router.get("/annotate/{name}/object.jpg")
def object_image(name: str, tile_id: str, x: float, y: float, request: Request, person: str | None = Depends(viewer)) -> Response:
    meta = session_meta(name, request)
    state = get_state()
    half = int(CROP_UM[meta["stain"]] / state.project.pixel_size_um / 2)
    crop = render.object_crop(state, meta["stain"], tile_id, x, y, half)
    ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(crop, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
    return Response(encoded.tobytes(), media_type="image/jpeg", headers={"Cache-Control": "private, max-age=86400"})
