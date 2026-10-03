from __future__ import annotations

import re
import threading
import time
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

from stainid.api import render
from stainid.api.routes.images import OVERLAYS
from stainid.api.session import admin_only, annotator, viewer
from stainid.api.state import get_state
from stainid.api.util import records
from stainid.outputs import objects
from stainid.review import annotations as store
from stainid.review.sampling import DEFAULT_LABELS

router = APIRouter(tags=["annotate"])
BUSY_SECONDS = 15 * 60
LAYERS = {"NeuN": ["dab", "exclusions"], "6E10": ["dab", "exclusions"], "AT8": ["dab", "exclusions", "threads"]}
DEFAULT_TAGS = ["mostly white matter", "poor image quality", "fold or tear", "unusable"]
_BUSY: dict[tuple[str, str], dict[str, float]] = {}
_BUSY_LOCK = threading.Lock()


def root() -> Path:
    return get_state().project.output("reviews")


def project(name: str) -> dict:
    if not re.fullmatch(r"[A-Za-z0-9_\-]+", name) or name not in store.names(root()):
        raise HTTPException(404, f"no annotation project {name}")
    return store.meta(root(), name)


def field(meta: dict, tile_id: str) -> dict:
    if tile_id not in meta["fields"]:
        raise HTTPException(404, f"{tile_id} is not part of this annotation project")
    return get_state().tile(tile_id)


def busy(name: str, tile_id: str | None = None, person: str | None = None) -> dict[str, set[str]]:
    """Who has which field open (seen in the last 15 minutes)."""
    now = time.time()
    with _BUSY_LOCK:
        if tile_id and person:
            _BUSY.setdefault((name, tile_id), {})[person] = now
        return {tile: {p for p, t in people.items() if now - t < BUSY_SECONDS} for (n, tile), people in _BUSY.items() if n == name}


def ai_objects(tile: dict) -> pd.DataFrame:
    state = get_state()
    found = objects(state.project, state.tiles[state.tiles.tile_id == tile["tile_id"]], tile["stain"])
    return found if not found.empty else pd.DataFrame(columns=["x", "y", "radius_px", "model_class", "probability", "area_um2"])


class NewProject(BaseModel):
    name: str
    title: str = ""
    stain: str
    n_fields: int = 20
    per_group: bool = True
    seed: int = 0
    tile_ids: list[str] | None = None
    labels: list[str] | None = None
    tags: list[str] | None = None
    tools: list[str] = ["object", "point", "polygon", "tag"]
    ai_view: str = "full"
    per_field: int = 1
    instructions: str = ""


class Mark(BaseModel):
    id: str
    type: str
    label: str = ""
    points: list[list[float]] = []
    note: str = ""
    deleted: bool = False


class Marks(BaseModel):
    records: list[Mark]


@router.get("/annotate")
def list_projects(person: str | None = Depends(viewer)) -> list[dict]:
    out = []
    for name in store.names(root()):
        meta = store.meta(root(), name)
        done = store.done_by(store.current(root(), name))
        out.append({**{k: meta[k] for k in ("name", "title", "stain", "tools", "ai_view", "per_field", "created")}, "fields": len(meta["fields"]),
                    "fields_covered": sum(len(done.get(t, set())) >= meta["per_field"] for t in meta["fields"]),
                    "done_by_me": sum(person in done.get(t, set()) for t in meta["fields"]) if person else 0})
    return out


@router.get("/annotate-defaults")
def defaults() -> dict:
    return {"labels": DEFAULT_LABELS, "tags": DEFAULT_TAGS, "tools": list(store.TOOLS), "ai_views": list(store.AI_VIEWS)}


@router.post("/annotate", dependencies=[Depends(admin_only)])
def create(request: NewProject) -> dict:
    if not re.fullmatch(r"[A-Za-z0-9_\-]+", request.name):
        raise HTTPException(400, "name may contain letters, digits, - and _ only")
    tiles = get_state().tiles
    tiles = tiles[tiles.stain == request.stain]
    if request.tile_ids:
        unknown = set(request.tile_ids) - set(tiles.tile_id)
        if unknown:
            raise HTTPException(400, f"unknown {request.stain} fields: {', '.join(sorted(unknown)[:5])}")
        chosen = list(request.tile_ids)
    else:
        if tiles.empty:
            raise HTTPException(400, f"no {request.stain} fields; select fields first")
        if request.per_group and "disease_group" in tiles and tiles.disease_group.notna().any():
            groups = tiles.dropna(subset=["disease_group"]).groupby("disease_group")
            take = max(1, request.n_fields // groups.ngroups)
            tiles = pd.concat([g.sample(min(len(g), take), random_state=request.seed) for _, g in groups])
        else:
            tiles = tiles.sample(min(len(tiles), request.n_fields), random_state=request.seed)
        chosen = list(tiles.sample(frac=1, random_state=request.seed).tile_id)
    try:
        meta = store.create(root(), request.name, request.title, request.stain, chosen, request.labels or DEFAULT_LABELS[request.stain],
                            request.tags if request.tags is not None else DEFAULT_TAGS, request.tools, request.ai_view, max(1, request.per_field),
                            request.instructions)
    except FileExistsError:
        raise HTTPException(409, "an annotation project with this name exists")
    except ValueError as error:
        raise HTTPException(400, str(error))
    return meta


@router.get("/annotate/{name}")
def detail(name: str, person: str | None = Depends(viewer)) -> dict:
    meta = project(name)
    done = store.done_by(store.current(root(), name))
    open_now = busy(name)
    fields = [{"tile_id": t, "done_by_me": bool(person) and person in done.get(t, set()), "annotators_done": len(done.get(t, set())),
               "open_by_others": len(open_now.get(t, set()) - {person})} for t in meta["fields"]]
    return {"meta": meta, "fields": fields, "annotator": person}


@router.get("/annotate/{name}/next")
def next_field(name: str, person: str = Depends(annotator)) -> dict:
    meta = project(name)
    return {"tile_id": store.next_field(meta, store.current(root(), name), person, busy(name))}


@router.get("/annotate/{name}/fields/{tile_id}")
def open_field(name: str, tile_id: str, person: str = Depends(annotator)) -> dict:
    meta = project(name)
    tile = field(meta, tile_id)
    busy(name, tile_id, person)
    state = get_state()
    x0, y0 = render.crop_origin(tile, state.project.context_px)
    found = ai_objects(tile) if meta["ai_view"] != "none" else pd.DataFrame()
    columns = ["x", "y", "radius_px"] + (["model_class", "probability", "area_um2"] if meta["ai_view"] == "full" else [])
    ai = {"objects": records(found[columns]) if not found.empty else [],
          "outlines": render.mask_outlines(state, tile_id) if meta["ai_view"] == "full" else [],
          "layers": LAYERS.get(tile["stain"], []) if meta["ai_view"] == "full" else []}
    mine = [r for r in store.current(root(), name) if r["annotator"] == person and r["tile_id"] == tile_id]
    box = {"x": int(tile["x_px"]) - x0, "y": int(tile["y_px"]) - y0, "width": int(tile["width_px"]), "height": int(tile["height_px"])}
    return {"tile_id": tile_id, "box": box, "pixel_size_um": state.project.pixel_size_um, "ai": ai, "mine": mine}


@router.post("/annotate/{name}/fields/{tile_id}")
def save(name: str, tile_id: str, request: Marks, person: str = Depends(annotator)) -> list[dict]:
    meta = project(name)
    tile = field(meta, tile_id)
    busy(name, tile_id, person)
    found = None
    clean = []
    for mark in request.records:
        record = {"id": mark.id[:80], "annotator": person, "tile_id": tile_id, "type": mark.type, "label": mark.label, "points": mark.points,
                  "note": mark.note[:2000], "deleted": mark.deleted}
        if mark.type not in store.TYPES or (mark.type in store.TOOLS and mark.type not in meta["tools"]):
            raise HTTPException(400, f"annotation type {mark.type} is not enabled here")
        if not mark.deleted:
            allowed = meta["tags"] if mark.type == "tag" else meta["labels"] if mark.type in ("object", "point", "polygon") else None
            if allowed is not None and mark.label not in allowed:
                raise HTTPException(400, f"unknown label {mark.label!r}")
            if any(len(p) != 2 for p in mark.points) or (mark.type == "point" and len(mark.points) != 1) or (mark.type == "polygon" and len(mark.points) < 3):
                raise HTTPException(400, "a point needs one [x, y]; an outline at least three")
        if mark.type == "object" and not mark.deleted:
            found = ai_objects(tile) if found is None else found
            match = found[(found.x.round() == round(mark.points[0][0])) & (found.y.round() == round(mark.points[0][1]))] if mark.points else found.iloc[:0]
            if match.empty:
                raise HTTPException(400, "no AI object at this position")
            hit = match.iloc[0]
            record["ai"] = {"x": float(hit.x), "y": float(hit.y), "model_class": str(hit.model_class),
                            "probability": None if pd.isna(hit.probability) else float(hit.probability)}
        clean.append(record)
    return store.append(root(), name, clean)


@router.get("/annotate/{name}/fields/{tile_id}/image.jpg")
def image(name: str, tile_id: str, person: str | None = Depends(viewer)) -> FileResponse:
    field(project(name), tile_id)
    return FileResponse(render.field_jpeg(get_state(), tile_id), headers={"Cache-Control": "private, max-age=86400"})


@router.get("/annotate/{name}/fields/{tile_id}/layer/{layer}.png")
def layer(name: str, tile_id: str, layer: str, person: str | None = Depends(viewer)) -> FileResponse:
    meta = project(name)
    tile = field(meta, tile_id)
    if meta["ai_view"] != "full" or layer not in LAYERS.get(tile["stain"], []):
        raise HTTPException(404, f"layer {layer} is not shown in this project")
    return FileResponse(OVERLAYS[layer](get_state(), tile_id), headers={"Cache-Control": "private, max-age=86400"})


@router.get("/annotate/{name}/summary", dependencies=[Depends(admin_only)])
def summary(name: str) -> dict:
    return store.summary(project(name), store.current(root(), name))


@router.get("/annotate/{name}/export.csv", dependencies=[Depends(admin_only)])
def export(name: str) -> Response:
    project(name)
    table = store.export(store.current(root(), name), get_state().tiles)
    return Response(table.to_csv(index=False), media_type="text/csv", headers={"Content-Disposition": f'attachment; filename="{name}_annotations.csv"'})
