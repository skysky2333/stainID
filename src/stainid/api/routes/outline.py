"""The segmentation check: outline sessions ("Outline every object", `labels.outline`) for the people outlining, and
for the admin their set-up and the scores (`segmentation.report`).

People outlining (`/api/annotate/<session>/outline/<field>`): their own objects in a field; click-to-outline with
Segment Anything (`masks.clicks`, from features made ahead by `api.sam_embeddings`); saving, editing and deleting
objects; marking the field complete; and what stainID detection found there (shown from the start; people may hide it).
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from stainid import cache
from stainid.api import render, sam_embeddings
from stainid.api.outlines import label_rings
from stainid.api.routes.annotate import NewSession, _known_fields, annotate_path, busy, field_tile, free_name, root, session_meta, viewer, writable
from stainid.api.session import annotator, require_admin
from stainid.api.state import get_state
from stainid.labels import outline, readiness, sessions, store
from stainid.labels import vocabulary as vocab
from stainid.labels.tasks import STAINS
from stainid.segmentation import proposals, report

router = APIRouter(tags=["segmentation check"])
admin = [Depends(require_admin)]
REFERENCE_SIMPLIFY_PX = 1.5


def _outline_meta(name: str, request: Request, write: bool = False) -> dict:
    meta = writable(session_meta(name, request)) if write else session_meta(name, request)
    if meta["kind"] != outline.KIND:
        raise HTTPException(404, f"{name} is not an outline session")
    return meta


def _ref(record_id: str) -> str:
    return record_id.rsplit(":", 1)[-1]


def _mine(meta: dict, tile_id: str, person: str) -> list[dict]:
    rows = outline.objects(store.current(root()), meta["name"], tile_id, person)
    return [{"ref": _ref(r.id), "value": r.value, "points": r.points, "x": float(r.x), "y": float(r.y)} for r in rows.itertuples()]


# ---------- people outlining ----------

class OutlineObject(BaseModel):
    ref: str
    value: str = ""
    points: list[list[float]] | None = None
    deleted: bool = False


class OutlineObjects(BaseModel):
    records: list[OutlineObject]


class Clicks(BaseModel):
    points: list[list[float]]  # [x, y, 1 on the object | 0 not on it], the first on it


class Completion(BaseModel):
    complete: bool


@router.get("/annotate/{name}/outline/{tile_id}")
def open_field(name: str, tile_id: str, request: Request, person: str = Depends(annotator)) -> dict:
    """A field to outline: its square in the picture, this person's objects, and whether they marked it complete."""
    meta = _outline_meta(name, request)
    tile = field_tile(meta, tile_id)
    busy(name, tile_id, person)
    state = get_state()
    x0, y0 = render.crop_origin(tile, state.project.context_px)
    box = {"x": int(tile["x_px"]) - x0, "y": int(tile["y_px"]) - y0, "width": int(tile["width_px"]), "height": int(tile["height_px"])}
    finished = outline.finished(store.current(root()), name)
    complete = person in finished.get(tile_id, set())
    ready = sam_embeddings.prepare_field(state, meta, tile_id, {t for t, people in finished.items() if person in people})
    return {"tile_id": tile_id, "box": box, "objects": _mine(meta, tile_id, person), "complete": complete,
            "segment_anything": state.project.model("sam").exists(), "segment_ready": ready["ready"], "pixel_size_um": state.project.pixel_size_um}


@router.get("/annotate/{name}/outline/{tile_id}/segment")
def segment_status(name: str, tile_id: str, request: Request, person: str = Depends(annotator)) -> dict:
    """Whether clicks on this field outline at once (`ready`), or Segment Anything is still preparing it (asked first)."""
    meta = _outline_meta(name, request)
    field_tile(meta, tile_id)
    return sam_embeddings.status(get_state(), tile_id)


@router.post("/annotate/{name}/outline/{tile_id}/segment")
def segment(name: str, tile_id: str, body: Clicks, request: Request, person: str = Depends(annotator)) -> dict:
    """Outlines Segment Anything proposes for the clicked object (see `masks.clicks.ClickOutliner.outline`)."""
    meta = _outline_meta(name, request)
    field_tile(meta, tile_id)
    if not body.points or any(len(p) != 3 for p in body.points):
        raise HTTPException(400, "send at least one click as [x, y, 1]")
    state = get_state()
    checkpoint = state.project.model("sam")
    if not checkpoint.exists():
        raise HTTPException(409, "The Segment Anything weights are not downloaded: draw the outline by hand, or ask the person running stainID "
                                 "to download them on the Models page.")
    return sam_embeddings.outline(state, tile_id, meta["stain"], [(float(x), float(y), bool(on)) for x, y, on in body.points])


@router.post("/annotate/{name}/outline/{tile_id}/objects")
def save_objects(name: str, tile_id: str, body: OutlineObjects, request: Request, person: str = Depends(annotator)) -> list[dict]:
    meta = _outline_meta(name, request, write=True)
    field_tile(meta, tile_id)
    busy(name, tile_id, person)
    try:
        words = vocab.of(get_state().project, meta["stain"])
        records = [outline.object_record(meta, words, person, tile_id, r.ref, r.value, r.points, r.deleted) for r in body.records]
    except ValueError as error:
        raise HTTPException(400, str(error))
    return store.append(root(), records)


@router.post("/annotate/{name}/outline/{tile_id}/complete")
def mark_complete(name: str, tile_id: str, body: Completion, request: Request, person: str = Depends(annotator)) -> dict:
    """The person says every object in the field is outlined (or takes that back)."""
    meta = _outline_meta(name, request, write=True)
    field_tile(meta, tile_id)
    return store.append(root(), [outline.complete_record(meta, person, tile_id, body.complete)])[0]


def _rings(layer: np.ndarray, labels: set[int] | None = None) -> list[list[int]]:
    rings = label_rings(layer, REFERENCE_SIMPLIFY_PX)
    return [ring for label, pieces in rings.items() if labels is None or label in labels for ring in pieces]


def _candidate_rings(project, stain: str, tile_id: str) -> list[list[int]] | None:
    target = proposals.path(project, stain, tile_id)

    def make() -> list[list[int]] | None:
        saved = proposals.load(project, stain, tile_id)
        if saved is None:
            return None
        out = []
        for layer in range(len(saved["layers"])):
            wanted = {int(v) for v, k in zip(saved["label"], saved["layer"]) if int(k) == layer}
            out += _rings(saved["layers"][layer], wanted)
        return out

    return cache.memo(("candidate rings", str(target)), [target], make)


@router.get("/annotate/{name}/outline/{tile_id}/reference")
def reference(name: str, tile_id: str, request: Request, person: str | None = Depends(viewer)) -> dict:
    """What the pipeline found in the field, for comparison after outlining: the objects it counted (circles, and
    their Segment Anything outlines where the Outline objects step made them) and, once the check has found them,
    every candidate it considered."""
    meta = _outline_meta(name, request)
    tile = field_tile(meta, tile_id)
    state = get_state()
    rgb = render.field_rgb(state, tile_id)
    analysis = state.analysis_tiles
    core_tiles = analysis[(analysis.core_id == tile["core_id"]) & (analysis.stain == meta["stain"])] if not analysis.empty else analysis
    found = proposals.detections(state.project, tile, core_tiles, rgb.shape[:2])
    return {"counted": [{"x": round(float(r.x), 1), "y": round(float(r.y), 1), "r": round(float(r.radius_px), 1)} for r in found.itertuples()],
            "outlines": [[v for xy in o["points"] for v in xy] for o in render.mask_outlines(state, tile_id)],
            "candidates": _candidate_rings(state.project, meta["stain"], tile_id)}


# ---------- set-up (admin) ----------

def _pick(stain: str, n: int, tmas: list[str], seed: int) -> list[str]:
    state = get_state()
    tiles = state.labelling_tiles
    tiles = tiles[(tiles.stain == stain) & tiles.tma.astype(str).isin(set(tmas))] if not tiles.empty else tiles
    if tiles.empty:
        raise HTTPException(400, f"There are no {stain} fields in {_where(tmas)}.")
    blocked = readiness.blockers(state.project, tiles)
    ready = tiles[[b is None for b in blocked]]
    if ready.empty:
        first = pd.Series([b["reason"] for b in blocked if b]).value_counts().index[0]
        raise HTTPException(400, f"No {stain} field in {_where(tmas)} can be outlined yet. {first}")
    labels = store.current(root())
    avoid = set(sessions.labelled_by(labels, stain)) | set(sessions.taken(root()))
    return outline.pick_fields(ready, _levels(stain, ready, seed), n, seed, avoid)


def _where(tmas: list[str]) -> str:
    return f"TMA {', '.join(tmas)}" if get_state().project.is_tma else f"batch {', '.join(tmas)}"


def _levels(stain: str, tiles: pd.DataFrame, seed: int) -> pd.Series:
    return outline.density_levels(outline.field_density(get_state().project, stain, tiles), seed)


def _density_of(stain: str, tile_ids: list[str], seed: int) -> dict[str, str]:
    """The density third of these fields among the stain's ready fields of the same TMAs."""
    state = get_state()
    tiles = state.labelling_tiles
    chosen = state.tiles[state.tiles.tile_id.isin(set(tile_ids))]
    tiles = tiles[(tiles.stain == stain) & tiles.tma.isin(set(chosen.tma))] if not tiles.empty else tiles
    levels = _levels(stain, pd.concat([tiles, chosen]).drop_duplicates("tile_id"), seed)
    return {t: str(levels.get(t, "unknown")) for t in tile_ids}


def proposals_job(stain: str, session: str) -> None:
    """Find the candidates of a check's fields in the background, so the scores and the reference layer are quick."""
    get_state().jobs.submit("score_segmentation", {"stain": stain, "session": session, "proposals_only": True})


def _all_tmas(stain: str) -> list[str]:
    tiles = get_state().labelling_tiles
    return sorted({str(t) for t in tiles[tiles.stain == stain].tma}, key=lambda t: (len(t), t)) if not tiles.empty else []


def _default_tmas(stain: str) -> list[str]:
    return outline.default_tmas(_all_tmas(stain), get_state().project.is_tma)


def create_outline_session(body: NewSession) -> dict:
    """A segmentation check for one stain: `n_fields` fields chosen by `outline.pick_fields` (or `tile_ids`), detection shown."""
    if body.stain not in STAINS:
        raise HTTPException(400, f"unknown stain {body.stain}")
    title = body.title.strip() or outline.title(body.stain)
    name = body.name or free_name(title, body.stain, outline.KIND)
    if name in sessions.names(root()) or name == sessions.REFERENCE:
        raise HTTPException(409, "a session with this name exists")
    tmas = body.tmas or _default_tmas(body.stain)
    seed = len(sessions.names(root()))
    fields = _known_fields(body.stain, body.tile_ids) if body.tile_ids else _pick(body.stain, max(1, body.n_fields), [str(t) for t in tmas], seed)
    benchmark = {"tmas": [str(t) for t in tmas], "density": _density_of(body.stain, fields, seed),
                 "density_measure": "objects detection counted per mm² of tissue, in thirds of the stain's ready fields on these TMAs"}
    meta = sessions.create(root(), name, title=title, stain=body.stain, kind=outline.KIND, tools=["outline"], ai_view="full",
                           per_field=max(1, body.per_field), rule="random", fields=fields, instructions=body.instructions, benchmark=benchmark)
    proposals_job(body.stain, name)
    if body.open:
        meta = sessions.update(root(), name, open=True)
        sam_embeddings.prepare_session(get_state(), meta, force=True)
    return meta


def add_outline_fields(meta: dict, n: int, tile_ids: list[str] | None) -> dict:
    tmas = meta.get("benchmark", {}).get("tmas") or _default_tmas(meta["stain"])
    more = _known_fields(meta["stain"], tile_ids) if tile_ids else _pick(meta["stain"], n, tmas, len(meta["fields"]) + 1)
    more = [t for t in more if t not in set(meta["fields"])]
    if not more:
        raise HTTPException(400, f"no more {meta['stain']} fields can be added")
    benchmark = {**meta.get("benchmark", {}), "density": {**meta.get("benchmark", {}).get("density", {}), **_density_of(meta["stain"], more, len(meta["fields"]))}}
    sessions.update(root(), meta["name"], benchmark=benchmark)
    updated = sessions.add_fields(root(), meta["name"], more)
    proposals_job(meta["stain"], meta["name"])
    sam_embeddings.prepare_session(get_state(), updated, force=True)
    return updated


class NewCheck(BaseModel):
    stains: list[str]
    n_fields: int = outline.DEFAULT_FIELDS
    tmas: list[str] | None = None
    open: bool = True


@router.post("/segmentation/sessions", dependencies=admin)
def create_checks(body: NewCheck) -> list[dict]:
    """One outline session per stain, named "Segmentation check – <stain>"."""
    return [create_outline_session(NewSession(stain=stain, kind=outline.KIND, n_fields=body.n_fields, tmas=body.tmas, open=body.open))
            for stain in body.stains]


@router.get("/segmentation", dependencies=admin)
def overview() -> dict:
    """The segmentation check: each stain's outline sessions and how far they are, the TMAs fields can come from, and
    the latest scores with the files to download."""
    state = get_state()
    labels = store.current(root())
    present = [s for s in STAINS if not state.tiles.empty and s in set(state.tiles.stain)]
    checks = []
    for meta in report.outline_sessions(state.project):
        sam_embeddings.prepare_session(state, meta)
        done = outline.finished(labels, meta["name"])
        checks.append({"name": meta["name"], "title": meta.get("title", meta["name"]), "stain": meta["stain"], "open": meta["open"],
                       "created": meta["created"], "fields": len(meta["fields"]), "complete": sum(bool(done.get(t)) for t in meta["fields"]),
                       "objects": int(len(outline.objects(labels, meta["name"]))), "people": sorted({p for ps in done.values() for p in ps}),
                       "link": annotate_path(meta["name"]), "tmas": meta.get("benchmark", {}).get("tmas", [])})
    path = report.report_path(state.project)
    try:
        scores = cache.memo(("segmentation report", str(path)), [path], lambda: json.loads(path.read_text()))
    except FileNotFoundError:
        scores = None
    folder = proposals.folder(state.project)
    files = [{"label": label, "path": state.project.relative(folder / name), "exists": (folder / name).exists()}
             for label, name in (("Scores (JSON)", "report.json"), ("One row per outlined object", "objects.csv"),
                                 ("One row per counted object", "detections.csv"), ("One row per field", "fields.csv"))]
    return {"stains": present, "checks": checks, "report": scores, "files": files, "segment_anything": state.project.model("sam").exists(),
            "tmas": {s: {"all": _all_tmas(s), "default": _default_tmas(s)} for s in present}, "tma_study": state.project.is_tma,
            "default_fields": outline.DEFAULT_FIELDS, "now": time.time(), "imported_sessions": len(report.imported_sessions(state.project))}
