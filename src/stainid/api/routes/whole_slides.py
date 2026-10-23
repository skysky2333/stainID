"""Whole slides in the Cores view: a whole-slide project's regions put back together (`stainid.imageset.slides`,
`stainid.api.slide_pyramid`). Every answer is the regions' own (the Cores view's routes of `cohort.py` and `areas.py`,
the same functions), each region's drawing moved by its offset into slide pixels, so the page draws a slide as it draws a
core. The regions stay the units of the pipeline, the labels and the tables: nothing here writes anything but the
slide's picture for viewing."""
from __future__ import annotations

import gzip
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse, Response

from stainid import cache, candidates
from stainid.api import slide_pyramid, slide_units, store_view
from stainid.api.session import require_admin
from stainid.api.state import ProjectState, get_state
from stainid.imageset.slides import Slide

router = APIRouter(tags=["whole slides"])
ADMIN = [Depends(require_admin)]
CACHE = {"Cache-Control": "public, max-age=86400"}
IMMUTABLE = {"Cache-Control": "public, max-age=31536000, immutable"}
PLAIN = SimpleNamespace(headers={})  # a request that takes no gzip: the regions' answers are read as plain JSON


def _slide(slide_id: str, stain: str | None = None) -> tuple[ProjectState, Slide]:
    state = get_state()
    try:
        found = slide_pyramid.slide(state, slide_id)
    except KeyError as error:
        raise HTTPException(404, str(error).strip("'")) from None
    if stain is not None and stain != found.stain:
        raise HTTPException(404, f"{slide_id} is a {found.stain} slide, not {stain}")
    return state, found


def _decoded(response) -> dict:
    if isinstance(response, dict):
        return response
    body = response.body
    if response.headers.get("content-encoding") == "gzip":
        body = gzip.decompress(body)
    return json.loads(body)


def _json(content: dict, request: Request) -> Response:
    from stainid.api.routes.cohort import _json as as_json  # noqa: PLC0415

    return as_json(content, request)


def _inputs(state: ProjectState, s: Slide, *kinds: str) -> list:
    """The regions' candidate-store files of these kinds (what a merged answer is kept until)."""
    out = []
    for r in s.regions:
        for kind in kinds:
            out.append(candidates.threads_path(state.project, r.core_id) if kind == "threads" else candidates.path(state.project, s.stain, r.core_id, kind))
    return out


# ---------------------------------------------------------------- the list, one slide, its picture

@router.get("/whole-slides")
def slide_list(prepare: bool = False) -> dict:
    """The slides of a whole-slide project (empty in other projects): name, donor, region, stain, size (slide px), how
    many regions and analysed tiles, and whether its picture is made (`prepare`: the ones not made are queued)."""
    state = get_state()
    found = slide_pyramid.slides(state)
    if prepare:
        for slide_id in slide_pyramid.pending(state):
            slide_pyramid.ask(state, slide_id)
    tiles = state.analysis_tiles
    counts = tiles.groupby("core_id").size().to_dict() if not tiles.empty else {}
    out = []
    for s in found.values():
        out.append({**{k: v for k, v in s.to_api().items() if k != "regions"}, "regions": len(s.regions),
                    "tiles": int(sum(counts.get(c, 0) for c in s.cores)), "ready": slide_pyramid.ready(state, s.slide_id)})
    return {"slides": out, "unit": "slide" if found else "core"}


def _info(state: ProjectState, s: Slide) -> dict:
    ids = list(slide_pyramid.slides(state))
    at = ids.index(s.slide_id)
    images = state.core_images
    first = images[images.core_id == s.regions[0].core_id].iloc[0].to_dict() if not images.empty else {}
    return {"core_id": s.slide_id, "slide_id": s.slide_id, "unit": "slide", "tma": s.tma, "core_label": s.name, "donor_id": s.donor_id or None,
            "sample_region_id": str(first.get("sample_region_id") or "") or None, "region": s.region or None,
            "disease_group": (str(first.get("disease_group")) if pd.notna(first.get("disease_group")) and first.get("disease_group") != "" else None),
            "technical_replicate": None,
            "images": [{"stain": s.stain, "native_width_px": s.width, "native_height_px": s.height, "pixel_width_um": state.project.pixel_size_um,
                        "tissue_status": "complete"}],
            "neighbours": {"previous": ids[at - 1] if at > 0 else None, "next": ids[at + 1] if at < len(ids) - 1 else None},
            "regions": [[r.core_id, r.x, r.y, r.width, r.height] for r in s.regions], "origin": list(s.origin)}


@router.get("/whole-slides/of/{core_id}")
def slide_of(core_id: str, x: float | None = None, y: float | None = None) -> dict:
    """The slide a region lies on, and a point of the region (core px) in slide px: where a link to a region goes."""
    state = get_state()
    s = slide_pyramid.slide_of(state, core_id)
    if s is None:
        raise HTTPException(404, f"{core_id} is not a region of a whole slide")
    region = s.find(core_id)
    return {"slide_id": s.slide_id, "stain": s.stain, "x": None if x is None else x + region.x, "y": None if y is None else y + region.y,
            "region": [region.x, region.y, region.width, region.height]}


@router.get("/whole-slides/{slide_id}")
def slide_detail(slide_id: str) -> dict:
    state, s = _slide(slide_id)
    return _info(state, s)


@router.get("/whole-slides/{slide_id}/{stain}/picture")
def slide_picture(slide_id: str, stain: str) -> dict:
    state, s = _slide(slide_id, stain)
    return slide_pyramid.picture(state, s.slide_id)


@router.get("/images/whole-slides/{slide_id}/{stain}/tiles/{level}/{col}_{row}.jpg")
def slide_tile(slide_id: str, stain: str, level: int, col: int, row: int, v: str = "") -> Response:
    state, s = _slide(slide_id, stain)
    try:
        data = slide_pyramid.tile(state, s.slide_id, level, col, row)
    except KeyError as error:
        raise HTTPException(404, str(error).strip("'")) from None
    return Response(data, media_type="image/jpeg", headers=IMMUTABLE if v else CACHE)


@router.get("/images/whole-slides/{slide_id}/{stain}/tiles/{level}/{col}_{row}_dab.png")
def slide_tile_dab(slide_id: str, stain: str, level: int, col: int, row: int, v: str = "") -> Response:
    state, s = _slide(slide_id, stain)
    try:
        data = slide_pyramid.dab_tile(state, s.slide_id, level, col, row)
    except KeyError as error:
        raise HTTPException(404, str(error).strip("'")) from None
    return Response(data, media_type="image/png", headers=IMMUTABLE if v else CACHE)


@router.get("/images/whole-slides/{slide_id}/{stain}.jpg")
def slide_thumbnail(slide_id: str, stain: str, size: int = 512) -> FileResponse:
    state, s = _slide(slide_id, stain)
    try:
        return FileResponse(slide_pyramid.thumbnail(state, s.slide_id, size), headers={"Cache-Control": "private, max-age=600"})
    except KeyError as error:
        raise HTTPException(404, str(error).strip("'")) from None


@router.get("/images/whole-slides/{slide_id}/{stain}/crop.jpg")
def slide_crop(slide_id: str, stain: str, x0: float, y0: float, x1: float, y1: float, size: int = 1024) -> Response:
    """A box of the slide (slide px) at most `size` px across."""
    state, s = _slide(slide_id, stain)
    try:
        data = slide_pyramid.crop_jpeg(state, s.slide_id, (x0, y0, x1, y1), size)
    except KeyError as error:
        raise HTTPException(404, str(error).strip("'")) from None
    return Response(data, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=600"})


# ---------------------------------------------------------------- what detection found, in slide px

def _objects(state: ProjectState, s: Slide) -> dict:
    from stainid.api.routes import cohort  # noqa: PLC0415

    parts = [(r, cohort.core_objects(r.core_id, s.stain)) for r in s.regions]
    rows, boxes = [], []
    for r, part in parts:
        rows += [[row[0] + r.x, row[1] + r.y, *row[2:]] for row in part["rows"]]
        boxes += [[b[0] + r.x, b[1] + r.y, b[2], b[3]] for b in part["tiles"]]
    states = {p["status"] for _, p in parts}
    status = "done" if states == {"done"} else "not_run" if states <= {"not_run"} else "partial"
    threads = [p["threads"] for _, p in parts if p.get("threads")]
    unproposed = [p.get("unproposed") for _, p in parts]
    first = parts[0][1] if parts else {}
    return {"status": status, "tiles_done": sum(p["tiles_done"] for _, p in parts), "tiles_total": sum(p["tiles_total"] for _, p in parts),
            "columns": first.get("columns", []), "rows": rows, "tiles": boxes, "outlines": all(p["outlines"] for _, p in parts),
            "threads": {"length_mm": round(sum(t["length_mm"] for t in threads), 2), "skeleton_px": int(sum(t["skeleton_px"] for t in threads))} if threads else None,
            "source": "v2" if all(p.get("source") == "v2" for _, p in parts) else "current",
            "unproposed": None if any(u is None for u in unproposed) else int(sum(unproposed)), "outline_default": first.get("outline_default")}


@router.get("/whole-slides/{slide_id}/{stain}/objects")
def slide_objects(slide_id: str, stain: str, request: Request) -> Response:
    """Every object of every region (`cohort.core_objects`), in slide px."""
    state, s = _slide(slide_id, stain)
    found = cache.memo(("slide objects", str(state.root), s.slide_id), _inputs(state, s, "candidates", "unproposed"), lambda: _objects(state, s), every=5)
    return _json(found, request)


@router.get("/whole-slides/{slide_id}/{stain}/outlines")
def slide_outlines(slide_id: str, stain: str, request: Request, outline: str = "default") -> Response:
    """Every region's outlines (`cohort.core_outlines`), in slide px."""
    from stainid.api.routes import cohort  # noqa: PLC0415

    if outline not in store_view.OUTLINE_CHOICES:
        raise HTTPException(400, f"outline must be one of {', '.join(store_view.OUTLINE_CHOICES)}")
    state, s = _slide(slide_id, stain)

    def make() -> dict:
        rows, statuses = [], set()
        for r in s.regions:
            part = _decoded(cohort.core_outlines(r.core_id, s.stain, PLAIN, outline))
            statuses.add(part["status"])
            rows += [[row[0] + r.x, row[1] + r.y, row[2], [slide_units.flat_moved(ring, r) for ring in row[3]]] for row in part["rows"]]
        return {"status": "done" if "done" in statuses else "not_made", "columns": cohort.OUTLINE_COLUMNS, "outline": outline, "rows": rows}

    found = cache.memo(("slide outlines", str(state.root), s.slide_id, outline), _inputs(state, s, "candidates", "outlines"), make, every=5)
    return _json(found, request)


@router.get("/whole-slides/{slide_id}/{stain}/threads")
def slide_threads(slide_id: str, stain: str, request: Request, prepare: bool = True) -> Response:
    """The AT8 threads of every region (`cohort.core_threads`), in slide px."""
    from stainid.api.routes import cohort  # noqa: PLC0415

    if stain != "AT8":
        raise HTTPException(404, "neuropil threads are traced on AT8 only")
    state, s = _slide(slide_id, stain)

    def make() -> dict:
        parts = [(r, _decoded(cohort.core_threads(r.core_id, s.stain, PLAIN, prepare))) for r in s.regions]
        rows = [[row[0], slide_units.flat_moved(row[1], r)] for r, p in parts for row in p.get("rows", [])]
        statuses = [p.get("status") for _, p in parts]
        status = "done" if all(x == "done" for x in statuses) else next((x for x in ("tracing", "waiting", "failed") if x in statuses), "not_run")
        measured = [p["measured"] for _, p in parts if p.get("measured")]
        return {"status": status, "tiles_done": sum(int(p.get("tiles_done") or 0) for _, p in parts),
                "tiles_total": sum(int(p.get("tiles_total") or 0) for _, p in parts), "pieces": len(rows),
                "length_mm": round(sum(float(p.get("length_mm") or 0) for _, p in parts), 2), "columns": ["width_um", "points"], "rows": rows,
                "measured": {"length_mm": round(sum(m["length_mm"] for m in measured), 2), "skeleton_px": int(sum(m["skeleton_px"] for m in measured))}
                if measured else None}

    inputs = _inputs(state, s, "threads")
    if not all(p.exists() for p in inputs):  # some region still traced from its image: asked afresh until all are saved
        return _json(make(), request)
    return _json(cache.memo(("slide threads", str(state.root), s.slide_id), inputs, make, every=5), request)


@router.get("/whole-slides/{slide_id}/{stain}/unproposed")
def slide_unproposed(slide_id: str, stain: str, request: Request) -> Response:
    state, s = _slide(slide_id, stain)

    def make() -> dict:
        rows, statuses = [], set()
        for r in s.regions:
            part = store_view.unproposed(state.project, r.core_id, s.stain)
            statuses.add(part["status"])
            rows += [[row[0] + r.x, row[1] + r.y, row[2], row[3], [slide_units.flat_moved(ring, r) for ring in row[4]]] for row in part["rows"]]
        return {"status": "done" if "done" in statuses else "not_run", "columns": ["x", "y", "area_um2", "dab_over_threshold", "rings"], "rows": rows}

    return _json(cache.memo(("slide unproposed", str(state.root), s.slide_id), _inputs(state, s, "unproposed"), make, every=5), request)


# ---------------------------------------------------------------- excluded areas: one picture of the slide

@router.get("/whole-slides/{slide_id}/{stain}/excluded")
def slide_excluded(slide_id: str, stain: str) -> dict:
    state, s = _slide(slide_id, stain)
    return slide_units.excluded_info(state.project, s)


@router.get("/whole-slides/{slide_id}/{stain}/excluded.png")
def slide_excluded_png(slide_id: str, stain: str) -> Response:
    state, s = _slide(slide_id, stain)
    png = slide_units.excluded_png(state.project, s)
    if png is None:
        raise HTTPException(404, f"the excluded areas of {slide_id} are not made yet (the v2 proposals make them)")
    return Response(png, media_type="image/png", headers={"Cache-Control": "private, max-age=86400"})


# ---------------------------------------------------------------- people's work on the slide (admin)

@router.get("/whole-slides/{slide_id}/{stain}/annotations", dependencies=ADMIN)
def slide_annotations(slide_id: str, stain: str) -> dict:
    """The Annotations layer of every region (`areas.core_annotations`), in slide px."""
    from stainid.api.routes import areas  # noqa: PLC0415

    state, s = _slide(slide_id, stain)
    parts = [(r, slide_units.moved(areas.core_annotations(r.core_id, s.stain), r)) for r in s.regions]
    counts: dict[str, int] = {}
    for _, p in parts:
        for k, v in p["counts"].items():
            counts[k] = counts.get(k, 0) + int(v or 0)
    matters = [p["matter"] for _, p in parts if p.get("matter")]
    matter = None
    if matters:
        latest = max(matters, key=lambda m: m.get("time") or 0)
        matter = {**latest, "regions": [region for m in matters for region in m.get("regions", [])]}
    return {"objects": [o for _, p in parts for o in p["objects"]], "added": [a for _, p in parts for a in p["added"]],
            "edits": [e for _, p in parts for e in p.get("edits", [])], "counts": counts, "matter": matter,
            "areas": {"regions": [g for _, p in parts for g in (p.get("areas") or {}).get("regions", [])],
                      "agreement": slide_units.agreement([(p.get("areas") or {}).get("agreement") for _, p in parts]),
                      "edits": sorted((e for _, p in parts for e in (p.get("areas") or {}).get("edits", [])), key=lambda e: e.get("time") or 0)}}


@router.get("/whole-slides/{slide_id}/{stain}/status", dependencies=ADMIN)
def slide_status(slide_id: str, stain: str) -> dict:
    """The minimap of every region (`areas.core_status`), in slide px."""
    from stainid.api.routes import areas  # noqa: PLC0415

    state, s = _slide(slide_id, stain)
    parts = [(r, areas.core_status(r.core_id, s.stain)) for r in s.regions]
    counts: dict[str, int] = {}
    sessions: dict[str, dict] = {}
    checked = None
    for _, p in parts:
        for k, v in p["counts"].items():
            counts[k] = counts.get(k, 0) + int(v or 0)
        for found in p["sessions"]:
            kept = sessions.setdefault(found["name"], {**found, "tiles": []})
            kept["tiles"] = sorted(set(kept["tiles"]) | set(found["tiles"]))
        people = set(p["areas_checked_by"])
        checked = people if checked is None else checked & people
    return {"core_id": s.slide_id, "stain": s.stain, "width": s.width, "height": s.height,
            "tiles": [{**t, "x": t["x"] + r.x, "y": t["y"] + r.y} for r, p in parts for t in p["tiles"]],
            "not_analysed": [{**t, "x": t["x"] + r.x, "y": t["y"] + r.y} for r, p in parts for t in p["not_analysed"]],
            "counts": counts, "areas_checked_by": sorted(checked or set()), "sessions": list(sessions.values())}


@router.get("/whole-slides/{slide_id}/{stain}/threshold-preview", dependencies=ADMIN)
def slide_threshold_preview(slide_id: str, stain: str) -> dict:
    from stainid.api.routes import areas  # noqa: PLC0415

    state, s = _slide(slide_id, stain)
    parts = [areas.threshold_preview(r.core_id, s.stain) for r in s.regions]
    found = [p["detection"] for p in parts if p.get("detection") is not None]
    return {"detection": float(np.median(found)) if found else None, "columns": next((p["columns"] for p in parts if p.get("columns")), []),
            "rows": [row for p in parts for row in p["rows"]]}
