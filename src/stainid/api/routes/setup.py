"""The guided setup (Slides -> Cores -> Map -> Donors -> Checks -> Run): thin routes over `stainid.setup`."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import cv2
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel

from stainid.api import render
from stainid.api.state import get_state, reload_project
from stainid.project import save_project
from stainid.setup import checks, donors, guide, maps
from stainid.setup.files import parse_text, read_tables
from stainid.setup.naming import Groups, IdStyle
from stainid.slides.dearray import ORIENTATIONS, slide_slug
from stainid.slides.table import read_slides, set_slide_grid
from stainid.tables import read_csv
from stainid.workflows.steps import BY_ID

router = APIRouter(prefix="/setup", tags=["setup"])
MAIN = ("1 · Set up", "2 · Detect", "3 · Results")


class GridRequest(BaseModel):
    slide_path: str
    rows: int | None = None
    columns: int | None = None
    orientation: str = ""
    run: bool = True


class CheckedRequest(BaseModel):
    slide_path: str
    checked: bool = True


class TextRequest(BaseModel):
    text: str
    name: str = "Pasted map"


class GridCells(BaseModel):
    grid: list[list[str]]
    source: str = ""


class DonorRequest(BaseModel):
    rows: list[list[str]]
    mapping: dict[str, str] | None = None
    keep: str = "study"
    mode: str = "update"


class FixRequest(BaseModel):
    action: str
    args: dict = {}


def _groups() -> Groups:
    return Groups(get_state().project.config.get("groups"))


def _style() -> IdStyle:
    return IdStyle.from_config(get_state().project.config)


def _sample_ids() -> maps.SampleIds:
    project = get_state().project
    return maps.sample_ids_for(project.config, maps.read_layout_rows(project.input("tma_layout")))


def _rows(path: Path) -> list[dict[str, str]]:
    return read_csv(path) if path.exists() else []


def _steps() -> list[dict]:
    from stainid.api.routes.jobs import steps

    return steps()


def _remaining(steps: list[dict]) -> list[dict]:
    busy = {s["id"] for s in steps if s.get("job") and s["job"]["status"] in ("running", "queued")}
    return [s for s in steps if s["stage"] in MAIN and s["command"] and s["id"] != "masks" and s["id"] not in busy
            and (s["progress"]["state"] != "done" or s["progress"].get("outdated"))]


# ---------------------------------------------------------------- overview and run

@router.get("")
def overview() -> dict:
    project = get_state().project
    steps = _steps()
    screens = guide.screens(project, steps)
    state = {s["id"]: s["state"] for s in screens}
    remaining = _remaining(steps)
    return {"screens": screens, "set_up": all(state[s] in ("done", "optional") for s in ("slides", "cores", "map")),
            "estimate": guide.estimate(project, [s["id"] for s in remaining]), "remaining": [{"id": s["id"], "title": s["title"]} for s in remaining]}


@router.post("/run")
def run_everything() -> dict:
    """Queue every step that is not done yet (except the optional outlines) with its default options; the queue starts
    them in order. Steps made with an older model start over."""
    jobs = get_state().jobs
    queued = []
    for step in _remaining(_steps()):
        options = {o.key: o.default for o in BY_ID[step["id"]].options}
        if "fresh" in options and step["progress"].get("outdated"):
            options["fresh"] = True
        queued.append(jobs.describe(jobs.submit(step["id"], options)))
    return {"queued": queued}


# ---------------------------------------------------------------- cores

@router.get("/cores")
def cores() -> dict:
    project = get_state().project
    return {"slides": guide.slide_reports(project), "rows": int(project.config["tma"]["rows"]), "columns": int(project.config["tma"]["columns"]),
            "orientations": ORIENTATIONS}


@router.put("/cores/grid")
def set_grid(request: GridRequest) -> dict:
    """Give one slide its own grid (or clear it: found automatically) and find its cores again."""
    state = get_state()
    if request.orientation not in ORIENTATIONS:
        raise HTTPException(400, f"Unknown orientation {request.orientation}")
    if bool(request.rows) != bool(request.columns) or any(v is not None and not 1 <= v <= 26 for v in (request.rows, request.columns)):
        raise HTTPException(400, "Give both the rows and the columns (1-26), or neither to have them found automatically")
    try:
        set_slide_grid(state.project.input("slides_table"), request.slide_path, request.rows, request.columns, request.orientation)
    except KeyError:
        raise HTTPException(404, f"{request.slide_path} is not in the slides table")
    job = state.jobs.describe(state.jobs.submit("dearray", {"slide": Path(request.slide_path).name})) if request.run else None
    return {"job": job, **cores()}


@router.post("/cores/checked")
def checked(request: CheckedRequest) -> dict:
    guide.mark_checked(get_state().project, request.slide_path, request.checked)
    return cores()


@lru_cache(maxsize=24)
def _picture(path: str, mtime: float):
    return cv2.imread(path, cv2.IMREAD_COLOR)


@router.get("/thumbs/{tma}/{label}.jpg")
def thumbnail(tma: str, label: str, size: int = 160, stain: str = "") -> Response:
    """A small picture of one core position, cut from the slide preview Find cores kept (or the exported core image)."""
    state = get_state()
    project = state.project
    rows = [r for r in _rows(project.input("core_manifest")) if r["tma"] == tma and r["core_label"] == label]
    rows.sort(key=lambda r: (r["stain"] != stain, r["stain"]))
    size = max(32, min(size, 512))
    for row in rows:
        slug = slide_slug(Path(row["slide_path"]))
        grids = project.output("qc") / "grids"
        for picture, scale_file in ((grids / "previews" / f"{slug}.jpg", grids / "previews" / f"{slug}.json"), (grids / f"{slug}.jpg", None)):
            if not picture.exists() or not row.get("preview_scale_x"):
                continue
            scale = json.loads(scale_file.read_text())["scale"] if scale_file and scale_file.exists() else 1.0
            image = _picture(str(picture), picture.stat().st_mtime)
            if image is None:
                continue
            x = float(row["center_x_px"]) / float(row["preview_scale_x"]) * scale
            y = float(row["center_y_px"]) / float(row["preview_scale_y"]) * scale
            radius = max(4, int(float(row["diameter_x_px"]) / float(row["preview_scale_x"]) * scale / 2))
            crop = image[max(0, int(y) - radius): int(y) + radius, max(0, int(x) - radius): int(x) + radius]
            if crop.size:
                ok, encoded = cv2.imencode(".jpg", cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 82])
                return Response(encoded.tobytes(), media_type="image/jpeg", headers={"Cache-Control": "max-age=300"})
        try:
            path = render.core_thumbnail(state, f"{project.tma_prefix}{tma}_{label}", row["stain"], size)
            return Response(path.read_bytes(), media_type="image/jpeg", headers={"Cache-Control": "max-age=300"})
        except (KeyError, FileNotFoundError):
            continue
    raise HTTPException(404, f"No picture of TMA {tma} {label} yet")


# ---------------------------------------------------------------- map

def _sizes() -> dict[str, tuple[int, int]]:
    """Rows and columns of each TMA: from Find cores, else the slides table, else the project setting."""
    project = get_state().project
    sizes: dict[str, tuple[int, int]] = {}
    for row in _rows(project.input("core_manifest")):
        r, c = maps.split_position(row["core_label"])
        old = sizes.get(row["tma"], (0, 0))
        sizes[row["tma"]] = (max(old[0], r + 1), max(old[1], c + 1))
    default = (int(project.config["tma"]["rows"]), int(project.config["tma"]["columns"]))
    for slide in read_slides(project.input("slides_table")):
        if slide.get("tma") and slide["tma"] not in sizes:
            sizes[slide["tma"]] = (int(slide["rows"]), int(slide["columns"])) if slide.get("rows") and slide.get("columns") else default
    return sizes


@router.get("/map")
def map_overview() -> dict:
    project = get_state().project
    sizes = _sizes()
    layout = maps.read_layout_rows(project.input("tma_layout"))
    positions: dict[str, list[str]] = {}
    for row in _rows(project.input("core_manifest")):
        positions.setdefault(row["tma"], [])
        if row["core_label"] not in positions[row["tma"]]:
            positions[row["tma"]].append(row["core_label"])
    slides = read_slides(project.input("slides_table"))
    tmas = sorted(set(sizes) | {r["tma"] for r in layout}, key=lambda t: (len(t), t))
    out = []
    for tma in tmas:
        saved = [r for r in layout if r["tma"] == tma]
        cells = maps.parse_map(maps.grid_of(saved), _groups(), _style(), _sample_ids()) if saved else []
        out.append({"tma": tma, "size": list(sizes.get(tma, (0, 0))), "slides": [s["stain"] for s in slides if s["tma"] == tma],
                    "positions": positions.get(tma, []), "saved": maps.summary(cells) if saved else None,
                    "sources": sorted({r.get("map_source", "") for r in saved} - {""})})
    return {"tmas": out, "path": project.relative(project.input("tma_layout")), "id_style": project.config.get("donor_ids", {}),
            "tma_prefix": project.tma_prefix, "sample_ids": _sample_ids().describe(), "sample_ids_setting": project.config.get("sample_region_ids", {})}


@router.get("/map/{tma}")
def map_of(tma: str) -> dict:
    project = get_state().project
    size = _sizes().get(tma, (int(project.config["tma"]["rows"]), int(project.config["tma"]["columns"])))
    saved = maps.map_of(project.input("tma_layout"), tma)
    grid = maps.grid_of(saved, size)
    return {"tma": tma, "size": list(size), "grid": grid, "saved": bool(saved), **_parsed(grid)}


def _parsed(grid: list[list[str]]) -> dict:
    cells = maps.parse_map(grid, _groups(), _style(), _sample_ids())
    return {"cells": cells, "summary": maps.summary(cells)}


async def _tables(file: UploadFile) -> list:
    try:
        return read_tables(file.filename or "file", await file.read())
    except ValueError as error:
        raise HTTPException(400, str(error))


def _found_maps(tables: list) -> dict:
    known = sorted(_sizes(), key=lambda t: (len(t), t))
    found = maps.read_maps(tables, _groups(), known)
    if not found:
        raise HTTPException(400, "No map was found in this file: it should hold a grid with one cell per core position")
    return {"maps": [{**m, **_parsed(m["grid"])} for m in found]}


@router.post("/map/read")
async def read_map_file(file: UploadFile = File(...)) -> dict:
    return _found_maps(await _tables(file))


@router.post("/map/read-text")
def read_map_text(request: TextRequest) -> dict:
    from stainid.setup.files import Table

    return _found_maps([Table(request.name, parse_text(request.text), source=request.name)])


@router.post("/map/parse")
def parse(request: GridCells) -> dict:
    return _parsed(request.grid)


@router.put("/map/{tma}")
def save_map(tma: str, request: GridCells) -> dict:
    state = get_state()
    project = state.project
    cells = maps.parse_map(request.grid, _groups(), _style(), _sample_ids())
    rows = maps.layout_rows(tma, cells, request.source)
    try:
        maps.save_map(project.input("tma_layout"), tma, rows)
    except ValueError as error:
        raise HTTPException(400, str(error))
    if not project.config.get("groups"):
        codes = sorted({c["disease_group"] for c in cells if c["disease_group"]})
        if codes:
            save_project(project, {"groups": [{"code": c, "label": c} for c in codes]})
    reload_project(project.root)
    return map_overview()


@router.put("/id-style")
def id_style(style: dict) -> dict:
    project = get_state().project
    try:
        digits = {key: int(style.get(key, default)) for key, default in (("digits", 4), ("prefixed_digits", 3))}
    except (TypeError, ValueError):
        raise HTTPException(400, "Give the number of digits as a whole number from 0 to 8")
    if not all(0 <= v <= 8 for v in digits.values()):
        raise HTTPException(400, "Give the number of digits as a whole number from 0 to 8")
    save_project(project, {"donor_ids": digits})
    reload_project(project.root)
    return map_overview()


@router.put("/sample-ids")
def sample_ids(rule: dict) -> dict:
    """The `sample_region_ids` setting (used while no saved map shows the project's own naming)."""
    project = get_state().project
    if rule.get("region", "name") not in ("name", "code"):
        raise HTTPException(400, "region must be name or code")
    save_project(project, {"sample_region_ids": {"plain_prefix": str(rule.get("plain_prefix", "")), "prefixed_prefix": str(rule.get("prefixed_prefix", "")),
                                                 "region": rule.get("region", "name")}})
    reload_project(project.root)
    return map_overview()


# ---------------------------------------------------------------- donors

@router.get("/donors")
def donor_overview() -> dict:
    project = get_state().project
    rows = donors.read_donor_table(project.input("donor_metadata"))
    return {"path": project.relative(project.input("donor_metadata")), "count": len(rows), "columns": donors.columns_of(rows),
            "fields": {k: v[0] for k, v in donors.FIELDS.items()}, "preview": rows[:8]}


@router.post("/donors/read")
async def read_donor_file(file: UploadFile = File(...)) -> dict:
    tables = await _tables(file)
    return {"tables": [{"name": t.name, "source": t.source, "rows": t.rows} for t in tables]}


def _sheet(request: DonorRequest):
    project = get_state().project
    layout = maps.read_layout_rows(project.input("tma_layout"))
    on_map = {r["donor_id"] for r in layout if r.get("donor_id")}
    mapping = {int(k): v for k, v in (request.mapping or {}).items()}
    try:
        sheet = donors.read_donors(request.rows, _groups(), _style(), mapping or None, donors.prefixes(on_map) if layout else None)
    except ValueError as error:
        raise HTTPException(400, str(error))
    study = {r["tma"] for r in layout} | {s["tma"] for s in read_slides(project.input("slides_table"))}
    kept = donors.select_donors(sheet.records, on_map, study, request.keep)
    return sheet, kept, on_map


@router.post("/donors/preview")
def donor_preview(request: DonorRequest) -> dict:
    sheet, kept, on_map = _sheet(request)
    found = {r["donor_id"] for r in kept}
    notes = list(sheet.notes)
    if on_map and request.keep != "all":
        notes.append(f"{len(kept)} of {len(sheet.records)} donors are kept: those on the TMA maps or on a TMA of this study "
                     f"({len(on_map & found)} of the {len(on_map)} donors on the maps are in the sheet).")
    return {"header_row": sheet.header_row, "columns": sheet.columns, "records": kept[:60], "total": len(sheet.records), "kept": len(kept),
            "notes": notes, "table_columns": donors.columns_of(kept), "fields": {k: v[0] for k, v in donors.FIELDS.items()}}


@router.put("/donors")
def save_donors(request: DonorRequest) -> dict:
    project = get_state().project
    _, kept, _ = _sheet(request)
    try:
        count = donors.save_donors(project.input("donor_metadata"), kept, request.mode)
    except ValueError as error:
        raise HTTPException(400, str(error))
    reload_project(project.root)
    return {"saved": project.relative(project.input("donor_metadata")), "count": count, "added": len(kept)}


# ---------------------------------------------------------------- checks

@router.get("/checks")
def issues() -> list[dict]:
    return checks.project_issues(get_state().project)


@router.post("/checks/fix")
def fix(request: FixRequest) -> dict:
    project = get_state().project
    try:
        message = checks.apply_fix(project, request.action, request.args)
    except (ValueError, KeyError) as error:
        raise HTTPException(400, str(error))
    reload_project(project.root)
    return {"message": message, "issues": checks.project_issues(get_state().project)}

