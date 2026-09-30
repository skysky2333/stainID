from __future__ import annotations

import io
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

from stainid.api.state import get_state, open_project, recent_projects
from stainid.api.util import records
from stainid.project import DEFAULTS, Project, save_project
from stainid.slides.layout import REQUIRED_COLUMNS, read_layout
from stainid.tables import read_csv

router = APIRouter(tags=["project"])

HELP = {
    "name": "A name for this study, shown at the top of the app.",
    "pixel_size_um": "Micrometres per pixel of the scans; only used when a table has no pixel size.",
    "field_context_px": "Extra margin (pixels) read around each analysis field so edge objects are measured whole.",
    "tma.prefix": "Text put in front of the TMA number in core names, e.g. LIP- gives LIP-3_B-2.",
    "tma.rows": "Number of core rows on each TMA slide.",
    "tma.columns": "Number of core columns on each TMA slide (columns are lettered A, B, C …).",
    "groups": "Diagnostic group codes used in the TMA map, the label to show for each, in plot order.",
    "inputs.slides_dir": "Folder with the whole-slide scans.",
    "inputs.slides_table": "Which scan is which TMA and stain (made on the Slides step).",
    "inputs.core_manifest": "Core table (made by Find cores).",
    "inputs.tma_layout": "TMA map: donor, region and group per core position.",
    "inputs.donor_metadata": "Optional: one row per donor with age, sex, APOE and other covariates.",
    "inputs.manual_exclusions": "Optional: hand-drawn regions to ignore (folds, bubbles).",
    "inputs.core_images": "Core image table (made by Choose analysis fields).",
    "inputs.field_manifest": "All candidate fields (made by Choose analysis fields).",
    "inputs.tile_manifest": "The analysed fields (made by Choose analysis fields).",
    "inputs.calibration": "Stain thresholds (made by Calibrate).",
}
UPLOADS = {"tma_layout": "TMA map", "donor_metadata": "donor information", "slides_table": "slides table"}


class ConfigChange(BaseModel):
    changes: dict


class OpenRequest(BaseModel):
    path: str
    name: str = ""


class PathRequest(BaseModel):
    path: str


def _paths(project: Project) -> dict:
    return {section: {key: {"path": project.relative(project.resolve(section, key)), "exists": project.resolve(section, key).exists(),
                            "help": HELP.get(f"{section}.{key}", "")}
                      for key in project.config[section]} for section in ("inputs", "models", "outputs")}


@router.get("/project")
def project_info() -> dict:
    state = get_state()
    project = state.project
    return {"name": project.name, "root": str(project.root), "configured": project.is_configured, "config": project.config,
            "defaults": DEFAULTS, "help": HELP, "pixel_size_um": project.pixel_size_um, "stains": project.stains,
            "tma_prefix": project.tma_prefix, "groups": state.groups(), "paths": _paths(project), "recent": recent_projects()}


@router.put("/project/config")
def update_config(request: ConfigChange) -> dict:
    state = get_state()
    open_project(save_project(state.project, request.changes).root)
    return project_info()


@router.post("/project/open")
def open_existing(request: OpenRequest) -> dict:
    path = Path(request.path).expanduser()
    if not (path / "stainid.yaml").exists():
        raise HTTPException(404, f"{path} is not a stainID project yet (it has no stainid.yaml). Use 'Create project' to start one there.")
    _switch(path)
    return project_info()


@router.post("/project/create")
def create(request: OpenRequest) -> dict:
    path = Path(request.path).expanduser()
    if (path / "stainid.yaml").exists():
        raise HTTPException(409, f"{path} already contains a stainID project; open it instead.")
    save_project(Project(root=path.resolve(), config=DEFAULTS), {"name": request.name or path.name})
    _switch(path)
    return project_info()


def _switch(path: Path) -> None:
    try:
        open_project(path)
    except RuntimeError as error:
        raise HTTPException(409, str(error))


@router.get("/summary")
def summary() -> dict:
    state = get_state()
    tiles = state.tiles
    donors = state.donors
    source = donors if not donors.empty else tiles.drop_duplicates("donor_id") if not tiles.empty else donors
    groups = source.disease_group.value_counts().to_dict() if "disease_group" in source else {}
    return {
        "donors": int(source.donor_id.nunique()) if "donor_id" in source else 0,
        "groups": {k: int(v) for k, v in groups.items()},
        "tmas": sorted(tiles.tma.astype(str).unique().tolist(), key=lambda t: (len(t), t)) if not tiles.empty else [],
        "cores": int(tiles.core_id.nunique()) if not tiles.empty else 0,
        "donor_regions": int(tiles.sample_region_id.nunique()) if not tiles.empty else 0,
        "fields": {k: int(v) for k, v in tiles.stain.value_counts().items()} if not tiles.empty else {},
        "status": state.output_status(),
    }


@router.get("/calibration")
def calibration() -> list[dict]:
    return records(get_state().calibration())


@router.get("/fs")
def browse(path: str = "", show: str = "") -> dict:
    """Folder browser for picking paths (the app runs on this computer, so this lists local folders)."""
    folder = Path(path).expanduser() if path else get_state().root
    folder = folder if folder.is_dir() else folder.parent
    entries = sorted((p for p in folder.iterdir() if not p.name.startswith(".")), key=lambda p: p.name.lower())
    suffixes = {".csv": {".csv"}, "slides": {".vsi", ".svs", ".ndpi", ".scn", ".mrxs", ".czi", ".tif", ".tiff"},
                "models": {".joblib", ".pth", ".pt", ""}}.get(show, set())
    shortcuts = [{"label": "Project folder", "path": str(get_state().root)}, {"label": "Home", "path": str(Path.home())}]
    shortcuts += [{"label": p.name, "path": str(p)} for p in sorted(Path("/Volumes").iterdir())] if Path("/Volumes").exists() else []
    return {"path": str(folder), "parent": str(folder.parent) if folder.parent != folder else None,
            "dirs": [p.name for p in entries if p.is_dir() and p.suffix.lower() not in suffixes],
            "files": [p.name for p in entries if p.is_file() and (not suffixes or p.suffix.lower() in suffixes)], "shortcuts": shortcuts}


@router.post("/reveal")
def reveal(request: PathRequest) -> dict:
    """Show a file or folder in Finder / Explorer / the file manager."""
    path = _inside_project(request.path, must_exist=False)
    while not path.exists():
        path = path.parent
    if sys.platform == "darwin":
        subprocess.run(["open", "-R", str(path)], check=True)
    elif sys.platform == "win32":
        subprocess.run(["explorer", "/select,", str(path)])
    else:
        subprocess.run(["xdg-open", str(path if path.is_dir() else path.parent)], check=True)
    return {"ok": True}


def _inside_project(path: str, must_exist: bool = True, contained: bool = False) -> Path:
    root = get_state().root.resolve()
    target = (root / path).resolve()
    if contained and root not in target.parents:
        raise HTTPException(403, "only files inside the project folder can be downloaded")
    if must_exist and not target.exists():
        raise HTTPException(404, f"{path} does not exist yet")
    return target


@router.get("/download")
def download(path: str) -> FileResponse:
    target = _inside_project(path, contained=True)
    if not target.is_file():
        raise HTTPException(400, "only files can be downloaded")
    return FileResponse(target, filename=target.name)


@router.get("/templates/{kind}.csv")
def template(kind: str) -> Response:
    """A CSV to fill in, pre-filled with every core position / donor the project already knows about."""
    project = get_state().project
    if kind == "tma_layout":
        manifest = project.input("core_manifest")
        positions = sorted({(r["tma"], r["core_label"]) for r in read_csv(manifest)}) if manifest.exists() else []
        frame = pd.DataFrame([{"tma": t, "core_label": c} for t, c in positions], columns=["tma", "core_label"])
        for column in REQUIRED_COLUMNS[2:] + ("cerad", "braak"):
            frame[column] = ""
    elif kind == "donor_metadata":
        layout = project.input("tma_layout")
        donors = sorted({r["donor_id"] for r in read_csv(layout) if r["donor_id"]}) if layout.exists() else []
        groups = {r["donor_id"]: r["disease_group"] for r in read_csv(layout)} if layout.exists() else {}
        frame = pd.DataFrame([{"donor_id": d, "disease_group": groups.get(d, "")} for d in donors], columns=["donor_id", "disease_group"])
        for column in ("age", "sex", "apoe_e4_count", "pmd_hours"):
            frame[column] = ""
    else:
        raise HTTPException(404, kind)
    return Response(frame.to_csv(index=False), media_type="text/csv", headers={"Content-Disposition": f'attachment; filename="{kind}_template.csv"'})


@router.post("/upload/{kind}")
async def upload(kind: str, file: UploadFile = File(...)) -> dict:
    if kind not in UPLOADS:
        raise HTTPException(404, kind)
    project = get_state().project
    content = await file.read()
    frame = pd.read_csv(io.BytesIO(content), dtype=str, keep_default_na=False)
    target = project.input(kind)
    staged = target.with_name(f".{target.name}.upload")
    staged.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(staged, index=False)
    problems = _validate(kind, staged, frame, project)
    if problems:
        staged.unlink()
        raise HTTPException(400, f"The {UPLOADS[kind]} was not saved: {problems}")
    if target.exists():
        target.rename(target.with_name(f"{target.stem}.replaced-{time.strftime('%Y%m%d-%H%M%S')}{target.suffix}"))
    staged.rename(target)
    if kind == "tma_layout" and not project.config["groups"]:
        codes = [c for c in dict.fromkeys(frame["disease_group"]) if c]
        project = save_project(project, {"groups": [{"code": c, "label": c} for c in codes]})
    open_project(project.root)
    return {"saved": project.relative(target), "rows": len(frame)}


def _validate(kind: str, path: Path, frame: pd.DataFrame, project: Project) -> str:
    if kind == "tma_layout":
        try:
            layout = read_layout(path)
        except ValueError as error:
            return str(error)
        manifest = project.input("core_manifest")
        known = {(r["tma"], r["core_label"]) for r in read_csv(manifest)} if manifest.exists() else set()
        missing = sorted(known - set(layout))
        return f"{len(missing)} core positions have no row, e.g. {missing[:3]}" if missing else ""
    if kind == "donor_metadata":
        missing = [c for c in ("donor_id", "disease_group") if c not in frame]
        return f"missing columns: {', '.join(missing)}" if missing else ""
    missing = [c for c in ("slide_path", "tma", "stain") if c not in frame]
    return f"missing columns: {', '.join(missing)}" if missing else ""
