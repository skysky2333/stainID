"""The Projects page: every project this server knows, with its status; open, create (optionally with another
project's settings) and remove from the list. These routes belong to the server, not to one project."""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from stainid.api import registry, state
from stainid.project import CONFIG_NAME, DEFAULTS, Project, load_project, save_project

router = APIRouter(prefix="/projects", tags=["projects"])

# What "use the settings of" copies from another project, in the words of the Projects page.
COPYABLE = {
    "groups": ("Diagnostic groups", ("groups",)),
    "stains": ("Stains", ("stains",)),
    "models": ("Models and their settings", ("models", "acceptance", "precision")),
    "naming": ("TMA layout and naming", ("tma", "pixel_size_um", "donor_ids", "sample_region_ids")),
}


class OpenRequest(BaseModel):
    path: str


class CreateRequest(BaseModel):
    path: str
    name: str = ""
    copy_from: str = ""  # a project id: start with (some of) its settings
    settings: list[str] = list(COPYABLE)  # which (keys of COPYABLE)


class FolderSetting(BaseModel):
    path: str


def known() -> list[registry.Entry]:
    """The registry, plus the projects open in this server and the one it was started with (`--project`)."""
    start = Path(os.environ.get("STAINID_PROJECT", "."))
    extra = [s.root for s in state.open_states() if s.project.is_configured] + ([start] if (start / CONFIG_NAME).exists() else [])
    return registry.entries(extra)


def _entry(project_id: str) -> registry.Entry:
    entry = next((e for e in known() if e.id == project_id), None)
    if entry is None:
        raise HTTPException(404, "This project is not on the list of this server")
    return entry


def describe(entry: registry.Entry) -> dict:
    return {**registry.as_dict(entry), "status": state.project_status(Path(entry.path))}


@router.get("")
def projects() -> dict:
    """The projects list, most recently opened first, and the projects folder setting."""
    folder = registry.projects_dir()
    return {"projects": [describe(e) for e in known()], "default": state.default_id(), "projects_dir": str(folder) if folder else "",
            "projects_dir_from": "command line" if os.environ.get(registry.PROJECTS_DIR_ENV) else "setting" if folder else "",
            "copyable": {key: label for key, (label, _) in COPYABLE.items()}}


@router.post("/open")
def open_folder(request: OpenRequest) -> dict:
    path = Path(request.path).expanduser()
    if not (path / CONFIG_NAME).exists():
        raise HTTPException(404, f"{path} is not a stainID project (it has no {CONFIG_NAME}). Use “New study” to start one there.")
    opened = state.open_project(path)
    return describe(_entry(opened.id))


def copied_settings(source: Project, keys: list[str]) -> dict:
    """The parts of another project's settings to start a new study with. Model files stay where they are: their
    locations are written out in full, so the new study uses the same files."""
    out: dict = {}
    for key in keys:
        if key not in COPYABLE:
            raise HTTPException(400, f"unknown setting group {key}")
        for setting in COPYABLE[key][1]:
            out[setting] = source.config[setting]
    if "models" in out:
        out["models"] = {name: str(source.model(name)) for name in out["models"]}
    return out


@router.post("/create")
def create(request: CreateRequest) -> dict:
    path = Path(request.path).expanduser()
    if not request.path.strip():
        raise HTTPException(400, "Choose a folder for the new study")
    if (path / CONFIG_NAME).exists():
        raise HTTPException(409, f"{path} already holds a stainID project; open it instead.")
    if path.exists() and not path.is_dir():
        raise HTTPException(400, f"{path} is a file, not a folder")
    changes = {"name": request.name.strip() or path.name}
    if request.copy_from:
        source = _entry(request.copy_from)
        changes = {**copied_settings(load_project(source.path), request.settings), **changes}
    save_project(Project(root=path.resolve(), config=DEFAULTS), changes)
    opened = state.open_project(path)
    return describe(_entry(opened.id))


@router.delete("/{project_id}")
def remove(project_id: str) -> dict:
    """Take a project off the list. Its folder and files stay as they are; steps still running carry on."""
    entry = _entry(project_id)
    registry.forget(entry.path)
    still_running = not state.close_project(project_id)
    return {"removed": entry.id, "still_running": still_running}


@router.put("/settings")
def settings(request: FolderSetting) -> dict:
    """The projects folder: every subfolder with a stainid.yaml is listed."""
    if request.path.strip() and not Path(request.path).expanduser().is_dir():
        raise HTTPException(400, f"{request.path} is not a folder")
    registry.set_projects_dir(request.path)
    return projects()
