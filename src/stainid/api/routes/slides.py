from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from stainid.api.state import get_state, open_project
from stainid.slides.table import save_slides, scan_slides

router = APIRouter(tags=["slides"])


class SlideRows(BaseModel):
    rows: list[dict[str, str]]


@router.get("/slides")
def slides() -> dict:
    project = get_state().project
    folder = project.input("slides_dir")
    return {"folder": project.relative(folder), "folder_exists": folder.is_dir(), "stains": list(project.stains),
            "saved": project.input("slides_table").exists(), "rows": scan_slides(folder, project.input("slides_table"), list(project.stains))}


@router.put("/slides")
def save(request: SlideRows) -> dict:
    project = get_state().project
    try:
        save_slides(project.input("slides_table"), request.rows, list(project.stains))
    except ValueError as error:
        raise HTTPException(400, str(error))
    open_project(project.root)
    return slides()


@router.get("/grids")
def grids() -> list[str]:
    folder = get_state().project.output("qc") / "grids"
    return sorted(p.name for p in folder.glob("*.jpg")) if folder.exists() else []


@router.get("/grids/{name}")
def grid_image(name: str) -> FileResponse:
    path = get_state().project.output("qc") / "grids" / Path(name).name
    if not path.exists():
        raise HTTPException(404, name)
    return FileResponse(path)
