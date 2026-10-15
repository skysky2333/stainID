from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

from stainid.api.state import get_state, reload_project
from stainid.slides.dearray import slide_slug
from stainid.slides.overview import slide_picture
from stainid.slides.table import find_slides, guess_prefix, read_slides, save_slides, slide_rows
from stainid.tables import read_csv

router = APIRouter(tags=["slides"])


class SlideRows(BaseModel):
    rows: list[dict[str, str | bool | dict | list | None]]


def _tma_key(tma: str) -> tuple:
    return (0, int(tma), "") if tma.isdigit() else (1, 0, tma)


def known_tmas(project) -> list[str]:
    """The TMAs the project already knows: in the slides table, on a saved map or in the core table."""
    found = {r.get("tma", "") for r in read_slides(project.input("slides_table"))}
    for key in ("tma_layout", "core_manifest"):
        path = project.input(key)
        if path.exists():
            found |= {r.get("tma", "") for r in read_csv(path)}
    return sorted((t for t in found if t), key=_tma_key)


def _prefix(project, folder: Path) -> str:
    """The core-name prefix to read file names with: the project's, or the one the file names suggest while it is the default."""
    if project.tma_prefix != "TMA-":
        return project.tma_prefix
    return guess_prefix(find_slides(folder), list(project.stains)) or project.tma_prefix


@router.get("/slides")
def slides() -> dict:
    project = get_state().project
    folder = project.input("slides_dir")
    stains = list(project.stains)
    rows = slide_rows(folder, project.input("slides_table"), stains, _prefix(project, folder))
    return {"folder": project.relative(folder), "folder_path": str(folder), "folder_exists": folder.is_dir(), "stains": stains,
            "saved": project.input("slides_table").exists(), "rows": rows, "tmas": known_tmas(project),
            "missing": [r for r in read_slides(project.input("slides_table")) if not Path(r["slide_path"]).exists()],
            "tma_prefix": project.tma_prefix, "prefix_guess": guess_prefix(find_slides(folder), stains),
            "table": project.relative(project.input("slides_table"))}


@router.put("/slides")
def save(request: SlideRows) -> dict:
    project = get_state().project
    try:
        save_slides(project.input("slides_table"), [{k: v for k, v in r.items() if isinstance(v, str)} for r in request.rows], list(project.stains))
    except ValueError as error:
        raise HTTPException(400, str(error))
    reload_project(project.root)
    return slides()


@router.get("/slides/picture")
def picture(path: str, size: int = 320) -> Response:
    """A small picture of one scan (its overview and label, or the Find cores preview), for recognising it by its label."""
    state = get_state()
    project = state.project
    known = {str(p) for p in find_slides(project.input("slides_dir"))} | {r["slide_path"] for r in read_slides(project.input("slides_table"))}
    if path not in known:
        raise HTTPException(404, "not a scan of this project")
    slide = Path(path)
    size = max(64, min(size, 800))
    stamp = int(slide.stat().st_mtime) if slide.exists() else 0
    cached = state.cache_dir("slide_pictures") / f"{slide_slug(slide)}-{stamp}-{size}.jpg"
    if not cached.exists():
        image = slide_picture(slide, project.output("qc"), size)
        if image is None:
            raise HTTPException(404, "this scan has no picture that can be read quickly")
        ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
        temporary = cached.with_suffix(".tmp")
        temporary.write_bytes(encoded.tobytes())
        temporary.replace(cached)
    return FileResponse(cached, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=3600"})


@router.get("/grids")
def grids() -> list[str]:
    folder = get_state().project.output("qc") / "grids"
    return sorted(p.name for p in folder.glob("*.jpg")) if folder.exists() else []


@lru_cache(maxsize=64)
def _smaller(path: str, mtime: float, width: int) -> bytes:
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(404, "this picture cannot be read")
    if image.shape[1] > width:
        image = cv2.resize(image, (width, round(image.shape[0] * width / image.shape[1])), interpolation=cv2.INTER_AREA)
    return cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()


@router.get("/grids/{name}")
def grid_image(name: str, width: int = 0) -> Response:
    """A Find cores picture; with `width` a smaller copy (the full pictures are several MB each)."""
    path = get_state().project.output("qc") / "grids" / Path(name).name
    if not path.exists():
        raise HTTPException(404, name)
    if not width:
        return FileResponse(path)
    picture = _smaller(str(path), path.stat().st_mtime, max(160, min(width, 2400)))
    return Response(picture, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=3600"})
