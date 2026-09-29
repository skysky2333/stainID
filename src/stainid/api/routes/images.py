from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from stainid.api import render
from stainid.api.state import get_state

router = APIRouter(tags=["images"])
CACHE = {"Cache-Control": "public, max-age=86400"}
OVERLAYS = {"exclusions": render.exclusion_overlay, "dab": render.dab_overlay, "threads": render.thread_overlay}


@router.get("/images/cores/{core_id}/{stain}.jpg")
def core_thumbnail(core_id: str, stain: str, size: int = 512) -> FileResponse:
    try:
        return FileResponse(render.core_thumbnail(get_state(), core_id, stain, min(size, 2048)), headers=CACHE)
    except (KeyError, FileNotFoundError) as error:
        raise HTTPException(404, str(error))


@router.get("/images/fields/{tile_id}.jpg")
def field_image(tile_id: str) -> FileResponse:
    try:
        return FileResponse(render.field_jpeg(get_state(), tile_id), headers=CACHE)
    except KeyError as error:
        raise HTTPException(404, str(error))


@router.get("/images/fields/{tile_id}/{layer}.png")
def field_overlay(tile_id: str, layer: str) -> FileResponse:
    if layer not in OVERLAYS:
        raise HTTPException(404, f"unknown layer {layer}")
    return FileResponse(OVERLAYS[layer](get_state(), tile_id), headers=CACHE)


@router.get("/fields/{tile_id}/outlines")
def outlines(tile_id: str) -> list[dict]:
    return render.mask_outlines(get_state(), tile_id)
