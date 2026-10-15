from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, Response

from stainid.api import pyramid, render
from stainid.api.state import get_state

router = APIRouter(tags=["images"])
CACHE = {"Cache-Control": "public, max-age=86400"}
OVERLAYS = {"exclusions": render.exclusion_overlay, "dab": render.dab_overlay, "threads": render.thread_overlay}


@router.get("/images/cores/{core_id}/{stain}.jpg")
def core_thumbnail(core_id: str, stain: str, size: int = 512) -> FileResponse:
    try:
        return FileResponse(render.core_thumbnail(get_state(), core_id, stain, max(32, min(size, pyramid.THUMB_BASE))), headers=CACHE)
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


IMMUTABLE = {"Cache-Control": "public, max-age=31536000, immutable"}  # tile addresses carry the image version (?v=)


@router.get("/cores/{core_id}/{stain}/picture")
def core_picture(core_id: str, stain: str) -> dict:
    """The deep-zoom levels of a core image once it is prepared; until then where its preparation stands (asking
    again keeps it queued)."""
    try:
        return pyramid.picture(get_state(), core_id, stain)
    except (KeyError, FileNotFoundError) as error:
        raise HTTPException(404, str(error).strip("'"))


@router.get("/images/cores/{core_id}/{stain}/tiles/{level}/{col}_{row}.jpg")
def core_tile(core_id: str, stain: str, level: int, col: int, row: int, v: str = "") -> Response:
    try:
        data = pyramid.tile(get_state(), core_id, stain, level, col, row)
    except KeyError as error:
        raise HTTPException(404, str(error).strip("'"))
    return Response(data, media_type="image/jpeg", headers=IMMUTABLE if v else CACHE)


@router.get("/images/cores/{core_id}/{stain}/tiles/{level}/{col}_{row}_dab.png")
def core_tile_dab(core_id: str, stain: str, level: int, col: int, row: int, v: str = "") -> Response:
    try:
        data = pyramid.dab_tile(get_state(), core_id, stain, level, col, row)
    except KeyError as error:
        raise HTTPException(404, str(error).strip("'"))
    return Response(data, media_type="image/png", headers=IMMUTABLE if v else CACHE)
