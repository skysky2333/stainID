from __future__ import annotations

import cv2
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, Response

from stainid.api import render
from stainid.api.state import get_state
from stainid.api.util import records
from stainid.outputs import objects

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


def _box(core_id: str, stain: str, x: int, y: int, size: int) -> tuple[int, int, int]:
    try:
        return render.region_box(get_state(), core_id, stain, x, y, min(size, 3000))
    except KeyError as error:
        raise HTTPException(404, str(error))


@router.get("/images/cores/{core_id}/{stain}/region.jpg")
def core_region(core_id: str, stain: str, x: int, y: int, size: int = 1600) -> Response:
    x0, y0, size = _box(core_id, stain, x, y, size)
    rgb = render.core_region(get_state(), core_id, stain, x0, y0, size)
    ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
    return Response(encoded.tobytes(), media_type="image/jpeg", headers=CACHE)


@router.get("/images/cores/{core_id}/{stain}/region_dab.png")
def core_region_dab(core_id: str, stain: str, x: int, y: int, size: int = 1600) -> Response:
    x0, y0, size = _box(core_id, stain, x, y, size)
    state = get_state()
    rgba = render.region_dab(state, core_id, stain, render.core_region(state, core_id, stain, x0, y0, size))
    ok, encoded = cv2.imencode(".png", cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
    return Response(encoded.tobytes(), media_type="image/png", headers=CACHE)


@router.get("/cores/{core_id}/{stain}/region")
def core_region_objects(core_id: str, stain: str, x: int, y: int, size: int = 1600) -> dict:
    """Where the region lies in the core image, and the detections inside it (region pixel coordinates)."""
    x0, y0, size = _box(core_id, stain, x, y, size)
    state = get_state()
    tiles = state.tiles[(state.tiles.core_id == core_id) & (state.tiles.stain == stain)]
    found = objects(state.project, tiles, stain)
    out = []
    if not found.empty:
        origin = {t["tile_id"]: render.crop_origin(t, state.project.context_px) for t in tiles.to_dict("records")}
        found = found.assign(x=found.x + found.tile_id.map(lambda t: origin[t][0]) - x0, y=found.y + found.tile_id.map(lambda t: origin[t][1]) - y0)
        found = found[found.x.between(0, size) & found.y.between(0, size)].drop_duplicates(["x", "y"])
        out = records(found[["x", "y", "radius_px", "model_class", "probability", "area_um2"]])
    return {"x0": x0, "y0": y0, "size": size, "objects": out}
