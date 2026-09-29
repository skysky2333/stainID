from __future__ import annotations

from fastapi import APIRouter, HTTPException

from stainid.api.state import get_state
from stainid.api.util import records
from stainid.outputs import objects

router = APIRouter(tags=["cohort"])
CORE_COLUMNS = ["core_id", "tma", "core_label", "donor_id", "sample_region_id", "region", "disease_group", "technical_replicate"]


@router.get("/cores")
def cores(tma: str | None = None) -> list[dict]:
    tiles = get_state().tiles
    if tma:
        tiles = tiles[tiles.tma.astype(str) == tma]
    table = tiles.groupby(CORE_COLUMNS, dropna=False).stain.apply(lambda s: sorted(set(s))).reset_index(name="stains")
    return records(table)


@router.get("/tmas/{tma}/layout")
def layout(tma: str) -> list[dict]:
    frame = get_state().layout
    return records(frame[frame.tma.astype(str) == tma])


@router.get("/cores/{core_id}")
def core(core_id: str) -> dict:
    state = get_state()
    tiles = state.tiles[state.tiles.core_id == core_id]
    if tiles.empty:
        raise HTTPException(404, f"unknown core {core_id}")
    images = state.core_images[state.core_images.core_id == core_id]
    first = tiles.iloc[0]
    return {
        **{k: (None if str(first[k]) == "nan" else str(first[k])) for k in CORE_COLUMNS},
        "images": records(images[["stain", "native_width_px", "native_height_px", "tissue_fraction", "tissue_status"]]) if not images.empty else [],
        "fields": records(tiles[["tile_id", "stain", "selection_order", "x_px", "y_px", "width_px", "height_px", "nested_sample"]]),
    }


@router.get("/fields/{tile_id}/objects")
def field_objects(tile_id: str) -> dict:
    state = get_state()
    tiles = state.tiles[state.tiles.tile_id == tile_id]
    if tiles.empty:
        raise HTTPException(404, f"unknown field {tile_id}")
    stain = tiles.iloc[0].stain
    found = objects(state.project, tiles, stain)
    return {"tile_id": tile_id, "stain": stain, "objects": records(found.drop(columns=["tile_id", "stain"], errors="ignore"))}
