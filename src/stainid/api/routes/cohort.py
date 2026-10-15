from __future__ import annotations

import re

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException

from stainid.api import placeholders, pyramid, render
from stainid.api.state import ProjectState, get_state
from stainid.api.util import records
from stainid.outputs import objects

router = APIRouter(tags=["cohort"])
CORE_COLUMNS = ["core_id", "tma", "core_label", "donor_id", "sample_region_id", "region", "disease_group", "technical_replicate"]
GRID_THUMB = 256
OBJECT_COLUMNS = ["x", "y", "radius_px", "model_class", "probability", "area_um2"]


def _tma_order(tma: str) -> tuple:
    return (len(tma), tma)


def _position(label: str) -> tuple[int, str]:
    """'B-12' -> (12, 'B'): reading order is row by row."""
    column, _, row = str(label).partition("-")
    return (int(row) if row.isdigit() else 0, column)


def _cores(state: ProjectState) -> pd.DataFrame:
    """One row per core and stain that has an image (else per analysed core and stain)."""
    source = state.core_images if not state.core_images.empty else state.analysis_tiles
    if source.empty:
        return source
    source = source.drop_duplicates(["core_id", "stain"])
    missing = [c for c in CORE_COLUMNS if c not in source]
    if missing and not state.analysis_tiles.empty:  # an older core list: the rest of the core's details come from the tiles
        details = state.analysis_tiles.drop_duplicates("core_id")[["core_id", *[c for c in missing if c in state.analysis_tiles]]]
        source = source.merge(details, on="core_id", how="left")
    return source


@router.get("/cohort")
def cohort() -> dict:
    """The TMAs and stains there are pictures of (cheap: no pipeline outputs are looked at)."""
    cores = _cores(get_state())
    if cores.empty:
        return {"tmas": [], "stains": []}
    tmas = [{"tma": tma, "cores": int(group.core_id.nunique()), "stains": sorted(group.stain.unique())}
            for tma, group in sorted(cores.groupby(cores.tma.astype(str)), key=lambda item: _tma_order(item[0]))]
    return {"tmas": tmas, "stains": sorted(cores.stain.unique())}


@router.get("/cores")
def cores(tma: str | None = None) -> list[dict]:
    tiles = get_state().analysis_tiles
    if tma:
        tiles = tiles[tiles.tma.astype(str) == tma]
    table = tiles.groupby(CORE_COLUMNS, dropna=False).stain.apply(lambda s: sorted(set(s))).reset_index(name="stains")
    return records(table)


@router.get("/tmas/{tma}/layout")
def layout(tma: str) -> list[dict]:
    frame = get_state().layout
    return records(frame[frame.tma.astype(str) == tma])


def _detection(state: ProjectState, tiles: pd.DataFrame, stain: str) -> tuple[str, int, int]:
    """Whether detection has run on these tiles of one core: ('done' | 'partial' | 'not_run', tiles done, tiles)."""
    if tiles.empty:
        return "not_run", 0, 0
    if stain == "NeuN":
        folder = state.project.output("neun") / "tiles"
        done = sum((folder / f"{t}_objects.csv").exists() for t in tiles.tile_id)
    else:
        done = len(tiles) if (state.project.output("fields") / "parts" / f"{tiles.core_id.iloc[0]}_{stain}_objects.csv").exists() else 0
    return ("done" if done == len(tiles) else "partial" if done else "not_run"), int(done), int(len(tiles))


def _quick_detection(state: ProjectState, tiles: pd.DataFrame, stain: str) -> str:
    """The same, from at most two file checks per core (for whole grids)."""
    if tiles.empty:
        return "not_run"
    if stain != "NeuN":
        return _detection(state, tiles, stain)[0]
    folder = state.project.output("neun") / "tiles"
    first, last = ((folder / f"{t}_objects.csv").exists() for t in (tiles.tile_id.iloc[0], tiles.tile_id.iloc[-1]))
    return "done" if first and last else "partial" if first or last else "not_run"


@router.get("/tmas/{tma}/grid")
def grid(tma: str, stain: str, prepare: bool = True) -> dict:
    """Every position of one TMA for one stain: the core there, whether its thumbnail can be asked for now (`thumb`;
    the ones that need their whole image decoded first are queued when `prepare`), a tiny stand-in picture until the
    thumbnail is made (`placeholder`), and whether detection has run on it."""
    state = get_state()
    cores = _cores(state)
    cores = cores[(cores.tma.astype(str) == tma) & (cores.stain == stain)] if not cores.empty else cores
    by_label = {str(r["core_label"]): r for r in cores.to_dict("records")}
    layout = state.layout[state.layout.tma.astype(str) == tma] if not state.layout.empty else state.layout
    positions = {str(r["core_label"]): r for r in layout.to_dict("records")}
    labels = sorted(set(positions) | set(by_label), key=_position)
    tiles = state.analysis_tiles
    tiles = tiles[(tiles.tma.astype(str) == tma) & (tiles.stain == stain)] if not tiles.empty else tiles
    tiles_by_core = dict(list(tiles.groupby("core_id"))) if not tiles.empty else {}
    pictured = not state.core_images.empty
    cells, waiting, stand_ins = [], 0, None
    for label in labels:
        core, place = by_label.get(label), positions.get(label, {})
        cell = {"core_label": label, "tissue_control": _text(place.get("tissue_control")), "core_id": None}
        if core is not None:
            core_id = str(core["core_id"])
            thumb = pyramid.thumbnail_state(state, core_id, stain, GRID_THUMB) if pictured else None
            if thumb == "slow" and prepare:
                pyramid.prepare_thumbnail(state, core_id, stain)
                waiting += 1
            if thumb != "ready" and stand_ins is None:
                stand_ins = placeholders.for_tma(state.project, tma, stain)
            cell.update({k: _text(core.get(k, place.get(k))) for k in ("core_id", "donor_id", "region", "disease_group", "technical_replicate")},
                        thumb=thumb in ("ready", "quick"), placeholder=(stand_ins or {}).get(label) if thumb != "ready" else None,
                        detection=_quick_detection(state, tiles_by_core.get(core_id, pd.DataFrame()), stain))
        cells.append(cell)
    columns = sorted({_position(c["core_label"])[1] for c in cells}, key=lambda c: (len(c), c))
    rows = sorted({_position(c["core_label"])[0] for c in cells})
    done = sum(c.get("detection") == "done" for c in cells)
    return {"tma": tma, "stain": stain, "columns": columns, "rows": rows, "cells": cells, "thumbs_waiting": waiting,
            "detection": {"done": done, "total": sum(c["core_id"] is not None for c in cells)}}


def _text(value) -> str | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return str(int(value)) if isinstance(value, float) and value.is_integer() else str(value)


def _neighbours(state: ProjectState, core_id: str, tma: str) -> dict:
    cores = _cores(state)
    ids = sorted(set(cores[cores.tma.astype(str) == tma].core_id), key=lambda c: _position(re.sub(r"^.*?_", "", c)))
    at = ids.index(core_id) if core_id in ids else -1
    return {"previous": ids[at - 1] if at > 0 else None, "next": ids[at + 1] if 0 <= at < len(ids) - 1 else None}


@router.get("/cores/{core_id}")
def core(core_id: str) -> dict:
    state = get_state()
    tiles = state.analysis_tiles
    tiles = tiles[tiles.core_id == core_id] if not tiles.empty else tiles
    images = state.core_images[state.core_images.core_id == core_id] if not state.core_images.empty else state.core_images
    if tiles.empty and images.empty:
        raise HTTPException(404, f"unknown core {core_id}")
    first = (tiles if not tiles.empty else images).iloc[0]
    image_columns = [c for c in ("stain", "native_width_px", "native_height_px", "pixel_width_um", "tissue_fraction", "tissue_status") if c in images]
    return {
        **{k: _text(first.get(k)) for k in CORE_COLUMNS},
        "images": records(images[image_columns].sort_values("stain")) if not images.empty else [],
        "neighbours": _neighbours(state, core_id, str(first["tma"])),
    }


@router.get("/cores/{core_id}/{stain}/objects")
def core_objects(core_id: str, stain: str) -> dict:
    """Everything detection found in one core image, in core-image pixels (as rows of `columns`), and the tiles it ran on."""
    state = get_state()
    tiles = state.analysis_tiles
    tiles = tiles[(tiles.core_id == core_id) & (tiles.stain == stain)] if not tiles.empty else tiles
    status, done, total = _detection(state, tiles, stain)
    rows = []
    found = objects(state.project, tiles, stain) if done else pd.DataFrame()
    if not found.empty:
        origin = {t["tile_id"]: render.crop_origin(t, state.project.context_px) for t in tiles.to_dict("records")}
        found = found.assign(x=(found.x + found.tile_id.map(lambda t: origin[t][0])).round(1),
                             y=(found.y + found.tile_id.map(lambda t: origin[t][1])).round(1)).drop_duplicates(["x", "y"])
        found = found.assign(radius_px=found.radius_px.round(1), probability=found.probability.round(3), area_um2=found.area_um2.round(1))
        rows = [list(r.values()) for r in records(found[OBJECT_COLUMNS])]
    boxes = tiles[["x_px", "y_px", "width_px", "height_px"]].astype(int).values.tolist() if not tiles.empty else []
    return {"status": status, "tiles_done": done, "tiles_total": total, "columns": OBJECT_COLUMNS, "rows": rows, "tiles": boxes}


@router.get("/fields/{tile_id}/objects")
def field_objects(tile_id: str) -> dict:
    state = get_state()
    tiles = state.tiles[state.tiles.tile_id == tile_id]
    if tiles.empty:
        raise HTTPException(404, f"unknown field {tile_id}")
    stain = tiles.iloc[0].stain
    found = objects(state.project, tiles, stain)
    return {"tile_id": tile_id, "stain": stain, "objects": records(found.drop(columns=["tile_id", "stain"], errors="ignore"))}
