"""Where a label's position is measured, and the same position on the whole core image.

A label names a tile and a position. On a tile of the project (`sampling.tiles.known_tiles`) the position is in the
field picture: the tile with a margin of `context_px` around it, the margin clipped where the core image ends
(`field_origin`), as in every pipeline output, candidate pool and labelling page. Three kinds of labels differ:

- a pilot field of an earlier study (`<reviews>/fields/<tile_id>.png`, no margin): pixels of that picture (`PILOT`);
- a core id instead of a tile id (the earliest reference labels were made on whole cores): pixels of the core image
  (`CORE`);
- labels of the `TILE_FRAME` sources, measured from the tile's own corner without the margin: the CNN outlier
  adjudication (its object table subtracted the margin). `in_field_frame` moves those into the field picture, so they
  sit on their objects and line up with the candidates, the AI's calls and other people's labels of the same objects.
  The label log keeps them as they were imported.
"""
from __future__ import annotations

import pandas as pd

FIELD, PILOT, CORE = "field", "pilot", "core"
TILE_FRAME = frozenset({"import: CNN outlier adjudication"})


def field_origin(tile: dict, context_px: int) -> tuple[int, int]:
    """Where the field picture of a tile starts on its core image."""
    return max(0, int(tile["x_px"]) - context_px), max(0, int(tile["y_px"]) - context_px)


def margin(tile: dict, context_px: int) -> tuple[int, int]:
    """How far the tile's own corner is from the corner of its field picture (the margin, smaller at the core's edge)."""
    x0, y0 = field_origin(tile, context_px)
    return int(tile["x_px"]) - x0, int(tile["y_px"]) - y0


def in_field_frame(labels: pd.DataFrame, tiles: pd.DataFrame, context_px: int) -> pd.DataFrame:
    """The labels with every position on a project tile in field-picture pixels (a copy when anything moves)."""
    if labels.empty or "source" not in labels or tiles.empty:
        return labels
    moving = labels.source.isin(TILE_FRAME) & labels.tile_id.isin(set(tiles.tile_id)) & labels.x.notna()
    if not moving.any():
        return labels
    boxes = tiles.drop_duplicates("tile_id").set_index("tile_id")
    shift = pd.DataFrame([margin(boxes.loc[t], context_px) for t in labels.tile_id[moving]], index=labels.index[moving], columns=["dx", "dy"])
    out = labels.copy()
    out.loc[moving, "x"] = out.x[moving].astype(float) + shift.dx
    out.loc[moving, "y"] = out.y[moving].astype(float) + shift.dy
    return out


def core_position(tile: dict | None, x: float, y: float, context_px: int) -> tuple[float, float]:
    """A field-picture position (or, without a tile, a core-image position) on the core image."""
    if tile is None:
        return float(x), float(y)
    x0, y0 = field_origin(tile, context_px)
    return x0 + float(x), y0 + float(y)
