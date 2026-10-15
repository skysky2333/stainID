"""The candidates of the segmentation check's fields, and what the pipeline did with them.

Whole-core detection keeps only the objects it counted for 6E10 and AT8, so the candidates it looked at and rejected
are found again here, once per field of an outline session, with the detection step's own segmentation
(`training.candidates.segment_with_masks`). They are a result of the check, not a picture for viewing:
`<tables>/segmentation_benchmark/proposals/<stain>/<field>.npz` holds the label images (`layers`) and, per
candidate, its key into them (`layer`, `label`), its centre (`x`, `y`, field pixels) and its kind (`source`), with
the stain threshold and `VERSION` it was made with (made again when either changes).

Which candidates the pipeline counted is decided when scoring, from the detection results as they are then
(`detections`), so scores follow a new detection run without finding the candidates again.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from stainid.outputs import objects as pipeline_objects
from stainid.project import Project
from stainid.tables import read_csv, save_npz_atomic

VERSION = 1
MATCH_PX = 4.0  # a counted object is the candidate whose centre lies this close (as the label pool matches them)
SOURCE = ("source_kind", "candidate_source", "candidate_class")


def crop_origin(tile: dict, context_px: int) -> tuple[int, int]:
    """Where a field's picture (its square and margin) starts in the core image, as the detection step cuts it."""
    return max(0, int(tile["x_px"]) - context_px), max(0, int(tile["y_px"]) - context_px)


def folder(project: Project) -> Path:
    return project.output("tables") / "segmentation_benchmark"


def path(project: Project, stain: str, tile_id: str) -> Path:
    return folder(project) / "proposals" / stain / f"{tile_id}.npz"


def _threshold(loader, tile: dict) -> float:
    return float(loader.calibration[(tile["tma"], tile["stain"])])


def current(project: Project, tile: dict, loader) -> bool:
    """Whether the field's saved candidates were made with this version and the slide's current stain threshold."""
    target = path(project, tile["stain"], tile["tile_id"])
    if not target.exists():
        return False
    with np.load(target) as saved:
        return int(saved["version"]) == VERSION and np.isclose(float(saved["threshold"]), _threshold(loader, tile))


def compute(project: Project, tile: dict, loader) -> int:
    """Find and save every candidate of one field (with its margin) and its mask; returns how many."""
    from stainid.training.candidates import segment_with_masks

    field = loader.load(tile)
    rows, layers = segment_with_masks(tile["stain"], loader, field)
    stack = np.stack([np.asarray(layer) for layer in layers]).astype(np.uint32)
    if stack.max(initial=0) < 65536:
        stack = stack.astype(np.uint16)
    inner = np.array([field.inner[0].start, field.inner[1].start, field.inner[0].stop, field.inner[1].stop])
    target = path(project, tile["stain"], tile["tile_id"])
    target.parent.mkdir(parents=True, exist_ok=True)
    save_npz_atomic(target, layers=stack, layer=np.array([r["mask_layer"] for r in rows], np.int32),
                    label=np.array([r["mask_label"] for r in rows], np.int64),
                    x=np.array([float(r["centroid_x_px"]) for r in rows]), y=np.array([float(r["centroid_y_px"]) for r in rows]),
                    source=np.array([next((str(r[k]) for k in SOURCE if r.get(k)), "") for r in rows]),
                    inner=inner, threshold=np.array(field.threshold), version=np.array(VERSION), pixel_size_um=np.array(field.pixel_size_um))
    return len(rows)


def load(project: Project, stain: str, tile_id: str) -> dict | None:
    """A field's saved candidates (`compute`), None when they have not been found yet."""
    target = path(project, stain, tile_id)
    if not target.exists():
        return None
    with np.load(target) as saved:
        out = {k: saved[k] for k in saved.files}
    out["box"] = (slice(int(out["inner"][0]), int(out["inner"][2])), slice(int(out["inner"][1]), int(out["inner"][3])))
    return out


def _overlaps(tile: dict, other: dict, context_px: int, size: tuple[int, int]) -> bool:
    x0, y0 = crop_origin(tile, context_px)
    ox, oy, ow, oh = (int(other[k]) for k in ("x_px", "y_px", "width_px", "height_px"))
    return ox < x0 + size[1] and ox + ow > x0 and oy < y0 + size[0] and oy + oh > y0


def detections(project: Project, tile: dict, core_tiles: pd.DataFrame, size: tuple[int, int]) -> pd.DataFrame:
    """The objects the pipeline counted in this field's picture (its own and its neighbours' squares, which overlap its
    margin), in this field's pixels: x, y, radius_px, tile_id (the square that counted it)."""
    near = core_tiles[[_overlaps(tile, t, project.context_px, size) for t in core_tiles.to_dict("records")]]
    found = pipeline_objects(project, near, tile["stain"])
    if found.empty:
        return pd.DataFrame(columns=["x", "y", "radius_px", "tile_id"])
    if "model_class" in found:
        found = found[found.model_class != "rejected"]
    origins = {t["tile_id"]: crop_origin(t, project.context_px) for t in near.to_dict("records")}
    x0, y0 = crop_origin(tile, project.context_px)
    dx = found.tile_id.map(lambda t: origins[t][0] - x0)
    dy = found.tile_id.map(lambda t: origins[t][1] - y0)
    moved = pd.DataFrame({"x": found.x.astype(float) + dx, "y": found.y.astype(float) + dy, "radius_px": found.radius_px.astype(float),
                          "tile_id": found.tile_id})
    seen = moved.x.between(0, size[1], inclusive="left") & moved.y.between(0, size[0], inclusive="left")  # only those in this picture
    return moved[seen].reset_index(drop=True)


def counted(saved: dict, found: pd.DataFrame) -> tuple[list[int], pd.DataFrame]:
    """Which saved candidates the pipeline counted (each counted object goes to the nearest candidate centre within
    `MATCH_PX`), and the counted objects no candidate explains (found by an older segmentation: the candidates are out
    of date for them)."""
    x, y = saved["x"], saved["y"]
    chosen: list[int] = []
    unexplained = []
    for row in found.itertuples():
        d = np.hypot(x - row.x, y - row.y) if len(x) else np.array([])
        i = int(d.argmin()) if len(d) else -1
        if i >= 0 and d[i] <= MATCH_PX and i not in chosen:
            chosen.append(i)
        else:
            unexplained.append(row.Index)
    return chosen, found.loc[unexplained]


def outlines(project: Project, tile: dict) -> np.ndarray | None:
    """The Segment Anything outlines of the field's own counted objects (the Outline objects step), as a label image in
    field pixels; None when the step has not outlined this field."""
    stain = tile["stain"]
    labels = project.output("masks") / stain / "labels" / f"{tile['tile_id']}.npz"
    parts = project.output("masks") / stain / "parts" / f"{tile['core_id']}_{stain}_objects.csv"
    if not labels.exists() or not parts.exists():
        return None
    kept = {int(float(r["label"])) for r in read_csv(parts) if r.get("tile_id") == tile["tile_id"] and r.get("mask_status") == "ok" and r.get("label")}
    with np.load(labels) as saved:
        image = saved["labels"]
    return np.where(np.isin(image, list(kept)), image, 0)
