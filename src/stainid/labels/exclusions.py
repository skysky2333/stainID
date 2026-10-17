"""Areas annotators mark as bad on a field (the field tool's exclusion brush, key E): one `exclude` label per person and
field (`exclude:<tile_id>`), its rings in field-picture pixels (`points`: a list of rings; `value`: which of them are
holes, e.g. "holes:0,1").

The page paints a mask and sends it as a PNG (at 1/`scale` of the field picture); `from_png` turns it into rings with
cv2.findContours. Candidates whose centre lies inside a person's marked area are "excluded by the annotator": that
person's labels on them are left out of scoring, agreement and training (`counted`). The analyses do not use them
unless the project setting `labels.apply_exclusions` is on (`for_analysis`, read by the v2 proposals as hand-drawn
exclusions); it is off by default, as the study's rule is that every tile is analysed with no hand exclusions.
"""
from __future__ import annotations

import base64

import cv2
import numpy as np
import pandas as pd

from stainid.labels import frames

TASK = "exclude"


def label_id(tile_id: str) -> str:
    return f"{TASK}:{tile_id}"


def from_png(data: str, scale: float) -> tuple[list[list[list[float]]], list[bool]]:
    """The rings of a painted mask (a PNG, as a data URL or base64; painted = any pixel with alpha or red above 127),
    in field-picture pixels, and which rings are holes."""
    raw = base64.b64decode(data.split(",", 1)[1] if data.startswith("data:") else data)
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError("the painted area is not a PNG picture")
    if image.ndim == 3:
        mask = image[..., 3] if image.shape[2] == 4 else image[..., 2]
    else:
        mask = image
    painted = (mask > 127).astype(np.uint8)
    contours, hierarchy = cv2.findContours(painted, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    rings, holes = [], []
    for i, contour in enumerate(contours):
        if len(contour) < 3:
            continue
        simple = cv2.approxPolyDP(contour, 0.8, True)[:, 0, :] if len(contour) > 8 else contour[:, 0, :]
        if len(simple) < 3:
            continue
        rings.append([[round(float(x + 0.5) * scale, 1), round(float(y + 0.5) * scale, 1)] for x, y in simple])
        holes.append(bool(hierarchy[0][i][3] >= 0))  # a contour inside another: a hole (erased)
    return rings, holes


def holes_text(holes: list[bool]) -> str:
    return "holes:" + ",".join("1" if h else "0" for h in holes)


def rings_of(label) -> tuple[list[np.ndarray], list[bool]]:
    """The rings of an `exclude` label and which are holes."""
    points = label.get("points") if isinstance(label, dict) else getattr(label, "points", None)
    value = (label.get("value") if isinstance(label, dict) else getattr(label, "value", "")) or ""
    rings = [np.asarray(r, np.float32) for r in (points or []) if r is not None and len(r) >= 3]
    flags = str(value).removeprefix("holes:").split(",") if str(value).startswith("holes:") else []
    return rings, [flags[i] == "1" if i < len(flags) else False for i in range(len(rings))]


def inside(rings: list[np.ndarray], holes: list[bool], x: float, y: float) -> bool:
    """Whether a point is in the marked area: inside an outer ring and not inside one of the holes."""
    def within(ring):
        return cv2.pointPolygonTest(ring.reshape(-1, 1, 2), (float(x), float(y)), False) >= 0

    return any(within(r) for r, h in zip(rings, holes) if not h) and not any(within(r) for r, h in zip(rings, holes) if h)


def counted(labels: pd.DataFrame) -> pd.DataFrame:
    """The labels without the ones a person made inside an area they marked as bad on that field (their class and
    outline labels on objects whose centre lies there): what scoring, agreement and training read."""
    if labels.empty or "task" not in labels or not (labels.task == TASK).any():
        return labels
    marks = labels[labels.task == TASK]
    areas = {(r.source, r.tile_id): rings_of(r) for r in marks.itertuples()}
    objects = labels[labels.task.isin(["class", "outline"]) & labels.x.notna()]
    drop = [i for i, r in zip(objects.index, objects.itertuples())
            if (r.source, r.tile_id) in areas and inside(*areas[(r.source, r.tile_id)], float(r.x), float(r.y))]
    return labels.drop(index=drop) if drop else labels


def marked_area_px(labels: pd.DataFrame) -> dict[tuple[str, str], float]:
    """(person, field) -> the area they marked as bad there, in field-picture pixels (outer rings less holes)."""
    out = {}
    for r in labels[labels.task == TASK].itertuples() if "task" in labels else []:
        rings, holes = rings_of(r)
        out[(r.source, r.tile_id)] = max(0.0, sum((-1 if h else 1) * float(cv2.contourArea(ring)) for ring, h in zip(rings, holes)))
    return out


def in_core(labels: pd.DataFrame, tiles: pd.DataFrame, context_px: int, core_id: str, stain: str) -> list[dict]:
    """The areas annotators marked on the fields of one core image, in core pixels: [{source, tile_id, rings, holes}]."""
    if labels.empty or "task" not in labels:
        return []
    mine = tiles[(tiles.core_id.astype(str) == str(core_id)) & (tiles.stain == stain)].drop_duplicates("tile_id").set_index("tile_id")
    marks = labels[(labels.task == TASK) & labels.tile_id.isin(set(mine.index))]
    out = []
    for r in marks.itertuples():
        x0, y0 = frames.field_origin(mine.loc[r.tile_id], context_px)
        rings, holes = rings_of(r)
        out.append({"source": r.source, "tile_id": r.tile_id, "holes": holes,
                    "rings": [(ring + np.array([x0, y0], np.float32)).round(1).tolist() for ring in rings]})
    return out


def for_analysis(project, labels: pd.DataFrame, tiles: pd.DataFrame) -> dict[str, list[dict]]:
    """The marked areas as hand-drawn exclusions per core image (the `manual_exclusions` format: image path ->
    [{"polygon_core_px": …}]), only when the project setting `labels.apply_exclusions` is on (off by default)."""
    if not (project.config.get("labels") or {}).get("apply_exclusions") or labels.empty or "task" not in labels:
        return {}
    marks = labels[labels.task == TASK]
    boxes = tiles.drop_duplicates("tile_id").set_index("tile_id")
    out: dict[str, list[dict]] = {}
    for r in marks.itertuples():
        if r.tile_id not in boxes.index:
            continue
        tile = boxes.loc[r.tile_id]
        x0, y0 = frames.field_origin(tile, project.context_px)
        rings, holes = rings_of(r)
        for ring, hole in zip(rings, holes):
            if not hole:  # holes are left out: the analysis takes the outer areas
                out.setdefault(str(tile["image_path"]), []).append(
                    {"polygon_core_px": (ring + np.array([x0, y0], np.float32)).round().astype(int).tolist(), "source": f"annotator {r.source}"})
    return out
