"""Areas annotators mark as bad on a field (the field tool's exclusion brush, key E): one `exclude` label per person and
field (`exclude:<tile_id>`), its rings in field-picture pixels (`points`: a list of rings) with each ring's type and
whether it is a hole in the area of its type (`value`, e.g. "kinds:1,1,include;holes:0,1,0"; older marks "holes:0,1",
all of type 1). The types are `KINDS`: 1 fold or tissue damage, 2 artifact, 3 no tissue, 4 out of focus or bad
staining, and "include": a part of an automatic exclusion (a fold, an artifact … the detection found) the annotator
says is fine (Shift-drag), an override.

The page paints a mask per type and sends them as PNGs (at 1/`scale` of the field picture); `from_pngs` turns them into
rings with cv2.findContours. Candidates whose centre lies inside a person's marked area (any type but "include") are
"excluded by the annotator": that person's labels on them are left out of scoring, agreement and training (`counted`).
The analyses do not use the marks unless the project setting `labels.apply_exclusions` is on (`for_analysis`, read by
the v2 proposals as hand-drawn exclusions, types and include overrides carried along); it is off by default, as the
study's rule is that every tile is analysed with no hand exclusions.
"""
from __future__ import annotations

import base64

import cv2
import numpy as np
import pandas as pd

from stainid.labels import frames

TASK = "exclude"
INCLUDE = "include"
KINDS = {"1": "fold or tissue damage", "2": "artifact (scratch, bubble, debris, linear artifact)", "3": "no tissue / empty",
         "4": "out of focus or bad staining", INCLUDE: "cleared by the annotator (part of an automatic exclusion that is fine)"}


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


def value_text(holes: list[bool], kinds: list[str]) -> str:
    """The `value` of an `exclude` label: each ring's type and whether it is a hole."""
    return "kinds:" + ",".join(kinds) + ";" + holes_text(holes)


def from_pngs(pngs: dict[str, str], scale: float) -> tuple[list[list[list[float]]], list[bool], list[str]]:
    """The rings of the painted masks, one PNG per type (`KINDS`), with each ring's hole flag and type."""
    rings, holes, kinds = [], [], []
    for kind, data in pngs.items():
        if kind not in KINDS:
            raise ValueError(f"unknown type of excluded area {kind!r}")
        if not data:
            continue
        found, flags = from_png(data, scale)
        rings += found
        holes += flags
        kinds += [kind] * len(found)
    return rings, holes, kinds


def parts_of(label) -> tuple[list[np.ndarray], list[bool], list[str]]:
    """The rings of an `exclude` label, which are holes, and each one's type."""
    points = label.get("points") if isinstance(label, dict) else getattr(label, "points", None)
    value = str((label.get("value") if isinstance(label, dict) else getattr(label, "value", "")) or "")
    rings = [np.asarray(r, np.float32) for r in (points or []) if r is not None and len(r) >= 3]
    fields = dict(part.split(":", 1) for part in value.split(";") if ":" in part)
    flags = fields.get("holes", "").split(",") if fields.get("holes") else []
    kinds = fields.get("kinds", "").split(",") if fields.get("kinds") else []
    return (rings, [flags[i] == "1" if i < len(flags) else False for i in range(len(rings))],
            [kinds[i] if i < len(kinds) and kinds[i] else "1" for i in range(len(rings))])


def rings_of(label) -> tuple[list[np.ndarray], list[bool], list[str]]:
    """The rings of an `exclude` label, which are holes and their types (`parts_of`)."""
    return parts_of(label)


def inside(rings: list[np.ndarray], holes: list[bool], kinds_or_x, x: float | None = None, y: float | None = None) -> bool:
    """Whether a point is in the marked area: inside an outer ring of a type other than "include" and not inside one of
    that type's holes. (Also `inside(rings, holes, x, y)` for rings all of one type.)"""
    if y is None:
        kinds, x, y = ["1"] * len(rings), kinds_or_x, x
    else:
        kinds = kinds_or_x
    return bool(inside_many(rings, holes, kinds, [x], [y])[0])


def _within(ring: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Which of the points are inside a ring or on it (even-odd ray casting for all of them at once; the edge counts as
    inside, as with cv2.pointPolygonTest >= 0)."""
    x0, y0 = ring[:, 0][:, None], ring[:, 1][:, None]
    x1, y1 = np.roll(ring[:, 0], -1)[:, None], np.roll(ring[:, 1], -1)[:, None]
    px, py = xs[None, :], ys[None, :]
    crosses = (y0 > py) != (y1 > py)
    with np.errstate(divide="ignore", invalid="ignore"):
        at = x0 + (py - y0) * (x1 - x0) / (y1 - y0)
    odd = (crosses & (px < at)).sum(axis=0) % 2 == 1
    dx, dy = x1 - x0, y1 - y0  # on an edge (within 1e-6 px)
    length = dx * dx + dy * dy
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.clip(np.where(length > 0, ((px - x0) * dx + (py - y0) * dy) / length, 0.0), 0.0, 1.0)
    edge = (np.hypot(px - (x0 + t * dx), py - (y0 + t * dy)) <= 1e-6).any(axis=0)
    return odd | edge


def inside_many(rings: list[np.ndarray], holes: list[bool], kinds, xs, ys=None) -> np.ndarray:
    """`inside` for many points at once (also `inside_many(rings, holes, xs, ys)` for rings all of one type)."""
    if ys is None:
        kinds, xs, ys = ["1"] * len(rings), kinds, xs
    xs, ys = np.asarray(xs, np.float64), np.asarray(ys, np.float64)
    out = np.zeros(len(xs), bool)
    for kind in set(kinds) - {INCLUDE}:  # each type's area: its outer rings less its holes
        outer = np.zeros(len(xs), bool)
        hole = np.zeros(len(xs), bool)
        for ring, is_hole, k in zip(rings, holes, kinds):
            if k != kind:
                continue
            found = _within(np.asarray(ring, np.float64), xs, ys)
            if is_hole:
                hole |= found
            else:
                outer |= found
        out |= outer & ~hole
    return out


def counted(labels: pd.DataFrame) -> pd.DataFrame:
    """The labels without the ones a person made inside an area they marked as bad on that field (their class and
    outline labels on objects whose centre lies there): what scoring, agreement and training read. Only the labels of
    the (person, field) pairs that have a marked area are looked at, each pair's at once."""
    if labels.empty or "task" not in labels or not (labels.task == TASK).any():
        return labels
    marks = labels[labels.task == TASK].drop_duplicates(["source", "tile_id"], keep="last")
    areas = {(r.source, r.tile_id): rings_of(r) for r in marks.itertuples()}
    keys = pd.MultiIndex.from_arrays([labels.source, labels.tile_id])
    near = keys.isin(list(areas)) & labels.task.isin(["class", "outline"]).to_numpy() & labels.x.notna().to_numpy()
    if not near.any():
        return labels
    objects = labels[near]
    drop = []
    for (source, tile_id), group in objects.groupby(["source", "tile_id"], sort=False):
        rings, holes, kinds = areas[(source, tile_id)]
        if rings:
            hit = inside_many(rings, holes, kinds, group.x.astype(float).to_numpy(), group.y.astype(float).to_numpy())
            drop.extend(group.index[hit])
    return labels.drop(index=drop) if drop else labels


def marked_area_px(labels: pd.DataFrame) -> dict[tuple[str, str], float]:
    """(person, field) -> the area they marked as bad there, in field-picture pixels (outer rings less holes)."""
    out = {}
    for r in labels[labels.task == TASK].itertuples() if "task" in labels else []:
        rings, holes, kinds = rings_of(r)
        out[(r.source, r.tile_id)] = max(0.0, sum((-1 if h else 1) * float(cv2.contourArea(ring)) for ring, h, k in zip(rings, holes, kinds)
                                                  if k != INCLUDE))
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
        rings, holes, kinds = rings_of(r)
        out.append({"source": r.source, "tile_id": r.tile_id, "holes": holes, "kinds": kinds,
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
        rings, holes, kinds = rings_of(r)
        for ring, hole, kind in zip(rings, holes, kinds):
            if not hole:  # holes are left out: the analysis takes the outer areas
                out.setdefault(str(tile["image_path"]), []).append(
                    {"polygon_core_px": (ring + np.array([x0, y0], np.float32)).round().astype(int).tolist(), "source": f"annotator {r.source}",
                     "kind": kind, "type": KINDS.get(kind, kind), "include": kind == INCLUDE})
    return out
