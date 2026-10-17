"""The edits of the excluded areas as the label log stores them (the write side; `stainid.exclusions` owns the resolved
layer every view, the analysis, training and scoring read).

One edit is one `exclude` label: what a person painted and cleared on a field with the brush (key E) in one save, with
id `exclude:<tile_id>:<token>` (a new one per save, so each save is its own edit and the log is the history), its rings
in field-picture pixels (`points`: a list of rings), and each ring's type and whether it is a hole in the area of its
type (`value`, e.g. "kinds:1,1,-2;holes:0,1,0"; older labels "holes:0,1", all of type 1). The types are `KINDS`:
1 fold or tissue damage, 2 artifact, 3 no tissue, 4 out of focus or bad staining (painted: added to that type's area,
taking the pixels from any other type), "-1"–"-4" (removed from that type's area, detection's included: Shift-drag in
the field tool), and "include": cleared, counted whatever detection or anyone earlier said (Shift-drag in "Mark bad
areas"). An edit is resolved per type into masks by `stainid.exclusions` (removals first, then the rest). Edits of the hand-drawn areas of the earlier AI-review rounds
(source "AI reviewer") have the core id as `tile_id` and rings in core pixels (`stainid.exclusions.import_reviewer`).

The page paints a mask per type and sends them as PNGs (at 1/`scale` of the field picture); `from_pngs` turns them into
rings with cv2.findContours.
"""
from __future__ import annotations

import base64
import secrets
import time

import cv2
import numpy as np

TASK = "exclude"
INCLUDE = "include"
KINDS = {"1": "fold or tissue damage", "2": "artifact (scratch, bubble, debris, linear artifact)", "3": "no tissue / empty",
         "4": "out of focus or bad staining", INCLUDE: "cleared: counted, whatever was said before",
         "-1": "taken out of the fold area", "-2": "taken out of the artifact area", "-3": "taken out of the no-tissue area",
         "-4": "taken out of the out-of-focus area"}


def label_id(tile_id: str) -> str:
    """A new edit's id: one per save."""
    return f"{TASK}:{tile_id}:{int(time.time() * 1000)}{secrets.token_hex(3)}"


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
