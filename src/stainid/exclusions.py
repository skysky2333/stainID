"""The excluded areas, in two layers per core image and stain, each with one reader. This module owns both.

- `automatic(project, core_id, stain)`: what the analysis uses, and nothing else does. It is detection's own areas from
  the stage-2 mask (`<outputs.v2>/masks/<stain>/<core_id>.png`, 1/`SCALE` of the core image, bit flags
  `v2.proposals.BITS`): no tissue, linear artifacts, folds (6E10 and AT8; NeuN counts folded tissue) and the AT8 edge
  guard. The hand-drawn bit is not read. Tissue in no analysed tile is "not analysed" in the views (`view`).
- `annotated(project, core_id, stain)`: people's areas. These are the `exclude` labels in the label log (written by
  `labels.exclusions`: the brush of "Mark bad areas" and of the object rounds, and "AI reviewer" for the hand-drawn
  polygons of the earlier AI-review rounds, `import_reviewer`), applied in time order with the latest edit winning. A
  painted part has its type (1–4) and joins that type's area (painting over another type gives the pixels to the
  painted one); a removed part ("-1"–"-4", Shift-drag in the field tool) leaves that type's area, detection's included;
  a cleared part ("include", "Mark bad areas") says "this area is fine" whatever the type. Each type's area is one
  merged region set (`regions`). Like every human annotation they only train
  and evaluate models (the object models, and the bad-area detector: `automatic` against people's areas). The analysis
  never reads them.

`final_say` puts people's areas over detection's: what people last said about each pixel, else detection's area. It is
the label set for training and evaluating, never an analysis input. Labels on objects there are left out of training,
scoring and the model comparison (`drop_excluded`), and the object rounds draw those objects faded and do not ask about
them.

Codes (`CODES`): 0 counted, 1 fold or tissue damage, 2 artifact, 3 no tissue / empty, 4 out of focus or bad staining
(people only), 5 AT8 edge guard, 6 not analysed (views only), 7 white matter left out by stage 5 (views only, when
`v2.white_matter.analysis` is "exclude"), `OUTSIDE` (255) outside every tile.

Read API:
- `automatic(...) -> Layer | None`, None before stage 2. `digest` is the stage-2 mask's content, for the manifests of
  the stages that read it.
- `annotated(...) -> Annotated | None`: codes 0 untouched, 1–4 painted, `CLEARED` (9) marked fine; `regions()` as
  polygons; `edits`, the history.
- `final_say(...)`, `shares` / `codes_at` (objects' shares in it) and `drop_excluded`.
- `candidate_fractions` (shares in the automatic layer, for stage 4) and `view` (what the views draw of the automatic
  layer).
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from stainid import cache
from stainid.tables import write_text_atomic

SCALE = 8  # core-image px per layer px (stainid.v2.proposals.MASK_SCALE)
TISSUE, LINEAR, FOLD, MANUAL, EDGE_GUARD, TILE = 1, 2, 4, 8, 16, 32  # stainid.v2.proposals.BITS
COUNTED, OUTSIDE = 0, 255
CLEARED = 9  # `annotated`: a part people marked fine
CODES = {1: "fold or tissue damage", 2: "artifact (scratch, bubble, debris, linear artifact)", 3: "no tissue / empty",
         4: "out of focus or bad staining", 5: "AT8 edge guard (brown rim along an edge)",
         6: "not analysed: tissue in no analysed tile (an image QC calls empty, a square with too little tissue)",
         7: "white matter (found by the algorithm; left out of the analysis)"}
NOT_ANALYSED = 6  # only in the views' codes (`view`): the analysis counts nothing outside the tiles anyway
WHITE_MATTER = 7  # only in the views' codes (`view`), when `v2.white_matter.analysis` is "exclude": stage 5 leaves it out
EXCLUDED_AT = 0.5  # an object with at least this share of its outline in excluded areas is excluded
REVIEWER = "AI reviewer"  # the source of the hand-drawn areas of the earlier AI-review rounds
TASK, INCLUDE = "exclude", "include"  # labels.exclusions.TASK / INCLUDE


# ---------------------------------------------------------------- the edits

@dataclass(frozen=True)
class Edit:
    """One edit of the layer: its rings in core pixels, each ring's type ("1"–"4" or "include") and hole flag."""
    time: float
    source: str
    id: str
    rings: tuple
    holes: tuple
    kinds: tuple
    session: str = ""  # the labelling session it was saved in ("" for none)

    def key(self) -> tuple:
        return (round(self.time, 6), self.source, self.id, self.session, self.kinds, self.holes,
                tuple(tuple(np.round(r, 1).ravel().tolist()) for r in self.rings))


def _parts(points, value) -> tuple[list[np.ndarray], list[bool], list[str]]:
    from stainid.labels import exclusions as marks  # noqa: PLC0415  (the format of an edit)

    return marks.parts_of({"points": points, "value": value})


def _tiles(project) -> pd.DataFrame:
    from stainid.sampling.tiles import known_tiles  # noqa: PLC0415

    return known_tiles(project)


def _labels(project) -> pd.DataFrame:
    from stainid.labels import store  # noqa: PLC0415

    return store.current(project.output("reviews"))


def _log(project) -> Path:
    from stainid.labels import store  # noqa: PLC0415

    return store.log_path(project.output("reviews"))


def _tile_paths(project) -> list:
    from stainid.sampling import tiles as tiles_module  # noqa: PLC0415

    return list(tiles_module._paths(project))


def all_edits(project) -> dict[tuple[str, str], list[Edit]]:
    """(stain, core_id) -> its edits, oldest first; kept until a label is saved or a tile table changes."""
    def make() -> dict[tuple[str, str], list[Edit]]:
        labels = _labels(project)
        if labels.empty or "task" not in labels or not (labels.task == TASK).any():
            return {}
        rows = labels[labels.task == TASK]
        tiles = _tiles(project)
        boxes = tiles.drop_duplicates("tile_id").set_index("tile_id") if not tiles.empty else pd.DataFrame()
        cores = set(tiles.core_id.astype(str)) if "core_id" in tiles else set()
        out: dict[tuple[str, str], list[Edit]] = {}
        for r in rows.itertuples():
            tile_id = str(r.tile_id)
            if tile_id in boxes.index:  # a field: rings in field-picture pixels
                tile = boxes.loc[tile_id]
                x0, y0 = max(0, int(tile["x_px"]) - project.context_px), max(0, int(tile["y_px"]) - project.context_px)
                core_id, stain = str(tile["core_id"]), str(tile["stain"])
            elif tile_id in cores:  # a core image: rings in core pixels
                x0, y0, core_id, stain = 0, 0, tile_id, str(r.stain)
            else:
                continue
            rings, holes, kinds = _parts(r.points, r.value)
            if not rings:
                continue
            shift = np.array([x0, y0], np.float32)
            out.setdefault((stain, core_id), []).append(Edit(float(r.time or 0.0), str(r.source), str(r.id),
                                                             tuple(np.asarray(g, np.float32) + shift for g in rings), tuple(holes), tuple(kinds),
                                                             str(getattr(r, "session", "") or "")))
        for edits in out.values():
            edits.sort(key=lambda e: (e.time, e.source, e.id))
        return out

    return cache.memo(("exclusion edits", str(project.root)), [_log(project), *_tile_paths(project)], make)


Scope = str | tuple[str, str] | None  # a labelling session, or (session, person): one person's edits in it


def edits(project, core_id: str, stain: str, ai: bool = True, session: Scope = None) -> list[Edit]:
    """The edits of one core image and stain, oldest first (the history: who, when, which type). `ai` False: people's
    only, without the AI reviewer's (`labels.store.is_ai_label`), as the annotation pages start from them. `session`: only
    the edits saved in that labelling session (inside a session nothing another session said counts); (session, person):
    only that person's there (an extra annotation of a finished field starts from detection's areas and their own)."""
    found = all_edits(project).get((stain, str(core_id)), [])
    if session is not None:
        name, person = session if isinstance(session, tuple) else (session, None)
        found = [e for e in found if e.session == name and (person is None or e.source == person)]
    if ai:
        return found
    from stainid.labels.store import is_ai_label  # noqa: PLC0415

    return [e for e in found if not is_ai_label(e.source)]


# ---------------------------------------------------------------- the layer

def mask_path(project, core_id: str, stain: str) -> Path:
    from stainid import candidates  # noqa: PLC0415

    return candidates.root(project) / "masks" / stain / f"{core_id}.png"


def masks_folder(project, stain: str) -> Path:
    from stainid import candidates  # noqa: PLC0415

    return candidates.root(project) / "masks" / stain


def codes_of(bits: np.ndarray, stain: str) -> np.ndarray:
    """Detection's own areas as codes: counted tissue, linear artifacts (2), folds (1, not NeuN), the AT8 edge guard (5)
    and no tissue (3); outside every tile `OUTSIDE`."""
    codes = np.full(bits.shape, OUTSIDE, np.uint8)
    tile = (bits & TILE) > 0
    codes[tile] = COUNTED
    codes[tile & ((bits & LINEAR) > 0)] = 2
    if stain != "NeuN":
        codes[tile & ((bits & FOLD) > 0)] = 1
    if stain == "AT8":
        codes[tile & ((bits & EDGE_GUARD) > 0)] = 5
    codes[tile & ((bits & TISSUE) == 0)] = 3
    return codes


def rasterize(rings, holes, shape: tuple[int, int], scale: int = SCALE, offset: tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    """The area of rings (core px; outer rings less holes) on a grid of `scale` core px per cell starting at `offset`
    (core px): a cell is in when its centre is."""
    out = np.zeros(shape, np.uint8)
    for ring, hole in sorted(zip(rings, holes), key=lambda p: p[1]):  # outer rings first, then the holes cut out
        grid = (np.asarray(ring, np.float64) - np.asarray(offset)) / scale - 0.5
        cv2.fillPoly(out, [np.round(grid * 16).astype(np.int32)], 0 if hole else 1, lineType=cv2.LINE_8, shift=4)
    return out > 0


def removal(kind: str) -> int | None:
    """The type a removal kind ("-1"–"-4") takes pixels out of; None for any other kind."""
    text = str(kind)
    return int(text[1:]) if text.startswith("-") and text[1:].isdigit() else None


def edit_areas(edit: Edit, shape: tuple[int, int]):
    """An edit's parts as (kind, cells) in the order they apply: removals first (out of one type's area), then what was
    cleared and painted. Within one save a pixel is painted one type at most (painting a type takes it off the others)."""
    order = sorted(dict.fromkeys(edit.kinds), key=lambda k: 0 if removal(k) is not None else 1)
    for kind in order:
        rings = [r for r, k in zip(edit.rings, edit.kinds) if k == kind]
        holes = [h for h, k in zip(edit.holes, edit.kinds) if k == kind]
        yield kind, rasterize(rings, holes, shape)


def _painted_code(kind: str) -> int:
    return int(kind) if str(kind).isdigit() else 1


def apply(codes: np.ndarray, edit: Edit) -> None:
    """One edit onto the codes, in place: a removed part leaves its type's area (counted), a cleared part is counted,
    a painted part takes the painted type (also over another type's pixels: the merge of `annotated`)."""
    analysed = codes != OUTSIDE
    for kind, area in edit_areas(edit, codes.shape):
        area &= analysed
        out_of = removal(kind)
        if out_of is not None:
            codes[area & (codes == out_of)] = COUNTED
        else:
            codes[area] = COUNTED if kind == INCLUDE else _painted_code(kind)


@dataclass
class Layer:
    """Excluded areas of one core image and stain at 1/`scale` (the automatic layer, or people's final say):
    `codes` (`CODES`, 0 counted, `OUTSIDE`), `bits` (the stage-2 mask as made) and `digest`."""
    codes: np.ndarray
    bits: np.ndarray
    digest: str
    scale: int = SCALE
    edits: list = field(default_factory=list)

    def excluded(self) -> np.ndarray:
        return (self.codes != COUNTED) & (self.codes != OUTSIDE)

    def valid(self) -> np.ndarray:
        """Counted tissue: analysed and not excluded."""
        return self.codes == COUNTED

    def analysed(self) -> np.ndarray:
        return self.codes != OUTSIDE

    def regions(self, simplify_px: float = 0.6) -> list[dict]:
        """Each excluded type's area (1–5) as merged polygons in core pixels (`regions_of`)."""
        return regions_of(self.codes, self.scale, {code: str(code) for code in range(1, 6)}, simplify_px)

    def of_type(self, code: int) -> np.ndarray:
        """One type's area (cells): detection's and, in `final_say`, people's edits merged into it."""
        return self.codes == code

    def cells(self, x0: float, y0: float, x1: float, y1: float) -> tuple[slice, slice]:
        """The rows and columns of the layer under a core-pixel box (clipped to the layer)."""
        h, w = self.codes.shape
        c0, r0 = max(0, int(x0) // self.scale), max(0, int(y0) // self.scale)
        c1, r1 = min(w, -(-int(x1) // self.scale)), min(h, -(-int(y1) // self.scale))
        return slice(r0, max(r0, r1)), slice(c0, max(c0, c1))

    def at(self, xs, ys) -> np.ndarray:
        """The code at core-pixel positions (`OUTSIDE` beyond the layer)."""
        xs, ys = np.asarray(xs, np.float64), np.asarray(ys, np.float64)
        h, w = self.codes.shape
        c, r = np.floor(xs / self.scale).astype(np.int64), np.floor(ys / self.scale).astype(np.int64)
        ok = (c >= 0) & (c < w) & (r >= 0) & (r < h)
        out = np.full(len(xs), OUTSIDE, np.uint8)
        out[ok] = self.codes[r[ok], c[ok]]
        return out

    def fractions(self, outlines, xs, ys) -> np.ndarray:
        """Each object's excluded share: of the analysed cells its outline (a flat core-pixel polygon "x0 y0 x1 y1 …",
        a list of numbers or an (n, 2) array) covers, the share in an excluded area; an object covering no cell, or with
        no outline, is 1 or 0 by the cell at its centre (`xs`, `ys`)."""
        xs, ys = np.asarray(xs, np.float64), np.asarray(ys, np.float64)
        centre = self.at(xs, ys)
        out = ((centre != COUNTED) & (centre != OUTSIDE)).astype(np.float64)
        excluded = self.excluded()
        analysed = self.analysed()
        h, w = self.codes.shape
        for i, polygon in enumerate(outlines):
            ring = _ring(polygon)
            if ring is None:
                continue
            lo, hi = ring.min(axis=0), ring.max(axis=0)
            c0, r0 = max(0, int(lo[0] // self.scale)), max(0, int(lo[1] // self.scale))
            c1, r1 = min(w, int(hi[0] // self.scale) + 1), min(h, int(hi[1] // self.scale) + 1)
            if c1 <= c0 or r1 <= r0:
                continue
            inside = rasterize([ring], [False], (r1 - r0, c1 - c0), self.scale, (c0 * self.scale, r0 * self.scale))
            inside &= analysed[r0:r1, c0:c1]
            if inside.any():
                out[i] = float(excluded[r0:r1, c0:c1][inside].mean())
        return out


def _ring(polygon) -> np.ndarray | None:
    if polygon is None or (isinstance(polygon, float) and polygon != polygon):
        return None
    if isinstance(polygon, str):
        if not polygon.strip():
            return None
        polygon = polygon.split(";")[0].split()
    values = np.asarray(polygon, np.float64).ravel()
    if values.size < 6:
        return None
    return values[: values.size // 2 * 2].reshape(-1, 2)


_READ_BYTES = threading.local()  # the mask file read for its digest, so the layer made right after decodes it without a second read


def _mask_digest(path: Path) -> str | None:
    """The stage-2 mask's content digest, kept until the file changes."""
    def make() -> str:
        data = path.read_bytes()
        _READ_BYTES.last = (str(path), data)
        return hashlib.sha1(data).hexdigest()

    if cache.signature(path) is None:
        return None
    return cache.memo(("exclusion mask digest", str(path)), [path], make)


def digest(project, core_id: str, stain: str) -> str | None:
    """The automatic layer's digest (the stage-2 mask's content), for the manifests of the stages that read it; None
    before stage 2 made the mask. People's edits never change it."""
    found = _mask_digest(mask_path(project, core_id, stain))
    return None if found is None else found[:16]


def digests(project, stain: str, core_ids) -> dict[str, str | None]:
    return {str(c): digest(project, str(c), stain) for c in core_ids}


def edits_digest(project, core_id: str, stain: str, ai: bool = True, session: Scope = None) -> str:
    """People's edits of a core image, as a short text: changes exactly when one is saved."""
    keys = [e.key() for e in edits(project, core_id, stain, ai, session)]
    return hashlib.sha1(json.dumps(keys, default=str).encode()).hexdigest()[:16]


@dataclass
class Annotated:
    """People's areas of one core image and stain on the 1/`scale` grid: `codes` (0 untouched, 1–4 painted that type,
    `CLEARED` marked fine; the latest edit wins), the `edits` (the history) and `digest` (`edits_digest`)."""
    codes: np.ndarray
    edits: list
    digest: str
    scale: int = SCALE

    def painted(self) -> np.ndarray:
        return (self.codes >= 1) & (self.codes <= 4)

    def cleared(self) -> np.ndarray:
        return self.codes == CLEARED

    def regions(self, simplify_px: float = 0.6) -> list[dict]:
        """The areas as polygons in core pixels: [{kind: "1"–"4" or "include", rings: [outer, hole, …]}], each type's
        area merged (one entry per connected part of it)."""
        return regions_of(self.codes, self.scale, {code: str(code) for code in range(1, 5)} | {CLEARED: INCLUDE}, simplify_px)


def regions_of(codes: np.ndarray, scale: int, kinds: dict[int, str], simplify_px: float = 0.6) -> list[dict]:
    """The cells of each code in `kinds` as polygons in core pixels ({kind, rings: [outer, hole, …]}), one entry per
    connected part: each type's area as one merged region set."""
    out = []
    for code in sorted(set(np.unique(codes).tolist()) & set(kinds)):
        mask = (codes == code).astype(np.uint8)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is None:
            continue
        kind = kinds[code]
        for i, contour in enumerate(contours):
            if hierarchy[0][i][3] >= 0:
                continue  # a hole: with its outer ring
            rings = [contour]
            child = hierarchy[0][i][2]
            while child >= 0:
                rings.append(contours[child])
                child = hierarchy[0][child][0]
            shaped = []
            for ring in rings:
                simple = cv2.approxPolyDP(ring, simplify_px, True)[:, 0, :] if len(ring) > 4 else ring[:, 0, :]
                if len(simple) >= 3:
                    shaped.append([[round((float(x) + 0.5) * scale, 1), round((float(y) + 0.5) * scale, 1)] for x, y in simple])
            if shaped:
                out.append({"kind": kind, "rings": shaped})
    return out


def _grid_shape(project, core_id: str, stain: str) -> tuple[int, int] | None:
    """The 1/`SCALE` grid of a core image: the stage-2 mask's, else from the image's size."""
    path = mask_path(project, core_id, stain)
    if path.exists():
        bits = _bits(path)
        if bits is not None:
            return bits.shape
    from stainid.tables import read_csv  # noqa: PLC0415

    table = project.input("core_images")
    for row in read_csv(table) if table.exists() else []:
        if str(row.get("core_id")) == str(core_id) and row.get("stain") == stain and row.get("native_width_px"):
            return -(-int(float(row["native_height_px"])) // SCALE), -(-int(float(row["native_width_px"])) // SCALE)
    return None


def annotated(project, core_id: str, stain: str, ai: bool = True, session: Scope = None) -> Annotated | None:
    """People's areas of one core image and stain (every edit in time order, the latest winning); None when the image
    is unknown. Only training and evaluation read them. `ai` False: without the AI reviewer's (the annotation pages)."""
    shape = _grid_shape(project, core_id, stain)
    if shape is None:
        return None
    found = edits(project, core_id, stain, ai, session)
    codes = np.zeros(shape, np.uint8)
    auto = automatic(project, core_id, stain) if found else None
    final = auto.codes.copy() if auto is not None and auto.codes.shape == shape else np.zeros(shape, np.uint8)  # people's say so far
    for edit in found:
        for kind, area in edit_areas(edit, shape):
            out_of = removal(kind)
            if out_of is not None:  # taken out of that type's area (detection's or people's): marked fine there
                hit = area & (final == out_of)
                codes[hit], final[hit] = CLEARED, COUNTED
            elif kind == INCLUDE:
                codes[area], final[area] = CLEARED, COUNTED
            else:
                codes[area] = final[area] = _painted_code(kind)
    return Annotated(codes, found, edits_digest(project, core_id, stain, ai, session))


def agreement(auto: "Layer", people: Annotated) -> dict:
    """How people's areas compare with detection's on one core image (cells of the layer grid): what people say is
    excluded (detection's areas they left, plus what they painted, less what they cleared) against detection's areas."""
    analysed = auto.analysed()
    detected = auto.excluded() & analysed
    final = (detected & ~people.cleared()) | (people.painted() & analysed)
    both = int((detected & final).sum())
    total = int(detected.sum()) + int(final.sum())
    return {"cells_people_excluded": int(final.sum()), "cells_auto_excluded": int(detected.sum()), "cells_both": both,
            "cleared_auto_cells": int((detected & people.cleared()).sum()), "painted_new_cells": int((people.painted() & analysed & ~detected).sum()),
            "cells_analysed": int(analysed.sum()), "f1": round(2 * both / total, 3) if total else None}


def _bits(path: Path, data: bytes | None = None) -> np.ndarray | None:
    bits = cv2.imread(str(path), cv2.IMREAD_UNCHANGED) if data is None else cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if bits is None:
        return None
    return bits if bits.ndim == 2 else bits[..., 0]


_LAYERS: OrderedDict[tuple, Layer] = OrderedDict()
_LAYERS_LIMIT = 24
_SHARES: OrderedDict[tuple, tuple[tuple, dict]] = OrderedDict()  # (project, stain, core) -> (made from, what was worked out)
_SHARES_LIMIT = 100_000  # every core image and stain of a cohort (a pass over all of them must not push its own out)
_LOCK = threading.Lock()


def _layer(project, core_id: str, stain: str, people: bool, ai: bool = True, session: Scope = None) -> Layer | None:
    path = mask_path(project, core_id, stain)
    made_from = digest(project, core_id, stain)
    if made_from is None:
        return None
    found = edits(project, core_id, stain, ai, session) if people else []
    if people:
        made_from = made_from + ":" + edits_digest(project, core_id, stain, ai, session)
    key = (str(project.root), stain, str(core_id), people, ai or not people, session if people else None)
    with _LOCK:
        kept = _LAYERS.get(key)
        if kept is not None and kept.digest == made_from:
            _LAYERS.move_to_end(key)
            return kept
    last = getattr(_READ_BYTES, "last", None)
    _READ_BYTES.last = None
    bits = _bits(path, last[1] if last is not None and last[0] == str(path) else None)
    if bits is None:
        return None
    codes = codes_of(bits, stain)
    for edit in found:
        apply(codes, edit)
    layer = Layer(codes, bits, made_from, SCALE, found)
    with _LOCK:
        _LAYERS[key] = layer
        _LAYERS.move_to_end(key)
        while len(_LAYERS) > _LAYERS_LIMIT:
            _LAYERS.popitem(last=False)
    return layer


def automatic(project, core_id: str, stain: str) -> Layer | None:
    """Detection's automatic areas of one core image and stain, what the analysis uses (and only it); None before stage 2
    made the core's mask. People's edits never change it."""
    return _layer(project, core_id, stain, people=False)


def final_say(project, core_id: str, stain: str, ai: bool = True, session: Scope = None) -> Layer | None:
    """People's final say over detection's areas: detection's areas with every edit on top in time order (a painted
    part excluded with its type, a cleared part counted). It is the label set for training and evaluating (objects there
    are left out of training and scoring, `drop_excluded`), never an analysis input; None before stage 2. `ai` False
    (the annotation pages: Mark bad areas, the brush, the faded objects of Fix the field): without the AI reviewer's
    edits (`labels.store.is_ai_label`), detection's areas and people's edits only. `session`: people's edits of that
    labelling session only (a session's pages)."""
    return _layer(project, core_id, stain, people=True, ai=ai, session=session)


_TISSUE: OrderedDict[tuple, np.ndarray] = OrderedDict()  # (image, its signature, shape) -> tissue cells


def _tissue(project, image_path: str, shape: tuple[int, int] | None) -> np.ndarray | None:
    """The core image's tissue at 1/`SCALE` as the tiling finds it (`qc.core_qc.tissue_mask` of the 1/8 picture),
    on a grid of `shape` (else its own); None when the image cannot be read."""
    from stainid import disk_cache  # noqa: PLC0415
    from stainid.imaging.core_images import read_bgr  # noqa: PLC0415
    from stainid.qc.core_qc import tissue_mask  # noqa: PLC0415

    path = project.root / image_path
    signature = cache.signature(path)
    key = (str(path), signature, shape)
    with _LOCK:
        kept = _TISSUE.get(key)
    if kept is not None:
        return kept
    # kept on disk by the image's signature: a server started again reads a small picture instead of the core image
    kept_file = disk_cache.path(project.root, "tissue", hashlib.sha1(repr((str(path), signature)).encode()).hexdigest()[:24] + ".png")
    tissue = cv2.imread(str(kept_file), cv2.IMREAD_GRAYSCALE) if kept_file.exists() else None
    if tissue is not None:
        disk_cache.used(kept_file)
        tissue = (tissue > 0).astype(np.uint8)
    else:
        bgr = read_bgr(path, cv2.IMREAD_REDUCED_COLOR_8)
        if bgr is None:
            return None
        tissue = tissue_mask(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))[0].astype(np.uint8)
        ok, encoded = cv2.imencode(".png", tissue * 255)
        if ok:
            try:
                disk_cache.write(kept_file, encoded.tobytes())
            except OSError:
                pass
    if shape is not None and tissue.shape != shape:
        tissue = cv2.resize(tissue, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    found = tissue > 0
    with _LOCK:
        _TISSUE[key] = found
        while len(_TISSUE) > 64:
            _TISSUE.popitem(last=False)
    return found


def _image_of(project, core_id: str, stain: str) -> tuple[str | None, bool]:
    """The core image's path and whether it has analysed tiles (from the tile table, else the core image table)."""
    tiles = _tiles(project)
    mine = tiles[(tiles.core_id.astype(str) == str(core_id)) & (tiles.stain.astype(str) == stain)] if "core_id" in tiles else tiles.iloc[:0]
    if len(mine):
        return str(mine.image_path.iloc[0]), True
    from stainid.tables import read_csv  # noqa: PLC0415

    table = project.input("core_images")
    for row in read_csv(table) if table.exists() else []:
        if str(row.get("core_id")) == str(core_id) and row.get("stain") == stain:
            return str(row["image_path"]), False
    return None, False


@dataclass
class View:
    """What the views draw of one core image and stain as "Excluded from analysis": the automatic codes with tissue
    outside every tile as `NOT_ANALYSED`, the scale and a version (changes with the layer and the white matter), and
    `white`: the white matter the algorithm found (`v2.white_matter`) on the same grid, None before its step ran. While
    `v2.white_matter.analysis` is off it is drawn over the codes and counted; with "exclude" its counted cells are
    `WHITE_MATTER` (stage 5 leaves them out), so the counted cells stay what the measures divide by."""
    codes: np.ndarray
    scale: int
    version: str
    white: np.ndarray | None = None


def view(project, core_id: str, stain: str) -> View | None:
    """The automatic layer as the Cores view and the annotation pages draw it (`automatic`, plus "not analysed" tissue);
    an image without any analysed tile is all "not analysed"; None before stage 2 made the mask of an image with tiles."""
    image, tiled = _image_of(project, core_id, stain)
    layer = automatic(project, core_id, stain)
    if layer is None and (tiled or image is None):
        return None
    if layer is not None:
        codes = layer.codes.copy()
        tissue = _tissue(project, image, codes.shape) if image else None
        if tissue is not None:
            codes[(codes == OUTSIDE) & tissue] = NOT_ANALYSED
        white, version = white_matter_of(project, core_id, stain, codes.shape)
        if white is not None and _white_excluded(project):
            codes[white & (codes == COUNTED)] = WHITE_MATTER
            version += "-x"
        return View(codes, layer.scale, layer.digest + version, white)
    tissue = _tissue(project, image, None)
    if tissue is None:
        return None
    codes = np.where(tissue, NOT_ANALYSED, OUTSIDE).astype(np.uint8)
    return View(codes, SCALE, "untiled-" + hashlib.sha1(repr(cache.signature(project.root / image)).encode()).hexdigest()[:12])


def _white_excluded(project) -> bool:
    from stainid.v2 import white_matter  # noqa: PLC0415

    try:
        return white_matter.excluded(project)
    except ValueError:  # an unknown setting: the step and stage 5 stop on it; the views draw it as counted
        return False


def white_matter_of(project, core_id: str, stain: str, shape: tuple[int, int]) -> tuple[np.ndarray | None, str]:
    """The algorithm's white matter of a core image on a layer's grid (`v2.white_matter.read`, cut or padded to
    `shape`) where the layer counts tissue, and a version text ("" before the step made it)."""
    from stainid.v2 import white_matter  # noqa: PLC0415

    found = white_matter.read(project, core_id, stain)
    if found is None:
        return None, ""
    out = np.zeros(shape, bool)
    h, w = min(shape[0], found.shape[0]), min(shape[1], found.shape[1])
    out[:h, :w] = found[:h, :w]
    return out, "-wm" + white_matter.digest(project, core_id, stain)


def signature(project, stain: str | None = None) -> tuple:
    """Changes whenever an edit is saved or stage 2 rewrites a mask (of `stain`, else of any stain): for caches of
    results made from the layers."""
    from stainid import candidates  # noqa: PLC0415

    folders = [masks_folder(project, stain)] if stain else sorted((candidates.root(project) / "masks").glob("*"))
    return cache.signature(_log(project)), tuple(cache.signature(f) for f in folders)


def forget() -> None:
    with _LOCK:
        _LAYERS.clear()
        _TISSUE.clear()
        _SHARES.clear()


# ---------------------------------------------------------------- objects in excluded areas

def quick_stamp(project, stain: str) -> tuple:
    """The stain's masks and candidate folders and the edit log: while none of them changed, no core's mask, candidates
    or edits did (files are replaced whole, which changes their folder), so `_known` need not look at each core's files."""
    from stainid import candidates  # noqa: PLC0415

    return cache.signature(masks_folder(project, stain)), cache.signature(candidates.folder(project, stain)), cache.signature(_log(project))


def _known(project, stain: str, core_id: str, quick: tuple | None = None) -> dict | None:
    """What was worked out about one core's objects (candidate shares, codes at positions), kept until its layer or its
    candidate table changes; None before stage 2. `quick` (`quick_stamp`, for passes over many cores): trusted while
    the folders are as they were when the entry was made."""
    from stainid import candidates  # noqa: PLC0415

    key = (str(project.root), stain, str(core_id))
    if quick is not None:
        with _LOCK:
            kept = _SHARES.get(key)
        if kept is not None and len(kept) > 2 and kept[2] == quick:
            return kept[1]
    made_from = cache.signature(mask_path(project, core_id, stain))  # the file is replaced whole: its signature will do here
    if made_from is None:
        return None
    path = candidates.path(project, stain, core_id, "candidates")
    stamp = (made_from, edits_digest(project, core_id, stain), cache.signature(path))
    with _LOCK:
        kept = _SHARES.get(key)
        if kept is None or kept[0] != stamp:
            kept = (stamp, {}, quick)
        elif quick is not None:
            kept = (stamp, kept[1], quick)
        _SHARES[key] = kept
        _SHARES.move_to_end(key)
        while len(_SHARES) > _SHARES_LIMIT:
            _SHARES.popitem(last=False)
    return kept[1]


def shares(project, stain: str, core_id: str, ids, quick: tuple | None = None, ai: bool = True,
           session: Scope = None) -> dict[str, float]:
    """The excluded share of these candidates of one core in people's final say (`final_say`, `Layer.fractions` of their stain
    outlines; a candidate the store no longer has by the cell at the position in its id); empty before stage 2. Only
    the candidates asked about are worked out (and kept until the layer or the store changes). `ai` False: in the final
    say without the AI reviewer's edits (`final_say`), as the annotation pages draw it."""
    from stainid import candidates  # noqa: PLC0415

    known = _known(project, stain, core_id, quick)
    if known is None:
        return {}
    ids = [str(c) for c in dict.fromkeys(ids)]
    tag = ("share" if ai else "people share", session)
    missing = [c for c in ids if (tag, c) not in known]
    if missing:
        layer = final_say(project, core_id, stain, ai=ai, session=session)
        rows = candidates.read_core(project, stain, core_id, ["candidate_id", "x_core", "y_core", "outline"])
        rows = rows[rows.candidate_id.isin(missing)].drop_duplicates("candidate_id") if len(rows) else rows
        found = dict(zip(rows.candidate_id, layer.fractions(rows.outline.tolist(), rows.x_core, rows.y_core))) if len(rows) else {}
        gone = [c for c in missing if c not in found]
        if gone:
            xy = np.array([candidates.parse_id(c)[2:] for c in gone], np.float64)
            codes = layer.at(xy[:, 0], xy[:, 1])
            found |= {c: float(code not in (COUNTED, OUTSIDE)) for c, code in zip(gone, codes)}
        with _LOCK:
            known.update({(tag, c): float(v) for c, v in found.items()})
    return {c: known[(tag, c)] for c in ids if (tag, c) in known}


def codes_at(project, stain: str, core_id: str, xs, ys, quick: tuple | None = None) -> np.ndarray:
    """People's final say at core-pixel positions (`final_say`; rounded to the pixel; `OUTSIDE` before stage 2)."""
    known = _known(project, stain, core_id, quick)
    points = [(round(float(x)), round(float(y))) for x, y in zip(xs, ys)]
    if known is None:
        return np.full(len(points), OUTSIDE, np.uint8)
    missing = [p for p in dict.fromkeys(points) if ("at",) + p not in known]
    if missing:
        layer = final_say(project, core_id, stain)
        xy = np.array(missing, np.float64)
        codes = layer.at(xy[:, 0], xy[:, 1])
        with _LOCK:
            known.update({("at",) + p: int(c) for p, c in zip(missing, codes)})
    return np.array([known[("at",) + p] for p in points], np.uint8)


def candidate_fractions(project, stain: str, core_id: str, frame: pd.DataFrame) -> pd.Series:
    """Each candidate's share in detection's automatic areas (`automatic`), by candidate id, for stage 4: of `frame`
    (candidate_id, x_core, y_core, outline); empty before stage 2."""
    layer = automatic(project, core_id, stain)
    if layer is None:
        return pd.Series(dtype=float)
    outline = frame["outline"].tolist() if "outline" in frame else [None] * len(frame)
    return pd.Series(layer.fractions(outline, frame.x_core, frame.y_core), index=frame.candidate_id.to_numpy(object))


_DROPPED: OrderedDict[tuple, tuple[float, list]] = OrderedDict()  # (project, labels, tasks, inputs) -> (when, the rows dropped)
DROPPED_FOR = 60.0  # seconds a result is used on the folders' signatures alone; then every core is looked at again


def _drop_inputs(project, labels: pd.DataFrame) -> tuple:
    """What `drop_excluded` depends on besides the labels: the masks and candidate folders of the stains, the edits and
    the tile tables (files are replaced atomically, so their folders' signatures change with them)."""
    from stainid import candidates  # noqa: PLC0415

    stains = sorted(set(labels.stain.astype(str))) if "stain" in labels else []
    return (tuple(cache.signature(masks_folder(project, s)) for s in stains),
            tuple(cache.signature(candidates.folder(project, s)) for s in stains),
            cache.signature(_log(project)), tuple(cache.signature(p) for p in _tile_paths(project)))


def drop_excluded(project, labels: pd.DataFrame, tasks=("class", "outline")) -> pd.DataFrame:
    """The labels (as the log stores them) without those about objects in excluded areas by people's final say
    (`final_say`: their areas over detection's), what training, scoring and the model comparison read: a label naming a
    candidate goes when its excluded share is at least `EXCLUDED_AT`; one without (an object a person added, an older
    label) when the cell at its position is excluded. The rows dropped are kept for the same labels and inputs (the
    Labels page and the comparison ask several times, and a cohort has thousands of core images)."""
    if labels.empty or "task" not in labels:
        return labels
    about = labels.task.isin(list(tasks)).to_numpy()
    if not about.any():
        return labels
    columns = [c for c in ("id", "task", "stain", "tile_id", "x", "y", "candidate", "time") if c in labels]
    fingerprint = (len(labels), int(pd.util.hash_pandas_object(labels[columns].astype(str), index=True).sum()))
    key = (str(project.root), fingerprint, tuple(tasks), _drop_inputs(project, labels))
    with _LOCK:
        kept = _DROPPED.get(key)
    if kept is not None and time.monotonic() - kept[0] < DROPPED_FOR:
        return labels.drop(index=kept[1]) if kept[1] else labels
    saved = _saved_drops(project, key) if kept is None else None  # a server started again: what it worked out before
    drop = saved if saved is not None else _to_drop(project, labels, about)
    if saved is None:
        _save_drops(project, key, drop)
    with _LOCK:
        _DROPPED[key] = (time.monotonic(), drop)
        _DROPPED.move_to_end(key)
        while len(_DROPPED) > 16:
            _DROPPED.popitem(last=False)
    return labels.drop(index=drop) if drop else labels


def _drops_file(project, key: tuple):
    import hashlib  # noqa: PLC0415

    from stainid import disk_cache  # noqa: PLC0415

    return disk_cache.answer_file(project.root, f"excluded-labels-{hashlib.sha1(repr(key).encode()).hexdigest()[:20]}.json")


def _saved_drops(project, key: tuple) -> list | None:
    from stainid import disk_cache  # noqa: PLC0415

    path = _drops_file(project, key)
    try:
        found = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if found.get("key") != repr(key) or found.get("code") != cache.code_stamp():
        return None
    disk_cache.used(path)
    return found["drop"]


def _save_drops(project, key: tuple, drop: list) -> None:
    """Keep the rows dropped for these labels and inputs on disk (the newest few), for the next server."""
    try:
        path = _drops_file(project, key)
        from stainid import disk_cache  # noqa: PLC0415

        write_text_atomic(path, json.dumps({"key": repr(key), "code": cache.code_stamp(), "drop": [d.item() if hasattr(d, "item") else d for d in drop]}))
        disk_cache.wrote(path)
        old = sorted(path.parent.glob("excluded-labels-*.json"), key=lambda p: p.stat().st_mtime)[:-8]
        for stale in old:
            stale.unlink(missing_ok=True)
    except (OSError, TypeError, ValueError):
        pass


def _to_drop(project, labels: pd.DataFrame, about) -> list:
    from stainid import candidates  # noqa: PLC0415
    from stainid.labels import frames  # noqa: PLC0415

    rows = labels[about]
    named = rows.candidate.fillna("").astype(str) if "candidate" in rows else pd.Series("", index=rows.index)
    drop: list = []
    by_core: dict[tuple[str, str], list] = {}
    for index, cid in named[named.str.len() > 0].items():
        try:
            stain, core_id, _, _ = candidates.parse_id(cid)
        except ValueError:
            continue
        by_core.setdefault((stain, core_id), []).append((index, cid))
    quick: dict[str, tuple] = {}

    def stamp_of(stain: str) -> tuple:
        if stain not in quick:
            quick[stain] = quick_stamp(project, stain)
        return quick[stain]

    def named_drops(part):
        (stain, core_id), items = part
        found = shares(project, stain, core_id, [c for _, c in items], stamp_of(stain))
        return [i for i, c in items if found.get(c, 0.0) >= EXCLUDED_AT]

    jobs = []
    rest = rows[(named.str.len() == 0).to_numpy() & rows.x.notna().to_numpy()]
    if len(rest):
        tiles = _tiles(project)
        boxes = tiles.drop_duplicates("tile_id").set_index("tile_id") if not tiles.empty else pd.DataFrame()
        known = rest[rest.tile_id.isin(set(boxes.index))] if len(boxes) else rest.iloc[:0]
        known = frames.in_field_frame(known, tiles, project.context_px)  # every position in field-picture pixels
        for tile_id, group in known.groupby("tile_id", sort=False):
            tile = boxes.loc[tile_id]
            x0, y0 = max(0, int(tile["x_px"]) - project.context_px), max(0, int(tile["y_px"]) - project.context_px)
            jobs.append((str(tile["stain"]), str(tile["core_id"]), group.index, group.x.astype(float).to_numpy() + x0, group.y.astype(float).to_numpy() + y0))

    def position_drops(part):
        stain, core_id, index, xs, ys = part
        codes = codes_at(project, stain, core_id, xs, ys, stamp_of(stain))
        return list(index[(codes != COUNTED) & (codes != OUTSIDE)])

    for part in by_core.items():
        drop += named_drops(part)
    for part in jobs:
        drop += position_drops(part)
    return drop


# ---------------------------------------------------------------- the AI reviewer's areas

def import_reviewer(project, path: Path, kind: str = "2", source: str = REVIEWER) -> int:
    """The hand-drawn areas of the earlier AI-review rounds (the `manual_exclusions` format: core image path ->
    [{"polygon_core_px", "exclusion_id", …}]) as people's areas of source `source` (type `kind`, default artifact),
    dated when the file was written, so later edits (clearing them, say) win. Areas already imported are not added
    again. Returns how many were added."""
    from stainid.labels import exclusions as marks  # noqa: PLC0415
    from stainid.labels import store  # noqa: PLC0415

    found = json.loads(Path(path).read_text())
    tiles = _tiles(project)
    by_image = tiles.drop_duplicates("image_path").set_index("image_path") if "image_path" in tiles else pd.DataFrame()
    labels = _labels(project)
    have = set(labels.id[labels.source == source]) if not labels.empty else set()
    when = Path(path).stat().st_mtime
    records = []
    for image, polygons in found.items():
        if image not in by_image.index:
            continue
        tile = by_image.loc[image]
        for n, polygon in enumerate(polygons):
            ring = [[float(x), float(y)] for x, y in polygon["polygon_core_px"]]
            if len(ring) < 3:
                continue
            ident = f"{TASK}:{tile['core_id']}:{polygon.get('exclusion_id') or n}"
            if ident in have:
                continue
            records.append({"id": ident, "source": source, "stain": str(tile["stain"]), "task": TASK, "tile_id": str(tile["core_id"]),
                            "x": None, "y": None, "points": [ring], "value": marks.value_text([False], [kind]), "time": when,
                            "session": ""})
    if records:
        store.append(project.output("reviews"), records)
    return len(records)
