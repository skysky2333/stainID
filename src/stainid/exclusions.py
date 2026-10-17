"""The excluded areas: one resolved layer per core image and stain, which every view, the analysis, training and scoring
read. This module owns it; nothing else combines detection's areas with people's edits.

The layer is made from
- detection's automatic areas, the stage-2 mask (`<outputs.v2>/masks/<stain>/<core_id>.png`, 1/`SCALE` of the core image,
  bit flags `v2.proposals.BITS`): no tissue, linear artifacts, folds (6E10 and AT8; NeuN counts folded tissue) and the
  AT8 edge guard (the hand-drawn bit is not read: hand-drawn areas are edits, below);
- the edits, `exclude` labels in the label log (`labels.store`, written by `labels.exclusions`): rings, each with a type.
  Painting type T sets its pixels to T; clearing ("include", Shift-drag) sets them to counted, whatever detection said.
  The edits are applied in time order, so the latest edit wins at each pixel. Their sources are people (the field
  tool's brush, on a field: rings in field-picture pixels) and "AI reviewer" (the hand-drawn polygons of the earlier
  AI-review rounds, `import_reviewer`: rings in core pixels, the label's `tile_id` the core id).

The label log keeps every edit (who, when, which type) as history; only the resolved layer is ever drawn or used.

Codes (`CODES`): 0 counted, 1 fold or tissue damage, 2 artifact, 3 no tissue / empty, 4 out of focus or bad staining,
5 AT8 edge guard, `OUTSIDE` (255) outside every tile (not analysed; edits do not change it). The views draw `view`: the
same codes with the tissue outside every tile as 6, "not analysed" (from the core image's tissue at 1/8, as the tiling
finds it), so what is drawn matches what the analysis counts; an image without tiles is all "not analysed".

Read API: `resolved(project, core_id, stain) -> Layer | None` (None before stage 2 made the core's mask), `digest` (changes
exactly when the layer can change: for manifests), `shares` / `candidate_fractions` (candidates' excluded shares),
`codes_at`, `drop_excluded` (labels on excluded objects left out of scoring, agreement and training) and `signature`
(for caches of results made from the layer).
"""
from __future__ import annotations

import hashlib
import json
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from stainid import cache

SCALE = 8  # core-image px per layer px (stainid.v2.proposals.MASK_SCALE)
TISSUE, LINEAR, FOLD, MANUAL, EDGE_GUARD, TILE = 1, 2, 4, 8, 16, 32  # stainid.v2.proposals.BITS
COUNTED, OUTSIDE = 0, 255
CODES = {1: "fold or tissue damage", 2: "artifact (scratch, bubble, debris, linear artifact)", 3: "no tissue / empty",
         4: "out of focus or bad staining", 5: "AT8 edge guard (brown rim along an edge)",
         6: "not analysed: tissue in no analysed tile (an image QC calls empty, a square with too little tissue)"}
NOT_ANALYSED = 6  # only in the views' codes (`view`): the analysis counts nothing outside the tiles anyway
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

    def key(self) -> tuple:
        return (round(self.time, 6), self.source, self.id, self.kinds, self.holes,
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
                                                             tuple(np.asarray(g, np.float32) + shift for g in rings), tuple(holes), tuple(kinds)))
        for edits in out.values():
            edits.sort(key=lambda e: (e.time, e.source, e.id))
        return out

    return cache.memo(("exclusion edits", str(project.root)), [_log(project), *_tile_paths(project)], make)


def edits(project, core_id: str, stain: str) -> list[Edit]:
    """The edits of one core image and stain, oldest first (the history: who, when, which type)."""
    return all_edits(project).get((stain, str(core_id)), [])


# ---------------------------------------------------------------- the layer

def mask_path(project, core_id: str, stain: str) -> Path:
    from stainid import candidates  # noqa: PLC0415

    return candidates.root(project) / "masks" / stain / f"{core_id}.png"


def masks_folder(project, stain: str) -> Path:
    from stainid import candidates  # noqa: PLC0415

    return candidates.root(project) / "masks" / stain


def automatic(bits: np.ndarray, stain: str) -> np.ndarray:
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


def apply(codes: np.ndarray, edit: Edit) -> None:
    """One edit onto the codes, in place: each type's area set to that type, a cleared area to counted."""
    analysed = codes != OUTSIDE
    for kind in dict.fromkeys(edit.kinds):
        rings = [r for r, k in zip(edit.rings, edit.kinds) if k == kind]
        holes = [h for h, k in zip(edit.holes, edit.kinds) if k == kind]
        area = rasterize(rings, holes, codes.shape) & analysed
        codes[area] = COUNTED if kind == INCLUDE else int(kind) if str(kind).isdigit() else 1


@dataclass
class Layer:
    """The resolved excluded areas of one core image and stain at 1/`scale`: `codes` (`CODES`, 0 counted, `OUTSIDE`),
    `bits` (the stage-2 mask as made) and `digest` (`digest`)."""
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


def _mask_digest(path: Path) -> str | None:
    """The stage-2 mask's content digest, kept until the file changes."""
    if not path.exists():
        return None
    return cache.memo(("exclusion mask digest", str(path)), [path], lambda: hashlib.sha1(path.read_bytes()).hexdigest())


def digest(project, core_id: str, stain: str) -> str | None:
    """What the core's layer is made from, as a short text: changes exactly when the stage-2 mask's content or the core's
    edits change (for the manifests of the stages that use the layer); None before stage 2 made the mask."""
    found = _mask_digest(mask_path(project, core_id, stain))
    if found is None:
        return None
    keys = [e.key() for e in edits(project, core_id, stain)]
    return hashlib.sha1(json.dumps([found, keys], default=str).encode()).hexdigest()[:16]


def digests(project, stain: str, core_ids) -> dict[str, str | None]:
    return {str(c): digest(project, str(c), stain) for c in core_ids}


_LAYERS: OrderedDict[tuple, Layer] = OrderedDict()
_LAYERS_LIMIT = 24
_SHARES: OrderedDict[tuple, tuple[tuple, dict]] = OrderedDict()  # (project, stain, core) -> (made from, what was worked out)
_LOCK = threading.Lock()


def resolved(project, core_id: str, stain: str) -> Layer | None:
    """The resolved layer of one core image and stain (detection's areas, then every edit in time order); None before
    stage 2 made the core's mask. Kept (a few cores) until the mask or the core's edits change."""
    path = mask_path(project, core_id, stain)
    made_from = digest(project, core_id, stain)
    if made_from is None:
        return None
    key = (str(project.root), stain, str(core_id))
    with _LOCK:
        kept = _LAYERS.get(key)
        if kept is not None and kept.digest == made_from:
            _LAYERS.move_to_end(key)
            return kept
    bits = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if bits is None:
        return None
    bits = bits if bits.ndim == 2 else bits[..., 0]
    codes = automatic(bits, stain)
    found = edits(project, core_id, stain)
    for edit in found:
        apply(codes, edit)
    layer = Layer(codes, bits, made_from, SCALE, found)
    with _LOCK:
        _LAYERS[key] = layer
        _LAYERS.move_to_end(key)
        while len(_LAYERS) > _LAYERS_LIMIT:
            _LAYERS.popitem(last=False)
    return layer


_TISSUE: OrderedDict[tuple, np.ndarray] = OrderedDict()  # (image, its signature, shape) -> tissue cells


def _tissue(project, image_path: str, shape: tuple[int, int] | None) -> np.ndarray | None:
    """The core image's tissue at 1/`SCALE` as the tiling finds it (`qc.core_qc.tissue_mask` of the 1/8 picture),
    on a grid of `shape` (else its own); None when the image cannot be read."""
    from stainid.imaging.core_images import read_bgr  # noqa: PLC0415
    from stainid.qc.core_qc import tissue_mask  # noqa: PLC0415

    path = project.root / image_path
    key = (str(path), cache.signature(path), shape)
    with _LOCK:
        kept = _TISSUE.get(key)
    if kept is not None:
        return kept
    bgr = read_bgr(path, cv2.IMREAD_REDUCED_COLOR_8)
    if bgr is None:
        return None
    tissue = tissue_mask(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))[0].astype(np.uint8)
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
    """What the views draw of one core image and stain: the resolved codes with tissue outside every tile as
    `NOT_ANALYSED`, the scale and a version (changes with the layer)."""
    codes: np.ndarray
    scale: int
    version: str


def view(project, core_id: str, stain: str) -> View | None:
    """The layer as the Cores view and the annotation pages draw it (`resolved`, plus "not analysed" tissue); an image
    without any analysed tile is all "not analysed"; None before stage 2 made the mask of an image with tiles."""
    image, tiled = _image_of(project, core_id, stain)
    layer = resolved(project, core_id, stain)
    if layer is None and (tiled or image is None):
        return None
    if layer is not None:
        codes = layer.codes.copy()
        tissue = _tissue(project, image, codes.shape) if image else None
        if tissue is not None:
            codes[(codes == OUTSIDE) & tissue] = NOT_ANALYSED
        return View(codes, layer.scale, layer.digest)
    tissue = _tissue(project, image, None)
    if tissue is None:
        return None
    codes = np.where(tissue, NOT_ANALYSED, OUTSIDE).astype(np.uint8)
    return View(codes, SCALE, "untiled-" + hashlib.sha1(repr(cache.signature(project.root / image)).encode()).hexdigest()[:12])


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

def _known(project, stain: str, core_id: str) -> dict | None:
    """What was worked out about one core's objects (candidate shares, codes at positions), kept until its layer or its
    candidate table changes; None before stage 2."""
    from stainid import candidates  # noqa: PLC0415

    made_from = digest(project, core_id, stain)
    if made_from is None:
        return None
    path = candidates.path(project, stain, core_id, "candidates")
    stamp = (made_from, cache.signature(path))
    key = (str(project.root), stain, str(core_id))
    with _LOCK:
        kept = _SHARES.get(key)
        if kept is None or kept[0] != stamp:
            kept = (stamp, {})
            _SHARES[key] = kept
        _SHARES.move_to_end(key)
        while len(_SHARES) > 4096:
            _SHARES.popitem(last=False)
    return kept[1]


def shares(project, stain: str, core_id: str, ids) -> dict[str, float]:
    """The excluded share of these candidates of one core by the resolved layer (`Layer.fractions` of their stain
    outlines; a candidate the store no longer has by the cell at the position in its id); empty before stage 2. Only
    the candidates asked about are worked out (and kept until the layer or the store changes)."""
    from stainid import candidates  # noqa: PLC0415

    known = _known(project, stain, core_id)
    if known is None:
        return {}
    ids = [str(c) for c in dict.fromkeys(ids)]
    missing = [c for c in ids if ("share", c) not in known]
    if missing:
        layer = resolved(project, core_id, stain)
        rows = candidates.read_core(project, stain, core_id, ["candidate_id", "x_core", "y_core", "outline"])
        rows = rows[rows.candidate_id.isin(missing)].drop_duplicates("candidate_id") if len(rows) else rows
        found = dict(zip(rows.candidate_id, layer.fractions(rows.outline.tolist(), rows.x_core, rows.y_core))) if len(rows) else {}
        gone = [c for c in missing if c not in found]
        if gone:
            xy = np.array([candidates.parse_id(c)[2:] for c in gone], np.float64)
            codes = layer.at(xy[:, 0], xy[:, 1])
            found |= {c: float(code not in (COUNTED, OUTSIDE)) for c, code in zip(gone, codes)}
        with _LOCK:
            known.update({("share", c): float(v) for c, v in found.items()})
    return {c: known[("share", c)] for c in ids if ("share", c) in known}


def codes_at(project, stain: str, core_id: str, xs, ys) -> np.ndarray:
    """The layer's code at core-pixel positions (rounded to the pixel; `OUTSIDE` before stage 2)."""
    known = _known(project, stain, core_id)
    points = [(round(float(x)), round(float(y))) for x, y in zip(xs, ys)]
    if known is None:
        return np.full(len(points), OUTSIDE, np.uint8)
    missing = [p for p in dict.fromkeys(points) if ("at",) + p not in known]
    if missing:
        layer = resolved(project, core_id, stain)
        xy = np.array(missing, np.float64)
        codes = layer.at(xy[:, 0], xy[:, 1])
        with _LOCK:
            known.update({("at",) + p: int(c) for p, c in zip(missing, codes)})
    return np.array([known[("at",) + p] for p in points], np.uint8)


def candidate_fractions(project, stain: str, core_id: str, frame: pd.DataFrame | None = None) -> pd.Series:
    """Each candidate's excluded share by the resolved layer, by candidate id: of `frame` (candidate_id, x_core,
    y_core, outline: stage 4 before the store is written), else of every candidate in the store; empty before stage 2."""
    if frame is None:
        from stainid import candidates  # noqa: PLC0415

        rows = candidates.read_core(project, stain, core_id, ["candidate_id"])
        found = shares(project, stain, core_id, rows.candidate_id.tolist()) if len(rows) else {}
        return pd.Series(found, dtype=float)
    layer = resolved(project, core_id, stain)
    if layer is None:
        return pd.Series(dtype=float)
    outline = frame["outline"].tolist() if "outline" in frame else [None] * len(frame)
    return pd.Series(layer.fractions(outline, frame.x_core, frame.y_core), index=frame.candidate_id.to_numpy(object))


def drop_excluded(project, labels: pd.DataFrame, tasks=("class", "outline")) -> pd.DataFrame:
    """The labels (as the log stores them) without those about objects in excluded areas, what scoring, agreement and
    training read: a label naming a candidate goes when the candidate's excluded share is at least `EXCLUDED_AT`; one
    without (an object a person added, an older label) when the cell at its position is excluded."""
    if labels.empty or "task" not in labels:
        return labels
    about = labels.task.isin(list(tasks)).to_numpy()
    if not about.any():
        return labels
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
    for (stain, core_id), items in by_core.items():
        found = shares(project, stain, core_id, [c for _, c in items])
        drop += [i for i, c in items if found.get(c, 0.0) >= EXCLUDED_AT]
    rest = rows[(named.str.len() == 0).to_numpy() & rows.x.notna().to_numpy()]
    if len(rest):
        tiles = _tiles(project)
        boxes = tiles.drop_duplicates("tile_id").set_index("tile_id") if not tiles.empty else pd.DataFrame()
        known = rest[rest.tile_id.isin(set(boxes.index))] if len(boxes) else rest.iloc[:0]
        known = frames.in_field_frame(known, tiles, project.context_px)  # every position in field-picture pixels
        for tile_id, group in known.groupby("tile_id", sort=False):
            tile = boxes.loc[tile_id]
            x0, y0 = max(0, int(tile["x_px"]) - project.context_px), max(0, int(tile["y_px"]) - project.context_px)
            codes = codes_at(project, str(tile["stain"]), str(tile["core_id"]), group.x.astype(float) + x0, group.y.astype(float) + y0)
            drop += list(group.index[(codes != COUNTED) & (codes != OUTSIDE)])
    return labels.drop(index=drop) if drop else labels


# ---------------------------------------------------------------- the AI reviewer's areas

def import_reviewer(project, path: Path, kind: str = "2", source: str = REVIEWER) -> int:
    """The hand-drawn areas of the earlier AI-review rounds (the `manual_exclusions` format: core image path ->
    [{"polygon_core_px", "exclusion_id", …}]) as edits of source `source` (type `kind`, default artifact), dated when the
    file was written, so people's later edits (clearing them, say) win. Areas already imported are not added again.
    Returns how many were added."""
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
