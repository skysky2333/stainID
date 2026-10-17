"""The v2 candidate store as the annotation pages read it: every candidate the detection proposed (accepted and
rejected), with its stable id, position, outline and the AI's call.

The store is written by the v2 detection stages (see the v2 contract): one table per stain and core image,
`data/analysis/v2/candidates/<stain>/<core_id>.parquet` (`.csv.gz` without pyarrow), one row per candidate:
`candidate_id` (f"{stain}:{core_id}:{round(x_core)}:{round(y_core)}"), `x_core`, `y_core` (core-image pixels), `tile_id`,
`area_um2`, `radius_px`, `sources`, `proposal_threshold` (the level it was proposed at, as a multiple of the slide's stain
threshold), raw feature columns, `outline` / `sam_outline` (polygons in core pixels: "x0 y0 x1 y1 …", rings of a
`sam_outline` separated by ";"), `sam_status`, the three Segment Anything masks of the candidate's point prompt
(`sam_masks`, small to large, with `sam_scores` and the default one `sam_default`), and the stage-4 call: `probability`,
`model_id`, `decision` (`accepted` or `rejected`), `reason` (`accepted`, `rejected_by_model`, `rule:<why>`), `class`,
`subtype` (label-vocabulary ids).

This module only reads it (through `stainid.candidates` when that module is installed, else straight from the files) and
turns it into what the labelling pages need: positions in a field picture (`stainid.labels.frames`), the AI's call in
the stain's label classes, and old labels matched onto candidate ids by position. `from_pool` writes a store from the
current candidate pool and detections, for trying the annotation rounds out before the v2 detection has run.
"""
from __future__ import annotations

import json
import os
import threading
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from stainid import cache
from stainid.labels import frames
from stainid.labels import vocabulary as vocab
from stainid.labels.tasks import NEGATIVE
from stainid.project import Project
from stainid.tables import write_text_atomic

FOLDER = Path("data/analysis/v2/candidates")
SUFFIXES = (".parquet", ".csv.gz")
ACCEPTED = "accepted"
MATCH_UM = {"NeuN": 4.0, "6E10": 6.0, "AT8": 4.0}  # as the contract's MATCH_RADIUS_UM: how far an old label may sit from a candidate's centre to be about it
DEFAULT_MATCH_UM = 5.0
POLYGONS = ("outline", "sam_outline")
MASKS = ("sam_masks", "sam_scores", "sam_default")
# columns that are not measurements of the object (everything else numeric is a raw feature)
NOT_FEATURES = {"candidate_id", "x_core", "y_core", "tile_id", "x_tile", "y_tile", "bbox", "sources", "proposal_threshold", "outline",
                "sam_outline", "sam_status", "sam_masks", "sam_scores", "sam_default", "x_field", "y_field", "probability", "model_id", "decision", "reason", "class", "subtype", "core_id", "stain",
                "radius_px", "x", "y", "own"}


def folder(project: Project, stain: str | None = None) -> Path:
    try:
        base = project.output("v2") / "candidates"
    except KeyError:
        base = project.root / FOLDER
    return base / stain if stain else base


def candidate_id(stain: str, core_id: str, x_core: float, y_core: float) -> str:
    return f"{stain}:{core_id}:{round(float(x_core))}:{round(float(y_core))}"


def path(project: Project, stain: str, core_id: str) -> Path | None:
    """The store file of one core image (None when the detection has not written it)."""
    names = cache.names(folder(project, stain))
    for suffix in SUFFIXES:
        if f"{core_id}{suffix}" in names:
            return folder(project, stain) / f"{core_id}{suffix}"
    return None


def cores(project: Project, stain: str) -> frozenset[str]:
    """The core images of this stain that have candidates."""
    out = set()
    for name in cache.names(folder(project, stain)):
        for suffix in SUFFIXES:
            if name.endswith(suffix) and not name.startswith("."):
                out.add(name[: -len(suffix)])
    return frozenset(out)


def _missing(value) -> bool:
    return value is None or (isinstance(value, float) and np.isnan(value)) or (isinstance(value, str) and not value.strip())


def _ring(value) -> np.ndarray:
    points = np.asarray(value, dtype=float) if len(value) else np.zeros((0, 2))
    if points.ndim == 1:
        points = points[: len(points) // 2 * 2].reshape(-1, 2)
    return points.reshape(-1, 2)


def _area(ring: np.ndarray) -> float:
    x, y = ring[:, 0], ring[:, 1]
    return abs(float((x * np.roll(y, -1) - np.roll(x, -1) * y).sum())) / 2


def _polygon(value) -> list[list[float]] | None:
    """A stored polygon as [[x, y], …]: from "x0 y0 x1 y1 …" (several rings separated by ";": the largest), a JSON
    text, a list of [x, y] or a flat list."""
    if _missing(value):
        return None
    if isinstance(value, str):
        text = value.strip()
        if text[0] in "[{":
            try:
                value = json.loads(text)
            except ValueError:
                return None
        else:
            rings = [_ring([float(v) for v in part.replace(",", " ").split()]) for part in text.split(";") if part.strip()]
            value = max(rings, key=_area).tolist() if rings else []
    ring = _ring(value)
    return ring.round(1).tolist() if len(ring) >= 3 else None


def _polygons(value) -> list[list[list[float]]] | None:
    """A list of polygons (the Segment Anything masks): a list, a JSON text, or polygon texts separated by "|"."""
    if _missing(value):
        return None
    if isinstance(value, str):
        text = value.strip()
        if text[0] == "[":
            try:
                value = json.loads(text)
            except ValueError:
                return None
        else:
            value = text.split("|")
    found = [_polygon(v) for v in value]
    return found if any(found) else None


def _numbers(value) -> list[float] | None:
    if _missing(value):
        return None
    if isinstance(value, str):
        text = value.strip()
        value = json.loads(text) if text[0] == "[" else text.replace("|", " ").replace(";", " ").replace(",", " ").split()
    return [float(v) for v in value]


def encode(polygon) -> str:
    """A polygon as the store writes it: "x0 y0 x1 y1 …" (core pixels, whole numbers)."""
    if isinstance(polygon, str):
        return polygon
    return " ".join(str(int(round(v))) for point in polygon for v in point) if polygon else ""


def _module_reader():
    """`stainid.candidates.read_core(project, stain, core_id)` of the v2 detection, when it is installed."""
    try:
        from stainid import candidates as v2  # noqa: PLC0415 - optional: written by the v2 detection
    except ImportError:
        return None
    return getattr(v2, "read_core", None)


def _read_file(target: Path, columns: list[str] | None = None) -> pd.DataFrame:
    if target.suffix == ".parquet":
        return pd.read_parquet(target, columns=columns)
    return pd.read_csv(target, usecols=(lambda c: c in columns) if columns else None, dtype={"tile_id": str, "candidate_id": str, "sources": str, "decision": str, "reason": str, "class": str,
                                      "subtype": str, "sam_status": str, "model_id": str})


def _normalised(frame: pd.DataFrame, stain: str, core_id: str) -> pd.DataFrame:
    frame = frame.copy()
    for column, default in (("decision", ""), ("reason", ""), ("class", ""), ("subtype", ""), ("sources", ""), ("sam_status", ""), ("model_id", "")):
        frame[column] = frame[column].fillna(default).astype(str) if column in frame else default
    for column in ("probability", "proposal_threshold", "area_um2", "radius_px"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce") if column in frame else np.nan
    for column in POLYGONS:
        frame[column] = [_polygon(v) for v in frame[column]] if column in frame else None
    frame["sam_masks"] = [_polygons(v) for v in frame.sam_masks] if "sam_masks" in frame else None
    frame["sam_scores"] = [_numbers(v) for v in frame.sam_scores] if "sam_scores" in frame else None
    frame["sam_default"] = pd.to_numeric(frame.sam_default, errors="coerce") if "sam_default" in frame else np.nan
    if "candidate_id" not in frame:
        frame["candidate_id"] = [candidate_id(stain, core_id, x, y) for x, y in zip(frame.x_core, frame.y_core)]
    frame["candidate_id"] = frame.candidate_id.astype(str)
    frame["x_core"], frame["y_core"] = frame.x_core.astype(float), frame.y_core.astype(float)
    frame["tile_id"] = frame.tile_id.astype(str) if "tile_id" in frame else ""
    frame["core_id"] = core_id
    return frame.reset_index(drop=True)


def radius(row, pixel_size_um: float) -> float:
    """A candidate's radius in pixels: as stored, else from its area, else 8."""
    stored = float(row.radius_px) if pd.notna(row.radius_px) else np.nan
    if np.isfinite(stored) and stored > 0:
        return stored
    area = float(row.area_um2) if pd.notna(row.area_um2) else np.nan
    return float(np.sqrt(area / np.pi) / pixel_size_um) if np.isfinite(area) and area > 0 else 8.0


POSITIONS = ["candidate_id", "x_core", "y_core", "tile_id"]
CALLS = [*POSITIONS, "decision", "class", "subtype"]
_WHOLE: OrderedDict[str, tuple[tuple, pd.DataFrame]] = OrderedDict()
_WHOLE_LIMIT = 8  # a whole NeuN or AT8 core with its outlines and three masks parsed is a few hundred MB to a GB in memory
_WHOLE_LOCK = threading.Lock()


def read_core(project: Project, stain: str, core_id: str, columns: list[str] | None = None) -> pd.DataFrame | None:
    """Every candidate of one core image (None when it has none). Whole tables are kept for the last few cores read
    (until their file changes); with `columns` (e.g. `POSITIONS`, `CALLS`) only those are read and parsed, for passes over
    many cores. Callers must not change it."""
    target = path(project, stain, core_id)
    if target is None:
        return None
    reader = _module_reader()
    if columns is not None:
        found = reader(project, stain, core_id, list(columns)) if reader else _read_file(target, list(columns))
        return _normalised(found, stain, core_id)
    stamp = cache.signature(target)
    with _WHOLE_LOCK:
        kept = _WHOLE.get(str(target))
        if kept is not None and kept[0] == stamp:
            _WHOLE.move_to_end(str(target))
            return kept[1]
    found = _normalised(reader(project, stain, core_id) if reader else _read_file(target), stain, core_id)
    with _WHOLE_LOCK:
        _WHOLE[str(target)] = (stamp, found)
        _WHOLE.move_to_end(str(target))
        while len(_WHOLE) > _WHOLE_LIMIT:
            _WHOLE.popitem(last=False)
    return found


EMPTY = ["candidate_id", "x_core", "y_core", "tile_id", "core_id", "radius_px", "area_um2", "probability", "decision", "reason", "class",
         "subtype", "proposal_threshold", "sources", "outline", "sam_outline", "sam_status"]


def for_cores(project: Project, stain: str, core_ids) -> pd.DataFrame:
    parts = [f for c in sorted(set(core_ids)) if (f := read_core(project, stain, c)) is not None and len(f)]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=EMPTY)


def at_level(frame: pd.DataFrame, level: float | None) -> pd.DataFrame:
    """The candidates a session at this "how generous" level shows: proposed at `level` times the slide's stain threshold
    or higher (candidates with no level, e.g. from Cellpose, always). None: every candidate."""
    if level is None or frame.empty:
        return frame
    return frame[frame.proposal_threshold.isna() | (frame.proposal_threshold >= float(level) - 1e-9)]


def in_tiles(frame: pd.DataFrame, tiles: pd.DataFrame, context_px: int) -> pd.DataFrame:
    """The candidates that belong to these tiles (their own `tile_id`), with `x`, `y` in their field picture."""
    if frame.empty:
        return frame.assign(x=[], y=[])
    boxes = tiles.drop_duplicates("tile_id").set_index("tile_id")
    mine = frame[frame.tile_id.isin(set(boxes.index))]
    origin = {t: frames.field_origin(boxes.loc[t], context_px) for t in set(mine.tile_id)}
    return mine.assign(x=[x - origin[t][0] for x, t in zip(mine.x_core, mine.tile_id)],
                       y=[y - origin[t][1] for y, t in zip(mine.y_core, mine.tile_id)])


def in_field(project: Project, stain: str, tile: dict, context_px: int, level: float | None = None) -> pd.DataFrame:
    """Every candidate whose centre lies in this field's picture (the tile and its margin), with `x`, `y` and the
    outlines in that picture's pixels and `own` (the candidate is this tile's, not a neighbour's seen in the margin)."""
    found = read_core(project, stain, str(tile["core_id"]))
    if found is None or found.empty:
        return pd.DataFrame(columns=[*EMPTY, "x", "y", "own"])
    x0, y0 = frames.field_origin(tile, context_px)
    x1 = int(tile["x_px"]) + int(tile["width_px"]) + context_px
    y1 = int(tile["y_px"]) + int(tile["height_px"]) + context_px
    near = at_level(found[(found.x_core >= x0) & (found.x_core < x1) & (found.y_core >= y0) & (found.y_core < y1)], level)
    shift = lambda poly: None if not poly else [[px - x0, py - y0] for px, py in poly]  # noqa: E731
    return near.assign(x=near.x_core - x0, y=near.y_core - y0, own=near.tile_id == str(tile["tile_id"]),
                       outline=[shift(p) for p in near.outline], sam_outline=[shift(p) for p in near.sam_outline],
                       sam_masks=[None if not m else [shift(p) for p in m] for m in near.sam_masks]).reset_index(drop=True)


def _get(row, name: str):
    return row.get(name) if isinstance(row, dict) else getattr(row, name, None)


def masks(row) -> tuple[list[list[list[float]] | None], int]:
    """The outlines one can switch between for a candidate, small to large, and the one shown first: its three
    Segment Anything masks and their default when the store has them, else its one outline (Segment Anything's, else
    the candidate's own)."""
    found = _get(row, "sam_masks")
    if found and any(found):
        default = _get(row, "sam_default")
        scores = _get(row, "sam_scores") or []
        if default is None or pd.isna(default) or not 0 <= int(default) < len(found) or not found[int(default)]:
            usable = [i for i, m in enumerate(found) if m]
            default = max(usable, key=lambda i: scores[i] if i < len(scores) else 0.0)
        return list(found), int(default)
    one = _get(row, "sam_outline") or _get(row, "outline")
    return ([one] if one else []), 0


def ai_outline(row) -> list[list[float]] | None:
    """The AI's outline of a candidate: the default of its masks (`masks`)."""
    found, default = masks(row)
    return found[default] if found else None


def accepted(decision: str) -> bool:
    return str(decision) == ACCEPTED


def by_model(frame: pd.DataFrame) -> pd.Series:
    """Candidates the model decided (accepted, or rejected by its score rather than by a rule)."""
    return (frame.decision == ACCEPTED) | frame.reason.isin(["rejected_by_model"]) | (frame.decision == "rejected_by_model")


def negative_class(vocabulary: vocab.Vocabulary) -> str:
    """The class meaning "not the stain's object" (what "remove" records): the stain's usual negative call, else its
    first shown negative class."""
    identity = vocabulary.resolve(NEGATIVE.get(vocabulary.stain, ""))[0]
    if vocabulary.identity(identity) is not None:
        return vocabulary.identity(identity).id
    return next((c.id for c in vocabulary.visible if c.role == vocab.NEGATIVE), "")


def ai_call(vocabulary: vocab.Vocabulary, decision: str, klass: str, subtype: str) -> tuple[str, str]:
    """The AI's call on a candidate as (identity, subtype) ids of the stain's label classes: accepted ones their class
    (the stain's first positive class when the class is not one of the labels'), rejected ones the negative class."""
    if not accepted(decision):
        return negative_class(vocabulary), ""
    identity, sub = vocabulary.resolve(klass, subtype)
    found = vocabulary.identity(identity) if identity else None
    if found is None or found.role != vocab.POSITIVE:
        identity, sub = vocabulary.resolve(klass)  # a model class like "compact": the old words map it
        found = vocabulary.identity(identity) if identity else None
        if found is None or found.role != vocab.POSITIVE:
            return next((c.id for c in vocabulary.classes if c.role == vocab.POSITIVE), ""), ""
    return found.id, sub if sub and found.subtype(sub) else ""


def match_px(stain: str, pixel_size_um: float) -> float:
    return MATCH_UM.get(stain, DEFAULT_MATCH_UM) / max(1e-6, pixel_size_um)


_LIGHT: OrderedDict[str, tuple[tuple, dict]] = OrderedDict()
_LIGHT_LIMIT = 1200  # every core image of a cohort: a core's ids, positions and calls are a few MB


def _light(project: Project, stain: str, core_id: str) -> dict | None:
    """One core's candidates as matching needs them, compactly: ids (bytes), positions (float32) and the call of each
    as a code into the core's few (decision, class, subtype) combinations. Kept until the store file changes."""
    target = path(project, stain, core_id)
    if target is None:
        return None
    stamp = cache.signature(target)
    with _WHOLE_LOCK:
        kept = _LIGHT.get(str(target))
        if kept is not None and kept[0] == stamp:
            _LIGHT.move_to_end(str(target))
            return kept[1]
    frame = read_core(project, stain, core_id, CALLS)
    if frame is None or frame.empty:
        found = None
    else:
        combos = pd.MultiIndex.from_arrays([frame.decision, frame["class"], frame.subtype])
        codes, uniques = pd.factorize(combos)
        found = {"ids": frame.candidate_id.to_numpy().astype("S"), "xy": np.c_[frame.x_core.to_numpy(), frame.y_core.to_numpy()].astype(np.float32),
                 "codes": codes.astype(np.int32), "combos": list(uniques)}
    with _WHOLE_LOCK:
        _LIGHT[str(target)] = (stamp, found)
        _LIGHT.move_to_end(str(target))
        while len(_LIGHT) > _LIGHT_LIMIT:
            _LIGHT.popitem(last=False)
    return found


def _nearest(project: Project, stain: str, labels: pd.DataFrame, tiles: pd.DataFrame):
    """For each core with labels: (the core's compact candidates, the labels' index, the candidate row of each label or
    -1) — the nearest candidate within `MATCH_UM` of each label (tile_id, x, y in its field picture, `labels.frames`)."""
    if labels.empty:
        return
    boxes = tiles.drop_duplicates("tile_id").set_index("tile_id") if not tiles.empty else pd.DataFrame(columns=["core_id"])
    have = cores(project, stain)
    placed = labels[labels.x.notna() & (labels.tile_id.isin(set(boxes.index)) | labels.tile_id.isin(have))]  # a field, or a whole core
    if placed.empty:
        return
    radius = match_px(stain, project.pixel_size_um)
    core_of = boxes.core_id.astype(str).to_dict()
    for core_id, group in placed.groupby(placed.tile_id.map(lambda t: core_of.get(t, t))):
        if core_id not in have:
            continue
        found = _light(project, stain, core_id)
        if found is None:
            continue
        origin = np.array([frames.field_origin(boxes.loc[t], project.context_px) if t in core_of else (0, 0) for t in group.tile_id], float)
        points = np.c_[group.x.astype(float).to_numpy(), group.y.astype(float).to_numpy()] + origin
        distance, index = cKDTree(found["xy"]).query(points, distance_upper_bound=radius)
        yield found, group.index, np.where(np.isfinite(distance), index, -1)


def match(project: Project, stain: str, labels: pd.DataFrame, tiles: pd.DataFrame) -> pd.Series:
    """The candidate each label (tile_id, x, y in its field picture, `labels.frames`) is about: the nearest candidate of
    the store within `MATCH_UM`, else None. Labels made before candidate ids existed are matched this way."""
    out = pd.Series(None, index=labels.index, dtype=object)
    for found, index, rows in _nearest(project, stain, labels, tiles):
        hit = rows >= 0
        out.loc[index[hit]] = [v.decode() for v in found["ids"][rows[hit]]]
    return out


def calls_at(project: Project, stain: str, labels: pd.DataFrame, tiles: pd.DataFrame) -> pd.Series:
    """The AI's call (identity id) of the candidate each label is about (`match`), None where there is no candidate."""
    out = pd.Series(None, index=labels.index, dtype=object)
    vocabulary = vocab.of(project, stain)
    for found, index, rows in _nearest(project, stain, labels, tiles):
        hit = rows >= 0
        calls = [ai_call(vocabulary, *combo)[0] for combo in found["combos"]]
        out.loc[index[hit]] = [calls[c] for c in found["codes"][rows[hit]]]
    return out


def feature_names(frame: pd.DataFrame) -> list[str]:
    """The raw feature columns of a store table (numeric columns that are not ids, positions, outlines or calls)."""
    return [c for c in frame.columns if c not in NOT_FEATURES and pd.api.types.is_numeric_dtype(frame[c])]


# ---------- a store from the current candidate pool (for trying the rounds out before the v2 detection) ----------

def write_core(project: Project, stain: str, core_id: str, frame: pd.DataFrame) -> Path:
    target = folder(project, stain) / f"{core_id}.csv.gz"
    target.parent.mkdir(parents=True, exist_ok=True)
    out = frame.copy()
    for column in POLYGONS:
        if column in out:
            out[column] = [encode(p) for p in out[column]]
    if "sam_masks" in out:
        out["sam_masks"] = ["|".join(encode(p) for p in m) if m else "" for m in out.sam_masks]
    if "sam_scores" in out:
        out["sam_scores"] = ["|".join(f"{v:.3f}" for v in m) if m else "" for m in out.sam_scores]
    temporary = target.with_name(f".{target.name}.tmp")
    out.to_csv(temporary, index=False, compression="gzip")
    os.replace(temporary, target)
    return target


def _circle(x: float, y: float, r: float, n: int = 16) -> list[list[float]]:
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.c_[x + r * np.cos(t), y + r * np.sin(t)].round(1).tolist()


def from_pool(project: Project, stain: str, tiles: pd.DataFrame) -> dict[str, int]:
    """Write a candidate store from the candidate pool of the prepared fields (`labels.pool`) and the AI's calls there,
    one file per core: what the annotation rounds read, until the v2 detection writes the real one. Returns the number of
    candidates per core."""
    from stainid.labels import pool

    vocabulary = vocab.of(project, stain)
    boxes = tiles[tiles.stain == stain].drop_duplicates("tile_id").set_index("tile_id")
    rows: dict[str, list[dict]] = {}
    for tile_id in sorted(pool.prepared(project, stain) & set(boxes.index)):
        found = pool.load_field(project, stain, tile_id)
        tile = boxes.loc[tile_id]
        x0, y0 = frames.field_origin(tile, project.context_px)
        names = pool.feature_names(project, stain) or []
        for r in found.to_dict("records"):
            xc, yc = x0 + float(r["x"]), y0 + float(r["y"])
            inside = int(tile.x_px) <= xc < int(tile.x_px) + int(tile.width_px) and int(tile.y_px) <= yc < int(tile.y_px) + int(tile.height_px)
            identity, subtype = vocabulary.resolve(r.get("ai_value"))
            positive = vocabulary.role(identity) == vocab.POSITIVE
            radius = float(r.get("radius_px") or 8.0)
            sizes = [_circle(xc, yc, radius * f) for f in (0.6, 1.0, 1.6)]  # stand-ins for the three Segment Anything masks
            rows.setdefault(str(tile.core_id), []).append({
                "candidate_id": candidate_id(stain, tile.core_id, xc, yc), "x_core": round(xc, 1), "y_core": round(yc, 1), "tile_id": tile_id,
                "own": inside, "area_um2": round(float(np.pi * (radius * project.pixel_size_um) ** 2), 1), "radius_px": radius,
                "sources": str(r.get("candidate_source") or "pool"), "proposal_threshold": 1.0, "outline": sizes[1],
                "sam_outline": sizes[1], "sam_status": "ok", "sam_masks": sizes, "sam_scores": [0.8, 0.9, 0.7], "sam_default": 1,
                "probability": r.get("ai_probability"), "model_id": "pool", "decision": ACCEPTED if positive else "rejected",
                "reason": ACCEPTED if positive else "rejected_by_model", "class": identity if positive else "",
                "subtype": subtype if positive else "", **{k: r.get(k) for k in names}})
    out = {}
    for core_id, found in rows.items():
        frame = pd.DataFrame(found).sort_values("own", ascending=False).drop_duplicates("candidate_id")  # a field's own copy first
        frame = frame.drop(columns=["own"]).reset_index(drop=True)
        write_core(project, stain, core_id, frame)
        out[core_id] = len(frame)
    write_text_atomic(folder(project, stain) / ".from_pool.json", json.dumps({"cores": out}))
    return out


# ---------- training from the decisions on candidates ----------

EXTRA = ("equivalent_diameter_um", "normalized_inner_dab", "soma_dab_p90", "soma_area_um2", "candidate_source")  # read by training too


def model_inputs(frame: pd.DataFrame) -> list[str]:
    """The model's input features a store table holds: its `m_<feature>` columns (without the prefix), else every raw
    feature column."""
    named = [c[2:] for c in frame.columns if c.startswith("m_")]
    return named or feature_names(frame)


def training_rows(project: Project, stain: str, objects: pd.DataFrame, tiles: pd.DataFrame,
                  names: list[str] | None) -> tuple[dict, list[str] | None]:
    """The features of the candidates these labelled objects (tile_id, x, y in field pictures) are about, for training:
    {row index: {feature: value, …, x, y, tma, candidate_id}}, and the feature names (`names`, else the store's own model
    inputs). Candidates whose table lacks one of the names are left out."""
    if objects.empty:
        return {}, names
    matched = match(project, stain, objects, tiles).dropna()
    if matched.empty:
        return {}, names
    tma = tiles.drop_duplicates("tile_id").set_index("tile_id").tma.astype(str)
    out = {}
    for core_id, group in matched.groupby(matched.map(lambda c: ":".join(str(c).split(":")[1:-2]))):
        found = read_core(project, stain, core_id)
        if found is None or found.empty:
            continue
        names = names or model_inputs(found)
        columns = {n: f"m_{n}" if f"m_{n}" in found else n for n in names}
        if any(c not in found for c in columns.values()):
            continue
        rows = found.drop_duplicates("candidate_id").set_index("candidate_id")
        for index, cid in group.items():
            row = rows.loc[cid]
            out[index] = {**{n: row[c] for n, c in columns.items()}, **{k: row[k] for k in EXTRA if k in row and k not in columns},
                          "x": float(objects.x[index]), "y": float(objects.y[index]), "tma": tma.get(objects.tile_id[index], ""), "candidate_id": cid}
    return out, names
