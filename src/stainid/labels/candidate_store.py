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

This module does not read the files itself: `stainid.candidates` is the one reader (and writer) of the store, and the
outline drawn first is the one `candidates.outline_options` chooses. Here the rows are turned into what the labelling
pages need: outlines as point lists, positions in a field picture (`stainid.labels.frames`), the AI's call in the
stain's label classes, and old labels matched onto candidate ids by position (`match`, the one position matcher).
`from_pool` writes a store from the current candidate pool and detections, for trying the annotation rounds out before
the v2 detection has run.
"""
from __future__ import annotations

import json
import threading
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from stainid import cache
from stainid import candidates as v2
from stainid.labels import frames
from stainid.labels import vocabulary as vocab
from stainid.project import Project
from stainid.tables import write_text_atomic

ACCEPTED = "accepted"
MATCH_UM = v2.MATCH_RADIUS_UM  # how far an old label may sit from a candidate's centre to be about it
DEFAULT_MATCH_UM = v2.DEFAULT_MATCH_RADIUS_UM
POLYGONS = ("outline", "sam_outline")
MASKS = ("sam_masks", "sam_scores", "sam_default")
# columns that are not measurements of the object (everything else numeric is a raw feature)
NOT_FEATURES = {"candidate_id", "x_core", "y_core", "tile_id", "x_tile", "y_tile", "bbox", "sources", "proposal_threshold", "outline",
                "sam_outline", "sam_status", "sam_masks", "sam_scores", "sam_default", "x_field", "y_field", "probability", "model_id", "decision", "reason", "class", "subtype", "core_id", "stain",
                "radius_px", "x", "y", "own"}


def folder(project: Project, stain: str | None = None) -> Path:
    return v2.folder(project, stain)


candidate_id = v2.candidate_id


def path(project: Project, stain: str, core_id: str) -> Path | None:
    """The store file of one core image (None when the detection has not written it)."""
    return v2.stored(project, stain, core_id)


def cores(project: Project, stain: str) -> frozenset[str]:
    """The core images of this stain that have candidates."""
    return v2.cores(project, stain)


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


def _text_ring(part: str) -> np.ndarray:
    """One "x0 y0 x1 y1 …" ring as an (n, 2) array, parsed by numpy (not number by number)."""
    return _ring(np.array(part.replace(",", " ").split(), dtype=float))


def _polygon_array(value) -> np.ndarray | None:
    """A stored polygon as an (n, 2) array rounded to 0.1 px (`_polygon`), None when there is none."""
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
            rings = [_text_ring(part) for part in text.split(";") if part.strip()]
            ring = (rings[0] if len(rings) == 1 else max(rings, key=_area)) if rings else np.zeros((0, 2))
            return ring.round(1) if len(ring) >= 3 else None
    ring = _ring(value)
    return ring.round(1) if len(ring) >= 3 else None


def _polygon(value) -> list[list[float]] | None:
    """A stored polygon as [[x, y], …]: from "x0 y0 x1 y1 …" (several rings separated by ";": the largest), a JSON
    text, a list of [x, y] or a flat list."""
    ring = _polygon_array(value)
    return None if ring is None else ring.tolist()


def _polygon_arrays(value) -> list[np.ndarray | None] | None:
    """A list of polygons (the Segment Anything masks) as arrays: a list, a JSON text, or polygon texts separated by "|"."""
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
    found = [_polygon_array(v) for v in value]
    return found if any(f is not None for f in found) else None


def _polygons(value) -> list[list[list[float]]] | None:
    """A list of polygons (the Segment Anything masks): a list, a JSON text, or polygon texts separated by "|"."""
    found = _polygon_arrays(value)
    return None if found is None else [None if f is None else f.tolist() for f in found]


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


def _displayed(project: Project, stain: str, frame: pd.DataFrame) -> pd.DataFrame:
    """`outline_default` (which outline is shown first) filled from the stain's setting where the store has none (a store
    assembled before the setting, or the pool's)."""
    if "outline_default" in frame and frame.outline_default.ne("").all():
        return frame
    try:
        from stainid.v2.params import default_outline  # noqa: PLC0415

        setting = default_outline(project, stain)
    except (ImportError, ValueError, AttributeError):
        return frame
    current = frame.outline_default if "outline_default" in frame else pd.Series("", index=frame.index)
    return frame.assign(outline_default=current.where(current.ne(""), setting))


def _normalised(frame: pd.DataFrame, stain: str, core_id: str) -> pd.DataFrame:
    frame = frame.copy()
    for column, default in (("decision", ""), ("reason", ""), ("class", ""), ("subtype", ""), ("sources", ""), ("sam_status", ""), ("model_id", ""),
                            ("outline_default", "")):
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
    if columns is not None:
        return _displayed(project, stain, _normalised(v2.read_core(project, stain, core_id, list(columns)), stain, core_id))
    stamp = cache.signature(target)
    with _WHOLE_LOCK:
        kept = _WHOLE.get(str(target))
        if kept is not None and kept[0] == stamp:
            _WHOLE.move_to_end(str(target))
            return kept[1]
    found = _displayed(project, stain, _normalised(v2.read_core(project, stain, core_id), stain, core_id))
    with _WHOLE_LOCK:
        _WHOLE[str(target)] = (stamp, found)
        _WHOLE.move_to_end(str(target))
        while len(_WHOLE) > _WHOLE_LIMIT:
            _WHOLE.popitem(last=False)
    return found


EMPTY = ["candidate_id", "x_core", "y_core", "tile_id", "core_id", "radius_px", "area_um2", "probability", "decision", "reason", "class",
         "subtype", "proposal_threshold", "sources", "outline", "sam_outline", "sam_status"]


def for_cores(project: Project, stain: str, core_ids, columns: list[str] | None = None) -> pd.DataFrame:
    """Every candidate of these cores (with `columns`, e.g. `LIGHT`, only those read)."""
    parts = [f for c in sorted(set(core_ids)) if (f := read_core(project, stain, c, columns)) is not None and len(f)]
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


FIELD_COLUMNS = ["candidate_id", "x_core", "y_core", "tile_id", "radius_px", "area_um2", "probability", "threshold", "model_id", "decision",
                 "reason", "class", "subtype", "proposal_threshold", "sources", "outline", "sam_outline", "sam_status", "sam_masks", "sam_scores",
                 "sam_default", "outline_default"]  # what the annotation pages show of a candidate (no raw features)
LIGHT = ["candidate_id", "x_core", "y_core", "tile_id", "radius_px", "area_um2", "probability", "threshold", "model_id", "decision", "reason",
         "class", "subtype", "proposal_threshold", "sources", "sam_status"]  # ranking and listing candidates: no outlines
_FIELDS: OrderedDict[tuple, pd.DataFrame] = OrderedDict()
_FIELDS_LIMIT = 96  # fields' candidates kept parsed (a few hundred rows each)


def read_box(project: Project, stain: str, core_id: str, x0: float, y0: float, x1: float, y1: float,
             columns: list[str] | None = FIELD_COLUMNS) -> pd.DataFrame | None:
    """The candidates of one core image whose centre lies in [x0, x1) × [y0, y1) (core pixels), as `read_core` gives
    them (None when the core has none), from `candidates.read_bbox` (a parquet store is filtered while it is read and
    only `columns` are parsed: a field's few hundred rows, not the core's thousands of outlines)."""
    if path(project, stain, core_id) is None:
        return None
    columns = None if columns is None else list(dict.fromkeys([*columns, "x_core", "y_core", "tile_id"]))
    return _displayed(project, stain, _normalised(v2.read_bbox(project, stain, core_id, x0, y0, x1, y1, columns), stain, core_id))


def _shifted(polygon, x0: int, y0: int):
    return None if not polygon else (np.asarray(polygon, dtype=float) - (x0, y0)).tolist()


def field_key(project: Project, stain: str, tile: dict, context_px: int, level: float | None) -> tuple:
    target = path(project, stain, str(tile["core_id"]))
    x0, y0 = frames.field_origin(tile, context_px)
    return (str(target), cache.signature(target) if target is not None else None, str(tile["tile_id"]), x0, y0,
            int(tile["x_px"]) + int(tile["width_px"]) + context_px, int(tile["y_px"]) + int(tile["height_px"]) + context_px, level)


def in_field(project: Project, stain: str, tile: dict, context_px: int, level: float | None = None) -> pd.DataFrame:
    """Every candidate whose centre lies in this field's picture (the tile and its margin), with `x`, `y` and the
    outlines in that picture's pixels and `own` (the candidate is this tile's, not a neighbour's seen in the margin).
    Only the field's candidates are read (`read_box`); the last fields asked are kept until the store file changes.
    Callers must not change it."""
    key = field_key(project, stain, tile, context_px, level)
    with _WHOLE_LOCK:
        kept = _FIELDS.get(key)
        if kept is not None:
            _FIELDS.move_to_end(key)
            return kept
    x0, y0 = key[3:5]
    found = field_rows(project, stain, tile, context_px)
    if found is None or found.empty:
        out = pd.DataFrame(columns=[*EMPTY, "x", "y", "own"])
    else:
        near = at_level(found, level)
        out = near.assign(x=near.x_core - x0, y=near.y_core - y0, own=near.tile_id == str(tile["tile_id"]),
                          outline=[_shifted(p, x0, y0) for p in near.outline], sam_outline=[_shifted(p, x0, y0) for p in near.sam_outline],
                          sam_masks=[None if not m else [_shifted(p, x0, y0) for p in m] for m in near.sam_masks]).reset_index(drop=True)
    _keep(key, out)
    return out


def _keep(key: tuple, frame) -> None:
    with _WHOLE_LOCK:
        _FIELDS[key] = frame
        _FIELDS.move_to_end(key)
        while len(_FIELDS) > _FIELDS_LIMIT:
            _FIELDS.popitem(last=False)


def field_rows(project: Project, stain: str, tile: dict, context_px: int) -> pd.DataFrame | None:
    """The candidates whose centre lies in this field's picture as the store has them (core pixels, every level; None
    when the core has none), kept like `in_field`. Callers must not change it."""
    key = ("rows", *field_key(project, stain, tile, context_px, None))
    with _WHOLE_LOCK:
        if key in _FIELDS:
            _FIELDS.move_to_end(key)
            return _FIELDS[key]
    found = read_box(project, stain, str(tile["core_id"]), *key[4:8])
    _keep(key, found)
    return found


def field_cached(project: Project, stain: str, tile: dict, context_px: int, level: float | None = None) -> bool:
    """Whether `in_field` has this field's candidates in memory already (answers at once)."""
    key = field_key(project, stain, tile, context_px, level)
    with _WHOLE_LOCK:
        return key in _FIELDS


def _get(row, name: str):
    return row.get(name) if isinstance(row, dict) else getattr(row, name, None)


def masks(row) -> tuple[list[list[list[float]] | None], int]:
    """The outlines one can switch between for a candidate (point lists) and the one shown first, as
    `candidates.outline_options` chooses them for every view: Segment Anything's three masks small to large ("mask
    0–2"), then the stain outline ("mask 3")."""
    return v2.outline_options(row)


CHOICES = v2.CHOICES


def choices(row) -> tuple[list[list[list[float]] | None], int]:
    """The two outlines Tab switches between for a candidate ([Segment Anything's, the stain's], point lists, None
    where missing) and the one shown first: `candidates.outline_choices`."""
    return v2.outline_choices(row)


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
    """The class meaning "not the stain's object" (what Del records, drawn dotted grey like detection's own rejects): the
    stain's single reject (`Vocabulary.reject`)."""
    return vocabulary.reject


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


def calls_of(project: Project, stain: str, candidate_ids) -> pd.DataFrame:
    """Detection's call of each of these candidates as the v2 store has it, the one source of detection's calls on
    every page: indexed by candidate id, `identity` and `subtype` (label-class ids, `ai_call`), `value` ("identity /
    subtype" as labels write it), `probability` and `radius_px`. Ids not in the store are left out."""
    ids = sorted({str(c) for c in candidate_ids if isinstance(c, str) and c})
    columns = ["identity", "subtype", "value", "probability", "radius_px"]
    if not ids:
        return pd.DataFrame(columns=columns)
    cores = set()
    for c in ids:
        try:
            cores.add(v2.parse_id(c)[1])
        except ValueError:
            continue
    found = for_cores(project, stain, cores, ["candidate_id", "x_core", "y_core", "decision", "class", "subtype", "probability", "radius_px", "area_um2"])
    found = found[found.candidate_id.isin(set(ids))].drop_duplicates("candidate_id") if len(found) else found
    if found.empty:
        return pd.DataFrame(columns=columns)
    words = vocab.of(project, stain)
    calls = [ai_call(words, d, k, t) for d, k, t in zip(found.decision, found["class"], found.subtype)]
    return pd.DataFrame({"identity": [c[0] for c in calls], "subtype": [c[1] for c in calls],
                         "value": [f"{i} / {t}" if t else i for i, t in calls],
                         "probability": pd.to_numeric(found.probability, errors="coerce").to_numpy(),
                         "radius_px": [radius(r, project.pixel_size_um) for r in found.itertuples()]},
                        index=pd.Index(found.candidate_id.astype(str), name="candidate_id"))


def match_px(stain: str, pixel_size_um: float) -> float:
    return MATCH_UM.get(stain, DEFAULT_MATCH_UM) / max(1e-6, pixel_size_um)


_LIGHT: OrderedDict[str, tuple[tuple, dict | None]] = OrderedDict()
_LIGHT_BYTES = 768 * 2**20  # every core image of a cohort fits: a core's ids, positions and calls are kB to a few MB
_LIGHT_SIZE = [0]
_INDEXED: set[tuple[str, str]] = set()  # (project, stain) whose index on disk was read into `_LIGHT` by this process
_READ_THREADS = 8  # cores read at once when many are not known yet (the parquet reader lets go of the GIL)


def _bytes(found: dict | None) -> int:  # (its KD-tree, made when first matched, is about as big again)
    return 64 if found is None else found["ids"].nbytes + found["xy"].nbytes + found["codes"].nbytes + 64 * (1 + len(found["combos"]))


def _keep_light(key: str, stamp, found: dict | None) -> None:
    with _WHOLE_LOCK:
        old = _LIGHT.pop(key, None)
        if old is not None:
            _LIGHT_SIZE[0] -= _bytes(old[1])
        _LIGHT[key] = (stamp, found)
        _LIGHT_SIZE[0] += _bytes(found)
        while _LIGHT_SIZE[0] > _LIGHT_BYTES and len(_LIGHT) > 1:
            _, (_, gone) = _LIGHT.popitem(last=False)
            _LIGHT_SIZE[0] -= _bytes(gone)


def _index_file(project: Project, stain: str) -> Path:
    from stainid import disk_cache  # noqa: PLC0415

    return disk_cache.path(project.root, "candidate_index", f"{stain}.npz")


def _load_index(project: Project, stain: str) -> None:
    """Read this stain's index of every core's compact candidates (written by `save_index`) into memory once, so a
    cold server matches labels without opening every core's store file. Entries still carry their file's stamp:
    `_light` uses one only while the file is unchanged."""
    key = (str(project.root), stain)
    if key in _INDEXED:
        return
    _INDEXED.add(key)
    try:
        file = _index_file(project, stain)
        if not file.exists():
            return
        with np.load(file, allow_pickle=False) as data:
            paths, stamps, offsets = data["paths"], data["stamps"], data["offsets"]
            ids, xy, codes, combos = data["ids"], data["xy"], data["codes"], json.loads(str(data["combos"]))
        from stainid import disk_cache  # noqa: PLC0415

        disk_cache.used(file)
    except (OSError, ValueError, KeyError):
        return
    for n, target in enumerate(paths):
        start, end = int(offsets[n]), int(offsets[n + 1])
        found = None if combos[n] is None else {"ids": ids[start:end], "xy": xy[start:end], "codes": codes[start:end],
                                                "combos": [tuple(c) for c in combos[n]]}
        with _WHOLE_LOCK:
            known = str(target) in _LIGHT
        if not known:
            _keep_light(str(target), (int(stamps[n][0]), int(stamps[n][1])), found)


def save_index(project: Project, stain: str) -> None:
    """Write what `_light` knows of this stain's cores to the project's cache (`candidate_index`), atomically."""
    from stainid import disk_cache  # noqa: PLC0415

    prefix = str(folder(project, stain))
    with _WHOLE_LOCK:
        entries = [(k, v) for k, v in _LIGHT.items() if k.startswith(prefix) and v[0] is not None]
    if not entries:
        return
    paths, stamps, offsets, combos = [], [], [0], []
    ids, xy, codes = [], [], []
    for key, (stamp, found) in entries:
        paths.append(key)
        stamps.append(stamp)
        if found is None:
            combos.append(None)
            offsets.append(offsets[-1])
            continue
        ids.append(found["ids"])
        xy.append(found["xy"])
        codes.append(found["codes"])
        combos.append([list(c) for c in found["combos"]])
        offsets.append(offsets[-1] + len(found["ids"]))
    import io  # noqa: PLC0415

    buffer = io.BytesIO()
    width = max([a.dtype.itemsize for a in ids] or [1])
    np.savez(buffer, paths=np.array(paths), stamps=np.array(stamps, dtype=np.int64).reshape(-1, 2), offsets=np.array(offsets, dtype=np.int64),
             ids=np.concatenate(ids).astype(f"S{width}") if ids else np.zeros(0, "S1"),
             xy=np.concatenate(xy) if xy else np.zeros((0, 2), np.float32), codes=np.concatenate(codes) if codes else np.zeros(0, np.int32),
             combos=np.array(json.dumps(combos)))
    try:
        disk_cache.write(_index_file(project, stain), buffer.getvalue())
    except OSError:
        pass


def _light(project: Project, stain: str, core_id: str) -> dict | None:
    """One core's candidates as matching needs them, compactly: ids (bytes), positions (float32) and the call of each
    as a code into the core's few (decision, class, subtype) combinations. Kept until the store file changes (also on
    disk between restarts: `save_index`)."""
    target = path(project, stain, core_id)
    if target is None:
        return None
    _load_index(project, stain)
    stamp = cache.signature(target)
    with _WHOLE_LOCK:
        kept = _LIGHT.get(str(target))
        if kept is not None and kept[0] == stamp:
            _LIGHT.move_to_end(str(target))
            return kept[1]
    _READ[0] += 1
    frame = v2.read_core(project, stain, core_id, CALLS)  # the raw columns only: nothing parsed
    if not frame.empty and "candidate_id" not in frame:
        frame = frame.assign(candidate_id=[candidate_id(stain, core_id, a, b) for a, b in zip(frame.x_core, frame.y_core)])
    for column in ("decision", "class", "subtype"):
        if not frame.empty:
            frame[column] = frame[column].fillna("").astype(str) if column in frame else ""
    if frame.empty or "x_core" not in frame:
        found = None
    else:
        combos = pd.MultiIndex.from_arrays([frame.decision, frame["class"], frame.subtype])
        codes, uniques = pd.factorize(combos)
        found = {"ids": frame.candidate_id.to_numpy().astype("S"), "xy": np.c_[frame.x_core.to_numpy(), frame.y_core.to_numpy()].astype(np.float32),
                 "codes": codes.astype(np.int32), "combos": list(uniques)}
    _keep_light(str(target), stamp, found)
    return found


_READ = [0]  # core store files read by `_light` (to tell when the index on disk is worth writing again)
SAVE_AFTER = 20  # a pass that read this many store files writes the index again


def _lights(project: Project, stain: str, core_ids) -> dict[str, dict | None]:
    """`_light` of many cores: the ones known and unchanged looked up (one listing of the folder, one stat per file),
    the others read side by side; the index on disk written again when many were read."""
    _load_index(project, stain)
    base = folder(project, stain)
    names = cache.names(base)
    out: dict[str, dict | None] = {}
    missing = []
    for core_id in core_ids:
        target = next((base / f"{core_id}{s}" for s in (".parquet", ".csv.gz") if f"{core_id}{s}" in names), None)
        if target is None:
            out[core_id] = None
            continue
        kept = _LIGHT.get(str(target))
        if kept is not None and kept[0] == cache.signature(target):
            out[core_id] = kept[1]
        else:
            missing.append(core_id)
    before = _READ[0]
    if len(missing) > 4:
        from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415

        with ThreadPoolExecutor(_READ_THREADS) as pool:
            out.update(zip(missing, pool.map(lambda c: _light(project, stain, c), missing)))
    else:
        out.update({c: _light(project, stain, c) for c in missing})
    if _READ[0] - before >= SAVE_AFTER:
        save_index(project, stain)
    return out


def _nearest(project: Project, stain: str, labels: pd.DataFrame, tiles: pd.DataFrame):
    """For each core with labels: (the core's compact candidates, the labels' index, the candidate row of each label or
    -1) — the nearest candidate within `MATCH_UM` of each label (tile_id, x, y in its field picture, `labels.frames`)."""
    if labels.empty:
        return
    boxes = tiles.drop_duplicates("tile_id").set_index("tile_id") if not tiles.empty else pd.DataFrame(columns=["core_id"])
    have = cores(project, stain)
    if not have or "core_id" not in boxes:  # no store, or fields that do not say their core: nothing to match onto
        return
    placed = labels[labels.x.notna() & (labels.tile_id.isin(set(boxes.index)) | labels.tile_id.isin(have))]  # a field, or a whole core
    if placed.empty:
        return
    radius = match_px(stain, project.pixel_size_um)
    core_of = boxes.core_id.astype(str).to_dict()
    groups = [(c, g) for c, g in placed.groupby(placed.tile_id.map(lambda t: core_of.get(t, t))) if c in have]
    light = _lights(project, stain, [c for c, _ in groups])
    origins: dict[str, tuple] = {}
    for core_id, group in groups:
        found = light[core_id]
        if found is None:
            continue
        origin = np.array([origins[t] if t in origins else origins.setdefault(t, frames.field_origin(boxes.loc[t], project.context_px) if t in core_of else (0, 0))
                           for t in group.tile_id], float)  # one look-up per field, not per label
        points = np.c_[group.x.astype(float).to_numpy(), group.y.astype(float).to_numpy()] + origin
        tree = found.get("tree")
        if tree is None:
            tree = found["tree"] = cKDTree(found["xy"])  # made once per core (until its file changes)
        distance, index = tree.query(points, distance_upper_bound=radius)
        yield found, group.index, np.where(np.isfinite(distance), index, -1)


def core_calls(project: Project, stain: str, core_ids) -> pd.DataFrame:
    """candidate_id, x_core, y_core, decision, class, subtype of every candidate of these cores, from the compact
    copies matching keeps (`_light`, also on disk): a pass over a whole cohort without parsing its store tables."""
    core_ids = sorted(set(core_ids) & cores(project, stain))
    parts = []
    for light in _lights(project, stain, core_ids).values():
        if light is None or not len(light["ids"]):
            continue
        combos = np.array(light["combos"], dtype=object).reshape(-1, 3)
        picked = combos[light["codes"]]
        parts.append(pd.DataFrame({"candidate_id": light["ids"].astype(str), "x_core": light["xy"][:, 0].astype(float),
                                   "y_core": light["xy"][:, 1].astype(float), "decision": picked[:, 0], "class": picked[:, 1], "subtype": picked[:, 2]}))
    columns = ["candidate_id", "x_core", "y_core", "decision", "class", "subtype"]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=columns)


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
    """Write one core's store from parsed rows (outlines as point lists), through `candidates.write_table`."""
    from stainid import practice  # noqa: PLC0415

    practice.guard("the candidate store")
    target = folder(project, stain) / f"{core_id}.csv.gz"
    out = frame.copy()
    for column in POLYGONS:
        if column in out:
            out[column] = [encode(p) for p in out[column]]
    if "sam_masks" in out:
        out["sam_masks"] = ["|".join(encode(p) for p in m) if isinstance(m, (list, tuple)) and m else "" for m in out.sam_masks]
    if "sam_scores" in out:
        out["sam_scores"] = ["|".join(f"{v:.3f}" for v in m) if isinstance(m, (list, tuple)) and m else "" for m in out.sam_scores]
    return v2.write_table(target, out)


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
    matched = match(project, stain, objects, tiles)
    if "candidate" in objects:  # the consensus's own key (`store.object_keys`): the candidate, matched by position only where unnamed
        named = objects.candidate.fillna("").astype(str)
        matched = named.where(named != "", matched)
    matched = matched.dropna()
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
            scaled = {k: row[k] for k in row.index if str(k).startswith("s_") and k not in columns}  # slide-scaled 6E10 inputs
            out[index] = {**{n: row[c] for n, c in columns.items()}, **{k: row[k] for k in EXTRA if k in row and k not in columns}, **scaled,
                          "x": float(objects.x[index]), "y": float(objects.y[index]), "tma": tma.get(objects.tile_id[index], ""), "candidate_id": cid}
    return out, names
