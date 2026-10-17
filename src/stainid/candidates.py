"""The candidate store: every object detection proposed in a core image, with its outline, raw features and the AI's call.

Written by the v2 detection stages (`stainid.v2`), read by the viewer and the annotation pages. One table per stain and
core image, `<outputs.v2>/candidates/<stain>/<core_id>.csv.gz` (`.parquet` is read too), assembled from the stage outputs:
proposals (stage 2: geometry, sources, features), outlines (stage 3: Segment Anything) and calls (stage 4: probability,
decision, reason, class). Before stage 4 has run for a core, `read_core` serves the stage-2 proposals with empty calls.

Encodings (see also the contract in the v2 brief):
- `candidate_id` = "<stain>:<core_id>:<round(x_core)>:<round(y_core)>" (a later candidate rounding to the same pixel gets
  "~2", "~3"); labels attach to it, and labels without one are matched onto it by position (`labels.candidate_store.match`,
  within `MATCH_RADIUS_UM`);
- coordinates: `x_core`, `y_core` core-image pixels; `x_tile`, `y_tile` the tile's own frame; `x_field`, `y_field` the
  frame of the labelling field picture (the tile with its context margin);
- `bbox` "x0 y0 x1 y1"; `outline` and each ring of `sam_outline` (";"-separated) a flat "x0 y0 x1 y1 …" of core pixels;
  `sam_masks` Segment Anything's three outputs small → large, "|"-separated polygons, with `sam_scores` (predicted IoU)
  and `sam_stability` ("|"-separated, same order) and `sam_default` (0–2, -1 when none survived the runaway cap);
- `outline_default`: which outline is the object's own for display, "stain" (`outline`) or "sam" (the default Segment
  Anything output, `outline` where it fell back), the stain's `v2.outlines.default_outline`;
- `threshold`: the model's probability cut-off the call used;
- `m_<name>`: the input `<name>` of the current stain models; plain names: generic measurements.

Also per core: thread segments (`threads/<core_id>.csv.gz`, AT8) and stained regions no candidate covers
(`unproposed/<stain>/<core_id>.csv.gz`).

This is the one reader of these files: the stages, the viewer and the annotation pages (`labels.candidate_store`, which
parses outlines into point lists for the labelling pages) all read through `read_core` / `read_bbox`, and every view
draws the outline `outline_options` chooses.
"""
from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import pandas as pd

from stainid import cache

KINDS = ("candidates", "proposals", "outlines", "calls", "unproposed", "tiles")  # per stain and core image
MATCH_RADIUS_UM = {"NeuN": 4.0, "6E10": 6.0, "AT8": 4.0}
DEFAULT_MATCH_RADIUS_UM = 5.0
TEXT_COLUMNS = ("candidate_id", "stain", "core_id", "tile_id", "bbox", "sources", "source", "outline", "outline_default", "sam_outline", "sam_status",
                "sam_masks", "sam_scores", "sam_stability", "model_id", "decision", "reason", "class", "subtype", "points", "kind", "legacy_class",
                "legacy_source_class", "flag_legacy_drop", "dab_q", "core_dab_q", "region_id", "density_windows", "density_window_px")
CALL_COLUMNS = ("probability", "threshold", "model_id", "decision", "reason", "class", "subtype")
OUTLINE_COLUMNS = ("sam_outline", "sam_status", "sam_masks", "sam_scores", "sam_stability", "sam_default")
_CACHE: dict[tuple[str, tuple | None], tuple[tuple[int, int], pd.DataFrame, int]] = {}
_CACHE_LIMIT = 64
_CACHE_BYTES = 2 * 2**30  # a whole NeuN or AT8 core store is a few hundred MB in memory: bound the cache by size too
_GUARD = threading.Lock()


# ---------------------------------------------------------------- ids and encodings

def candidate_id(stain: str, core_id: str, x_core: float, y_core: float) -> str:
    return f"{stain}:{core_id}:{round(float(x_core))}:{round(float(y_core))}"


def parse_id(value: str) -> tuple[str, str, int, int]:
    """(stain, core_id, x, y) of a candidate id (the suffix of a de-collided id is ignored)."""
    stain, rest = value.split(":", 1)
    core_id, x, y = rest.rsplit(":", 2)
    return stain, core_id, int(x), int(y.split("~", 1)[0])


def unique_ids(stain: str, core_id: str, xs, ys) -> list[str]:
    """Stable ids of a core's candidates in the given order; a repeat of a rounded position gets "~2", "~3", …"""
    seen: dict[str, int] = {}
    out = []
    for x, y in zip(xs, ys):
        base = candidate_id(stain, core_id, x, y)
        seen[base] = seen.get(base, 0) + 1
        out.append(base if seen[base] == 1 else f"{base}~{seen[base]}")
    return out


def encode_points(points) -> str:
    """A flat [x0, y0, x1, y1, …] (or an (n, 2) array) as "x0 y0 x1 y1 …" of integers."""
    flat = np.rint(np.asarray(points, dtype=float).reshape(-1)).astype(np.int64)
    return " ".join(map(str, flat.tolist()))


def encode_rings(rings) -> str:
    return ";".join(encode_points(r) for r in rings)


def points(text) -> list[int]:
    """The flat point list of an `outline` / thread `points` cell ([] for an empty one)."""
    if text is None or (isinstance(text, float) and text != text):
        return []
    return [int(v) for v in str(text).split()]


def rings(text) -> list[list[int]]:
    """The polygons of a `sam_outline` (or `outline`) cell, each a flat [x0, y0, x1, y1, …]."""
    if text is None or (isinstance(text, float) and text != text) or not str(text).strip():
        return []
    return [points(part) for part in str(text).split(";") if part.strip()]


def _empty(text) -> bool:
    return text is None or (isinstance(text, float) and text != text) or not str(text).strip()


def sam_masks(text) -> list[list[int]]:
    """The three Segment Anything outputs of a candidate (small, medium, large), "|"-separated in the store, each a flat
    polygon ([] where SAM gave nothing usable); [] when the candidate was not outlined."""
    if _empty(text):
        return []
    return [points(part) for part in str(text).split("|")]


def sam_scores(text) -> list[float]:
    """A "|"-separated number per SAM output (`sam_scores`: predicted IoU; `sam_stability`: stability score)."""
    if _empty(text):
        return []
    return [float(v) if v.strip() else float("nan") for v in str(text).split("|")]


def bbox(text) -> tuple[int, int, int, int] | None:
    values = points(text)
    return tuple(values) if len(values) == 4 else None


# ---------------------------------------------------------------- where things are

def root(project) -> Path:
    try:
        return project.output("v2")
    except KeyError:  # a project configured before the v2 outputs had their own entry
        return project.root / "data" / "analysis" / "v2"


def folder(project, stain: str | None = None, kind: str = "candidates") -> Path:
    return root(project) / kind / stain if stain else root(project) / kind


def stored(project, stain: str, core_id: str, kind: str = "candidates") -> Path | None:
    """The table of one core image of one kind when it exists (`.parquet` first), else None; from the folder's listing
    (`cache.names`), so asking about many cores lists the folder once."""
    names = cache.names(folder(project, stain, kind))
    for suffix in (".parquet", ".csv.gz"):
        if f"{core_id}{suffix}" in names:
            return folder(project, stain, kind) / f"{core_id}{suffix}"
    return None


def cores(project, stain: str, kind: str = "candidates") -> frozenset[str]:
    """The core images of this stain that have a table of this kind."""
    out = set()
    for name in cache.names(folder(project, stain, kind)):
        for suffix in (".parquet", ".csv.gz"):
            if name.endswith(suffix) and not name.startswith("."):
                out.add(name[: -len(suffix)])
    return frozenset(out)


def path(project, stain: str, core_id: str, kind: str = "candidates") -> Path:
    """The table of one core image (stain) of one kind; `.parquet` when that is what exists, else `.csv.gz`."""
    folder = root(project) / kind / stain
    parquet = folder / f"{core_id}.parquet"
    return parquet if parquet.exists() else folder / f"{core_id}.csv.gz"


def threads_path(project, core_id: str) -> Path:
    folder = root(project) / "threads"
    parquet = folder / f"{core_id}.parquet"
    return parquet if parquet.exists() else folder / f"{core_id}.csv.gz"


# ---------------------------------------------------------------- reading and writing tables

def _dtypes(columns) -> dict:
    return {c: str for c in columns if c in TEXT_COLUMNS}


def _size(frame: pd.DataFrame) -> int:
    """Rough bytes of a frame: its arrays, plus about 400 bytes per text cell (outlines are long)."""
    text = sum(frame[c].dtype == object for c in frame.columns)
    return int(frame.memory_usage(index=True, deep=False).sum()) + 400 * text * len(frame)


def _remember(key, stamp, frame: pd.DataFrame) -> None:
    size = _size(frame)
    with _GUARD:
        _CACHE.pop(key, None)
        while _CACHE and (len(_CACHE) >= _CACHE_LIMIT or sum(v[2] for v in _CACHE.values()) + size > _CACHE_BYTES):
            _CACHE.pop(next(iter(_CACHE)))
        if size <= _CACHE_BYTES:
            _CACHE[key] = (stamp, frame, size)


def _parquet_ok() -> bool:
    try:
        import pyarrow  # noqa: F401, PLC0415 - optional: without it the store stays csv.gz
    except ImportError:
        return False
    return True


PARQUET = _parquet_ok()


def twin(file: Path) -> Path:
    """The `.parquet` file standing for a `.csv.gz` name (the stages name their outputs `<core_id>.csv.gz`)."""
    return file.with_name(file.name[: -len(".csv.gz")] + ".parquet") if file.name.endswith(".csv.gz") else file


def _for_parquet(frame: pd.DataFrame) -> pd.DataFrame:
    """Text columns as text ("" for none, as the csv.gz store reads back), other object columns numeric where they are."""
    out = frame.copy()
    for column in out.columns:
        if out[column].dtype != object and not pd.api.types.is_string_dtype(out[column].dtype):
            continue
        values = [v for v in out[column] if v is not None and v == v and v != ""]
        if column not in TEXT_COLUMNS and values and all(isinstance(v, (bool, np.bool_)) for v in values):
            out[column] = pd.array([v if isinstance(v, (bool, np.bool_)) else None for v in out[column]], dtype="boolean")
            continue
        if column not in TEXT_COLUMNS:
            numbers = pd.to_numeric(out[column], errors="coerce")
            if numbers.notna().sum() == len(values):
                out[column] = numbers
                continue
        empty = "" if column in TEXT_COLUMNS else None  # as the csv.gz store reads back: "" for text, NaN otherwise
        out[column] = [empty if v is None or (isinstance(v, float) and np.isnan(v)) or (empty is None and v == "") else str(v) for v in out[column]]
    return out


_SCHEMAS: dict[str, tuple[tuple[int, int], frozenset[str]]] = {}


def _schema(file: Path, stamp: tuple[int, int]) -> frozenset[str]:
    import pyarrow.parquet as pq  # noqa: PLC0415

    kept = _SCHEMAS.get(str(file))
    if kept is not None and kept[0] == stamp:
        return kept[1]
    names = frozenset(pq.read_schema(file).names)
    _SCHEMAS[str(file)] = (stamp, names)
    return names


def _read_parquet(file: Path, wanted: tuple[str, ...] | list[str] | None, filters: list | None = None) -> pd.DataFrame:
    """A parquet table (only `wanted` columns, only rows passing `filters`) decoded as the csv.gz store reads back: text
    columns "" for none, other missing values NaN."""
    import pyarrow.parquet as pq  # noqa: PLC0415

    columns = None
    if wanted is not None:
        stat = file.stat()
        present = _schema(file, (stat.st_mtime_ns, stat.st_size))
        columns = [c for c in wanted if c in present]
    frame = pq.read_table(file, columns=columns, filters=filters).to_pandas()
    for column in frame.columns:
        if column in TEXT_COLUMNS:
            frame[column] = frame[column].fillna("").astype(object)
        elif isinstance(frame[column].dtype, pd.BooleanDtype) or frame[column].dtype == object:  # None as NaN, as csv.gz reads back
            frame[column] = frame[column].astype(object).where(frame[column].notna(), np.nan)
    return frame


def read_table(file: Path, columns: list[str] | None = None) -> pd.DataFrame:
    """A store table (cached by mtime and size); an empty frame when the file does not exist. With `columns`, only those
    are read (the whole table when it is in memory already): a stage-5 pass or a viewer layer parses a fraction of a
    core's columns. A `.csv.gz` name reads its `.parquet` twin when there is one."""
    file = Path(file)
    if file.name.endswith(".csv.gz") and twin(file).exists():
        file = twin(file)
    try:
        stat = file.stat()
    except FileNotFoundError:
        return pd.DataFrame(columns=columns or [])
    stamp = (stat.st_mtime_ns, stat.st_size)
    wanted = None if columns is None else tuple(dict.fromkeys(columns))
    with _GUARD:
        whole = _CACHE.get((str(file), None))
        part = _CACHE.get((str(file), wanted)) if wanted is not None else None
    if whole is not None and whole[0] == stamp:
        frame = whole[1]
    elif part is not None and part[0] == stamp:
        frame = part[1]
    else:
        if file.suffix == ".parquet":
            frame = _read_parquet(file, wanted)
        else:
            header = pd.read_csv(file, nrows=0).columns
            use = list(header) if wanted is None else [c for c in header if c in wanted]
            frame = pd.read_csv(file, usecols=use or list(header[:1]), dtype=_dtypes(use) if use else str, keep_default_na=False,
                                na_values={c: [""] for c in use if c not in TEXT_COLUMNS}, low_memory=False)
            if not use:  # none of the wanted columns: the rows without columns, as the whole table would give
                frame = frame.iloc[:, 0:0]
        _remember((str(file), wanted), stamp, frame)
    if wanted is not None:
        return frame[[c for c in wanted if c in frame.columns]].copy()
    return frame.copy()


def write_table(file: Path, frame: pd.DataFrame, keep_csv: bool = False) -> Path:
    """Write a store table atomically: as the `.parquet` twin of a `.csv.gz` name when pyarrow is installed (the csv.gz
    is removed unless `keep_csv`), else csv.gz with floats to 8 significant digits. Returns the file written."""
    import os
    import tempfile

    file = Path(file)
    file.parent.mkdir(parents=True, exist_ok=True)
    if PARQUET and file.name.endswith(".csv.gz"):
        target = twin(file)
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix=f".{target.name}.", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
        try:
            _for_parquet(frame).to_parquet(temporary, index=False, compression="zstd")
            with temporary.open("rb+") as handle:
                os.fsync(handle.fileno())
            temporary.replace(target)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        if not keep_csv:
            file.unlink(missing_ok=True)
        return target
    with tempfile.NamedTemporaryFile(dir=file.parent, prefix=f".{file.name}.", suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        frame.to_csv(temporary, index=False, compression={"method": "gzip", "compresslevel": 5, "mtime": 0}, float_format="%.8g")
        with temporary.open("rb+") as handle:
            os.fsync(handle.fileno())
        temporary.replace(file)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return file


# ---------------------------------------------------------------- the API the viewer and annotation use

def read_core(project, stain: str, core_id: str, columns: list[str] | None = None) -> pd.DataFrame:
    """Every candidate of one core image: the assembled store, else (stage 4 not run yet) the stage-2 proposals with the
    stage 3 / 4 columns added empty. Empty frame when stage 2 has not run for this core."""
    store = path(project, stain, core_id, "candidates")
    frame = read_table(store, columns)
    if not store.exists():
        frame = read_table(path(project, stain, core_id, "proposals"), columns)
        if frame.empty and not len(frame.columns):
            return frame
        for column in (*OUTLINE_COLUMNS, *CALL_COLUMNS):
            if (columns is None or column in columns) and column not in frame:
                frame[column] = np.nan if column in ("probability", "threshold", "sam_default") else ""
    if (columns is None or "outline_default" in columns) and "outline_default" not in frame:  # assembled before the setting
        from stainid.v2.params import default_outline

        frame = frame.assign(outline_default=default_outline(project, stain))
    return frame


def read_tile(project, stain: str, tile_id: str, core_id: str | None = None, columns: list[str] | None = None) -> pd.DataFrame:
    core_id = core_id or tile_id.rsplit(f"_{stain}_", 1)[0]
    frame = read_core(project, stain, core_id, None if columns is None else list(dict.fromkeys([*columns, "tile_id"])))
    frame = frame[frame.tile_id == tile_id] if "tile_id" in frame else frame.iloc[0:0]
    return frame if columns is None else frame[[c for c in columns if c in frame]]


def _whole_cached(file: Path) -> bool:
    try:
        stat = file.stat()
    except FileNotFoundError:
        return False
    with _GUARD:
        whole = _CACHE.get((str(file), None))
    return whole is not None and whole[0] == (stat.st_mtime_ns, stat.st_size)


def read_bbox(project, stain: str, core_id: str, x0: float, y0: float, x1: float, y1: float, columns: list[str] | None = None) -> pd.DataFrame:
    """The candidates whose centroid lies in [x0, x1) × [y0, y1) (core pixels), as `read_core` gives them. An assembled
    parquet store not in memory is filtered while it is read (a field's few hundred rows, not a core's thousands of
    outlines); otherwise the core is read (or taken from memory) and filtered."""
    store = twin(path(project, stain, core_id, "candidates"))
    if store.suffix == ".parquet" and store.exists() and not _whole_cached(store):
        wanted = None if columns is None else tuple(dict.fromkeys([*columns, "x_core", "y_core"]))
        bounds = [("x_core", ">=", float(x0)), ("x_core", "<", float(x1)), ("y_core", ">=", float(y0)), ("y_core", "<", float(y1))]
        frame = _read_parquet(store, wanted, bounds)
        if (columns is None or "outline_default" in columns) and "outline_default" not in frame:
            from stainid.v2.params import default_outline

            frame = frame.assign(outline_default=default_outline(project, stain))
        return frame if columns is None else frame[[c for c in columns if c in frame]]
    frame = read_core(project, stain, core_id, None if columns is None else list(dict.fromkeys([*columns, "x_core", "y_core"])))
    if frame.empty:
        return frame
    inside = (frame.x_core >= x0) & (frame.x_core < x1) & (frame.y_core >= y0) & (frame.y_core < y1)
    frame = frame[inside]
    return frame if columns is None else frame[[c for c in columns if c in frame]]


def by_id(project, ids: list[str], columns: list[str] | None = None) -> pd.DataFrame:
    """Candidates by id (any stains and cores), in the order asked; unknown ids are left out."""
    wanted: dict[tuple[str, str], list[str]] = {}
    for value in ids:
        stain, core_id, _, _ = parse_id(value)
        wanted.setdefault((stain, core_id), []).append(value)
    frames = []
    for (stain, core_id), values in wanted.items():
        frame = read_core(project, stain, core_id, None if columns is None else list(dict.fromkeys(["candidate_id", *columns])))
        if not frame.empty:
            frames.append(frame[frame.candidate_id.isin(values)])
    if not frames:
        return pd.DataFrame(columns=columns or ["candidate_id"])
    out = pd.concat(frames, ignore_index=True).set_index("candidate_id", drop=False)
    return out.loc[[v for v in ids if v in out.index]].reset_index(drop=True)


def _given(value) -> bool:
    """A cell that holds something: not None / NaN, not empty text, not an empty list."""
    if value is None or (isinstance(value, float) and value != value):
        return False
    if isinstance(value, str):
        return bool(value.strip())
    try:
        return len(value) > 0
    except TypeError:
        return True


def _number(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if number != number else number


def _flat(value) -> list[int]:
    """A polygon in either encoding (store text, the largest ring of several; or a list of points) as flat whole pixels."""
    if isinstance(value, str):
        found = rings(value)
        return max(found, key=len) if found else []
    return np.rint(np.asarray(value, dtype=float).reshape(-1)).astype(np.int64).tolist()


def outline_options(row) -> tuple[list, int]:
    """The outlines one can switch between for a candidate and the index of the one drawn by default: the one rule every
    view and page follows. With Segment Anything's three outputs: those (small → large, so "mask 0–2" keep their
    meaning, None where SAM gave nothing) and then the candidate's stain outline ("mask 3"); drawn by default is the
    stain outline where `outline_default` is "stain" or Segment Anything fell back (`sam_default` -1), else the stored
    default output (the best-scored usable one when none is stored). Without them: its one outline (`sam_outline`, or
    the stain outline where `outline_default` is "stain" or SAM gave none).

    Works on a row as the store has it (text cells: the options are texts) and as `labels.candidate_store` parses it
    (lists of points: the options are those); `row` is a dict, a Series or an itertuples row."""
    get = row.get if hasattr(row, "get") else (lambda name: getattr(row, name, None))
    own = get("outline") if _given(get("outline")) else None
    found = get("sam_masks")
    found = (str(found).split("|") if isinstance(found, str) else list(found)) if _given(found) else []
    found = [m if _given(m) else None for m in found]
    prefer_own = get("outline_default") == "stain"
    if any(m is not None for m in found):
        found = found + ([own] if own is not None else [])
        default = _number(get("sam_default"))
        if own is not None and (prefer_own or (default is not None and default < 0)):
            return found, len(found) - 1
        if default is None or not 0 <= int(default) < min(3, len(found)) or found[int(default)] is None:
            kept = _flat(get("sam_outline")) if _given(get("sam_outline")) else None
            same = [i for i, m in enumerate(found[:3]) if m is not None and kept is not None and _flat(m) == kept]
            if same:  # no default stored, but the output stage 3 kept (`sam_outline`) is one of them
                return found, same[0]
            scores = get("sam_scores")
            scores = (sam_scores(scores) if isinstance(scores, str) else list(scores)) if _given(scores) else []
            usable = [i for i, m in enumerate(found[:3]) if m is not None]
            default = max(usable, key=lambda i: scores[i] if i < len(scores) and scores[i] == scores[i] else 0.0)
        return found, int(default)
    sam = get("sam_outline") if _given(get("sam_outline")) and get("sam_status") in ("ok", "", None) else None
    one = (own if own is not None else sam) if prefer_own else (sam if sam is not None else own)
    return ([one] if one is not None else []), 0


def outline_rings(row) -> list[list[int]]:
    """The outline to draw for a candidate (`outline_options`' default) as rings of flat core-pixel points."""
    found, default = outline_options(row)
    chosen = found[default] if found else None
    if chosen is None:
        return []
    if isinstance(chosen, str):
        return rings(chosen)
    flat = np.asarray(chosen, dtype=float).reshape(-1)
    return [np.rint(flat).astype(np.int64).tolist()] if len(flat) else []


def outlines(project, stain: str, core_id: str, decision: str | None = None) -> list[dict]:
    """Every candidate's id, centroid, call and outline rings (core pixels) of one core image; `decision` limits them."""
    frame = read_core(project, stain, core_id, ["candidate_id", "x_core", "y_core", "decision", "class", "subtype", "probability",
                                               "outline", "sam_outline", "sam_status", "sam_masks", "sam_scores", "sam_stability",
                                               "sam_default", "threshold", "outline_default"])
    if decision is not None and "decision" in frame:
        frame = frame[frame.decision == decision]
    out = []
    for r in frame.to_dict("records"):
        default = r.get("sam_default")
        out.append({"candidate_id": r["candidate_id"], "x": r["x_core"], "y": r["y_core"], "decision": r.get("decision", ""),
                    "class": r.get("class", ""), "subtype": r.get("subtype", ""), "probability": r.get("probability"),
                    "rings": outline_rings(r), "outline": points(r.get("outline")), "sam_masks": sam_masks(r.get("sam_masks")),
                    "sam_scores": sam_scores(r.get("sam_scores")), "sam_stability": sam_scores(r.get("sam_stability")),
                    "threshold": r.get("threshold"), "outline_default": r.get("outline_default", ""),
                    "sam_default": None if default is None or default != default or int(default) < 0 else int(default)})
    return out


def threads(project, core_id: str) -> pd.DataFrame:
    """The AT8 thread skeleton of one core image as saved by stage 2 (one row per segment)."""
    return read_table(threads_path(project, core_id))


def unproposed(project, stain: str, core_id: str, include_threads: bool = False, include_excluded: bool = False) -> pd.DataFrame:
    """Stained regions (DAB at or above the slide threshold, at least the stain's minimum size) that no candidate covers.
    Regions mostly on traced AT8 threads or in excluded areas are left out unless asked for."""
    frame = read_table(path(project, stain, core_id, "unproposed"))
    if frame.empty:
        return frame
    if not include_threads and "thread_fraction" in frame:
        frame = frame[~(frame.thread_fraction.fillna(0) >= 0.5)]
    if not include_excluded and "excluded_fraction" in frame:
        frame = frame[~(frame.excluded_fraction.fillna(0) >= 0.5)]
    return frame


def model_matrix(frame: pd.DataFrame, names: list[str]) -> np.ndarray:
    """The rows' values of a model's features (`m_<name>` columns, else plain `<name>`), as floats (NaN where missing)."""
    columns = []
    for name in names:
        column = f"m_{name}" if f"m_{name}" in frame else name
        columns.append(pd.to_numeric(frame[column], errors="coerce").to_numpy(float) if column in frame else np.full(len(frame), np.nan))
    return np.column_stack(columns) if columns else np.zeros((len(frame), 0))


def forget() -> None:
    with _GUARD:
        _CACHE.clear()


__all__ = [
    "MATCH_RADIUS_UM", "by_id", "candidate_id", "cores", "encode_points", "encode_rings", "folder", "model_matrix",
    "outline_options", "outline_rings", "outlines", "parse_id", "path", "points", "read_bbox", "read_core", "read_table", "read_tile",
    "rings", "stored", "threads", "threads_path", "unique_ids", "unproposed", "write_table",
]
