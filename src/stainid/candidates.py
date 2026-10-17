"""The candidate store: every object detection proposed in a core image, with its outline, raw features and the AI's call.

Written by the v2 detection stages (`stainid.v2`), read by the viewer and the annotation pages. One table per stain and
core image, `<outputs.v2>/candidates/<stain>/<core_id>.csv.gz` (`.parquet` is read too), assembled from the stage outputs:
proposals (stage 2: geometry, sources, features), outlines (stage 3: Segment Anything) and calls (stage 4: probability,
decision, reason, class). Before stage 4 has run for a core, `read_core` serves the stage-2 proposals with empty calls.

Encodings (see also the contract in the v2 brief):
- `candidate_id` = "<stain>:<core_id>:<round(x_core)>:<round(y_core)>" (a later candidate rounding to the same pixel gets
  "~2", "~3"); labels attach to it, and `match` finds a candidate by position when ids change between runs;
- coordinates: `x_core`, `y_core` core-image pixels; `x_tile`, `y_tile` the tile's own frame; `x_field`, `y_field` the
  frame of the labelling field picture (the tile with its context margin);
- `bbox` "x0 y0 x1 y1"; `outline` and each ring of `sam_outline` (";"-separated) a flat "x0 y0 x1 y1 …" of core pixels;
  `sam_masks` Segment Anything's three outputs small → large, "|"-separated polygons, with `sam_scores` (predicted IoU)
  and `sam_stability` ("|"-separated, same order) and `sam_default` (0–2, -1 when none survived the runaway cap);
- `threshold`: the model's probability cut-off the call used;
- `m_<name>`: the input `<name>` of the current stain models; plain names: generic measurements.

Also per core: thread segments (`threads/<core_id>.csv.gz`, AT8) and stained regions no candidate covers
(`unproposed/<stain>/<core_id>.csv.gz`).
"""
from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import pandas as pd

KINDS = ("candidates", "proposals", "outlines", "calls", "unproposed", "tiles")  # per stain and core image
MATCH_RADIUS_UM = {"NeuN": 4.0, "6E10": 6.0, "AT8": 4.0}
DEFAULT_MATCH_RADIUS_UM = 5.0
TEXT_COLUMNS = ("candidate_id", "stain", "core_id", "tile_id", "bbox", "sources", "source", "outline", "sam_outline", "sam_status",
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
    return project.output("v2")


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
            if wanted is None:
                frame = pd.read_parquet(file)
            else:
                import pyarrow.parquet as pq  # noqa: PLC0415

                present = set(pq.read_schema(file).names)
                frame = pd.read_parquet(file, columns=[c for c in wanted if c in present])
            for column in frame.columns:
                if column in TEXT_COLUMNS:
                    frame[column] = frame[column].fillna("").astype(object)
                elif isinstance(frame[column].dtype, pd.BooleanDtype) or frame[column].dtype == object:  # None as NaN, as csv.gz reads back
                    frame[column] = frame[column].astype(object).where(frame[column].notna(), np.nan)
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
    if store.exists():
        return frame
    frame = read_table(path(project, stain, core_id, "proposals"), columns)
    if frame.empty and not len(frame.columns):
        return frame
    for column in (*OUTLINE_COLUMNS, *CALL_COLUMNS):
        if (columns is None or column in columns) and column not in frame:
            frame[column] = np.nan if column in ("probability", "threshold", "sam_default") else ""
    return frame


def read_tile(project, stain: str, tile_id: str, core_id: str | None = None, columns: list[str] | None = None) -> pd.DataFrame:
    core_id = core_id or tile_id.rsplit(f"_{stain}_", 1)[0]
    frame = read_core(project, stain, core_id, None if columns is None else list(dict.fromkeys([*columns, "tile_id"])))
    frame = frame[frame.tile_id == tile_id] if "tile_id" in frame else frame.iloc[0:0]
    return frame if columns is None else frame[[c for c in columns if c in frame]]


def read_bbox(project, stain: str, core_id: str, x0: float, y0: float, x1: float, y1: float, columns: list[str] | None = None) -> pd.DataFrame:
    """The candidates whose centroid lies in [x0, x1) × [y0, y1) (core pixels)."""
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


def match(project, stain: str, core_id: str, x_core: float, y_core: float, radius_um: float | None = None,
          frame: pd.DataFrame | None = None) -> str | None:
    """The id of the candidate nearest a position (core pixels) within the stain's match radius, else None: how a label
    made on an earlier run (or before ids existed) finds its candidate."""
    frame = read_core(project, stain, core_id, ["candidate_id", "x_core", "y_core"]) if frame is None else frame
    if frame.empty:
        return None
    radius_px = (radius_um or MATCH_RADIUS_UM.get(stain, DEFAULT_MATCH_RADIUS_UM)) / project.pixel_size_um
    distance = np.hypot(frame.x_core.to_numpy(float) - float(x_core), frame.y_core.to_numpy(float) - float(y_core))
    nearest = int(np.argmin(distance))
    return str(frame.candidate_id.iloc[nearest]) if distance[nearest] <= radius_px else None


def match_many(project, stain: str, core_id: str, xy: np.ndarray, radius_um: float | None = None) -> list[str | None]:
    """`match` for many positions at once (an (n, 2) array of core pixels); each candidate is given to one position at most."""
    from scipy.spatial import cKDTree

    frame = read_core(project, stain, core_id, ["candidate_id", "x_core", "y_core"])
    xy = np.asarray(xy, float).reshape(-1, 2)
    if frame.empty or not len(xy):
        return [None] * len(xy)
    radius_px = (radius_um or MATCH_RADIUS_UM.get(stain, DEFAULT_MATCH_RADIUS_UM)) / project.pixel_size_um
    k = min(8, len(frame))
    distance, nearest = cKDTree(frame[["x_core", "y_core"]].to_numpy(float)).query(xy, k=k, distance_upper_bound=radius_px)
    distance, nearest = distance.reshape(len(xy), k), nearest.reshape(len(xy), k)
    pairs = sorted((d, p, int(n)) for p in range(len(xy)) for d, n in zip(distance[p], nearest[p]) if np.isfinite(d))
    out: list[str | None] = [None] * len(xy)
    taken: set[int] = set()
    for _, position, candidate in pairs:  # closest pairs first, each candidate and position used once
        if out[position] is None and candidate not in taken:
            taken.add(candidate)
            out[position] = str(frame.candidate_id.iloc[candidate])
    return out


def outline_rings(row) -> list[list[int]]:
    """The outline to draw for a candidate: its default Segment Anything output when one was kept, else the candidate
    outline."""
    sam = rings(row.get("sam_outline")) if row.get("sam_status") == "ok" else []
    return sam or ([points(row.get("outline"))] if points(row.get("outline")) else [])


def outlines(project, stain: str, core_id: str, decision: str | None = None) -> list[dict]:
    """Every candidate's id, centroid, call and outline rings (core pixels) of one core image; `decision` limits them."""
    frame = read_core(project, stain, core_id, ["candidate_id", "x_core", "y_core", "decision", "class", "subtype", "probability",
                                               "outline", "sam_outline", "sam_status", "sam_masks", "sam_scores", "sam_stability",
                                               "sam_default", "threshold"])
    if decision is not None and "decision" in frame:
        frame = frame[frame.decision == decision]
    out = []
    for r in frame.to_dict("records"):
        default = r.get("sam_default")
        out.append({"candidate_id": r["candidate_id"], "x": r["x_core"], "y": r["y_core"], "decision": r.get("decision", ""),
                    "class": r.get("class", ""), "subtype": r.get("subtype", ""), "probability": r.get("probability"),
                    "rings": outline_rings(r), "outline": points(r.get("outline")), "sam_masks": sam_masks(r.get("sam_masks")),
                    "sam_scores": sam_scores(r.get("sam_scores")), "sam_stability": sam_scores(r.get("sam_stability")),
                    "threshold": r.get("threshold"),
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
    "MATCH_RADIUS_UM", "by_id", "candidate_id", "encode_points", "encode_rings", "match", "match_many", "model_matrix",
    "outline_rings", "outlines", "parse_id", "path", "points", "read_bbox", "read_core", "read_table", "read_tile", "rings",
    "threads", "threads_path", "unique_ids", "unproposed", "write_table",
]
