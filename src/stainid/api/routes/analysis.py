from __future__ import annotations

import os
import threading
import time
from collections import OrderedDict
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException

from stainid.analysis.dictionary import TOPICS, describe, measure
from stainid.api.state import get_state
from stainid.api.util import records

router = APIRouter(tags=["analysis"])
GROUP_COLUMNS = ("disease_group", "region", "tma")
ID_COLUMNS = ("sample_region_id", "core_id", "donor_id", "tma", "region", "disease_group")
TABLES = {  # name: (title, description)
    "results_donor_region.csv": ("All measures, per donor-region", "Main results: one row per donor and brain region, every stain."),
    "results_core.csv": ("All measures, per core", "Main results, one row per core."),
    "neun_donor_region.csv": ("Neurons, per donor-region", "NeuN neuron measures per donor-region."),
    "fields_donor_region.csv": ("Plaques and tau, per donor-region", "6E10 plaque and AT8 tau measures per donor-region."),
    "masks_donor_region.csv": ("Object shapes, per donor-region", "Object-outline shape measures per donor-region."),
    "neun_core.csv": ("Neurons, per core", "NeuN neuron measures per core."),
    "fields_core.csv": ("Plaques and tau, per core", "6E10 plaque and AT8 tau measures per core."),
    "masks_core.csv": ("Object shapes, per core", "Object-outline shape measures per core."),
}
NOT_MEASURES = {"tma", "technical_replicate", "selection_order", "label"}

# The tables change only when a step writes them, so what is read from them is kept: folder listings by the folder's
# modification time (and read again at least every LISTING_FRESH seconds), headers and whole tables by each file's
# modification time and size. Listing ~250 tables or plotting a measure then needs no folder scans or file reads, which
# are slow on USB and network drives, above all while a long run writes to the same drive.
_lock = threading.Lock()
_folders: dict[Path, tuple[int, float, list[Path], list[Path]]] = {}
_headers: dict[Path, tuple[tuple[int, int], list[str]]] = {}
_frames: OrderedDict[Path, tuple[tuple[int, int], pd.DataFrame]] = OrderedDict()
FRAMES_KEPT = 12
LISTING_FRESH = 60.0


def _folder(folder: Path) -> tuple[list[Path], list[Path]]:
    """The CSV files and the subfolders of a folder."""
    try:
        mtime = folder.stat().st_mtime_ns
    except FileNotFoundError:
        return [], []
    with _lock:
        known = _folders.get(folder)
    if known and known[0] == mtime and time.monotonic() - known[1] < LISTING_FRESH:
        return known[2], known[3]
    files, folders = [], []
    with os.scandir(folder) as entries:
        for entry in entries:
            if entry.is_dir():
                folders.append(Path(entry.path))
            elif entry.name.endswith(".csv") and entry.is_file():
                files.append(Path(entry.path))
    with _lock:
        _folders[folder] = (mtime, time.monotonic(), sorted(files), sorted(folders))
    return sorted(files), sorted(folders)


def _csv_tree(folder: Path) -> list[Path]:
    files, folders = _folder(folder)
    return files + [path for sub in folders for path in _csv_tree(sub)]


def _stamp(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return stat.st_mtime_ns, stat.st_size


def _header(path: Path, stamp: tuple[int, int]) -> list[str]:
    with _lock:
        known = _headers.get(path)
    if known and known[0] == stamp:
        return known[1]
    with path.open() as handle:
        header = handle.readline().strip().split(",")
    with _lock:
        _headers[path] = (stamp, header)
    return header


def read_table(path: Path) -> pd.DataFrame:
    """The whole table (kept in memory until the file changes). Callers must not modify it."""
    stamp = _stamp(path)
    with _lock:
        known = _frames.get(path)
        if known and known[0] == stamp:
            _frames.move_to_end(path)
            return known[1]
    frame = pd.read_csv(path, dtype={"donor_id": str, "tma": str})
    with _lock:
        _frames[path] = (stamp, frame)
        while len(_frames) > FRAMES_KEPT:
            _frames.popitem(last=False)
    return frame


def tables_root() -> Path:
    return get_state().project.output("tables")


def resolve(name: str) -> Path:
    path = (tables_root() / name).resolve()
    if tables_root().resolve() not in path.parents or path.suffix != ".csv" or not path.exists():
        raise HTTPException(404, name)
    return path


@router.get("/tables")
def tables() -> list[dict]:
    base = tables_root()
    project = get_state().project
    out = []
    for path in _folder(base)[0] + sorted(_csv_tree(base / "biology")):
        stamp = _stamp(path)
        header = _header(path, stamp)
        name = str(path.relative_to(base))
        title, description = TABLES.get(name, (name, ""))
        out.append({"name": name, "title": title, "description": description, "main": name in TABLES, "path": project.relative(path),
                    "columns": len(header), "size_kb": round(stamp[1] / 1024, 1), "modified": stamp[0] / 1e9,
                    "donor_region": "sample_region_id" in header, "contrasts": "std_effect" in header or "effect_sd" in header})
    return sorted(out, key=lambda t: (not t["main"], list(TABLES).index(t["name"]) if t["main"] else 0, t["name"]))


@router.get("/tables/{name:path}/columns")
def columns(name: str) -> dict:
    frame = read_table(resolve(name))
    numeric = [c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c]) and c not in NOT_MEASURES and not c.endswith("_id")]
    return {"columns": list(frame.columns), "numeric": numeric, "groupable": [c for c in GROUP_COLUMNS if c in frame],
            "descriptions": {c: describe(c) for c in frame.columns if describe(c)},
            "measures": [measure(c) for c in numeric], "topics": TOPICS, "rows": len(frame)}


@router.get("/tables/{name:path}/rows")
def rows(name: str, offset: int = 0, limit: int = 100, sort: str = "", desc: bool = False, q: str = "", columns: str = "") -> dict:
    """One page of a table, optionally sorted by a column and filtered to rows whose names contain `q`."""
    frame = read_table(resolve(name))
    if q:
        text = [c for c in frame.columns if c in ID_COLUMNS or not pd.api.types.is_numeric_dtype(frame[c])]
        hit = pd.Series(False, index=frame.index)
        for column in text:
            hit |= frame[column].astype(str).str.contains(q, case=False, regex=False, na=False)
        frame = frame[hit]
    if sort in frame:
        frame = frame.sort_values(sort, ascending=not desc, na_position="last", kind="stable")
    if columns:
        keep = [c for c in dict.fromkeys(columns.split(",")) if c in frame]
        frame = frame[keep]
    limit = max(1, min(limit, 5000))
    return {"total": len(frame), "offset": offset, "columns": list(frame.columns), "rows": records(frame.iloc[offset: offset + limit])}


@router.get("/tables/{name:path}/feature")
def feature(name: str, column: str, group: str = "disease_group", facet: str | None = None) -> dict:
    frame = read_table(resolve(name))
    if "sampling_extent" in frame:
        frame = frame[frame.sampling_extent == frame.sampling_extent.iloc[0]]
    if column not in frame or group not in frame:
        raise HTTPException(400, "unknown column")
    keep = [c for c in dict.fromkeys(["sample_region_id", "core_id", "donor_id", group, facet, column]) if c and c in frame]
    return {"column": column, "group": group, "facet": facet, "points": records(frame[keep].dropna(subset=[column]))}
