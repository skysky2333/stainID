"""Every label in one append-only log: who said what about which object (or field), for which task.

A label is {id, source, stain, task, tile_id, x, y, value, points, session, confirmed, time, deleted}. `source` is a
person's name or `import: <set>`; the pipeline's own call is not stored here, it is read from the results. Objects are
identified by field and position (rounded pixel), so labels from different people and sessions about the same object
line up. The latest version of (source, id) wins; edits and deletions keep the history.

Tasks: `class` (the object's class, also used for objects people add), `outline` (polygon), `tag`, `note` and `done`
(a person finished reviewing the field).
"""
from __future__ import annotations

import fcntl
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from stainid.labels.tasks import SKIP, positive

COLUMNS = ["id", "source", "stain", "task", "tile_id", "x", "y", "value", "points", "session", "confirmed", "time", "deleted"]
_CACHE: dict[Path, tuple[tuple[int, int], pd.DataFrame]] = {}


def log_path(root: Path) -> Path:
    return root / "labels" / "labels.jsonl"


def object_id(task: str, tile_id: str, x: float, y: float) -> str:
    return f"{task}:{tile_id}:{round(x)}:{round(y)}"


def append(root: Path, records: list[dict]) -> list[dict]:
    path = log_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    now = time.time()
    stamped = [{**{k: None for k in COLUMNS}, "confirmed": False, "deleted": False, "session": "", **r, "time": r.get("time") or now} for r in records]
    with path.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.write("".join(json.dumps(r) + "\n" for r in stamped))
        handle.flush()
        fcntl.flock(handle, fcntl.LOCK_UN)
    return stamped


def history(root: Path) -> pd.DataFrame:
    path = log_path(root)
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    stat = path.stat()
    key = (stat.st_mtime_ns, stat.st_size)
    if path in _CACHE and _CACHE[path][0] == key:
        return _CACHE[path][1]
    with path.open(encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_SH)
        lines = handle.read().splitlines()
        fcntl.flock(handle, fcntl.LOCK_UN)
    frame = pd.DataFrame([json.loads(line) for line in lines if line], columns=COLUMNS)
    _CACHE[path] = (key, frame)
    return frame


def current(root: Path) -> pd.DataFrame:
    """The latest version of every label that is not deleted."""
    frame = history(root)
    latest = frame.drop_duplicates(["source", "id"], keep="last")
    return latest[~latest.deleted.astype(bool)].reset_index(drop=True)


def consensus(labels: pd.DataFrame, stain: str) -> pd.DataFrame:
    """One row per labelled object: majority value (ties unresolved), number of people, agreement, majority positive."""
    rows = labels[(labels.stain == stain) & (labels.task == "class") & ~labels.value.isin(SKIP)].copy()
    if rows.empty:
        return pd.DataFrame(columns=["tile_id", "x", "y", "value", "n", "agreement", "positive", "sources"])
    rows["rx"], rows["ry"] = rows.x.round().astype(int), rows.y.round().astype(int)
    out = []
    for (tile, rx, ry), group in rows.groupby(["tile_id", "rx", "ry"], sort=False):
        counts = group.value.value_counts()
        top = counts.index[0] if len(counts) == 1 or counts.iloc[0] > counts.iloc[1] else ""
        votes = np.array([positive(stain, v) for v in group.value])
        out.append({"tile_id": tile, "x": float(group.x.iloc[0]), "y": float(group.y.iloc[0]), "value": top, "n": len(group),
                    "agreement": float(counts.iloc[0] / len(group)), "positive": bool(votes.mean() > 0.5) if votes.mean() != 0.5 else None,
                    "sources": ", ".join(sorted(group.source))})
    return pd.DataFrame(out)


def reviewed_fields(labels: pd.DataFrame, stain: str) -> dict[str, set[str]]:
    """Fields someone finished reviewing, and who."""
    done = labels[(labels.stain == stain) & (labels.task == "done")]
    return {tile: set(group.source) for tile, group in done.groupby("tile_id")}
