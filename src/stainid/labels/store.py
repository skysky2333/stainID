"""Every label in one append-only log: who said what about which object (or field), for which task.

A label is {id, source, stain, task, tile_id, x, y, value, points, session, confirmed, time, deleted, subtype, candidate}. `source` is a
person's name or `import: <set>`; the pipeline's own call is not stored here, it is read from the results. Objects are
identified by field and position (rounded pixel), so labels from different people and sessions about the same object
line up. The latest version of (source, id) wins; edits and deletions keep the history.

Tasks: `class` (the object's class, also used for objects people add: `value` is its identity and `subtype` its
subtype, see `labels.vocabulary`; labels made before subtypes existed have one value and no subtype),`outline` (polygon), `tag`, `note` and `done`
(a person finished reviewing the field).
"""
from __future__ import annotations

import fcntl
import json
import time
from pathlib import Path

import pandas as pd

from stainid.labels import vocabulary as vocab
from stainid.labels.vocabulary import Vocabulary

COLUMNS = ["id", "source", "stain", "task", "tile_id", "x", "y", "value", "points", "session", "confirmed", "time", "deleted", "subtype", "candidate"]
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


def signature(root: Path) -> tuple[int, int] | None:
    """Changes whenever a label is saved."""
    path = log_path(root)
    if not path.exists():
        return None
    stat = path.stat()
    return stat.st_mtime_ns, stat.st_size


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


def classes(labels: pd.DataFrame, stain: str, vocabulary: Vocabulary | None = None) -> pd.DataFrame:
    """The class labels of this stain, each with its `identity`, `subtype` (ids, "" for none) and `role` in the stain's
    vocabulary (labels made before subtypes existed mapped onto them, `vocabulary.LEGACY`)."""
    vocabulary = vocabulary or vocab.defaults(stain)
    rows = labels[(labels.stain == stain) & (labels.task == "class")]
    subtypes = rows.subtype if "subtype" in rows else pd.Series(None, index=rows.index, dtype=object)
    pairs = {pair: vocabulary.resolve(*pair) for pair in set(zip(rows.value.map(vocab.text), subtypes.map(vocab.text)))}
    resolved = [pairs[(vocab.text(v), vocab.text(s))] for v, s in zip(rows.value, subtypes)]
    identity = [p[0] for p in resolved]
    roles = {i: vocabulary.role(i) for i in set(identity)}
    return rows.assign(identity=identity, subtype=[p[1] for p in resolved], role=[roles[i] for i in identity])


def majority(values: pd.Series) -> tuple[str, float]:
    """The most common value ("" when two are as common) and its share."""
    counts = values.value_counts()
    if counts.empty:
        return "", 0.0
    top = counts.index[0] if len(counts) == 1 or counts.iloc[0] > counts.iloc[1] else ""
    return str(top), float(counts.iloc[0] / len(values))


def consensus(labels: pd.DataFrame, stain: str, vocabulary: Vocabulary | None = None) -> pd.DataFrame:
    """One row per labelled object: the majority identity (`value`, "" when people are split), the majority subtype
    among the people who gave one for that identity, number of people, agreement on the identity, and whether most say
    the object is positive (None on a tie). Labels that say nothing ("unsure") are left out."""
    rows = classes(labels, stain, vocabulary)
    rows = rows[rows.role != vocab.SKIP].copy()
    if rows.empty:
        return pd.DataFrame(columns=["tile_id", "x", "y", "value", "subtype", "n", "agreement", "positive", "sources"])
    rows["rx"], rows["ry"] = rows.x.round().astype(int), rows.y.round().astype(int)
    out = []
    for (tile, rx, ry), group in rows.groupby(["tile_id", "rx", "ry"], sort=False):
        top, agreement = majority(group.identity)
        typed = group.subtype[(group.identity == top) & (group.subtype != "")] if top else group.subtype.iloc[:0]
        votes = (group.role == vocab.POSITIVE).to_numpy()
        out.append({"tile_id": tile, "x": float(group.x.iloc[0]), "y": float(group.y.iloc[0]), "value": top, "subtype": majority(typed)[0],
                    "n": len(group), "agreement": agreement, "positive": bool(votes.mean() > 0.5) if votes.mean() != 0.5 else None,
                    "sources": ", ".join(sorted(group.source))})
    return pd.DataFrame(out)


def reviewed_fields(labels: pd.DataFrame, stain: str) -> dict[str, set[str]]:
    """Fields someone finished reviewing, and who."""
    done = labels[(labels.stain == stain) & (labels.task == "done")]
    return {tile: set(group.source) for tile, group in done.groupby("tile_id")}
