"""Every label in one append-only log: who said what about which object (or field), for which task.

A label is {id, source, stain, task, tile_id, x, y, value, points, session, confirmed, time, deleted, subtype, candidate}. `source` is a
person's name or `import: <set>`; the pipeline's own call is not stored here, it is read from the results. Objects are
identified by field and position (rounded pixel), so labels from different people and sessions about the same object
line up. The latest version of (source, id) wins; edits and deletions keep the history.

Tasks: `class` (the object's class, also used for objects people add: `value` is its identity and `subtype` its
subtype, see `labels.vocabulary`; labels made before subtypes existed have one value and no subtype),`outline` (polygon), `tag`, `note` and `done`
(a person finished reviewing the field; its `value` says which review, see `reviewed_fields`).
"""
from __future__ import annotations

import fcntl
import json
import time
from pathlib import Path

import numpy as np
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
    kept = _CACHE.get(path)
    if kept is not None and kept[0] == key:
        return kept[1]
    with path.open("rb") as handle:
        fcntl.flock(handle, fcntl.LOCK_SH)
        start = 0
        if kept is not None and len(kept) > 4 and kept[3] == stat.st_ino and kept[2] <= stat.st_size:  # the log only grows: read what was added
            handle.seek(kept[2])
            start = kept[2]
        data = handle.read()
        fcntl.flock(handle, fcntl.LOCK_UN)
    end = data.rfind(b"\n") + 1  # a line still being written is read next time
    records = (kept[4] if start else []) + [json.loads(line) for line in data[:end].decode("utf-8").splitlines() if line]
    frame = pd.DataFrame(records, columns=COLUMNS)  # from every record at once: the same columns as reading the whole log
    _CACHE[path] = (key, frame, start + end, stat.st_ino, records)
    return frame


_CURRENT: dict[Path, tuple[tuple, pd.DataFrame]] = {}


def current(root: Path) -> pd.DataFrame:
    """The latest version of every label that is not deleted (kept until a label is saved; a copy is returned)."""
    frame = history(root)
    path = log_path(root)
    key = (_CACHE[path][0], id(frame)) if path in _CACHE else None
    kept = _CURRENT.get(path)
    if key is not None and kept is not None and kept[0] == key:
        return kept[1].copy()
    latest = frame.drop_duplicates(["source", "id"], keep="last")
    latest = latest[~latest.deleted.astype(bool)].reset_index(drop=True)
    if key is not None:
        _CURRENT[path] = (key, latest)
    return latest.copy()


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
    group = rows.groupby(["tile_id", "rx", "ry"], sort=False).ngroup().to_numpy()  # in the order objects first come
    frame = pd.DataFrame({"group": group, "identity": rows.identity.astype(str).to_numpy(object), "subtype": rows.subtype.astype(str).to_numpy(object)})
    n = np.bincount(group)
    top, share = _majorities(frame.group.to_numpy(), frame.identity.to_numpy(object), len(n))
    typed = frame[(frame.subtype != "") & (frame.identity.to_numpy(object) == top[frame.group.to_numpy()]) & (top[frame.group.to_numpy()] != "")]
    subtype, _ = _majorities(typed.group.to_numpy(), typed.subtype.to_numpy(object), len(n))
    positive = np.bincount(group, weights=(rows.role == vocab.POSITIVE).to_numpy(float), minlength=len(n)) / n
    first = pd.Series(np.arange(len(group))).groupby(group).min().to_numpy()
    sources = pd.Series(rows.source.to_numpy(object)).groupby(group).agg(lambda s: ", ".join(sorted(s)))
    return pd.DataFrame({"tile_id": list(rows.tile_id.to_numpy(object)[first]), "x": rows.x.astype(float).to_numpy()[first],
                         "y": rows.y.astype(float).to_numpy()[first], "value": list(top), "subtype": list(subtype), "n": n.tolist(),
                         "agreement": share.tolist(), "positive": [bool(p > 0.5) if p != 0.5 else None for p in positive],
                         "sources": list(sources.to_numpy(object))})


def _majorities(group: np.ndarray, values: np.ndarray, groups: int) -> tuple[np.ndarray, np.ndarray]:
    """`majority` of `values` within each of `groups` groups at once: the most common value ("" on a tie or for a group
    without values) and its share."""
    top = np.full(groups, "", dtype=object)
    share = np.zeros(groups)
    if not len(group):
        return top, share
    counted = pd.DataFrame({"group": group, "value": values}).groupby(["group", "value"], sort=False).size().reset_index(name="count")
    counted = counted.sort_values(["group", "count"], ascending=[True, False], kind="stable")
    best = counted.drop_duplicates("group")
    second = counted[counted.duplicated("group")].drop_duplicates("group").set_index("group")["count"]
    runner = second.reindex(best.group).fillna(0).to_numpy()
    total = np.bincount(group, minlength=groups)
    clear = best["count"].to_numpy() > runner
    top[best.group.to_numpy()[clear]] = best.value.to_numpy(object)[clear]
    share[best.group.to_numpy()] = best["count"].to_numpy() / total[best.group.to_numpy()]
    return top, share


def reviewed_fields(labels: pd.DataFrame, stain: str, kind: str = "") -> dict[str, set[str]]:
    """Fields someone finished reviewing, and who. `kind` is the review a `done` label closes (its `value`): "" a
    whole-field review (older sessions and "Fix the field"), `labels.rounds.CALLS` a field of "Check the AI's calls"."""
    done = labels[(labels.stain == stain) & (labels.task == "done")]
    done = done[done.value.fillna("").astype(str) == kind]
    return {tile: set(group.source) for tile, group in done.groupby("tile_id")}
