"""Every label in one append-only log: who said what about which object (or field), for which task.

A label is {id, source, stain, task, tile_id, x, y, value, points, session, confirmed, time, deleted, subtype, candidate}. `source` is a
person's name or `import: <set>`; the pipeline's own call is not stored here, it is read from the results. A label's
`id` is its field and position (rounded pixel); the latest version of (source, id) wins, edits and deletions keep the
history. What a class label is *about* (`object_keys`) is the candidate it names (`candidate`), else the candidate found
at its position (labels made before candidate ids existed, `labels.decisions`), else its field and position: the one
key `consensus` groups by, for training, the Labels page and the rounds alike.

Tasks: `class` (the object's class, also used for objects people add: `value` is its identity and `subtype` its
subtype, see `labels.vocabulary`; labels made before subtypes existed have one value and no subtype),`outline` (polygon), `tag`, `note` and `done`
(a person finished reviewing the field; its `value` says which review, see `reviewed_fields`).
"""
from __future__ import annotations

import fcntl
import json
import time
from contextvars import ContextVar
from pathlib import Path

import numpy as np
import pandas as pd

from stainid.labels import vocabulary as vocab
from stainid.labels.vocabulary import Vocabulary

COLUMNS = ["id", "source", "stain", "task", "tile_id", "x", "y", "value", "points", "session", "confirmed", "time", "deleted", "subtype", "candidate",
           "meta"]  # meta: where a label comes from (a split or merge, `labels.objectedit`), else None
_CACHE: dict[Path, tuple[tuple[int, int], pd.DataFrame]] = {}


def log_path(root: Path) -> Path:
    return root / "labels" / "labels.jsonl"


def object_id(task: str, tile_id: str, x: float, y: float) -> str:
    return f"{task}:{tile_id}:{round(x)}:{round(y)}"


ATTRIBUTION: ContextVar[str | None] = ContextVar("stainid_account", default=None)  # the signed-in account (set by the app)


def append(root: Path, records: list[dict]) -> list[dict]:
    """Add labels to the log; each carries the account id of the signed-in person (`account`) besides their name."""
    path = log_path(root)
    account = ATTRIBUTION.get()
    path.parent.mkdir(parents=True, exist_ok=True)
    now = time.time()
    stamped = [{**{k: None for k in COLUMNS}, "confirmed": False, "deleted": False, "session": "", **r, "time": r.get("time") or now} for r in records]
    if account:
        stamped = [r if r.get("account") else {**r, "account": account} for r in stamped]
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


CONSENSUS = ["object", "candidate", "tile_id", "x", "y", "value", "subtype", "n", "agreement", "positive", "sources", "latest", "latest_subtype",
             "time"]


def object_keys(labels: pd.DataFrame, matched: pd.Series | None = None) -> pd.Series:
    """What each label is about: the candidate it names (`candidate`), else `matched` (the candidate found at its
    position, by label index), else its field and rounded position ("at:<tile>:<x>:<y>")."""
    out = pd.Series("", index=labels.index, dtype=object)
    if labels.empty:
        return out
    named = labels.candidate.fillna("").astype(str) if "candidate" in labels else out
    found = matched.reindex(labels.index).fillna("").astype(str) if matched is not None and len(matched) else out
    xs = pd.to_numeric(labels.x, errors="coerce").round()
    ys = pd.to_numeric(labels.y, errors="coerce").round()
    at = ["at:%s:%s:%s" % (t, "" if x != x else int(x), "" if y != y else int(y)) for t, x, y in zip(labels.tile_id, xs, ys)]
    return pd.Series(np.where(named != "", named, np.where(found != "", found, at)), index=labels.index, dtype=object)


def latest_say(rows: pd.DataFrame, keys: pd.Series) -> pd.DataFrame:
    """`rows` with their `object` (`keys`), keeping each person's latest label about each object (by `time`, then the
    order of the rows), in the order of the rows."""
    if rows.empty:
        return rows.assign(object=pd.Series(dtype=object))
    time = pd.to_numeric(rows.time, errors="coerce").fillna(0.0).to_numpy(float) if "time" in rows else np.zeros(len(rows))
    frame = pd.DataFrame({"source": rows.source.astype(str).to_numpy(object), "object": keys.reindex(rows.index).to_numpy(object),
                          "time": time, "position": np.arange(len(rows))})
    keep = frame.sort_values(["time", "position"], kind="stable").drop_duplicates(["source", "object"], keep="last").position.sort_values()
    return rows.iloc[keep.to_numpy()].assign(object=frame.object.to_numpy(object)[keep.to_numpy()])


def tally(rows: pd.DataFrame) -> pd.DataFrame:
    """One row per object of these labels (each with `object`, `source`, `identity`, `subtype`, `role`, optional
    `time`, `tile_id`, `x`, `y`; one per person and object, `latest_say`), in the order objects first come: the
    majority identity (`value`, "" when people are split), the majority subtype among the people who gave one for that
    identity ("" on a tie), number of people, agreement on the identity, whether most say the object is positive (None
    on a tie), who, the latest identity and subtype said (`latest`, `latest_subtype`) and when. Labels that say nothing
    ("unsure") are left out. The one counting rule: training's truth, the Labels page and the rounds' calls."""
    rows = rows[rows.role != vocab.SKIP]
    if rows.empty:
        return pd.DataFrame(columns=CONSENSUS)
    group, objects = pd.factorize(rows.object.to_numpy(object))  # in the order objects first come
    identity, subtype = rows.identity.astype(str).to_numpy(object), rows.subtype.fillna("").astype(str).to_numpy(object)
    n = np.bincount(group, minlength=len(objects))
    top, share = _majorities(group, identity, len(n))
    typed = (subtype != "") & (identity == top[group]) & (top[group] != "")
    sub, _ = _majorities(group[typed], subtype[typed], len(n))
    positive = np.bincount(group, weights=(rows.role == vocab.POSITIVE).to_numpy(float), minlength=len(n)) / n
    time = pd.to_numeric(rows.time, errors="coerce").fillna(0.0).to_numpy(float) if "time" in rows else np.zeros(len(rows))
    order = np.lexsort((np.arange(len(rows)), time))  # by time, then the order of the rows
    last = pd.Series(order).groupby(group[order]).last().reindex(range(len(n))).to_numpy(int)  # the latest label of each object
    first = pd.Series(np.arange(len(group))).groupby(group).min().to_numpy()
    sources = pd.Series(rows.source.astype(str).to_numpy(object)).groupby(group).agg(lambda s: ", ".join(sorted(s))).to_numpy(object)
    column = (lambda name, fill: rows[name].to_numpy(object)[first] if name in rows else np.full(len(n), fill, dtype=object))  # noqa: E731
    keys = np.asarray(objects, dtype=object)
    return pd.DataFrame({"object": list(keys), "candidate": ["" if str(k).startswith("at:") else k for k in keys],
                         "tile_id": list(column("tile_id", "")), "x": pd.to_numeric(pd.Series(column("x", np.nan)), errors="coerce").to_numpy(float),
                         "y": pd.to_numeric(pd.Series(column("y", np.nan)), errors="coerce").to_numpy(float),
                         "value": list(top), "subtype": list(sub), "n": n.tolist(), "agreement": share.tolist(),
                         "positive": [bool(p > 0.5) if p != 0.5 else None for p in positive], "sources": list(sources),
                         "latest": list(identity[last]), "latest_subtype": list(subtype[last]),
                         "time": pd.Series(time).groupby(group).max().to_numpy(float).tolist()}, columns=CONSENSUS)


def consensus(labels: pd.DataFrame, stain: str, vocabulary: Vocabulary | None = None, matched: pd.Series | None = None) -> pd.DataFrame:
    """One row per labelled object of this stain (`tally`), the object being the candidate a label names, else the
    one `matched` at its position, else its field and position (`object_keys`); each person's latest label about it
    counts. `tile_id`, `x`, `y` are those of the object's first label. `labels.decisions.consensus` matches old labels
    onto candidates first."""
    rows = classes(labels, stain, vocabulary)
    if rows.empty:
        return pd.DataFrame(columns=CONSENSUS)
    return tally(latest_say(rows, object_keys(rows, matched)))


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
