"""Every label, in append-only logs: who said what about which object (or field), for which task.

One log per labelling session and per label set, all under `<reviews>/labels/` (`file_of`):

    labels/sessions/<session>.jsonl   people's labels in a labelling session
    labels/imports/<set>.jsonl        an imported label set ("import: <set>"; an imported set with a session: that session)
    labels/ai/<session or source>.jsonl  the AI's labels (an AI review session's, the earlier "AI reviewer")
    labels/other/<source>.jsonl       anything else made outside a session
    labels/changes.log                one line more per save: what caches watch (`log_path`, `signature`)

Training, the Labels and Models pages and the Cores view read the union of them all (`current(root)`), as one log;
inside a session only its own labels count (`current(root, session=name)`). Deleting a session moves its logs into its
archive (`remove_session`): nothing reads them any more. A project of an earlier release has one `labels/labels.jsonl`:
`split_legacy` (the `stainid labels split` command, with the app stopped) splits it once; the app does not open a
project in the old layout, nor one with both (`check_layout`).

A label is {id, source, stain, task, tile_id, x, y, value, points, session, confirmed, time, deleted, subtype, candidate}. `source` is a
person's name or `import: <set>`; the pipeline's own call is not stored here, it is read from the results. A label's
`id` is its field and position (rounded pixel); the latest version of (session, source, id) wins, edits and deletions
keep the history (a session's labels are its own: nothing said in another session replaces or deletes them). What a class label is *about* (`object_keys`) is the candidate it names (`candidate`), else the candidate found
at its position (labels made before candidate ids existed, `labels.decisions`), else its field and position: the one
key `consensus` groups by, for training, the Labels page and the rounds alike.

Tasks: `class` (the object's class, also used for objects people add: `value` is its identity and `subtype` its
subtype, see `labels.vocabulary`; labels made before subtypes existed have one value and no subtype),`outline` (polygon), `tag`, `note` and `done`
(a person finished reviewing the field; its `value` says which review, see `reviewed_fields`).
"""
from __future__ import annotations

import fcntl
import json
import re
import shutil
import time
from contextvars import ContextVar
from pathlib import Path

import numpy as np
import pandas as pd

from stainid.labels import vocabulary as vocab
from stainid.labels.vocabulary import Vocabulary

AI_REVIEWER = "AI reviewer"  # the earlier AI-review rounds' hand-drawn areas (`stainid.exclusions.REVIEWER`)


def is_ai_label(source) -> bool:
    """Whether a label (object or area) is the AI's, not a person's: the imported reference sets ("import: …"), the
    earlier AI reviewer ("AI reviewer") and AI review sessions (accounts "ai:<name>"). The annotation pages (Fix the
    field, Mark bad areas, the brush, the tutorial) never show them nor start from them: people correct the pipeline's
    output, not the AI's. The Labels and Models pages and the Cores view's Annotations layer keep them, marked as AI;
    training and evaluation take them as their session is included. The page has the same rule
    (webapp/src/labels/aiLabels.ts)."""
    text = "" if source is None else str(source)
    return text.startswith("import:") or text == AI_REVIEWER or text.startswith("ai:")

COLUMNS = ["id", "source", "stain", "task", "tile_id", "x", "y", "value", "points", "session", "confirmed", "time", "deleted", "subtype", "candidate",
           "meta"]  # meta: where a label comes from (a split or merge, `labels.objectedit`), else None
_CACHE: dict[Path, tuple] = {}  # one log -> (its signature, frame, bytes read, inode, records)
GROUPS = ("sessions", "imports", "ai", "other")  # the folders of the logs (see the module's docstring)
LEGACY, BEFORE_SPLIT, CHANGES = "labels.jsonl", "labels.jsonl.before_split", "changes.log"


class LayoutError(RuntimeError):
    """The label store is in an earlier release's layout (one labels.jsonl), or in both layouts at once."""


def folder(root: Path) -> Path:
    return root / "labels"


def log_path(root: Path) -> Path:
    """The file whose signature changes with every save to any log (`changes.log`): what caches of results made from
    the labels name as their input."""
    return folder(root) / CHANGES


def legacy_path(root: Path) -> Path:
    return folder(root) / LEGACY


def _slug(text: str) -> str:
    out = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text).strip()).strip("._")
    return out or "unnamed"


def file_of(root: Path, record: dict) -> Path:
    """The log a label belongs in: its session's (people's under sessions/, an imported set's under imports/, the AI's
    under ai/), else its import or AI set's by `source`, else other/<source>."""
    session = str(record.get("session") or "")
    source = str(record.get("source") or "")
    base = folder(root)
    if session:
        group = "imports" if source.startswith("import:") else "ai" if is_ai_label(source) else "sessions"
        return base / group / f"{_slug(session)}.jsonl"
    if source.startswith("import:"):
        return base / "imports" / f"{_slug(source[len('import:'):])}.jsonl"
    if is_ai_label(source):
        return base / "ai" / f"{_slug(source)}.jsonl"
    return base / "other" / f"{_slug(source)}.jsonl"


def session_files(root: Path, name: str) -> list[Path]:
    """The logs of one session (people's, and an imported or AI session's), those that exist."""
    base = folder(root)
    return [path for path in (base / group / f"{_slug(name)}.jsonl" for group in ("sessions", "imports", "ai")) if path.exists()]


_FILES: dict[str, tuple] = {}


def files(root: Path) -> list[Path]:
    """Every log of the store (not the archives of deleted sessions); listed again when changes.log changes (a log is
    only made or archived together with a line there)."""
    stamp = signature(root)
    kept = _FILES.get(str(root))
    if stamp is not None and kept is not None and kept[0] == stamp:
        return kept[1]
    base = folder(root)
    found = sorted(path for group in GROUPS for path in (base / group).glob("*.jsonl"))
    _FILES[str(root)] = (stamp, found)
    return found


def layout(root: Path) -> str:
    """"split" (this release's), "legacy" (one labels.jsonl), "empty"; LayoutError when both are there."""
    legacy = legacy_path(root).exists()
    split = log_path(root).exists() or bool(files(root))
    if legacy and split:
        raise LayoutError(f"{folder(root)} has both the old labels.jsonl and the split logs: the split was not finished or labels were "
                          "written to the old log after it. Stop the app, look at both, and keep one layout.")
    return "legacy" if legacy else "split" if split else "empty"


def check_layout(root: Path) -> None:
    """Refuses (LayoutError) a store the app cannot use: in the old layout (split it first) or in both."""
    if layout(root) == "legacy":
        raise LayoutError(f"{folder(root)} still has one labels.jsonl (an earlier release): with the app stopped, run "
                          f"`stainid --project <project> labels split` once, then start the app again.")


def object_id(task: str, tile_id: str, x: float, y: float) -> str:
    return f"{task}:{tile_id}:{round(x)}:{round(y)}"


ATTRIBUTION: ContextVar[str | None] = ContextVar("stainid_account", default=None)  # the signed-in account (set by the app)


def _append_lines(path: Path, lines: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.write(lines)
        handle.flush()
        fcntl.flock(handle, fcntl.LOCK_UN)


def _changed(root: Path) -> None:
    """One line more in changes.log: every cache made from the labels looks again (the log only grows, so its size
    changes even within the file system's time resolution)."""
    _append_lines(log_path(root), f"{time.time():.6f}\n")


def append(root: Path, records: list[dict]) -> list[dict]:
    """Add labels, each to its log (`file_of`); each carries the account id of the signed-in person (`account`) besides
    their name. Never while practising (`stainid.practice`), never into a store in the old layout."""
    from stainid import practice  # noqa: PLC0415

    practice.guard("the label log")
    if legacy_path(root).exists():
        check_layout(root)
    account = ATTRIBUTION.get()
    now = time.time()
    stamped = [{**{k: None for k in COLUMNS}, "confirmed": False, "deleted": False, "session": "", **r, "time": r.get("time") or now} for r in records]
    if account:
        stamped = [r if r.get("account") else {**r, "account": account} for r in stamped]
    by_file: dict[Path, list[dict]] = {}
    for record in stamped:
        by_file.setdefault(file_of(root, record), []).append(record)
    for path, rows in by_file.items():
        _append_lines(path, "".join(json.dumps(r) + "\n" for r in rows))
    if stamped:
        _changed(root)
    return stamped


def signature(root: Path) -> tuple[int, int] | None:
    """Changes whenever a label is saved (to any log) or a session's logs are archived."""
    path = log_path(root)
    if not path.exists():
        return None
    stat = path.stat()
    return stat.st_mtime_ns, stat.st_size


def _file_history(path: Path) -> pd.DataFrame:
    """One log, read again only where it grew."""
    stat = path.stat()
    key = (stat.st_mtime_ns, stat.st_size)
    kept = _CACHE.get(path)
    if kept is not None and kept[0] == key and len(kept) > 4 and kept[3] == stat.st_ino:
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


_UNION: dict[tuple, tuple[tuple, pd.DataFrame]] = {}


def history(root: Path, session: str | None = None) -> pd.DataFrame:
    """Every version of every label, oldest first: of the whole store (every log, as one), or of one session."""
    if legacy_path(root).exists():
        check_layout(root)
    paths = session_files(root, session) if session is not None else files(root)
    frames = [_file_history(path) for path in paths]
    key = (str(root), session)
    stamp = tuple((str(path), id(frame)) for path, frame in zip(paths, frames))
    kept = _UNION.get(key)
    if kept is not None and kept[0] == stamp:
        return kept[1]
    if not frames:
        frame = pd.DataFrame(columns=COLUMNS)
    elif len(frames) == 1:
        frame = frames[0]
    else:
        joined = pd.concat([f for f in frames if len(f)] or frames[:1], ignore_index=True)
        order = pd.to_numeric(joined.time, errors="coerce").fillna(0.0)
        frame = joined.iloc[order.sort_values(kind="stable").index].reset_index(drop=True)  # one log, in the order labels were saved
    if session is not None and len(frame):
        own = frame.session.fillna("").astype(str) == session  # a set's log named like the session holds no label of it
        if not own.all():
            frame = frame[own].reset_index(drop=True)
    _UNION[key] = (stamp, frame)
    return frame


def raw_records(root: Path) -> list[dict]:
    """Every line of every log as saved (all fields, `account` too), oldest first: for checks and exports."""
    out = []
    for path in files(root):
        out += [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return sorted(out, key=lambda r: float(r.get("time") or 0.0))


def line_count(root: Path) -> int:
    """How many lines all the logs hold (every save adds lines, none is rewritten)."""
    return sum(1 for path in files(root) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


_CURRENT: dict[tuple, tuple[int, pd.DataFrame]] = {}


def current(root: Path, session: str | None = None) -> pd.DataFrame:
    """The latest version of every label that is not deleted (a copy): of the whole store, or of one session only (what
    counts inside it: its fields' states, its progress, who is on it, what its fields start from)."""
    frame = history(root, session)
    key = (str(root), session)
    kept = _CURRENT.get(key)
    if kept is not None and kept[0] == id(frame):
        return kept[1].copy()
    keys = ["session", "source", "id"] if "session" in frame else ["source", "id"]  # a session's labels are its own: one
    latest = frame.drop_duplicates(keys, keep="last")  # in another session (or a delete there) never replaces them
    latest = latest[~latest.deleted.astype(bool)].reset_index(drop=True)
    _CURRENT[key] = (id(frame), latest)
    return latest.copy()


def remove_session(root: Path, name: str, archive: Path) -> int:
    """A deleted session's logs go into its archive (`archive`, beside the session's own files): nothing reads them any
    more, so its labels leave training, scoring and every page; an admin can bring them back by hand. How many labels
    left (the current ones)."""
    from stainid import practice  # noqa: PLC0415

    practice.guard("the label log")
    paths = session_files(root, name)
    count = len(current(root, name)) if paths else 0
    if not paths:
        return 0
    archive.mkdir(parents=True, exist_ok=True)
    for path in paths:
        target = archive / f"labels-{path.parent.name}.jsonl"
        with path.open("r+", encoding="utf-8") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)  # no label is added meanwhile
            lines = [line for line in handle.read().splitlines() if line.strip()]
            own = [str(json.loads(line).get("session") or "") == name for line in lines]
            if all(own):
                shutil.move(str(path), str(target))
            else:  # a label set's log of the same name: only the session's own lines go
                target.write_text("".join(line + "\n" for line, mine in zip(lines, own) if mine), encoding="utf-8")
                handle.seek(0)
                handle.truncate()
                handle.write("".join(line + "\n" for line, mine in zip(lines, own) if not mine))
                handle.flush()
            fcntl.flock(handle, fcntl.LOCK_UN)
    _changed(root)
    return count


def split_legacy(root: Path) -> dict:
    """The one-off move from the earlier release's single labels.jsonl to one log per session and set: every line goes,
    in its order, to its log (`file_of`); the counts are checked; the old file is kept as labels.jsonl.before_split. Run
    it with the app stopped. Refuses when there is nothing to split or the split logs exist already."""
    source = legacy_path(root)
    if not source.exists():
        raise LayoutError(f"{source} does not exist: nothing to split")
    if log_path(root).exists() or files(root):
        raise LayoutError(f"{folder(root)} has split logs already: the split was done (or started); look before running it again")
    by_file: dict[Path, list[str]] = {}
    total = 0
    with source.open("rb") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)  # no writer while it is read
        data = handle.read()
        fcntl.flock(handle, fcntl.LOCK_UN)
    for line in data.decode("utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        by_file.setdefault(file_of(root, record), []).append(line.rstrip("\n"))
        total += 1
    written: dict[str, int] = {}
    try:
        for path, lines in by_file.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(path.name + ".part")
            tmp.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
            tmp.replace(path)
            written[str(path.relative_to(folder(root)))] = sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
        if sum(written.values()) != total or any(written[str(p.relative_to(folder(root)))] != len(lines) for p, lines in by_file.items()):
            raise LayoutError(f"the split logs hold {sum(written.values())} labels, the old log {total}: nothing was changed")
    except BaseException:
        for path in by_file:
            path.unlink(missing_ok=True)
        raise
    source.rename(folder(root) / BEFORE_SPLIT)
    _changed(root)
    return {"labels": total, "files": dict(sorted(written.items())), "kept": str(folder(root) / BEFORE_SPLIT)}


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
