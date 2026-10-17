"""People's decisions on the v2 candidates: one shared store for both annotation rounds, every session and every person.

A decision is a class label (`labels.store`) about a candidate (`labels.candidate_store`): the latest label of each
person about each candidate counts. Labels that name the candidate (`candidate`) are taken as they are; labels made
before candidate ids existed (earlier sessions, imported label sets) are matched onto candidates by position
(`candidate_store.match`), so nothing anyone labelled is asked again.

A candidate is *decided* once `needed` people gave it a class that says something (not "unsure"): it then leaves every
round-1 list at once and shows as "checked by <name>" in round 2. Someone who answered "unsure" is not asked again,
but the candidate stays open for the others.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from stainid import cache
from stainid import exclusions as excluded_areas
from stainid.labels import candidate_store, frames, store
from stainid.labels import vocabulary as vocab
from stainid.project import Project

COLUMNS = ["candidate", "source", "identity", "subtype", "role", "time", "session", "confirmed"]
_LOCK = threading.Lock()
_LEGACY: dict[tuple[str, str], tuple[tuple, pd.Series]] = {}  # (project, stain) -> (what it was made from, candidate of each old label)
_STATE: dict[tuple[str, str], tuple[tuple, State]] = {}
_MAKING: dict[tuple[str, str], threading.Lock] = {}


def _has_candidate(labels: pd.DataFrame) -> pd.Series:
    if "candidate" not in labels:
        return pd.Series(False, index=labels.index)
    return labels.candidate.fillna("").astype(str).str.len() > 0


def _store_signature(project: Project, stain: str) -> tuple:
    names = cache.names(candidate_store.folder(project, stain))
    return cache.signature(candidate_store.folder(project, stain)), len(names)


def _legacy(project: Project, stain: str, rows: pd.DataFrame, tiles: pd.DataFrame) -> pd.Series:
    """The candidate each old label (one without a candidate id) is about, kept until those labels or the store change."""
    digest = (len(rows), int(pd.util.hash_pandas_object(rows[["id", "source", "time"]].astype(str), index=False).sum()) if len(rows) else 0,
              _store_signature(project, stain))
    key = (str(project.root), stain)
    with _LOCK:
        kept = _LEGACY.get(key)
    if kept is not None and kept[0] == digest:
        return kept[1]
    matched = candidate_store.match(project, stain, frames.in_field_frame(rows, tiles, project.context_px), tiles)
    with _LOCK:
        _LEGACY[key] = (digest, matched)
    return matched


def table(project: Project, stain: str, labels: pd.DataFrame, tiles: pd.DataFrame) -> pd.DataFrame:
    """Every person's latest decision on each candidate of this stain: candidate, source, identity, subtype, role
    (`vocabulary` roles: positive, negative, skip), time, session and whether it confirmed the AI's call."""
    rows = labels[(labels.stain == stain) & (labels.task == "class")]
    if rows.empty:
        return pd.DataFrame(columns=COLUMNS)
    named = _has_candidate(rows)
    direct = rows[named]
    old = rows[~named & rows.x.notna()]
    matched = _legacy(project, stain, old, tiles) if len(old) else pd.Series(dtype=object)
    old = old.assign(candidate=matched.reindex(old.index)).dropna(subset=["candidate"])
    both = pd.concat([direct, old]) if len(old) else direct
    if both.empty:
        return pd.DataFrame(columns=COLUMNS)
    both = store.classes(both, stain, vocab.of(project, stain))
    both = both.assign(time=pd.to_numeric(both.time, errors="coerce").fillna(0.0), confirmed=both.confirmed.fillna(False).astype(bool),
                       session=both.session.fillna("").astype(str))
    latest = both.sort_values("time", kind="stable").drop_duplicates(["source", "candidate"], keep="last")
    return latest[COLUMNS].reset_index(drop=True)


@dataclass
class State:
    """Who decided what about each candidate (as the label log stands)."""
    decided: dict[str, list[str]] = field(default_factory=dict)  # candidate -> people whose class says something, oldest first
    answered: dict[str, set[str]] = field(default_factory=dict)  # candidate -> everyone who answered (also "unsure")
    calls: dict[str, tuple[str, str]] = field(default_factory=dict)  # candidate -> the class most deciders gave (identity, subtype)
    times: dict[str, float] = field(default_factory=dict)  # candidate -> when it was last decided
    rows: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=COLUMNS))

    def count(self, candidate: str) -> int:
        return len(self.decided.get(candidate, ()))

    def done(self, candidate: str, needed: int) -> bool:
        return self.count(candidate) >= max(1, needed)

    def by(self, candidate: str) -> list[str]:
        return self.decided.get(candidate, [])


def _most_common(frame: pd.DataFrame, column: str) -> pd.Series:
    """Per candidate, the most common value of `column` among its rows (in their order); a tie goes to the value that
    came first, as `value_counts` orders them."""
    counted = frame.groupby(["candidate", column], sort=False).position.agg(["size", "min"]).reset_index()
    counted = counted.sort_values(["size", "min"], ascending=[False, True], kind="stable").drop_duplicates("candidate")
    return counted.set_index("candidate")[column]


def _state_of(rows: pd.DataFrame) -> State:
    """Who decided what, counted for every candidate at once (grouped, not candidate by candidate: a submitted tile
    records each of its candidates)."""
    out = State(rows=rows)
    if rows.empty:
        return out
    for candidate, source in zip(rows.candidate.to_numpy(object), rows.source.to_numpy(object)):
        out.answered.setdefault(candidate, set()).add(source)
    said = rows[rows.role != vocab.SKIP].sort_values("time", kind="stable")
    if said.empty:
        return out
    said = pd.DataFrame({"candidate": said.candidate.to_numpy(object), "source": said.source.to_numpy(object),
                         "identity": said.identity.astype(str).to_numpy(object), "subtype": said.subtype.fillna("").astype(str).to_numpy(object),
                         "time": said.time.to_numpy(float), "position": np.arange(len(said))})
    for candidate, source in zip(*(said.drop_duplicates(["candidate", "source"])[c].to_numpy(object) for c in ("candidate", "source"))):
        out.decided.setdefault(candidate, []).append(source)
    identity = _most_common(said, "identity")
    typed = said[(said.subtype != "") & (said.identity.to_numpy(object) == identity.reindex(said.candidate).to_numpy(object))]
    subtype = _most_common(typed, "subtype") if len(typed) else pd.Series(dtype=object)
    latest = said.groupby("candidate", sort=False).time.max()
    subtypes = subtype.to_dict()
    out.calls = {c: (identity[c], subtypes.get(c, "")) for c in out.decided}
    out.times = {c: float(t) for c, t in latest.items()}
    return out


def _signature(project: Project, stain: str) -> tuple:
    return store.signature(project.output("reviews")), _store_signature(project, stain), excluded_areas.signature(project, stain)


def cached(project: Project, stain: str) -> bool:
    """Whether `state` answers at once (made since the last label was saved)."""
    with _LOCK:
        kept = _STATE.get((str(project.root), stain))
    return kept is not None and kept[0] == _signature(project, stain)


def state(project: Project, stain: str, tiles: pd.DataFrame, stale_ok: bool = False) -> State:
    """The decisions as the label log stands now, made again whenever a label is saved (once: parallel requests wait
    for the one making it). `stale_ok` (the pages' heartbeat): while it is being made again, the previous one."""
    root = project.output("reviews")
    signature = _signature(project, stain)
    key = (str(project.root), stain)
    with _LOCK:
        kept = _STATE.get(key)
        making = _MAKING.setdefault(key, threading.Lock())
    if kept is not None and kept[0] == signature:
        return kept[1]
    if stale_ok and kept is not None:
        if not making.acquire(blocking=False):
            return kept[1]
    else:
        making.acquire()
    try:
        with _LOCK:
            kept = _STATE.get(key)
        signature = _signature(project, stain)
        if kept is not None and kept[0] == signature:
            return kept[1]
        made = _state_of(table(project, stain, excluded_areas.drop_excluded(project, store.current(root)), tiles))  # not on excluded objects
        with _LOCK:
            _STATE[key] = (signature, made)
        return made
    finally:
        making.release()


def forget() -> None:
    with _LOCK:
        _LEGACY.clear()
        _STATE.clear()
