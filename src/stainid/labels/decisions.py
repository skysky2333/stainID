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

import pandas as pd

from stainid import cache
from stainid.labels import candidate_store, frames, store
from stainid.labels import vocabulary as vocab
from stainid.project import Project

COLUMNS = ["candidate", "source", "identity", "subtype", "role", "time", "session", "confirmed"]
_LOCK = threading.Lock()
_LEGACY: dict[tuple[str, str], tuple[tuple, pd.Series]] = {}  # (project, stain) -> (what it was made from, candidate of each old label)
_STATE: dict[tuple[str, str], tuple[tuple, State]] = {}


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


def _state_of(rows: pd.DataFrame) -> State:
    out = State(rows=rows)
    if rows.empty:
        return out
    out.answered = {c: set(g) for c, g in rows.groupby("candidate").source}
    said = rows[rows.role != vocab.SKIP].sort_values("time", kind="stable")
    for candidate, group in said.groupby("candidate", sort=False):
        out.decided[candidate] = list(dict.fromkeys(group.source))
        top = group.identity.value_counts()
        identity = str(top.index[0])
        typed = group.subtype[(group.identity == identity) & (group.subtype.fillna("") != "")]
        out.calls[candidate] = (identity, str(typed.value_counts().index[0]) if len(typed) else "")
        out.times[candidate] = float(group.time.max())
    return out


def state(project: Project, stain: str, tiles: pd.DataFrame) -> State:
    """The decisions as the label log stands now, made again whenever a label is saved."""
    root = project.output("reviews")
    signature = (store.signature(root), _store_signature(project, stain))
    key = (str(project.root), stain)
    with _LOCK:
        kept = _STATE.get(key)
    if kept is not None and kept[0] == signature:
        return kept[1]
    made = _state_of(table(project, stain, store.current(root), tiles))
    with _LOCK:
        _STATE[key] = (signature, made)
    return made


def forget() -> None:
    with _LOCK:
        _LEGACY.clear()
        _STATE.clear()
