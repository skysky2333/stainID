"""People's decisions on the v2 candidates: one shared store for both annotation rounds, every session and every person.

A decision is a class label (`labels.store`) about a candidate (`labels.candidate_store`): the latest label of each
person about each candidate counts. Labels that name the candidate (`candidate`) are taken as they are; labels made
before candidate ids existed (earlier sessions, imported label sets) are matched onto candidates by position
(`candidate_store.match`), so nothing anyone labelled is asked again.

One key and one count for every reader: `store.object_keys` (the candidate id, the position only where no candidate is
named or found) and `store.tally` (the majority, "" when split). `table` / `state` are the decisions on candidates
(the rounds, scoring, the Cores view); `consensus` is the same count over every labelled object, candidates and the
objects people added that no candidate covers (training's truth, the Labels page).

A candidate is *decided* once `needed` people gave it a class that says something (not "unsure"): it shows as
"checked by <name>" in "Fix the field", with the class most of them gave (the latest one said when they are split).
Someone who answered "unsure" is not asked again, but the candidate stays open for the others.
"""
from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass, field

import pandas as pd

from stainid import cache
from stainid import exclusions as excluded_areas
from stainid.labels import candidate_store, frames, store
from stainid.labels import vocabulary as vocab
from stainid.project import Project

COLUMNS = ["candidate", "source", "identity", "subtype", "role", "time", "session", "confirmed"]
_LOCK = threading.Lock()
_LEGACY: OrderedDict[tuple, pd.Series] = OrderedDict()  # (project, stain, what it was made from) -> candidate of each old label
_LEGACY_LIMIT = 8  # the rounds' labels, training's and the Labels page's: each its own rows
_STATE: dict[tuple[str, str], tuple[tuple, State]] = {}
_PEOPLE: dict[tuple[str, str], tuple[State, State]] = {}  # (project, stain) -> (the state it was made from, people's only)
_MAKING: dict[tuple[str, str], threading.Lock] = {}


def _has_candidate(labels: pd.DataFrame) -> pd.Series:
    if "candidate" not in labels:
        return pd.Series(False, index=labels.index)
    return labels.candidate.fillna("").astype(str).str.len() > 0


def _store_signature(project: Project, stain: str) -> tuple:
    names = cache.names(candidate_store.folder(project, stain))
    return cache.signature(candidate_store.folder(project, stain)), len(names)


def _legacy(project: Project, stain: str, rows: pd.DataFrame, tiles: pd.DataFrame, in_field: bool = False) -> pd.Series:
    """The candidate each old label (one without a candidate id) is about, kept until those labels or the store change.
    `in_field`: the positions are in field-picture pixels already (`frames.in_field_frame`)."""
    digest = (len(rows), int(pd.util.hash_pandas_object(rows[["id", "source", "time", "x", "y"]].astype(str), index=False).sum()) if len(rows) else 0,
              _store_signature(project, stain), in_field)
    key = (str(project.root), stain, digest)
    with _LOCK:
        kept = _LEGACY.get(key)
        if kept is not None:
            _LEGACY.move_to_end(key)
            return kept
    moved = rows if in_field else frames.in_field_frame(rows, tiles, project.context_px)
    matched = candidate_store.match(project, stain, moved, tiles)
    with _LOCK:
        _LEGACY[key] = matched
        while len(_LEGACY) > _LEGACY_LIMIT:
            _LEGACY.popitem(last=False)
    return matched


def _unnamed(rows: pd.DataFrame) -> pd.DataFrame:
    """Labels to match onto candidates by position: those naming none, made where an object is, and not pieces of a
    split or merge (their own objects, `labels.objectedit`)."""
    from stainid.labels import objectedit  # noqa: PLC0415

    return rows[~_has_candidate(rows) & rows.x.notna() & ~objectedit.pieces_mask(rows)]


def table(project: Project, stain: str, labels: pd.DataFrame, tiles: pd.DataFrame) -> pd.DataFrame:
    """Every person's latest decision on each candidate of this stain: candidate, source, identity, subtype, role
    (`vocabulary` roles: positive, negative, skip), time, session and whether it confirmed the AI's call. Old labels no
    candidate is found for are left out (they are about no candidate)."""
    rows = labels[(labels.stain == stain) & (labels.task == "class")]
    if rows.empty:
        return pd.DataFrame(columns=COLUMNS)
    old = _unnamed(rows)
    matched = _legacy(project, stain, old, tiles) if len(old) else None
    rows = store.classes(rows, stain, vocab.of(project, stain))
    keyed = store.latest_say(rows, store.object_keys(rows, matched))
    keyed = keyed[~keyed.object.astype(str).str.startswith("at:")]
    if keyed.empty:
        return pd.DataFrame(columns=COLUMNS)
    keyed = keyed.assign(candidate=keyed.object, time=pd.to_numeric(keyed.time, errors="coerce").fillna(0.0),
                         confirmed=keyed.confirmed.fillna(False).astype(bool), session=keyed.session.fillna("").astype(str))
    return keyed[COLUMNS].reset_index(drop=True)


def matched(project: Project, stain: str, rows: pd.DataFrame, tiles: pd.DataFrame) -> pd.Series | None:
    """The candidate each label of `rows` naming none is about, found by position (None when every label names one):
    the `matched` of `store.object_keys`. `rows` in field-picture pixels (`frames.in_field_frame`)."""
    old = _unnamed(rows) if len(rows) else rows
    return _legacy(project, stain, old, tiles, in_field=True) if len(old) else None


def keys(project: Project, stain: str, rows: pd.DataFrame, tiles: pd.DataFrame) -> pd.Series:
    """What each label (field-picture pixels) is about, as every reader keys objects (`store.object_keys`)."""
    return store.object_keys(rows, matched(project, stain, rows, tiles))


def consensus(project: Project, stain: str, labels: pd.DataFrame, tiles: pd.DataFrame) -> pd.DataFrame:
    """One row per labelled object of this stain (`store.consensus`): old labels matched onto candidates first, so a
    candidate's labels count together whatever field or position they were made at. `labels` in field-picture pixels
    (`frames.in_field_frame`), as training and the Labels page read them."""
    rows = labels[(labels.stain == stain) & (labels.task == "class")] if len(labels) else labels
    return store.consensus(rows, stain, vocab.of(project, stain), matched(project, stain, rows, tiles))


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
    for candidate, source in zip(*(said.drop_duplicates(["candidate", "source"])[c].to_numpy(object) for c in ("candidate", "source"))):
        out.decided.setdefault(candidate, []).append(source)
    counted = store.tally(said.assign(object=said.candidate))  # the one count (training's truth too); split: the latest said
    out.calls = {c: ((v, s) if v else (latest, latest_subtype)) for c, v, s, latest, latest_subtype in
                 zip(counted.object, counted.value, counted.subtype, counted.latest, counted.latest_subtype)}
    out.times = {c: float(t) for c, t in zip(counted.object, counted.time)}
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


def people(project: Project, stain: str, tiles: pd.DataFrame, stale_ok: bool = False) -> State:
    """`state` without the AI's labels (`store.is_ai_label`: the imported reference sets, AI review sessions): what the
    annotation pages show and start from (Fix the field: who checked an object, the class it starts with). Kept until
    `state` is made again."""
    full = state(project, stain, tiles, stale_ok)
    key = (str(project.root), stain)
    with _LOCK:
        kept = _PEOPLE.get(key)
    if kept is not None and kept[0] is full:
        return kept[1]
    rows = full.rows
    ai = rows.source.map(store.is_ai_label).astype(bool) if len(rows) else pd.Series(False, index=rows.index)
    made = _state_of(rows[~ai]) if ai.any() else full
    with _LOCK:
        _PEOPLE[key] = (full, made)
    return made


def forget() -> None:
    with _LOCK:
        _LEGACY.clear()
        _STATE.clear()
        _PEOPLE.clear()
