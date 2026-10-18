"""Where labelling stands, per stain: coverage, what people said, how often people agree, and how the AI compares.

How the AI compares is measured on every real object people labelled (`detection`), not only on the objects the
detection step proposed: a real object with no pool candidate at its position (within `pool.MATCH_PX`) on a prepared
field is one the AI never proposed, a missed object like one it proposed but rejected. The older figure, among proposed
objects only (`ai`), is kept for comparison.
"""
from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path

import pandas as pd

from stainid import cache
from stainid import exclusions as excluded_areas
from stainid.labels import added, candidate_store, compare, decisions, frames, pool, store
from stainid.labels import vocabulary as vocab
from stainid.labels.tasks import positive
from stainid.project import Project

_CACHE: dict[tuple, tuple[tuple, dict]] = {}
_MAKING: dict[tuple, threading.Lock] = {}  # one count at a time per project and stain (the warm-up and a page share it)
_MAKING_LOCK = threading.Lock()
SPLIT = "people disagree"  # an object people labelled differently, as many for one class as for another


def match(consensus: pd.DataFrame, objects: pd.DataFrame, stain: str, prepared: set[str] | None = None,
          extra: pd.DataFrame | None = None, stored: pd.Series | None = None) -> pd.DataFrame:
    """Each consensus label with the AI call of the pool candidate at its position, and whether the detection step
    proposed the object at all: True (a candidate is there), False (no candidate, on a field in `prepared`: the AI never
    proposed it) or None (no candidate on a field that is not prepared, so it cannot tell). `extra`: measured added
    objects (`labels.added`); one that turned out to be a candidate counts as that candidate. `stored`: the AI's call of
    the v2 store candidate at each row's position (`candidate_store.calls_at`, None where there is none)."""
    by_tile = {t: g for t, g in objects.groupby("tile_id")}
    extra_by_tile = {t: g for t, g in extra.groupby("tile_id")} if extra is not None and len(extra) else {}
    prepared = prepared or set()
    calls, proposed = [], []
    for row in consensus.itertuples():
        candidates = by_tile.get(row.tile_id)
        i = pool.nearest(candidates, row.x, row.y)
        hit = added.resolve(extra_by_tile.get(row.tile_id), row.x, row.y) if i is None else None
        if i is not None:
            calls.append(candidates.iloc[i].ai_value)
            proposed.append(True)
        elif stored is not None and stored.get(row.Index) is not None:
            calls.append(stored[row.Index])
            proposed.append(True)
        elif hit is not None and pd.notna(hit.candidate_x):
            calls.append(hit.ai_value)
            proposed.append(True)
        else:
            calls.append(None)
            proposed.append(False if row.tile_id in prepared else None)
    calls = [None if isinstance(c, float) and pd.isna(c) else c for c in calls]
    return consensus.assign(ai_value=calls, ai_positive=[None if c is None else positive(stain, c) for c in calls], proposed=proposed)


def detection(matched: pd.DataFrame) -> dict:
    """How many of the real objects people labelled the AI finds, and why it misses the others: never proposed (no
    candidate there) or proposed but rejected. Real objects on fields that are not prepared cannot be checked."""
    real = matched[matched.positive.astype(object).eq(True)]
    never = int(real.proposed.astype(object).eq(False).sum())
    proposed = real[real.proposed.astype(object).eq(True) & real.ai_positive.notna()]
    found = int(proposed.ai_positive.astype(bool).sum())
    rejected = int(len(proposed) - found)
    n = found + rejected + never
    share = (lambda k: k / n if n else None)
    return {"real": n, "found": found, "never_proposed": never, "rejected": rejected, "recall": share(found),
            "never_proposed_share": share(never), "rejected_share": share(rejected), "not_checked": int(len(real) - n)}


def class_counts(consensus: pd.DataFrame, vocabulary: vocab.Vocabulary) -> list[dict]:
    """Labelled objects per identity (in the order of their keys, then classes it no longer has), with how many of
    them carry each subtype and how many no type; objects people are split on are left out."""
    by_identity = {i: g for i, g in consensus.groupby("value") if i} if len(consensus) else {}
    known = [c.id for c in vocabulary.by_key()]  # the order of the keys, as every page lists the classes
    out = []
    for identity in [*known, *sorted(set(by_identity) - set(known))]:
        found, group = vocabulary.identity(identity), by_identity.get(identity)
        if group is None and (found is None or found.hidden):
            continue
        n = 0 if group is None else len(group)
        subtypes = pd.Series(dtype=int) if group is None else group.subtype[group.subtype != ""].value_counts()
        names = [(s.id, s.name) for s in found.subtypes if not s.hidden or s.id in subtypes] if found else []
        names += [(s, s) for s in subtypes.keys() if s not in {i for i, _ in names}]
        out.append({"id": identity, "name": found.name if found else identity, "role": vocabulary.role(identity), "objects": n,
                    "subtypes": [{"id": i, "name": name, "objects": int(subtypes.get(i, 0))} for i, name in names],
                    "untyped": n - int(subtypes.sum())})
    return out


def stain_overview(project: Project, stain: str, labels: pd.DataFrame, tiles: pd.DataFrame) -> dict:
    fields = set(tiles.tile_id[tiles.stain == stain])
    mine = labels[labels.stain == stain]
    reviewed = store.reviewed_fields(labels, stain)
    vocabulary = vocab.of(project, stain)
    consensus = decisions.consensus(project, stain, labels, tiles)  # by candidate, as training counts
    objects = pool.all_objects(project, stain)
    prepared = pool.prepared(project, stain)
    on_store = candidate_store.cores(project, stain)
    if on_store and not tiles.empty:  # fields whose core has v2 candidates are prepared too
        prepared = prepared | set(tiles.tile_id[(tiles.stain == stain) & tiles.core_id.astype(str).isin(on_store)])
    stored = candidate_store.calls_at(project, stain, consensus, tiles) if on_store and len(consensus) else None
    matched = match(consensus, objects, stain, prepared, added.all_added(project, stain), stored) if len(consensus) \
        else consensus.assign(ai_value=[], ai_positive=[], proposed=[])
    judged = matched[matched.ai_positive.notna() & matched.positive.notna()]
    human, ai = judged.positive.astype(bool).to_numpy(), judged.ai_positive.astype(bool).to_numpy()
    multi = consensus[consensus.n >= 2]
    outlines = mine[mine.task == "outline"]
    return {
        "stain": stain,
        "fields": len(fields),
        "fields_prepared": len(prepared & fields),
        "fields_reviewed": len(set(reviewed) & fields),
        "fields_with_labels": len(set(consensus.tile_id) & fields),
        "objects_labelled": int(len(consensus)),
        "objects_added": int(matched.ai_value.isna().sum()),
        "values": {k: int(v) for k, v in pd.Series([vocabulary.display(i, s) if i else SPLIT for i, s in zip(consensus.value, consensus.subtype)],
                                                     dtype=object).value_counts().items()},
        "classes": class_counts(consensus, vocabulary),
        "outlines": int(outlines.drop_duplicates("id").shape[0]),
        "outlines_edited": int((~outlines.confirmed.astype(bool)).sum()),
        "people": sorted(set(mine.source)),
        "agreement": float((multi.agreement == 1).mean()) if len(multi) else None,
        "objects_with_several_labels": int(len(multi)),
        "ai": {"n": int(len(judged)), "agreement": float((human == ai).mean()) if len(judged) else None,
               "precision": float((human & ai).sum() / ai.sum()) if ai.sum() else None,
               "recall": float((human & ai).sum() / human.sum()) if human.sum() else None},
        "detection": detection(matched),
        "added": added.summary(project, stain, labels),
        "tags": {k: int(v) for k, v in mine[mine.task == "tag"].drop_duplicates(["tile_id", "value"]).value.value_counts().items()},
        # people against detection on the candidates they decided (the comparison the Cores view's Annotations layer shows)
        "vs_detection": compare.summary(project, stain, tiles) if on_store else None,
    }


def _saved_path(project: Project, stain: str) -> Path:
    from stainid import disk_cache  # noqa: PLC0415

    return disk_cache.path(project.root, "label_overview", f"{stain}.json")


def _saved(project: Project, stain: str, digest: str) -> dict | None:
    """The overview a server worked out before from the same inputs (it outlives restarts: a cold server answers at
    once when nothing changed since)."""
    from stainid import disk_cache  # noqa: PLC0415

    path = _saved_path(project, stain)
    try:
        found = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if found.get("digest") != digest:
        return None
    disk_cache.used(path)
    return found.get("overview")


def _save(project: Project, stain: str, digest: str, value: dict) -> None:
    from stainid import disk_cache  # noqa: PLC0415

    try:
        disk_cache.write(_saved_path(project, stain), json.dumps({"digest": digest, "overview": value}).encode())
    except (OSError, TypeError, ValueError):
        pass


def cached_stain_overview(project: Project, stain: str, root: Path, tiles: pd.DataFrame) -> dict:
    """`stain_overview`, worked out again only when a label is saved, a pool field is (re)built, detection writes this
    stain's candidates or masks, or an area is edited; kept in memory and in the project's cache (`label_overview`)."""
    key = (str(project.root), stain)
    stamp = (store.signature(root), pool.signature(project, stain), added.signature(project, stain), len(tiles), vocab.of(project, stain),
             cache.signature(candidate_store.folder(project, stain)), excluded_areas.signature(project, stain))
    kept = _CACHE.get(key)
    if kept is not None and kept[0] == stamp:
        return kept[1]
    with _MAKING_LOCK:
        lock = _MAKING.setdefault(key, threading.Lock())
    with lock:
        kept = _CACHE.get(key)
        if kept is not None and kept[0] == stamp:
            return kept[1]
        digest = hashlib.sha1(json.dumps([key, stamp], default=repr).encode()).hexdigest()
        value = _saved(project, stain, digest)
        if value is None:
            labels = frames.in_field_frame(excluded_areas.drop_excluded(project, store.current(root)), tiles, project.context_px)
            value = stain_overview(project, stain, labels, tiles)
            _save(project, stain, digest, value)
        _CACHE[key] = (stamp, value)
        return value
