"""Where labelling stands, per stain: coverage, what people said, how often people agree, and how the AI compares.

How the AI compares is measured on every real object people labelled (`detection`), not only on the objects the
detection step proposed: a real object with no pool candidate at its position (within `pool.MATCH_PX`) on a prepared
field is one the AI never proposed, a missed object like one it proposed but rejected. The older figure, among proposed
objects only (`ai`), is kept for comparison.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from stainid.labels import added, frames, pool, store
from stainid.labels.tasks import positive
from stainid.project import Project

UNTYPED = {"plaque": "plaque (type not recorded)", "tau+ neuron": "tau+ neuron (type not recorded)"}
_CACHE: dict[tuple, tuple[tuple, dict]] = {}


def match(consensus: pd.DataFrame, objects: pd.DataFrame, stain: str, prepared: set[str] | None = None,
          extra: pd.DataFrame | None = None) -> pd.DataFrame:
    """Each consensus label with the AI call of the pool candidate at its position, and whether the detection step
    proposed the object at all: True (a candidate is there), False (no candidate, on a field in `prepared`: the AI never
    proposed it) or None (no candidate on a field that is not prepared, so it cannot tell). `extra`: measured added
    objects (`labels.added`); one that turned out to be a candidate counts as that candidate."""
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


def stain_overview(project: Project, stain: str, labels: pd.DataFrame, tiles: pd.DataFrame) -> dict:
    fields = set(tiles.tile_id[tiles.stain == stain])
    mine = labels[labels.stain == stain]
    reviewed = store.reviewed_fields(labels, stain)
    consensus = store.consensus(labels, stain)
    objects = pool.all_objects(project, stain)
    prepared = pool.prepared(project, stain)
    matched = match(consensus, objects, stain, prepared, added.all_added(project, stain)) if len(consensus) \
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
        "values": {UNTYPED.get(k, k): int(v) for k, v in consensus.value.replace("", "people disagree").value_counts().items()},
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
    }


def cached_stain_overview(project: Project, stain: str, root: Path, tiles: pd.DataFrame) -> dict:
    """`stain_overview`, worked out again only when a label is saved or a pool field is (re)built."""
    key = (str(project.root), stain)
    stamp = (store.signature(root), pool.signature(project, stain), added.signature(project, stain), len(tiles))
    if key not in _CACHE or _CACHE[key][0] != stamp:
        _CACHE[key] = (stamp, stain_overview(project, stain, frames.in_field_frame(store.current(root), tiles, project.context_px), tiles))
    return _CACHE[key][1]
