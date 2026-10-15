"""Where labelling stands, per stain: coverage, what people said, how often people agree, and how the AI compares."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from stainid.labels import pool, store
from stainid.labels.tasks import positive
from stainid.project import Project

UNTYPED = {"plaque": "plaque (type not recorded)", "tau+ neuron": "tau+ neuron (type not recorded)"}
_CACHE: dict[tuple, tuple[tuple, dict]] = {}


def match(consensus: pd.DataFrame, objects: pd.DataFrame, stain: str) -> pd.DataFrame:
    """Each consensus label with the AI call of the pool candidate at its position (None where no candidate is near)."""
    by_tile = {t: g for t, g in objects.groupby("tile_id")}
    calls = []
    for row in consensus.itertuples():
        candidates = by_tile.get(row.tile_id)
        i = pool.nearest(candidates, row.x, row.y)
        calls.append(None if i is None else candidates.iloc[i].ai_value)
    return consensus.assign(ai_value=calls, ai_positive=[None if c is None else positive(stain, c) for c in calls])


def stain_overview(project: Project, stain: str, labels: pd.DataFrame, tiles: pd.DataFrame) -> dict:
    fields = set(tiles.tile_id[tiles.stain == stain])
    mine = labels[labels.stain == stain]
    reviewed = store.reviewed_fields(labels, stain)
    consensus = store.consensus(labels, stain)
    objects = pool.all_objects(project, stain)
    matched = match(consensus, objects, stain) if len(consensus) else consensus.assign(ai_value=[], ai_positive=[])
    judged = matched[matched.ai_positive.notna() & matched.positive.notna()]
    human, ai = judged.positive.astype(bool).to_numpy(), judged.ai_positive.astype(bool).to_numpy()
    multi = consensus[consensus.n >= 2]
    outlines = mine[mine.task == "outline"]
    prepared = pool.prepared(project, stain)
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
        "tags": {k: int(v) for k, v in mine[mine.task == "tag"].drop_duplicates(["tile_id", "value"]).value.value_counts().items()},
    }


def cached_stain_overview(project: Project, stain: str, root: Path, tiles: pd.DataFrame) -> dict:
    """`stain_overview`, worked out again only when a label is saved or a pool field is (re)built."""
    key = (str(project.root), stain)
    stamp = (store.signature(root), pool.signature(project, stain), len(tiles))
    if key not in _CACHE or _CACHE[key][0] != stamp:
        _CACHE[key] = (stamp, stain_overview(project, stain, store.current(root), tiles))
    return _CACHE[key][1]
