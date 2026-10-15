"""Labelling sessions: what other people are asked to label, and in which order.

A session is `sessions/<name>/meta.json`; its labels go into the shared label store, tagged with the session name.
- `fields`: review whole fields. The AI's calls and outlines are pre-filled; people correct them, add what was missed,
  and press Finished, which records every call they left untouched as confirmed by them.
- `queue`: one object at a time, chosen by a rule (where the AI is unsure, where people or the AI disagree, objects
  with one label so far, or at random) from the fields whose candidate pool is prepared. The rule is applied anew for
  every item, so the queue follows new data and new labels.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from stainid.labels.tasks import positive
from stainid.tables import write_text_atomic

FOLDER = "sessions"
KINDS = ("fields", "queue")
FIELD_TOOLS = ("class", "point", "outline", "tag")
AI_VIEWS = ("full", "positions", "none")
RULES = {"uncertain": "where the AI is unsure (probability 0.3–0.7)", "disagree": "where people disagree with each other or with the AI",
         "second": "objects one other person has labelled", "random": "any candidate"}


def folder(root: Path, name: str) -> Path:
    return root / FOLDER / name


def create(root: Path, name: str, **meta) -> dict:
    target = folder(root, name)
    if target.exists():
        raise FileExistsError(target)
    if meta["kind"] not in KINDS or meta["ai_view"] not in AI_VIEWS or not set(meta["tools"]) <= set(FIELD_TOOLS) or not meta["tools"]:
        raise ValueError(f"kind must be one of {KINDS}, tools from {FIELD_TOOLS}, ai_view from {AI_VIEWS}")
    if meta["kind"] == "queue" and meta["rule"] not in RULES:
        raise ValueError(f"rule must be one of {list(RULES)}")
    target.mkdir(parents=True)
    full = {"name": name, **meta, "created": time.time(), "open": False}
    write_text_atomic(target / "meta.json", json.dumps(full, indent=1))
    return full


def meta(root: Path, name: str) -> dict:
    return json.loads((folder(root, name) / "meta.json").read_text())


def update(root: Path, name: str, **changes) -> dict:
    updated = {**meta(root, name), **changes}
    write_text_atomic(folder(root, name) / "meta.json", json.dumps(updated, indent=1))
    return updated


def remove(root: Path, name: str) -> Path:
    """Moves the session aside (into sessions/.deleted); its labels stay in the label store."""
    target = root / FOLDER / ".deleted" / f"{name}-{time.strftime('%Y%m%d-%H%M%S')}"
    target.parent.mkdir(parents=True, exist_ok=True)
    return Path(shutil.move(folder(root, name), target))


def names(root: Path) -> list[str]:
    return sorted(p.parent.name for p in (root / FOLDER).glob("*/meta.json"))


def pick_fields(tiles: pd.DataFrame, reviewed: set[str], n: int, seed: int) -> list[str]:
    """Fields nobody has reviewed yet, spread over TMAs and groups, in shuffled order (so the order reveals nothing)."""
    pool = tiles[~tiles.tile_id.isin(reviewed)].sample(frac=1, random_state=seed)
    buckets = [list(g.tile_id) for _, g in pool.groupby(["tma", pool.disease_group.fillna("")], sort=False)]
    picked: list[str] = []
    while len(picked) < n and any(buckets):
        for bucket in buckets:
            if bucket and len(picked) < n:
                picked.append(bucket.pop())
    return list(np.random.default_rng(seed).permutation(picked))


def taken(root: Path) -> dict[str, list[str]]:
    """The sessions each field belongs to."""
    out: dict[str, list[str]] = {}
    for name in names(root):
        for tile in meta(root, name).get("fields", []):
            out.setdefault(tile, []).append(name)
    return out


def add_fields(root: Path, name: str, tile_ids: list[str]) -> dict:
    """Add these fields to a session (ones it has already are left as they are)."""
    current = meta(root, name)
    have = set(current["fields"])
    return update(root, name, fields=current["fields"] + [t for t in dict.fromkeys(tile_ids) if t not in have])


def remove_fields(root: Path, name: str, tile_ids: list[str]) -> dict:
    """Take these fields out of a session. Labels already made on them stay in the label store."""
    current = meta(root, name)
    drop = set(tile_ids)
    return update(root, name, fields=[t for t in current["fields"] if t not in drop])


def labelled_by(labels: pd.DataFrame, stain: str) -> dict[str, set[str]]:
    """Who has saved any label of this stain on each field."""
    mine = labels[labels.stain == stain]
    return {tile: set(group.source) for tile, group in mine.groupby("tile_id")}


def field_states(session: dict, reviewed: dict[str, set[str]], labelled: dict[str, set[str]], prepared: set[str],
                 busy: dict[str, set[str]]) -> list[dict]:
    """Where each field of a whole-field session stands: `preparing` (its objects are still being found), `not_started`,
    `in_progress` (someone has labels on it or has it open, but fewer than `per_field` people finished it) or `done`;
    with who finished it, who is working on it, and who has it open now."""
    out = []
    for tile in session["fields"]:
        done_by = sorted(reviewed.get(tile, set()))
        working = sorted(labelled.get(tile, set()) - set(done_by))
        open_by = sorted(busy.get(tile, set()))
        if tile not in prepared:
            state = "preparing"
        elif len(done_by) >= session.get("per_field", 1):
            state = "done"
        elif done_by or working or open_by:
            state = "in_progress"
        else:
            state = "not_started"
        out.append({"tile_id": tile, "state": state, "done_by": done_by, "working": working, "open_by": open_by})
    return out


def next_field(session: dict, reviewed: dict[str, set[str]], annotator: str, busy: dict[str, set[str]]) -> str | None:
    """Fields that still need annotators first, then fields nobody else has open."""
    todo = [t for t in session["fields"] if annotator not in reviewed.get(t, set())]
    order = {t: i for i, t in enumerate(session["fields"])}

    def rank(tile: str) -> tuple:
        n = len(reviewed.get(tile, set()))
        return n >= session["per_field"], bool(busy.get(tile, set()) - {annotator}), n, order[tile]

    return min(todo, key=rank) if todo else None


def next_object(session: dict, objects: pd.DataFrame, labels: pd.DataFrame, annotator: str) -> dict | None:
    """The next object for this person under the session's rule: fewest labels first, in an order of their own."""
    stain = session["stain"]
    human = labels[(labels.stain == stain) & (labels.task == "class")]
    key = lambda f: f.tile_id + ":" + f.x.astype(float).round().astype(int).astype(str) + ":" + f.y.astype(float).round().astype(int).astype(str)  # noqa: E731
    objects = objects.assign(key=key(objects))
    by_object = human.assign(key=key(human)).groupby("key")
    sources = by_object.source.agg(set)
    values = by_object.value.agg(list)
    mine = {k for k, s in sources.items() if annotator in s}
    candidates = objects[~objects.key.isin(mine)]
    count = candidates.key.map(lambda k: len(sources.get(k, ())))
    rule = session["rule"]
    if rule == "uncertain":
        candidates = candidates[candidates.ai_probability.between(0.3, 0.7)]
    elif rule == "second":
        candidates = candidates[count.loc[candidates.index] == 1]
    elif rule == "disagree":
        def disputed(row) -> bool:
            votes = [positive(stain, v) for v in values.get(row.key, []) if v != "unsure"]
            return bool(votes) and (len(set(votes)) > 1 or votes[0] != positive(stain, row.ai_value))
        candidates = candidates[candidates.apply(disputed, axis=1)] if len(candidates) else candidates
    if candidates.empty:
        return None
    seed = int(hashlib.sha256(annotator.encode()).hexdigest()[:8], 16)
    order = np.random.default_rng(seed).permutation(len(candidates))
    ranked = candidates.iloc[order].assign(n=count.loc[candidates.index].iloc[order].to_numpy()).sort_values("n", kind="stable")
    row = ranked.iloc[0]
    return {"tile_id": row.tile_id, "x": float(row.x), "y": float(row.y), "radius_px": float(row.radius_px), "ai_value": row.ai_value,
            "ai_probability": float(row.ai_probability), "labels_so_far": int(row.n), "remaining": int(len(candidates))}
