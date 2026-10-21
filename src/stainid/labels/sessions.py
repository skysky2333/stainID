"""Labelling sessions: what other people are asked to label, and in which order.

A session is `sessions/<name>/meta.json`; its labels go into the shared label store, tagged with the session name.
- `fields`: review whole fields. The AI's calls and outlines are pre-filled; people correct them, add what was missed,
  and press Finished, which records every call they left untouched as confirmed by them.
- `queue`: one object at a time, chosen by a rule (where the AI is unsure, where people or the AI disagree, objects
  with one label so far, or at random) from the fields whose candidate pool is prepared. The page asks for the next
  few objects at a time; those are held for that person for a while (`Reservations`), so people labelling at once get
  different objects. The rule is applied anew on every ask, so the queue follows new data and new labels.
- `outline`: outline every object of whole fields, to measure segmentation (`labels.outline`); no longer made, kept readable and scored.

A field is one square tile of a core (about 560 µm, 2048 pixels), shown to people as one picture.

Every label belongs to a session. Labels with no session (the label sets imported from before labelling sessions
existed, `source` "import: <set>") form one read-only pseudo-session, `REFERENCE`. Models can be trained on any
combination of sessions (`in_sessions`).
"""
from __future__ import annotations

import hashlib
import json
import shutil
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd

from stainid import practice
from stainid.labels import store
from stainid.labels import vocabulary as vocab
from stainid.labels.vocabulary import Vocabulary
from stainid.tables import write_text_atomic

FOLDER = "sessions"
KINDS = ("fields", "queue", "outline", "areas", "ai_review")  # areas: "Mark bad areas" (`labels.areas`); ai_review: read-only AI reviewer labels (`review.ai_ingest`)
READ_ONLY_KINDS = ("ai_review",)  # label sets nobody labels in: no fields to prepare, no join key, not in anyone's sets
KIND_TITLES = {"fields": "check whole fields", "queue": "one object at a time", "outline": "outline every object (older segmentation check)",
               "areas": "Mark bad areas", "ai_review": "AI review · read only"}  # the Labels page shows these (`kind_title`)
FIELD_TOOLS = ("class", "point", "outline", "tag", "area")


def label_set_only(meta: dict) -> bool:
    """A read-only label set (an AI review round): its labels are looked at and trained on, nobody labels in it."""
    return meta.get("kind") in READ_ONLY_KINDS


def kind_title(meta: dict) -> str:
    """What kind of session it is, in words; a kind this release does not know reads as its name, never as nothing."""
    kind = str(meta.get("kind") or "")
    return KIND_TITLES.get(kind) or kind.replace("_", " ") or "labelling session"
AI_VIEWS = ("full", "positions", "none")
RULES = {"uncertain": "where detection is unsure (probability 0.3–0.7)", "disagree": "where people disagree with each other or with detection",
         "second": "objects one other person has labelled", "random": "any candidate",
         "informative": "the most informative first: near detection’s threshold, then rejected but strong, then accepted but weak"}


REFERENCE = "reference"
REFERENCE_TITLE = "AI reference labels"


def session_of(labels: pd.DataFrame) -> pd.Series:
    """The session of each label: its own, or `REFERENCE` for labels made outside sessions."""
    named = labels.session.fillna("").astype(str) if "session" in labels else pd.Series("", index=labels.index)
    return named.where(named != "", REFERENCE)


def in_sessions(labels: pd.DataFrame, chosen: list[str] | None) -> pd.DataFrame:
    """Only the labels of these sessions (all labels when `chosen` is empty or None)."""
    return labels[session_of(labels).isin(set(chosen))] if chosen else labels


def label_sets(labels: pd.DataFrame, stain: str, titles: dict[str, str], vocabulary: Vocabulary | None = None) -> list[dict]:
    """Every session with object labels of this stain: how many labels of each class (identity and subtype, as
    `Vocabulary.display` names them), from how many objects, fields and people. `titles` names the real sessions; the
    reference labels come last."""
    vocabulary = vocabulary or vocab.defaults(stain)
    mine = store.classes(labels, stain, vocabulary)
    groups = session_of(mine)
    out = []
    for name, group in mine.groupby(groups):
        objects = group.drop_duplicates(["tile_id", "x", "y"])
        shown = pd.Series([vocabulary.display(i, s) for i, s in zip(group.identity, group.subtype)], dtype=object)
        out.append({"name": name, "title": REFERENCE_TITLE if name == REFERENCE else titles.get(name, name), "reference": name == REFERENCE,
                    "labels": int(len(group)), "objects": int(len(objects)), "fields": int(group.tile_id.nunique()),
                    "classes": {str(k): int(v) for k, v in shown.value_counts().items()}, "people": sorted(set(group.source))})
    return sorted(out, key=lambda s: (s["reference"], s["title"].lower()))


def folder(root: Path, name: str) -> Path:
    return root / FOLDER / name


ON_CREATE: list = []  # called with (root, name) when a session is made: the app gives it its join key (`api.routes.accounts`)


def create(root: Path, name: str, **meta) -> dict:
    practice.guard("the sessions")
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
    for hook in ON_CREATE:
        try:
            hook(root, name)
        except Exception:  # noqa: BLE001 - a session outside the app (an import on the command line) gets its key when first shown
            pass
    return full


def meta(root: Path, name: str) -> dict:
    return json.loads((folder(root, name) / "meta.json").read_text())


def update(root: Path, name: str, **changes) -> dict:
    practice.guard("the sessions")
    updated = {**meta(root, name), **changes}
    write_text_atomic(folder(root, name) / "meta.json", json.dumps(updated, indent=1))
    return updated


def remove(root: Path, name: str) -> Path:
    """Moves the session aside (into sessions/.deleted); its labels stay in the label store."""
    practice.guard("the sessions")
    target = root / FOLDER / ".deleted" / f"{name}-{time.strftime('%Y%m%d-%H%M%S')}"
    target.parent.mkdir(parents=True, exist_ok=True)
    return Path(shutil.move(folder(root, name), target))


def names(root: Path) -> list[str]:
    return sorted(p.parent.name for p in (root / FOLDER).glob("*/meta.json"))


def pick_fields(tiles: pd.DataFrame, reviewed: set[str], n: int, seed: int) -> list[str]:
    """Fields nobody has reviewed yet, spread over TMAs and groups, in shuffled order (so the order reveals nothing)."""
    pool = tiles[~tiles.tile_id.isin(reviewed)].sample(frac=1, random_state=seed)
    n = len(pool) if n < 0 else n  # -1: every one
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
    """Take these fields out of a session, and out of its `scope` (a round session's: the fields it draws on, which
    adding fields keeps equal to them). Labels already made on them stay in the label store."""
    current = meta(root, name)
    drop = set(tile_ids)
    changes = {"fields": [t for t in current["fields"] if t not in drop]}
    if current.get("scope") is not None:
        changes["scope"] = [t for t in current["scope"] if t not in drop]
    return update(root, name, **changes)


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


RESERVE_SECONDS = 120.0  # how long objects handed to someone (the next few on their screen) are kept from others


def object_keys(frame: pd.DataFrame) -> pd.Series:
    """`tile:x:y` (rounded pixels) of each row: one key per object, shared by the pool and the labels on it."""
    rounded = lambda v: v.astype(float).round().astype(int).astype(str)  # noqa: E731
    return frame.tile_id.astype(str) + ":" + rounded(frame.x) + ":" + rounded(frame.y)


class Reservations:
    """Objects of a queue session handed to someone and not yet answered, so others get other objects meanwhile. A hold
    lasts `seconds` and is renewed whenever the same person is handed the object again; answering releases it."""

    def __init__(self, seconds: float = RESERVE_SECONDS, clock=time.monotonic):
        self.seconds, self.clock = seconds, clock
        self._held: dict[tuple, dict[str, dict[str, tuple[float, int]]]] = {}  # session -> object key -> person -> (until, place)
        self._handed = 0  # the place of each object in the order it was handed out
        self._lock = threading.Lock()

    def held(self, session: tuple) -> dict[str, set[str]]:
        """Who holds each object of this session now."""
        now = self.clock()
        with self._lock:
            objects = self._held.get(session, {})
            for key in [k for k, people in objects.items() if not any(until > now for until, _ in people.values())]:
                del objects[key]
            return {k: {p for p, (until, _) in people.items() if until > now} for k, people in objects.items()}

    def order(self, session: tuple, person: str) -> dict[str, int]:
        """The objects this person holds, by the order they were handed to them (so they stay in that order on their
        screen whatever changes meanwhile)."""
        now = self.clock()
        with self._lock:
            return {k: people[person][1] for k, people in self._held.get(session, {}).items() if person in people and people[person][0] > now}

    def hold(self, session: tuple, person: str, keys: list[str]) -> None:
        practice.guard("the holds")
        now = self.clock()
        until = now + self.seconds
        with self._lock:
            objects = self._held.setdefault(session, {})
            for key in keys:
                people = objects.setdefault(key, {})
                before = people.get(person)
                if before is not None and before[0] > now:
                    people[person] = (until, before[1])  # renewed, in its place
                else:
                    self._handed += 1
                    people[person] = (until, self._handed)

    def release(self, session: tuple, person: str, keys: list[str]) -> None:
        practice.guard("the holds")
        with self._lock:
            objects = self._held.get(session, {})
            for key in keys:
                objects.get(key, {}).pop(person, None)


def _rule_filter(session: dict, candidates: pd.DataFrame, count: pd.Series, values: pd.Series, vocabulary: Vocabulary) -> pd.DataFrame:
    """The candidates the session's rule asks about (`count`: labels so far, `values`: what people said, by key)."""
    rule = session["rule"]
    if rule == "uncertain":
        return candidates[candidates.ai_probability.between(0.3, 0.7)]
    if rule == "second":
        return candidates[count.loc[candidates.index] == 1]
    if rule == "disagree" and len(candidates):
        def disputed(row) -> bool:  # people disagree on the identity's role, or with the AI (types do not count)
            votes = [vocabulary.positive(v) for v in values.get(row.key, []) if not vocabulary.skipped(v)]
            return bool(votes) and (len(set(votes)) > 1 or votes[0] != vocabulary.positive(row.ai_value))
        return candidates[candidates.apply(disputed, axis=1)]
    return candidates


def _labels_by_object(session: dict, labels: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Who labelled each object (by key) and what they said."""
    human = labels[(labels.stain == session["stain"]) & (labels.task == "class")]
    by_object = human.assign(key=object_keys(human)).groupby("key")
    return by_object.source.agg(set), by_object.value.agg(list)


def may_serve(session: dict, objects: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Every object this queue could hand to someone now or after more labels come in (the pictures to make ahead):
    all candidates of the rule `random`, the AI-unsure ones of `uncertain`, and the labelled ones of `second` and
    `disagree` (only objects someone labelled can be asked about a second time or be disputed)."""
    sources, _ = _labels_by_object(session, labels)
    objects = objects.assign(key=object_keys(objects))
    if session["rule"] == "uncertain":
        return objects[objects.ai_probability.between(0.3, 0.7)]
    if session["rule"] in ("second", "disagree"):
        return objects[objects.key.isin(set(sources.index))]
    return objects


def next_objects(session: dict, objects: pd.DataFrame, labels: pd.DataFrame, annotator: str, count: int = 1,
                 ready: pd.Series | None = None, held: dict[str, set[str]] | None = None,
                 vocabulary: Vocabulary | None = None, handed: dict[str, int] | None = None) -> tuple[list[dict], int]:
    """The next `count` objects for this person under the session's rule, and how many the rule leaves them in all.

    Order: objects already handed to this person (still on their screen), in the order they were handed (`handed`:
    object key -> place, from `Reservations.order`); then objects nobody else holds (`held`: object key -> people),
    so two people at once get different objects while there are any; among those, objects whose picture is `ready`
    (a boolean per row of `objects`) first, then fewest labels, counting each person holding one as a label more;
    ties in an order of the person's own. Pictures being made meanwhile change only what comes after the handed ones."""
    held, handed = held or {}, handed or {}
    sources, values = _labels_by_object(session, labels)
    objects = objects.assign(key=object_keys(objects), ready=True if ready is None else ready.to_numpy(bool))
    mine = {k for k, s in sources.items() if annotator in s}
    candidates = objects[~objects.key.isin(mine)]
    count_so_far = candidates.key.map(lambda k: len(sources.get(k, ())))
    candidates = _rule_filter(session, candidates, count_so_far, values, vocabulary or vocab.defaults(session["stain"]))
    if candidates.empty:
        return [], 0
    seed = int(hashlib.sha256(annotator.encode()).hexdigest()[:8], 16)
    order = np.random.default_rng(seed).permutation(len(candidates))
    ranked = candidates.iloc[order].assign(n=count_so_far.loc[candidates.index].iloc[order].to_numpy())
    holders = ranked.key.map(lambda k: held.get(k, set()))
    ranked = ranked.assign(handed=holders.map(lambda h: annotator in h), others=holders.map(lambda h: len(h - {annotator})))
    ranked = ranked.assign(place=ranked.key.map(lambda k: handed.get(k, 0)).where(ranked.handed, 0), taken=ranked.others > 0,
                           busy=ranked.n + ranked.others)
    ranked = ranked.sort_values(["handed", "place", "taken", "ready", "busy"], ascending=[False, True, True, False, True], kind="stable")
    out = [{"tile_id": row.tile_id, "x": float(row.x), "y": float(row.y), "radius_px": float(row.radius_px), "ai_value": row.ai_value,
            "ai_probability": float(row.ai_probability), "labels_so_far": int(row.n), "remaining": int(len(candidates)), "key": row.key}
           for row in ranked.head(max(1, count)).itertuples()]
    return out, int(len(candidates))
