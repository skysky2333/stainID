"""The two annotation rounds over the v2 candidates: corrections only, every candidate shown, one shared decision store.

- Round 1, "Check the AI's calls" (a session of kind `queue`): one candidate at a time, accepted and rejected ones, with the
  AI's class, type, score and outline; people keep the call, change the class or type, remove it (not an object) or say
  they are unsure. The most informative come first (`rank`): near the decision threshold, then rejected but strong, then
  accepted but weak. A session's candidates are chosen when it is made (`choose`) and kept in `sessions/<name>/items.json`.
- Round 2, "Fix the field" (kind `fields`): whole fields with every candidate and the AI's outlines; people add what was
  missed, fix outlines and may change any call (recorded exactly like a round-1 decision on that candidate).

Both read and write the same decisions (`labels.decisions`): a candidate decided in either round, in any session, is done
for everyone once `labels_needed` people decided it. Candidates and fields handed to someone are held for them for a short
while (`sessions.Reservations`), renewed while their page is open, so people working at once get different ones.

A round session's meta holds `source: "candidates"`, `round` (1 or 2), `scope` (the tiles it draws from), `level` (how
generous: the lowest proposal level shown, `candidate_store.at_level`; None = every candidate) and `labels_needed`.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from stainid import cache
from stainid.labels import candidate_store, sessions
from stainid.labels.decisions import State
from stainid.project import Project
from stainid.tables import write_text_atomic

SOURCE = "candidates"
KIND = {1: "queue", 2: "fields"}
TITLE = {1: "Check the AI’s calls", 2: "Fix the field"}
RULE = "informative"
NEAR = 0.15  # how close to the decision threshold a score is "near" it
LEVELS = {"all": None, "0.5": 0.5, "0.75": 0.75, "1": 1.0, "1.5": 1.5}  # how generous: lowest proposal level shown (× the slide threshold)
DEFAULT_OBJECTS = 500
FIELD_HOLD_SECONDS = 90.0


def is_round(meta: dict) -> bool:
    return meta.get("source") == SOURCE


def round_of(meta: dict) -> int:
    return int(meta.get("round") or (1 if meta["kind"] == "queue" else 2))


def needed(meta: dict) -> int:
    return max(1, int(meta.get("labels_needed") or meta.get("per_field") or 1))


def level(meta: dict) -> float | None:
    value = meta.get("level")
    return None if value is None else float(value)


# ---------- ordering ----------

def decision_threshold(frame: pd.DataFrame) -> float:
    """The score above which the AI accepts: the store's `threshold` column when it has one, else the cut that best
    separates the model's accepted from its rejected candidates, else 0.5."""
    if "threshold" in frame and pd.to_numeric(frame.threshold, errors="coerce").notna().any():
        return float(pd.to_numeric(frame.threshold, errors="coerce").median())
    model = frame[candidate_store.by_model(frame) & frame.probability.notna()]
    if model.empty or model.decision.nunique() < 2:
        return 0.5
    p = model.probability.to_numpy(float)
    yes = (model.decision == candidate_store.ACCEPTED).to_numpy()
    cuts = np.unique(p)
    right = [((p >= c) == yes).sum() for c in cuts]
    return float(cuts[int(np.argmax(right))])


def rank(frame: pd.DataFrame, threshold: float | None = None) -> pd.DataFrame:
    """The candidates, most informative first, with their `tier`: 0 near the threshold (closest first), 1 rejected but
    strong (highest score first), 2 accepted but weak (lowest score first)."""
    if frame.empty:
        return frame.assign(tier=[])
    threshold = decision_threshold(frame) if threshold is None else threshold
    p = frame.probability.astype(float)
    accepted = frame.decision.map(candidate_store.accepted).astype(bool)
    near = (p - threshold).abs() <= NEAR
    tier = np.where(near.fillna(False), 0, np.where(~accepted, 1, 2))
    key = np.where(tier == 0, (p - threshold).abs(), np.where(tier == 1, -p.fillna(-1.0), p.fillna(1.0)))
    ordered = frame.assign(tier=tier, _key=key).sort_values(["tier", "_key", "candidate_id"], kind="stable")
    return ordered.drop(columns=["_key"])


# ---------- round 1: the objects of a session ----------

def items_path(root: Path, name: str) -> Path:
    return sessions.folder(root, name) / "items.json"


def scope_cores(tiles: pd.DataFrame, scope: list[str]) -> list[str]:
    return sorted(set(tiles[tiles.tile_id.isin(set(scope))].core_id.astype(str))) if not tiles.empty else []


def candidates_in(project: Project, stain: str, tiles: pd.DataFrame, scope: list[str], at: float | None) -> pd.DataFrame:
    """The candidates of these tiles (their own), at the session's level, with field-picture positions."""
    mine = tiles[(tiles.stain == stain) & tiles.tile_id.isin(set(scope))]
    found = candidate_store.for_cores(project, stain, scope_cores(mine, scope))
    return candidate_store.in_tiles(candidate_store.at_level(found, at), mine, project.context_px)


def choose(project: Project, stain: str, tiles: pd.DataFrame, scope: list[str], at: float | None, n: int, decided: State,
           needs: int = 1) -> list[str]:
    """The candidates a new round-1 session asks about: the `n` most informative of its tiles (all when n is 0) that are
    not decided already."""
    found = candidates_in(project, stain, tiles, scope, at)
    if found.empty:
        return []
    found = found[[not decided.done(c, needs) for c in found.candidate_id]]
    ordered = rank(found)
    return list(ordered.candidate_id.head(n) if n > 0 else ordered.candidate_id)


def save_items(root: Path, name: str, ids: list[str]) -> None:
    write_text_atomic(items_path(root, name), json.dumps({"candidates": ids}))


def load_items(root: Path, name: str) -> list[str]:
    target = items_path(root, name)
    return cache.memo(("round items", str(target)), [target], lambda: json.loads(target.read_text())["candidates"] if target.exists() else [])


def items(project: Project, meta: dict, tiles: pd.DataFrame) -> pd.DataFrame:
    """A round-1 session's candidates in its order (those still in the store), with field-picture positions."""
    root = project.output("reviews")
    ids = load_items(root, meta["name"])
    target = items_path(root, meta["name"])

    def make() -> pd.DataFrame:
        found = candidates_in(project, meta["stain"], tiles, meta.get("scope", []), None)
        if found.empty:
            return found.assign(order=[])
        order = {c: i for i, c in enumerate(ids)}
        mine = found[found.candidate_id.isin(order)]
        return mine.assign(order=mine.candidate_id.map(order)).sort_values("order").reset_index(drop=True)

    return cache.memo(("round items frame", str(target), meta["stain"]), [target, candidate_store.folder(project, meta["stain"])], make, folders=True)


def open_for(frame: pd.DataFrame, decided: State, person: str, needs: int, held: dict[str, set[str]]) -> pd.Series:
    """Which of these candidates this person may be handed: not answered by them, not decided, and not held by others
    who would decide it."""
    ids = frame.candidate_id
    answered = ids.map(lambda c: person in decided.answered.get(c, ()))
    taken = ids.map(lambda c: decided.count(c) + len(held.get(c, set()) - {person}))
    return ~answered & (taken < needs)


def serve(meta: dict, frame: pd.DataFrame, decided: State, person: str, held: dict[str, set[str]], count: int,
          ready: pd.Series | None = None) -> tuple[pd.DataFrame, int]:
    """The next `count` candidates for this person, and how many are left for them: those already handed to them first
    (still on their page), then those whose picture is made, in the session's order."""
    if frame.empty:
        return frame, 0
    free = open_for(frame, decided, person, needed(meta), held)
    left = frame[free].assign(ready=True if ready is None else ready[free].to_numpy(bool))
    left = left.assign(handed=left.candidate_id.map(lambda c: person in held.get(c, set())))
    ordered = left.sort_values(["handed", "ready", "order"], ascending=[False, False, True], kind="stable")
    return ordered.head(max(1, count)), int(len(left))


def gone(keys: list[str], decided: State, person: str, needs: int, held: dict[str, set[str]]) -> list[str]:
    """Of the candidates on someone's page, those that are no longer theirs to answer: decided by others meanwhile (or
    taken by others while their hold had run out)."""
    out = []
    for key in keys:
        others = held.get(key, set()) - {person}
        if person in decided.answered.get(key, ()) or decided.done(key, needs) or (others and decided.count(key) + len(others) >= needs):
            out.append(key)
    return out


def progress(meta: dict, frame: pd.DataFrame, decided: State) -> dict:
    """How far a round-1 session is: candidates decided (by anyone, in any round) of all, and by whom."""
    ids = list(frame.candidate_id) if len(frame) else []
    needs = needed(meta)
    done = [c for c in ids if decided.done(c, needs)]
    people: dict[str, int] = {}
    for c in ids:
        for person in decided.by(c):
            people[person] = people.get(person, 0) + 1
    return {"total": len(ids), "decided": len(done), "people": [{"person": p, "decided": n} for p, n in sorted(people.items(), key=lambda kv: -kv[1])]}


# ---------- round 2: fields ----------

def field_states(meta: dict, reviewed: dict[str, set[str]], working: dict[str, set[str]], held: dict[str, set[str]], ready: set[str]) -> list[dict]:
    """Where each field of a round-2 session stands (as `sessions.field_states`); `open_by` are the people holding it now."""
    return sessions.field_states({**meta, "per_field": needed(meta)}, reviewed, working, ready, held)


def next_field(meta: dict, reviewed: dict[str, set[str]], person: str, held: dict[str, set[str]], ready: set[str]) -> tuple[str | None, int]:
    """The next field for this person and how many fields others are on: one they hold already, else the first field
    that still needs people and that others are not on (fields others hold are not handed out twice)."""
    needs = needed(meta)
    todo = [t for t in meta["fields"] if t in ready and person not in reviewed.get(t, set()) and len(reviewed.get(t, set())) < needs]
    mine = [t for t in todo if person in held.get(t, set())]
    if mine:
        return mine[0], 0
    free = [t for t in todo if len(reviewed.get(t, set())) + len(held.get(t, set()) - {person}) < needs]
    return (free[0] if free else None), len(todo) - len(free)
