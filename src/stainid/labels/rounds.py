"""The two annotation rounds over the v2 candidates: corrections only, every candidate shown, one shared decision store.

- Round 1, "Check the AI's calls" (a session of kind `queue`): one candidate at a time, accepted and rejected ones, with the
  AI's class, type, score and outline; people keep the call, change the class or type, remove it (not an object) or say
  they are unsure. The most informative come first (`rank`): near the decision threshold, then rejected but strong, then
  accepted but weak. A session's candidates are chosen when it is made (`choose`) and kept in `sessions/<name>/items.json`.
- Round 2, "Fix the field" (kind `fields`): whole fields with every candidate and the AI's outlines; people add what was
  missed, fix outlines and may change any call (recorded exactly like a round-1 decision on that candidate).

Both write the same decisions (`labels.decisions`, keyed by candidate id). The unit people claim is a tile (field), in
both rounds: opening one holds it for that person (`sessions.Reservations`, renewed while their page is open), so the
others are handed other tiles meanwhile; once someone saves it as finished (a `done` label: `value` CALLS in round 1, ""
in round 2) it is done for that round, and once `labels_needed` people (default 1) finished it nobody else is handed it.
Round 1 walks the objects of one tile after the other. Nothing reaches the label log until the person saves the tile:
all its answers at once (`routes.rounds.save_tile`).

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
TITLE = {1: "Check detection’s calls", 2: "Fix the field"}
RULE = "informative"
NEAR = 0.15  # how close to the decision threshold a score is "near" it
LEVELS = {"all": None, "0.5": 0.5, "0.75": 0.75, "1": 1.0, "1.5": 1.5}  # how generous: lowest proposal level shown (× the slide threshold)
DEFAULT_OBJECTS = 500
TILE_HOLD_SECONDS = 90.0  # a tile stays someone's this long after their page last asked (every few seconds while open)
CALLS = "calls"  # the `value` of the `done` label that finishes a tile of round 1 (round 2's has none)


def is_round(meta: dict) -> bool:
    return meta.get("source") == SOURCE


def round_of(meta: dict) -> int:
    return int(meta.get("round") or (1 if meta["kind"] == "queue" else 2))


def needed(meta: dict) -> int:
    return max(1, int(meta.get("labels_needed") or meta.get("per_field") or 1))


def level(meta: dict) -> float | None:
    value = meta.get("level")
    return None if value is None else float(value)


def done_kind(meta: dict) -> str:
    """The `value` of the `done` label that finishes a tile of this round (`store.reviewed_fields`)."""
    return CALLS if round_of(meta) == 1 else ""


def done_id(meta: dict, tile_id: str) -> str:
    return f"done:{CALLS}:{tile_id}" if round_of(meta) == 1 else f"done:{tile_id}"


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
    found = candidate_store.for_cores(project, stain, scope_cores(mine, scope), candidate_store.LIGHT)  # no outlines: `with_shapes`
    return candidate_store.in_tiles(candidate_store.at_level(found, at), mine, project.context_px)


SHAPES = ("outline", "sam_outline", "sam_masks", "sam_scores", "sam_default")


def with_shapes(project: Project, stain: str, frame: pd.DataFrame, tile: dict) -> pd.DataFrame:
    """These candidates of one tile (`candidates_in`) with their outlines and masks (core pixels), read for that tile only."""
    if frame.empty:
        return frame
    rows = candidate_store.field_rows(project, stain, tile, project.context_px)
    if rows is None or rows.empty:
        return frame
    rows = rows.drop_duplicates("candidate_id").set_index("candidate_id")
    at = [rows.index.get_loc(c) if c in rows.index else -1 for c in frame.candidate_id]
    return frame.assign(**{c: [rows[c].iloc[i] if i >= 0 else None for i in at] for c in SHAPES if c in rows})


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


def tiles_of(frame: pd.DataFrame) -> list[str]:
    """The tiles of a round-1 session's candidates, in the order its most informative candidate on each comes."""
    return list(dict.fromkeys(frame.tile_id)) if len(frame) else []


def open_items(frame: pd.DataFrame, finished: dict[str, set[str]], needs: int) -> pd.DataFrame:
    """A round-1 session's candidates on tiles that still need people (whose pictures are worth making)."""
    if frame.empty:
        return frame
    return frame[[len(finished.get(t, set())) < needs for t in frame.tile_id]]


# ---------- tiles: who is handed which, and how far a session is (both rounds) ----------

def next_tile(tiles: list[str], finished: dict[str, set[str]], person: str, held: dict[str, set[str]], needs: int,
              ready: set[str] | None = None, skip: set[str] | None = None, prefer: set[str] | None = None) -> tuple[str | None, int]:
    """The next tile for this person and how many tiles still to do are others' now: one they hold already (their work
    in progress), else the first tile (of those `prefer`red first, e.g. with their pictures made) that still needs people
    and that others are not on. A tile someone else is editing, or that enough people finished, is not handed out
    again; `skip` are tiles this person put aside for now."""
    skip = skip or set()
    todo = [t for t in tiles if (ready is None or t in ready) and person not in finished.get(t, set()) and len(finished.get(t, set())) < needs]
    mine = [t for t in todo if person in held.get(t, set()) and t not in skip]
    if mine:
        return mine[0], 0
    free = [t for t in todo if len(finished.get(t, set())) + len(held.get(t, set()) - {person}) < needs]
    pick = [t for t in free if t not in skip]
    if prefer is not None:
        pick = sorted(pick, key=lambda t: t not in prefer)  # stable: the session's order among each
    return (pick[0] if pick else None), len(todo) - len(free)


def progress(tiles: list[str], finished: dict[str, set[str]], held: dict[str, set[str]], needs: int) -> dict:
    """How far a round session is, in tiles (live): done (enough people finished it), in progress (someone has it open
    or finished it and more people are needed), left (nobody on it yet), and how many tiles each person finished."""
    done = [t for t in tiles if len(finished.get(t, set())) >= needs]
    working = [t for t in tiles if t not in set(done) and (finished.get(t) or held.get(t))]
    people: dict[str, int] = {}
    for t in tiles:
        for person in finished.get(t, set()):
            people[person] = people.get(person, 0) + 1
    return {"total": len(tiles), "decided": len(done), "open_now": len(working), "left": len(tiles) - len(done) - len(working),
            "people": [{"person": p, "decided": n} for p, n in sorted(people.items(), key=lambda kv: (-kv[1], kv[0]))]}


# ---------- round 2: fields ----------

ALL_FIELDS = -1  # "how many fields": every remaining one


def remaining_fields(fields: list[str], finished: set[str], assigned: set[str]) -> list[str]:
    """The fields a new "Fix the field" session can draw from: those not annotated yet (nobody finished them) and not
    assigned to an open session, in the given order."""
    return [t for t in dict.fromkeys(fields) if t not in finished and t not in assigned]


def draw_fields(remaining: list[str], n: int, seed: int, first: set[str] | None = None) -> list[str]:
    """n of the remaining fields drawn at random, without repeats (n < 0, or more than there are: all of them), in the
    drawn order; the same seed gives the same draw. `first`: fields drawn before any other (those whose core image's
    bad areas a person checked), at random among themselves, then the rest at random."""
    rng = np.random.default_rng(seed)
    if first is None:
        if n < 0 or n >= len(remaining):
            return [remaining[i] for i in rng.permutation(len(remaining))]
        return [remaining[i] for i in rng.choice(len(remaining), size=n, replace=False)]
    pools = [[t for t in remaining if t in first], [t for t in remaining if t not in first]]
    drawn = [pool[i] for pool in pools for i in rng.permutation(len(pool))]
    return drawn if n < 0 else drawn[:n]


def from_checked(fields: list[str], tiles: pd.DataFrame, checked: dict[tuple[str, str], set[str]]) -> set[str]:
    """The fields whose core image's bad areas someone checked ("Mark bad areas", `status.areas_checked`)."""
    if tiles.empty or not checked:
        return set()
    rows = tiles[tiles.tile_id.isin(set(fields))].drop_duplicates("tile_id")
    return {t for t, c, s in zip(rows.tile_id, rows.core_id.astype(str), rows.stain.astype(str)) if checked.get((c, s))}

def field_states(meta: dict, reviewed: dict[str, set[str]], working: dict[str, set[str]], held: dict[str, set[str]], ready: set[str]) -> list[dict]:
    """Where each field of a round-2 session stands (as `sessions.field_states`); `open_by` are the people holding it now."""
    return sessions.field_states({**meta, "per_field": needed(meta)}, reviewed, working, ready, held)
