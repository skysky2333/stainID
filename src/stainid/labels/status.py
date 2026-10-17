"""Where review stands, tile by tile and core by core: the one reader the Cores view's minimap, the Labels page and
the round pages use. Everything comes from the label log (`labels.store`): a tile is reviewed once someone finished it
in an object round (a `done` label, `store.reviewed_fields`), in progress while someone holds it (the rounds' claims) or
has labelled on it without finishing, else not annotated; a core image's areas are checked once someone saved it in
"Mark bad areas" (a `done` label of value `AREAS` on the core id).
"""
from __future__ import annotations

import pandas as pd

from stainid.labels import store

AREAS = "areas"  # the `value` of the `done` label of a core image checked for bad areas (`tile_id` the core id)
CALLS = "calls"  # labels.rounds.CALLS: a tile finished in "Check detection's calls"
REVIEWED, IN_PROGRESS, NOT_ANNOTATED, NOT_ANALYSED = "reviewed", "in_progress", "not_annotated", "not_analysed"
STATES = (REVIEWED, IN_PROGRESS, NOT_ANNOTATED, NOT_ANALYSED)
OBJECT_TASKS = ("class", "outline", "tag")


def reviewed_tiles(labels: pd.DataFrame, stain: str) -> dict[str, set[str]]:
    """Tiles someone finished in an object round (either round), and who."""
    out: dict[str, set[str]] = {}
    for kind in ("", CALLS):
        for tile, people in store.reviewed_fields(labels, stain, kind).items():
            out.setdefault(tile, set()).update(people)
    return out


def tile_states(labels: pd.DataFrame, stain: str, tile_ids, held: dict[str, set[str]] | None = None) -> dict[str, dict]:
    """Each tile's object-review state: {state, by (who finished it), working (who holds it or labelled on it)}."""
    held = held or {}
    done = reviewed_tiles(labels, stain)
    mine = labels[(labels.stain == stain) & labels.task.isin(OBJECT_TASKS)] if not labels.empty else labels
    touched = {t: set(g.source) for t, g in mine.groupby("tile_id")} if len(mine) else {}
    out = {}
    for tile in tile_ids:
        by = sorted(done.get(tile, set()))
        working = sorted((held.get(tile, set()) | touched.get(tile, set())) - set(by))
        state = REVIEWED if by else IN_PROGRESS if working else NOT_ANNOTATED
        out[tile] = {"state": state, "by": by, "working": working}
    return out


def areas_checked(labels: pd.DataFrame, stain: str | None = None) -> dict[tuple[str, str], set[str]]:
    """(core_id, stain) -> who checked that core image's bad areas."""
    if labels.empty or "task" not in labels:
        return {}
    done = labels[(labels.task == "done") & (labels.value.fillna("").astype(str) == AREAS)]
    if stain is not None:
        done = done[done.stain == stain]
    return {(str(core), str(s)): set(group.source) for (core, s), group in done.groupby(["tile_id", "stain"])}


def counts(states: dict[str, dict]) -> dict[str, int]:
    found = {s: 0 for s in STATES}
    for value in states.values():
        found[value["state"]] += 1
    return found
