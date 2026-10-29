"""The order a session's picker shows its fields (or cores) in, which "Next" follows: a person who finished a field (Save
and next, Finish field, Skip for now) gets the next one after it in the order they opened it from.

- "natural" (the picker's Natural order): by TMA, then core (both in natural order: "LIP-2" before "LIP-10", "A-2"
  before "A-10"), then the tile's place in the core (top to bottom, left to right);
- "confidence" (the picker's By confidence list): least confident first (the highest average closeness of the field's
  objects to the acceptance threshold), fields without objects last, the session's order among equals.

`rotated` starts the order after the field the person is on, so "Next" goes on from there and comes round to the start.
Plain functions: the field page (`routes.rounds.next_field`) and the Grey / white matter cores (`routes.matter`) use
them, and the pickers (`webapp/src/labels/sessionfields.tsx`, `matter.tsx`) sort the same way.
"""
from __future__ import annotations

import re

NATURAL, CONFIDENCE = "natural", "confidence"
ORDERS = (NATURAL, CONFIDENCE)
OTHER_GROUP = "Other fields"  # the picker's group of fields without a TMA


def natural_key(text) -> tuple:
    """Text sorted as people read it: the numbers in it as numbers, letters without case."""
    parts = re.split(r"(\d+)", str(text or ""))
    return tuple((0, int(p), "") if p.isdigit() else (1, 0, p.casefold()) for p in parts if p != "")


def natural(places: list[dict]) -> list[str]:
    """Field ids in the Natural order: each place {tile_id, tma_name or tma, core or core_id, box [x, y, w, h] or None},
    in the session's order."""
    def key(item: tuple[int, dict]) -> tuple:
        i, f = item
        box = f.get("box") or None
        return (natural_key(f.get("tma_name") or f.get("tma") or OTHER_GROUP), natural_key(f.get("core") or f.get("core_id") or "—"),
                (0, box[1], box[0]) if box else (1, 0, 0), i)

    return [f["tile_id"] for _, f in sorted(enumerate(places), key=key)]


def by_confidence(fields: list[str], scores: dict[str, dict]) -> list[str]:
    """Field ids least confident first (`scores`: tile -> {mean, objects}, as `routes.annotate.field_confidence`)."""
    def score(tile: str) -> float:
        s = scores.get(tile) or {}
        return float(s["mean"]) if s.get("objects") and s.get("mean") is not None else -1.0

    return [t for _, t in sorted(enumerate(fields), key=lambda item: (-score(item[1]), item[0]))]


def rotated(order: list[str], after: str | None) -> list[str]:
    """The order starting just after `after` (and coming round to it last); as it is when `after` is not in it."""
    if not after or after not in order:
        return list(order)
    at = order.index(after)
    return order[at + 1:] + order[:at + 1]


def cores_natural(rows: list[dict]) -> list[str]:
    """Core ids in the Grey / white matter picker's order: by TMA (or donor), then core, both natural; each row
    {core_id, tma, core}."""
    return [r["core_id"] for _, r in sorted(enumerate(rows), key=lambda item: (natural_key(item[1].get("tma")), natural_key(item[1].get("core") or item[1]["core_id"]), item[0]))]
