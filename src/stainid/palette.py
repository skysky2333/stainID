"""The one palette of every colour the pages draw with (`palette.json`): the label classes by colour slot
(`Vocabulary.colour_slots`), the inactive / unsure / unknown states, the bad-area types, the layers drawn over the
tissue and the markers, all distinct, also from the white highlight and dark edge of drawn shapes (tests/test_review_consistency.py). The web app imports the same file (webapp/src/palette.ts),
the server draws its pictures with it and serves it at /api/palette."""
from __future__ import annotations

import json
from functools import cache
from pathlib import Path

PATH = Path(__file__).with_name("palette.json")


@cache
def palette() -> dict:
    """The palette, without its note."""
    return {k: v for k, v in json.loads(PATH.read_text()).items() if not k.startswith("_")}


def rgb(hexcode: str) -> tuple[int, int, int]:
    """(r, g, b) of "#rrggbb"."""
    return int(hexcode[1:3], 16), int(hexcode[3:5], 16), int(hexcode[5:7], 16)


def area(kind: str) -> tuple[int, int, int]:
    """A bad-area type's colour ("1"–"4", "taken_out", "edge_guard", "not_analysed")."""
    return rgb(palette()["areas"][kind])


def overlay(name: str) -> tuple[int, int, int]:
    return rgb(palette()["overlays"][name])
