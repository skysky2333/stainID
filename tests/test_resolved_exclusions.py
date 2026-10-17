"""The resolved excluded areas (`stainid.exclusions`): detection's areas with people's edits applied in time order, the
one layer every view, the analysis, training and scoring read."""
from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from stainid import cache, exclusions
from stainid.labels import exclusions as marks
from stainid.labels import store
from stainid.project import load_project
from stainid.v2.proposals import BITS, MASK_SCALE

CORE, STAIN, TILE = "LIP-1_B-2", "AT8", "LIP-1_B-2_AT8_S01"


@pytest.fixture()
def project(synthetic_project, monkeypatch):
    monkeypatch.chdir(synthetic_project)
    cache.forget()
    exclusions.forget()
    found = load_project(synthetic_project)
    yield found
    cache.forget()
    exclusions.forget()


def write_mask(project, stain: str = STAIN) -> np.ndarray:
    """150 × 150 cells (the 1200-px core): analysed tissue everywhere, a fold over rows 0–19, a linear artifact on row
    100, no tissue in the bottom-right corner."""
    bits = np.full((150, 150), BITS["tile"] | BITS["tissue"], np.uint8)
    bits[0:20, :] |= BITS["fold"]
    bits[100, :] |= BITS["linear"]
    bits[140:, 140:] = BITS["tile"]
    target = exclusions.mask_path(project, CORE, stain)
    target.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(target), bits)
    return bits


def square(x0, y0, x1, y1):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def edit(project, source, rings, kinds, when, tile_id=TILE, ident=None, holes=None):
    """An `exclude` label as the field tool saves it (rings in field px of `tile_id`)."""
    store.append(project.output("reviews"), [{"id": ident or f"exclude:{tile_id}:{source}:{when}", "source": source, "stain": STAIN,
                                              "task": "exclude", "tile_id": tile_id, "x": None, "y": None, "points": rings,
                                              "value": marks.value_text(holes or [False] * len(rings), kinds), "time": when}])


def field_origin(project):
    return 300 - project.context_px, 300 - project.context_px


def cell(project, x, y):
    """The layer cell of a field-px position on TILE."""
    ox, oy = field_origin(project)
    return (oy + y) // MASK_SCALE, (ox + x) // MASK_SCALE


def test_automatic_codes_and_no_layer_before_stage_2(project):
    assert exclusions.resolved(project, CORE, STAIN) is None and exclusions.digest(project, CORE, STAIN) is None
    write_mask(project)
    layer = exclusions.resolved(project, CORE, STAIN)
    assert layer.scale == MASK_SCALE == exclusions.SCALE
    assert layer.codes[5, 5] == 1 and layer.codes[100, 5] == 2 and layer.codes[145, 145] == 3 and layer.codes[50, 50] == 0
    write_mask(project, "NeuN")
    assert exclusions.resolved(project, CORE, "NeuN").codes[5, 5] == 0  # NeuN counts folded tissue


def test_paint_clear_and_overlapping_edits_latest_wins(project):
    write_mask(project)
    before = exclusions.digest(project, CORE, STAIN)
    ox, oy = field_origin(project)
    # Ann paints "out of focus" over a square of plain tissue
    edit(project, "Ann", [square(200, 200, 280, 280)], ["4"], 100.0)
    layer = exclusions.resolved(project, CORE, STAIN)
    assert layer.digest != before
    assert layer.codes[cell(project, 240, 240)] == 4 and layer.codes[cell(project, 320, 240)] == 0
    # Bob paints an artifact half over it later: the overlap is his
    edit(project, "Bob", [square(240, 200, 320, 280)], ["2"], 200.0)
    layer = exclusions.resolved(project, CORE, STAIN)
    assert layer.codes[cell(project, 220, 240)] == 4 and layer.codes[cell(project, 260, 240)] == 2 and layer.codes[cell(project, 310, 240)] == 2
    # Ann then clears a strip through both: counted again, whoever painted it
    edit(project, "Ann", [square(200, 230, 320, 250)], ["include"], 300.0)
    layer = exclusions.resolved(project, CORE, STAIN)
    assert layer.codes[cell(project, 220, 240)] == 0 and layer.codes[cell(project, 300, 240)] == 0
    assert layer.codes[cell(project, 220, 210)] == 4 and layer.codes[cell(project, 300, 270)] == 2
    # a clear over detection's fold removes it (the fold covers core rows 0–159)
    edit(project, "Cy", [[[ox + 0.0 - ox, 0.0 - oy], [400.0, 0.0 - oy], [400.0, 159.0 - oy], [0.0, 159.0 - oy]]], ["include"], 400.0)
    layer = exclusions.resolved(project, CORE, STAIN)
    assert (layer.codes[2:19, (ox + 8) // 8:(ox + 392) // 8] == 0).all() and layer.codes[5, 140] == 1
    assert [e.source for e in exclusions.edits(project, CORE, STAIN)] == ["Ann", "Bob", "Ann", "Cy"]  # the history, oldest first


def test_a_hole_in_a_painted_area_is_left_as_it_was(project):
    write_mask(project)
    edit(project, "Ann", [square(200, 200, 300, 300), square(230, 230, 270, 270)], ["1", "1"], 10.0, holes=[False, True])
    layer = exclusions.resolved(project, CORE, STAIN)
    assert layer.codes[cell(project, 210, 210)] == 1 and layer.codes[cell(project, 250, 250)] == 0


def test_fractions_and_dropped_labels(project):
    write_mask(project)
    layer = exclusions.resolved(project, CORE, STAIN)
    inside = "0 0 80 0 80 80 0 80"  # in the fold
    half = "0 140 80 140 80 180 0 180"  # rows 17–22: half in the fold
    plain = "400 400 480 400 480 480 400 480"
    shares = layer.fractions([inside, half, plain, None], [40, 40, 440, 600], [40, 160, 440, 20])
    assert shares[0] == 1.0 and 0.3 < shares[1] < 0.7 and shares[2] == 0.0 and shares[3] == 1.0  # no outline: the centre cell
    import pandas as pd

    ox, oy = field_origin(project)
    labels = pd.DataFrame([{"id": "a", "source": "Ann", "stain": STAIN, "task": task, "tile_id": TILE, "x": x, "y": y, "candidate": ""}
                           for task, x, y in (("class", 240.0, 240.0), ("class", 240.0, 804.0 - oy), ("tag", 240.0, 804.0 - oy))])
    kept = exclusions.drop_excluded(project, labels)  # the second sits on the linear artifact (core row 100 × 8)
    assert kept.index.tolist() == [0, 2]


def test_ai_reviewer_areas_are_edits_people_can_clear(project, tmp_path):
    write_mask(project)
    drawn = tmp_path / "manual.json"
    drawn.write_text(json.dumps({"data/cores/LIP-1/B-2/AT8.png": [{"exclusion_id": "E01", "polygon_core_px": square(600, 600, 700, 700)}],
                                 "data/cores/elsewhere.png": [{"polygon_core_px": square(0, 0, 9, 9)}]}))
    assert exclusions.import_reviewer(project, drawn) == 1
    assert exclusions.import_reviewer(project, drawn) == 0  # not twice
    layer = exclusions.resolved(project, CORE, STAIN)
    assert layer.codes[650 // 8, 650 // 8] == 2
    first = exclusions.edits(project, CORE, STAIN)
    assert [(e.source, e.kinds) for e in first] == [(exclusions.REVIEWER, ("2",))]
    ox, oy = field_origin(project)
    edit(project, "Ann", [square(600 - ox, 600 - oy, 700 - ox, 700 - oy)], ["include"], first[0].time + 10)
    assert exclusions.resolved(project, CORE, STAIN).codes[650 // 8, 650 // 8] == 0
    assert [e.source for e in exclusions.edits(project, CORE, STAIN)] == [exclusions.REVIEWER, "Ann"]
