# ruff: noqa: F811  (the rounds fixture is shared with test_annotation_rounds)
"""Each candidate's outlines (Tab: in the field tool the object under the mouse, in the Cores view every object): the
stain outline and Segment Anything's three sizes as stage 3 stored them; Tab cycles stain -> SAM small -> SAM medium,
the stain's default first. The one a person leaves an object on is kept per candidate in the label log (by name, with
its points; or the one Segment Anything drew live; "sam" of earlier releases is the output stage 3 kept), a changed
option counts as an outline fix, and the analysis keeps the automatic default whatever people chose."""
from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from test_annotation_rounds import TILE, _save, _store_row, rounds_project  # noqa: E402, F401

from stainid import candidates  # noqa: E402
from stainid.api import state, store_view  # noqa: E402
from stainid.labels import candidate_store, compare, decisions, outline_fixes, store  # noqa: E402

CORE = "LIP-1_B-2"
ORIGIN = 172  # the field's picture starts at (172, 172) in the core image
SWITCHED, KEPT, LEFT, NO_SAM, LEGACY = (400.0, 400.0), (460.0, 400.0), (520.0, 400.0), (400.0, 480.0), (460.0, 480.0)


def square(x: float, y: float, half: float) -> list[list[float]]:
    return [[x - half, y - half], [x + half, y - half], [x + half, y + half], [x - half, y + half]]


def _row(x: float, y: float, sam: bool = True) -> dict:
    """A candidate whose stain outline is 16 px across and whose Segment Anything masks are 12, 24 and 40 px across, the
    medium one kept by stage 3 (none at all with `sam` False)."""
    if not sam:
        return _store_row(x, y, 0.9, True, outline=square(x, y, 8), sam_status="failed")
    masks = [square(x, y, h) for h in (6, 12, 20)]
    return _store_row(x, y, 0.9, True, outline=square(x, y, 8), sam_outline=masks[1], sam_status="ok", sam_masks=masks,
                      sam_scores=[0.7, 0.9, 0.8], sam_default=1)


def test_the_outlines_the_one_shown_first_and_what_tab_moves_to():
    masks = [square(0, 0, h) for h in (6, 12, 20)]
    row = {"outline": square(0, 0, 8), "sam_outline": masks[1], "sam_status": "ok", "sam_masks": masks, "sam_default": 1}
    assert candidates.outline_choices({**row, "outline_default": "stain"}) == ([square(0, 0, 8), *masks], 0)  # 6E10, AT8: the stain's
    assert candidates.outline_choices({**row, "outline_default": "sam"}) == ([square(0, 0, 8), *masks], 2)  # NeuN: the SAM output stage 3 kept
    assert [candidates.tab_cycle(i, 0) for i in (0, 1, 2)] == [1, 2, 0]  # stain -> SAM small -> SAM medium -> stain
    assert [candidates.tab_cycle(i, 3) for i in (3, 0, 1, 2)] == [0, 1, 2, 3]  # SAM large only where it is shown first
    assert candidates.choice_index("sam", {**row, "outline_default": "stain"}) == 2  # "sam" of earlier pages: what stage 3 kept
    assert candidates.choice_index("sam_small", row) == 1 and candidates.choice_index("drawn", row) is None
    for default in ("stain", "sam"):  # the one shown first is always the analysis' own outline
        found, shown = candidates.outline_options({**row, "outline_default": default})
        options, first = candidates.outline_choices({**row, "outline_default": default})
        assert options[first] == found[shown]
    fell_back = {"outline": square(0, 0, 8), "sam_status": "failed", "outline_default": "sam"}
    assert candidates.outline_choices(fell_back) == ([square(0, 0, 8), None, None, None], 0)  # no SAM: the stain outline, SAM asked live


def test_the_chosen_outline_is_kept_per_candidate_and_counts_as_a_fix_and_the_analysis_is_unaffected(rounds_project):
    admin, (ann, _), _ = rounds_project
    project, tiles = state.get_state().project, state.get_state().tiles
    rows = [_row(*SWITCHED), _row(*KEPT), _row(*LEFT), _row(*NO_SAM, sam=False), _row(*LEGACY)]
    candidate_store.write_core(project, "6E10", CORE, pd.DataFrame(rows))
    decisions.forget()
    ids = {(r["x_core"], r["y_core"]): r["candidate_id"] for r in rows}
    analysis = (store_view.outline_rows(project, CORE, "6E10"), candidates.shown_area_um2(candidates.read_core(project, "6E10", CORE)).tolist())

    name = admin.post("/api/labels/sessions", json={"stain": "6E10", "round": 2, "open": True, "tile_ids": [TILE]}).json()["name"]
    field = ann.get(f"/api/annotate/{name}/fields/{TILE}").json()
    shown = {o["candidate_id"]: o for o in field["objects"]}
    assert all(len(o["masks"]) == 4 and o["mask"] == 0 for o in shown.values())  # [stain, SAM small, medium, large], 6E10's stain outline first
    every = ann.get(f"/api/annotate/{name}/fields/{TILE}/masks").json()
    assert every[ids[NO_SAM]][1] is None and every[ids[SWITCHED]][2] is not None

    live = [[x - ORIGIN, y - ORIGIN] for x, y in square(*NO_SAM, 11)]  # what Segment Anything drew live for the one without
    records = [{"task": "outline", "candidate": ids[SWITCHED], "points": [[0, 0], [1, 0], [1, 1]], "value": "sam_medium"},  # Tab, Tab: SAM medium
               {"task": "class", "candidate": ids[KEPT], "value": "plaque", "subtype": "compact", "mask": 0},  # a call, on the default
               {"task": "outline", "candidate": ids[NO_SAM], "points": live, "value": "sam_small"},  # asked live
               {"task": "outline", "candidate": ids[LEGACY], "points": [[0, 0], [1, 0], [1, 1]], "value": "sam"}]  # a page of an earlier release
    saved = _save(ann, name, TILE, records, submit=True)
    assert saved.status_code == 200, saved.text
    assert _save(ann, name, TILE, [{"task": "outline", "candidate": ids[NO_SAM], "value": "sam_small"}]).status_code == 400  # SAM live needs its outline

    labels = store.current(project.output("reviews"))
    outline = {r.candidate: r for r in labels[labels.task == "outline"].itertuples()}
    assert outline[ids[SWITCHED]].value == "sam_medium" and not outline[ids[SWITCHED]].confirmed  # a changed option
    assert outline[ids[SWITCHED]].points == [[x - ORIGIN, y - ORIGIN] for x, y in square(*SWITCHED, 12)]  # the stored SAM's own points
    assert outline[ids[KEPT]].value == "stain" and outline[ids[KEPT]].confirmed  # the default, left as it was
    assert outline[ids[LEFT]].value == "stain" and outline[ids[LEFT]].confirmed  # untouched: kept as left when the field was finished
    assert outline[ids[NO_SAM]].value == "sam_small" and outline[ids[NO_SAM]].points == live
    assert outline[ids[LEGACY]].value == "sam_medium" and outline[ids[LEGACY]].points == [[x - ORIGIN, y - ORIGIN] for x, y in square(*LEGACY, 12)]

    again = ann.get(f"/api/annotate/{name}/fields/{TILE}").json()  # the choice comes back with the field
    assert {m["candidate"]: m["value"] for m in again["mine"] if m["task"] == "outline"}[ids[SWITCHED]] == "sam_medium"

    fixed = compare.fixed_outlines(project, "6E10", tiles, CORE)
    assert set(fixed.candidate_id) == {ids[SWITCHED], ids[NO_SAM], ids[LEGACY]} and set(fixed.option) == {"sam_medium", "sam_small"}
    seen = outline_fixes.summary(project, "6E10", tiles)
    assert seen["switched"] == 3 and seen["picked"] == {"sam_medium": 2, "sam_small": 1} and seen["objects"] == 3
    kept = outline_fixes.rows(project, "6E10", tiles)
    assert set(kept.kind) == {"switched"} and kept.set_index("candidates").loc[ids[SWITCHED], "iou_sam_medium"] == 1.0

    after = (store_view.outline_rows(project, CORE, "6E10"), candidates.shown_area_um2(candidates.read_core(project, "6E10", CORE)).tolist())
    assert after == analysis  # the analysis keeps the automatic default


def test_the_cores_view_shows_either_outline(rounds_project):
    admin, _, _ = rounds_project
    project = state.get_state().project
    rows = [_row(*SWITCHED), _row(*NO_SAM, sam=False)]
    candidate_store.write_core(project, "6E10", CORE, pd.DataFrame(rows))
    sam = {(x, y): rings for x, y, _, rings in store_view.outline_rows(project, CORE, "6E10", "sam")}
    stain = {(x, y): rings for x, y, _, rings in store_view.outline_rows(project, CORE, "6E10", "stain")}
    assert sam[SWITCHED] != stain[SWITCHED] and len(sam[SWITCHED][0]) == 8  # the SAM outline stage 3 kept
    assert sam[NO_SAM] == stain[NO_SAM]  # none stored: the stain outline
    found = admin.get(f"/api/cores/{CORE}/6E10/objects").json()
    assert found["source"] == "v2" and found["outline_default"] == "stain"
    assert admin.get(f"/api/cores/{CORE}/6E10/outlines?outline=sam").status_code == 200


def test_a_split_piece_asks_segment_anything_with_its_siblings_as_context(rounds_project, monkeypatch):
    import numpy as np

    from stainid.api import sam_embeddings

    admin, (ann, _), _ = rounds_project
    project = state.get_state().project
    name = admin.post("/api/labels/sessions", json={"stain": "6E10", "round": 2, "open": True, "tile_ids": [TILE]}).json()["name"]
    weights = project.model("sam")
    weights.parent.mkdir(parents=True, exist_ok=True)
    weights.write_bytes(b"weights")
    asked = {}

    def outline(state_, tile_id, stain, points, box=None, sizes=False, avoid=None):
        asked.update(points=points, box=box, sizes=sizes, avoid=avoid)
        return {"options": [{"points": square(100, 100, 5)}, {"points": square(100, 100, 9)}], "chosen": 0}

    monkeypatch.setattr(sam_embeddings, "status", lambda *a, **k: {"ready": True})
    monkeypatch.setattr(sam_embeddings, "field_picture", lambda *a, **k: np.zeros((800, 800, 3), np.uint8))
    monkeypatch.setattr(sam_embeddings, "outline", outline)
    sibling = square(140, 100, 10)
    answer = ann.post(f"/api/annotate/{name}/fields/{TILE}/segment",
                      json={"points": [[100, 100, 1], [140, 100, 0]], "box": [85, 85, 115, 115], "sizes": True, "avoid": [sibling]})
    assert answer.status_code == 200 and len(answer.json()["options"]) == 2
    assert asked == {"points": [(100.0, 100.0, True), (140.0, 100.0, False)], "box": (85.0, 85.0, 115.0, 115.0), "sizes": True, "avoid": [sibling]}
    assert ann.post(f"/api/annotate/{name}/fields/{TILE}/segment", json={"points": [[100, 100, 1]], "box": [5, 5, 1, 1]}).status_code == 400
