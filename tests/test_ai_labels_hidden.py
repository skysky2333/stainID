# ruff: noqa: F811  (the rounds fixture is shared from test_annotation_rounds)
"""AI labels never show while annotating: the AI's object labels (imported reference sets "import: …", AI review
sessions "ai:<name>") and its bad areas ("AI reviewer", "ai:<name>") neither show nor set where anything starts in Fix
the field, Mark bad areas or the brush; people correct the pipeline's output, not the AI's. They stay in the training
and evaluation layers and in the Cores view's Annotations layer, marked as AI."""
from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from test_annotation_rounds import TILE, _plain_mask, rounds_project  # noqa: E402,F401

from stainid import exclusions  # noqa: E402
from stainid.api import state  # noqa: E402
from stainid.labels import decisions, store  # noqa: E402
from stainid.labels import exclusions as marks  # noqa: E402
from stainid.sampling.tiles import known_tiles  # noqa: E402

CORE = "LIP-1_B-2"


def test_the_one_rule_of_what_an_ai_label_is():
    for source in ("import: reference_6e10", "import: CNN outlier adjudication", "AI reviewer", "ai:round4"):
        assert store.is_ai_label(source), source
    for source in ("Ann", "", None, "aileen", "reviewer"):
        assert not store.is_ai_label(source), source


def test_a_field_and_a_core_with_ai_labels_open_exactly_as_detection_made_them(rounds_project):
    admin, (ann, _), rows = rounds_project
    project = state.get_state().project
    _plain_mask(project)
    areas = admin.post("/api/labels/sessions", json={"stain": "all", "kind": "areas", "open": True, "order": "random", "seed": 1}).json()["name"]
    fix = admin.post("/api/labels/sessions", json={"stain": "6E10", "round": 2, "open": True, "tile_ids": [TILE]}).json()["name"]
    pages = (f"/api/annotate/{areas}/cores/{CORE}/6E10/areas", f"/api/annotate/{areas}/cores/{CORE}/6E10/annotated",
             f"/api/annotate/{fix}/fields/{TILE}/areas", f"/api/annotate/{fix}/fields/{TILE}/annotated")
    before = {p: ann.get(p).json() for p in pages}
    field = ann.get(f"/api/annotate/{fix}/fields/{TILE}").json()
    core = ann.get(f"/api/annotate/{areas}/cores/{CORE}").json()

    # the AI labels: areas painted over the candidates, and calls on them that differ from detection's
    ring = [[376.0, 456.0], [432.0, 456.0], [432.0, 512.0], [376.0, 512.0]]  # over the candidate at (400, 480), clear of the fold
    area = {"stain": "6E10", "task": exclusions.TASK, "tile_id": CORE, "x": None, "y": None, "points": [ring],
            "value": marks.value_text([False], ["2"]), "session": "", "time": 5.0}
    cid = rows[4]["candidate_id"]  # at (460, 480): (288, 308) in the field picture, outside the AI's area
    call = {"stain": "6E10", "task": "class", "tile_id": TILE, "x": 288.0, "y": 308.0, "value": "not a plaque", "candidate": cid, "session": "", "time": 6.0}
    store.append(project.output("reviews"), [{**area, "id": "exclude:ai1", "source": "AI reviewer"}, {**area, "id": "exclude:ai2", "source": "ai:round4"},
                                             {**call, "id": "class:ai1", "source": "import: reference_6e10"},
                                             {**call, "id": "class:ai2", "source": "ai:round4", "session": "ai-round4"}])

    # Fix the field: every object as detection made it, nobody's check, nothing faded by the AI's areas
    again = ann.get(f"/api/annotate/{fix}/fields/{TILE}").json()
    assert again["objects"] == field["objects"] and again["checked"] == field["checked"] == {}
    live = ann.post(f"/api/annotate/{fix}/live", json={"tile_id": TILE}).json()
    assert live.get("checked", {}) == {}
    # Mark bad areas and the brush: the automatic layer and people's edits only
    for page in pages:
        assert ann.get(page).json() == before[page], page
    assert ann.get(f"/api/annotate/{areas}/cores/{CORE}").json()["images"][0]["edits"] == core["images"][0]["edits"] == 0

    # elsewhere they stay, marked as AI: training's layers and decisions, the Cores view's Annotations layer
    assert exclusions.final_say(project, CORE, "6E10").codes[60, 50] == 2
    covered = rows[3]["candidate_id"]  # left out of training by the AI's area, never faded by it in Fix the field
    assert exclusions.shares(project, "6E10", CORE, [covered])[covered] >= exclusions.EXCLUDED_AT
    assert exclusions.shares(project, "6E10", CORE, [covered], ai=False)[covered] < exclusions.EXCLUDED_AT
    assert not next(o for o in again["objects"] if o["candidate_id"] == covered).get("excluded")
    assert decisions.state(project, "6E10", known_tiles(project)).by(cid) == ["import: reference_6e10", "ai:round4"]
    layer = admin.get(f"/api/cores/{CORE}/6E10/annotations").json()
    assert {r["kind"] for r in layer["areas"]["regions"]} == {"2"} and all(e["ai"] for e in layer["areas"]["edits"])
    assert any(o["candidate_id"] == cid for o in layer["objects"])
