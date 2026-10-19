"""The acceptance threshold of Fix the field and of the Cores view's preview re-decides the model's part only, as
stage 4 would: p >= threshold passes (a missing score is 0); detection's rules (a speck, a deposit on a vessel wall
called CAA) keep their call at every value. So at 0 every candidate no rule decided (outside excluded areas) counts, at
1 only those scoring 1, and raising the threshold never counts more."""
# ruff: noqa: F811  (the rounds fixture is shared with test_annotation_rounds)
from __future__ import annotations

import math

import pandas as pd
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from test_annotation_rounds import TILE, _save, _store_row, rounds_project  # noqa: E402, F401

from stainid.api import state  # noqa: E402
from stainid.labels import candidate_store, decisions, starting, store  # noqa: E402

CORE = "LIP-1_B-2"
DETECTED, REJECT, COUNTED = ("plaque", "compact"), ("not a plaque", ""), ("plaque", "diffuse")


def test_score_at_or_above_the_threshold_counts_and_a_missing_score_is_zero():
    call = lambda p, t, passed=COUNTED: starting.start_call(p, passed, t, REJECT, REJECT)  # noqa: E731
    for p in (0.0, None, math.nan, 0.3, 1.0):
        assert call(p, 0.0) == COUNTED  # at 0: everything
    assert call(1.0, 1.0) == COUNTED and call(0.999, 1.0) == REJECT  # at 1: only p = 1
    assert call(0.29, 29 * 0.01) == COUNTED  # the page's step (0.29000000000000004): p = 0.29 counts at 0.29
    assert call(0.5, 0.5) == COUNTED and call(0.4999, 0.5) == REJECT
    assert call(0.9, 0.0, passed=None) == REJECT  # an excluded area: detection's call, whatever the threshold


def _project_with_rules():
    """The field's candidates: detection's accepted and rejected ones, a speck and a vascular deposit (rules after the
    model), one with no score and one with score 0, one in an area detection left out of the analysis."""
    project = state.get_state().project
    rows = [_store_row(400.0, 400.0, 0.9, True, threshold=0.5), _store_row(460.0, 400.0, 0.45, False, threshold=0.5),
            _store_row(520.0, 400.0, 0.8, False, threshold=0.5, reason="rule:speck"),
            _store_row(400.0, 480.0, 0.7, False, threshold=0.5, reason="rule:vascular_or_edge", **{"class": "CAA"}),
            _store_row(460.0, 480.0, float("nan"), False, threshold=0.5), _store_row(520.0, 480.0, 0.0, False, threshold=0.5),
            _store_row(580.0, 480.0, 0.95, False, threshold=0.5, reason="rule:excluded")]
    candidate_store.write_core(project, "6E10", CORE, pd.DataFrame(rows))
    decisions.forget()
    return project, [r["candidate_id"] for r in rows]


def test_the_field_and_the_cores_view_count_every_candidate_at_zero(rounds_project):
    admin, (ann, _), _ = rounds_project
    project, ids = _project_with_rules()
    accepted, weak, speck, vascular, unscored, zero, excluded = ids
    name = admin.post("/api/labels/sessions", json={"stain": "6E10", "round": 2, "open": True, "tile_ids": [TILE]}).json()["name"]
    field = ann.get(f"/api/annotate/{name}/fields/{TILE}").json()
    objects = {o["candidate_id"]: o for o in field["objects"]}
    assert objects[vascular]["ai_value"] == "CAA"  # detection's own call on a vascular deposit: CAA, not "not a plaque"
    assert objects[speck]["rule"] == "rule:speck" and objects[vascular]["rule"] == "rule:vascular_or_edge"
    for c in (speck, vascular, excluded):  # a rule's call, or an excluded area: detection's call at every value
        assert objects[c]["if_accepted"] is None, c
    for c in (accepted, weak, unscored, zero):
        assert objects[c]["if_accepted"][0] == "plaque", c
    # the count never grows with the threshold (the page's rule, as the server saves untouched objects)
    counted = []
    for step in range(101):
        t = step / 100
        counted.append(sum(starting.start_call(o["ai_probability"], o["if_accepted"], t, (o["ai_value"], o["ai_subtype"]), REJECT)[0] == "plaque"
                           for o in objects.values() if not o.get("excluded")))
    assert all(a >= b for a, b in zip(counted, counted[1:])), counted
    assert counted[0] == 4 and counted[-1] == 0  # at 0: every candidate no rule decided; at 1: none scores 1
    # the Cores view's preview: a score for each (0 when missing) and the call once it passes
    preview = admin.get(f"/api/cores/{CORE}/6E10/threshold-preview").json()
    rows = {r[0]: r for r in preview["rows"]}
    assert rows[unscored][1] == 0.0 and rows[zero][1] == 0.0
    assert all(rows[c][2] == "" for c in (speck, vascular, excluded)) and rows[speck][4] == "rule:speck"
    assert all(rows[c][2] == "plaque" for c in (accepted, weak, unscored, zero))
    # Save and continue at 0, untouched: every candidate no rule decided counts; the rules' calls stay
    assert _save(ann, name, TILE, [{"task": "threshold", "value": "0"}], submit=True).status_code == 200
    labels = store.current(project.output("reviews"))
    calls = labels[(labels.source == "Ann") & (labels.task == "class")].set_index("candidate")
    assert all(calls.loc[c, "value"] == "plaque" for c in (accepted, weak, unscored, zero))
    assert calls.loc[speck, "value"] == "not a plaque" and calls.loc[vascular, "value"] == "CAA"
    assert calls.loc[excluded, "value"] == "not a plaque"


def test_at_one_only_a_score_of_one_counts(rounds_project):
    admin, (ann, _), _ = rounds_project
    project = state.get_state().project
    rows = [_store_row(400.0, 400.0, 1.0, True, threshold=0.5), _store_row(460.0, 400.0, 0.99, True, threshold=0.5),
            _store_row(520.0, 400.0, float("nan"), False, threshold=0.5)]
    candidate_store.write_core(project, "6E10", CORE, pd.DataFrame(rows))
    decisions.forget()
    name = admin.post("/api/labels/sessions", json={"stain": "6E10", "round": 2, "open": True, "tile_ids": [TILE]}).json()["name"]
    assert _save(ann, name, TILE, [{"task": "threshold", "value": "1"}], submit=True).status_code == 200
    labels = store.current(project.output("reviews"))
    calls = labels[(labels.source == "Ann") & (labels.task == "class")].set_index("candidate")
    assert calls.loc[rows[0]["candidate_id"], "value"] == "plaque"
    assert calls.loc[rows[1]["candidate_id"], "value"] == "not a plaque" and calls.loc[rows[2]["candidate_id"], "value"] == "not a plaque"
