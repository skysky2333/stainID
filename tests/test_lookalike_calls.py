# ruff: noqa: F811  (the rounds fixture is shared with test_annotation_rounds)
"""A rule call that names a class is that class, not a reject: a deposit detection's vessel rule calls CAA is CAA (6E10
key Q) everywhere (Fix the field, at every slider value, saved as CAA when left; the Cores view; the label comparisons);
an AT8 candidate the model does not count whose stain lies along threads is a thread (key E). Rules that only reject
(a speck, a duplicate, an excluded area …) stay the stain's reject, dotted grey."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from stainid.labels import candidate_store
from stainid.labels import vocabulary as vocab
from stainid.v2 import classify

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from test_annotation_rounds import TILE, _save, _store_row, rounds_project  # noqa: E402, F401

CORE = "LIP-1_B-2"


def test_each_rule_maps_to_the_class_it_names_or_to_the_reject():
    plaques, tau, neun = vocab.defaults("6E10"), vocab.defaults("AT8"), vocab.defaults("NeuN")
    call = candidate_store.ai_call
    assert call(plaques, "rejected", "CAA", "", "rule:vascular_or_edge") == ("CAA", "")
    assert plaques.identity("CAA").key == "q"
    assert call(plaques, "rejected", "not a plaque", "", "rule:speck") == ("not a plaque", "")  # no speck class: a reject
    assert call(plaques, "rejected", "not a plaque", "", "rejected_by_model") == ("not a plaque", "")
    assert call(tau, "rejected", "thread", "", "rejected_by_model") == ("thread", "")  # the model's reject lying on threads
    assert tau.identity("thread").key == "e"
    assert call(tau, "rejected", "thread", "", "rule:thread_like") == ("thread", "")  # the rule (off by default) names it
    for rule in ("rule:small", "rule:irregular", "rule:low_contrast"):  # a shape rule left it out, its stain lies along threads
        assert call(tau, "rejected", "thread", "", rule) == ("thread", ""), rule
    for rule in ("rule:excluded", "rule:duplicate", "rule:edge", "rule:fold"):  # no object of its own: the reject
        assert call(tau, "rejected", "thread", "", rule) == ("not tau", ""), rule
    for rule in ("rule:excluded", "rule:near_profile", "rule:pale_nucleus", "rule:shape", "rule:edge", "rule:fold", "rule:irregular"):
        assert call(neun, "rejected", "not a neuron", "", rule) == ("not a neuron", ""), rule  # NeuN has no look-alike
    assert call(plaques, "accepted", "plaque", "compact", "accepted") == ("plaque", "compact")
    assert set(classify.CLASS_RULES) == {"rule:vascular_or_edge", "rule:thread_like"}


class _Fixed(classify.Model):
    def probability(self, rows):
        return np.asarray(rows["p"], float)


def test_stage_4_names_the_look_alikes_and_rejects_the_rest():
    project = SimpleNamespace(pixel_size_um=0.5)
    at8 = pd.DataFrame({"candidate_id": list("abcdef"), "x_core": [0.0, 500, 900, 10, 1500, 2000], "y_core": 0.0, "source": "ring",
                        "m_normalized_contrast": 1.0, "excluded_fraction": [0, 0, 0.9, 0, 0, 0], "thread_fraction": [0, 0.8, 0.8, 0.8, 0.9, 0.7],
                        "legacy_class": ["", "", "", "", "", "dab_body"], "m_soma_area_um2": [99, 99, 99, 99, 99, 10], "p": [0.9, 0.2, 0.2, 0.85, 0.9, 0.9]})
    calls = classify.call(project, "AT8", at8, _Fixed("AT8", {}, None, "m", 0.5, []), {"excluded_fraction": 0.5}).set_index("candidate_id")
    assert calls.loc["a", "class"] == "tau+ neuron" and calls.loc["e", "class"] == "tau+ neuron"  # counted: never a thread
    assert calls.loc["b", "reason"] == "rejected_by_model" and calls.loc["b", "class"] == "thread"
    assert calls.loc["c", "reason"] == "rule:excluded" and calls.loc["c", "class"] == "not tau"
    assert calls.loc["d", "reason"] == "rule:duplicate" and calls.loc["d", "class"] == "not tau"
    assert calls.loc["f", "reason"] == "rule:small" and calls.loc["f", "class"] == "thread"
    plaques = pd.DataFrame({"candidate_id": list("xyz"), "x_core": [0.0, 100, 200], "y_core": 0.0, "p": 0.9,
                            "m_equivalent_diameter_um": [20.0, 8.0, 20.0], "lumen_contact_fraction": [0.0, 0.0, 0.6], "normalized_inner_dab": 0.0})
    calls = classify.call(project, "6E10", plaques, _Fixed("6E10", {}, None, "m", 0.5, []), {}).set_index("candidate_id")
    assert list(calls["class"]) == ["plaque", "not a plaque", "CAA"]
    assert list(calls.reason) == ["accepted", "rule:speck", "rule:vascular_or_edge"]


def test_start_call_below_the_threshold_is_the_named_look_alike():
    from stainid.labels import starting

    tangle, thread, reject = ("tau+ neuron", "pretangle"), ("thread", ""), ("not tau", "")
    assert starting.start_call(0.4, tangle, 0.5, tangle, reject, thread) == thread
    assert starting.start_call(0.6, tangle, 0.5, thread, reject, thread) == tangle
    assert starting.start_call(0.4, tangle, 0.5, tangle, reject) == reject
    caa = ("CAA", "")
    assert all(starting.start_call(0.7, None, t / 10, caa, ("not a plaque", "")) == caa for t in range(11))  # fixed: CAA at every value


def _caa_core(project):
    rows = [_store_row(400.0, 400.0, 0.9, True, threshold=0.5),
            _store_row(400.0, 480.0, 0.7, False, threshold=0.5, reason="rule:vascular_or_edge", **{"class": "CAA"}),
            _store_row(520.0, 400.0, 0.8, False, threshold=0.5, reason="rule:speck")]
    candidate_store.write_core(project, "6E10", CORE, pd.DataFrame(rows))
    return [r["candidate_id"] for r in rows]


def test_a_vessel_deposit_is_caa_on_every_page(rounds_project):
    from stainid.api import state
    from stainid.api.store_view import LOOKALIKE, found
    from stainid.labels import compare, decisions, starting, store

    admin, (ann, _), _ = rounds_project
    project, tiles = state.get_state().project, state.get_state().tiles
    plaque, vascular, speck = _caa_core(project)
    decisions.forget()
    name = admin.post("/api/labels/sessions", json={"stain": "6E10", "round": 2, "open": True, "tile_ids": [TILE]}).json()["name"]
    objects = {o["candidate_id"]: o for o in ann.get(f"/api/annotate/{name}/fields/{TILE}").json()["objects"]}
    caa = objects[vascular]
    assert (caa["ai_value"], caa["if_accepted"], caa["rule"]) == ("CAA", None, "rule:vascular_or_edge")
    for step in range(101):  # selected as CAA at every slider value: the slider never changes it
        assert starting.start_call(caa["ai_probability"], caa["if_accepted"], step / 100, ("CAA", ""), ("not a plaque", ""),
                                   caa.get("if_rejected")) == ("CAA", "")
    assert objects[speck]["ai_value"] == "not a plaque"
    # the Cores view: CAA solid in its class (model_class "lookalike"), the speck a dotted reject
    view = found(project, CORE, "6E10").set_index("candidate_id")
    assert view.loc[vascular, "model_class"] == LOOKALIKE and view.loc[vascular, "class"] == "CAA"
    assert view.loc[speck, "model_class"] == "rejected" and view.loc[speck, "class"] == ""
    preview = {r[0]: r for r in admin.get(f"/api/cores/{CORE}/6E10/threshold-preview").json()["rows"]}
    assert preview[vascular][5] is True and preview[plaque][5] is False  # fixed at every value
    # an untouched field saved at another threshold: CAA recorded as CAA
    assert _save(ann, name, TILE, [{"task": "threshold", "value": "0.2"}], submit=True).status_code == 200
    labels = store.current(project.output("reviews"))
    calls = labels[(labels.source == "Ann") & (labels.task == "class")].set_index("candidate")
    assert calls.loc[vascular, "value"] == "CAA" and calls.loc[speck, "value"] == "not a plaque"
    # the label comparisons: a person's CAA agrees with detection's (one core, and every core from the compact index)
    for core_id in (CORE, None):
        rows = compare.versus_detection(project, "6E10", tiles, core_id).set_index("candidate_id")
        assert rows.loc[vascular, "kind"] == "agree", core_id
    assert candidate_store.core_calls(project, "6E10", [CORE]).set_index("candidate_id").loc[vascular, "reason"] == "rule:vascular_or_edge"


def test_an_at8_model_reject_on_threads_starts_as_a_thread(rounds_project):
    from stainid.api import state
    from stainid.labels import starting

    project = state.get_state().project
    ids = []
    rows = []
    for x, p, accepted, thread in ((400.0, 0.9, True, 0.0), (460.0, 0.9, True, 0.8), (520.0, 0.2, False, 0.8), (580.0, 0.2, False, 0.8)):
        row = _store_row(x, 400.0, p, accepted, threshold=0.5, thread_fraction=thread)
        row.update({"candidate_id": candidate_store.candidate_id("AT8", CORE, x, 400.0), "class": "tau+ neuron" if accepted else "thread",
                    "subtype": "pretangle" if accepted else ""})
        rows.append(row)
        ids.append(row["candidate_id"])
    rows[3].update({"reason": "rule:excluded", "class": "thread"})  # an older store: a rejecting rule with the thread class
    candidate_store.write_core(project, "AT8", CORE, pd.DataFrame(rows))
    below = starting.if_rejected(project, "AT8", CORE)
    words = vocab.defaults("AT8")
    named = {c: candidate_store.ai_call(words, *below[c])[0] if below[c] is not None else None for c in ids}
    assert named == {ids[0]: "not tau", ids[1]: "thread", ids[2]: "thread", ids[3]: None}  # a rule decided ids[3]: no slider call
    assert candidate_store.ai_call(words, "rejected", "thread", "", "rule:excluded")[0] == "not tau"


def test_a_thread_like_stain_is_a_reject_reason_with_its_own_name_and_key():
    tau = vocab.defaults("AT8")
    thread = tau.identity("thread")
    assert thread.name == "thread-like stain (not a neuron)" and thread.key == "e" and thread.role == vocab.NEGATIVE
    assert tau.rejecting("thread") and tau.rejecting("not tau") and not tau.rejecting("astroglial tau") and not tau.rejecting("tau+ neuron")
    assert not vocab.defaults("6E10").rejecting("CAA")  # CAA stays a solid look-alike
    assert any(k["identity"] == "thread" and k["key"] == "e" for g in tau.keymap()["groups"] for k in g["items"])  # its key kept
    assert next(c for c in tau.to_api()["classes"] if c["id"] == "thread")["rejects"] is True
    saved = vocab.from_config("AT8", [{"id": "thread", "name": "thread or dystrophic neurite", "key": "e"}])
    assert saved.identity("thread").name == "thread-like stain (not a neuron)" and saved.rejecting("thread")  # older settings
    mine = vocab.from_config("AT8", [{"id": "fibre", "name": "fibre", "role": "negative", "key": "t", "rejects": True}])
    assert mine.rejecting("fibre")
