# ruff: noqa: F811  (the rounds fixture is shared with test_annotation_rounds)
"""One source and one look across the review pages: detection's calls from the v2 store only, the Annotations layer's
comparison (old labels matched by position, a label without a type agreeing, cached), quick and plain refusals for
annotators, the "Mark bad areas" order, and one palette of distinct colours."""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402
from test_annotation_rounds import TILE, _save, rounds_project  # noqa: E402, F401

from stainid import palette  # noqa: E402
from stainid.api import state, store_view  # noqa: E402
from stainid.labels import areas, compare, decisions, frames, objectedit, overview, store  # noqa: E402

CORE = "LIP-1_B-2"
ROOT = Path(__file__).resolve().parents[1]


def _session(admin) -> str:
    return admin.post("/api/labels/sessions", json={"stain": "6E10", "round": 2, "open": True, "tile_ids": [TILE]}).json()["name"]


def _field_xy(row) -> tuple[float, float]:
    project, tiles = state.get_state().project, state.get_state().tiles
    tile = tiles[tiles.tile_id == TILE].iloc[0]
    x0, y0 = frames.field_origin(tile, project.context_px)
    return float(row["x_core"]) - x0, float(row["y_core"]) - y0


def test_a_label_without_a_type_agrees_and_old_labels_on_candidates_are_not_added(rounds_project):
    admin, (ann, ben), rows = rounds_project
    project, tiles = state.get_state().project, state.get_state().tiles
    name = _session(admin)
    ann.get(f"/api/annotate/{name}/fields/{TILE}")
    compact = rows[0]["candidate_id"]  # detection: plaque · compact
    assert _save(ann, name, TILE, [{"task": "class", "candidate": compact, "value": "plaque"}]).status_code == 200  # no type given
    x, y = _field_xy(rows[4])  # an old label, without a candidate id, 2 px off an accepted candidate
    store.append(project.output("reviews"), [{"id": "old-1", "source": "Ben", "stain": "6E10", "task": "class", "tile_id": TILE,
                                              "x": x + 2, "y": y - 1, "value": "plaque", "subtype": "compact", "time": 5.0}])
    decisions.forget()
    found = compare.versus_detection(project, "6E10", tiles, CORE).set_index("candidate_id")
    assert found.loc[compact, "kind"] == "agree"  # a label without a type says nothing against detection's type
    assert compare.agrees(("plaque", ""), ("plaque", "compact")) and not compare.agrees(("plaque", "diffuse"), ("plaque", "compact"))
    assert found.loc[rows[4]["candidate_id"], "kind"] == "agree"  # the old label counts on its candidate
    assert compare.added_objects(project, "6E10", tiles, CORE).empty  # and is no object detection missed
    layer = admin.get(f"/api/cores/{CORE}/6E10/annotations").json()
    assert layer["counts"]["added"] == 0 and layer["counts"]["changed"] == 0 and layer["counts"]["agree"] == 2


def test_the_comparison_is_worked_out_once_per_stain_until_the_labels_change(rounds_project, monkeypatch):
    admin, (ann, _), rows = rounds_project
    project, tiles = state.get_state().project, state.get_state().tiles
    name = _session(admin)
    ann.get(f"/api/annotate/{name}/fields/{TILE}")
    _save(ann, name, TILE, [{"task": "class", "candidate": rows[1]["candidate_id"], "value": "not a plaque"}])
    time.sleep(2.7)  # past the cache's racy window (cache.RACY_SECONDS): the log's signature is trusted
    first = compare.versus_detection(project, "6E10", tiles, CORE)  # worked out (here or by a warm-up in the background)
    made = []
    real = compare._versus_detection
    monkeypatch.setattr(compare, "_versus_detection", lambda *a, **k: made.append(1) or real(*a, **k))
    again = compare.versus_detection(project, "6E10", tiles, CORE)
    every = compare.versus_detection(project, "6E10", tiles)
    assert not made and first.equals(again) and len(every) == len(first)  # every core, and again: from what was kept
    _save(ann, name, TILE, [{"task": "class", "candidate": rows[2]["candidate_id"], "value": "plaque"}])
    assert len(compare.versus_detection(project, "6E10", tiles, CORE)) == 2 and made  # a new label: worked out again


def test_detection_s_calls_on_the_labels_page_come_from_the_store(rounds_project):
    admin, (ann, _), rows = rounds_project
    project, tiles = state.get_state().project, state.get_state().tiles
    name = _session(admin)
    ann.get(f"/api/annotate/{name}/fields/{TILE}")
    _save(ann, name, TILE, [{"task": "class", "candidate": rows[0]["candidate_id"], "value": "plaque", "subtype": "compact"},
                            {"task": "class", "candidate": rows[2]["candidate_id"], "value": "plaque", "subtype": "diffuse"}])
    labels = frames.in_field_frame(store.current(project.output("reviews")), tiles, project.context_px)
    seen = overview.stain_overview(project, "6E10", labels, tiles)
    assert seen["ai"]["n"] == 2 and seen["ai"]["precision"] == 1.0 and seen["ai"]["recall"] == 0.5  # rows[2] is a store reject
    assert seen["detection"]["found"] == 1 and seen["detection"]["rejected"] == 1
    calls = store_view.found(project, CORE, "6E10").set_index("candidate_id")
    assert calls.loc[rows[0]["candidate_id"], "class"] == "plaque" and calls.loc[rows[0]["candidate_id"], "subtype"] == "compact"


def test_merging_objects_far_apart_is_refused_at_once(rounds_project):
    admin, (ann, _), _ = rounds_project
    name = _session(admin)
    ann.get(f"/api/annotate/{name}/fields/{TILE}")
    a = [[10, 10], [30, 10], [30, 30], [10, 30]]
    far = [[600, 600], [620, 600], [620, 620], [600, 620]]
    started = time.monotonic()
    refused = ann.post(f"/api/annotate/{name}/fields/{TILE}/merge", json={"outlines": [a, far]})
    assert time.monotonic() - started < 1.5 and refused.status_code == 400 and "µm apart" in refused.json()["detail"]
    assert objectedit.merge_gap([a, [[34, 10], [50, 10], [50, 30], [34, 30]]]) == pytest.approx(4.0)
    assert objectedit.merge_gap([a, [[20, 20], [40, 20], [40, 40], [20, 40]]]) == 0.0  # overlapping


def test_annotators_never_see_a_raw_server_error(rounds_project, monkeypatch):
    import pickle

    from stainid.api import sam_embeddings

    admin, (ann, _), _ = rounds_project
    project = state.get_state().project
    name = _session(admin)
    ann.get(f"/api/annotate/{name}/fields/{TILE}")
    weights = project.model("sam")
    weights.parent.mkdir(parents=True, exist_ok=True)
    weights.write_bytes(b"not weights")

    def broken(*args, **kwargs):
        raise pickle.UnpicklingError("invalid load key, 'n'.")

    monkeypatch.setattr(sam_embeddings, "status", lambda *a, **k: {"ready": True})
    monkeypatch.setattr(sam_embeddings, "field_picture", lambda *a, **k: np.zeros((800, 800, 3), np.uint8))
    monkeypatch.setattr(sam_embeddings, "outline", broken)
    answer = ann.post(f"/api/annotate/{name}/fields/{TILE}/segment", json={"points": [[100, 100, 1]]})
    assert answer.status_code == 503 and "One-click outlines are not working" in answer.json()["detail"]
    assert "Unpickling" not in answer.text and "pickle" not in answer.text
    # an error nobody expected: a plain message for annotators (the detail goes to the log)
    monkeypatch.setattr(sam_embeddings, "field_picture", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("cuda: device-side assert")))
    plain = TestClient(admin.app, raise_server_exceptions=False)
    plain.post("/api/session/annotator", json={"name": "Ann"})
    seen = plain.post(f"/api/annotate/{name}/fields/{TILE}/segment", json={"points": [[100, 100, 1]]})
    assert seen.status_code == 500 and "cuda" not in seen.text and "Something went wrong" in seen.json()["detail"]


def test_likely_problems_come_first_and_about_a_tenth_get_a_second_check():
    signals = {f"C{i:02d}": {"excluded_share": 0.05, "tissue_share": 0.9, "unproposed_per_mm2": 1.0 + 0.01 * i, "dab_share": 0.1} for i in range(40)}
    signals["C07"] = {**signals["C07"], "excluded_share": 0.6}  # a large excluded share
    signals["C20"] = {**signals["C20"], "tissue_share": 0.2}  # little tissue
    scores = areas.priority(signals)
    assert scores["C07"] >= 2 and scores["C20"] >= 2 and max(v for k, v in scores.items() if k not in ("C07", "C20")) < 2
    cores = sorted(signals)
    first = areas.order(cores, scores, "problems", seed=3)
    assert set(first[:2]) == {"C07", "C20"} and scores[first[0]] >= scores[first[1]]  # highest first
    assert sorted(first) == cores and areas.order(cores, scores, "problems", seed=3) == first  # every core, reproducibly
    assert areas.order(cores, scores, "random", seed=3) == areas.order(cores, {}, "random", seed=3)  # random: scores ignored
    many = [f"core{i}" for i in range(4000)]
    second = areas.second_checks(many, seed=11)
    assert 0.085 < len(second) / len(many) < 0.115 and second == areas.second_checks(many, seed=11)
    assert second != areas.second_checks(many, seed=12)
    assert areas.needed({"second": second[:3]}, second[0]) == 2 and areas.needed({"second": second[:3]}, "elsewhere") == 1


def _lab(hexcode: str) -> np.ndarray:
    rgb = np.array(palette.rgb(hexcode)) / 255.0
    rgb = np.where(rgb > 0.04045, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
    xyz = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]]) @ rgb / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])


def test_one_palette_of_distinct_colours_everywhere(rounds_project):
    admin, (ann, _), _ = rounds_project
    colours = palette.palette()
    every = {f"classes[{i}]": c for i, c in enumerate(colours["classes"])}
    every |= {f"{group}.{k}": v for group, d in colours.items() if isinstance(d, dict) for k, v in d.items()}
    assert len({v.lower() for v in every.values()}) == len(every)  # no two the same
    names = sorted(every)
    near = min((float(np.linalg.norm(_lab(every[a]) - _lab(every[b]))), a, b) for i, a in enumerate(names) for b in names[i + 1:])
    assert near[0] >= 15, near  # and none hard to tell apart (CIE Lab)
    assert {k.type: k.color for k in store_view.EXCLUDED_KINDS if k.type} == {k: palette.area(k) for k in ("1", "2", "3", "4")}
    assert ann.get("/api/palette").json() == colours  # served, also to annotators
    # the pages take every colour from the palette: no colour written into the web app's code
    literal = re.compile(r"['\"]#[0-9a-fA-F]{6}['\"]")
    allowed = {"'#000000'", '"#000000"', "'#ffffff'", '"#ffffff"'}  # the dark edge and the white highlight of drawn shapes
    found = [(p.name, m) for p in (ROOT / "webapp" / "src").rglob("*.ts*") for m in literal.findall(p.read_text()) if m not in allowed]
    assert not found, found
    assert json.loads((ROOT / "src" / "stainid" / "palette.json").read_text())["areas"] == colours["areas"]
