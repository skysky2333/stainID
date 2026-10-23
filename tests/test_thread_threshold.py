# ruff: noqa: F811  (the rounds fixture is shared with test_annotation_rounds)
"""The AT8 thread threshold of "Fix the field" (`labels.thread_threshold`): threads mode's network at every thread
threshold from one ridge response, the step at detection's threshold being stage 2's thread output; one label per
field and person; the Models page's summary; the tutorial's threads practice keeps nothing."""
from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
import pytest

from stainid.labels import thread_threshold as tt
from stainid.stains.tau.threads import thread_ladder, thread_network


def _threads_picture(seed: int = 0) -> np.ndarray:
    """Light tissue with brown threads of several strengths (some faint: found only at lower thresholds)."""
    rng = np.random.default_rng(seed)
    rgb = np.full((400, 400, 3), 222, np.uint8)
    rgb += rng.integers(0, 6, rgb.shape, dtype=np.uint8)
    for n, (x0, y0, x1, y1) in enumerate(rng.integers(20, 380, size=(18, 4))):
        level = [60, 110, 150][n % 3]
        cv2.line(rgb, (int(x0), int(y0)), (int(x1), int(y1)), (int(level * 0.7), int(level * 0.85), level), 2)  # BGR-ish brown in RGB order
    return rgb


def test_the_ladder_at_detection_s_threshold_is_the_thread_network():
    rgb = _threads_picture()
    valid = np.ones(rgb.shape[:2], bool)
    from stainid.imaging.color import rgb_to_hed

    threshold = float(np.percentile(np.maximum(rgb_to_hed(rgb)[..., 2], 0), 99.5)) * 0.5
    ladder = thread_ladder(rgb, valid, threshold, 0.5, tt.FACTORS)
    alone = thread_network(rgb, valid, threshold, 0.5)
    at = ladder[tt.DETECTION]
    for key in ("line", "skeleton", "branchpoints", "endpoints", "width_um"):
        assert np.array_equal(at[key], alone[key]), key
    lengths = [int(n["skeleton"].sum()) for n in ladder]
    assert lengths[0] > lengths[tt.DETECTION] > 0 and lengths[0] >= lengths[-1]  # lower thresholds find more
    assert tt.FACTORS[0] == 0.5 and tt.FACTORS[-1] == 2.0 and len(tt.FACTORS) == 17 and tt.FACTORS[tt.DETECTION] == 1.0


def test_the_field_ladder_is_stage_2_s_threads_at_detection_s_threshold(synthetic_project):
    from stainid import candidates
    from stainid.project import load_project
    from stainid.v2 import proposals, run

    picture = synthetic_project / "data/cores/LIP-1/B-2/AT8.png"
    rgb = cv2.cvtColor(cv2.imread(str(picture)), cv2.COLOR_BGR2RGB)
    rng = np.random.default_rng(1)
    for x0, y0, x1, y1 in rng.integers(320, 800, size=(14, 4)):
        cv2.line(rgb, (int(x0), int(y0)), (int(x1), int(y1)), (95, 70, 45), 2)
    cv2.imwrite(str(picture), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    project = load_project(synthetic_project)
    tiles = run.groups(project, ["AT8"])[("LIP-1_B-2", "AT8")]
    calibration, nuclei = run.proposal_inputs(project)
    proposals.propose_core(project, "AT8", "LIP-1_B-2", tiles, calibration[(tiles[0]["tma"], "AT8")], nuclei)
    saved = candidates.threads(project, "LIP-1_B-2")
    tile = pd.read_csv(project.analysis_tiles()).query("stain == 'AT8'").iloc[0].to_dict()
    ladder = tt.compute(project, tile)
    assert ladder["status"] == "done" and len(ladder["steps"]) == len(tt.FACTORS)
    stage2 = float(saved[saved.tile_id == tile["tile_id"]].length_um.sum())
    assert stage2 > 0 and ladder["steps"][tt.DETECTION]["length_um"] == pytest.approx(stage2, abs=0.2)
    assert len({s["length_um"] for s in ladder["steps"]}) > 1  # the threshold changes what is found (in no fixed order on a synthetic picture)
    assert all(len(line) >= 4 and len(line) % 2 == 0 for line in ladder["steps"][tt.DETECTION]["lines"])
    wide = tt.compute(project, tile, wide=True)  # the simple neurite brush's: 0.25x-4x, the standard steps inside it to the pixel
    assert wide["ladder"] == "wide" and len(wide["steps"]) == len(tt.WIDE_FACTORS) and wide["detection"] == tt.WIDE_DETECTION
    assert [s["lines"] for s in wide["steps"][tt.WIDE_OFFSET:tt.WIDE_OFFSET + len(tt.FACTORS)]] == [s["lines"] for s in ladder["steps"]]
    assert wide["steps"][0]["length_um"] >= ladder["steps"][0]["length_um"]


def test_the_summary_of_people_s_thread_thresholds():
    rows = []
    for n, (tma, factor) in enumerate([("1", 0.8409), ("1", 0.8409), ("1", 0.7071), ("2", 1.0), ("2", 1.0), ("2", 1.0905), ("2", 1.0)]):
        rows.append({"task": tt.TASK, "source": f"P{n % 3}", "session": "s", "tile_id": f"t{n}", "value": f"{factor:.4f}",
                     "points": {"length_um": 100.0 / factor, "detection_length_um": 100.0}})
    rows.append({"task": tt.TASK, "source": "ai:round4", "session": "a", "tile_id": "t0", "value": "2.0000", "points": {}})  # never counted
    labels = pd.DataFrame(rows)
    tiles = pd.DataFrame({"tile_id": [f"t{n}" for n in range(7)], "tma": ["1", "1", "1", "2", "2", "2", "2"], "slide_id": ["a", "a", "b", "c", "c", "c", "c"]})
    found = tt.summary(labels, tiles)
    assert found["fields"] == 7 and found["people"] == 3 and all(r["source"] != "ai:round4" for r in found["rows"])
    tma = {g["tma"]: g for g in found["by_tma"]}
    assert tma["1"]["median"] == 0.8409 and tma["2"]["median"] == 1.0 and tma["1"]["length_ratio"] == pytest.approx(
        sum(100 / f for f in (0.8409, 0.8409, 0.7071)) / 300, abs=1e-3)
    assert found["suggested"] == 1.0  # the median of every field
    assert [p["tma"] for p in found["per_tma"]] == ["1"] and found["per_tma"][0]["suggested"] == 0.8409  # TMA 1 differs, 3 fields
    assert {g["slide"] for g in found["by_slide"]} == {"a", "b", "c"}
    assert tt.summary(pd.DataFrame(columns=["task", "source", "value", "tile_id"]), tiles)["fields"] == 0
    record = tt.record({"steps": [{"length_um": float(i)} for i in range(17)], "dab_threshold": 0.2}, 0.84)
    assert record["value"] == "0.8409" and not record["confirmed"] and record["points"]["length_um"] == 6.0 and record["points"]["detection_length_um"] == 8.0
    assert tt.record({"steps": [], "dab_threshold": None}, 1.0)["confirmed"] is True


pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from test_annotation_rounds import _save, _store_row, rounds_project  # noqa: E402, F401

AT8_TILE = "LIP-1_B-2_AT8_S01"
LADDER = {"status": "done", "factors": list(tt.FACTORS), "detection": tt.DETECTION, "dab_threshold": 0.05, "ridge_threshold": 0.0125,
          "steps": [{"factor": f, "length_um": round(500 / f, 1), "lines": [[10, 10, 50, 60]]} for f in tt.FACTORS], "stamp": "x"}


def test_a_field_s_thread_threshold_is_one_label_and_comes_back(rounds_project, monkeypatch):
    from stainid.api import state
    from stainid.labels import candidate_store, decisions, store

    admin, (ann, _), _ = rounds_project
    project = state.get_state().project
    rows = [{**_store_row(400.0, 400.0, 0.9, True), "candidate_id": candidate_store.candidate_id("AT8", "LIP-1_B-2", 400.0, 400.0), "tile_id": AT8_TILE,
             "class": "tau+ neuron", "subtype": "pretangle"}]
    candidate_store.write_core(project, "AT8", "LIP-1_B-2", pd.DataFrame(rows))
    decisions.forget()
    monkeypatch.setattr(tt, "field_ladder", lambda project, tile, wide=False: LADDER)
    monkeypatch.setattr(tt, "ready", lambda project, tile, wide=False: True)
    made = admin.post("/api/labels/sessions", json={"stain": "AT8", "round": 2, "open": True, "tile_ids": [AT8_TILE]})
    assert made.status_code == 200, made.text
    name = made.json()["name"]
    field = ann.get(f"/api/annotate/{name}/fields/{AT8_TILE}").json()
    assert field["threads"] == {"factor": None} and "threads" not in field["layers"]  # no thread layer while annotating objects
    assert ann.get(f"/api/annotate/{name}/fields/{AT8_TILE}/threads").json()["steps"][tt.DETECTION]["length_um"] == 500.0
    assert _save(ann, name, AT8_TILE, [{"task": tt.TASK, "value": "3"}]).status_code == 400  # outside 0.5–2
    assert _save(ann, name, AT8_TILE, [{"task": tt.TASK, "value": "0.84"}]).status_code == 200
    labels = store.current(project.output("reviews"))
    (saved,) = labels[labels.task == tt.TASK].to_dict("records")
    assert saved["id"] == tt.label_id(AT8_TILE) and saved["value"] == "0.8409" and saved["source"] == "Ann" and saved["session"] == name
    assert saved["points"]["length_um"] == pytest.approx(500 / 0.8409, abs=0.1) and saved["points"]["detection_length_um"] == 500.0
    assert saved["points"]["thread_threshold"] == pytest.approx(0.0125 * 0.8409, abs=1e-5) and not saved["confirmed"]
    assert ann.get(f"/api/annotate/{name}/fields/{AT8_TILE}").json()["threads"] == {"factor": 0.8409}
    # Enter without moving it: detection's is right
    assert _save(ann, name, AT8_TILE, [{"task": tt.TASK, "value": "1"}]).status_code == 200
    again = store.current(project.output("reviews"))
    assert again[again.task == tt.TASK].iloc[-1]["confirmed"] and len(again[again.task == tt.TASK]) == 1  # one per field and person
    summary = admin.get("/api/models/thread-detector").json()
    assert summary["fields"] == 1 and summary["suggested"] == 1.0
    # another stain's field has no threads step
    six = admin.post("/api/labels/sessions", json={"stain": "6E10", "round": 2, "open": True, "tile_ids": ["LIP-1_B-2_6E10_S01"]}).json()["name"]
    assert _save(ann, six, "LIP-1_B-2_6E10_S01", [{"task": tt.TASK, "value": "1"}]).status_code == 400
    assert ann.get(f"/api/annotate/{six}/fields/LIP-1_B-2_6E10_S01").json()["threads"] is None
    assert ann.get(f"/api/annotate/{name}/fields/{AT8_TILE}/threads/areas").status_code == 200  # the "not counted" shade
    assert ann.get(f"/api/annotate/{six}/fields/LIP-1_B-2_6E10_S01/threads/areas").status_code == 404


def test_the_threads_practice_keeps_nothing(rounds_project):
    import json

    from test_annotation_rounds import _log_lines

    from stainid import tutorial

    admin, (ann, _), _ = rounds_project
    folder = tutorial.folder() / tutorial.THREADS
    folder.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(folder / "field.jpg"), np.full((64, 64, 3), 200, np.uint8))
    (folder / "threads.json").write_text(json.dumps({"tile_id": "x", "credit": "Images: test", "width": 64, "height": 64, **LADDER}))
    before = _log_lines()
    for client in (ann, admin):
        found = client.get("/api/annotate/practice-threads/ladder")
        assert found.status_code == 200 and found.json()["credit"] == "Images: test" and len(found.json()["steps"]) == 17
        assert client.get("/api/annotate/practice-threads/picture.jpg").headers["content-type"] == "image/jpeg"
    assert _log_lines() == before


def test_threads_mode_shades_what_is_not_counted_from_people_s_final_say_without_ai(monkeypatch):
    import base64
    from types import SimpleNamespace

    from stainid import exclusions
    from stainid.api import store_view

    codes = np.zeros((40, 40), np.uint8)
    codes[:, 30:] = 3  # glass on the right
    codes[5:15, 5:15] = 1  # a fold
    codes[38:, :] = exclusions.OUTSIDE  # beyond the tiles: not shaded
    asked = {}

    def final_say(project, core_id, stain, ai=True, session=None):
        asked["ai"] = ai
        return SimpleNamespace(codes=codes, scale=16, digest="d")

    monkeypatch.setattr(exclusions, "final_say", final_say)
    project = SimpleNamespace(context_px=64)
    tile = {"core_id": "c", "stain": "AT8", "x_px": 64, "y_px": 64, "width_px": 512, "height_px": 512}
    found = store_view.not_counted(project, tile)
    assert asked["ai"] is False  # people's areas by the existing rule, no AI areas
    assert found["status"] == "done" and (found["x"], found["y"]) == (0, 0) and found["width"] == found["height"] == 640
    picture = cv2.imdecode(np.frombuffer(base64.b64decode(found["png"].split(",", 1)[1]), np.uint8), cv2.IMREAD_UNCHANGED)
    assert picture[10, 10, 3] == 255 and picture[20, 35, 3] == 255 and picture[20, 20, 3] == 0 and picture[39, 20, 3] == 0
    assert len(found["labels"]) == 2 and all(0 <= x <= 640 and 0 <= y <= 640 for x, y in found["labels"])
    codes[:] = exclusions.COUNTED
    assert store_view.not_counted(project, tile)["png"] is None
