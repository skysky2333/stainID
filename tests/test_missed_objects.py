"""Objects the AI never proposed: they count as missed in the AI's accuracy, are measured where people added them, and
train the stain model."""
from __future__ import annotations

import json

import cv2
import numpy as np
import pandas as pd
import pytest

from stainid.labels import added, overview, pool, store
from stainid.project import load_project
from stainid.training.candidates import Field, measure_at
from stainid.training.train import train_model

FIELD, OTHER = "T", "U"


def _pool(project, stain: str, rows: list[dict], tile_id: str = FIELD, features: list[str] | None = None) -> None:
    folder = pool.folder(project, stain)
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).assign(complete=True).to_csv(folder / f"{tile_id}.csv", index=False)
    if features:
        pool._check_names(project, stain, features)


def _label(person: str, x: float, y: float, value: str, tile_id: str = FIELD, session: str = "round_1", stain: str = "6E10") -> dict:
    return {"id": store.object_id("class", tile_id, x, y), "source": person, "stain": stain, "task": "class", "tile_id": tile_id,
            "x": x, "y": y, "value": value, "session": session}


def test_accuracy_counts_objects_the_ai_never_proposed(tmp_path):
    project = load_project(tmp_path)
    root = project.output("reviews")
    candidates = [("compact plaque", 10.0), ("not a plaque", 50.0), ("diffuse plaque", 90.0), ("not a plaque", 130.0)]
    _pool(project, "6E10", [{"x": x, "y": 10.0, "radius_px": 8.0, "ai_value": v, "ai_probability": 0.5, "tma": "1"} for v, x in candidates])
    store.append(root, [
        _label("Ann", 10.0, 10.0, "compact plaque"),    # proposed, found
        _label("Ann", 50.0, 10.0, "diffuse plaque"),    # proposed but rejected
        _label("Ann", 90.0, 10.0, "not a plaque"),      # the AI's false positive
        _label("Ann", 130.0, 10.0, "not a plaque"),
        _label("Ann", 300.0, 300.0, "compact plaque"),  # added: the AI never proposed it
        _label("Ann", 400.0, 400.0, "not a plaque"),    # added, but not a plaque: no miss
        _label("Ann", 20.0, 20.0, "compact plaque", tile_id=OTHER),  # a field that is not prepared: cannot tell
    ])
    tiles = pd.DataFrame({"tile_id": [FIELD, OTHER], "stain": "6E10"})
    seen = overview.stain_overview(project, "6E10", store.current(root), tiles)
    assert seen["detection"] == {"real": 3, "found": 1, "never_proposed": 1, "rejected": 1, "recall": 1 / 3, "never_proposed_share": 1 / 3,
                                 "rejected_share": 1 / 3, "not_checked": 1}
    assert seen["ai"]["n"] == 4 and seen["ai"]["recall"] == 0.5  # the old figure: among proposed objects only
    assert seen["added"] == {"objects": 2, "measured": 0, "could_not_be_measured": 0, "waiting": 2}
    assert seen["objects_added"] == 3

    # measured, the added plaque turns out to be the candidate at (130, 10) (the click was off its centre): proposed, rejected
    folder = added.folder(project, "6E10")
    folder.mkdir(parents=True)
    pd.DataFrame([{"x": 300.0, "y": 300.0, "tma": "1", "added_by_person": True, "measured": True, "candidate_x": 130.0, "candidate_y": 10.0,
                   "ai_value": "not a plaque"},
                  {"x": 400.0, "y": 400.0, "tma": "1", "added_by_person": True, "measured": False, "reason": "no stained object"}]).to_csv(
        folder / f"{FIELD}.csv", index=False)
    seen = overview.stain_overview(project, "6E10", store.current(root), tiles)
    assert seen["detection"]["never_proposed"] == 0 and seen["detection"]["rejected"] == 2
    assert seen["added"] == {"objects": 1, "measured": 0, "could_not_be_measured": 1, "waiting": 0}  # the one that is a candidate is left out


def test_an_added_object_is_measured_at_the_click():
    rng = np.random.default_rng(0)
    rgb = np.clip(rng.normal((200, 175, 200), 6, size=(800, 800, 3)), 0, 255).astype(np.uint8)
    cv2.circle(rgb, (400, 400), 30, (150, 100, 60), -1)
    field = Field({"tile_id": FIELD, "tma": "1", "stain": "6E10"}, rgb, (slice(100, 700), slice(100, 700)), 0.27, 0.05, np.zeros((800, 800), bool))
    row, features, names, how = measure_at("6E10", None, field, 410.0, 392.0)  # off the centre, inside the plaque
    assert row is not None and abs(row["centroid_x_px"] - 400) < 2 and abs(row["centroid_y_px"] - 400) < 2
    assert len(features) == len(names) and np.isfinite(features).mean() > 0.9 and how == "as the detection step segments it"
    nothing, _, _, why = measure_at("6E10", None, field, 200.0, 600.0)
    assert nothing is None and "no stained object" in why
    manual = np.zeros((800, 800), bool)
    manual[550:650, 150:250] = True  # an area marked by hand is left out, as in the detection step
    marked = Field(field.tile, rgb, field.inner, 0.27, 0.05, manual)
    assert "left out" in measure_at("6E10", None, marked, 200.0, 600.0)[3]


@pytest.fixture()
def client(synthetic_project, monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from stainid.api import state
    from stainid.api.app import create_app
    from stainid.api.routes import annotate

    monkeypatch.chdir(synthetic_project)
    monkeypatch.setattr(annotate, "prepare_job", lambda *args, **kwargs: None)
    rng = np.random.default_rng(1)
    rgb = np.clip(rng.normal((200, 175, 200), 6, size=(1200, 1200, 3)), 0, 255).astype(np.uint8)
    cv2.circle(rgb, (560, 560), 30, (150, 100, 60), -1)  # field (388, 388): the field picture starts 128 px before the tile at 300
    cv2.imwrite(str(synthetic_project / "data/cores/LIP-1/B-2/6E10.png"), rgb[:, :, ::-1])
    state.reset()
    model = state.get_state().project.model("amyloid")
    model.parent.mkdir(parents=True, exist_ok=True)
    model.write_bytes(b"model")
    admin = TestClient(create_app(), client=("127.0.0.1", 50000))
    assert admin.post("/api/session/admin", json={"password": "admin"}).status_code == 200
    yield admin
    state.reset()


def test_people_add_missed_objects_and_they_are_measured(client, monkeypatch):
    from people import annotator

    from stainid.api.routes import annotate
    from stainid.api.state import get_state

    threads = []
    measure = annotate.measure_added_soon
    monkeypatch.setattr(annotate, "measure_added_soon", lambda *args: threads.append(measure(*args)))
    project = get_state().project
    tile = "LIP-1_B-2_6E10_S01"
    _pool(project, "6E10", [{"x": 100.0, "y": 100.0, "radius_px": 10.0, "ai_value": "not a plaque", "ai_probability": 0.2, "tma": "1"}], tile_id=tile)
    client.post("/api/labels/sessions", json={"name": "round_1", "stain": "6E10", "tile_ids": [tile], "open": True})
    ann = annotator(client, "Ann", "round_1")
    url = "/api/annotate/round_1/labels"
    ann.post(url, json={"records": [{"task": "class", "tile_id": tile, "x": 100.0, "y": 100.0, "value": "not a plaque"}]})
    assert threads == []  # a label on a candidate needs no measuring
    ann.post(url, json={"records": [{"task": "class", "tile_id": tile, "x": 392.0, "y": 385.0, "value": "diffuse plaque"},
                                    {"task": "class", "tile_id": tile, "x": 600.0, "y": 150.0, "value": "compact plaque"}]})
    for thread in threads:
        thread and thread.join(30)
    rows = added.load_field(project, "6E10", tile).set_index("x")
    assert rows.added_by_person.all() and bool(rows.loc[392.0, "measured"]) and not bool(rows.loc[600.0, "measured"])
    assert abs(rows.loc[392.0, "measured_x"] - 388) < 3 and rows.loc[600.0, "reason"]
    work = ann.get(f"/api/annotate/round_1/fields/{tile}").json()
    assert sorted((a["x"], a["state"]) for a in work["added"]) == [(100.0, "proposed"), (392.0, "measured"), (600.0, "could_not_be_measured")]
    assert ann.get(f"/api/annotate/round_1/fields/{tile}/added").json() == work["added"]
    plaques = next(s for s in client.get("/api/labels/overview").json()["stains"] if s["stain"] == "6E10")
    assert plaques["detection"]["never_proposed"] == 2 and plaques["detection"]["recall"] == 0
    assert plaques["added"] == {"objects": 2, "measured": 1, "could_not_be_measured": 1, "waiting": 0}
    assert [r["x"] for r in store.raw_records(project.output("reviews"))] == [100.0, 392.0, 600.0]  # the logs only grew


def test_person_added_objects_train_the_model_and_are_reported(tmp_path):
    project = load_project(tmp_path)
    rng = np.random.default_rng(2)
    names = ["size", "stain_mean", "context"]
    n = 60
    x = rng.normal(size=(n, 3))
    truth = x[:, 0] + x[:, 1] > 0
    frame = pd.DataFrame({"x": np.arange(n) * 20.0, "y": 10.0, "radius_px": 8.0, "ai_value": "not a tau+ neuron", "ai_probability": 0.5,
                          "tma": [str(i % 3 + 1) for i in range(n)], "soma_dab_p90": 0.2})
    _pool(project, "AT8", pd.concat([frame, pd.DataFrame(x, columns=names)], axis=1).to_dict("records"), features=names)
    root = project.output("reviews")
    value = {True: "tau+ neuron: tangle", False: "not a tau+ neuron"}
    store.append(root, [_label("Ann", i * 20.0, 10.0, value[bool(t)], stain="AT8") for i, t in enumerate(truth)])
    extra = rng.normal(loc=1.0, size=(9, 3))  # nine tau+ neurons the AI never proposed, on all three TMAs
    rows = [{"x": 50.0 + 25 * i, "y": 500.0, "radius_px": 8.0, "tma": str(i % 3 + 1), "soma_dab_p90": 0.2, "added_by_person": True, "measured": True,
             "measured_how": "as the detection step segments it", **dict(zip(names, extra[i]))} for i in range(9)]
    rows.append({"x": 900.0, "y": 900.0, "tma": "1", "added_by_person": True, "measured": False, "reason": "no stained object could be segmented at this spot"})
    added.folder(project, "AT8").mkdir(parents=True)
    pd.DataFrame(rows).to_csv(added.path(project, "AT8", FIELD), index=False)
    store.append(root, [_label("Ann", r["x"], r["y"], "tau+ neuron: tangle", stain="AT8") for r in rows])

    report = json.loads((train_model(project, "AT8") / "report.json").read_text())
    assert report["validation"] == "leave-one-TMA-out" and report["new_model"]["n"] == n + 9
    found = report["person_added"]
    assert found["objects"] == 9 and found["positive"] == 9 and found["could_not_be_measured"] == 1
    detection = found["detection"]
    assert detection["never_proposed"] == 10 and detection["real"] == int(truth.sum()) + 10
    assert detection["found"] + detection["rejected"] == int(truth.sum())
