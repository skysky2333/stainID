"""Choosing labelling fields: by hand from a picker or automatically, only fields that are ready, and each session's
fields with where they stand."""
from __future__ import annotations

import shutil

import pandas as pd
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402
from people import annotator  # noqa: E402

from stainid.api import state  # noqa: E402
from stainid.api.app import create_app  # noqa: E402

READY, WAITING = "LIP-1_B-2_6E10_S01", "LIP-1_C-2_6E10_W0000"


@pytest.fixture()
def client(synthetic_project, monkeypatch):
    """The synthetic project with a 6E10 model and a second core, C-2, that detection has not reached yet."""
    from stainid.api.routes import annotate

    monkeypatch.chdir(synthetic_project)
    monkeypatch.setattr(annotate, "prepare_job", lambda *args, **kwargs: None)
    monkeypatch.setattr(annotate, "measure_added_soon", lambda *args, **kwargs: None)
    root = synthetic_project
    analysis = root / "data" / "analysis"
    tiles = pd.read_csv(analysis / "cohort_whole_core_tiles.csv", dtype=str)
    extra = tiles[tiles.stain == "6E10"].assign(tile_id=WAITING, core_id="LIP-1_C-2", core_label="C-2", image_path="data/cores/LIP-1/C-2/6E10.png")
    pd.concat([tiles, extra]).to_csv(analysis / "cohort_whole_core_tiles.csv", index=False)
    images = pd.read_csv(analysis / "core_images.csv", dtype=str)
    pd.concat([images, images[images.stain == "6E10"].assign(core_id="LIP-1_C-2", image_path="data/cores/LIP-1/C-2/6E10.png")]).to_csv(
        analysis / "core_images.csv", index=False)
    (root / "data/cores/LIP-1/C-2").mkdir(parents=True)
    shutil.copy(root / "data/cores/LIP-1/B-2/6E10.png", root / "data/cores/LIP-1/C-2/6E10.png")
    state.reset()
    model = state.get_state().project.model("amyloid")
    model.parent.mkdir(parents=True, exist_ok=True)
    model.write_bytes(b"model")
    admin = TestClient(create_app(), client=("127.0.0.1", 50000))
    assert admin.post("/api/session/admin", json={"password": "admin"}).status_code == 200
    yield admin
    state.reset()


def _prepare(tile_id: str) -> None:
    from stainid.labels import pool

    project = state.get_state().project
    folder = pool.folder(project, "6E10")
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"x": 100.0, "y": 200.0, "radius_px": 20.0, "ai_value": "compact plaque", "ai_probability": 0.9, "tma": "1", "complete": True,
                   "f1": 1.0}]).to_csv(folder / f"{tile_id}.csv", index=False)


def test_picker_shows_cores_that_are_not_ready_with_the_step_that_comes_first(client):
    page = client.get("/api/labels/picker", params={"stain": "6E10"}).json()
    cores = {c["core_id"]: c for c in page["cores"]}
    assert list(cores) == ["LIP-1_B-2", "LIP-1_C-2"] and page["total_fields"] == 2 and page["ready_fields"] == 1
    assert cores["LIP-1_B-2"]["blocked"] is None and cores["LIP-1_B-2"]["fields"][0]["blocked"] is None
    waiting = cores["LIP-1_C-2"]["blocked"]
    assert waiting["step"] == "fields" and waiting["link"] == "/workflow#fields" and "has not looked at this core" in waiting["reason"]
    assert cores["LIP-1_B-2"]["width"] == 1200 and {"x_px", "y_px", "width_px", "height_px"} <= set(cores["LIP-1_B-2"]["fields"][0])
    assert page["facets"]["tmas"] == ["1"]
    neun = client.get("/api/labels/picker", params={"stain": "NeuN"}).json()  # no NeuN model: its field says where to get one
    assert neun["ready_fields"] == 0 and neun["cores"][0]["blocked"]["link"] == "/models"
    assert client.get(f"/api/labels/fields/{READY}/thumb.jpg").headers["content-type"] == "image/jpeg"


def test_sessions_take_only_ready_fields_and_list_where_each_stands(client):
    auto = client.post("/api/labels/sessions", json={"name": "auto", "stain": "6E10", "n_fields": 5}).json()
    assert auto["fields"] == [READY]  # the core detection has not reached is left out, the session is still made
    refused = client.post("/api/labels/sessions", json={"name": "by_hand", "stain": "6E10", "tile_ids": [WAITING]})
    assert refused.status_code == 400 and "Detect plaques and tau" in refused.json()["detail"]
    assert client.post("/api/labels/sessions/auto/fields", json={"tile_ids": [WAITING]}).status_code == 400

    fields = client.get("/api/labels/sessions/auto/fields").json()
    assert [(f["tile_id"], f["state"], f["core_label"]) for f in fields] == [(READY, "preparing", "B-2")]
    _prepare(READY)
    client.post("/api/labels/sessions/auto/open", json={"open": True})
    ann = annotator(client, "Ann", "auto")
    assert client.get("/api/labels/sessions/auto/fields").json()[0]["state"] == "not_started"
    ann.post("/api/annotate/auto/labels", json={"records": [{"task": "class", "tile_id": READY, "x": 100.0, "y": 200.0, "value": "diffuse plaque"}]})
    assert client.get("/api/labels/sessions/auto/fields").json()[0] | {"state": "in_progress", "working": ["Ann"], "done_by": []} == \
        client.get("/api/labels/sessions/auto/fields").json()[0]
    mine = ann.get("/api/annotate/auto").json()["fields"][0]  # the person labelling: by TMA and core, with the same state
    assert (mine["tma"], mine["core"], mine["state"], mine["working"]) == ("1", "B-2", "in_progress", ["Ann"])
    ann.post(f"/api/annotate/auto/fields/{READY}/finish")
    assert client.get("/api/labels/sessions/auto/fields").json()[0]["done_by"] == ["Ann"]
    assert ann.get("/api/annotate/auto").json()["fields"][0]["done_by"] == ["Ann"]
    assert ann.get(f"/api/annotate/auto/fields/{READY}/thumb.jpg").status_code == 200
    core = ann.get(f"/api/annotate/auto/fields/{READY}/core").json()  # the smaller zoom levels beside the field
    assert core["width"] > 0 and core["height"] > 0 and core["field"][0] > 0 and core["field"][1] > 0
    assert 0 <= core["origin"][0] < core["width"] and 0 <= core["origin"][1] < core["height"]
    assert ann.get(core["thumb"]).headers["content-type"] == "image/jpeg"

    removed = client.post("/api/labels/sessions/auto/fields/remove", json={"tile_ids": [READY]}).json()
    assert removed["fields"] == [] and client.get("/api/labels/sessions/auto/fields").json() == []
    plaques = next(s for s in client.get("/api/labels/overview").json()["stains"] if s["stain"] == "6E10")
    assert plaques["objects_labelled"] == 1  # labels on a removed field are kept
    assert client.post("/api/labels/sessions/auto/fields", json={"tile_ids": [READY]}).json()["fields"] == [READY]


def test_every_label_belongs_to_a_session_and_each_can_be_examined(client):
    from stainid.labels import store

    root = state.get_state().project.output("reviews")
    store.append(root, [{"id": f"class:{READY}:100:200", "source": "import: reference_6e10", "stain": "6E10", "task": "class", "tile_id": READY,
                         "x": 100.0, "y": 200.0, "value": "compact plaque"}])
    client.post("/api/labels/sessions", json={"name": "round_1", "title": "Round 1", "stain": "6E10", "tile_ids": [READY], "open": True})
    ann = annotator(client, "Ann", "round_1")
    ann.post("/api/annotate/round_1/labels", json={"records": [{"task": "class", "tile_id": READY, "x": 100.0, "y": 200.0, "value": "diffuse plaque"}]})
    sets = client.get("/api/labels/sets", params={"stain": "6E10"}).json()
    assert [(s["name"], s["title"], s["classes"]) for s in sets] == [("round_1", "Round 1", {"plaque · diffuse": 1}),
                                                                       ("reference", "AI reference labels", {"plaque · compact (dense-core)": 1})]
    assert client.get("/api/labels/overview").json()["reference"]["sets"] == ["import: reference_6e10"]
    seen = client.get("/api/labels/sessions/reference/objects").json()
    assert seen["reference"] and seen["stain"] == "6E10" and seen["classes"] == {"plaque · compact (dense-core)": 1}
    found = seen["objects"][0]
    assert found["tile_id"] == READY and found["frame"] == "field" and found["value"] == "plaque · compact (dense-core)" and found["ai_value"] == "plaque · compact (dense-core)"
    assert client.get("/api/labels/sessions/round_1/objects", params={"values": "plaque · compact (dense-core)"}).json()["objects"] == []
    assert client.get("/api/labels/sessions/round_1/objects").json()["objects"][0]["labels"][0]["source"] == "Ann"
    assert client.post("/api/labels/sessions", json={"name": "reference", "stain": "6E10", "n_fields": 1}).status_code == 400
