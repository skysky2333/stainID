from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from stainid.api import state  # noqa: E402
from stainid.api.app import create_app  # noqa: E402


@pytest.fixture()
def client(synthetic_project, monkeypatch, tmp_path_factory):
    monkeypatch.chdir(synthetic_project)
    monkeypatch.setattr(state, "SETTINGS", tmp_path_factory.mktemp("home") / "app.json")
    state._STATE = None
    yield TestClient(create_app())
    state._STATE = None


def test_summary_and_cores(client):
    summary = client.get("/api/summary").json()
    assert summary["cores"] == 1 and summary["donors"] == 1
    assert summary["status"]["6E10"]["detection"]["total"] == 1
    cores = client.get("/api/cores?tma=1").json()
    assert cores[0]["core_id"] == "LIP-1_B-2" and cores[0]["stains"] == ["6E10", "AT8", "NeuN"]


def test_images_and_objects(client):
    thumb = client.get("/api/images/cores/LIP-1_B-2/NeuN.jpg?size=128")
    assert thumb.status_code == 200 and thumb.headers["content-type"] == "image/jpeg"
    field = client.get("/api/images/fields/LIP-1_B-2_6E10_S01.jpg")
    assert field.status_code == 200
    layer = client.get("/api/images/fields/LIP-1_B-2_6E10_S01/dab.png")
    assert layer.status_code == 200
    objects = client.get("/api/fields/LIP-1_B-2_6E10_S01/objects").json()["objects"]
    assert len(objects) == 8 and {o["model_class"] for o in objects} == {"compact", "diffuse"}


def test_review_roundtrip(client):
    created = client.post("/api/reviews", json={"name": "plaques_r1", "stain": "6E10", "n": 4, "per_group": False})
    assert created.status_code == 200, created.text
    name = created.json()["name"]
    data = client.get(f"/api/reviews/{name}/items").json()
    assert len(data["items"]) == 4 and "disease_group" not in data["items"][0]
    first = data["items"][0]["review_id"]
    assert client.post(f"/api/reviews/{name}/labels", json={"review_id": first, "label": "compact_plaque"}).status_code == 200
    assert client.get(f"/api/reviews/{name}/image/{first}.jpg").status_code == 200
    listed = {s["name"]: s for s in client.get("/api/reviews").json()}
    assert listed[name]["labelled"] == 1


def test_tables(client):
    names = [t["name"] for t in client.get("/api/tables").json()]
    assert "slide_dab_calibration.csv" in names
    rows = client.get("/api/tables/slide_dab_calibration.csv/rows").json()
    assert rows["total"] == 3


def test_steps_describe_needs_and_progress(client):
    steps = {s["id"]: s for s in client.get("/api/steps").json()}
    assert steps["select"]["progress"] == {"state": "done", "done": 3, "total": 3, "unit": "fields", "note": "1 cores", "outdated": False}
    assert steps["fields"]["progress"]["total"] == 2
    assert [n["id"] for n in steps["fields"]["needs"]][:2] == ["tile_manifest", "calibration"]
    assert steps["slides"]["command"] is None and steps["dearray"]["options"][0]["key"] == "redo"


def test_settings_save_and_reopen(client, synthetic_project):
    info = client.put("/api/project/config", json={"changes": {"tma": {"prefix": "LIP-"}, "groups": [{"code": "AD", "label": "Alzheimer"}]}}).json()
    assert info["tma_prefix"] == "LIP-" and info["groups"][0] == {"code": "AD", "label": "Alzheimer"}
    assert "prefix: LIP-" in (synthetic_project / "stainid.yaml").read_text()


def test_create_open_project_and_templates(client, tmp_path):
    new = tmp_path / "study2"
    created = client.post("/api/project/create", json={"path": str(new), "name": "Second study"}).json()
    assert created["name"] == "Second study" and created["configured"]
    assert client.post("/api/project/open", json={"path": str(tmp_path / "missing")}).status_code == 404
    listing = client.get("/api/fs", params={"path": str(new)}).json()
    assert listing["path"] == str(new) and "stainid.yaml" in listing["files"]
    assert client.get("/api/templates/tma_layout.csv").text.startswith("tma,core_label,donor_id")


def test_upload_tma_map_validates(client, synthetic_project):
    bad = client.post("/api/upload/tma_layout", files={"file": ("map.csv", b"tma,core_label\n1,B-2\n")})
    assert bad.status_code == 400 and "missing columns" in bad.json()["detail"]
    good = b"tma,core_label,donor_id,region,disease_group\n1,B-2,0001,frontal,AD\n"
    saved = client.post("/api/upload/tma_layout", files={"file": ("map.csv", good)})
    assert saved.status_code == 200 and saved.json()["rows"] == 1


def test_download_is_limited_to_project(client):
    assert client.get("/api/download", params={"path": "data/analysis/slide_dab_calibration.csv"}).status_code == 200
    assert client.get("/api/download", params={"path": "../../etc/hosts"}).status_code == 403


def test_settings_can_change_while_steps_run(client, synthetic_project, tmp_path, monkeypatch):
    runner = state.get_state().jobs
    monkeypatch.setattr(runner, "_start", lambda job: None)
    job = runner.submit("summarize", {})
    job.status = "running"
    assert client.put("/api/project/config", json={"changes": {"name": "Renamed"}}).status_code == 200
    assert state.get_state().jobs is runner
    assert client.post("/api/project/create", json={"path": str(tmp_path / "other")}).status_code == 409
    job.status = "finished"


def test_tma_map_upload_fills_groups(client):
    good = b"tma,core_label,donor_id,region,disease_group\n1,B-2,0001,frontal,AD\n1,C-2,0002,frontal,CT\n"
    assert client.post("/api/upload/tma_layout", files={"file": ("map.csv", good)}).status_code == 200
    assert [g["code"] for g in client.get("/api/project").json()["groups"]] == ["AD", "CT"]


def test_missing_inputs_say_how_to_get_them(client, tmp_path):
    client.post("/api/project/create", json={"path": str(tmp_path / "fresh")})
    needs = {n["id"]: n for s in client.get("/api/steps").json() for n in s["needs"]}
    assert needs["core_manifest"]["made_by"]["title"] == "Find cores"
    assert needs["model_cellpose"]["download"] == "cellpose" and needs["model_cellpose"]["made_by"] is None
    assert needs["tma_layout"]["template"] == "tma_layout" and "core_label" in needs["tma_layout"]["format"]
    assert needs["model_neun"]["setting"] == "models.neun" and needs["slides_table"]["made_by"]["id"] == "slides"
