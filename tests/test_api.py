from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from stainid.api import state  # noqa: E402
from stainid.api.app import create_app  # noqa: E402


@pytest.fixture()
def client(synthetic_project, monkeypatch):
    monkeypatch.chdir(synthetic_project)
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
