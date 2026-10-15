"""Looking at labels: every labelled object as the crop the model sees, centred on the object whatever frame its label
uses (field picture with margin, tile-frame imports, pilot pictures, whole-core labels named by a core id)."""
from __future__ import annotations

import time

import cv2
import numpy as np
import pandas as pd
import pytest

from stainid.labels import frames

TILE = "LIP-1_B-2_6E10_S01"  # x_px = y_px = 300, 512 px, in a 1200 px core image; the field margin is 128 px
DOT = (500, 520)  # a bright green dot on the core image, where every test label points
PILOT, PILOT_DOT = "M001_6E10", (100, 120)


def test_tile_frame_imports_move_into_the_field_picture_with_the_margin_the_core_edge_leaves():
    tiles = pd.DataFrame([{"tile_id": "a", "x_px": 300, "y_px": 300}, {"tile_id": "edge", "x_px": 0, "y_px": 2048}])
    labels = pd.DataFrame([
        {"source": "import: CNN outlier adjudication", "tile_id": "a", "x": 200.0, "y": 220.0},
        {"source": "import: CNN outlier adjudication", "tile_id": "edge", "x": 10.0, "y": 10.0},
        {"source": "import: round3_6e10", "tile_id": "a", "x": 328.0, "y": 348.0},
        {"source": "import: CNN outlier adjudication", "tile_id": "LIP-1_B-2", "x": 500.0, "y": 520.0},  # a core id: core pixels
    ])
    moved = frames.in_field_frame(labels, tiles, 128)
    assert moved[["x", "y"]].values.tolist() == [[328.0, 348.0], [10.0, 138.0], [328.0, 348.0], [500.0, 520.0]]
    assert labels.x.iloc[0] == 200.0  # the labels given are not changed
    assert frames.core_position({"x_px": 300, "y_px": 300}, 328, 348, 128) == DOT
    assert frames.core_position({"x_px": 0, "y_px": 2048}, 10, 138, 128) == (10, 2058)
    assert frames.core_position(None, *DOT, 128) == DOT


pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from stainid.api import state  # noqa: E402
from stainid.api.app import create_app  # noqa: E402


@pytest.fixture()
def client(synthetic_project, monkeypatch):
    monkeypatch.chdir(synthetic_project)
    core = synthetic_project / "data/cores/LIP-1/B-2/6E10.png"
    image = cv2.imread(str(core))
    cv2.circle(image, DOT, 6, (0, 255, 0), -1)
    cv2.imwrite(str(core), image)
    pilot = np.full((512, 512, 3), 230, np.uint8)
    cv2.circle(pilot, PILOT_DOT, 6, (0, 255, 0), -1)
    (synthetic_project / "data/annotations/fields").mkdir(parents=True)
    cv2.imwrite(str(synthetic_project / f"data/annotations/fields/{PILOT}.png"), pilot)
    state.reset()
    admin = TestClient(create_app(), client=("127.0.0.1", 50000))
    assert admin.post("/api/session/admin", json={"password": "admin"}).status_code == 200
    yield admin
    state.reset()


def _label(source: str, tile_id: str, x: float, y: float, value: str) -> dict:
    return {"id": f"class:{tile_id}:{round(x)}:{round(y)}", "source": source, "stain": "6E10", "task": "class", "tile_id": tile_id,
            "x": x, "y": y, "value": value}


def _objects(client, **params) -> dict:
    """The listing, asked again until every crop on it is ready (core images are read in the background)."""
    for _ in range(100):
        page = client.get("/api/labels/sessions/reference/objects", params={"stain": "6E10", **params}).json()
        if all(o["crop"]["state"] == "ready" for o in page["objects"]):
            return page
        time.sleep(0.1)
    raise AssertionError(f"crops not ready: {[o['crop'] for o in page['objects']]}")


def test_every_label_source_gets_a_crop_centred_on_its_object(client):
    from stainid.labels import store

    root = state.get_state().project.output("reviews")
    store.append(root, [
        _label("import: round3_6e10", TILE, 328.0, 348.0, "compact plaque"),  # field picture, margin included
        _label("import: CNN outlier adjudication", TILE, 200.0, 220.0, "plaque"),  # same object, measured from the tile corner
        _label("import: reference_6e10", "LIP-1_B-2", *map(float, DOT), "diffuse plaque"),  # a core id: whole-core pixels
        _label("import: reference_6e10", PILOT, *map(float, PILOT_DOT), "not a plaque"),  # a pilot field of an earlier study
        _label("import: round3_6e10", TILE, 100.0, 200.0, "not a plaque"),  # the AI found a compact plaque here
    ])
    page = _objects(client)
    assert page["all"] == 4 and page["total"] == 4  # the two labels of the dot's field object are one object
    by = {(o["tile_id"], round(o["x"]), round(o["y"])): o for o in page["objects"]}
    field = by[(TILE, 328, 348)]
    # both say plaque; the label without a type does not count against the one that gives it
    assert field["frame"] == "field" and field["value"] == "plaque · compact (dense-core)" and {lab["value"] for lab in field["labels"]} == {
        "plaque · compact (dense-core)", "plaque (type not recorded)"}
    assert (field["identity"], field["subtype"]) == ("plaque", "compact")
    assert (field["core_x"], field["core_y"]) == DOT and field["core_label"] == "B-2" and field["tma"] == "1"
    core = by[("LIP-1_B-2", *DOT)]
    assert core["frame"] == "core" and core["core_label"] == "B-2" and (core["core_x"], core["core_y"]) == DOT
    assert by[(PILOT, *PILOT_DOT)]["frame"] == "pilot"
    missed = by[(TILE, 100, 200)]
    assert missed["ai_value"] == "plaque · compact (dense-core)" and missed["ai"] == "disagrees" and missed["radius_px"] > 10
    assert page["classes"] == {"plaque · compact (dense-core)": 1, "plaque · diffuse": 1, "not a plaque": 2} and page["ai"]["disagrees"] == 2  # the AI found no plaque at the field object
    assert page["vocabulary"][:5] == ["plaque (type not recorded)", "plaque · compact (dense-core)", "plaque · diffuse", "plaque · cored-neuritic",
                                      "plaque · small, too small to type"]
    assert [g["name"] for g in page["groups"]][:2] == ["plaque", "CAA / vascular amyloid"] and page["groups"][-1]["name"] == "people disagree"

    size = page["crop"]["size_px"]
    for o in (field, core, by[(PILOT, *PILOT_DOT)]):
        response = client.get(f"/api/labels/crops/6E10/{o['tile_id']}.jpg", params={"x": o["x"], "y": o["y"]})
        assert response.status_code == 200 and response.headers["content-type"] == "image/jpeg", o
        crop = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
        assert crop.shape[:2] == (size, size)
        b, g, r = crop[size // 2, size // 2].astype(int)
        assert g > 200 and r < 80 and b < 80, (o["tile_id"], (b, g, r))  # the green dot is in the middle

    only = client.get("/api/labels/sessions/reference/objects", params={"stain": "6E10", "values": "not a plaque", "ai": "disagrees"}).json()
    assert [(o["tile_id"], round(o["x"])) for o in only["objects"]] == [(TILE, 100)]
    assert client.get("/api/labels/sessions/reference/objects", params={"stain": "6E10", "values": "-"}).json()["objects"] == []
    assert client.get("/api/labels/crops/6E10/nowhere.jpg", params={"x": 1, "y": 1}).status_code == 404
