"""The Cores grid and a core's own page say the same about detection: the store the page shows answers first, and leftovers
of an older run do not make a core look part-way."""
from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from stainid import candidates as store  # noqa: E402
from stainid.api import state  # noqa: E402
from stainid.api.app import create_app  # noqa: E402
from stainid.project import load_project  # noqa: E402

CORE = "LIP-1_B-2"


@pytest.fixture()
def client(synthetic_project, monkeypatch):
    monkeypatch.chdir(synthetic_project)
    state.reset()
    admin = TestClient(create_app(), client=("127.0.0.1", 50000))
    assert admin.post("/api/session/admin", json={"password": "admin"}).status_code == 200
    yield admin
    state.reset()


def _two_tiles(root) -> None:
    """NeuN with two tiles, of which an older run finished one: the old outputs say part-way."""
    path = root / "data" / "analysis" / "cohort_whole_core_tiles.csv"
    tiles = pd.read_csv(path)
    second = tiles[tiles.stain == "NeuN"].assign(tile_id=f"{CORE}_NeuN_W0002", x_px=812)
    pd.concat([tiles, second]).to_csv(path, index=False)
    folder = root / "data" / "analysis" / "cohort_neun" / "tiles"
    folder.mkdir(parents=True)
    pd.DataFrame(columns=["centroid_x_px", "centroid_y_px"]).to_csv(folder / f"{CORE}_NeuN_S01_objects.csv", index=False)


def _in_the_store(root, stain: str) -> None:
    frame = pd.DataFrame({"candidate_id": [f"{stain}:{CORE}:10:10"], "x_core": [10.0], "y_core": [10.0], "radius_px": [5.0], "area_um2": [40.0],
                          "probability": [0.9], "decision": ["accepted"], "reason": ["accepted"], "class": ["neuron"], "subtype": [""], "m_from_ring": [0],
                          "outline": ["0 0 20 0 20 20"], "sam_outline": [""], "sam_status": [""], "sam_area_um2": [float("nan")]})
    store.write_table(store.root(load_project(root)) / "candidates" / stain / f"{CORE}.csv.gz", frame)


def _grid(client, stain: str) -> dict:
    grid = client.get(f"/api/tmas/1/grid?stain={stain}&prepare=false").json()
    return next(c for c in grid["cells"] if c["core_id"] == CORE) | {"summary": grid["detection"]}


def test_the_grid_and_the_core_s_page_agree_when_the_store_has_the_core(client, synthetic_project):
    _two_tiles(synthetic_project)
    state.reset()
    assert _grid(client, "NeuN")["detection"] == "partial"  # no store: the older outputs, on the grid and on the page
    page = client.get(f"/api/cores/{CORE}/NeuN/objects").json()
    assert (page["status"], page["tiles_done"], page["tiles_total"]) == ("partial", 1, 2)

    _in_the_store(synthetic_project, "NeuN")
    state.reset()
    cell = _grid(client, "NeuN")
    assert cell["detection"] == "done" and cell["summary"] == {"done": 1, "total": 1}
    page = client.get(f"/api/cores/{CORE}/NeuN/objects").json()
    assert (page["status"], page["source"], page["tiles_done"], page["tiles_total"]) == ("done", "v2", 2, 2)


def test_the_store_answers_for_every_stain_and_a_core_without_it_still_reads_as_not_run(client, synthetic_project):
    assert _grid(client, "AT8")["detection"] == "not_run"
    assert client.get(f"/api/cores/{CORE}/AT8/objects").json()["status"] == "not_run"
    _in_the_store(synthetic_project, "AT8")
    assert _grid(client, "AT8")["detection"] == "done"
    assert client.get(f"/api/cores/{CORE}/AT8/objects").json()["status"] == "done"


def test_the_setting_that_shows_the_older_outputs_keeps_the_grid_on_them_too(client, synthetic_project):
    _in_the_store(synthetic_project, "AT8")
    (synthetic_project / "stainid.yaml").write_text("name: synthetic\nv2:\n  viewer: current\n")
    state.reset()
    assert _grid(client, "AT8")["detection"] == "not_run"
    assert client.get(f"/api/cores/{CORE}/AT8/objects").json()["status"] == "not_run"
