# ruff: noqa: F811  (the rounds fixture is shared from test_annotation_rounds)
"""The grey / white matter algorithm (v2/white_matter.py): white matter is tissue with almost no NeuN+ neurons next to
grey matter that has them, confirmed on the NeuN image by its cells; a NeuN stain too pale to read is "undetermined";
a band along the tissue's edge (layer I) is no white matter; the NeuN area is carried to the other stains through the
registration; the step writes its own masks (never detection's) and is redone only when an input changes; the views
draw it over the excluded areas without counting it out; the "Grey / white matter" round starts from it on NeuN."""
from __future__ import annotations

import json

import cv2
import numpy as np
import pandas as pd
import pytest

from stainid.v2 import white_matter as wm

CELL_UM = 17.5


def _grid(white_rows: int = 40, size: int = 100, neurons_per_cell: float = 2.0):
    """A square core of grey matter (`neurons_per_cell` NeuN+ neurons per cell, about 6,500/mm²) whose last
    `white_rows` rows have none."""
    tissue = np.ones((size, size), np.float32)
    rng = np.random.default_rng(1)
    neurons = rng.poisson(neurons_per_cell, (size, size)).astype(np.float32)
    neurons[size - white_rows:] = 0
    return tissue, neurons


def test_white_matter_is_where_the_neurons_stop():
    tissue, neurons = _grid()
    white, info = wm.detect(tissue, neurons, CELL_UM)
    assert info["status"] == "white" and info["reference_per_mm2"] > 1000
    truth = np.zeros_like(white)
    truth[60:] = True
    assert (white & truth).sum() / (white | truth).sum() > 0.9  # the edge lies at the drop in neuron density
    assert 0.35 < info["white_share"] < 0.45


def test_a_core_without_white_matter_is_grey_and_a_pale_neun_core_is_undetermined():
    tissue, neurons = _grid(white_rows=0)
    white, info = wm.detect(tissue, neurons, CELL_UM)
    assert info["status"] == "grey" and not white.any()
    white, info = wm.detect(tissue, np.zeros_like(neurons), CELL_UM)  # the NeuN stain failed: no neuron anywhere
    assert info["status"] == "undetermined" and "NeuN" in info["reason"] and not white.any()


def test_a_thin_band_along_the_edge_is_no_white_matter():
    """Layer I under the pia: a band a few hundred µm wide, poor in neurons, along the tissue's edge."""
    tissue, neurons = _grid(white_rows=0)
    neurons[:, :15] = 0  # 260 µm along the left edge
    white, info = wm.detect(tissue, neurons, CELL_UM)
    assert info["status"] == "grey" and not white.any()


def test_the_neun_image_confirms_each_part_by_its_cells_and_stain():
    tissue, neurons = _grid()
    more_cells = np.ones_like(tissue)
    more_cells[60:] = 1.5  # white matter: more cells (oligodendrocytes)
    dab = np.ones_like(tissue)
    white, info = wm.detect(tissue, neurons, CELL_UM, look=lambda: (more_cells, dab))
    assert info["status"] == "white" and info["parts"] and info["parts"][0]["kept"]
    same = np.ones_like(tissue)
    fewer = same.copy()
    fewer[60:] = 0.9  # grey matter only poor in NeuN+ neurons: no more cells, as much NeuN stain
    white, info = wm.detect(tissue, neurons, CELL_UM, look=lambda: (fewer, same))
    assert info["status"] == "grey" and not white.any() and not info["parts"][0]["kept"]
    pale = np.ones_like(tissue)
    pale[60:] = 0.3  # no NeuN-stained neuropil: kept on the stain alone
    white, info = wm.detect(tissue, neurons, CELL_UM, look=lambda: (fewer, pale))
    assert info["status"] == "white"


def test_cells_are_counted_on_the_hematoxylin():
    bgr = np.full((256, 256, 3), 235, np.uint8)
    rng = np.random.default_rng(2)
    for x, y in rng.integers(10, 246, size=(60, 2)):
        cv2.circle(bgr, (int(x), int(y)), 3, (150, 60, 60), -1)  # small blue nuclei (BGR)
    cells, dab = wm.look_at(bgr, 1.1, (4, 4), 64)
    assert 45 <= cells.sum() <= 62 and dab.max() < 0.1


def test_the_area_is_carried_with_the_registration_matrix():
    white = np.zeros((50, 50), bool)
    white[10:20, 10:20] = True
    analysed = np.ones((50, 50), bool)
    moved = wm.carry(white, np.array([[1, 0, 80], [0, 1, 0], [0, 0, 1]], float), analysed)  # 80 core px = 10 cells right
    assert moved[10:20, 20:30].all() and moved.sum() == 100
    analysed[:, 25:] = False  # only inside the other image's analysed tissue
    assert wm.carry(white, np.array([[1, 0, 80], [0, 1, 0], [0, 0, 1]], float), analysed).sum() == 50


@pytest.fixture()
def wm_project(synthetic_project, monkeypatch):
    """The synthetic core with stage-2 masks of its three stains, NeuN calls (neurons in the top 2/3 only) and the
    registration table (6E10 80 px left of NeuN); the algorithm's sizes scaled to the tiny core."""
    from stainid import candidates, exclusions
    from stainid.project import load_project
    from stainid.v2.proposals import BITS

    project = load_project(synthetic_project)
    for stain in ("NeuN", "6E10", "AT8"):
        path = exclusions.mask_path(project, "LIP-1_B-2", stain)
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), np.full((150, 150), BITS["tile"] | BITS["tissue"], np.uint8))
    rng = np.random.default_rng(3)
    xy = rng.uniform(0, 1200, size=(3000, 2))
    xy = xy[xy[:, 1] < 800]
    frame = pd.DataFrame({"candidate_id": [f"NeuN:LIP-1_B-2:{x:.0f}:{y:.0f}:{i}" for i, (x, y) in enumerate(xy)],
                          "x_core": xy[:, 0], "y_core": xy[:, 1], "decision": "accepted"})
    target = candidates.folder(project, "NeuN", "candidates") / "LIP-1_B-2.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(target)
    columns = ["core_id", "moving_stain", "reference_stain", "status", "registered_tissue_dice"] + [f"native_m{i}{j}" for i in range(2) for j in range(3)]
    path = project.input("registration")
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([["LIP-1_B-2", "6E10", "NeuN", "pass", 0.95, 1, 0, 80, 0, 1, 0], ["LIP-1_B-2", "AT8", "NeuN", "pass", 0.9, 1, 0, 0, 0, 1, 0]],
                 columns=columns).to_csv(path, index=False)
    small = {"sigma_um": 20.0, "open_um": 15.0, "min_area_mm2": 0.001, "fill_holes_mm2": 0.0001, "min_depth_um": 15.0, "min_reference_per_mm2": 10.0,
             "min_tissue_mm2": 0.01}
    for key, value in small.items():
        monkeypatch.setitem(wm.DEFAULTS, key, value)
    monkeypatch.setattr(wm, "_look", lambda project, core_id, shape: (np.ones(shape, np.float32), np.ones(shape, np.float32) * 0.1))
    return project


def test_the_step_writes_its_own_masks_and_redoes_only_what_changed(wm_project):
    from stainid import exclusions

    project = wm_project
    before = {s: exclusions.digest(project, "LIP-1_B-2", s) for s in ("NeuN", "6E10", "AT8")}
    found = wm.run(project, log=lambda text: None)
    assert set(found.stain) == {"NeuN", "6E10", "AT8"} and set(found.status) == {"white"}
    neun, other = wm.read(project, "LIP-1_B-2", "NeuN"), wm.read(project, "LIP-1_B-2", "6E10")
    assert neun[120:, :].mean() > 0.8 and neun[:80, :].mean() < 0.05  # the bottom third, where there are no neurons
    assert other[120:, 5:].mean() > 0.8 and other[120:, -10:].sum() < neun[120:, -10:].sum()  # 10 cells further left on 6E10
    assert {s: exclusions.digest(project, "LIP-1_B-2", s) for s in ("NeuN", "6E10", "AT8")} == before  # detection's masks untouched
    assert wm.todo(project) == []  # up to date
    assert wm.status_of(project, "LIP-1_B-2", "NeuN")["status"] == "white"
    table = pd.read_csv(wm.summary_path(project))
    assert len(table) == 3 and table.white_share.between(0.2, 0.5).all()
    project.config.setdefault("v2", {})["white_matter"] = {"seed": 0.05}  # a setting changed: the core is redone
    assert [core for core, _, _ in wm.todo(project)] == ["LIP-1_B-2"]
    project.config["v2"]["white_matter"] = {"analysis": "exclude"}  # not decided yet: refused, not ignored
    with pytest.raises(ValueError, match="re-run"):
        wm.settings(project)


def test_the_views_draw_it_over_the_excluded_areas_and_still_count_it(wm_project):
    from stainid import exclusions
    from stainid.api import store_view

    project = wm_project
    exclusions.forget()
    plain = exclusions.view(project, "LIP-1_B-2", "NeuN")
    assert plain.white is None and next(k for k in store_view.excluded_info(project, "LIP-1_B-2", "NeuN")["kinds"]
                                        if k["id"] == "white_matter")["fraction"] == 0
    wm.run(project, log=lambda text: None)
    exclusions.forget()
    shown = exclusions.view(project, "LIP-1_B-2", "NeuN")
    assert shown.white is not None and shown.white.any() and shown.version != plain.version
    assert (shown.codes == plain.codes).all()  # counted as before: the analysis' layer is unchanged
    info = store_view.excluded_info(project, "LIP-1_B-2", "NeuN")
    kind = next(k for k in info["kinds"] if k["id"] == "white_matter")
    assert kind["counted"] and 0.2 < kind["fraction"] < 0.5
    picture = cv2.imdecode(np.frombuffer(store_view.excluded_png(project, "LIP-1_B-2", "NeuN"), np.uint8), cv2.IMREAD_UNCHANGED)
    assert picture[140, :, 3].max() > 0 and picture[20, :, 3].max() == 0  # drawn in the white matter only


def test_people_start_from_the_proposal_on_neun_and_are_scored_against_it(wm_project, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from test_annotation_rounds import annotator

    from stainid.api import state
    from stainid.api.app import create_app
    from stainid.labels import store

    monkeypatch.chdir(wm_project.root)
    state.reset()
    admin = TestClient(create_app(), client=("127.0.0.1", 50000))
    assert admin.post("/api/session/admin", json={"password": "admin"}).status_code == 200
    ann = annotator(admin, "Ann", every=True)
    name = admin.post("/api/labels/sessions", json={"stain": "all", "kind": "matter", "open": True, "seed": 1}).json()["name"]
    core = "LIP-1_B-2"
    assert ann.get(f"/api/annotate/{name}/matter/cores/{core}/NeuN/area").json()["proposal"] is None  # before the step: "new"
    wm.run(wm_project, log=lambda text: None)
    area = ann.get(f"/api/annotate/{name}/matter/cores/{core}/NeuN/area").json()
    assert area["from"] == "proposed" and area["proposal"]["status"] == "white" and area["kinds"]["white"].startswith("data:image/png")
    ann.get(f"/api/annotate/{name}/matter/cores/{core}")
    saved = ann.post(f"/api/annotate/{name}/matter/cores/{core}/NeuN/save", json={"accepted": True})  # the proposal as it is
    assert saved.status_code == 200, saved.text
    labels = store.current(wm_project.output("reviews"))
    row = labels[(labels.task == "matter") & (labels.stain == "NeuN")].iloc[-1]
    assert row.meta["accepted"] and row.meta["proposal"]["status"] == "white" and len(row.points) >= 1
    scores = admin.get("/api/labels/matter").json()["algorithm"]
    assert scores["per_stain"]["NeuN"]["images"] == 1 and scores["per_stain"]["NeuN"]["mean_iou"] > 0.9
    state.reset()


def test_the_human_labels_are_not_read_by_the_step(wm_project):
    """Only the algorithm's inputs: a saved white-matter label does not change what the step finds."""
    wm.run(wm_project, log=lambda text: None)
    first = json.loads(wm.manifest_path(wm_project, "LIP-1_B-2").read_text())["digest"]
    assert "labels" not in json.dumps(json.loads(wm.manifest_path(wm_project, "LIP-1_B-2").read_text())["inputs"])
    assert wm.todo(wm_project) == [] and first
