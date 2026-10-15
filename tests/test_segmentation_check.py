"""The segmentation check: objects people outlined against the candidates, on synthetic fields with known objects and
deliberately missed, merged, split, badly outlined and rejected candidates."""
from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
import pytest

from stainid.labels import outline
from stainid.segmentation import scoring

SHAPE = (600, 600)


def circle(x: float, y: float, r: float, n: int = 48) -> list[list[float]]:
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.c_[x + r * np.cos(a), y + r * np.sin(a)].round(1).tolist()


def field() -> tuple[list[scoring.Truth], scoring.Candidates, list[int], dict[str, int]]:
    """Seven objects and the candidates of each case; returns the objects, the candidates, which candidates the
    pipeline counted and the column of each named candidate."""
    truths = {"found": circle(80, 80, 20), "missed": circle(200, 80, 20), "merged_a": circle(320, 80, 15), "merged_b": circle(352, 80, 15),
              "split": circle(150, 300, 45), "poor": circle(400, 300, 20), "rejected": circle(500, 80, 20), "unsure": circle(500, 500, 20)}
    layer = np.zeros(SHAPE, np.uint16)
    names: dict[str, int] = {}

    def add(name: str, draw) -> None:
        names[name] = len(names) + 1
        mask = np.zeros(SHAPE, np.uint8)
        draw(mask)
        layer[mask.astype(bool) & (layer == 0)] = names[name]

    add("found", lambda m: cv2.circle(m, (80, 80), 20, 1, -1))
    add("merged", lambda m: cv2.ellipse(m, (336, 80), (34, 17), 0, 0, 360, 1, -1))
    for i, (x0, x1) in enumerate(((105, 135), (135, 165), (165, 196))):  # the big object cut into three upright strips
        add(f"split_{i}", lambda m, x0=x0, x1=x1: cv2.rectangle(m, (x0, 255), (x1 - 1, 345), 1, -1))
    add("poor", lambda m: cv2.circle(m, (425, 300), 20, 1, -1))  # shifted by more than its radius: IoU about 0.2
    add("rejected", lambda m: cv2.circle(m, (500, 80), 20, 1, -1))
    add("nothing", lambda m: cv2.circle(m, (300, 500), 15, 1, -1))
    add("on_unsure", lambda m: cv2.circle(m, (500, 500), 18, 1, -1))
    candidates = scoring.Candidates(layer[None], [(0, v) for v in names.values()])
    objects = []
    for name, points in truths.items():
        truth = scoring.Truth(points, "unsure" if name == "unsure" else "plaque")
        truth.rasterize(SHAPE)
        objects.append(truth)
    column = {name: value - 1 for name, value in names.items()}
    counted = [column[n] for n in ("found", "merged", "split_0", "poor", "nothing", "on_unsure")]
    return objects, candidates, counted, column


def test_each_kind_of_segmentation_failure_is_counted_as_such():
    truths, candidates, counted, column = field()
    inter = scoring.intersections(truths, candidates)
    proposed = scoring.score_set(truths, candidates, inter, list(range(len(candidates.keys))))
    detected = scoring.score_set(truths, candidates, inter, counted)
    names = ["found", "missed", "merged_a", "merged_b", "split", "poor", "rejected", "unsure"]
    by_name = dict(zip(names, zip(proposed.outcome, detected.outcome)))
    assert by_name["found"] == ("found", "found")
    assert by_name["missed"] == ("missed", "missed")
    assert by_name["merged_a"] == by_name["merged_b"] == ("merged", "merged")
    assert by_name["split"][0] == "split" and proposed.pieces[names.index("split")] == 3
    assert by_name["split"][1] == "poor outline"  # only one of the three pieces was counted: it fits the object badly
    assert by_name["poor"] == ("poor outline", "poor outline")
    assert by_name["rejected"] == ("found", "missed")
    outcomes = [scoring.end_to_end(p, d) for p, d in zip(proposed.outcome, detected.outcome)]
    assert outcomes[:7] == ["counted", "missed", "merged", "merged", "split", "poor outline", "rejected"]
    # what each counted candidate is: right, a merge, a piece, a poor outline, nothing, and one on an unsure object
    assert detected.extra == ["right", "covers several objects", "piece of an object", "poor outline", "nothing there", "unsure"]
    assert proposed.best_iou[0] > 0.9 and 0.1 < proposed.best_iou[names.index("poor")] < 0.4
    assert proposed.boundary[0] > 0.8 and proposed.boundary[names.index("poor")] < 0.4


def test_matching_at_the_lower_iou_finds_the_loosely_outlined_object():
    truths, candidates, _, column = field()
    inter = scoring.intersections(truths, candidates)
    loose = scoring.score_set(truths[5:6], candidates, inter[5:6], [column["poor"]])
    assert loose.outcome == ["poor outline"] and not loose.found_low[0]
    near = scoring.Truth(circle(410, 300, 20), "plaque")  # moved closer: IoU between 0.3 and 0.5
    near.rasterize(SHAPE)
    scores = scoring.score_set([near], candidates, scoring.intersections([near], candidates), [column["poor"]])
    assert scores.outcome == ["poor outline"] and scores.found_low[0] and 0.3 <= scores.best_iou[0] < 0.5


def test_benchmark_fields_spread_over_strata_and_density():
    rows = []
    for tma in ("1", "2", "3"):
        for group in ("AD", "CT"):
            for region in ("frontal", "occipital"):
                for core in range(3):
                    for i in range(6):
                        rows.append({"tile_id": f"{tma}_{group}_{region}_{core}_{i}", "core_id": f"{tma}_{group}_{region}_{core}", "tma": tma,
                                     "disease_group": group, "region": region})
    tiles = pd.DataFrame(rows)
    density = pd.Series(np.arange(len(tiles)) % 9, index=tiles.tile_id, dtype=float)
    levels = outline.density_levels(density, seed=1)
    assert levels.value_counts().to_dict() == {"sparse": 72, "medium": 72, "dense": 72}
    picked = outline.pick_fields(tiles, levels, 24, seed=3)
    chosen = tiles.set_index("tile_id").loc[picked]
    assert len(set(picked)) == 24
    assert chosen.tma.value_counts().to_dict() == {"1": 8, "2": 8, "3": 8}
    assert chosen.groupby(["tma", "disease_group", "region"]).size().eq(2).all()  # every cell twice
    assert levels.loc[picked].value_counts().to_dict() == {"sparse": 8, "medium": 8, "dense": 8}
    assert chosen.core_id.nunique() == 24  # one field per core while cores are left
    assert outline.pick_fields(tiles, levels, 24, seed=3) == picked
    assert outline.default_tmas(["1", "8", "12", "3"]) == ["1", "3"] and outline.default_tmas(["8", "9"]) == ["8", "9"]


# ---------------------------------------------------------------- the whole check through the app

TILE = "LIP-1_B-2_6E10_S01"
ORIGIN = 172  # the field's picture starts at (300 - 128, 300 - 128) in the core image
PLAQUES = [(250, 250), (450, 250), (250, 450), (450, 450)]  # field pixels; the first two are counted by the pipeline
FAINT = (560, 360)  # a real plaque too faint for the segmentation
STRAY = (600, 150)  # counted by an older segmentation: no candidate there now, nothing outlined


@pytest.fixture()
def client(synthetic_project, monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from stainid.api import state
    from stainid.api.app import create_app
    from stainid.api.routes import outline as outline_routes

    monkeypatch.chdir(synthetic_project)
    jobs = []
    monkeypatch.setattr(outline_routes, "proposals_job", lambda stain, session: jobs.append((stain, session)))
    rgb = np.full((1200, 1200, 3), 225, np.uint8)
    for x, y in PLAQUES:
        cv2.circle(rgb, (ORIGIN + x, ORIGIN + y), 40, (110, 70, 30), -1)
    cv2.circle(rgb, (ORIGIN + FAINT[0], ORIGIN + FAINT[1]), 30, (215, 210, 205), -1)
    cv2.imwrite(str(synthetic_project / "data/cores/LIP-1/B-2/6E10.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    parts = synthetic_project / "data/analysis/cohort_v3/parts/LIP-1_B-2_6E10_objects.csv"
    common = pd.read_csv(parts, dtype=str).iloc[0].to_dict()
    pd.DataFrame([{**common, "centroid_x_px": x, "centroid_y_px": y, "plaque_class": "compact", "plaque_probability": 0.9, "plaque_area_um2": 380.0,
                   "dense_core_fraction": 0.3, "equivalent_diameter_um": 22.0} for x, y in [*PLAQUES[:2], STRAY]]).to_csv(parts, index=False)
    state.reset()
    model = state.get_state().project.model("amyloid")
    model.parent.mkdir(parents=True, exist_ok=True)
    model.write_bytes(b"model")
    admin = TestClient(create_app(), client=("127.0.0.1", 50000))
    assert admin.post("/api/session/admin", json={"password": "admin"}).status_code == 200
    admin.jobs = jobs
    yield admin
    state.reset()


def test_outline_session_from_set_up_to_scores(client):
    from fastapi.testclient import TestClient

    from stainid.api import state
    from stainid.segmentation import proposals, report

    created = client.post("/api/segmentation/sessions", json={"stains": ["6E10"], "n_fields": 40}).json()
    assert [c["title"] for c in created] == ["Segmentation check – 6E10"] and created[0]["fields"] == [TILE] and created[0]["open"]
    name = created[0]["name"]
    assert created[0]["kind"] == "outline" and created[0]["ai_view"] == "none" and created[0]["benchmark"]["tmas"] == ["1"]
    assert client.jobs == [("6E10", name)]  # its candidates are found in the background

    ann = TestClient(client.app)
    ann.post("/api/session/annotator", json={"name": "Ann"})
    detail = ann.get(f"/api/annotate/{name}").json()
    assert detail["meta"]["classes"] == ["compact plaque", "diffuse plaque", "small plaque", "unsure"]
    assert detail["fields"] == [{"tile_id": TILE, "done_by_me": False, "annotators_done": 0, "open_by_others": 0, "prepared": True}]
    assert ann.get(f"/api/annotate/{name}/next").json()["tile_id"] == TILE
    work = ann.get(f"/api/annotate/{name}/outline/{TILE}").json()
    assert work["box"] == {"x": 128, "y": 128, "width": 512, "height": 512} and work["objects"] == [] and not work["complete"]
    assert ann.get(f"/api/annotate/{name}/fields/{TILE}/image.jpg").headers["content-type"] == "image/jpeg"

    records = [{"ref": f"p{i}", "value": "compact plaque", "points": circle(x, y, 40)} for i, (x, y) in enumerate([*PLAQUES, FAINT])]
    records.append({"ref": "odd", "value": "unsure", "points": circle(600, 600, 10)})
    assert len(ann.post(f"/api/annotate/{name}/outline/{TILE}/objects", json={"records": records}).json()) == 6
    moved = ann.post(f"/api/annotate/{name}/outline/{TILE}/objects", json={"records": [{"ref": "p4", "value": "diffuse plaque", "points": circle(*FAINT, 30)}]})
    assert moved.json()[0]["id"] == f"object:{name}:{TILE}:p4" and moved.json()[0]["x"] == pytest.approx(FAINT[0], abs=0.5)
    bad = ann.post(f"/api/annotate/{name}/outline/{TILE}/objects", json={"records": [{"ref": "x", "value": "not a plaque", "points": circle(1, 1, 5)}]})
    assert bad.status_code == 400
    assert ann.post(f"/api/annotate/{name}/outline/{TILE}/segment", json={"points": [[250, 250, 1]]}).status_code == 409  # no SAM weights
    assert len(ann.get(f"/api/annotate/{name}/outline/{TILE}").json()["objects"]) == 6
    reference = ann.get(f"/api/annotate/{name}/outline/{TILE}/reference").json()
    assert [(c["x"], c["y"]) for c in reference["counted"]] == [(250, 250), (450, 250), STRAY] and reference["candidates"] is None

    ann.post(f"/api/annotate/{name}/outline/{TILE}/complete", json={"complete": True})
    assert ann.get(f"/api/annotate/{name}").json()["fields"][0]["done_by_me"]
    listed = client.get("/api/labels/overview").json()["sessions"]
    assert [(s["kind"], s["fields_covered"], s["fields_prepared"], s["labels"]) for s in listed] == [("outline", 1, 1, 6)]
    fields = client.get(f"/api/labels/sessions/{name}/fields").json()
    assert fields[0]["state"] == "done" and fields[0]["done_by"] == ["Ann"]
    assert client.get("/api/labels/sets", params={"stain": "6E10"}).json() == []  # outlines never train a model

    project = state.get_state().project
    report.run(project, "6E10", proposals_only=True)
    saved = proposals.load(project, "6E10", TILE)
    assert len(saved["x"]) == 4 and saved["layers"].shape == (1, 768, 768)
    assert ann.get(f"/api/annotate/{name}/outline/{TILE}/reference").json()["candidates"]
    scores = report.run(project, "6E10")["stains"][0]
    assert scores["objects"] == 5 and scores["fields_scored"] == 1 and scores["people"] == ["Ann"]
    assert scores["outcomes"] == {"counted": 2, "rejected": 2, "missed": 1, "merged": 0, "split": 0, "poor outline": 0}
    assert scores["proposal_recall"] == 0.8 and scores["detection_recall"] == 0.4 and scores["detection_precision"] == pytest.approx(2 / 3, abs=1e-3)
    assert scores["counted_outcomes"]["nothing there"] == 1 and scores["not_found_again"] == 1
    assert scores["by_tma"][0]["key"] == "1" and scores["by_density"][0]["objects"] == 5
    assert sum(b["objects"] for b in scores["by_size"]) == 5 and 0.6 < scores["proposal_iou_median"] < 0.8  # the segmentation grows each plaque a little
    assert "never proposed" in report.plain(scores)
    objects = pd.read_csv(proposals.folder(project) / "objects.csv")
    assert len(objects) == 6 and set(objects.outcome) == {"counted", "rejected", "missed"}
    shown = client.get("/api/segmentation").json()
    assert shown["checks"][0]["complete"] == 1 and shown["report"]["stains"][0]["objects"] == 5 and all(f["exists"] for f in shown["files"])
    assert shown["tma_study"] and shown["report"]["imported"] == []


def test_polygons_imported_from_a_dataset_are_scored_per_rater(tmp_path):
    """An image-set project made from a dataset package: the rater who drew polygons is scored, the boxes are not."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    from package_files import fake_plaques, regions_with_boxes

    from stainid.imageset import package, prepare
    from stainid.project import load_project
    from stainid.sampling.tiles import table
    from stainid.segmentation import report

    made = package.create_project(regions_with_boxes(tmp_path / "dataset"), tmp_path / "project")
    prepare.prepare_images(made, log=lambda *a, **k: None)
    project = load_project(made.root)
    cores = pd.read_csv(project.input("core_images"), dtype=str)
    pd.DataFrame([{"tma": t, "stain": "6E10", "threshold_dab_od": 0.05} for t in sorted(set(cores.tma))]).to_csv(project.input("calibration"), index=False)
    tiles = table(project.analysis_tiles())
    fake_plaques(project.root, tiles, {"r0": [(200.0, 300.0)], "r1": [(200.0, 450.0)]}, classes="diffuse")
    found = report.truths(project)
    assert [(t.imported, t.meta["imported"]["rater"]) for t in found] == [(True, "expert1")]  # the consensus drew boxes only
    scores = report.run(project)
    assert scores["stains"] == [] and len(scores["imported"]) == 1
    rater = scores["imported"][0]
    assert rater["rater"] == "expert1" and rater["objects"] == 4 and rater["fields_scored"] == 2
    assert rater["outcomes"]["missed"] == 0 and sum(rater["outcomes"].values()) == 4  # every plaque is proposed in some form
    assert rater["counted"] == 2 and "Synthetic boxes: expert1" in report.plain(rater)


def test_a_field_without_objects_still_counts_what_the_ai_counted_there():
    from stainid.segmentation import report

    objects = pd.DataFrame(columns=report.OBJECT_COLUMNS)
    counted = pd.DataFrame([{"tile_id": "f", "tma": "1", "outcome": "nothing there", "right_at_low_iou": False}] * 2)
    figures = report.summarize(objects, counted)
    assert figures["objects"] == 0 and figures["detection_recall"] is None and figures["detection_precision"] == 0.0
    assert figures["counted_outcomes"]["nothing there"] == 2
