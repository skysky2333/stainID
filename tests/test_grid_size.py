"""Finding a slide's grid size automatically (Find cores on slides without their own rows / columns)."""
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from stainid.project import load_project
from stainid.slides import dearray
from stainid.slides.dearray import choose_grid, core_like, estimate_grid, oriented, tissue_blobs
from stainid.slides.table import read_slides, set_slide_grid
from stainid.slides.vsi import SceneInfo, SlidePreview
from stainid.workflows.slides import dearray_slides, read_grid_report

PITCH = 120


def array_mask(rows: int, columns: int, missing=(), debris: int = 0, tilt: float = 0.0, seed: int = 0) -> np.ndarray:
    """A tissue mask of a rows x columns TMA: round cores on a slightly tilted lattice, a few missing, some crumbs."""
    rng = np.random.default_rng(seed)
    mask = np.zeros((rows * PITCH + 200, columns * PITCH + 260), np.uint8)
    for row in range(rows):
        for column in range(columns):
            if (row, column) in missing:
                continue
            x = 130 + column * PITCH - row * tilt * PITCH + rng.normal(0, 3)
            y = 100 + row * PITCH + column * tilt * PITCH + rng.normal(0, 3)
            cv2.circle(mask, (int(x), int(y)), int(rng.uniform(40, 48)), 1, -1)
    for _ in range(debris):
        cv2.circle(mask, (int(rng.uniform(0, mask.shape[1])), int(rng.uniform(0, mask.shape[0]))), 4, 1, -1)
    return mask.astype(bool)


@pytest.mark.parametrize("rows, columns", [(5, 6), (4, 6), (3, 8), (6, 4)])
def test_the_grid_size_is_found_when_the_slide_has_none(rows, columns):
    mask = array_mask(rows, columns, missing={(1, 2), (rows - 1, 0)}, debris=6, tilt=0.02)
    choice = choose_grid(mask, 1.0, default=(5, 6))
    assert (choice.rows, choice.columns, choice.source) == (rows, columns, "auto")
    assert len(choice.fit.inlier_components) >= rows * columns - 2
    assert choice.score > 0.85


def test_a_4_by_6_array_in_a_5_by_6_study_gets_4_rows():
    mask = array_mask(4, 6)
    choice = choose_grid(mask, 1.0, default=(5, 6))
    assert (choice.rows, choice.columns) == (4, 6)
    scores = {(r, c): s for r, c, s in choice.tried}
    assert scores[4, 6] > (scores.get((5, 6)) or 0)


def test_the_project_default_is_kept_when_it_fits():
    mask = array_mask(5, 6, missing={(0, 0), (0, 1), (4, 5)})
    choice = choose_grid(mask, 1.0, default=(5, 6))
    assert (choice.rows, choice.columns) == (5, 6)


def test_a_size_that_is_set_is_used_as_it_is():
    mask = array_mask(4, 6)
    choice = choose_grid(mask, 1.0, rows=4, columns=6, default=(5, 6))
    assert (choice.rows, choice.columns, choice.source, choice.tried) == (4, 6, "set", ())
    assert (choose_grid(mask, 1.0, rows=5, columns=6).rows, choose_grid(mask, 1.0, rows=5, columns=6).source) == (5, "set")


def test_counting_rows_and_columns_ignores_crumbs():
    mask = array_mask(3, 7, debris=20, seed=3)
    assert estimate_grid(core_like(tissue_blobs(mask), mask.shape)) == (3, 7)


def test_an_empty_slide_says_so():
    with pytest.raises(ValueError, match="No grid"):
        choose_grid(np.zeros((600, 800), bool), 1.0)


def test_orientation_relabels_positions():
    assert oriented(0, 0, 5, 6) == (0, 0)
    assert oriented(0, 0, 5, 6, "rotate_180") == (4, 5)
    assert oriented(1, 2, 5, 6, "flip_left_right") == (1, 3)
    assert oriented(1, 2, 5, 6, "flip_top_bottom") == (3, 2)
    with pytest.raises(ValueError):
        oriented(0, 0, 5, 6, "sideways")


def preview_of(mask: np.ndarray, path: Path) -> SlidePreview:
    rgb = np.repeat((235 - mask * 90)[..., None], 3, axis=2).astype(np.uint8)
    scene = SceneInfo(0, "full", mask.shape[1], mask.shape[0])
    return SlidePreview(path=path, rgb=rgb, full_scene=scene, preview_scene=scene, scale_x=1.0, scale_y=1.0, pixel_width_um=1.0, pixel_height_um=1.0)


@pytest.fixture()
def slide_project(tmp_path, monkeypatch):
    """Two slides in a 5 x 6 study: TMA 1 is 5 x 6, TMA 2 is a 4 x 6 array; no slide has its own grid."""
    (tmp_path / "stainid.yaml").write_text(yaml.safe_dump({"name": "grids", "tma": {"prefix": "LIP-", "rows": 5, "columns": 6}}))
    slides = tmp_path / "data" / "slides"
    slides.mkdir(parents=True)
    masks = {"TMA 1 NeuN": array_mask(5, 6, missing={(2, 2)}), "TMA 2 NeuN": array_mask(4, 6, seed=1)}
    for name in masks:
        (slides / f"{name}.vsi").write_text("")
    (tmp_path / "data" / "slides.csv").write_text("slide_path,tma,stain\n" + "".join(f"{slides / n}.vsi,{n[4]},NeuN\n" for n in masks))
    monkeypatch.setattr(dearray, "read_slide_preview", lambda path, target: preview_of(masks[Path(path).stem], Path(path)))
    return load_project(tmp_path)


def test_find_cores_records_the_size_it_chose(slide_project):
    dearray_slides(slide_project)
    slides = {Path(s["slide_path"]).stem: s for s in read_slides(slide_project.input("slides_table"))}
    assert (slides["TMA 2 NeuN"]["rows"], slides["TMA 2 NeuN"]["columns"], slides["TMA 2 NeuN"]["grid_source"]) == ("4", "6", "auto")
    assert slides["TMA 1 NeuN"]["rows"] == "5"
    manifest = pd_rows(slide_project.input("core_manifest"))
    assert sum(r["tma"] == "2" for r in manifest) == 24 and sum(r["tma"] == "1" for r in manifest) == 30
    report = read_grid_report(slide_project, Path(slides["TMA 2 NeuN"]["slide_path"]))
    assert report["rows"] == 4 and report["grid_source"] == "auto" and report["tried"]
    grids = slide_project.output("qc") / "grids"
    assert (grids / "TMA_2_NeuN.jpg").exists() and (grids / "previews" / "TMA_2_NeuN.jpg").exists()


def test_a_size_set_by_hand_wins_and_one_slide_can_be_found_again(slide_project):
    dearray_slides(slide_project)
    table = slide_project.input("slides_table")
    two = next(s["slide_path"] for s in read_slides(table) if s["tma"] == "2")
    set_slide_grid(table, two, 4, 6, "rotate_180")
    dearray_slides(slide_project, only=["TMA 2 NeuN.vsi"])
    rows = [r for r in pd_rows(slide_project.input("core_manifest")) if r["tma"] == "2"]
    first = min(rows, key=lambda r: float(r["lattice_center_x_px"]) + float(r["lattice_center_y_px"]))
    assert first["core_label"] == "F-4"  # the top-left core of a slide turned 180 degrees is the map's last position
    slide = next(s for s in read_slides(table) if s["tma"] == "2")
    assert (slide["grid_source"], slide["orientation"]) == ("manual", "rotate_180")
    assert read_grid_report(slide_project, Path(two))["grid_source"] == "set"


def test_a_slide_that_fails_does_not_stop_the_others(slide_project, monkeypatch):
    masks = {"TMA 1 NeuN": np.zeros((700, 900), bool), "TMA 2 NeuN": array_mask(4, 6, seed=1)}
    monkeypatch.setattr(dearray, "read_slide_preview", lambda path, target: preview_of(masks[Path(path).stem], Path(path)))
    with pytest.raises(ValueError, match="No cores found on 1 of 2 slides"):
        dearray_slides(slide_project)
    assert {r["tma"] for r in pd_rows(slide_project.input("core_manifest"))} == {"2"}
    report = json.loads((slide_project.output("qc") / "grids" / "TMA_1_NeuN.json").read_text())
    assert "No grid" in report["error"]


def pd_rows(path):
    from stainid.tables import read_csv

    return read_csv(path)


def test_the_cores_screen_shows_each_slide_and_core_pictures(slide_project, monkeypatch, tmp_path_factory):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from stainid.api import state
    from stainid.api.app import create_app

    dearray_slides(slide_project)
    monkeypatch.setenv("STAINID_PROJECT", str(slide_project.root))
    monkeypatch.setattr(state, "SETTINGS", tmp_path_factory.mktemp("home") / "app.json")
    state._STATE = None
    client = TestClient(create_app(), client=("127.0.0.1", 50000))
    client.post("/api/session/admin", json={"password": "admin"})
    slides = client.get("/api/setup/cores").json()["slides"]
    two = next(s for s in slides if s["tma"] == "2")
    assert two["found"] and two["image"] == "TMA_2_NeuN.jpg" and two["report"]["rows"] == 4 and two["grid_source"] == "auto"
    assert not two["checked"]
    marked = client.post("/api/setup/cores/checked", json={"slide_path": two["slide_path"]}).json()["slides"]
    assert next(s for s in marked if s["tma"] == "2")["checked"]
    picture = client.get("/api/setup/thumbs/2/A-1.jpg?size=96")
    assert picture.status_code == 200 and picture.headers["content-type"] == "image/jpeg"
    assert client.get("/api/setup/thumbs/2/F-5.jpg").status_code == 404
    overview = client.get("/api/setup").json()
    assert {s["id"]: s["state"] for s in overview["screens"]}["cores"] == "done"
    state._STATE = None
