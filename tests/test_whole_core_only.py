"""Whole cores are the only analysis: tiles without a field-selection step, calibration and labelling fields from the
whole-core tiles, and older projects' sampled fields still readable where their labels refer to them."""
from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
import pytest

from stainid.project import load_project, save_project
from stainid.tables import read_csv, write_csv

CORE = {"tma": "1", "core_label": "B-2", "core_role": "biological", "slide_path": "slides/x.vsi", "center_x_px": "0", "center_y_px": "0",
        "diameter_x_px": "1200", "diameter_y_px": "1000", "source_pixel_width_um": "0.274", "source_pixel_height_um": "0.274",
        "donor_id": "1001", "region": "frontal", "disease_group": "AD", "technical_replicate": "1", "sample_region_id": "1001_frontal",
        "tissue_status": "complete", "tissue_fraction": "0.8", "review_required": "false"}


def _core_project(root, stains=("NeuN",)):
    """A one-core project after Check core quality: a core table and a 1200 x 1000 core image with a dark disc of tissue."""
    image = np.full((1000, 1200, 3), 235, np.uint8)
    cv2.circle(image, (600, 500), 420, (150, 110, 90), -1)
    rows = []
    for stain in stains:
        path = root / "data" / "cores" / "LIP-1" / "B-2" / f"{stain}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), image)
        rows.append({**CORE, "stain": stain, "output_path": f"cores/LIP-1/B-2/{stain}.png", "output_width_px": "1200", "output_height_px": "1000"})
    write_csv(root / "data" / "core_manifest.csv", rows, list(rows[0]))
    return save_project(load_project(root), {"tma": {"prefix": "LIP-"}})


def test_an_old_sampled_fields_setting_is_read_and_ignored(tmp_path):
    (tmp_path / "stainid.yaml").write_text("name: old\nanalysis:\n  tiles: sampled_fields\n")
    project = load_project(tmp_path)
    assert project.config["analysis"]["tiles"] == "whole_core"  # data/analysis/whole_core_run.sh checks this value
    assert project.analysis_tiles() == project.input("whole_core_manifest")
    save_project(project, {"name": "renamed"})
    assert "analysis" not in (tmp_path / "stainid.yaml").read_text()


def test_tiles_need_no_field_selection_and_keep_earlier_field_ids(tmp_path):
    from stainid.sampling.whole_core import write_whole_core_manifest

    project = _core_project(tmp_path)
    write_whole_core_manifest(project, 512)  # no sampled-field table: every tile is a W tile
    tiles = read_csv(project.analysis_tiles())
    assert read_csv(project.input("core_images"))[0]["core_id"] == "LIP-1_B-2"  # the core image table is made here too
    assert {t["tile_id"] for t in tiles} >= {"LIP-1_B-2_NeuN_W0000", "LIP-1_B-2_NeuN_W0101"}
    assert all(t["nested_sample"] == "whole_core" and int(t["selection_order"]) >= 100 for t in tiles)
    # an older project: the square that was a sampled field keeps its id and order
    write_csv(project.input("field_manifest"), [{"tile_id": "LIP-1_B-2_NeuN_S01", "core_id": "LIP-1_B-2", "stain": "NeuN", "x_px": 512,
                                                 "y_px": 512, "selection_order": 1, "nested_sample": "primary_four"}],
              ["tile_id", "core_id", "stain", "x_px", "y_px", "selection_order", "nested_sample"])
    write_whole_core_manifest(project, 512)
    again = {t["tile_id"]: t for t in read_csv(project.analysis_tiles())}
    assert "LIP-1_B-2_NeuN_S01" in again and "LIP-1_B-2_NeuN_W0101" not in again and again["LIP-1_B-2_NeuN_S01"]["x_px"] == "512"
    assert len(again) == len(tiles)


def test_calibration_pools_every_tile_and_keeps_existing_thresholds(tmp_path, monkeypatch):
    from stainid.sampling.whole_core import write_whole_core_manifest
    from stainid.v2 import field
    from stainid.workflows.calibrate import calibrate_slides

    # the synthetic disc is dark and blue, which the linear-artifact mask (left out of the sample) would take for a scratch
    monkeypatch.setattr(field, "linear_artifact_mask", lambda rgb: np.zeros(rgb.shape[:2], bool))
    project = _core_project(tmp_path, ("NeuN", "AT8"))
    write_whole_core_manifest(project, 512)
    calibrate_slides(project, stride=1)
    rows = {r["stain"]: r for r in read_csv(project.input("calibration"))}
    assert set(rows) == {"NeuN", "AT8"} and float(rows["NeuN"]["threshold_dab_od"]) >= 0.01
    disc = np.pi * 420 ** 2  # the tissue of all tiles, not of one field
    assert 0.9 * disc < int(rows["NeuN"]["sampled_tissue_pixels"]) < 1.1 * disc
    table = project.input("calibration")
    pd.DataFrame([{**rows["NeuN"], "threshold_dab_od": 0.123}, rows["AT8"]]).to_csv(table, index=False)
    calibrate_slides(project)  # slides that have a threshold keep it
    assert float({r["stain"]: r for r in read_csv(table)}["NeuN"]["threshold_dab_od"]) == 0.123
    calibrate_slides(project, ["NeuN"], redo=True)  # unless asked again, and then only for these stains
    again = {r["stain"]: r for r in read_csv(table)}
    assert float(again["NeuN"]["threshold_dab_od"]) != 0.123 and again["AT8"]["threshold_dab_od"] == rows["AT8"]["threshold_dab_od"]


def test_calibration_and_nuclei_can_be_limited_to_some_core_images(tmp_path):
    from stainid.cli import build_parser
    from stainid.sampling.whole_core import write_whole_core_manifest
    from stainid.workflows.calibrate import calibrate_slides

    project = _core_project(tmp_path, ("NeuN",))
    write_whole_core_manifest(project, 512)
    with pytest.raises(ValueError, match="None of these core images"):
        calibrate_slides(project, stride=1, cores={"elsewhere"})
    calibrate_slides(project, stride=1, cores={"LIP-1_B-2"})
    assert [r["tma"] for r in read_csv(project.input("calibration"))] == ["1"]
    args = build_parser().parse_args(["nuclei", "--stain", "NeuN", "--core", "LIP-1_B-2", "--core", "LIP-1_B-3"])
    assert args.core == ["LIP-1_B-2", "LIP-1_B-3"]
    assert build_parser().parse_args(["calibrate", "--core", "cores.txt"]).core == ["cores.txt"]


def test_labelling_fields_come_from_full_whole_core_tiles_and_old_fields_stay_known(tmp_path):
    from stainid.sampling.tiles import known_tile_rows, known_tiles, labelling_tiles

    project = load_project(tmp_path)
    base = {"core_id": "C", "tma": "1", "stain": "AT8", "disease_group": "AD", "image_path": "x.png", "selection_order": 100}
    whole = [{**base, "tile_id": "C_AT8_S01", "x_px": 0, "y_px": 0, "width_px": 2048, "height_px": 2048, "preview_tissue_fraction": 0.9},
             {**base, "tile_id": "C_AT8_W0001", "x_px": 2048, "y_px": 0, "width_px": 2048, "height_px": 2048, "preview_tissue_fraction": 0.3},
             {**base, "tile_id": "C_AT8_W0002", "x_px": 4096, "y_px": 0, "width_px": 700, "height_px": 2048, "preview_tissue_fraction": 1.0},
             {**base, "tile_id": "C_AT8_W0100", "x_px": 0, "y_px": 2048, "width_px": 2048, "height_px": 2048, "preview_tissue_fraction": 0.6},
             {**base, "tile_id": "C_AT8_W0101", "x_px": 2048, "y_px": 2048, "width_px": 2048, "height_px": 2048, "preview_tissue_fraction": 0.05}]
    write_csv(project.analysis_tiles(), whole, list(whole[0]))
    assert list(labelling_tiles(project).tile_id) == ["C_AT8_S01", "C_AT8_W0001", "C_AT8_W0100"]  # full size, at least 10 % tissue (5 % stays out)
    old = [{**base, "tile_id": "C_AT8_S01", "x_px": 0, "y_px": 0, "width_px": 2048, "height_px": 2048},
           {**base, "tile_id": "C_AT8_S02", "x_px": 9000, "y_px": 0, "width_px": 2048, "height_px": 2048}]
    write_csv(project.input("tile_manifest"), old, list(old[0]))  # an older project's sampled fields, named by its labels
    assert list(known_tiles(project).tile_id) == ["C_AT8_S01", "C_AT8_W0001", "C_AT8_W0002", "C_AT8_W0100", "C_AT8_W0101", "C_AT8_S02"]
    assert known_tile_rows(project)["C_AT8_S02"]["x_px"] == "9000" and "C_AT8_S02" not in set(labelling_tiles(project).tile_id)


@pytest.fixture()
def client(synthetic_project, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from stainid.api import state
    from stainid.api.app import create_app

    monkeypatch.chdir(synthetic_project)
    state.reset()
    admin = TestClient(create_app(), client=("127.0.0.1", 50000))
    assert admin.post("/api/session/admin", json={"password": "admin"}).status_code == 200
    yield admin
    state.reset()


def test_core_view_has_no_sampled_fields_and_old_fields_still_open(client, synthetic_project):
    core = client.get("/api/cores/LIP-1_B-2").json()
    assert "fields" not in core and core["core_id"] == "LIP-1_B-2"
    old = pd.read_csv(synthetic_project / "data/analysis/cohort_whole_core_tiles.csv").assign(tile_id=lambda t: t.tile_id.str.replace("S01", "S02"))
    old.to_csv(synthetic_project / "data/analysis/cohort_primary_tiles.csv", index=False)  # a field of an older project, outside the tiles
    assert client.get("/api/images/fields/LIP-1_B-2_6E10_S02.jpg").status_code == 200
    paths = client.get("/api/project").json()["paths"]["inputs"]
    assert "whole_core_manifest" in paths and "tile_manifest" not in paths and "field_manifest" not in paths


def test_a_stain_named_like_a_number_stays_text(tmp_path):
    """A tile table whose only stain is 6E10 must not turn it into the number 6e10."""
    import pandas as pd

    from stainid.sampling.tiles import TEXT_COLUMNS
    path = tmp_path / "tiles.csv"
    path.write_text("tile_id,stain,tma,x_px\nLIP-1_A-1_6E10_W0001,6E10,1,0\n")
    assert pd.read_csv(path, dtype=TEXT_COLUMNS).stain.tolist() == ["6E10"]
