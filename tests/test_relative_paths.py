"""File paths in project tables are kept relative to the project folder (`stainid.project.stored_path`), so a project can
be moved together with its scans; tables written before (absolute paths) are read as they are."""
import json
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml
from test_grid_size import array_mask, preview_of

from stainid.imaging import chromogens, core_images
from stainid.project import load_project, located_path, stored_path
from stainid.sampling.analysis_manifest import analysis_rows
from stainid.setup import guide
from stainid.slides import dearray
from stainid.slides.table import read_slides, save_slides, set_slide_grid
from stainid.tables import read_csv, write_records
from stainid.workflows import steps
from stainid.workflows.slides import dearray_slides

STAINS = ["NeuN", "6E10", "AT8"]
NAMES = ("TMA 1 NeuN", "TMA 2 NeuN")


def _study(tmp_path: Path):
    (tmp_path / "study").mkdir()
    return load_project(tmp_path / "study")


def test_a_path_is_kept_relative_when_it_shares_a_folder_with_the_project():
    root = Path("/media/me/disk/lab/tma_study")
    assert stored_path(root, "/media/me/disk/lab/raw_scans/x.vsi") == "../raw_scans/x.vsi"
    assert stored_path(root, root / "data" / "slides.csv") == "data/slides.csv"
    assert stored_path(root, root) == "."
    assert stored_path(Path("/home/u/p"), "/data/scans/x.vsi") == "/data/scans/x.vsi"  # only the filesystem root in common
    assert stored_path(root, "data/cores/LIP-1/A-2/6E10.png") == "data/cores/LIP-1/A-2/6E10.png"  # relative already
    assert located_path(root, "../raw_scans/x.vsi") == Path("/media/me/disk/lab/raw_scans/x.vsi")
    assert located_path(root, "/data/scans/x.vsi") == Path("/data/scans/x.vsi")
    for path in ("/media/me/disk/lab/raw_scans/a b/x.vsi", "/media/me/disk/other/y.svs", "/media/me/disk/lab/tma_study/data/z.csv"):
        assert located_path(root, stored_path(root, path)) == Path(path)



def test_a_file_on_another_drive_is_kept_in_full(monkeypatch):
    """Two drives mounted side by side share their mount folder: a scan on the other drive is not `../../b/...`."""
    from stainid import project as project_module

    monkeypatch.setattr(project_module, "_device", lambda path: 2 if str(path).startswith("/media/me/b") else 1)
    assert stored_path("/media/me/a/study", "/media/me/b/scans/x.vsi") == "/media/me/b/scans/x.vsi"
    assert stored_path("/media/me/a/study", "/media/me/a/scans/x.vsi") == "../scans/x.vsi"

def test_project_relative_and_locate_use_the_same_rule(tmp_path):
    project = _study(tmp_path)
    scan = tmp_path / "scans" / "x.vsi"
    assert project.relative(scan) == "../scans/x.vsi" and project.locate("../scans/x.vsi") == scan
    assert project.relative(project.input("slides_table")) == "data/slides.csv"


@pytest.fixture()
def sibling_scans(tmp_path, monkeypatch):
    """A study in `a/study` whose two scans lie beside it in `a/scans`; Find cores reads a drawn preview of each."""
    root, scans = tmp_path / "a" / "study", tmp_path / "a" / "scans"
    root.mkdir(parents=True)
    scans.mkdir()
    (root / "stainid.yaml").write_text(yaml.safe_dump({"name": "moved", "inputs": {"slides_dir": "../scans"},
                                                       "tma": {"prefix": "LIP-", "rows": 5, "columns": 6}}))
    masks = {"TMA 1 NeuN": array_mask(5, 6, missing={(2, 2)}), "TMA 2 NeuN": array_mask(4, 6, seed=1)}
    for name in masks:
        (scans / f"{name}.vsi").write_text("")
    monkeypatch.setattr(dearray, "read_slide_preview", lambda path, target: preview_of(masks[Path(path).stem], Path(path)))
    return load_project(root)


def _register(project) -> None:
    rows = [{"slide_path": str(project.input("slides_dir") / f"{n}.vsi"), "tma": n[4], "stain": "NeuN"} for n in NAMES]
    save_slides(project.input("slides_table"), rows, STAINS, project.root)


def test_find_cores_keeps_the_scans_relative_in_every_table(sibling_scans, tmp_path, monkeypatch):
    project = sibling_scans
    _register(project)
    dearray_slides(project)
    kept = {f"../scans/{n}.vsi" for n in NAMES}
    assert {r["slide_path"] for r in read_csv(project.input("slides_table"))} == kept
    manifest = read_csv(project.input("core_manifest"))
    assert {r["slide_path"] for r in manifest} == kept
    grids = project.output("qc") / "grids"
    assert {json.loads((grids / f"{n.replace(' ', '_')}.json").read_text())["slide_path"] for n in NAMES} == kept
    # the core image table copies the core table's paths (relative to the same project folder)
    filled = [{**r, "core_role": "biological", "output_path": f"cores/LIP-{r['tma']}/{r['core_label']}/NeuN.png", "output_width_px": "10",
               "output_height_px": "10", "donor_id": "1", "region": "frontal", "disease_group": "AD", "technical_replicate": "1",
               "sample_region_id": "1_frontal", "tissue_status": "substantial", "tissue_fraction": "0.9", "review_required": "false"}
              for r in manifest]
    assert {r["slide_path"] for r in analysis_rows(filled, "LIP-")} == kept
    # read from anywhere: in full, from the project folder
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    slides = read_slides(project.input("slides_table"), project.root)
    assert all(Path(s["slide_path"]).is_file() and Path(s["slide_path"]).is_absolute() for s in slides)
    assert steps._dearray_status(project)["state"] == "done"
    reports = guide.slide_reports(project)
    assert all(r["found"] and Path(r["slide_path"]).is_file() for r in reports)
    guide.mark_checked(project, reports[0]["slide_path"])
    assert json.loads((grids / "TMA_1_NeuN.json").read_text())["slide_path"] == "../scans/TMA 1 NeuN.vsi"
    set_slide_grid(project.input("slides_table"), reports[1]["slide_path"], 4, 6, root=project.root)
    assert {r["slide_path"] for r in read_csv(project.input("slides_table"))} == kept


def test_tables_with_full_paths_still_load_and_match(sibling_scans, monkeypatch):
    """Tables written before: the scans in full. They are found and match as before; the next write keeps them relative."""
    project = sibling_scans
    _register(project)
    dearray_slides(project)
    table, manifest = project.input("slides_table"), project.input("core_manifest")
    report = project.output("qc") / "grids" / "TMA_1_NeuN.json"
    for path in (table, manifest):  # as written before
        write_records(path, [{**r, "slide_path": str(project.locate(r["slide_path"]))} for r in read_csv(path)])
    report.write_text(json.dumps({**json.loads(report.read_text()), "slide_path": str(project.input("slides_dir") / "TMA 1 NeuN.vsi")}))
    assert all(Path(r["slide_path"]).is_absolute() for path in (table, manifest) for r in read_csv(path))
    full = [str(project.input("slides_dir") / f"{n}.vsi") for n in NAMES]
    assert [s["slide_path"] for s in read_slides(table, project.root)] == [s["slide_path"] for s in read_slides(table)] == full
    assert {k: steps._dearray_status(project)[k] for k in ("state", "done", "total")} == {"state": "done", "done": 2, "total": 2}
    assert all(r["found"] for r in guide.slide_reports(project))
    monkeypatch.setattr(dearray, "read_slide_preview", lambda path, target: pytest.fail(f"{path} was looked at again"))
    dearray_slides(project)  # nothing to do
    set_slide_grid(table, full[0], 5, 6, root=project.root)
    guide.mark_checked(project, full[0])
    assert {r["slide_path"] for r in read_csv(table)} == {f"../scans/{n}.vsi" for n in NAMES}
    assert json.loads(report.read_text())["slide_path"] == "../scans/TMA 1 NeuN.vsi"
    assert steps._dearray_status(project)["done"] == 2  # relative slides table, full paths in the core table


def test_a_project_moved_with_its_scans_still_finds_them(sibling_scans, tmp_path, monkeypatch):
    _register(sibling_scans)
    dearray_slides(sibling_scans)
    shutil.move(tmp_path / "a", tmp_path / "b")
    project = load_project(tmp_path / "b" / "study")
    slides = read_slides(project.input("slides_table"), project.root)
    assert [s["slide_path"] for s in slides] == [str(tmp_path / "b" / "scans" / f"{n}.vsi") for n in NAMES]
    assert steps._dearray_status(project)["state"] == "done"
    assert all(r["found"] for r in guide.slide_reports(project))
    monkeypatch.setattr(dearray, "read_slide_preview", lambda path, target: pytest.fail(f"{path} was looked at again"))
    dearray_slides(project)  # every slide's cores are found already


def test_core_image_links_keep_their_source_relative(tmp_path):
    root = tmp_path / "study"
    source = tmp_path / "scans" / "S.png"
    source.parent.mkdir(parents=True)
    cv2.imwrite(str(source), np.full((64, 64, 3), 200, np.uint8))
    link = root / "data" / "images" / "links" / "S.json"
    dab = chromogens.recipe("DAB")
    core_images.write_link(link, source, (0, 0, 64, 64), 1.0, dab, True, (64, 64))
    assert json.loads(link.read_text())["source"] == "../scans/S.png"
    assert core_images.read_link(link)["source"] == str(source)
    with core_images.open_image(link) as image:
        assert image.size == (64, 64)
    # a link written before (its source in full) that says the same is left as it is: stage 2 goes by its time and size
    old = {**json.loads(link.read_text()), "source": str(source)}
    link.write_text(json.dumps(old, indent=1))
    os.utime(link, ns=(1_000_000_000, 1_000_000_000))
    core_images.write_link(link, source, (0, 0, 64, 64), 1.0, dab, True, (64, 64))
    assert link.stat().st_mtime_ns == 1_000_000_000 and json.loads(link.read_text())["source"] == str(source)
    core_images.write_link(link, source, (0, 0, 32, 32), 1.0, dab, True, (32, 32))  # says something else: written anew
    assert json.loads(link.read_text())["source"] == "../scans/S.png"


def test_the_image_list_keeps_its_files_relative(tmp_path):
    from stainid.imageset.table import read_images, resolve, save_images

    project = _study(tmp_path)
    image = tmp_path / "images" / "case1_AT8.png"
    save_images(project, [{"path": str(image), "stain": "AT8", "donor_id": "1"}, {"path": "own/case2_AT8.png", "stain": "AT8", "donor_id": "2"}])
    assert [r["path"] for r in read_csv(project.input("image_list"))] == ["../images/case1_AT8.png", "own/case2_AT8.png"]
    rows = read_images(project)
    assert [r["path"] for r in rows] == [str(image), str(project.root / "own" / "case2_AT8.png")]
    assert resolve(project, rows[0]) == image


def test_a_model_kept_in_full_before_is_the_same_model_now(tmp_path):
    from stainid.workflows.fields import check_models

    project = _study(tmp_path)
    model = tmp_path / "models" / "bundle.joblib"
    record = tmp_path / "study" / "provenance.json"
    record.write_text(json.dumps({"amyloid": str(model)}))  # as written before
    check_models(project, record, {"amyloid": model})
    with pytest.raises(ValueError, match="different model"):
        check_models(project, record, {"amyloid": tmp_path / "models" / "other.joblib"})


def test_a_folder_chosen_in_the_app_is_kept_relative(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from stainid.api import state
    from stainid.api.app import create_app

    root = tmp_path / "study"
    root.mkdir()
    (root / "stainid.yaml").write_text(yaml.safe_dump({"name": "picked"}))
    monkeypatch.setenv("STAINID_PROJECT", str(root))
    state.reset()
    client = TestClient(create_app(), client=("127.0.0.1", 50000))
    assert client.post("/api/session/admin", json={"password": "admin"}).status_code == 200
    changed = client.put("/api/project/config", json={"changes": {"inputs": {"slides_dir": str(tmp_path / "scans")}}})
    assert changed.status_code == 200
    assert yaml.safe_load((root / "stainid.yaml").read_text())["inputs"] == {"slides_dir": "../scans"}
    assert load_project(root).input("slides_dir") == tmp_path / "scans"
    state.reset()
