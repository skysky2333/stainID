"""Image-set and whole-slide projects: dataset packages, rescaling, colours, the setup per kind, imported labels and
scoring the models on them."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from package_files import fake_plaques, image_labels, paint, regions_with_boxes, tiles_with_points, two_chromogens, whole_slide

from stainid.imageset import annotations, package, prepare, scoring
from stainid.imaging import chromogens, core_images
from stainid.imaging.color import rgb_to_hed
from stainid.imaging.sources import open_source
from stainid.labels import sessions, store
from stainid.project import load_project
from stainid.sampling.tiles import table
from stainid.workflows.steps import describe


def _quiet(*args, **kwargs) -> None:
    pass


def _made(tmp_path, builder):
    built = builder(tmp_path / "dataset")
    folder = built[0] if isinstance(built, tuple) else built
    project = package.create_project(folder, tmp_path / "project")
    prepare.prepare_images(project, log=_quiet)
    return load_project(project.root), built


def test_tiles_package_is_rescaled_linked_and_its_raters_become_sessions(tmp_path):
    project, (folder, truth) = _made(tmp_path, tiles_with_points)
    assert project.kind == "images" and project.config["imaging"]["stains"]["4G8"] == {"family": "amyloid", "chromogen": "DAB", "antibody": "4G8"}
    assert project.config["dataset"]["class_map"]["cored"] == "compact plaque"
    cores = pd.read_csv(project.input("core_images"), dtype=str)
    assert list(cores.stain.unique()) == ["6E10"] and list(cores.antibody.unique()) == ["4G8"]
    assert set(cores.disease_group) == {"AD", "Control"} and set(cores.tma) == {"SD1", "SD2"}
    first = cores.iloc[0]
    assert first.image_path.endswith(".json") and int(first.native_width_px) == round(600 * 0.5 / 0.2738)
    # the picture is made in the project's cache, at the analysis scale; the dataset's own file is untouched
    with core_images.open_image(project.root / first.image_path) as image:
        assert image.size == (int(first.native_width_px), int(first.native_height_px))
    assert (project.root / "data" / "cache" / "images" / "tile0_6E10.png").exists()
    tiles = table(project.analysis_tiles())
    assert len(tiles) == 3 and set(tiles.width_px) == {int(first.native_width_px)}  # each image smaller than a tile is one tile
    placed = annotations.imported(project)
    assert set(placed.rater) == {"expert1", "expert2", "consensus"}
    # a point lands where the plaque is drawn, in core-image pixels (enlarged 1.83 times)
    scale = 0.5 / 0.2738
    mark = placed[(placed.rater == "expert1") & (placed.image_id == "tile0")].iloc[0]
    assert min(np.hypot(x * scale - mark.core_x, y * scale - mark.core_y) for x, y in truth["tile0"]) < 8 * scale
    # consensus: the 4 plaques both marked, not expert2's extra one
    assert (placed.rater == "consensus").sum() == 12
    names = sessions.names(project.output("reviews"))
    assert {"import-synthetic_points-expert1", "import-synthetic_points-expert2", "import-synthetic_points-consensus"} <= set(names)
    meta = sessions.meta(project.output("reviews"), "import-synthetic_points-consensus")
    assert meta["read_only"] and meta["stain"] == "6E10" and len(meta["fields"]) == 3
    labels = store.current(project.output("reviews"))
    assert set(labels.value[labels.session == "import-synthetic_points-expert2"]) == {"compact plaque", "diffuse plaque"}
    # importing again replaces the labels instead of doubling them
    annotations.import_package_labels(project, log=_quiet)
    labels = store.current(project.output("reviews"))
    assert (labels.session == "import-synthetic_points-expert1").sum() == 12


def test_scoring_against_points_per_rater_and_consensus(tmp_path):
    project, (folder, truth) = _made(tmp_path, tiles_with_points)
    tiles = table(project.analysis_tiles())
    scale = 0.5 / 0.2738
    found = {}
    for image, centres in truth.items():
        found[image] = [(x * scale, y * scale) for x, y in centres[:3]] + [(50.0, 50.0)]  # finds 3 of 4, one false alarm
    fake_plaques(project.root, tiles, found)
    result = scoring.score(project)
    rows = {(r["rater"], r["kind"]): r for r in result["rows"]}
    consensus = rows[("consensus", "objects")]
    assert consensus["marks"] == 12 and consensus["found"] == 9 and consensus["recall"] == 0.75
    assert consensus["ai_objects"] == 12 and consensus["ai_unmarked"] == 3 and consensus["precision"] == 0.75 and consensus["f1"] == 0.75
    assert rows[("expert2", "objects")]["marks"] == 15
    pair = result["agreement"][0]
    assert pair["rater"] == "expert1" and pair["other"] == "expert2" and pair["object_f1"] == pytest.approx(2 * 12 / 27, abs=1e-3)
    assert "found 75%" in result["summary"][0]
    assert scoring.scores_csv(project).exists() and scoring.saved(project)["rows"]


def test_regions_with_boxes_are_read_where_they_lie_and_recoloured(tmp_path):
    project, folder = _made(tmp_path, regions_with_boxes)
    cores = pd.read_csv(project.input("core_images"), dtype=str)
    assert cores.chromogen.tolist() == ["NovaRed", "NovaRed"]
    assert all(p.endswith(".json") for p in cores.image_path)  # NovaRed is turned into DAB, so the image is linked, not used as it is
    with core_images.open_image(project.root / cores.image_path.iloc[0]) as image:
        rgb = np.asarray(image.convert("RGB"))
    dab = rgb_to_hed(rgb)[..., 2]
    assert dab[300, 200] > 5 * dab[600, 600]  # the red deposit reads as brown stain, the tissue around it does not
    placed = annotations.imported(project)
    assert set(placed.rater) == {"consensus", "expert1"} and (placed.rater == "consensus").sum() == 6  # the package's own consensus, as it is
    assert set(placed.kind[placed.rater == "consensus"]) == {"box"} and set(placed.kind[placed.rater == "expert1"]) == {"polygon"}
    outlines = store.current(project.output("reviews"))
    outlines = outlines[(outlines.task == "outline") & (outlines.source == "import: synthetic_boxes expert1")]
    assert len(outlines) == 4 and all(len(points) == 4 for points in outlines.points)  # the polygons, on the tiles as outlines
    tiles = table(project.analysis_tiles())
    fake_plaques(project.root, tiles, {"r0": [(200.0, 300.0), (500.0, 300.0)], "r1": [(200.0, 450.0), (700.0, 700.0)]}, classes="diffuse")
    rows = {r["rater"]: r for r in scoring.score(project)["rows"]}
    assert rows["consensus"]["found"] == 3 and rows["consensus"]["marks"] == 6 and rows["consensus"]["ai_unmarked"] == 1
    assert rows["consensus"]["iou_recall"] is not None


def test_plain_dab_images_at_the_analysis_scale_are_used_as_they_are(tmp_path):
    project, folder = _made(tmp_path, image_labels)
    cores = pd.read_csv(project.input("core_images"), dtype=str)
    assert all(p.endswith(".png") and "raw" in p for p in cores.image_path)
    assert not (project.root / "data" / "images" / "links").exists()


def test_image_level_labels_and_kappa(tmp_path):
    project, folder = _made(tmp_path, image_labels)
    placed = annotations.imported(project)
    consensus = placed[placed.rater == "consensus"]
    assert sorted(consensus.image_id[consensus["class"] == "plaque"]) == ["c0", "c1"]  # 2 of 3 raters is a majority
    tiles = table(project.analysis_tiles())
    fake_plaques(project.root, tiles, {"c0": [(256.0, 256.0)], "c2": [(256.0, 256.0)]})
    result = scoring.score(project)
    image = next(r for r in result["rows"] if r["rater"] == "consensus" and r["kind"] == "image")
    assert image["class"] == "plaque" and image["images"] == 4 and image["found"] == 1 and image["missed"] == 1 and image["ai_unmarked"] == 1
    kappas = [p["image_kappa"] for p in result["agreement"]]
    assert 1.0 in kappas and min(kappas) < 1.0


def test_whole_slide_regions_are_tiled_and_points_follow_them(tmp_path):
    source = open_source(whole_slide(tmp_path / "probe")[0].parent / "raw" / "slide.tif")
    assert source.pixel_size_um == pytest.approx(0.5) and source.width == 3000
    assert len(source.levels) == (3 if source.reader == "tiff" else 1)  # OpenSlide reads the pyramid's base only when it is in sub-IFDs
    assert source.read(100, 100, 400, 400, len(source.levels) - 1).shape[:2] == tuple(round(400 / source.downsample(len(source.levels) - 1)) for _ in "xy")
    project, (folder, spots) = _made(tmp_path, whole_slide)
    assert project.kind == "slides" and project.config["imaging"]["rescale"] == "on_the_fly"
    cores = pd.read_csv(project.input("core_images"), dtype=str)
    assert len(cores) == 1 and set(cores.stain) == {"AT8"}  # the slide is smaller than one region: one region, cut down to its tissue
    box = [int(v) for v in cores.source_box.iloc[0].split()]
    assert box[0] > 50 and box[1] > 50 and box[0] + box[2] < 2950  # cut down to the tissue
    assert not (project.root / "data" / "cache" / "images").exists()  # on the fly: nothing is stored
    tiles = table(project.analysis_tiles())
    assert 2 <= len(tiles) < 6  # the empty tiles between the two pieces are left out
    placed = annotations.imported(project)
    assert len(placed) == 3 and set(placed.tile_id) <= set(tiles.tile_id)
    # with smaller regions each piece of tissue is a region of its own, and they do not overlap
    regions = prepare.tissue_regions(source, 0.5 / 0.2738, 0.5, region_px=2048)
    assert len(regions) >= 2 and all(not (a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3])
                                     for i, a in enumerate(regions) for b in regions[i + 1:])
    with core_images.open_image(project.root / cores.image_path.iloc[0]) as image:
        rgb = np.asarray(image.convert("RGB"))
    mark = placed[placed.core_id == cores.core_id.iloc[0]].iloc[0]
    assert rgb_to_hed(rgb)[int(mark.core_y), int(mark.core_x), 2] > 0.05  # the tangle is under its mark


def test_two_chromogen_slide_reads_only_the_stain_of_interest(tmp_path):
    folder, blue, brown = two_chromogens(tmp_path / "dataset")
    preview = package.preview(folder)
    entry = preview["stains"][0]
    assert entry["stain"] == "AT8+pTDP-43" and entry["family"] == "tau" and entry["chromogen"] == "blue" and entry["others"] == ["DAB"]
    project = package.create_project(folder, tmp_path / "project")
    prepare.prepare_images(project, log=_quiet)
    project = load_project(project.root)
    cores = pd.read_csv(project.input("core_images"), dtype=str)
    with core_images.open_image(project.root / cores.image_path.iloc[0]) as image:
        dab = rgb_to_hed(np.asarray(image.convert("RGB")))[..., 2]
    on_blue = min(dab[int(y), int(x)] for x, y in blue)
    on_brown = max(dab[int(y), int(x)] for x, y in brown)
    assert on_blue > 0.05 and on_brown < 0.3 * on_blue  # AT8 (blue) becomes the brown stain; pTDP-43 (brown) is left out


def test_chromogen_conversion_keeps_plain_dab_and_moves_fast_red_to_dab():
    rgb = paint(64, 64, [(32.0, 32.0, 10.0, "DAB")])
    assert chromogens.to_analysis_rgb(rgb, chromogens.recipe("DAB")) is rgb
    red = paint(64, 64, [(32.0, 32.0, 10.0, "FastRed")])
    converted = chromogens.to_analysis_rgb(red, chromogens.recipe("Fast Red"))
    reference = rgb_to_hed(rgb)[32, 32, 2]
    assert rgb_to_hed(converted)[32, 32, 2] == pytest.approx(reference, rel=0.25)
    assert rgb_to_hed(converted)[5, 5, 2] < 0.2 * reference


def test_setup_steps_follow_the_kind(tmp_path):
    project, _ = _made(tmp_path, tiles_with_points)
    ids = [s["id"] for s in describe(project)]
    assert "images" in ids and not {"slides", "dearray", "layout", "export", "qc", "tiles"} & set(ids)
    steps = {s["id"]: s for s in describe(project)}
    assert steps["images"]["progress"]["state"] == "done"
    assert steps["calibrate"]["needs"][0]["made_by"]["id"] == "images"
    from stainid.setup import guide

    screens = [s["id"] for s in guide.screens(project)]
    assert screens == ["images", "donors", "checks", "run"]


def test_image_list_from_a_folder_and_checks(tmp_path):
    from stainid.imageset import table as image_table
    from stainid.project import Project, save_project
    from stainid.setup.checks import apply_fix, project_issues

    folder = tmp_path / "scans"
    for donor, name in (("case1", "case1_4G8_frontal.png"), ("case1", "case1_AT8.png"), ("case2", "c2 PHF-1.tif")):
        (folder / donor).mkdir(parents=True, exist_ok=True)
        (folder / donor / name).write_bytes(b"")
    # an image keeps its id as the dataset writes it (its annotations name it so); ids are made only from file names
    kept = image_table.clean_rows([{"path": "a.png", "image_id": "2s-1-4G8_(AB17-24)_r1"}, {"path": "x/a b.png"}, {"path": "y/a b.png"}])
    assert [r["image_id"] for r in kept] == ["2s-1-4G8_(AB17-24)_r1", "a_b", "a_b_2"]
    rows = image_table.scan_folder(folder, ["4G8"], slides=False)
    assert [(r["donor_id"], r["stain"]) for r in rows] == [("case1", "4G8"), ("case1", "AT8"), ("case2", "PHF-1")]
    project = save_project(Project(root=tmp_path / "p", config=load_project(tmp_path / "p").config), {"kind": "images"})
    image_table.save_images(project, rows + [{"path": str(folder / "missing.png"), "stain": "IBA1", "donor_id": ""}])
    issues = {i["id"]: i for i in project_issues(project)}
    assert issues["images_without_donor"]["level"] == "problem" and issues["stains_not_analysed"]["level"] == "warning"
    assert "image_files_missing" in issues
    (project.root / "data" / "donor_metadata.csv").write_text("donor_id,disease_group\ncase1,AD\n")
    fix = next(i for i in project_issues(project) if i["id"] == "donors_not_in_table")["fixes"][0]
    assert apply_fix(project, fix["action"], fix["args"]) == "1 donor added to the donor table"


# ---------------------------------------------------------------- the web app's routes

@pytest.fixture()
def client(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from stainid.api import state
    from stainid.api.app import create_app

    home = tmp_path / "home"
    home.mkdir()
    (home / "stainid.yaml").write_text("name: Home\n")
    monkeypatch.setenv("STAINID_PROJECT", str(home))
    state.reset()
    client = TestClient(create_app(), client=("127.0.0.1", 50000))
    assert client.post("/api/session/admin", json={"password": "admin"}).status_code == 200
    yield client
    state.reset()


def test_package_import_setup_and_scoring_through_the_app(client, tmp_path):
    folder, truth = tiles_with_points(tmp_path / "dataset")
    preview = client.post("/api/projects/package-preview", json={"package": str(folder.parent)}).json()  # the dataset folder works too
    assert preview["kind"] == "images" and preview["images"] == 3 and preview["raters"] == {"expert1": 12, "expert2": 15}
    assert preview["stains"][0]["family"] == "amyloid" and preview["class_map"]["diffuse"] == "diffuse plaque" and preview["files_found"] == 3
    made = client.post("/api/projects/from-package", json={"package": str(folder), "path": str(tmp_path / "study"), "name": "Points",
                                                               "class_map": {"diffuse": "ignore"}}).json()
    assert made["kind"] == "images" and made["kind_label"] == "Image set"
    base = f"/api/projects/{made['id']}"
    setup = client.get(f"{base}/setup").json()
    assert [s["id"] for s in setup["screens"]] == ["images", "donors", "checks", "run"] and setup["set_up"]
    assert [r["id"] for r in setup["remaining"]][0] == "images"
    images = client.get(f"{base}/setup/images").json()
    assert images["total"] == 3 and images["donors"] == 2 and images["stains"][0]["analysed_as"] == "6E10"
    assert client.get(f"{base}/setup/images/tile0/picture.jpg?analysis=1").headers["content-type"] == "image/jpeg"
    changed = client.put(f"{base}/setup/images/imaging", json={"native_pixel_size_um": 0.5, "rescale": "on_import",
                                                                "stains": {"4G8": {"family": "amyloid", "chromogen": "Fast Red"}}}).json()
    assert changed["stains"][0]["chromogen"] == "FastRed" and changed["imaging"]["native_pixel_size_um"] == 0.5
    bad = client.put(f"{base}/setup/images/imaging", json={"stains": {"4G8": {"family": "glia"}}})
    assert bad.status_code == 400
    client.put(f"{base}/setup/images/imaging", json={"stains": {"4G8": {"family": "amyloid", "chromogen": "DAB"}}})
    project = load_project(tmp_path / "study")
    prepare.prepare_images(project, log=_quiet)  # what the Prepare images step runs
    imported = client.get(f"{base}/labels/imported").json()
    assert imported["placed"] == {"expert1": 12, "expert2": 12, "consensus": 12}  # expert2's diffuse plaque is ignored by the class map
    assert client.post(f"{base}/models/score").status_code == 400  # no detections yet
    tiles = table(project.analysis_tiles())
    fake_plaques(project.root, tiles, {image: [(x * 0.5 / 0.2738, y * 0.5 / 0.2738) for x, y in centres] for image, centres in truth.items()})
    scores = client.post(f"{base}/models/score").json()["scores"]
    assert {(r["rater"], r["recall"]) for r in scores["rows"]} == {("expert1", 1.0), ("expert2", 1.0), ("consensus", 1.0)}
    assert client.get(f"{base}/models/score.csv").text.startswith("stain,rater,kind")
    remapped = client.put(f"{base}/labels/imported/class-map", json={"class_map": {**imported["class_map"], "diffuse": "diffuse plaque"}}).json()
    assert remapped["placed"]["expert2"] == 15
    # imported sessions are read-only
    client.post("/api/session/annotator", json={"name": "Ann"})
    name = "import-synthetic_points-expert1"
    session = sessions.meta(project.output("reviews"), name)
    refused = client.post(f"{base}/annotate/{name}/labels", json={"records": [{"task": "class", "tile_id": session["fields"][0], "x": 5, "y": 5,
                                                                                  "value": "not a plaque"}]})
    assert refused.status_code == 403
    listed = {p["id"]: p for p in client.get("/api/projects").json()["projects"]}
    assert listed[made["id"]]["kind"] == "images" and listed[made["id"]]["status"]["state"] == "set_up"


def test_new_study_of_each_kind(client, tmp_path):
    for kind, rescale in (("images", "on_import"), ("slides", "on_the_fly")):
        made = client.post("/api/projects/create", json={"path": str(tmp_path / kind), "name": kind, "kind": kind}).json()
        project = load_project(tmp_path / kind)
        assert made["kind"] == kind and project.kind == kind and project.config["imaging"]["rescale"] == rescale
        steps = [s["id"] for s in client.get(f"/api/projects/{made['id']}/steps").json()]
        assert "images" in steps and "dearray" not in steps
        checks = client.get(f"/api/projects/{made['id']}/setup/checks").json()
        assert checks[0]["id"] == "no_images"
    assert client.post("/api/projects/create", json={"path": str(tmp_path / "x"), "kind": "puzzle"}).status_code == 400
    tma = client.post("/api/projects/create", json={"path": str(tmp_path / "tma"), "name": "tma"}).json()
    assert tma["kind"] == "tma" and "dearray" in [s["id"] for s in client.get(f"/api/projects/{tma['id']}/steps").json()]


def test_ome_zarr_is_read_level_by_level(tmp_path):
    from stainid.imaging.sources import available

    if not available()["zarr"]:
        pytest.skip('zarr does not work in this installation (pip install "stainid[slides]")')
    import zarr

    rgb = paint(400, 600, [(300.0, 200.0, 30.0, "DAB")])
    group = zarr.open_group(str(tmp_path / "slide.zarr"), mode="w")
    group.create_dataset("0", data=np.moveaxis(rgb, -1, 0), chunks=(3, 128, 128))
    group.create_dataset("1", data=np.moveaxis(rgb[::2, ::2], -1, 0), chunks=(3, 128, 128))
    axes = [{"name": "c", "type": "channel"}, {"name": "y", "type": "space", "unit": "micrometer"}, {"name": "x", "type": "space", "unit": "micrometer"}]
    group.attrs["multiscales"] = [{"version": "0.4", "axes": axes, "datasets": [
        {"path": "0", "coordinateTransformations": [{"type": "scale", "scale": [1, 0.11, 0.11]}]},
        {"path": "1", "coordinateTransformations": [{"type": "scale", "scale": [1, 0.22, 0.22]}]}]}]
    source = open_source(tmp_path / "slide.zarr")
    assert (source.width, source.height, len(source.levels)) == (600, 400, 2) and source.pixel_size_um == pytest.approx(0.11)
    assert np.array_equal(source.read(250, 150, 100, 100), rgb[150:250, 250:350])
    reduced = source.region((0, 0, 600, 400), 0.11 / 0.2738)  # to the analysis scale: read from the coarser level
    assert reduced.shape[:2] == (round(400 * 0.11 / 0.2738), round(600 * 0.11 / 0.2738))
