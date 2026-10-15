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
    assert set(placed.kind) == {"box"} and set(placed.rater) == {"consensus"}  # the package's own consensus is used as it is
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
