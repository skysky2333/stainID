"""Dataset packages as public datasets really come: downloaded bit by bit (images not downloaded yet), with stain colours
measured on the dataset's own slides, whole slides as plain Zarr arrays with JPEG-XR chunks, and many such projects on
one drive (the drive keeps a floor of free space)."""
from __future__ import annotations

import json
import os
import shutil
import time
from collections import namedtuple
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from package_files import paint, tiles_with_points
from PIL import Image

from stainid import disk_cache
from stainid.imageset import package, prepare, table
from stainid.imageset.checks import image_issues
from stainid.imaging import chromogens, settings
from stainid.imaging.color import rgb_to_hed
from stainid.imaging.sources import open_source
from stainid.project import load_project
from stainid.setup.checks import apply_fix

HOUR = 3600


def _quiet(*args, **kwargs) -> None:
    pass


def _partly_downloaded(root: Path) -> tuple[Path, Path]:
    """The three-tile package with tile1 not downloaded (no path) and tile2's file not there yet (path given)."""
    folder, _ = tiles_with_points(root)
    images = pd.read_csv(folder / "images.csv", dtype=str, keep_default_na=False)
    later = root / "raw" / "tiles" / "tile2.png"
    shutil.move(later, root / "tile2_later.png")
    images.loc[images.image_id == "tile1", "path"] = ""
    images.to_csv(folder / "images.csv", index=False)
    return folder, later


def test_images_not_downloaded_yet_are_counted_left_out_and_added_when_they_arrive(tmp_path):
    folder, later = _partly_downloaded(tmp_path / "dataset")
    preview = package.preview(folder)
    files = preview["files"]
    assert (files["listed"], files["present"], files["not_downloaded"], files["missing"]) == (3, 1, 1, 1)
    assert files["missing_examples"] == [str(later)]
    assert preview["stains"][0]["images"] == 3 and preview["stains"][0]["present"] == 1
    assert preview["cache_limit_gb"] == package.CACHE_LIMIT_GB

    project = package.create_project(folder, tmp_path / "project")
    assert [r["image_id"] for r in table.read_images(project)] == ["tile0"]  # the present one only; no package folder as an "image"
    assert project.config["cache_limit_gb"] == package.CACHE_LIMIT_GB
    status = package.dataset_files(project)
    assert (status["present"], status["in_study"], status["new"], status["not_downloaded"], status["missing"]) == (1, 1, 0, 1, 1)
    note = {i["id"]: i for i in image_issues(project)}["dataset_not_downloaded"]
    assert note["level"] == "note" and note["title"] == "Downloaded so far: 1 of 3 images of the dataset"
    prepare.prepare_images(project, log=_quiet)  # the study runs with what is there

    shutil.move(tmp_path / "dataset" / "tile2_later.png", later)  # downloaded since
    images = pd.read_csv(folder / "images.csv", dtype=str, keep_default_na=False)
    images.loc[images.image_id == "tile1", "path"] = "../raw/tiles/tile1.png"  # the data agent fills in the path
    images.to_csv(folder / "images.csv", index=False)
    project = load_project(project.root)
    assert package.dataset_files(project, fresh=True)["new"] == 2
    issues = {i["id"]: i for i in image_issues(load_project(project.root))}
    assert issues["dataset_new_files"]["fixes"][0]["action"] == "add_dataset_files"
    rows = table.read_images(project)
    rows[0]["region"] = "changed by hand"
    table.save_images(project, rows)
    assert apply_fix(project, "add_dataset_files", {}) == "2 newly downloaded images added to the image list"
    rows = table.read_images(project)
    assert [r["image_id"] for r in rows] == ["tile0", "tile1", "tile2"] and rows[0]["region"] == "changed by hand"
    assert apply_fix(project, "add_dataset_files", {}).startswith("No new files yet")


def test_a_package_with_nothing_downloaded_still_makes_a_study(tmp_path):
    folder, _ = tiles_with_points(tmp_path / "dataset")
    images = pd.read_csv(folder / "images.csv", dtype=str, keep_default_na=False)
    images["path"] = ""
    images.to_csv(folder / "images.csv", index=False)
    assert package.preview(folder)["files"]["present"] == 0
    project = package.create_project(folder, tmp_path / "project")
    assert table.read_images(project) == []
    issue = image_issues(project)[0]
    assert issue["level"] == "problem" and issue["title"] == "None of the dataset's 3 images are downloaded yet"
    assert issue["fixes"][0]["action"] == "add_dataset_files"


# ---------------------------------------------------------------- measured stain colours

MEASURED = {"blue": [0.7982, 0.5518, 0.2415], "DAB": [0.3683, 0.5634, 0.7396], "hematoxylin": [0.6848, 0.628, 0.3696]}  # SEA-AD AT slide


def _paint_with(vectors: dict[str, list[float]], spots: list[tuple[float, float, str]], size: int = 256) -> np.ndarray:
    unit = {k: np.array(v) / np.linalg.norm(v) for k, v in vectors.items()}
    od = np.full((size, size, 3), 0.02) + 0.2 * unit["hematoxylin"]
    yy, xx = np.mgrid[:size, :size]
    for x, y, chromogen in spots:
        od[(xx - x) ** 2 + (yy - y) ** 2 <= 18 ** 2] += 0.9 * unit[chromogen]
    return np.clip(255 * 10 ** (-od), 0, 255).astype(np.uint8)


def _two_colour_package(root: Path) -> tuple[Path, list[tuple[float, float]], list[tuple[float, float]]]:
    raw = root / "raw"
    raw.mkdir(parents=True)
    blue, brown = [(70.0, 70.0), (190.0, 180.0)], [(190.0, 60.0), (60.0, 190.0)]
    rgb = _paint_with(MEASURED, [(x, y, "blue") for x, y in blue] + [(x, y, "DAB") for x, y in brown])
    Image.fromarray(rgb).save(raw / "at.png")
    meta = {"id": "measured", "title": "Measured colours", "image_kind": "tiles",
            "stains": [{"stain": "AT8+pTDP-43", "family": "tau", "chromogen": "blue", "also_on_slide": [{"stain": "pTDP-43", "chromogen": "DAB"}],
                        "vectors": {"blue": MEASURED["blue"], "DAB": MEASURED["DAB"], "Hematoxylin": MEASURED["hematoxylin"]}}]}
    folder = root / "package"
    folder.mkdir()
    (folder / "dataset.yaml").write_text(yaml.safe_dump(meta, sort_keys=False))
    pd.DataFrame([{"image_id": "at1", "path": str(raw / "at.png"), "pixel_size_um": 0.2738, "stain": "AT8+pTDP-43", "donor_id": "T1"}]).to_csv(
        folder / "images.csv", index=False)
    return folder, blue, brown


def test_measured_stain_vectors_are_carried_into_the_project_and_used_in_unmixing(tmp_path):
    folder, blue, brown = _two_colour_package(tmp_path / "dataset")
    entry = package.preview(folder)["stains"][0]
    assert entry["vectors"] == MEASURED  # "Hematoxylin" named as stainID names it
    project = package.create_project(folder, tmp_path / "project")
    assert project.config["imaging"]["stains"]["AT8+pTDP-43"]["vectors"] == MEASURED
    recipe = settings.recipe_for(project, "AT8+pTDP-43")
    assert recipe.chromogens == ("hematoxylin", "blue", "DAB") and recipe.vectors == tuple(tuple(MEASURED[c]) for c in recipe.chromogens)

    rgb = np.asarray(Image.open(tmp_path / "dataset" / "raw" / "at.png").convert("RGB"))
    measured = chromogens.amounts(rgb, recipe)[..., 1]  # the blue (AT8) amount
    standard = chromogens.amounts(rgb, chromogens.recipe("blue", ["DAB"]))[..., 1]
    leak = lambda found: max(found[int(y), int(x)] for x, y in brown) / min(found[int(y), int(x)] for x, y in blue)  # noqa: E731
    assert leak(measured) < 0.02 < leak(standard)  # with the dataset's own colours, pTDP-43 (brown) no longer leaks into AT8

    prepare.prepare_images(project, log=_quiet)
    cores = pd.read_csv(load_project(project.root).input("core_images"), dtype=str)
    from stainid.imaging import core_images

    with core_images.open_image(project.root / cores.image_path.iloc[0]) as image:
        dab = rgb_to_hed(np.asarray(image.convert("RGB")))[..., 2]
    assert min(dab[int(y), int(x)] for x, y in blue) > 10 * max(dab[int(y), int(x)] for x, y in brown)


def test_stain_vectors_are_checked_and_kept_by_the_imaging_card(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from stainid.api import state
    from stainid.api.app import create_app

    folder, _, _ = _two_colour_package(tmp_path / "dataset")
    project = package.create_project(folder, tmp_path / "project")
    monkeypatch.setenv("STAINID_PROJECT", str(project.root))
    state.reset()
    client = TestClient(create_app(), client=("127.0.0.1", 50000))
    assert client.post("/api/session/admin", json={"password": "admin"}).status_code == 200
    overview = client.get("/api/setup/images").json()
    stain = overview["stains"][0]
    assert stain["vectors"] == MEASURED and stain["counterstain"] == "hematoxylin"
    assert overview["dataset_files"]["present"] == overview["dataset_files"]["in_study"] == 1
    draft = {"family": stain["family"], "chromogen": stain["chromogen"], "others": stain["others"], "vectors": stain["vectors"]}
    saved = client.put("/api/setup/images/imaging", json={"rescale": "on_import", "stains": {stain["stain"]: draft}}).json()
    assert saved["stains"][0]["vectors"] == MEASURED  # saving the card keeps the measured colours
    bad = client.put("/api/setup/images/imaging", json={"stains": {stain["stain"]: {**draft, "vectors": {"blue": [1, 2]}}}})
    assert bad.status_code == 400 and "three numbers" in bad.json()["detail"]
    found = client.post("/api/setup/images/look-for-new").json()
    assert found["added"] == 0 and found["overview"]["total"] == 1
    state.reset()


def test_clean_vectors_names_and_checks_colours():
    assert chromogens.clean_vectors({"Fast Red": [0.2, 0.8, 0.5]}) == {"FastRed": [0.2, 0.8, 0.5]}
    for bad in ({"glitter": [1, 1, 1]}, {"DAB": [0, 0, 0]}, {"DAB": "brown"}, [0.3, 0.5, 0.7]):
        with pytest.raises(ValueError):
            chromogens.clean_vectors(bad)


# ---------------------------------------------------------------- whole slides as plain Zarr v2 arrays

def _zarr_slide(path: Path, rgb: np.ndarray, chunk: int, compressor: dict | None, encode, skip: set[tuple[int, int]] = frozenset()) -> None:
    path.mkdir(parents=True)
    (path / ".zarray").write_text(json.dumps({"chunks": [chunk, chunk, 3], "compressor": compressor, "dtype": "|u1", "fill_value": 0,
                                              "filters": None, "order": "C", "shape": list(rgb.shape), "zarr_format": 2}))
    for cy in range(-(-rgb.shape[0] // chunk)):
        for cx in range(-(-rgb.shape[1] // chunk)):
            if (cy, cx) in skip:
                continue
            block = np.zeros((chunk, chunk, 3), np.uint8)  # unscanned parts of a chunk are black (fill value 0)
            part = rgb[cy * chunk:(cy + 1) * chunk, cx * chunk:(cx + 1) * chunk]
            block[:part.shape[0], :part.shape[1]] = part
            (path / f"{cy}.{cx}.0").write_bytes(encode(block))


def _tissue(height: int = 700, width: int = 900) -> np.ndarray:
    rgb = np.full((height, width, 3), 240, np.uint8)
    rgb[100:600, 150:750] = paint(500, 600, [(100.0, 100.0, 20.0, "DAB"), (400.0, 300.0, 20.0, "DAB")], seed=3)
    return rgb


@pytest.mark.parametrize("codec", ["raw", "jpegxr"])
def test_a_plain_zarr_slide_is_read_chunk_by_chunk(tmp_path, codec):
    rgb = _tissue()
    if codec == "jpegxr":
        imagecodecs = pytest.importorskip("imagecodecs")
        if not imagecodecs.JPEGXR.available:
            pytest.skip("this imagecodecs has no JPEG-XR")
        compressor, encode, tolerance = {"id": "imagecodecs_jpegxr", "level": 1.0}, lambda b: imagecodecs.jpegxr_encode(b, level=1.0), 3.0
    else:
        compressor, encode, tolerance = None, lambda b: b.tobytes(), 0.0
    path = tmp_path / "slide.zarr"
    _zarr_slide(path, rgb, 256, compressor, encode, skip={(0, 3)})
    source = open_source(path)
    assert source.reader == "zarr_array" and (source.width, source.height) == (900, 700) and source.pixel_size_um is None
    assert len(source.levels) == 2 and source.levels[1] == (450, 350)  # reduced levels made while reading, down to ~256 px
    box = source.read(200, 150, 400, 300)  # across four chunks
    assert box.shape == (300, 400, 3) and np.abs(box.astype(float) - rgb[150:450, 200:600]).mean() <= tolerance
    corner = source.read(800, 0, 100, 100)  # chunk (0, 3) is not stored, and the stored ones end in black: glass
    assert (corner == 255).all()
    edge = source.read(850, 650, 100, 100)  # runs past the slide: white
    assert edge.shape == (100, 100, 3) and edge[60:, 60:].min() == 255
    small = source.read(0, 0, 900, 700, level=1)
    assert small.shape == (350, 450, 3) and np.abs(small.astype(float) - rgb[::2, ::2]).mean() < 6
    thumb, scale = source.thumbnail(300)
    assert thumb.shape[1] == 300 and scale == pytest.approx(300 / 900)


def test_a_whole_slide_package_of_plain_zarr_slides_is_prepared(tmp_path):
    root = tmp_path / "dataset"
    _zarr_slide(root / "raw" / "s1.zarr", _tissue(1400, 1800), 512, None, lambda b: b.tobytes())
    folder = root / "package"
    folder.mkdir()
    (folder / "dataset.yaml").write_text(yaml.safe_dump({"id": "zarrs", "title": "Zarr slides", "image_kind": "wsi", "pixel_size_um": 0.5,
                                                         "stains": [{"stain": "AT8", "chromogen": "DAB"}]}))
    pd.DataFrame([{"image_id": "s1", "path": "../raw/s1.zarr", "stain": "AT8", "donor_id": "Z1"},
                  {"image_id": "s2", "path": "", "stain": "AT8", "donor_id": "Z2"}]).to_csv(folder / "images.csv", index=False)
    assert package.preview(folder)["files"] == {"listed": 2, "present": 1, "not_downloaded": 1, "missing": 0, "missing_examples": []}
    project = package.create_project(folder, tmp_path / "project")
    prepare.prepare_images(project, log=_quiet)
    cores = pd.read_csv(load_project(project.root).input("core_images"), dtype=str)
    assert set(cores.donor_id) == {"Z1"} and len(cores) >= 1


# ---------------------------------------------------------------- the drive's floor of free space

Usage = namedtuple("Usage", "total used free")


def _cache_file(root: Path, relative: str, size: int, age: float) -> Path:
    path = root / "data" / "cache" / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)
    when = time.time() - age

    os.utime(path, (when, when))
    return path


def test_min_free_space_floor():
    assert disk_cache.min_free_bytes(2000 * disk_cache.GB) == 50 * disk_cache.GB  # the smaller of 50 GB and 10 %
    assert disk_cache.min_free_bytes(200 * disk_cache.GB) == 20 * disk_cache.GB
    assert disk_cache.parse_min_free("80G, 5%") == [("bytes", 80e9), ("share", 0.05)]
    for bad in ("", "lots", "150%"):
        with pytest.raises(ValueError):
            disk_cache.parse_min_free(bad)


def test_a_drive_low_on_space_takes_the_least_recently_used_from_every_project_on_it(tmp_path, monkeypatch):
    first, second = tmp_path / "first", tmp_path / "second"
    for root in (first, second):
        root.mkdir()
        (root / "stainid.yaml").write_text("name: x\ncache_limit_gb: 100\n")  # far from their own limits
    oldest = _cache_file(first, "thumbs/A_base.jpg", 4000, 5 * HOUR)
    older = _cache_file(second, "fields/B.jpg", 3000, 4 * HOUR)
    newer = _cache_file(first, "fields/C.jpg", 3000, 2 * HOUR)
    in_use = _cache_file(second, "fields/D.jpg", 3000, 0)
    for root in (first, second):
        disk_cache.scan(root)
    monkeypatch.setattr(disk_cache, "_KNOWN", set())
    monkeypatch.setattr(disk_cache, "_WATCHED", set())
    disk_cache.know(first, second)
    monkeypatch.setenv(disk_cache.MIN_FREE_ENV, "10K")  # 10,000 bytes
    free = {"bytes": 20_000}
    monkeypatch.setattr(disk_cache.shutil, "disk_usage", lambda path: Usage(10**6, 10**6 - free["bytes"], free["bytes"]))
    assert disk_cache.free_space() == []  # enough free space: nothing is touched
    free["bytes"] = 5000  # below the floor of 10,000: back to 10,000 / 0.9 = 11,111, so 6,111 bytes are needed
    done = disk_cache.free_space()
    assert done[0]["files"] == 2 and done[0]["bytes"] == 7000
    assert not oldest.exists() and not older.exists() and newer.exists() and in_use.exists()  # across both projects, oldest first
    usage = disk_cache.usage(second)
    assert usage["last_cleanup"]["reason"] == "space"
    assert usage["drive"] == {"free_bytes": 5000, "total_bytes": 10**6, "min_free_bytes": 10000, "min_free_setting": "10K"}
    free["bytes"] = 0  # nothing left that is not in use: what is open stays
    disk_cache.free_space()
    assert in_use.exists() and not newer.exists()
