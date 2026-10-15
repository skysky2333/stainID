"""Core grid pictures without decoding whole core images: thumbnails from the quality pictures, written by export, and
the stand-ins cut from the slide picture."""
from __future__ import annotations

import base64
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from stainid.imaging import thumbnails
from stainid.qc.core_qc import OUTLINE_COLOURS, write_overlay
from stainid.slides.export import write_png

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from stainid.api import pyramid, state  # noqa: E402
from stainid.api.app import create_app  # noqa: E402

CORE = Path("data/cores/LIP-1/B-2")
CORE_COLOUR = (170, 120, 110)  # BGR, of the core on the slide pictures below (bluish, as haematoxylin)


@pytest.fixture()
def client(synthetic_project, monkeypatch):
    monkeypatch.chdir(synthetic_project)
    state.reset()
    admin = TestClient(create_app(), client=("127.0.0.1", 50000))
    assert admin.post("/api/session/admin", json={"password": "admin"}).status_code == 200
    yield admin
    state.reset()


def _quality_picture(project: Path, stain: str) -> Path:
    """What Check core quality keeps of a core image: the image reduced, the tissue and core outlines drawn on."""
    rgb = cv2.cvtColor(cv2.imread(str(project / CORE / f"{stain}.png")), cv2.COLOR_BGR2RGB)[::2, ::2].copy()
    tissue = np.zeros(rgb.shape[:2], bool)
    tissue[100:500, 120:480] = True
    footprint = np.zeros(rgb.shape[:2], np.uint8)
    cv2.circle(footprint, (300, 300), 280, 1, -1)
    path = project / "data" / "qc" / "cores" / "LIP-1" / "B-2" / f"{stain}.png"
    write_overlay(rgb, tissue, footprint.astype(bool), [], path)
    return path


def _no_outline_colours(bgr: np.ndarray) -> bool:
    b, g, r = (bgr[..., i].astype(int) for i in range(3))
    return not ((g - np.maximum(r, b) > 60).any() or ((r - b > 120) & (g > 120)).any())


def test_without_outlines_recovers_the_plain_picture():
    plain = np.full((300, 300, 3), (200, 190, 210), np.uint8)
    cv2.circle(plain, (150, 150), 60, (90, 60, 140), -1)
    drawn = plain.copy()
    for radius, colour in zip((130, 100, 40), OUTLINE_COLOURS):
        cv2.circle(drawn, (150, 150), radius, colour, 2)
    clean = thumbnails.without_outlines(drawn)
    assert not any(np.all(clean == colour, axis=2).any() for colour in OUTLINE_COLOURS)
    assert np.abs(clean.astype(int) - plain).max() < 40 and np.abs(clean.astype(int) - plain).mean() < 1
    assert thumbnails.without_outlines(plain) is plain  # nothing drawn: nothing to do


def test_export_writes_the_thumbnail_base_after_the_core_image(tmp_path):
    rgb = np.zeros((3000, 2500, 3), np.uint8)
    rgb[..., 0] = 200  # red, in RGB
    image, thumb = tmp_path / "cores" / "A-1.png", tmp_path / "thumbs" / "LIP-1_A-1_AT8_base.jpg"
    write_png(rgb, image, thumb)
    base = cv2.imread(str(thumb))
    assert base.shape[:2] == (2048, round(2500 * 2048 / 3000))
    assert abs(int(base[..., 2].mean()) - 200) < 3 and base[..., 0].mean() < 3  # BGR on disk: the colour is right
    assert thumb.stat().st_mtime >= image.stat().st_mtime  # newer than the image: counts as made from this version


def test_grid_thumbnail_comes_from_the_quality_picture_without_decoding_the_core(client, synthetic_project, monkeypatch):
    _quality_picture(synthetic_project, "AT8")
    monkeypatch.setattr(pyramid, "_read", lambda *args: pytest.fail("the whole core image was decoded"))
    grid = client.get("/api/tmas/1/grid?stain=AT8").json()
    assert grid["cells"][0]["thumb"] is True and grid["thumbs_waiting"] == 0  # ask for it at once: nothing queued
    picture = client.get("/api/images/cores/LIP-1_B-2/AT8.jpg?size=256")
    assert picture.status_code == 200
    bgr = cv2.imdecode(np.frombuffer(picture.content, np.uint8), cv2.IMREAD_COLOR)
    assert bgr.shape[:2] == (256, 256) and _no_outline_colours(bgr)
    assert pyramid.picture(state.get_state(), "LIP-1_B-2", "AT8", prepare=False)["thumb_size"] == 1024  # the viewer's stand-in
    assert client.get("/api/tmas/1/grid?stain=AT8").json()["cells"][0]["placeholder"] is None  # made: no stand-in needed


def test_quality_picture_of_an_older_image_is_not_used(synthetic_project, client):
    old = _quality_picture(synthetic_project, "NeuN")
    source = synthetic_project / CORE / "NeuN.png"
    os.utime(old, (source.stat().st_mtime - 60,) * 2)
    assert pyramid.thumbnail_state(state.get_state(), "LIP-1_B-2", "NeuN", 256) == "slow"


def _manifest(project: Path, slide: str, scale: float) -> np.ndarray:
    """One core of TMA 1 found at (900, 700) preview px, 400 px across, on a slide whose full scan is 4x the preview."""
    pd.DataFrame([{"slide": slide, "slide_path": f"/scans/{slide}.vsi", "tma": "1", "stain": "AT8", "core_label": "B-2",
                   "center_x_px": 3600, "center_y_px": 2800, "lattice_center_x_px": 3640, "lattice_center_y_px": 2800,
                   "diameter_x_px": 1600, "diameter_y_px": 1600, "preview_scale_x": 4, "preview_scale_y": 4,
                   "provisional_tissue_fraction": 0.81}]).to_csv(project / "data" / "core_manifest.csv", index=False)
    picture = np.full((round(1400 * scale), round(1800 * scale), 3), 235, np.uint8)
    cv2.circle(picture, (round(900 * scale), round(700 * scale)), round(190 * scale), CORE_COLOUR, -1)
    return picture


def _placeholder(client) -> np.ndarray:
    cell = client.get("/api/tmas/1/grid?stain=AT8&prepare=false").json()["cells"][0]
    assert cell["thumb"] is False and cell["placeholder"].startswith("data:image/jpeg;base64,")
    return cv2.imdecode(np.frombuffer(base64.b64decode(cell["placeholder"].split(",", 1)[1]), np.uint8), cv2.IMREAD_COLOR)


def test_grid_stand_in_is_cut_from_the_plain_slide_preview(client, synthetic_project):
    picture = _manifest(synthetic_project, "TMA LIP-1 AT8", 0.5)
    previews = synthetic_project / "data" / "qc" / "grids" / "previews"
    previews.mkdir(parents=True)
    cv2.imwrite(str(previews / "TMA_LIP-1_AT8.jpg"), picture)
    (previews / "TMA_LIP-1_AT8.json").write_text(json.dumps({"scale": 0.5}))
    small = _placeholder(client)
    assert small.shape[:2] == (32, 32)
    assert np.abs(small[16, 16].astype(int) - CORE_COLOUR).max() < 12  # the core in the middle
    assert small[0, 0].min() > 200  # the slide around it in the corners


def test_grid_stand_in_from_the_drawn_on_grid_picture_has_the_drawing_painted_over(client, synthetic_project):
    picture = _manifest(synthetic_project, "TMA LIP-1 AT8", 1.0)
    cv2.circle(picture, (900, 700), 200, (0, 0, 255), 5)  # as draw_qc marks a core
    cv2.drawMarker(picture, (900, 700), (0, 0, 255), cv2.MARKER_CROSS, 24, 3)
    cv2.putText(picture, "B-2 0.81", (700, 700), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    grids = synthetic_project / "data" / "qc" / "grids"
    grids.mkdir(parents=True)
    cv2.imwrite(str(grids / "TMA_LIP-1_AT8.jpg"), picture)
    small = _placeholder(client).astype(int)
    assert (small[..., 2] - small[..., 0] < 30).all()  # no red left
    assert np.abs(small[16, 16] - CORE_COLOUR).max() < 12
