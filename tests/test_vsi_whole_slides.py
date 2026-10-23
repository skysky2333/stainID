"""Whole-slide Olympus .vsi files read through Bio-Formats (the scan's series and its pyramid, not the label picture),
and adding a project to the app's list from the command line."""
from __future__ import annotations

import json

import numpy as np
import pytest

from stainid.imaging import sources
from stainid.imaging.sources import Series, Source, scan_series

# The series of a real .vsi (JOG 84 MFG 6E10): label, overview and their copies, a narrow first scan area over the
# marker ink, the scan of the section with its reduced copies, and the macro picture (no pixel size).
VSI = [Series(0, "label", 8021, 9366, 2.738), Series(1, "label resolution 1", 4011, 4683, 5.476),
       Series(6, "overview", 18032, 9148, 2.738), Series(7, "overview resolution 1", 9016, 4574, 5.476),
       Series(13, "20x_BF_01", 1420, 11190, 0.2738), Series(14, "20x_BF_01 resolution 1", 710, 5595, 0.5476),
       Series(19, "20x_BF_02", 111313, 79615, 0.2738), Series(21, "20x_BF_02 resolution 2", 27829, 19904, 1.0952),
       Series(20, "20x_BF_02 resolution 1", 55657, 39808, 0.5476), Series(28, "macro image", 991, 375, None)]


def test_the_scan_and_its_pyramid_are_chosen():
    levels = scan_series(VSI)
    assert [s.index for s in levels] == [19, 20, 21]  # the section's scan, finest first; not the label or the ink strip
    assert levels[0].pixel_size_um == pytest.approx(0.2738)
    assert [s.index for s in scan_series([VSI[0], VSI[4]])] == [13]  # a scan smaller than its label is still the scan
    assert [s.index for s in scan_series([Series(0, "image", 500, 400, None)])] == [0]  # no pixel size anywhere
    with pytest.raises(ValueError):
        scan_series([])


class FakeHandle:
    """A Bio-Formats reader stand-in: a gradient image and its 2x reduction."""

    def __init__(self):
        full = np.zeros((40, 60, 3), np.uint8)
        full[..., 0] = np.arange(60)[None, :]
        full[..., 1] = np.arange(40)[:, None]
        self.images = [full, full[::2, ::2]]
        self.levels = [Series(5, "scan", 60, 40, 0.27), Series(6, "scan resolution 1", 30, 20, 0.54)]
        self.reads = []

    def read(self, level, x0, y0, x1, y1):
        self.reads.append((level, x0, y0, x1, y1))
        image = self.images[level]
        return image[max(0, y0):max(0, y1), max(0, x0):max(0, x1)]


def fake_source() -> Source:
    handle = FakeHandle()
    return Source(sources.Path("slide.vsi"), 60, 40, [(60, 40), (30, 20)], 0.27, "bioformats", handle)


def test_regions_are_read_from_the_right_level_with_glass_beyond_the_edge():
    source = fake_source()
    box = source.read(10, 4, 8, 6)
    assert box.shape == (6, 8, 3) and box[0, 0, 0] == 10 and box[0, 0, 1] == 4
    small = source.region((0, 0, 60, 40), 0.5)  # half size: read from the reduced copy
    assert small.shape == (20, 30, 3) and source._handle.reads[-1][0] == 1
    edge = source.read(-4, -2, 10, 6)  # starts before the image: white glass there, the image after it
    assert edge.shape == (6, 10, 3)
    assert (edge[:2] == 255).all() and (edge[:, :4] == 255).all()
    assert edge[2, 4, 0] == 0 and edge[2, 4, 1] == 0
    past = source.read(55, 35, 10, 10)  # runs past the far edge
    assert past.shape == (10, 10, 3) and (past[5:] == 255).all() and past[0, 0, 0] == 55


def test_a_vsi_opens_through_the_scan_series(monkeypatch, tmp_path):
    made = []

    class Handle(FakeHandle):
        def __init__(self, path):
            super().__init__()
            made.append(path)

    monkeypatch.setattr(sources, "_BioHandle", Handle)
    monkeypatch.setattr(sources, "_BIO_HANDLES", {})
    slide = tmp_path / "JOG 84 MFG 6E10.vsi"
    slide.write_bytes(b"vsi")
    first = sources.open_source(slide)
    again = sources.open_source(slide)
    assert (first.width, first.height, first.levels, first.pixel_size_um, first.reader) == (60, 40, [(60, 40), (30, 20)], 0.27, "bioformats")
    assert len(made) == 1 and again._handle is first._handle  # one reader per file and process
    assert sources.describe(slide)["width_px"] == 60


def test_projects_are_added_to_the_list_without_opening_them(monkeypatch, tmp_path):
    from stainid.api import registry
    from stainid.cli import main

    monkeypatch.setattr(registry, "SETTINGS", tmp_path / "home" / "app.json")
    study, extra = tmp_path / "study", tmp_path / "extra"
    for folder, name in ((study, "Study"), (extra, "Extra slides")):
        folder.mkdir()
        (folder / "stainid.yaml").write_text(f"name: {name}\nkind: slides\n")
    registry.remember(study)
    main(["projects", "add", str(extra)])
    settings = json.loads(registry.SETTINGS.read_text())
    assert settings["recent"] == [str(study.resolve()), str(extra.resolve())]
    assert str(extra.resolve()) not in settings["opened"]  # not opened: the study stays first
    assert [e.name for e in registry.entries()] == ["Study", "Extra slides"]
    assert registry.add(extra) is False
    main(["projects", "remove", str(extra)])
    assert [e.name for e in registry.entries()] == ["Study"]
    main(["--project", str(extra), "projects", "add"])  # added again: shown again
    assert [e.name for e in registry.entries()] == ["Study", "Extra slides"]
    with pytest.raises(SystemExit):
        main(["projects", "add", str(tmp_path)])  # not a project
