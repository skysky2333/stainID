"""Excluded pixels are as if they were not there: a stained blob in a detected fold changes no candidate and no stage-2
measure (stain, tissue, nuclei, background, context); the slide calibration leaves the excluded pixels out; and people's
area edits do not touch the analysis (they only train and evaluate models)."""
from __future__ import annotations

import cv2
import numpy as np
from test_v2_pipeline import BLUE, BROWN, PALE, PS, blank, v2_project  # noqa: F401 - the fixture

from stainid.project import load_project
from stainid.v2 import proposals, run
from stainid.v2.field import Field
from stainid.v2.params import DEFAULTS

SETTINGS = DEFAULTS["proposals"]
SHARED = {k: v for k, v in SETTINGS.items() if k not in ("NeuN", "6E10", "AT8")}
MEASURES = ("analysis_area_mm2", "valid_area_mm2", "dab_area_mm2_at_0.5", "dab_area_mm2_at_1", "dab_area_mm2_at_2", "nucleus_count",
            "amyloid_extent_area_mm2", "at8_valid_area_mm2", "at8_positive_area_mm2", "at8_noncompact_area_mm2", "thread_skeleton_px",
            "seed_threshold_dab_od", "background_median_dab_od")


def _scene(stain: str, blob: bool) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(3)
    rgb = blank(600)
    rgb[:] = np.clip(rgb.astype(int) - rng.integers(0, 12, rgb.shape), 0, 255).astype(np.uint8)  # textured tissue
    nuclei = np.zeros(rgb.shape[:2], np.int32)
    for k, (x, y) in enumerate([(120, 120), (480, 140), (130, 470)], start=1):
        if stain == "6E10":
            cv2.circle(rgb, (x, y), 34, PALE, -1)
            cv2.circle(rgb, (x, y), 16, BROWN, -1)
        else:
            cv2.circle(rgb, (x, y), 15, BROWN, -1)
            cv2.circle(rgb, (x, y), 7, BLUE, -1)
            cv2.circle(nuclei, (x, y), 7, k, -1)
    if blob:  # a stained blob (and a nucleus) in the middle of the excluded area
        cv2.circle(rgb, (440, 440), 30, BROWN, -1)
        cv2.circle(rgb, (440, 440), 8, BLUE, -1)
        cv2.circle(nuclei, (440, 440), 8, 9, -1)
    return rgb, nuclei.astype(np.uint32)


def _square(y0: int, y1: int, x0: int, x1: int, shape=(600, 600)) -> np.ndarray:
    out = np.zeros(shape, bool)
    out[y0:y1, x0:x1] = True
    return out


FOLD = _square(340, 540, 340, 540)


def _field(stain: str, rgb: np.ndarray, nuclei: np.ndarray, excluded: np.ndarray | None) -> Field:
    h, w = rgb.shape[:2]
    tile = {"tile_id": "T_W0101", "stain": stain, "x_px": 0, "y_px": 0, "width_px": w, "height_px": h, "pixel_width_um": PS,
            "pixel_height_um": PS, "context_px": 0}
    field = Field(tile, rgb, (slice(0, h), slice(0, w)), (0, 0), 0.05, PS, np.zeros((h, w), bool), nuclei)
    automatic = np.zeros((h, w), bool) if excluded is None else excluded
    field.__dict__["linear"] = automatic  # detection's excluded area (NeuN leaves out linear artifacts, not folds)
    field.__dict__["fold"] = np.zeros((h, w), bool)
    return field


def _result(field: Field, stain: str) -> tuple[list, dict]:
    found = proposals.process_tile(field, stain, SETTINGS[stain], SHARED)
    rows = sorted((round(r["x_core"]), round(r["y_core"]), round(r["area_um2"], 3)) for r in found["rows"])
    return rows, {k: found["summary"][k] for k in MEASURES if k in found["summary"]}


def test_a_stained_blob_in_an_excluded_area_changes_no_candidate_and_no_measure():
    for stain in ("6E10", "AT8", "NeuN"):
        with_blob = _result(_field(stain, *_scene(stain, True), FOLD), stain)
        without = _result(_field(stain, *_scene(stain, False), FOLD), stain)
        assert with_blob == without, stain  # the same candidates and measures: numerators, denominators, background
        assert not any(340 <= x < 540 and 340 <= y < 540 for x, y, _ in with_blob[0]), stain
        plain = _result(_field(stain, *_scene(stain, True), None), stain)  # not excluded, the blob is found and counted
        assert any(abs(x - 440) < 12 and abs(y - 440) < 12 for x, y, _ in plain[0]), stain
        assert plain[1]["analysis_area_mm2"] > with_blob[1]["analysis_area_mm2"], stain
        assert plain[1]["dab_area_mm2_at_1"] > with_blob[1]["dab_area_mm2_at_1"], stain


def test_context_features_leave_excluded_pixels_out():
    from stainid.v2.context import context_features_many

    rgb, nuclei = _scene("NeuN", False)
    darker = rgb.copy()
    darker[FOLD] = (60, 40, 30)  # whatever lies in the excluded area
    centre = [(330.0, 330.0)]
    a = context_features_many(_field("NeuN", rgb, nuclei, FOLD), centre, [12.0])
    b = context_features_many(_field("NeuN", darker, nuclei, FOLD), centre, [12.0])
    assert a == b
    c = context_features_many(_field("NeuN", darker, nuclei, None), centre, [12.0])
    assert c != b


def test_the_slide_calibration_leaves_the_excluded_areas_out(tmp_path, monkeypatch):
    from PIL import Image

    from stainid.v2 import field as field_module
    from stainid.workflows import calibrate

    rgb, _ = _scene("6E10", False)
    darker = rgb.copy()
    darker[FOLD] = (120, 80, 50)  # dark, brown: it would raise the threshold
    tiles = [{"tile_id": "T", "core_id": "C", "stain": "6E10", "image_path": "c.png", "x_px": 0, "y_px": 0, "width_px": 600, "height_px": 600,
              "pixel_width_um": PS, "pixel_height_um": PS}]
    monkeypatch.setattr(field_module, "fold_mask", lambda rgb_, ps: FOLD[: rgb_.shape[0], : rgb_.shape[1]].copy())
    monkeypatch.setattr(field_module, "linear_artifact_mask", lambda rgb_: np.zeros(rgb_.shape[:2], bool))
    samples = []
    for picture in (rgb, darker):
        Image.fromarray(picture).save(tmp_path / "c.png")
        samples.append(calibrate.sample_core(str(tmp_path), "c.png", tiles, stride=1, context=0))
    assert np.array_equal(samples[0]["values"], samples[1]["values"])  # the excluded pixels are not in the sample
    assert not np.array_equal(samples[0]["all"], samples[1]["all"])
    assert samples[1]["excluded_px"] == int(FOLD.sum())


def test_area_edits_do_not_touch_the_analysis(v2_project):  # noqa: F811
    project = load_project(v2_project)
    calibration, nuclei = run.proposal_inputs(project)
    core, stain, tiles, _ = next(j for j in run.todo(project, "proposals", ["6E10"]))
    made = proposals.expected(project, stain, core, tiles, calibration[(tiles[0]["tma"], stain)], nuclei)
    assert not {"exclusions", "manual_exclusions"} & set(made["inputs"])  # neither people's areas nor the drawn polygons
