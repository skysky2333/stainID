"""Stage 2 proposes on all tissue: an excluded area (fold, linear artifact, drawn or painted) still gets its candidates,
recorded with their excluded share, so clearing it later needs no new proposals; the excluded pixels stay out of the
background statistics only; and editing the hand-drawn exclusions does not make stage 2 outdated."""
from __future__ import annotations

import json

import cv2
import numpy as np
from test_v2_pipeline import BLUE, BROWN, PALE, PS, blank, v2_project  # noqa: F401 - the fixture

from stainid.project import load_project
from stainid.v2 import proposals, run
from stainid.v2.field import Field
from stainid.v2.params import DEFAULTS

SETTINGS = DEFAULTS["proposals"]
SHARED = {k: v for k, v in SETTINGS.items() if k not in ("NeuN", "6E10", "AT8")}


def _field(rgb: np.ndarray, excluded: np.ndarray, nuclei: np.ndarray | None = None) -> Field:
    h, w = rgb.shape[:2]
    tile = {"tile_id": "T_W0101", "x_px": 0, "y_px": 0, "width_px": w, "height_px": h, "pixel_width_um": PS, "pixel_height_um": PS,
            "context_px": 0}
    return Field(tile, rgb, (slice(0, h), slice(0, w)), (0, 0), 0.05, PS, excluded, nuclei)


def _scene(stain: str) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    rgb = blank(500)
    nuclei = np.zeros(rgb.shape[:2], np.int32)
    centres = [(120, 120), (380, 380)]
    for k, (x, y) in enumerate(centres, start=1):
        if stain == "6E10":
            cv2.circle(rgb, (x, y), 40, PALE, -1)
            cv2.circle(rgb, (x, y), 18, BROWN, -1)
        elif stain == "AT8":
            cv2.circle(rgb, (x, y), 16, BROWN, -1)
            cv2.circle(rgb, (x, y), 8, BLUE, -1)
            cv2.circle(nuclei, (x, y), 8, k, -1)
        else:
            cv2.circle(rgb, (x, y), 14, BROWN, -1)
            cv2.circle(nuclei, (x, y), 7, k, -1)
    return rgb, nuclei.astype(np.uint32), centres


def _nearest(rows: list[dict], point: tuple[int, int]) -> dict:
    return min(rows, key=lambda r: np.hypot(r["x_core"] - point[0], r["y_core"] - point[1]))


def test_objects_in_an_excluded_area_are_still_proposed_with_their_excluded_share():
    for stain in ("6E10", "AT8", "NeuN"):
        rgb, nuclei, (clean, covered) = _scene(stain)
        excluded = np.zeros(rgb.shape[:2], bool)
        excluded[300:470, 300:470] = True  # e.g. a fold or a drawn exclusion over the second object
        rows = proposals.process_tile(_field(rgb, excluded, nuclei), stain, SETTINGS[stain], SHARED)["rows"]
        plain = proposals.process_tile(_field(rgb, np.zeros_like(excluded), nuclei), stain, SETTINGS[stain], SHARED)["rows"]
        inside = _nearest(rows, covered)
        assert np.hypot(inside["x_core"] - covered[0], inside["y_core"] - covered[1]) < 10, stain
        assert inside["excluded_fraction"] >= 0.5, stain  # what the downstream rule (stage 4) rejects unless it is cleared
        assert _nearest(rows, clean)["excluded_fraction"] == 0.0
        assert len(rows) == len(plain), stain  # the same candidates as without the exclusion
        assert sorted(round(r["area_um2"]) for r in rows) == sorted(round(r["area_um2"]) for r in plain), stain


def test_a_dark_excluded_band_stays_out_of_the_6e10_background_statistics():
    from stainid.stains.amyloid.segmentation import segment_amyloid_candidates

    rgb = blank(500)
    rgb[:, :300] = (150, 110, 80)  # a dark fold over most of the field
    cv2.circle(rgb, (420, 250), 30, BROWN, -1)
    fold = np.zeros(rgb.shape[:2], bool)
    fold[:, :300] = True
    with_fold, *_ = segment_amyloid_candidates(rgb, 0.05, PS, PS, None, background_exclusion=fold)
    without, *_ = segment_amyloid_candidates(rgb[:, 300:].copy(), 0.05, PS, PS)
    assert with_fold["seed_threshold_dab_od"] == without["seed_threshold_dab_od"]
    unguarded, *_ = segment_amyloid_candidates(rgb, 0.05, PS, PS)
    assert unguarded["seed_threshold_dab_od"] > with_fold["seed_threshold_dab_od"]


def test_hand_drawn_exclusions_are_not_a_stage_2_input(v2_project):  # noqa: F811
    project = load_project(v2_project)
    jobs = run.todo(project, "proposals", ["NeuN", "6E10", "AT8"])
    calibration, nuclei = run.proposal_inputs(project)
    core, stain, tiles, _ = next(j for j in jobs if j[1] == "6E10")
    proposals.propose_core(project, stain, core, tiles, calibration[(tiles[0]["tma"], stain)], nuclei)
    assert "manual_exclusions" not in proposals.inputs_of(project, stain, core, tiles, 0.05, nuclei)
    drawn = {tiles[0]["image_path"]: [{"polygon_core_px": [[100, 100], [300, 100], [300, 300], [100, 300]], "source": "test"}]}
    (v2_project / "data" / "annotations" / "cohort_manual_exclusions.json").write_text(json.dumps(drawn))
    project = load_project(v2_project)
    assert "6E10" not in [s for _, s, _, _ in run.todo(project, "proposals", ["6E10"])]
