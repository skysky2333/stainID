"""The linear-artifact mask relative to the core's tissue, and the bound on a slide's DAB threshold."""
import cv2
import numpy as np

from stainid.imaging.tissue import core_tissue_darkness, linear_artifact_mask
from stainid.v2 import params as v2_params
from stainid.workflows.calibrate import bounded


class _Project:
    context_px = 128

    def __init__(self, config=None):
        self.config = config or {}


def _tissue(level: int) -> np.ndarray:
    """A field of bluish tissue at this grey level with fine texture (seeded), and a dark scratch across it."""
    rng = np.random.default_rng(0)
    rgb = np.clip(np.full((600, 600, 3), (level - 10, level, level + 25), np.int16) + rng.integers(-12, 12, (600, 600, 1)), 0, 255).astype(np.uint8)
    cv2.line(rgb, (20, 40), (580, 500), (40, 40, 60), 9)
    return rgb


def test_a_scratch_is_found_on_pale_and_on_dark_tissue():
    for level in (200, 150):  # darkness about 0.2 and about 0.4 (darker than the fixed 0.35 cut)
        rgb = _tissue(level)
        mask = linear_artifact_mask(rgb, 0.1, core_tissue_darkness(rgb))
        assert mask[270, 300] or mask[260:290, 290:310].any()  # on the scratch
        assert mask.mean() < 0.1


def test_dark_tissue_itself_is_no_artifact_relative_to_the_tissue():
    rgb = _tissue(150)
    assert linear_artifact_mask(rgb).mean() > 0.5  # the fixed cut takes the whole field for a band
    assert linear_artifact_mask(rgb, 0.1, core_tissue_darkness(rgb)).mean() < 0.1


def test_pale_tissue_keeps_the_fixed_cut():
    rgb = _tissue(200)
    assert np.array_equal(linear_artifact_mask(rgb), linear_artifact_mask(rgb, 0.1, core_tissue_darkness(rgb)))


def test_the_setting_is_recorded_only_when_set():
    off = _Project({"v2": {"proposals": {"linear_artifact_contrast": None}}})  # what stage 2 recorded before the setting
    assert "linear_artifact_contrast" not in v2_params.stage_params(off, "proposals", "AT8")["shared"]
    assert v2_params.stage_params(_Project(), "proposals", "AT8")["shared"]["linear_artifact_contrast"] == 0.1
    assert v2_params.linear_contrast(off) is None and v2_params.linear_contrast(_Project()) == 0.1


def test_a_slide_threshold_is_held_to_the_stain_bound():
    project = _Project()
    high = bounded(project, {"tma": "1", "stain": "AT8", "threshold_dab_od": "0.08"})
    assert high["threshold_dab_od"] == 0.05 and high["calibrated_threshold_dab_od"] == 0.08
    assert bounded(project, {"tma": "1", "stain": "AT8", "threshold_dab_od": "0.031"})["threshold_dab_od"] == 0.031
    assert bounded(project, {"tma": "1", "stain": "6E10", "threshold_dab_od": "0.2"})["threshold_dab_od"] == 0.2
    again = bounded(_Project({"v2": {"calibration": {"max_threshold_dab_od": {"AT8": None}}}}), high)
    assert again["threshold_dab_od"] == 0.05 and again["calibrated_threshold_dab_od"] == 0.08  # kept: never raised
