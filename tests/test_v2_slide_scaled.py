"""Slide-scaled model inputs: stage 2 stores each DAB intensity in slide thresholds and the measures cut at absolute OD at
levels relative to the threshold (`s_` columns), and training uses them by default for the next models."""
from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
import pytest
from test_v2_pipeline import v2_project  # noqa: F401 - the synthetic three-stain project
from test_v2_tile_borders import BROWN, border_project, propose

from stainid.project import load_project
from stainid.training.train import slide_scaled, with_scaled
from stainid.v2 import amyloid

PALE_CORE = (150, 110, 80)  # DAB about 0.11 OD: a dense core on a pale slide


def test_stage_two_stores_slide_scaled_inputs(tmp_path):
    def draw(rgb):
        cv2.circle(rgb, (200, 200), 40, (195, 170, 150), -1)  # a halo
        cv2.circle(rgb, (200, 200), 14, BROWN, -1)  # with a dense core
        cv2.circle(rgb, (600, 600), 25, BROWN, -1)
    frame, _, _ = propose(border_project(tmp_path, draw))
    assert len(frame)
    for name in amyloid.SCALED_INTENSITY:
        assert np.allclose(frame[f"s_{name}"], frame[f"m_{name}"] / 0.05, equal_nan=True)
    for name in amyloid.SCALED_CORE:
        assert f"s_{name}" in frame
    # at a threshold above 0.035 OD (as on every slide but LIP-6) the relative seed is the legacy seed
    assert np.allclose(frame.s_seed_area_um2, frame.m_seed_area_um2)


def test_a_pale_slide_keeps_its_dense_core_in_the_scaled_inputs(tmp_path):
    def draw(rgb):
        cv2.circle(rgb, (200, 200), 40, (205, 190, 175), -1)
        cv2.circle(rgb, (200, 200), 14, PALE_CORE, -1)
    root = border_project(tmp_path, draw)
    pd.DataFrame([{"tma": 1, "stain": "6E10", "threshold_dab_od": 0.02}]).to_csv(root / "data/analysis/slide_dab_calibration.csv", index=False)
    frame, _, _ = propose(root)
    plaque = frame.loc[frame.area_um2.idxmax()]
    # the legacy core needs 0.10 OD and more than 1.2 × the 0.035 OD seed floor: none here; relative to the slide, one
    assert plaque.m_core_area_um2 < plaque.s_core_area_um2 and plaque.s_core_fraction > 0
    assert plaque.s_max_dab_od == pytest.approx(plaque.m_max_dab_od / 0.02)


def test_training_uses_scaled_inputs_by_default_for_6e10(tmp_path):
    root = border_project(tmp_path, lambda rgb: None)
    project = load_project(root)
    assert slide_scaled(project, "6E10") and slide_scaled(project, "NeuN") and slide_scaled(project, "AT8")
    names = ["deposit_area_um2", "max_dab_od", "core_fraction", "circularity"]
    data = pd.DataFrame({"tma": ["1", "1"], "deposit_area_um2": [500.0, 900.0], "max_dab_od": [0.2, 0.4], "core_fraction": [0.1, 0.2],
                         "circularity": [0.5, 0.6], "s_max_dab_od": [np.nan, 7.0], "s_core_fraction": [np.nan, 0.3]})
    out, scaled = with_scaled(project, "6E10", data, names)
    assert scaled == ["deposit_area_um2", "s_max_dab_od", "s_core_fraction", "circularity"]
    # stored values stay; a row without them gets its intensity over the TMA's threshold (0.05), its core measure empty
    assert out.s_max_dab_od.tolist() == pytest.approx([4.0, 7.0])
    assert np.isnan(out.s_core_fraction.iloc[0]) and out.s_core_fraction.iloc[1] == 0.3


def test_neun_and_at8_store_slide_scaled_inputs(v2_project):
    from test_v2_pipeline import _proposals

    from stainid import candidates as store
    from stainid.v2 import scaled

    project = load_project(v2_project)
    _proposals(project)
    for stain in ("NeuN", "AT8"):
        frame = store.read_table(store.path(project, stain, "LIP-1_B-2", "proposals"))
        assert len(frame)
        for name in scaled.INTENSITY[stain]:
            assert np.allclose(frame[f"s_{name}"], frame[f"m_{name}"] / 0.05, equal_nan=True), (stain, name)
    neun = store.read_table(store.path(project, "NeuN", "LIP-1_B-2", "proposals"))
    # at a 0.05 OD threshold the 0.012 OD floor does not bind: the relative stained area is the legacy one
    assert np.allclose(neun.s_stained_area_um2, neun.m_stained_area_um2, equal_nan=True)
    assert scaled.names("AT8", ["soma_dab_p90", "normalized_contrast", "ctx_center_dab_mean"]) == \
        ["s_soma_dab_p90", "normalized_contrast", "s_ctx_center_dab_mean"]
