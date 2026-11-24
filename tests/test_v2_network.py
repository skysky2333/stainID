"""A network model (stainid.v2.network, 2026-11-20): stage 4 decides from the scores stage 4a stored, with the same rules,
duplicate radius and cut-off as for a forest's probabilities; a network bundle never scores stored features; the
labelling pool falls back to the forest the bundle keeps."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from stainid.v2 import classify, network

FRAME = pd.DataFrame({"candidate_id": list("abcd"), "x_core": [0.0, 10.0, 900.0, 1500.0], "y_core": 0.0, "source": "ring",
                      "m_normalized_contrast": 1.0, "excluded_fraction": 0.0, "thread_fraction": 0.0, "legacy_class": "",
                      "m_soma_area_um2": 99.0, "m_soma_dab_p90": 0.5})
NET = {"kind": "network", "network_file": "network.pt", "nms_radius_um": 12.0, "mature_feature": "soma_dab_p90", "mature_p90_threshold": 0.2,
       "classifier": None, "feature_names": [], "fallback_threshold": 0.3}


def test_a_network_decides_from_the_given_scores_with_the_forest_rules():
    model = classify.Model("AT8", NET, None, "net", 0.5, [])
    assert model.network
    with pytest.raises(RuntimeError):
        model.probability(FRAME)  # its scores come from stage 4a, never from the stored features
    scores = np.array([0.9, 0.8, 0.4, 0.6])
    calls = classify.call(SimpleNamespace(pixel_size_um=0.5), "AT8", FRAME, model, {}, scores).set_index("candidate_id")
    assert calls.loc["a", "reason"] == "accepted" and calls.loc["a", "subtype"] == "mature tangle"
    assert calls.loc["b", "reason"] == "rule:duplicate"  # 5 um from a better-scored call: the bundle's 12 um radius
    assert calls.loc["c", "reason"] == "rejected_by_model" and calls.loc["d", "reason"] == "accepted"
    assert np.allclose(calls.probability.to_numpy(), scores)


def test_a_forest_has_no_network_stage():
    forest = classify.Model("AT8", {"classifier": None, "feature_names": []}, None, "forest", 0.3, [])
    assert not forest.network
    assert network.expected(SimpleNamespace(), "AT8", "core", forest) is None
    assert network.WIN_SIDE == 701 and network.CELL_PX == 263  # the training inputs at 0.2738 um/px


def test_the_window_is_white_beyond_the_image():
    from PIL import Image

    image = Image.new("RGB", (100, 100), (10, 20, 30))
    w = network.window(image, 5.0, 5.0)
    assert w.shape == (701, 701, 3)
    assert (w[350, 350] == [10, 20, 30]).all() and (w[0, 0] == 255).all()
