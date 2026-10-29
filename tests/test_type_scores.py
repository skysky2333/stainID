"""The continuous type scores of stage 5 (`stainid.v2.scores`): monotone, 0.5 at stage 4's cut for median objects, three
bins, and readouts that add columns only."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stainid.v2 import scores

TAU_BUNDLE = {"mature_p90_threshold": 0.16, "mature_feature": "soma_dab_p90"}
PLAQUE_BUNDLE = {"morphotype_threshold": 1.5}
QUANTILES = " ".join(f"{v:.4f}" for v in np.linspace(0.5, 3.0, 21))  # 0.5 … 3.0 slide thresholds


def _tau(p90, contrast=2.83, cv=0.597) -> pd.DataFrame:
    p90 = np.atleast_1d(np.asarray(p90, float))
    return pd.DataFrame({"m_soma_dab_p90": p90, "m_normalized_contrast": contrast, "m_soma_dab_cv": cv})


def _plaque(inner, quantiles=QUANTILES, radial=0.777, edge=0.717) -> pd.DataFrame:
    inner = np.atleast_1d(np.asarray(inner, float))
    return pd.DataFrame({"normalized_inner_dab": inner, "core_dab_q": quantiles, "s_radial_dab_contrast": radial, "boundary_contrast": edge})


def test_a_median_tau_neuron_scores_one_half_at_the_mature_cut():
    assert scores.compute("AT8", _tau(0.16), TAU_BUNDLE)[0] == pytest.approx(0.5)
    low, high = scores.compute("AT8", _tau([0.10, 0.30]), TAU_BUNDLE)
    assert low < 0.5 < high


def test_scores_are_monotone_in_every_feature():
    grid = np.linspace(0.05, 0.5, 20)
    assert np.all(np.diff(scores.compute("AT8", _tau(grid), TAU_BUNDLE)) > 0)
    for column in ("m_normalized_contrast", "m_soma_dab_cv"):
        frame = _tau(np.full(20, 0.16))
        frame[column] = np.linspace(0.0, 5.0, 20)
        assert np.all(np.diff(scores.compute("AT8", frame, TAU_BUNDLE)) > 0)
    for column, values in (("normalized_inner_dab", np.linspace(0.5, 4, 20)), ("s_radial_dab_contrast", np.linspace(-1, 3, 20)),
                           ("boundary_contrast", np.linspace(-1, 3, 20))):
        frame = _plaque(np.full(20, 1.5))
        frame[column] = values
        assert np.all(np.diff(scores.compute("6E10", frame, PLAQUE_BUNDLE)) > 0)
    # the dense-core fraction: the deposit's share at or above the morphotype cut-off grows with its quantiles
    frame = _plaque(np.full(5, 1.5), [" ".join(f"{v:.4f}" for v in np.linspace(0.5, top, 21)) for top in (1.6, 2.0, 2.5, 3.0, 4.0)])
    assert np.all(np.diff(scores.compute("6E10", frame, PLAQUE_BUNDLE)) > 0)


def test_the_score_follows_the_models_cut_and_a_scaled_feature():
    assert scores.compute("AT8", _tau(0.2), {"mature_p90_threshold": 0.2})[0] == pytest.approx(0.5)
    frame = _tau(0.01).assign(s_soma_dab_p90=4.0)
    assert scores.compute("AT8", frame, {"mature_p90_threshold": 4.0, "mature_feature": "s_soma_dab_p90"})[0] == pytest.approx(0.5)


def test_missing_features_count_as_their_centre_and_no_dab_level_gives_no_score():
    frame = _tau([0.16, 0.16, np.nan])
    frame.loc[1, "m_soma_dab_cv"] = np.nan
    values = scores.compute("AT8", frame, TAU_BUNDLE)
    assert values[0] == pytest.approx(values[1]) and np.isnan(values[2])
    plaque = _plaque([1.5]).drop(columns="s_radial_dab_contrast").assign(m_radial_dab_contrast=0.777 * 0.08, m_inner_mean_dab_od=1.5 * 0.08)
    assert scores.features("6E10", plaque, PLAQUE_BUNDLE).core_contrast.iloc[0] == pytest.approx(0.777)


def test_bins_and_readouts():
    between = (0.29, 0.62)
    assert list(scores.bins("AT8", np.array([0.1, 0.29, 0.5, 0.62, 0.9, np.nan]), between)) == \
        ["pretangle", "intermediate", "intermediate", "mature", "mature", ""]
    found = pd.DataFrame({"tile_id": "t", "candidate_id": list("abcdef"), "decision": ["accepted"] * 5 + ["rejected"],
                          "subtype": ["pretangle", "pretangle", "mature tangle", "mature tangle", "mature tangle", ""]}).assign(
        **_tau([0.05, 0.13, 0.16, 0.3, 0.5, 0.5]))
    rows = scores.object_rows("AT8", found, TAU_BUNDLE, between, {"t": {"core_id": "C"}})
    assert [r["candidate_id"] for r in rows] == list("abcde") and rows[0]["core_id"] == "C"
    assert [r["tau_maturity_bin"] for r in rows] == ["pretangle", "intermediate", "intermediate", "mature", "mature"]
    out = scores.readouts("AT8", rows, 2.0)
    assert list(out) == scores.readout_names("AT8")
    assert out["tau_maturity_scored_count"] == 5 and out["tau_maturity_mature_fraction"] == pytest.approx(0.4)
    assert out["tau_maturity_intermediate_density_mm2"] == pytest.approx(1.0)
    assert out["tau_maturity_score_median"] == pytest.approx(rows[2]["tau_maturity_score"])
    few = scores.readouts("AT8", rows[:2], 2.0)  # under 5 objects: no distribution or share, the densities all the same
    assert np.isnan(few["tau_maturity_score_median"]) and few["tau_maturity_pretangle_density_mm2"] == pytest.approx(0.5)


def test_a_small_plaque_has_a_score_and_no_bin():
    found = pd.DataFrame({"tile_id": "t", "candidate_id": ["a", "b"], "decision": "accepted", "subtype": ["small", "compact"]}).assign(**_plaque([3.0, 3.0]))
    rows = scores.object_rows("6E10", found, PLAQUE_BUNDLE, (0.48, 0.79), {})
    assert [r["plaque_compactness_bin"] for r in rows] == ["", "compact"] and np.isfinite(rows[0]["plaque_compactness_score"])


def test_cuts_come_from_the_project(tmp_path):
    from stainid.project import load_project, write_default_config

    write_default_config(tmp_path)
    project = load_project(tmp_path)
    assert scores.cuts(project, "AT8") == (0.29, 0.62) and scores.cuts(project, "6E10") == (0.48, 0.79)
    project.config["v2"] = {"scores": {"AT8": {"bins": [0.7, 0.2]}}}
    with pytest.raises(ValueError):
        scores.cuts(project, "AT8")
