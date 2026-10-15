from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from stainid.analysis import measure_stats as ms
from stainid.api import state


def cohort(seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Donor-region and core tables: 3 TMAs, 6 donors per group per TMA (CT, ASYMP, AD), two regions, two cores each.
    AD has many more tau neurons; `noise_um2` differs between the cores of a donor-region as much as between them."""
    rng = np.random.default_rng(seed)
    cores, regions = [], []
    for tma in ("1", "2", "3"):
        for group, tau in (("CT", 5.0), ("ASYMP", 10.0), ("AD", 60.0)):
            for d in range(6):
                donor = f"{tma}{group}{d}"
                level = rng.normal(0, 1)
                for region in ("frontal", "occipital"):
                    sid = f"{donor}_{region}"
                    true = tau * np.exp(rng.normal(0, 0.3))
                    values = [{"core_id": f"LIP-{tma}_{sid}_{k}", "sample_region_id": sid, "donor_id": donor, "tma": tma, "region": region,
                               "disease_group": group, "tau_neuron_density_mm2": true * np.exp(rng.normal(0, 0.1)),
                               "neun_median_profile_area_um2": 100 + 10 * level + rng.normal(0, 1), "noise_um2": rng.normal(0, 1)}
                              for k in (1, 2)]
                    cores += values
                    regions.append({**{k: v for k, v in values[0].items() if k != "core_id"},
                                    **{c: float(np.mean([v[c] for v in values])) for c in ("tau_neuron_density_mm2", "neun_median_profile_area_um2", "noise_um2")}})
    return pd.DataFrame(regions), pd.DataFrame(cores)


def test_icc_from_cores_and_which_study_value_describes_a_column():
    _, core = cohort()
    own = ms.computed_icc(core, "neun_median_profile_area_um2")
    assert own["pairs"] == 108 and own["icc"] > 0.9 and own["scale"] == "raw"
    assert ms.computed_icc(core, "tau_neuron_density_mm2")["scale"] == "log"  # densities on log10(x + 1)
    assert ms.computed_icc(core, "noise_um2")["icc"] < 0.3
    tables = {"neun_median_profile_area_um2": [{"icc": round(own["icc"], 3), "pairs": 108, "file": "a.csv", "scale": None}],
              "noise_um2": [{"icc": 0.9, "pairs": 108, "file": "a.csv", "scale": None}],  # a measure of the same name, other numbers
              "missing_um2": [{"icc": 0.6, "pairs": 50, "file": "b.csv", "scale": None}]}
    out = ms.reliability(["neun_median_profile_area_um2", "noise_um2", "missing_um2"], core, tables)
    assert out["neun_median_profile_area_um2"]["source"] == "study" and out["neun_median_profile_area_um2"]["file"] == "a.csv"
    assert out["noise_um2"]["source"] == "computed" and out["noise_um2"]["icc"] < 0.3
    assert out["missing_um2"] == {"icc": 0.6, "pairs": 50, "source": "study", "file": "b.csv", "scale": None}  # no core values to check
    disagreeing = {"x": [{"icc": 0.3, "pairs": 9, "file": "a.csv", "scale": None}, {"icc": 0.8, "pairs": 9, "file": "b.csv", "scale": None}]}
    assert ms.reliability(["x"], None, disagreeing) == {}  # two tables, two answers, nothing to tell which: none shown


def test_log_scale_entry_of_the_reliability_table_is_preferred(tmp_path):
    (tmp_path / "a_replicate_icc.csv").write_text("feature,pairs,icc\nneun_profile_density_mm2,90,0.716\n")
    (tmp_path / "replicate_reliability.csv").write_text("feature,pairs,icc\nlog_neun_density,90,0.737\n")
    found = ms.study_icc([tmp_path / "a_replicate_icc.csv", tmp_path / "replicate_reliability.csv"], lambda p: p.name)
    assert [e["icc"] for e in found["neun_profile_density_mm2"]] == [0.737, 0.716]
    assert ms.reliability(["neun_profile_density_mm2"], None, found)["neun_profile_density_mm2"]["scale"] == "log"


def test_fdr_and_quick_test():
    adjusted = ms.fdr(pd.Series([0.01, 0.04, 0.03, np.nan, 0.5]))
    assert np.allclose(adjusted.iloc[[0, 1, 2, 4]], [0.04, 0.0533333, 0.0533333, 0.5]) and np.isnan(adjusted.iloc[3])
    regions, _ = cohort()
    tau = ms.quick_test(regions, "tau_neuron_density_mm2", "AD", "ASYMP")
    assert tau["effect"] > 1 and tau["p"] < 1e-4 and tau["n"] == 72 and tau["terms"] == ["region", "tma"]
    assert ms.quick_test(regions, "noise_um2", "AD", "ASYMP")["p"] > 0.01
    few = regions[(regions.disease_group != "CT") | (regions.donor_id == "1CT0")]  # one control donor: too few values
    assert ms.quick_test(few, "tau_neuron_density_mm2", "AD", "CT") is None
    only_ad_tmas = regions[~((regions.tma == "3") & (regions.disease_group == "ASYMP"))]  # TMA 3 without ASYMAD donors is left out
    assert ms.quick_test(only_ad_tmas, "tau_neuron_density_mm2", "AD", "ASYMP")["n"] == 48
    assert ms.comparisons(["CT", "ASYMP", "AD"], {"AD": "AD", "ASYMP": "ASYMP", "CT": "CT"}) == [("AD", "ASYMP"), ("AD", "CT"), ("ASYMP", "CT")]
    assert ms.comparisons(["A", "B"], {}) == [("B", "A")]
    assert ms.adjustment(ms.model_terms(ms.FULL_MODEL)) == "amyloid burden, age, APOE ε4, brain region and TMA"


@pytest.fixture()
def client(synthetic_project, monkeypatch):
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from stainid.api.app import create_app
    monkeypatch.chdir(synthetic_project)
    state.reset()
    admin = TestClient(create_app(), client=("127.0.0.1", 50000))
    assert admin.post("/api/session/admin", json={"password": "admin"}).status_code == 200
    yield admin
    state.reset()


@pytest.fixture()
def results(synthetic_project):
    regions, core = cohort()
    analysis = synthetic_project / "data" / "analysis"
    (analysis / "biology").mkdir(exist_ok=True)
    regions.to_csv(analysis / "results_donor_region.csv", index=False)
    core.to_csv(analysis / "results_core.csv", index=False)
    return analysis, regions


def test_stats_endpoint_without_study_tables_computes_everything(client, results):
    stats = client.get("/api/tables/results_donor_region.csv/stats").json()
    assert stats["icc"]["neun_median_profile_area_um2"]["source"] == "computed" and stats["icc_files"] == [] and stats["contrast_files"] == []
    assert [c["key"] for c in stats["comparisons"]] == ["AD|ASYMP", "AD|CT", "ASYMP|CT"]
    first = stats["comparisons"][0]
    assert first["models"] == [{"key": "quick", "terms": ["region", "tma"], "adjusted": "brain region and TMA", "source": "quick", "default": True}]
    tau = first["tests"]["tau_neuron_density_mm2"]["quick"]
    assert tau["effect"] > 1 and tau["fdr"] < 0.01
    core_stats = client.get("/api/tables/results_core.csv/stats").json()  # per core: the same ICC, the donor-region tests
    assert core_stats["icc"] == stats["icc"] and core_stats["comparisons"][0]["tests"].keys() == first["tests"].keys()


def test_stats_endpoint_uses_study_tables_that_hold_the_same_numbers(client, results):
    analysis, regions = results
    icc = analysis / "biology" / "cohort_v3_replicate_icc.csv"
    own = ms.computed_icc(pd.read_csv(analysis / "results_core.csv"), "neun_median_profile_area_um2")["icc"]
    icc.write_text(f"feature,pairs,icc\nneun_median_profile_area_um2,108,{own:.3f}\nnoise_um2,108,0.9\n")
    stale = analysis / "biology" / "cohort_v2_replicate_icc.csv"
    stale.write_text("feature,pairs,icc\ntau_neuron_density_mm2,108,0.1\n")
    old = (analysis / "results_donor_region.csv").stat().st_mtime - 3600
    os.utime(stale, (old, old))  # made before the current results: not used
    wide = regions.assign(sampling_extent="sampled_fields", noise_um2=regions.noise_um2 + 5)  # the study's noise_um2 is another measure
    wide.to_csv(analysis / "cohort_morphology_donor_region_wide.csv", index=False)
    rows = [{"feature": f, "model": m, "n": 72, "n_ad": 36, "std_effect": e, "t": 3.0, "p": p}
            for f, e, p in (("tau_neuron_density_mm2", 1.5, 1e-6), ("noise_um2", 0.9, 0.001), ("not_on_screen", 0.1, 0.45))
            for m, p in (("is_ad + region + tma", p), (ms.FULL_MODEL, p * 2))]
    pd.DataFrame(rows).to_csv(analysis / "cohort_morphology_donor_region_wide_within_tma_contrasts.csv", index=False)

    stats = client.get("/api/tables/results_donor_region.csv/stats").json()
    assert stats["icc"]["neun_median_profile_area_um2"] == {"icc": round(own, 3), "pairs": 108, "source": "study",
                                                             "file": "data/analysis/biology/cohort_v3_replicate_icc.csv", "scale": "raw"}
    assert stats["icc"]["noise_um2"]["source"] == "computed" and stats["icc"]["tau_neuron_density_mm2"]["source"] == "computed"
    assert stats["icc_files"] == ["data/analysis/biology/cohort_v3_replicate_icc.csv"]
    first = stats["comparisons"][0]
    assert [(m["key"], m["source"], m["default"]) for m in first["models"]] == [
        ("is_ad + region + tma", "study", False), (ms.FULL_MODEL, "study", True), ("quick", "quick", False)]
    tau = first["tests"]["tau_neuron_density_mm2"]
    assert set(tau) == {"is_ad + region + tma", ms.FULL_MODEL} and tau[ms.FULL_MODEL]["effect"] == 1.5
    assert tau[ms.FULL_MODEL]["fdr"] == pytest.approx(6e-6)  # over the study table's measures for that model
    assert tau[ms.FULL_MODEL]["file"] == "data/analysis/cohort_morphology_donor_region_wide_within_tma_contrasts.csv"
    assert set(first["tests"]["noise_um2"]) == {"quick"}  # the study measured other numbers under this name
    assert all(set(c["tests"]["tau_neuron_density_mm2"]) == {"quick"} for c in stats["comparisons"][1:])  # the study tested AD vs ASYMAD only
    assert client.get("/api/tables/results_donor_region.csv/stats").json() == stats  # kept until a file changes
