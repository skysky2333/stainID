from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from joblib import load

from stainid.project import load_project
from stainid.review.store import create_set, save_label
from stainid.slides.layout import attach_layout, read_layout
from stainid.slides.table import guess, save_slides, scan_slides
from stainid.training.sets import LABELS, choose, pick_fields
from stainid.training.train import activate, train_model, trained_models


@pytest.mark.parametrize("name, tma, stain", [
    ("TMA LIP-3 AT8.vsi", "3", "AT8"), ("Block-04 6e10 scan.svs", "4", "6E10"), ("study_tma12_NeuN.ndpi", "12", "NeuN"), ("scan.tif", "", ""),
])
def test_guess_slide(name, tma, stain):
    assert guess(Path(name), ["NeuN", "6E10", "AT8"]) == {"slide_path": name, "tma": tma, "stain": stain}


def test_slides_table_scan_and_validation(tmp_path):
    for name in ("TMA 1 NeuN.vsi", "TMA 1 AT8.vsi", "notes.txt"):
        (tmp_path / name).touch()
    table = tmp_path / "slides.csv"
    rows = scan_slides(tmp_path, table, ["NeuN", "AT8"])
    assert [(r["tma"], r["stain"]) for r in rows] == [("1", "AT8"), ("1", "NeuN")]
    with pytest.raises(ValueError, match="more than one NeuN"):
        save_slides(table, [rows[1], {**rows[0], "stain": "NeuN"}], ["NeuN", "AT8"])
    save_slides(table, rows, ["NeuN", "AT8"])
    assert scan_slides(tmp_path, table, ["NeuN", "AT8"]) == rows


def test_layout_defaults_and_attach(tmp_path):
    layout = tmp_path / "map.csv"
    pd.DataFrame([{"tma": "1", "core_label": "A-1", "donor_id": "", "region": "", "disease_group": ""},
                  {"tma": "1", "core_label": "B-1", "donor_id": "7", "region": "frontal", "disease_group": "AD"},
                  {"tma": "1", "core_label": "C-1", "donor_id": "7", "region": "frontal", "disease_group": "AD"}]).to_csv(layout, index=False)
    positions = read_layout(layout)
    assert positions[("1", "A-1")]["core_role"] == "orientation_control"
    assert positions[("1", "C-1")]["sample_region_id"] == "7_frontal" and positions[("1", "C-1")]["technical_replicate"] == "2"
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame([{"tma": "1", "core_label": "B-1", "stain": s} for s in ("NeuN", "AT8")]).to_csv(manifest, index=False)
    attach_layout(manifest, layout)
    assert set(pd.read_csv(manifest, dtype=str).sample_region_id) == {"7_frontal"}
    pd.DataFrame([{"tma": "2", "core_label": "B-1", "stain": "NeuN"}]).to_csv(manifest, index=False)
    with pytest.raises(ValueError, match="No TMA map entry"):
        attach_layout(manifest, layout)


def test_pick_fields_spreads_over_tmas_and_choose_prefers_uncertain():
    tiles = [{"tile_id": f"{t}_{i}", "tma": t, "disease_group": g} for t in "12" for g in ("AD", "CT") for i in range(5)]
    picked = pick_fields(tiles, 4, seed=0)
    assert {(t["tma"], t["disease_group"]) for t in picked} == {("1", "AD"), ("1", "CT"), ("2", "AD"), ("2", "CT")}
    probability = np.array([0.01] * 10 + [0.5] * 10)
    chosen = choose(probability, 8, "uncertain", np.random.default_rng(0))
    assert len(set(chosen)) == 8 and (probability[chosen] == 0.5).sum() >= 4


def _training_set(project, stain: str, name: str, n: int = 60) -> None:
    rng = np.random.default_rng(1)
    names = ["size", "stain_mean", "context"]
    x = rng.normal(size=(n, 3))
    positive = x[:, 0] + x[:, 1] > 0
    items = pd.DataFrame({"tile_id": "T", "stain": stain, "x": 1.0, "y": 1.0, "tma": [str(i % 3 + 1) for i in range(n)], "item": range(n)})
    folder = create_set(project.output("reviews"), f"training/{name}", name, stain, items, LABELS[stain], "", 55.0)
    key = pd.read_csv(folder / "key.csv")
    features = pd.DataFrame(x[key["item"]], columns=names)
    features.insert(0, "review_id", key["review_id"])
    features["soma_dab_p90"] = rng.uniform(0.05, 0.3, n)
    features.to_csv(folder / "features.csv", index=False)
    meta = json.loads((folder / "meta.json").read_text())
    (folder / "meta.json").write_text(json.dumps({**meta, "purpose": "training", "feature_names": names}))
    for review_id, is_positive, p90 in zip(key["review_id"], positive[key["item"]], features["soma_dab_p90"]):
        label = ("tau+ neuron: tangle" if p90 > 0.17 else "tau+ neuron: pretangle") if is_positive else "not a tau+ neuron"
        save_label(project.output("reviews"), f"training/{name}", review_id, label)


def test_train_report_and_activate(tmp_path):
    project = load_project(tmp_path)
    with pytest.raises(ValueError, match="No labelled AT8"):
        train_model(project, "AT8")
    _training_set(project, "AT8", "r1")
    folder = train_model(project, "AT8")
    report = json.loads((folder / "report.json").read_text())
    assert report["validation"] == "leave-one-TMA-out" and report["new_model"]["auc"] > 0.85 and report["current_model"] is None
    bundle = load(folder / "bundle.joblib")
    assert bundle["feature_names"] == ["size", "stain_mean", "context"] and 0.05 < bundle["mature_p90_threshold"] < 0.3
    assert not trained_models(project)[0]["active"]
    project = activate(project, folder / "bundle.joblib")
    assert project.model("tau") == folder / "bundle.joblib" and trained_models(project)[0]["active"]
