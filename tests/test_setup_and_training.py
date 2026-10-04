from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from joblib import load

from stainid.labels import pool, sessions, store
from stainid.project import load_project
from stainid.slides.layout import attach_layout, read_layout
from stainid.slides.table import guess, save_slides, scan_slides
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


def test_pick_fields_spreads_over_tmas_and_skips_reviewed():
    tiles = pd.DataFrame([{"tile_id": f"{t}_{g}_{i}", "tma": t, "disease_group": g} for t in "12" for g in ("AD", "CT") for i in range(5)])
    picked = sessions.pick_fields(tiles, {"1_AD_0"}, 4, seed=0)
    assert {tuple(t.split("_")[:2]) for t in picked} == {("1", "AD"), ("1", "CT"), ("2", "AD"), ("2", "CT")} and "1_AD_0" not in picked


def _labelled_pool(project, n: int = 60) -> None:
    """A prepared AT8 field with n candidates, labelled by two people who disagree on a few objects."""
    rng = np.random.default_rng(1)
    names = ["size", "stain_mean", "context"]
    x = rng.normal(size=(n, 3))
    truth = x[:, 0] + x[:, 1] > 0
    p90 = rng.uniform(0.05, 0.3, n)
    frame = pd.DataFrame({"x": np.arange(n) * 20.0, "y": 10.0, "radius_px": 8.0, "ai_value": "not a tau+ neuron", "ai_probability": 0.5,
                          "tma": [str(i % 3 + 1) for i in range(n)], "complete": True, "soma_dab_p90": p90})
    folder = pool.folder(project, "AT8")
    folder.mkdir(parents=True)
    pd.concat([frame, pd.DataFrame(x, columns=names)], axis=1).to_csv(folder / "T.csv", index=False)
    pool._check_names(project, "AT8", names)
    labels = [("tau+ neuron: tangle" if q > 0.17 else "tau+ neuron: pretangle") if t else "not a tau+ neuron" for t, q in zip(truth, p90)]
    for person, flip in (("Ann", set()), ("Ben", {0, 1})):
        store.append(project.output("reviews"), [
            {"id": store.object_id("class", "T", i * 20.0, 10.0), "source": person, "stain": "AT8", "task": "class", "tile_id": "T", "x": i * 20.0, "y": 10.0,
             "value": "not a tau+ neuron" if i in flip else label} for i, label in enumerate(labels)])


def test_consensus_and_queue_rules(tmp_path):
    project = load_project(tmp_path)
    _labelled_pool(project)
    labels = store.current(project.output("reviews"))
    consensus = store.consensus(labels, "AT8").set_index("x")
    assert len(consensus) == 60 and consensus.loc[20.0, "n"] == 2
    objects = pool.all_objects(project, "AT8")
    disputed = sessions.next_object({"stain": "AT8", "rule": "disagree"}, objects, labels, "Cy")
    assert disputed is not None and disputed["labels_so_far"] == 2
    assert sessions.next_object({"stain": "AT8", "rule": "random"}, objects, labels, "Ann") is None


def test_train_report_and_activate(tmp_path):
    project = load_project(tmp_path)
    with pytest.raises(ValueError, match="No labelled AT8"):
        train_model(project, "AT8")
    _labelled_pool(project)
    folder = train_model(project, "AT8")
    report = json.loads((folder / "report.json").read_text())
    assert report["validation"] == "leave-one-TMA-out" and report["new_model"]["auc"] > 0.85 and report["current_model"] is None
    bundle = load(folder / "bundle.joblib")
    assert bundle["feature_names"] == ["size", "stain_mean", "context"] and 0.05 < bundle["mature_p90_threshold"] < 0.3
    assert not trained_models(project)[0]["active"]
    project = activate(project, folder / "bundle.joblib")
    assert project.model("tau") == folder / "bundle.joblib" and trained_models(project)[0]["active"]


def test_whole_core_grid_covers_the_image_without_overlap():
    from stainid.sampling.whole_core import grid

    assert grid(5000, 2048) == [(0, 2048), (2048, 2048), (4096, 904)]
    assert grid(4200, 2048) == [(0, 2048), (2048, 2048)]
