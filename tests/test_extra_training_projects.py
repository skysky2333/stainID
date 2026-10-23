"""Labels of other projects as extra, training-only cases (`training.extra_projects`): they join every fit and never
the out-of-fold check, the comparison with the model in use or anything else of the study; each model records what it
used; simple labels train through the study's own mapping."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import yaml
from joblib import load

from stainid.labels import pool, store
from stainid.project import load_project, save_project
from stainid.training.train import extra_projects, out_of_fold, train_model


def _pool(project, n: int, tmas: list[str], seed: int, simple_labels: bool = False, donor: str = "") -> None:
    """A prepared AT8 field "T" with n candidates labelled by one person: detailed classes, or the simple classes of a
    simple session ("tangle" / "neuritic_plaque" for real ones, "none" for the others)."""
    rng = np.random.default_rng(seed)
    names = ["size", "stain_mean", "context"]
    x = rng.normal(size=(n, 3))
    truth = x[:, 0] + x[:, 1] > 0
    frame = pd.DataFrame({"x": np.arange(n) * 20.0, "y": 10.0, "radius_px": 8.0, "ai_value": "not a tau+ neuron", "ai_probability": 0.5,
                          "tma": [tmas[i % len(tmas)] for i in range(n)], "complete": True, "soma_dab_p90": rng.uniform(0.05, 0.3, n)})
    if donor:
        frame["donor_id"] = donor
    folder = pool.folder(project, "AT8")
    folder.mkdir(parents=True)
    pd.concat([frame, pd.DataFrame(x, columns=names)], axis=1).to_csv(folder / "T.csv", index=False)
    pool._check_names(project, "AT8", names)
    rows = []
    for i, real in enumerate(truth):
        if simple_labels:
            value = ("tangle" if i % 2 else "neuritic_plaque") if real else "none"
            rows.append({"id": f"simple:T:{i * 20.0}:10.0", "source": "Cy", "stain": "AT8", "task": "simple", "tile_id": "T", "x": i * 20.0,
                         "y": 10.0, "value": value, "session": "simple_1"})
        else:
            rows.append({"id": store.object_id("class", "T", i * 20.0, 10.0), "source": "Ann", "stain": "AT8", "task": "class", "tile_id": "T",
                         "x": i * 20.0, "y": 10.0, "value": "tau+ neuron: tangle" if real else "not a tau+ neuron"})
    store.append(project.output("reviews"), rows)


@pytest.fixture()
def two_projects(tmp_path):
    study_root, extra_root = tmp_path / "study", tmp_path / "jog"
    for root, name, kind in ((study_root, "Study", "tma"), (extra_root, "JOG whole slides", "slides")):
        root.mkdir()
        (root / "stainid.yaml").write_text(yaml.safe_dump({"name": name, "kind": kind}))
    study = load_project(study_root)
    _pool(study, 60, ["1", "2", "3"], seed=1)
    extra = load_project(extra_root)
    _pool(extra, 40, ["JOG_159_HIPPO_AT8"], seed=2)
    return study, extra


def test_extra_cases_join_every_fit_and_are_never_tested():
    y = np.array([True, False] * 15)
    groups = np.array(["1"] * 10 + ["2"] * 10 + ["extra:a"] * 10)
    fit_only = groups == "extra:a"
    donors = np.array(["d1"] * 10 + ["d2"] * 10 + ["d9"] * 5 + ["d1"] * 5)  # half the extra cases share a donor with TMA 1
    seen = []
    oof, scheme = out_of_fold(lambda tr, te: (seen.append((tr, te)), np.zeros(len(te)))[1], len(y), y, groups, fit_only=fit_only, donors=donors)
    assert len(seen) == 2  # two folds: the study's two TMAs; the extra project never makes one
    for train, test in seen:
        assert not fit_only[test].any() and len(set(groups[test])) == 1  # tested: study rows of one TMA only
        assert not set(donors[train]) & set(donors[test])  # an extra case of a held-out donor is left out of that fit
        held = groups[test][0]
        expected = np.flatnonzero(fit_only & ((donors == "d9") | (held != "1")))
        assert set(expected) <= set(train)  # every other extra case is in the fit
    assert np.isnan(oof[fit_only]).all() and not np.isnan(oof[~fit_only]).any()
    assert scheme == "2-fold, whole TMAs per fold"


def test_a_model_fits_on_the_extra_project_and_is_scored_on_the_study_alone(two_projects):
    study, extra = two_projects
    alone = json.loads((train_model(study, "AT8") / "report.json").read_text())
    assert alone["extra_projects"] == [] and alone["fitted_on"] == {"objects": 60, "positives": alone["new_model"]["positives"],
                                                                    "this_project": 60, "extra_projects": 0}
    study = save_project(study, {"training": {"extra_projects": ["../jog"]}})  # relative to the study
    assert extra_projects(study) == [extra.root.resolve()]
    folder = train_model(study, "AT8")
    report = json.loads((folder / "report.json").read_text())
    assert report["validation"] == "leave-one-TMA-out"
    assert report["new_model"]["n"] == 60 and report["sources"] == {"Ann": 60}  # tested on the study's labels only
    assert report["fitted_on"]["objects"] == 100 and report["fitted_on"]["extra_projects"] == 40
    used = report["extra_projects"][0]
    assert used["name"] == "JOG whole slides" and used["path"] == str(extra.root.resolve()) and used["objects"] == 40
    assert used["positives"] + used["negatives"] == 40 and used["sessions"][0]["objects"] == 40
    bundle = load(folder / "bundle.joblib")
    assert bundle["training_n"] == 100  # the model itself learned from both
    no_extra = json.loads((train_model(study, "AT8", extra=False) / "report.json").read_text())
    assert no_extra["extra_projects"] == [] and no_extra["fitted_on"]["objects"] == 60
    # nothing of the study changed: its label store holds its own labels only
    assert set(store.current(study.output("reviews")).source) == {"Ann"} and len(store.current(study.output("reviews"))) == 60


def test_missing_or_empty_extra_projects_are_reported_not_fatal(two_projects, tmp_path):
    study, _ = two_projects
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "stainid.yaml").write_text("name: Empty\nkind: slides\n")
    study = save_project(study, {"training": {"extra_projects": [str(empty), str(tmp_path / "gone"), str(study.root), str(empty)]}})
    report = json.loads((train_model(study, "AT8") / "report.json").read_text())
    notes = {e["name"]: e.get("note") for e in report["extra_projects"]}
    assert notes == {"Empty": "no labels yet", "gone": "not found"} and len(report["extra_projects"]) == 2  # not the study itself, nor twice
    assert report["fitted_on"]["extra_projects"] == 0 and report["new_model"]["n"] == 60


def test_simple_labels_of_an_extra_project_train_through_the_study_mapping(tmp_path):
    study_root, extra_root = tmp_path / "study", tmp_path / "jog"
    for root in (study_root, extra_root):
        root.mkdir()
        (root / "stainid.yaml").write_text(yaml.safe_dump({"name": root.name}))
    study = load_project(study_root)
    _pool(study, 60, ["1", "2", "3"], seed=1)
    _pool(load_project(extra_root), 40, ["J"], seed=2, simple_labels=True)
    # the extra project maps "neuritic plaque" to tau+ neurons; the study leaves it out: the study's mapping counts
    (extra_root / "stainid.yaml").write_text(yaml.safe_dump({"name": "jog", "labels": {"simple": {"AT8": {"training": {"neuritic_plaque": "tau+ neuron"}}}}}))
    study = save_project(study, {"training": {"extra_projects": [str(extra_root)]},
                                 "labels": {"simple": {"AT8": {"training": {"neuritic_plaque": ""}}}}})
    report = json.loads((train_model(study, "AT8") / "report.json").read_text())
    used = report["extra_projects"][0]
    labels = store.current(load_project(extra_root).output("reviews"))
    tangles, plaques, none = ((labels.value == v).sum() for v in ("tangle", "neuritic_plaque", "none"))
    assert used["objects"] == tangles + none and used["positives"] == tangles and used["negatives"] == none  # plaques: not used
    assert used["simple_mapping"]["training"]["neuritic_plaque"] == "" and used["simple_mapping"]["labels"]["neuritic_plaque"] == plaques
    assert report["simple_mapping"] is None  # the study's own labels are detailed
    assert set(labels.task) == {"simple"}  # the extra project's labels stay as they were given


def test_the_models_page_chooses_the_extra_projects(two_projects, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from stainid.api import registry, state
    from stainid.api.app import create_app

    study, extra = two_projects
    registry.add(extra.root)
    monkeypatch.setenv("STAINID_PROJECT", str(study.root))
    state.reset()
    app = TestClient(create_app(), client=("127.0.0.1", 50000))
    assert app.post("/api/session/admin", json={"password": "admin"}).status_code == 200
    listed = app.get("/api/models/extra-projects").json()
    offered = {p["path"]: p for p in listed["projects"]}
    assert offered[str(extra.root.resolve())]["enabled"] is False and str(study.root.resolve()) not in offered
    chosen = app.put("/api/models/extra-projects", json={"projects": [str(extra.root)]}).json()
    on = next(p for p in chosen["projects"] if p["enabled"])
    assert on["name"] == "JOG whole slides" and on["labels"] == {"NeuN": 0, "6E10": 0, "AT8": 40}
    assert yaml.safe_load((study.root / "stainid.yaml").read_text())["training"] == {"extra_projects": [str(extra.root.resolve())]}
    assert app.put("/api/models/extra-projects", json={"projects": [str(study.root)]}).status_code == 400
    assert app.put("/api/models/extra-projects", json={"projects": [str(study.root.parent)]}).status_code == 404
    off = app.put("/api/models/extra-projects", json={"projects": []}).json()
    assert not any(p["enabled"] for p in off["projects"]) and "training" not in yaml.safe_load((study.root / "stainid.yaml").read_text())
    state.reset()
