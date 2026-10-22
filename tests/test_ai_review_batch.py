"""AI review batches: the fields and objects chosen, the picture sheets, and the reviewers' CSVs back into the label log
as one read-only session per stain (accounts `ai:<name>`), as "Fix the field" records the same answers."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from stainid.labels import candidate_store, sessions, store
from stainid.labels import vocabulary as vocab
from stainid.project import load_project
from stainid.review import ai_batch, ai_ingest

TILE = "LIP-1_B-2_6E10_S01"  # x_px = y_px = 300 in a 1200 px core image: field origin (172, 172)
CORE = "LIP-1_B-2"


def test_reviewers_labels_resolve_in_the_stain_vocabulary():
    words = vocab.defaults("6E10")
    assert [c.label for c in ai_batch.choices(words)] == ["plaque / compact", "plaque / diffuse", "plaque / cored-neuritic", "plaque / small", "CAA",
                                                          "intraneuronal", "not a plaque", "unsure"]
    assert [c.key for c in ai_batch.choices(words)][:5] == ["1", "2", "3", "4", "Q"]
    for said, want in [("plaque / compact", ("plaque", "compact")), ("Plaque/Diffuse", ("plaque", "diffuse")), ("q", ("CAA", "")),
                       ("CAA / vascular amyloid", ("CAA", "")), ("Del", ("not a plaque", "")), ("`", ("unsure", "")), ("unsure", ("unsure", "")),
                       ("compact plaque", ("plaque", "compact")), ("plaque", ("plaque", "")), ("vascular", ("CAA", ""))]:
        assert ai_batch.resolve_label(words, said) == want, said
    assert ai_batch.resolve_label(words, "tangle") is None and ai_batch.resolve_label(words, "") is None
    neun = vocab.defaults("NeuN")
    assert [c.label for c in ai_batch.choices(neun)] == ["NeuN+ neuron", "not a neuron", "unsure"]
    assert ai_batch.resolve_label(neun, "NeuN+ neuron / strong") == ("NeuN+ neuron", "")  # a hidden type: the identity
    at8 = vocab.defaults("AT8")
    assert ai_batch.resolve_label(at8, "tau+ neuron / pretangle") == ("tau+ neuron", "pretangle") and ai_batch.resolve_label(at8, "e") == ("thread", "")
    text = ai_batch.instructions(at8, "AT8", 4)
    assert "`neuritic plaque` | Q" in text and "`not tau` | Del" in text and "outline_ok" in text and "AT8-F03-M1" in text


def _objects(n: int, seed: int = 0, box: float = 2048) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0, box, size=(n, 2))
    p = rng.uniform(0, 1, n)
    return pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1], "probability": p, "accepted": p >= 0.5, "near": abs(p - 0.5) <= 0.15,
                         "decision": np.where(p >= 0.5, "accepted", "rejected"), "reason": np.where(p >= 0.5, "accepted", "rejected_by_model")})


def test_a_crowded_field_is_cut_to_a_sub_field_and_capped_by_a_stratified_sample():
    few = _objects(60)
    assert ai_batch.window(few, (0, 0, 2048, 2048), "rich") == ((0, 0, 2048, 2048), 1)  # at most 80: the whole field
    many = _objects(400)
    box, n = ai_batch.window(many, (0, 0, 2048, 2048), "rich")
    inside = many[(many.x >= box[0]) & (many.x < box[2]) & (many.y >= box[1]) & (many.y < box[3])]
    assert n == 3 and 20 <= len(inside) <= 80 and box[2] - box[0] == pytest.approx(2048 / 3)
    dense = _objects(5000)
    box, n = ai_batch.window(dense, (0, 0, 2048, 2048), "threshold")
    inside = dense[(dense.x >= box[0]) & (dense.x < box[2]) & (dense.y >= box[1]) & (dense.y < box[3])]
    kept = ai_batch.stratified(inside, 80, seed=1)
    assert n == ai_batch.GRIDS[-1] and len(inside) > 80 and len(kept) <= 80
    assert kept.accepted.any() and (~kept.accepted).any()  # every call kept


def test_halves_balance_categories_and_about_a_tenth_overlaps():
    fields = pd.DataFrame({"field": [f"F0{i}" for i in range(1, 9)], "category": ["rich"] * 4 + ["sparse"] * 2 + ["threshold"] * 2})
    halves = ai_batch.split_halves(fields, {f: 40 for f in fields.field}, seed=3)
    for category in ("rich", "sparse", "threshold"):
        assert sorted(halves[f] for f in fields.field[fields.category == category]) == sorted({"rich": "AABB", "sparse": "AB", "threshold": "AB"}[category])
    objects = pd.DataFrame({"field": np.repeat(fields.field, 40), "accepted": np.tile([True, False], 160)}).reset_index(drop=True)
    objects["half"] = objects.field.map(halves)
    also = ai_batch.overlap(objects, 0.10, seed=4)
    assert (also != "").sum() == 32 and set(also[objects.half == "A"]) == {"", "B"} and set(also[objects.half == "B"]) == {"", "A"}
    assert (also[objects.half == "A"] == "B").sum() == 16


def test_reviewers_disagreeing_on_an_object_make_it_unsure():
    words = vocab.defaults("6E10")
    said = pd.DataFrame({"review_id": ["X1", "X1", "X2", "X2", "X3", "X4", "X4"], "source": ["ai:a", "ai:b"] * 2 + ["ai:a", "ai:a", "ai:b"],
                         "identity": ["plaque", "plaque", "plaque", "CAA", "plaque", "not a plaque", "not a plaque"],
                         "subtype": ["compact", "diffuse", "compact", "", "compact", "", ""], "outline_ok": ["ok", "ok", "ok", "", "too big", "", ""]})
    out = ai_ingest.resolve_overlap(said, words).set_index(["review_id", "source"])
    assert list(out.loc["X1"].subtype) == ["", ""] and list(out.loc["X1"].identity) == ["plaque", "plaque"]  # a disputed type is dropped
    assert list(out.loc["X2"].identity) == ["unsure", "unsure"] and list(out.loc["X2"].said) == ["plaque / compact", "CAA"]
    assert out.loc[("X3", "ai:a")].agreed is None and out.loc[("X3", "ai:a")].subtype == "compact"  # one reviewer: as said
    assert list(out.loc["X4"].agreed) == [True, True]
    assert ai_ingest.account("Claude Reviewer 1") == "ai:claude-reviewer-1" and ai_ingest.account("ai:x") == "ai:x"


def _row(x: float, y: float, p: float) -> dict:
    accepted = p >= 0.5
    ring = [[x - 9, y - 9], [x + 9, y - 9], [x + 9, y + 9], [x - 9, y + 9]]
    return {"candidate_id": candidate_store.candidate_id("6E10", CORE, x, y), "x_core": x, "y_core": y, "tile_id": TILE, "area_um2": 25.0,
            "radius_px": 9.0, "sources": "dab@1", "proposal_threshold": 1.0, "outline": ring, "sam_outline": None, "sam_status": "",
            "probability": p, "threshold": 0.5, "model_id": "m@1", "decision": "accepted" if accepted else "rejected",
            "reason": "accepted" if accepted else "rejected_by_model", "class": "plaque" if accepted else "not a plaque",
            "subtype": "compact" if accepted else "", "outline_default": "stain", "mean_dab_od": 0.3}


@pytest.fixture()
def batch(synthetic_project, monkeypatch):
    monkeypatch.chdir(synthetic_project)
    project = load_project(synthetic_project)
    rng = np.random.default_rng(5)
    xy = rng.uniform(330, 780, size=(30, 2)).round()
    probability = np.r_[np.linspace(0.55, 0.95, 8), rng.uniform(0, 0.45, 22)]
    candidate_store.write_core(project, "6E10", CORE, pd.DataFrame([_row(x, y, p) for (x, y), p in zip(xy, probability)]))
    summary = ai_batch.build_stain(project, "6E10", round_=4, n_fields=1, seed=0, log=lambda *a, **k: None)
    return project, summary


def test_a_batch_is_built_with_sheets_overviews_and_assignments(batch):
    project, summary = batch
    out = ai_batch.stain_folder(project, "6E10", 4)
    manifest = pd.read_csv(out / "manifest.csv")
    assert summary["objects"] == len(manifest) == 30 and summary["fields"] == 1
    assert manifest.review_id.str.match(r"^6E10-F01-\d{3}$").all() and manifest.candidate_id.nunique() == 30
    assert set(manifest.half) == {"A"} and (manifest.also_in == "B").sum() == summary["overlap"] > 0
    assert set(manifest.outline_choice) == {"stain"} and all(len(json.loads(p)) == 4 for p in manifest.outline_points)
    assert (out / "sheets" / "A_01.jpg").exists() and (out / "sheets" / "A_02.jpg").exists() and (out / "sheets" / "B_x01.jpg").exists()
    assert (out / "overviews" / "F01.jpg").exists() and (out / "overviews" / "F01_raw.jpg").exists()
    template = pd.read_csv(out / "review_labels_A.csv")
    assert list(template.columns) == ai_batch.CSV_COLUMNS and len(template) == 30
    assert "plaque / compact" in (out / "REVIEW_INSTRUCTIONS.md").read_text() and "F01" in (out / "assignment_A.md").read_text()
    with pytest.raises(FileExistsError):  # a batch is never overwritten
        ai_batch.build_stain(project, "6E10", round_=4, n_fields=1)


def test_reviewers_csvs_become_a_read_only_session_of_ai_accounts(batch, monkeypatch):
    project, _ = batch
    out = ai_batch.stain_folder(project, "6E10", 4)
    manifest = pd.read_csv(out / "manifest.csv", keep_default_na=False)
    both = manifest[manifest.also_in == "B"].review_id.tolist()
    a = pd.DataFrame({"review_id": manifest.review_id, "label": np.where(manifest.accepted.astype(str) == "True", "plaque / compact", "not a plaque"),
                      "confidence": "high", "outline_ok": np.where(manifest.accepted.astype(str) == "True", "ok", ""), "rationale": "dense brown core",
                      "reviewer": "Alpha", "field_x": "", "field_y": ""})
    first = manifest.review_id.iloc[0]
    a.loc[a.review_id == first, ["label", "outline_ok"]] = ["plaque / diffuse", "too big"]
    a.loc[a.review_id == manifest.review_id.iloc[1], "label"] = "tangle"  # not a 6E10 label: reported, left out
    away = manifest.x.astype(float).min() - 100  # far from every candidate, inside the field
    on = manifest.iloc[2]
    a = pd.concat([a, pd.DataFrame([
        {"review_id": "6E10-F01-M1", "label": "plaque / diffuse", "confidence": "moderate", "rationale": "faint deposit", "reviewer": "Alpha",
         "field_x": str(max(130.0, away)), "field_y": "140"},
        {"review_id": "6E10-F01-M2", "label": "plaque / small", "confidence": "low", "reviewer": "Alpha", "field_x": str(on.x), "field_y": str(on.y)}])],
        ignore_index=True)
    a.to_csv(out / "review_labels_A.csv", index=False)
    b = pd.DataFrame({"review_id": both, "label": "CAA", "confidence": "low", "outline_ok": "ok", "rationale": "vessel wall", "reviewer": "beta",
                      "field_x": "", "field_y": ""})
    b.loc[0, "label"] = a.set_index("review_id").label[both[0]]  # agrees on one object
    b.to_csv(out / "review_labels_B.csv", index=False)

    root = project.output("reviews")
    dry = ai_ingest.ingest(project, "6E10", 4, dry_run=True, log=lambda *a, **k: None)
    assert dry["written"] > 0 and store.current(root).empty  # a dry run writes nothing
    report = ai_ingest.ingest(project, "6E10", 4, log=lambda *a, **k: None)
    name = ai_ingest.session_name("6E10", 4)
    assert name == "ai_review_round4_6e10" and report["title"] == "AI review round 4 · 6E10"
    meta = sessions.meta(root, name)
    assert meta["kind"] == "ai_review" and meta["read_only"] and not meta["open"] and meta["fields"] == [TILE]
    assert meta["ai_review"]["reviewers"] == ["ai:alpha", "ai:beta"]
    assert [r["label"] for r in report["invalid"]] == ["tangle"] and report["overlap_objects"] == len(both) and report["overlap_agreed"] == 1

    labels = store.current(root)
    assert set(labels.source) == {"ai:alpha", "ai:beta"} and set(labels.session) == {name}
    raw = store.raw_records(root)
    assert all(r["account"] == r["source"] for r in raw)
    classes = labels[labels.task == "class"]
    named = classes[classes.candidate.notna()].set_index(["source", "candidate"])
    by_id = manifest.set_index("review_id")
    row = named.loc[("ai:alpha", by_id.candidate_id[first])]
    assert (row.value, row.subtype, row.confirmed) == ("plaque", "diffuse", False) and row.meta["outline"] == "too big"
    assert row.tile_id == TILE and row.x == pytest.approx(float(by_id.x[first]), abs=0.05)
    rejected = by_id[by_id.accepted.astype(str) == "False"].index.difference([*both, *manifest.review_id.iloc[:3]])[0]
    assert bool(named.loc[("ai:alpha", by_id.candidate_id[rejected])].confirmed)  # detection's own call: confirmed
    split = named.loc[(slice(None), by_id.candidate_id[both[1]]), :]
    assert set(split.value) == {"unsure"} and {m["said"] for m in split.meta} == {"CAA", a.set_index("review_id").label[both[1]]}
    agreed = named.loc[(slice(None), by_id.candidate_id[both[0]]), :]
    assert agreed.value.nunique() == 1 and agreed.value.iloc[0] != "unsure"
    outlines = labels[labels.task == "outline"]
    assert len(outlines) and set(outlines.value) == {"stain"} and outlines.confirmed.all()  # as the app records an outline left as it is
    assert by_id.candidate_id[first] not in set(outlines.candidate)  # "too big": no outline confirmed
    added = classes[classes.candidate.isna()]
    assert len(added) == 1 and added.value.iloc[0] == "plaque" and added.meta.iloc[0]["missed"]  # an object people added
    assert report["missed"] == {"added": 1, f"a candidate lies there ({on.candidate_id})": 1}
    assert pd.read_csv(out / "outline_judgements.csv").outline_ok.tolist() == ["too big"]

    listed = sessions.label_sets(labels, "6E10", {name: meta["title"]})  # the Labels page and the Models page's choice
    assert [s["title"] for s in listed] == ["AI review round 4 · 6E10"] and set(listed[0]["people"]) == {"ai:alpha", "ai:beta"}
    assert len(sessions.in_sessions(labels, [name])) == len(labels) and sessions.in_sessions(labels, ["other"]).empty

    again = ai_ingest.ingest(project, "6E10", 4, log=lambda *a, **k: None)  # nothing changed: nothing written
    assert again["written"] == 0 and again["deleted"] == 0 and len(store.history(root)) == len(labels)
    (out / "review_labels_B.csv").unlink()  # beta's labels withdrawn: marked deleted, alpha's overlap labels as they said
    third = ai_ingest.ingest(project, "6E10", 4, log=lambda *a, **k: None)
    now = store.current(root)
    assert third["deleted"] > 0 and set(now.source) == {"ai:alpha"} and "unsure" not in set(now[now.task == "class"].value)

    from stainid.review import ai_ingest as module
    monkeypatch.setattr(module.excluded_areas, "drop_excluded", lambda project, frame, *a, **k: frame.iloc[1:])
    assert ai_ingest.ingest(project, "6E10", 4, dry_run=True, log=lambda *a, **k: None)["in_excluded_areas"] == 1
