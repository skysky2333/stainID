"""Object texture (`stainid.v2.radiomics`): pyradiomics per object and per field, and its columns in the results tables."""
from __future__ import annotations

import json

import cv2
import numpy as np
import pandas as pd
import pytest
import yaml
from test_v2_pipeline import _proposals, run_calls_inline, v2_project  # noqa: F401 - the fixture

from stainid import candidates as store
from stainid.analysis import mask_aggregation
from stainid.project import load_project
from stainid.v2 import manifest, missing, radiomics

CORE = "LIP-1_B-2"


def _ring(cx: int, cy: int, r: int, n: int = 32) -> str:
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return " ".join(f"{int(round(cx + r * np.cos(a)))} {int(round(cy + r * np.sin(a)))}" for a in angles)


def _engine():
    pytest.importorskip("radiomics")
    pytest.importorskip("SimpleITK")
    return radiomics.extractor(type("P", (), {"config": {}, "root": None})())


def test_stems_and_the_parameter_file_features_are_all_named():
    assert radiomics.stem("original_glcm_JointEntropy") == "glcm_joint_entropy"
    assert radiomics.stem("original_glcm_ClusterShade") == "glcm_cluster_shade"  # not in FEATURES: made from its name
    enabled = yaml.safe_load(radiomics.PARAMS_FILE.read_text())["featureClass"]
    keys = {f"original_{group}_{name}" for group, names in enabled.items() for name in names}
    assert keys == set(radiomics.FEATURES), "every feature the parameter file enables has a column name and a label"


def test_an_object_and_its_field_are_measured_and_a_speck_gets_nothing():
    engine = _engine()
    rng = np.random.default_rng(1)
    rgb = np.full((120, 120, 3), 230, np.uint8)
    yy, xx = np.ogrid[:120, :120]
    for cx, cy in ((40, 40), (85, 80)):
        disc = (yy - cy) ** 2 + (xx - cx) ** 2 < 15 ** 2
        rgb[disc] = rng.integers(40, 170, (int(disc.sum()), 3)).astype(np.uint8)
    grey = radiomics.grey_levels(rgb, 0.3)
    assert grey.dtype == np.float32 and 0 <= grey.min() and grey.max() <= 255
    origin = (1000, 2000)  # the crop's top-left in core pixels
    rows = pd.DataFrame({"candidate_id": ["a", "b", "speck"], "tile_id": "t",
                         "sam_outline": [_ring(1040, 2040, 14), _ring(1085, 2080, 14), "1010 2010 1010 2010 1010 2010"]})
    objects, fields = radiomics.measure_field(engine, grey, origin, rows, (0.27, 0.27), 4, np.array([True, False, True]))
    stems = [v[0] for v in radiomics.FEATURES.values()]
    assert [o["candidate_id"] for o in objects] == ["a", "b", "speck"]
    for row in objects[:2]:
        assert all(np.isfinite(row[s]) for s in stems)
    assert not set(stems) & set(objects[2])  # pyradiomics cannot measure one pixel: no values, never 0
    whole, grey_only = fields
    assert whole["scope"] == "whole" and whole["n_objects"] == 3 and grey_only["scope"] == "grey" and grey_only["n_objects"] == 2
    one = np.zeros((120, 120), np.uint8)
    cv2.fillPoly(one, [np.asarray(store.points(_ring(40, 40, 14)), np.int32).reshape(-1, 2)], 1)
    assert grey_only["mask_px"] < whole["mask_px"] and grey_only["mask_px"] >= one.sum()  # the field mask is the union of its objects
    assert all(np.isfinite(whole[s]) for s in stems)


def test_field_means_are_weighted_by_mask_area_and_skip_empty_fields():
    rows = [{"n_objects": 2, "mask_px": 300, "glcm_joint_entropy": 2.0}, {"n_objects": 1, "mask_px": 100, "glcm_joint_entropy": 6.0},
            {"n_objects": 0, "mask_px": 0, "glcm_joint_entropy": float("nan")}, {"n_objects": 3, "mask_px": 50, "glcm_joint_entropy": ""}]
    assert radiomics.field_means(rows, ["glcm_joint_entropy"]) == {"glcm_joint_entropy": pytest.approx(3.0)}
    assert np.isnan(radiomics.field_means(rows[2:], ["glcm_joint_entropy"])["glcm_joint_entropy"])


def test_object_medians_need_the_outline_measures_minimum():
    rows = [{"area_um2": "50", "rad_glcm_joint_entropy": str(v)} for v in range(1, 12)] + [{"area_um2": "50", "rad_glcm_joint_entropy": ""}]
    out = mask_aggregation.summarize("6E10", rows, 1.0, ["glcm_joint_entropy"])
    assert out["plaque_rad_glcm_joint_entropy_median"] == 6.0  # 11 plaques with a value (≥ 10), the empty one left out
    assert np.isnan(mask_aggregation.summarize("6E10", rows[:9], 1.0, ["glcm_joint_entropy"])["plaque_rad_glcm_joint_entropy_median"])
    assert "plaque_rad_glcm_joint_entropy_median" not in mask_aggregation.summarize("6E10", rows, 1.0)  # no texture: no column


def test_texture_columns_are_labelled_and_say_why_they_are_missing():
    from stainid.analysis.dictionary import describe, measure

    from stainid.analysis.dictionary import TOPICS, family

    for prefix in ("plaque", "tau_neuron", "neun_soma"):
        for column in (f"{prefix}_rad_glcm_joint_entropy_median", f"{prefix}_rad_field_firstorder_mean"):
            found, whole = measure(column), measure(f"{column}_whole_core")
            assert describe(column) and found["topic"] == "texture" in TOPICS and "texture" in found["label"]  # a section of their own
            assert whole["topic"] == "texture" and whole["label"] == f"{found['label']}, whole core" and whole["description"]
            assert family(column) == family(f"{column}_whole_core") == "texture"  # and their own FDR family
    assert family("plaque_area_median_um2") == family("plaque_area_median_um2_whole_core") == ""
    ran = {"tiles_6E10": 2, "plaque_texture_fields": 4, "plaque_textured": 3, "plaque_texture_fields_with_objects": 2}
    assert missing._reason("plaque_rad_glcm_joint_entropy_median", "a", {"a": ran}, None) == ("too_few_objects", 3)
    assert missing._reason("plaque_rad_field_glcm_joint_entropy", "a", {"a": {**ran, "plaque_texture_fields_with_objects": 0}}, None) == \
        ("too_few_objects", 3)
    assert missing._reason("plaque_rad_glcm_joint_entropy_median", "a", {"a": {"tiles_6E10": 2}}, None) == ("not_measured", 0)  # not run
    assert missing._reason("plaque_rad_glcm_joint_entropy_median", "a", {"a": {"tiles_6E10": 0}}, None) == ("no_image", 0)


def _outlined(project) -> None:
    """Stage 3 without Segment Anything: each accepted object's own outline as its kept outline."""
    for stain in radiomics.STAINS:
        target = store.root(project) / "candidates" / stain / f"{CORE}.csv.gz"  # write_table makes its .parquet twin
        frame = store.read_table(target).copy()
        accepted = frame.decision == "accepted"
        frame["sam_status"] = np.where(accepted, "ok", "")
        frame["sam_outline"] = np.where(accepted, frame.outline, "")
        for sam, own in (("sam_area_um2", "area_um2"), ("sam_solidity", "solidity"), ("sam_circularity", "circularity")):
            frame[sam] = frame[own] if own in frame else 1.0
        frame["sam_elongation"], frame["sam_normalized_mean_dab"] = 1.0, 1.0
        store.write_table(target, frame)
        manifest.write(project, "outlines", stain, CORE, {"stage": "outlines", "digest": "test"})


def test_the_step_end_to_end_into_the_results_tables(v2_project):  # noqa: F811 - the fixture
    _engine()
    project = load_project(v2_project)
    _proposals(project)
    run_calls_inline(project)
    _outlined(project)
    from stainid.v2.measures import write_measures

    write_measures(project)
    folder = store.root(project) / "measures"
    assert not [c for c in pd.read_csv(folder / "results_core.csv").columns if "_rad_" in c]  # never run: no texture columns
    assert sorted(radiomics.todo(project)) == sorted((CORE, s) for s in radiomics.STAINS)
    done = radiomics.run(project, log=lambda text: None)
    assert len(done) == 3 and radiomics.todo(project) == []
    assert radiomics.progress(project)["done"] == 3
    stems = [v[0] for v in radiomics.FEATURES.values()]
    accepted = {}
    for stain in radiomics.STAINS:
        objects, fields = radiomics.current(project, stain, CORE)
        found = store.read_core(project, stain, CORE)
        accepted[stain] = int(((found.decision == "accepted") & (found.sam_status == "ok")).sum())
        assert len(objects) == accepted[stain] and set(fields.tile_id) == {f"{CORE}_{stain}_W{r:02d}{c:02d}" for r in (0, 1) for c in (0, 1)}
        assert set(fields.scope) == {"whole"}  # no white matter
        assert fields.n_objects.sum() == accepted[stain]
    write_measures(project)
    for level, key in (("core", "core_id"), ("donor_region", "sample_region_id")):
        table = pd.read_csv(folder / f"results_{level}.csv")
        gone = pd.read_csv(folder / f"results_missing_{level}.csv").set_index("measure")
        for stain, prefix in radiomics.PREFIX.items():
            if not accepted[stain]:
                continue
            for name in stems:
                median, field = f"{prefix}_rad_{name}_median", f"{prefix}_rad_field_{name}"
                assert median in table and field in table
                assert np.isfinite(table[field].iloc[0]), field
                if accepted[stain] >= mask_aggregation.MINIMUM_OBJECTS[stain]:
                    assert np.isfinite(table[median].iloc[0]), median
                else:
                    assert gone.loc[median, "reason"] == "too_few_objects" and gone.loc[median, "n_objects"] == accepted[stain]
        ids = [c for c in missing.IDS if c in table]
        cells = {(str(i), c) for c in table.columns if c not in ids for i in table.loc[table[c].isna(), key]}
        gone = gone.reset_index()
        assert set(zip(gone[key].astype(str), gone.measure)) == cells  # every empty cell, the texture ones too, says why
    # a new setting makes every core image out of date; stage 5 then leaves its texture empty and says so
    config = yaml.safe_load((v2_project / "stainid.yaml").read_text())
    config["v2"] = {"radiomics": {"od_ceiling": 0.4}}
    (v2_project / "stainid.yaml").write_text(yaml.safe_dump(config))
    project = load_project(v2_project)
    assert sorted(radiomics.todo(project)) == sorted((CORE, s) for s in radiomics.STAINS)
    assert radiomics.current(project, "6E10", CORE) is None
    lines = write_measures(project)
    assert any("object texture: 3 core images out of date" in line for line in lines)
    table = pd.read_csv(folder / "results_core.csv")
    assert not [c for c in table.columns if "_rad_" in c]
    assert json.loads(radiomics.manifest_path(project, "6E10", CORE).read_text())["inputs"]["od_ceiling"] == 0.3


def test_texture_does_not_depend_on_the_object_orientation():
    engine = _engine()
    rng = np.random.default_rng(3)
    grey = np.zeros((80, 80), np.float32)
    grey[:, ::3] = 200
    grey += rng.integers(0, 40, grey.shape)
    yy, xx = np.ogrid[:80, :80]
    mask = (((yy - 40) / 30) ** 2 + ((xx - 40) / 12) ** 2 < 1).astype(np.uint8)
    turned = radiomics.features(engine, np.ascontiguousarray(np.rot90(grey)), np.ascontiguousarray(np.rot90(mask)), (0.27, 0.27))
    assert radiomics.features(engine, grey, mask, (0.27, 0.27)) == pytest.approx(turned)
