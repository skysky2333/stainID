"""The v2 detection stages on synthetic tiles: component-tree proposals, the NeuN split of merged profiles, de-duplication,
stable ids, calls from stored features only, the thread skeleton round trip, the candidate store API, manifests and
outdated detection, and the measures tables."""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

from stainid import candidates as store
from stainid.project import load_project
from stainid.v2 import classify, proposals, run
from stainid.v2.context import CONTEXT_NAMES, context_features_many
from stainid.v2.field import Field
from stainid.v2.geometry import Occupancy, Region, same_object
from stainid.v2.threads import rasterize, segments
from stainid.v2.tree import build_tree, tree_regions

PS = 0.2738
BROWN = (110, 70, 35)
PALE = (190, 150, 120)
BLUE = (110, 110, 170)


def field_of(rgb: np.ndarray, threshold: float = 0.05, nuclei: np.ndarray | None = None, margin: int = 0) -> Field:
    h, w = rgb.shape[:2]
    inner = (slice(margin, h - margin), slice(margin, w - margin))
    tile = {"tile_id": "T_W0101", "x_px": margin, "y_px": margin, "width_px": w - 2 * margin, "height_px": h - 2 * margin,
            "pixel_width_um": PS, "pixel_height_um": PS, "context_px": margin}
    return Field(tile, rgb, inner, (0, 0), threshold, PS, np.zeros((h, w), bool), nuclei)


def blank(size: int = 400) -> np.ndarray:
    return np.full((size, size, 3), 228, np.uint8)


# ---------------------------------------------------------------- component tree

def test_component_tree_splits_touching_objects_and_keeps_one_node_per_object():
    rgb = blank()
    cv2.circle(rgb, (150, 200), 22, PALE, -1)  # a pale halo joining two dark cells: one component at the low level
    cv2.circle(rgb, (250, 200), 22, PALE, -1)
    cv2.rectangle(rgb, (150, 185), (250, 215), PALE, -1)
    cv2.circle(rgb, (150, 200), 16, BROWN, -1)
    cv2.circle(rgb, (250, 200), 16, BROWN, -1)
    cv2.circle(rgb, (100, 330), 15, BROWN, -1)  # an isolated cell: one object, found at every level
    field = field_of(rgb)
    allowed = np.ones(field.shape, bool)
    roots = build_tree(field, [0.5, 1.0, 1.5], allowed, 20)
    assert len(roots) == 2  # the joined pair and the isolated cell
    regions = tree_regions(field, [0.5, 1.0, 1.5], allowed, 15.0, 650.0, 25.0, 6000.0, "split")
    centres = sorted((round(r.centroid()[0] / 10), round(r.centroid()[1] / 10)) for r in regions)
    assert centres == [(10, 33), (15, 20), (25, 20)]  # the pair is split, each object once
    assert all(r.level is not None and r.source.startswith("dab@") for r in regions)
    plaques = tree_regions(field, [0.5, 1.0], allowed, 20.0, 50000.0, 20.0, None, "roots")
    assert len(plaques) == 2  # roots mode: the halo with both cores is one deposit


def test_merged_profile_above_the_size_limit_is_cut_by_the_watershed():
    rgb = blank(500)
    for x in (160, 225, 290):  # three touching cells, together far above 650 µm²
        cv2.circle(rgb, (x, 250), 36, BROWN, -1)
    field = field_of(rgb)
    allowed = np.ones(field.shape, bool)
    regions = tree_regions(field, [1.0], allowed, 15.0, 650.0, 25.0, 6000.0, "split")
    assert len(regions) == 3 and all(r.source == "split@1" for r in regions)
    assert all(r.area_px * PS * PS <= 650.0 for r in regions)


def test_neun_proposals_split_merged_neurons_and_merge_nuclei_into_their_profile():
    from stainid.v2 import neun
    from stainid.v2.params import DEFAULTS

    rgb = blank(500)
    cv2.circle(rgb, (120, 120), 14, BROWN, -1)  # a neuron, its nucleus found by Cellpose too
    for x in (260, 325, 390):  # merged neurons
        cv2.circle(rgb, (x, 300), 36, BROWN, -1)
    cv2.circle(rgb, (120, 400), 9, BLUE, -1)  # a glial nucleus, no stain
    cv2.ellipse(rgb, (400, 120), (45, 7), 0, 0, 360, BROWN, -1)  # a long thin stained cell the legacy profile filter drops
    nuclei = np.zeros(rgb.shape[:2], np.int32)
    cv2.circle(nuclei, (120, 120), 8, 1, -1)
    cv2.circle(nuclei, (120, 400), 9, 2, -1)
    cv2.ellipse(nuclei, (400, 120), (20, 6), 0, 0, 360, 3, -1)
    field = field_of(rgb, nuclei=nuclei.astype(np.uint32))
    p = DEFAULTS["proposals"]
    regions = neun.propose(field, p["NeuN"], {k: v for k, v in p.items() if k not in ("NeuN", "6E10", "AT8")})
    by_source = {}
    for r in regions:
        by_source.setdefault(r.source, []).append(r)
    neuron = min(regions, key=lambda r: np.hypot(*(np.array(r.centroid()) - (120, 120))))
    assert neuron.source == "profile" and "cellpose" in neuron.sources  # one candidate, both sources recorded
    assert sum(len(v) for k, v in by_source.items() if k.startswith("split@")) == 3  # the merge cut by the tree
    glia = min(regions, key=lambda r: np.hypot(*(np.array(r.centroid()) - (120, 400))))
    assert glia.source == "cellpose"
    thin = min(regions, key=lambda r: np.hypot(*(np.array(r.centroid()) - (400, 120))))
    assert thin.source == "cellpose" and "profile" in thin.sources  # the legacy candidate there was the Cellpose cell


def test_at8_tile_with_almost_no_tissue_is_proposed_without_failing():
    from stainid.v2.params import DEFAULTS

    rgb = np.full((300, 300, 3), 245, np.uint8)  # glass, and a speck of tissue too small for the AT8 network analysis
    rgb[10:16, 10:16] = PALE
    settings = DEFAULTS["proposals"]
    shared = {k: v for k, v in settings.items() if k not in ("NeuN", "6E10", "AT8")}
    field = field_of(rgb)
    result = proposals.process_tile(field, "AT8", settings["AT8"], shared)
    assert result["rows"] == [] and result["threads"] == []
    assert result["summary"]["at8_valid_area_mm2"] == 0.0 and result["summary"]["thread_skeleton_px"] == 0
    assert proposals.mask_piece(field, "AT8").shape[0] > 0


def test_same_object_rules():
    occupancy = Occupancy((100, 100))
    kept = Region(10, 10, np.ones((20, 20), bool), "profile")
    occupancy.add(kept)
    other = Region(60, 60, np.ones((10, 10), bool), "profile")
    occupancy.add(other)
    args = dict(same_iou=0.5, same_inside=0.8, covered=0.3)
    assert same_object(Region(12, 12, np.ones((18, 18), bool), "dab@1"), occupancy, **args) == ("same", 0)
    assert same_object(Region(5, 5, np.ones((30, 30), bool), "dab@0.5"), occupancy, **args) == ("same", 0)  # holds it, < 3x
    assert same_object(Region(0, 0, np.ones((80, 80), bool), "dab@0.5"), occupancy, **args)[0] == "merge"
    assert same_object(Region(40, 40, np.ones((8, 8), bool), "dab@1"), occupancy, **args)[0] == "new"


def test_context_features_many_equal_the_per_crop_function():
    from stainid.stains.amyloid.classifier import context_features
    from stainid.stains.neun.audit import crop_with_padding

    rng = np.random.default_rng(3)
    rgb = rng.integers(60, 240, size=(300, 320, 3), dtype=np.uint8)
    field = field_of(rgb)
    centres = [(150.4, 160.2), (10.0, 12.0), (310.6, 295.5)]  # inside, and padded with white beyond the field
    batch = context_features_many(field, centres, [10.0, 6.0, 14.0], 55.0)
    size = max(64, round(55.0 / PS))
    for (x, y), diameter, got in zip(centres, [10.0, 6.0, 14.0], batch):
        want = context_features(crop_with_padding(rgb, x, y, size), diameter, context_um=55.0)
        assert set(got) == set(CONTEXT_NAMES)
        for name in CONTEXT_NAMES:
            assert got[name] == pytest.approx(want[name], rel=1e-9, abs=1e-12), name


# ---------------------------------------------------------------- threads

def test_thread_segments_cover_the_skeleton_once_and_cut_at_candidates():
    skeleton = np.zeros((120, 120), bool)
    cv2.line(skeleton.view(np.uint8), (10, 60), (110, 60), 1, 1)
    cv2.line(skeleton.view(np.uint8), (60, 10), (60, 110), 1, 1)
    skeleton = skeleton.astype(bool)
    width = np.where(skeleton, 1.5, 0.0)
    owner = np.zeros(skeleton.shape, np.int32)
    owner[50:70, 80:100] = 7  # a candidate the horizontal thread runs through
    found = segments(skeleton, width, owner, PS)
    assert sum(s["n_px"] for s in found) == skeleton.sum()
    assert sum(s["width_sum_um"] for s in found) == pytest.approx(width[skeleton].sum())
    assert {s["kind"] for s in found} == {"piece", "junction"}
    inside = [s for s in found if s["owner"] == 7]
    assert inside and sum(s["n_px"] for s in inside) == (skeleton & (owner == 7)).sum()
    drawn = rasterize([s["points"].ravel().tolist() for s in found], skeleton.shape)
    assert (drawn & skeleton).sum() >= 0.95 * skeleton.sum()


# ---------------------------------------------------------------- the store API

def test_store_ids_encodings_and_reads(tmp_path):
    project = load_project(tmp_path)
    ids = store.unique_ids("NeuN", "LIP-1_A-1", [10.2, 10.4, 30.0], [20.0, 19.9, 5.0])
    assert ids == ["NeuN:LIP-1_A-1:10:20", "NeuN:LIP-1_A-1:10:20~2", "NeuN:LIP-1_A-1:30:5"]
    assert store.parse_id(ids[1]) == ("NeuN", "LIP-1_A-1", 10, 20)
    assert store.points(store.encode_points([[1, 2], [3, 4], [5, 6]])) == [1, 2, 3, 4, 5, 6]
    assert store.rings("1 2 3 4 5 6;7 8 9 10 11 12") == [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]
    assert store.sam_masks("1 2 3 4 5 6||7 8 9 10 11 12")[1] == []
    assert store.sam_scores("0.9|0.5|0.1") == [0.9, 0.5, 0.1]
    frame = pd.DataFrame({"candidate_id": ids, "stain": "NeuN", "core_id": "LIP-1_A-1", "tile_id": ["T1", "T1", "T2"],
                          "x_core": [10.2, 10.4, 30.0], "y_core": [20.0, 19.9, 5.0], "area_um2": [50.0, 60.0, 70.0],
                          "outline": ["0 0 20 0 20 20", "1 1 2 2 3 1", "25 0 35 0 35 10"], "m_context_dab_mean": [0.1, 0.2, np.nan]})
    store.write_table(store.root(project) / "proposals" / "NeuN" / "LIP-1_A-1.csv.gz", frame)
    got = store.read_core(project, "NeuN", "LIP-1_A-1")  # no calls yet: proposals with empty call columns
    assert list(got.candidate_id) == ids and got.decision.eq("").all() and got.probability.isna().all()
    assert list(store.read_tile(project, "NeuN", "T2", "LIP-1_A-1").candidate_id) == [ids[2]]
    assert list(store.read_bbox(project, "NeuN", "LIP-1_A-1", 0, 0, 20, 30).candidate_id) == ids[:2]
    assert list(store.by_id(project, [ids[2], ids[0], "NeuN:LIP-1_A-1:99:99"]).candidate_id) == [ids[2], ids[0]]
    assert store.stored(project, "NeuN", "LIP-1_A-1", "proposals").suffix in (".parquet", ".gz")
    assert store.stored(project, "NeuN", "LIP-1_A-1") is None and store.cores(project, "NeuN", "proposals") == {"LIP-1_A-1"}
    matrix = store.model_matrix(got, ["context_dab_mean", "area_um2", "unknown"])
    assert matrix.shape == (3, 3) and np.isnan(matrix[2, 0]) and matrix[1, 1] == 60.0 and np.isnan(matrix[:, 2]).all()
    assert store.outlines(project, "NeuN", "LIP-1_A-1")[0]["rings"] == [[0, 0, 20, 0, 20, 20]]


# ---------------------------------------------------------------- the stages on a synthetic project

def _bundle(names: list[str], seed: int = 0) -> object:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(60, len(names)))
    y = (x[:, 0] > 0).astype(int)
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), RandomForestClassifier(n_estimators=10, random_state=0)).fit(x, y)


def _models(root: Path) -> dict:
    from stainid.stains.amyloid.classifier import MORPHOLOGY_FEATURES
    from stainid.stains.tau.neurons import TAU_FEATURES
    from stainid.v2.neun import MEASUREMENT

    neun_names = [*MEASUREMENT, "source_is_dab_profile", "proposal_is_positive_profile", "source_candidate_is_positive",
                  "source_candidate_is_review", *CONTEXT_NAMES]
    amyloid_names = [*MORPHOLOGY_FEATURES, *CONTEXT_NAMES]
    tau_names = [*TAU_FEATURES, *[f"ctx_{n}" for n in CONTEXT_NAMES], "from_ring"]
    folder = root / "data" / "models"
    folder.mkdir(parents=True, exist_ok=True)
    bundles = {
        "neun": {"classifier": _bundle(neun_names), "feature_names": neun_names, "usable_features": np.ones(len(neun_names), bool),
                 "context_threshold": 0.5, "addition_threshold": 0.6},
        "amyloid": {"identity_classifier": _bundle(amyloid_names), "identity_feature_names": amyloid_names, "identity_threshold": 0.5,
                    "morphotype_threshold": 1.5, "morphotype_minimum_diameter_um": 15.0},
        "tau": {"classifier": _bundle(tau_names), "feature_names": tau_names, "probability_threshold": 0.4, "nms_radius_um": 12.0,
                "mature_p90_threshold": 0.16},
    }
    paths = {}
    for key, bundle in bundles.items():
        (folder / key).mkdir(exist_ok=True)
        joblib.dump(bundle, folder / key / "bundle.joblib")
        paths[key] = f"data/models/{key}/bundle.joblib"
    return paths


@pytest.fixture()
def v2_project(tmp_path: Path) -> Path:
    """One core with three stains (900 px images, four 400 px tiles each), nuclei, slide thresholds and tiny models."""
    rng = np.random.default_rng(5)
    core_dir = tmp_path / "data" / "cores" / "LIP-1" / "B-2"
    core_dir.mkdir(parents=True)
    nuclei_dir = tmp_path / "data" / "analysis" / "tile_nuclei"
    nuclei_dir.mkdir(parents=True)
    tiles = []
    for stain in ("NeuN", "6E10", "AT8"):
        rgb = np.full((900, 900, 3), 225, np.uint8)
        nuclei = np.zeros((900, 900), np.int32)
        for k, (x, y) in enumerate(rng.integers(60, 840, size=(40, 2)), start=1):
            cv2.circle(rgb, (int(x), int(y)), 9, BLUE, -1)
            cv2.circle(nuclei, (int(x), int(y)), 9, k, -1)
            if k % 3 == 0:
                cv2.circle(rgb, (int(x), int(y)), 15, BROWN, 4)  # a stained ring / soma around some nuclei
        for x, y in rng.integers(80, 820, size=(12, 2)):
            cv2.circle(rgb, (int(x), int(y)), int(rng.integers(8, 30)), BROWN, -1)
        if stain == "AT8":
            for _ in range(15):
                p = rng.integers(50, 850, size=4)
                cv2.line(rgb, (int(p[0]), int(p[1])), (int(p[2]), int(p[3])), BROWN, 2)
        cv2.imwrite(str(core_dir / f"{stain}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        for row, y in enumerate((50, 450)):
            for col, x in enumerate((50, 450)):
                tile_id = f"LIP-1_B-2_{stain}_W{row:02d}{col:02d}"
                x0, y0 = max(0, x - 32), max(0, y - 32)
                np.savez_compressed(nuclei_dir / f"{tile_id}.npz", masks=nuclei[y0: y + 400 + 32, x0: x + 400 + 32].astype(np.uint32))
                tiles.append({"tile_id": tile_id, "core_id": "LIP-1_B-2", "tma": 1, "core_label": "B-2", "donor_id": "0001",
                              "sample_region_id": "BRC_0001_FR", "region": "frontal", "disease_group": "AD", "technical_replicate": 1,
                              "stain": stain, "image_path": f"data/cores/LIP-1/B-2/{stain}.png", "pixel_width_um": PS, "pixel_height_um": PS,
                              "selection_role": "whole_core_grid", "selection_order": row * 2 + col, "nested_sample": "whole_core",
                              "x_px": x, "y_px": y, "width_px": 400, "height_px": 400, "preview_tissue_fraction": 0.9})
    analysis = tmp_path / "data" / "analysis"
    pd.DataFrame(tiles).to_csv(analysis / "cohort_whole_core_tiles.csv", index=False)
    pd.DataFrame([{"tma": 1, "stain": s, "threshold_dab_od": 0.05} for s in ("NeuN", "6E10", "AT8")]).to_csv(analysis / "slide_dab_calibration.csv", index=False)
    (tmp_path / "data" / "annotations").mkdir()
    config = {"name": "v2 synthetic", "field_context_px": 32, "models": _models(tmp_path)}
    (tmp_path / "stainid.yaml").write_text(yaml.safe_dump(config))
    return tmp_path


def _proposals(project):
    jobs = run.todo(project, "proposals", ["NeuN", "6E10", "AT8"])
    calibration, nuclei = run.proposal_inputs(project)
    for core, stain, tiles, _ in jobs:
        proposals.propose_core(project, stain, core, tiles, calibration[(tiles[0]["tma"], stain)], nuclei)
    return jobs


def test_stages_end_to_end_with_stable_ids_and_outdated_detection(v2_project, monkeypatch):
    project = load_project(v2_project)
    jobs = _proposals(project)
    assert len(jobs) == 3
    assert run.todo(project, "proposals", ["NeuN", "6E10", "AT8"]) == []  # resumable: nothing left
    first = {s: store.read_table(store.path(project, s, "LIP-1_B-2", "proposals")) for s in ("NeuN", "6E10", "AT8")}
    for stain, frame in first.items():
        assert len(frame) and frame.candidate_id.is_unique and frame.candidate_id.str.startswith(f"{stain}:LIP-1_B-2:").all()
        assert frame.outline.map(lambda t: len(store.points(t)) >= 6).all()
        assert any(c.startswith("m_") for c in frame.columns)
        tiles = pd.read_csv(v2_project / "data/analysis/cohort_whole_core_tiles.csv")
        tiles = tiles[tiles.stain == stain].set_index("tile_id")
        for row in frame.itertuples():  # every candidate's centroid lies in its own tile
            t = tiles.loc[row.tile_id]
            assert t.x_px <= row.x_core < t.x_px + t.width_px and t.y_px <= row.y_core < t.y_px + t.height_px
    assert store.threads_path(project, "LIP-1_B-2").exists()
    assert store.threads(project, "LIP-1_B-2").n_px.sum() > 0
    # stage 4 reads no image
    import stainid.imaging.core_images as images

    monkeypatch.setattr(images, "open_image", lambda *a, **k: (_ for _ in ()).throw(AssertionError("stage 4 read an image")))
    run_calls_inline(project)
    for stain in ("NeuN", "6E10", "AT8"):
        found = store.read_core(project, stain, "LIP-1_B-2")
        assert found.decision.isin(["accepted", "rejected"]).all() and found.reason.ne("").all()
        assert found.model_id.str.contains("@").all()
    monkeypatch.undo()
    assert run.todo(project, "calls", ["NeuN", "6E10", "AT8"]) == []
    # the same inputs give the same ids
    _redo = run.todo(project, "proposals", ["AT8"], redo=True)
    calibration, nuclei = run.proposal_inputs(project)
    for core, stain, tiles, _ in _redo:
        proposals.propose_core(project, stain, core, tiles, calibration[(tiles[0]["tma"], stain)], nuclei)
    again = store.read_table(store.path(project, "AT8", "LIP-1_B-2", "proposals"))
    assert list(again.candidate_id) == list(first["AT8"].candidate_id)
    # a new rule makes only stage 4 outdated; a new proposal setting stage 2 (and so stage 4) too
    config = yaml.safe_load((v2_project / "stainid.yaml").read_text())
    config["v2"] = {"rules": {"AT8": {"min_contrast": 0.5}}}
    (v2_project / "stainid.yaml").write_text(yaml.safe_dump(config))
    project = load_project(v2_project)
    assert run.todo(project, "proposals", ["NeuN", "6E10", "AT8"]) == []
    assert [(c, s, why) for c, s, _, why in run.todo(project, "calls", ["NeuN", "6E10", "AT8"])] == [("LIP-1_B-2", "AT8", "outdated")]
    config["v2"]["proposals"] = {"NeuN": {"levels": [0.6, 1.0]}}
    (v2_project / "stainid.yaml").write_text(yaml.safe_dump(config))
    project = load_project(v2_project)
    assert [(s, why) for _, s, _, why in run.todo(project, "proposals", ["NeuN", "6E10", "AT8"])] == [("NeuN", "outdated")]
    status = run.status(project)
    assert status["proposals"]["NeuN"] == {"done": 0, "outdated": 1, "total": 1}
    assert status["calls"]["NeuN"]["outdated"] == 1


def run_calls_inline(project):
    for core, stain, _, _ in run.todo(project, "calls", ["NeuN", "6E10", "AT8"]):
        classify.classify_core(project, stain, core, classify.load_model(project, stain, jobs=1))


def test_rules_and_classes_from_stored_features(v2_project):
    project = load_project(v2_project)
    model = classify.load_model(project, "6E10", jobs=1)
    names = model.names
    frame = pd.DataFrame({"candidate_id": ["a", "b", "c", "d"], "x_core": [0, 50, 100, 150], "y_core": 0.0,
                          "excluded_fraction": [0, 0, 0, 0.9], "lumen_contact_fraction": [0, 0.5, 0, 0], "normalized_inner_dab": [2.0, 1.0, 1.0, 1.0],
                          **{f"m_{n}": 1.0 for n in names}})
    frame["m_equivalent_diameter_um"] = [20.0, 20.0, 8.0, 20.0]
    model.probability = lambda f: np.array([0.9, 0.9, 0.9, 0.9])
    calls = classify.call(project, "6E10", frame, model, {"excluded_fraction": 0.5, "speck_diameter_um": 10.0, "lumen_contact": 0.25})
    assert list(calls.reason) == ["accepted", "rule:vascular_or_edge", "rule:speck", "rule:excluded"]
    assert list(calls["class"]) == ["plaque", "CAA", "not a plaque", "not a plaque"] and calls.subtype.iloc[0] == "compact"
    tau = classify.load_model(project, "AT8", jobs=1)
    frame = pd.DataFrame({"candidate_id": ["a", "b", "c"], "x_core": [0.0, 10.0, 500.0], "y_core": 0.0, "excluded_fraction": 0.0,
                          "centroid_valid": True, "legacy_class": ["nucleus_ring", "nucleus_ring", "dab_body"], "thread_fraction": [0, 0, 0.9],
                          "m_normalized_contrast": [1.0, 1.0, 1.0], "m_soma_area_um2": [100.0, 100.0, 30.0], "m_soma_dab_p90": [0.3, 0.1, 0.1]})
    tau.probability = lambda f: np.array([0.9, 0.8, 0.9])
    calls = classify.call(project, "AT8", frame, tau, {"min_contrast": 0.3, "min_body_um2": 40.0})
    assert list(calls.reason) == ["accepted", "rule:duplicate", "rule:small"]  # 10 px apart: the less probable is a duplicate
    assert list(calls.subtype) == ["mature tangle", "", ""] and calls["class"].iloc[2] == "thread"


def test_measures_tables_from_the_store(v2_project):
    project = load_project(v2_project)
    _proposals(project)
    run_calls_inline(project)
    from stainid.v2.measures import fraction_at_least, write_measures

    assert fraction_at_least("0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20", 15.0) == pytest.approx(0.25)
    lines = write_measures(project)
    folder = store.root(project) / "measures"
    core = pd.read_csv(folder / "results_core.csv")
    assert len(core) == 1 and {"neun_profile_density_mm2", "v2_plaque_density_mm2", "thread_length_density_mm_per_mm2"} <= set(core.columns)
    wide = pd.read_csv(folder / "cohort_morphology_donor_region_wide.csv")
    assert {"amyloid_positive_area_fraction", "compact_plaque_fraction", "tau_positive_area_fraction", "neun_profile_density_mm2"} <= set(wide.columns)
    assert any("R tables" in line for line in lines)
    accepted = store.read_core(project, "NeuN", "LIP-1_B-2").decision.eq("accepted").sum()
    assert core.neun_positive_profile_count.iloc[0] == accepted


def test_outlines_keep_all_three_sam_outputs(v2_project, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("segment_anything")
    from test_sam_clicks import tiny_sam

    from stainid.masks import clicks
    from stainid.v2 import outlines

    monkeypatch.setattr(clicks, "build_model", tiny_sam)
    project = load_project(v2_project)
    _proposals(project)
    run_calls_inline(project)
    outliner = outlines.BatchOutliner(project.model("sam"), "cpu")
    tiles = [t for t in pd.read_csv(v2_project / "data/analysis/cohort_whole_core_tiles.csv").to_dict("records") if t["stain"] == "6E10"]
    result = outlines.outline_core(project, "6E10", "LIP-1_B-2", tiles, outliner, 0.05)
    table = store.read_table(store.path(project, "6E10", "LIP-1_B-2", "outlines"))
    assert result["new"] == len(table) > 0
    assert table.sam_masks.map(lambda t: len(store.sam_masks(t)) == 3).all()
    assert table.sam_scores.map(lambda t: len(store.sam_scores(t)) == 3).all() and table.sam_stability.map(lambda t: len(store.sam_scores(t)) == 3).all()
    ok = table[table.sam_status == "ok"]
    assert ok.sam_default.between(0, 2).all()
    assert table[table.sam_status != "ok"].sam_default.eq(-1).all()
    assembled = store.read_core(project, "6E10", "LIP-1_B-2")
    assert "sam_masks" in assembled and assembled.decision.ne("").all()
    from stainid.labels import candidate_store  # the annotation pages read the store as written here

    read = candidate_store.read_core(project, "6E10", "LIP-1_B-2")
    outlined = read[read.sam_masks.map(lambda m: bool(m) and sum(bool(p) for p in m) == 3)]
    assert len(outlined)
    row = outlined.iloc[0]
    found, default = candidate_store.masks(row)
    # the three masks small → large, then the stain outline; 6E10 shows its stain outline first (v2.outlines.default_outline)
    assert len(found) == 4 and found[3] == row.outline and len(row.sam_scores) == 3
    assert row.outline_default == "stain" and default == 3
    assert store.outline_rings(store.read_core(project, "6E10", "LIP-1_B-2").iloc[0].to_dict())[0] == \
        store.points(store.read_core(project, "6E10", "LIP-1_B-2").outline.iloc[0])
    as_sam = {**row.to_dict(), "outline_default": "sam"}  # as NeuN: the default mask, the stain outline where SAM fell back
    found, default = candidate_store.masks(as_sam)
    assert default == (int(row.sam_default) if row.sam_default >= 0 else 3)
    assert read.threshold.notna().all()
    again = outlines.outline_core(project, "6E10", "LIP-1_B-2", tiles, outliner, 0.05)  # nothing new to outline
    assert again["new"] == 0
    del torch


def test_viewer_reads_the_store_for_every_stain(v2_project, monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from stainid.api import state, threads
    from stainid.api.app import create_app

    project = load_project(v2_project)
    _proposals(project)
    run_calls_inline(project)
    pd.DataFrame([{"core_id": "LIP-1_B-2", "tma": 1, "core_label": "B-2", "stain": s, "image_path": f"data/cores/LIP-1/B-2/{s}.png",
                   "native_width_px": 900, "native_height_px": 900, "tissue_fraction": 0.9, "tissue_status": "complete"}
                  for s in ("NeuN", "6E10", "AT8")]).to_csv(v2_project / "data/analysis/core_images.csv", index=False)
    monkeypatch.chdir(v2_project)
    monkeypatch.setenv("STAINID_PROJECT", str(v2_project))
    monkeypatch.setattr(threads, "core_threads", lambda *a, **k: (_ for _ in ()).throw(AssertionError("threads traced again")))
    state.reset()
    client = TestClient(create_app(), client=("127.0.0.1", 50000))
    assert client.post("/api/session/admin", json={"password": "admin"}).status_code == 200
    for stain in ("NeuN", "6E10", "AT8"):
        found = client.get(f"/api/cores/LIP-1_B-2/{stain}/objects").json()
        assert found["source"] == "v2" and found["status"] == "done" and found["outlines"]
        frame = store.read_core(project, stain, "LIP-1_B-2")
        assert len(found["rows"]) == len(frame)  # accepted and rejected
        classes = {row[found["columns"].index("model_class")] for row in found["rows"]}
        assert "rejected" in classes or frame.decision.eq("accepted").all()
        shapes = client.get(f"/api/cores/LIP-1_B-2/{stain}/outlines").json()
        assert shapes["status"] == "done" and len(shapes["rows"]) == len(frame)
        unproposed = client.get(f"/api/cores/LIP-1_B-2/{stain}/unproposed").json()
        assert unproposed["status"] == "done" and found["unproposed"] == len(unproposed["rows"])
    saved = client.get("/api/cores/LIP-1_B-2/AT8/threads").json()
    assert saved["status"] == "done" and saved["saved"] and saved["rows"]
    state.reset()


def test_tiles_at_once_and_filter_threads_do_not_change_the_outputs(v2_project):
    """Stage 2 with several tiles in flight and filter threads writes exactly what one tile at a time writes."""
    import gzip

    from stainid.imaging import parallel

    project = load_project(v2_project)
    root = store.root(project)
    calibration, nuclei = run.proposal_inputs(project)
    jobs = run.todo(project, "proposals", ["NeuN", "6E10", "AT8"])

    def outputs(tiles_at_once: int, threads: int) -> dict:
        parallel.threads(threads)
        try:
            for core, stain, tiles, _ in jobs:
                proposals.propose_core(project, stain, core, tiles, calibration[(tiles[0]["tma"], stain)], nuclei, tiles_at_once)
        finally:
            parallel.threads(1)
        files = sorted(p for kind in ("proposals", "tiles", "unproposed", "threads", "masks") for p in (root / kind).rglob("*.*"))
        return {str(p.relative_to(root)): gzip.decompress(p.read_bytes()) if p.suffix == ".gz" else p.read_bytes() for p in files}

    one = outputs(1, 1)
    assert len(one) == 3 * 4 + 1  # proposals, tiles, unproposed and a mask per stain, AT8 threads
    assert outputs(3, 3) == one


def test_tile_processes_write_what_one_process_writes(v2_project, capsys):
    """Stage 2 with the tiles worked on in tile processes (the decoded image and the core-wide masks in shared memory),
    run as the runner does (several core images sharing the pool), writes exactly what one process writes, and leaves
    no shared memory behind."""
    from stainid.imaging import shared_arrays

    project = load_project(v2_project)
    root = store.root(project)
    calibration, nuclei = run.proposal_inputs(project)
    jobs = run.todo(project, "proposals", ["NeuN", "6E10", "AT8"])

    def saved() -> dict:
        files = sorted(p for kind in ("proposals", "tiles", "unproposed", "threads", "masks") for p in (root / kind).rglob("*.*"))
        return {str(p.relative_to(root)): p.read_bytes() for p in files}

    for core, stain, tiles, _ in jobs:
        proposals.propose_core(project, stain, core, tiles, calibration[(tiles[0]["tma"], stain)], nuclei)
    one = saved()
    before = set(shared_arrays.folder().glob(f"{shared_arrays.PREFIX}{os.getpid()}-*"))
    run.run_proposals(project, ["NeuN", "6E10", "AT8"], redo=True, workers=2, processes=2, threads=2)
    assert "2 tile processes (2 threads each)" in capsys.readouterr().out
    assert saved() == one
    assert set(shared_arrays.folder().glob(f"{shared_arrays.PREFIX}{os.getpid()}-*")) == before


def test_tile_processes_fit_the_queue_jobs_budget(monkeypatch):
    """Stage 2 starts as many tile processes as the processors and the memory its queue job was given allow."""
    monkeypatch.setenv("STAINID_JOB_CPUS", "24")
    monkeypatch.setenv("STAINID_JOB_RAM_GB", "34")
    assert run.default_processes(3) == int((34 - 1 - 3 * run.CORE_GB) / run.PROCESS_GB) == 22
    assert run.default_processes(3, threads=2) == 12
    monkeypatch.setenv("STAINID_JOB_RAM_GB", "200")
    assert run.default_processes(3) == 24
