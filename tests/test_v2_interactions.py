"""The spatial-interaction features of stage 5 (`stainid.v2.interactions`) and the reliability table (`stainid.v2.reliability`)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stainid.v2 import interactions as ix
from stainid.v2 import reliability

GRID_UM = ix.SCALE * 0.2738


def _square(x: float, y: float, half: float) -> np.ndarray:
    return np.array([[x - half, y - half], [x + half, y - half], [x + half, y + half], [x - half, y + half]], np.float32)


def _objects(centres: np.ndarray, half: float, subtype: str) -> pd.DataFrame:
    return pd.DataFrame({"x_core": centres[:, 0], "y_core": centres[:, 1], "subtype": subtype,
                         "ring": [_square(x, y, half) for x, y in centres]})


def test_rings_see_neurons_missing_next_to_plaques_and_the_null_keeps_random_at_one():
    rng = np.random.default_rng(0)
    shape = (400, 400)  # mask cells: 876 µm square
    counted = np.ones(shape, bool)
    plaques = _objects(rng.uniform(400, 2800, size=(25, 2)), 60, "compact")  # 120 px across: 33 µm
    nuclei = rng.uniform(0, 3200, size=(20000, 2))
    distance = np.min(np.hypot(*(nuclei[:, None, :] - plaques[["x_core", "y_core"]].to_numpy()[None]).transpose(2, 0, 1)), axis=1) * 0.2738
    cell_yx = (nuclei[:, 1] // ix.SCALE).astype(int), (nuclei[:, 0] // ix.SCALE).astype(int)
    groups = {"compact": plaques, "diffuse": plaques.iloc[0:0]}
    depleted = np.where(distance < 45, 0.1, 0.5)  # neuron-like probability: few neurons within ~30 µm of a plaque edge
    flat = np.full(len(nuclei), 0.5)
    for p, low in ((depleted, True), (flat, False)):
        found = ix.ring_stats("pl", shape, counted, groups, pd.DataFrame(columns=plaques.columns), cell_yx, p, GRID_UM, ix.PLAQUE_RINGS,
                              "LIP-1_A-1", nulls=9)
        ratio = ix.share_ratio(found, "pl", "all", "ring10_30")
        assert (ratio < 0.5) if low else ratio == pytest.approx(1.0, abs=0.05)
        assert found["pl_compact_ring10_30_cells_o"] == found["pl_all_ring10_30_cells_o"]  # every ring nucleus goes to a compact plaque


def test_ring_nulls_are_seeded_by_the_core():
    rng = np.random.default_rng(1)
    shape = (200, 200)
    counted = np.ones(shape, bool)
    plaques = _objects(rng.uniform(200, 1400, size=(10, 2)), 40, "diffuse")
    nuclei = rng.uniform(0, 1600, size=(3000, 2))
    cell_yx = (nuclei[:, 1] // ix.SCALE).astype(int), (nuclei[:, 0] // ix.SCALE).astype(int)
    args = (shape, counted, {"compact": plaques.iloc[0:0], "diffuse": plaques}, pd.DataFrame(columns=plaques.columns), cell_yx,
            rng.random(3000), GRID_UM, ix.PLAQUE_RINGS)
    assert ix.ring_stats("pl", *args, "LIP-1_A-1", nulls=5) == ix.ring_stats("pl", *args, "LIP-1_A-1", nulls=5)
    assert ix.ring_stats("pl", *args, "LIP-1_A-1", nulls=5) != ix.ring_stats("pl", *args, "LIP-2_A-1", nulls=5)


def test_microcolumn_band_counts_columns_along_the_axis():
    rng = np.random.default_rng(2)
    columns = np.array([[x + rng.normal(0, 2), y] for x in np.arange(0, 2000, 80) for y in rng.uniform(0, 2000, 30)])  # µm
    scattered = rng.uniform(0, 2000, size=(len(columns), 2))
    axis = np.tile([0.0, 1.0], (len(columns), 1))
    radial, lateral, _ = ix.column_counts(columns, axis)
    assert radial > 3 * lateral
    radial, lateral, _ = ix.column_counts(scattered, axis)
    assert radial == pytest.approx(lateral, rel=0.25)
    aligned, pairs = ix.alignment_counts(columns, axis)
    random_aligned, random_pairs = ix.alignment_counts(scattered, axis)
    assert random_aligned / random_pairs == pytest.approx(1 / 3, abs=0.03) and aligned / pairs > 1.03 * random_aligned / random_pairs


def test_maturity_mixing_is_one_under_random_labels():
    rng = np.random.default_rng(3)
    xy = rng.uniform(0, 3000, size=(400, 2))
    mature = rng.random(400) < 0.6
    found = ix.mix_stats(xy, mature)
    assert found["mix_100_mixed"] / found["mix_100_expected"] == pytest.approx(1.0, abs=0.2)
    segregated = xy[:, 0] > 1500  # mature tangles on one side only
    found = ix.mix_stats(xy, segregated)
    assert found["mix_100_mixed"] / found["mix_100_expected"] < 0.2


def test_colocation_quotient():
    rng = np.random.default_rng(4)
    xy = rng.uniform(0, 1000, size=(2000, 2))
    neuron = rng.random(2000) < 0.3
    s = ix.clq_stats(xy, neuron)
    assert ix._clq(s, "other") == pytest.approx(1.0, abs=0.15)
    s = ix.clq_stats(xy, xy[:, 0] < 300)  # neurons all on one side: an other nucleus seldom has a neuron as its neighbour
    assert ix._clq(s, "other") < 0.2


def test_coupling_sees_shared_fields_and_the_shift_null():
    rng = np.random.default_rng(5)
    ixs, iys = np.meshgrid(np.arange(30), np.arange(30))
    field = np.sin(ixs / 2.0) + np.cos(iys / 3.0) + rng.normal(0, 0.3, ixs.shape)
    bins = pd.DataFrame({"ix": ixs.ravel(), "iy": iys.ravel(), "tissue_mm2": 0.01, "plaque_um2": np.exp(field.ravel()) * 100,
                         "compact_um2": np.exp(field.ravel()) * 50, "diffuse_um2": rng.random(900), "thread_mm": np.exp(field.ravel()) * 0.1,
                         "tau": rng.poisson(2, 900).astype(float)})
    s = ix.coupling_stats(bins, "LIP-1_A-1")
    assert ix._mean(s, "cp_rho") > 0.8 and ix._mean(s, "cp_excess") > 0.3
    assert abs(ix._mean(s, "cp_diffuse")) < 0.15


def test_niches_and_their_contacts():
    bins = pd.DataFrame({"ix": np.repeat(np.arange(10), 10), "iy": np.tile(np.arange(10), 10)})
    damaged = bins.ix < 5
    bins["a"], bins["t"], bins["n"] = np.where(damaged, 0.9, 0.1), np.where(damaged, 0.9, 0.1), 0.5
    labels = ix.niche_labels(bins)
    s = ix.niche_stats("niche200", bins, labels)
    out = ix._niche_columns(s, "niche200")
    assert out["niche200_amyloid_tau_fraction"] == 0.5 and out["niche200_quiet_fraction"] == 0.5
    assert out["niche200_damage_quiet_contact_ratio"] < 0.3  # damage in one block: few damaged-quiet contacts
    assert out["niche200_amyloid_tau_cohesion"] > 1.5


def test_nuclei_from_stage_one_have_regionprops_shapes(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from skimage.measure import regionprops_table

    from stainid.v2 import measures

    labels = np.zeros((300, 300), np.uint32)
    yy, xx = np.mgrid[:300, :300]
    labels[((xx - 100) / 12) ** 2 + ((yy - 120) / 7) ** 2 <= 1] = 1
    labels[((xx - 200) / 14) ** 2 + ((yy - 200) / 14) ** 2 <= 1] = 2
    labels[((xx - 20) / 9) ** 2 + ((yy - 20) / 9) ** 2 <= 1] = 3  # centred in the context margin: another tile's
    np.savez_compressed(tmp_path / "T1.npz", masks=labels)
    tiles = pd.DataFrame([{"tile_id": "T1", "x_px": 1032, "y_px": 2032, "width_px": 236, "height_px": 236}])
    monkeypatch.setattr(measures, "_tiles", lambda project, stain, core_id: tiles)
    project = SimpleNamespace(output=lambda name: tmp_path, context_px=32, pixel_size_um=0.5)
    found = ix._npz_nuclei(project, "6E10", "LIP-1_A-1").sort_values("x_core").reset_index(drop=True)
    props = pd.DataFrame(regionprops_table(labels.astype(int), properties=("label", "centroid", "area", "eccentricity", "major_axis_length",
                                                                           "minor_axis_length")))
    props = props[props.label.isin([1, 2])].sort_values("centroid-1").reset_index(drop=True)
    assert len(found) == 2
    assert np.allclose(found.x_core, props["centroid-1"] + 1000) and np.allclose(found.y_core, props["centroid-0"] + 2000)
    assert np.allclose(found.area_um2, props.area * 0.25)
    assert np.allclose(found.major_um, props.major_axis_length * 0.5) and np.allclose(found.minor_um, props.minor_axis_length * 0.5)
    assert np.allclose(found.eccentricity, props.eccentricity)


def test_every_column_has_its_needs_and_the_reasons_are_known():
    from stainid.v2 import missing

    for column in ix.COLUMNS:
        for key, minimum, reason in ix.needs(column):
            assert reason in missing.REASONS and minimum >= 0 and key
    empty = ix.columns({}, 40.0)
    assert set(empty) == set(ix.COLUMNS) and all(np.isnan(v) for v in empty.values())


def test_replicate_icc_and_unique_variance():
    rng = np.random.default_rng(6)
    region = np.repeat(np.arange(40), 2)
    truth = rng.normal(size=40)
    assert reliability.icc(truth[region], region)[0] > 0.98
    assert reliability.icc(rng.normal(size=80), region)[0] < 0.3
    frame = pd.DataFrame({c: rng.random(60) for c in reliability.PRIMARY})
    frame["copy"] = frame.v2_plaque_density_mm2 * 2
    frame["new"] = rng.random(60)
    assert reliability.unique_variance(frame, "copy")[0] < 0.01
    assert reliability.unique_variance(frame, "new")[0] > 0.7


def test_every_interaction_column_has_a_plain_label():
    from stainid.analysis import dictionary

    for column in ix.COLUMNS:
        info = dictionary.measure(column)
        assert info["topic"] == "spatial" and info["label"] and info["description"], column


def test_attach_keeps_every_existing_cell_as_written(tmp_path):
    from stainid.v2 import interactions as ix
    (tmp_path / "results_core.csv").write_text("core_id,thread_mm,tangles\nA-1,1.66552540,3\nA-2,,0\n")
    (tmp_path / "results_counts_core.csv").write_text("core_id,thread_mm\nA-1,0.06105740\nA-2,0\n")
    core = pd.DataFrame({"core_id": ["A-1"], **{c: [np.nan] for c in ix.COLUMNS}, "ix_rings_plaques": [12.0]})
    core[ix.COLUMNS[0]] = 0.5
    ix.attach_tables(tmp_path, core, pd.DataFrame(columns=["sample_region_id"]))
    results = pd.read_csv(tmp_path / "results_core.csv", dtype=str, keep_default_na=False)
    assert results["thread_mm"].tolist() == ["1.66552540", ""] and results["tangles"].tolist() == ["3", "0"]
    assert results[ix.COLUMNS[0]].tolist() == ["0.5", ""] and set(ix.COLUMNS) <= set(results.columns)
    counts = pd.read_csv(tmp_path / "results_counts_core.csv", dtype=str, keep_default_na=False)
    assert counts["thread_mm"].tolist() == ["0.06105740", "0"] and counts["ix_rings_plaques"].tolist() == ["12.0", "0.0"]


def test_a_coupling_without_plaque_squares_is_too_few_objects_not_unmeasured():
    from stainid.v2 import missing

    ixs, iys = np.meshgrid(np.arange(8), np.arange(8))
    bins = pd.DataFrame({"ix": ixs.ravel(), "iy": iys.ravel(), "tissue_mm2": 0.01, "plaque_um2": 0.0, "compact_um2": 0.0,
                         "diffuse_um2": 0.0, "thread_mm": np.linspace(0.1, 1, 64), "tau": 1.0})
    s = ix.coupling_stats(bins, "LIP-1_A-1") | {"registered": 1.0}
    assert s["cp_bins"] == 64 and s["cp_plaque_bins"] == 0 and s["cp_thread_bins"] == 64
    assert np.isnan(ix.columns(s, 40.0)["plaque_thread_coupling_rho"])
    counts = {"tiles_NeuN": 1, "tiles_6E10": 1, "tiles_AT8": 1, **{f"ix_{k}": v for k, v in ix._counts(s).items()}}
    for column in ("plaque_thread_coupling_rho", "plaque_thread_coupling_excess", "plaque_tangle_coupling_excess",
                   "plaque_compact_thread_coupling", "plaque_thread_correlation_length_um"):
        assert missing._reason(column, "c", {"c": counts}, None) == ("too_few_objects", 0)
    bins["plaque_um2"] = np.r_[np.linspace(10, 100, 10), np.zeros(54)]  # plaques in 10 squares: measured
    s = ix.coupling_stats(bins, "LIP-1_A-1") | {"registered": 1.0}
    assert s["cp_plaque_bins"] == 10 and np.isfinite(ix.columns(s, 40.0)["plaque_thread_coupling_rho"])


def test_register_keeps_every_pair_the_table_has_no_row_for(synthetic_project, monkeypatch):
    from types import SimpleNamespace

    from stainid.project import load_project
    from stainid.registration import cores as registration
    from stainid.registration import rigid

    calls = []

    def fake(reference, moving):
        calls.append(1)
        return SimpleNamespace(matrix=np.array([[1.0, 0, 5], [0, 1.0, 0]]), status="review", final_dice=0.8)

    monkeypatch.setattr(rigid, "register_previews", fake)
    project = load_project(synthetic_project)
    rows = registration.register_missing(project)
    assert [(r["core_id"], r["target"], r["how"], r["status"]) for r in rows] == [
        ("LIP-1_B-2", "6E10", "computed", "review"), ("LIP-1_B-2", "AT8", "computed", "review")]
    for target in ("6E10", "AT8"):  # where carry_project and stage 5 read it
        assert registration.kept_path(project.root, "LIP-1_B-2", "NeuN", target).exists()
        assert ix.to_stain(project, "LIP-1_B-2", target) is not None
    again = registration.register_missing(project)
    assert [r["how"] for r in again] == ["kept", "kept"] and len(calls) == 2
    assert registration.carry_project(project, "LIP-1_B-2", "NeuN", "6E10").status == "review" and len(calls) == 2


def test_grid_distances_do_not_depend_on_memory_alignment():
    import cv2
    from scipy import ndimage

    rng = np.random.default_rng(0)
    canvas = np.full((300, 400), 255, np.uint8)
    for x, y in rng.integers(0, 400, (120, 2)):
        cv2.circle(canvas, (int(x), int(y) % 300), int(rng.integers(2, 12)), 0, -1)
    reference = ix._edt(canvas)
    for offset in range(1, 16):
        shifted = np.empty(canvas.size + offset, np.uint8)[offset:].reshape(canvas.shape)
        shifted[:] = canvas
        assert np.array_equal(ix._edt(shifted), reference)
    assert np.allclose(reference, ndimage.distance_transform_edt(canvas > 0), atol=1e-5)
