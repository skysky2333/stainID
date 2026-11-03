"""Stage 2's biology measurements (`stainid.v2.biology`) and their stage 5 readouts (`stainid.v2.biology_measures`)."""
from __future__ import annotations

from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd
import pytest
from test_v2_pipeline import _proposals, run_calls_inline, v2_project  # noqa: E402, F401 - the test project of the pipeline tests

from stainid.v2 import biology, biology_measures
from stainid.v2.field import Field
from stainid.v2.geometry import Region

PS = 0.5


def _soma_field(nucleus_dab: float):
    """A 100 x 100 field with a soma (a 30 x 30 square) and a nucleus in its upper-left part."""
    raw = np.zeros((100, 100), np.uint32)
    raw[40:50, 40:52] = 7  # 120 px = 30 µm²
    hematoxylin = np.full((100, 100), 0.02, np.float32)
    dab = np.full((100, 100), 0.01, np.float32)
    dab[35:65, 35:65] = 0.3
    hematoxylin[40:50, 40:52] = 0.2
    dab[40:50, 40:52] = nucleus_dab
    return SimpleNamespace(raw_nuclei=raw, nuclei=raw, hematoxylin=hematoxylin, dab=dab, pixel_size_um=PS, memo={})


def test_a_visible_nucleus_in_a_soma_and_its_offset():
    soma = Region(35, 35, np.ones((30, 30), bool), "test")
    seen = biology.soma_nucleus(_soma_field(0.05), soma)
    assert seen["soma_nucleus_visible"] == 1 and seen["soma_nucleus_area_um2"] == pytest.approx(30)
    assert seen["soma_nucleus_offset"] == pytest.approx(np.hypot(int(45.5) - 49.5, int(44.5) - 49.5) / np.sqrt(900 / np.pi), abs=0.05)
    hidden = biology.soma_nucleus(_soma_field(0.5), soma)  # the stain over the nucleus: not visible, still found
    assert hidden["soma_nucleus_visible"] == 0 and np.isfinite(hidden["soma_nucleus_offset"])
    elsewhere = biology.soma_nucleus(_soma_field(0.05), Region(70, 70, np.ones((20, 20), bool), "test"))
    assert elsewhere["soma_nucleus_visible"] == 0 and np.isnan(elsewhere["soma_nucleus_offset"])
    nothing = _soma_field(0.05)
    nothing.raw_nuclei = None
    assert all(np.isnan(v) for v in biology.soma_nucleus(nothing, soma).values())


def test_profile_numbers_tell_a_haloed_core_a_bare_core_and_a_diffuse_plaque():
    rings = lambda values: np.asarray(values, float) * 0.1  # noqa: E731  (OD; the slide threshold is 0.1)
    haloed = biology.profile_numbers(rings([4, 4, 1.8, 1.6, 1.5, 1.4, 1.2, 0.6, 0.4, 0.3] + [0.3] * 13), 0.1)
    bare = biology.profile_numbers(rings([4, 4, 0.5, 0.4, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3] + [0.3] * 13), 0.1)
    diffuse = biology.profile_numbers(rings([1.5, 1.5, 1.4, 1.4, 1.3, 1.2, 1.1, 0.6, 0.4, 0.3] + [0.3] * 13), 0.1)
    assert haloed["profile_core_radius_um"] == 4 and haloed["profile_halo_width_um"] == pytest.approx(10)
    assert bare["profile_halo_width_um"] == 0 and bare["profile_core_to_halo"] > haloed["profile_core_to_halo"] > diffuse["profile_core_to_halo"]
    assert diffuse["profile_core_contrast"] < 0.3 < haloed["profile_core_contrast"]
    assert biology.profile_numbers(rings([0.9] * 23), 0.1)["profile_core_radius_um"] == 0  # no core above the threshold


def _plaque_field():
    h = w = 240
    dab = np.full((h, w), 0.01, np.float32)
    r = np.hypot(*(np.mgrid[:h, :w] - 120)) * PS
    dab[r < 6] = 0.5
    dab[(r >= 6) & (r < 12)] = 0.15  # a halo above the threshold out to 12 µm
    return SimpleNamespace(dab=dab, valid=np.ones((h, w), bool), shape=(h, w), pixel_size_um=PS, threshold=0.1, memo={})


def test_radial_profile_of_a_deposit():
    plaque = Region(108, 108, np.hypot(*np.mgrid[-12:13, -12:13]) * PS < 6, "test")
    out = biology.radial_profile(_plaque_field(), plaque)
    assert len(out["radial_dab_um"].split()) == int(biology.PROFILE_UM / biology.RING_UM)
    assert out["profile_core_dab"] == pytest.approx(5.0, rel=0.05) and out["profile_core_radius_um"] == pytest.approx(6, abs=2)
    assert out["profile_extent_radius_um"] == pytest.approx(12, abs=2) and out["profile_background_dab"] == pytest.approx(0.1, rel=0.05)
    speck = Region(118, 118, np.ones((4, 4), bool), "test")  # under PROFILE_MIN_UM: no profile
    assert np.isnan(biology.radial_profile(_plaque_field(), speck)["profile_core_dab"])


def _vessel_field() -> Field:
    """Grey tissue with an elongated lumen lined by two elongated nuclei, and a round hole with none."""
    rgb = np.full((400, 400, 3), 170, np.uint8)
    raw = np.zeros((400, 400), np.int32)
    cv2.ellipse(rgb, (150, 150), (40, 10), 0, 0, 360, (250, 250, 250), -1)  # 40 x 10 µm lumen
    for label, y in ((1, 136), (2, 164)):
        cv2.ellipse(rgb, (150, y), (16, 3), 0, 0, 360, (70, 70, 160), -1)
        cv2.ellipse(raw, (150, y), (16, 3), 0, 0, 360, label, -1)
    cv2.circle(rgb, (300, 300), 10, (250, 250, 250), -1)  # 10 µm across: under the round-vessel size
    tile = {"tile_id": "T", "stain": "NeuN", "pixel_width_um": PS, "pixel_height_um": PS, "context_px": 0}
    return Field(tile, rgb, (slice(0, 400), slice(0, 400)), (1000, 2000), 0.1, PS, np.zeros((400, 400), bool), raw.astype(np.uint32))


def test_vessels_need_a_lining_of_elongated_nuclei():
    found = biology.vessels(_vessel_field(), "NeuN")
    by = {round(r["x_core"]) // 50: r for r in found}
    assert by[1150 // 50]["vessel"] and by[1150 // 50]["lining_along_wall"] == 2
    assert not by[1300 // 50]["vessel"] and by[1300 // 50]["lining_nuclei"] == 0
    mask = biology.vessel_mask(pd.DataFrame(found), (300, 300), 8, PS)
    assert (mask & biology.VESSEL_BITS["vessel"]).any() and (mask & biology.VESSEL_BITS["lumen"]).any() and (mask & biology.VESSEL_BITS["wall"]).any()


def test_readouts_need_their_minimum_and_a_measured_stage_2():
    tally = {"stain": "AT8", "measured": True, "vessels_known": True, "area_mm2": 2.0, "vessels": 4, "vessel_area_mm2": 0.002, "caa_vessels": 0,
             "nucleus_visible": np.array([1, 0, 0, 1, np.nan, 0]), "nucleus_offset": np.array([0.2, 0.5, np.nan, 0.3, np.nan, 0.4]),
             "subtype": np.array(["pretangle", "mature tangle", "mature tangle", "pretangle", "mature tangle", "mature tangle"]),
             "near_area_mm2": 0.1, "far_area_mm2": 1.9, "near_neurons": 2, "far_neurons": 4, "near_thread_mm": 0.1, "far_thread_mm": 0.4}
    out = biology_measures.columns("AT8", [tally])
    assert out["tau_neuron_nucleus_visible_fraction"] == pytest.approx(0.4) and np.isnan(out["tau_neuron_nucleus_visible_fraction_mature"])
    assert out["at8_vessel_density_mm2"] == 2 and out["tau_neuron_vessel_enrichment"] == pytest.approx((2 / 0.1) / (4 / 1.9))
    assert np.isnan(out["tau_neuron_nucleus_offset_median"])  # 4 neurons with a nucleus: under the minimum
    old = biology_measures.columns("AT8", [tally | {"measured": False, "vessels_known": False}])
    assert all(np.isnan(v) for v in old.values())
    counts = biology_measures.counts({"AT8": [tally]})
    assert counts["nucleus_known"] == 5 and counts["bio_measured_AT8"] == 1
    from stainid.v2 import missing

    assert missing._reason("tau_neuron_nucleus_visible_fraction_mature", "c", {"c": {"tiles_AT8": 1, **counts}}, None) == ("too_few_objects", 3)
    assert missing._reason("tau_neuron_nucleus_visible_fraction", "c", {"c": {"tiles_AT8": 1, "bio_measured_AT8": 0}}, None) == ("not_measured", 0)
    neun = {"stain": "NeuN", "measured": True, "vessels_known": False, "area_mm2": 1.0, "neuron_nucleus_area": np.full(25, 60.0),
            "large": 40, "large_in_neuron": 30, "large_weak": 4, "large_unstained": 6}
    out = biology_measures.columns("NeuN", [neun])
    assert out["neun_large_nucleus_density_mm2"] == 40 and out["neun_large_nucleus_unstained_fraction"] == pytest.approx(0.15)
    assert out["neun_neuron_nucleus_area_median_um2"] == 60 and np.isnan(out["neun_vessel_density_mm2"])


def test_stage_2_stores_the_biology_and_stage_5_reads_it(v2_project):  # noqa: F811
    """End to end on the test project: the new candidate columns and the vessel table and picture exist after stage 2,
    and every new results column is there after stage 5 (empty cells each with a reason)."""
    from stainid import candidates as store
    from stainid.project import load_project
    from stainid.v2.measures import write_measures

    project = load_project(v2_project)
    jobs = _proposals(project)
    for core, stain, _, _ in jobs:
        frame = store.read_table(store.path(project, stain, core, "proposals"))
        wanted = biology.PROFILE_COLUMNS if stain == "6E10" else biology.NUCLEUS_COLUMNS
        assert set(wanted) <= set(frame.columns) or frame.empty
        table = biology.read_vessels(project, stain, core)
        assert table is not None and list(table.columns) == ["vessel_id", *biology.VESSEL_COLUMNS]
        assert (store.root(project) / "vessels" / stain / f"{core}.png").exists()
    run_calls_inline(project)
    write_measures(project)
    folder = store.root(project) / "measures"
    core = pd.read_csv(folder / "results_core.csv")
    for stain in ("NeuN", "6E10", "AT8"):
        assert set(biology_measures.names_of(stain)) <= set(core.columns)
    gone = pd.read_csv(folder / "results_missing_core.csv")
    empty = {(c, m) for m in biology_measures.names_of("AT8") for c in core.loc[core[m].isna(), "core_id"]}
    assert empty <= set(zip(gone.core_id, gone.measure))


def test_a_large_round_space_and_a_perivascular_space_are_vessels():
    rgb = np.full((400, 400, 3), 170, np.uint8)
    cv2.circle(rgb, (120, 120), 30, (250, 250, 250), -1)  # 30 µm across, smooth
    cv2.circle(rgb, (290, 290), 40, (250, 250, 250), -1)
    cv2.circle(rgb, (290, 290), 20, (150, 150, 170), -1)  # the vessel inside its space
    tile = {"tile_id": "T", "stain": "AT8", "pixel_width_um": PS, "pixel_height_um": PS, "context_px": 0}
    field = Field(tile, rgb, (slice(0, 400), slice(0, 400)), (0, 0), 0.1, PS, np.zeros((400, 400), bool), np.zeros((400, 400), np.uint32))
    found = {round(r["x_core"]) // 100: r for r in biology.vessels(field, "AT8")}
    assert found[1]["vessel"] and found[2]["vessel"]
