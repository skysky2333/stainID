"""Leaving the white matter out of the analysis (`v2.white_matter.analysis: exclude`): the grey-matter pass
(v2/grey_matter.py) reproduces stage 2's tile measures exactly and redoes them with the white matter counted out; stage 5
drops the objects in it, takes its area out of every denominator, uses the pass's stained areas, keeps the whole-core
numbers as `<column>_whole_core` columns and a `whole_core/` folder, and refuses to mix in an outdated pass; the views
shade the white matter as left out, so what they draw as counted is still what the measures divide by; the NeuN-pale
flag is carried into the tables and nothing is left out for it."""
from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
import pytest
from test_v2_pipeline import PS, _proposals, run_calls_inline, v2_project  # noqa: F401 - the fixture

from stainid import candidates as store
from stainid import exclusions
from stainid.project import load_project
from stainid.v2 import grey_matter, white_matter
from stainid.v2.measures import write_measures

CORE = "LIP-1_B-2"
HALF = 56  # mask cells: from x 448 the 900 px core images are white matter (the right column of tiles starts at 450)


def _white(project, pale: bool = False) -> None:
    """The step's outputs, written by hand: the right half white matter on all three stains (or a NeuN-pale core)."""
    rows = []
    for stain in white_matter.STAINS:
        layer = exclusions.automatic(project, CORE, stain)
        white = np.zeros(layer.codes.shape, np.uint8)
        if not pale:
            white[:, HALF:] = white_matter.WHITE
        path = white_matter.mask_path(project, CORE, stain)
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), white)
        rows.append({"core_id": CORE, "stain": stain, "status": "undetermined" if pale else "white",
                     "reason": white_matter.PALE_REASON + " …" if pale else "", "white_share": 0.0 if pale else 0.5,
                     "white_mm2": 0.0 if pale else 0.03, "tissue_mm2": 0.06, "reference_per_mm2": 20.0, "neurons": 3,
                     "registration": "", "version": white_matter.VERSION})
    pd.DataFrame(rows, columns=white_matter.COLUMNS).to_csv(white_matter.summary_path(project), index=False)


@pytest.fixture()
def detected(v2_project):  # noqa: F811
    project = load_project(v2_project)
    _proposals(project)
    run_calls_inline(project)
    exclusions.forget()
    return project


def test_the_pass_reproduces_stage_two_and_counts_the_white_matter_out(detected):
    project = detected
    _white(project)
    for stain in grey_matter.STAINS:
        white = white_matter.read(project, CORE, stain)
        frame = grey_matter.measure_core(project, stain, CORE, white, grey_matter._tile_rows(project, stain, CORE), check=True)
        stored = store.read_table(store.path(project, stain, CORE, "tiles")).set_index("tile_id").loc[frame.tile_id]
        for column in grey_matter.COLUMNS[stain]:  # without leaving anything out: exactly stage 2's numbers
            assert np.allclose(frame[f"whole_{column}"].astype(float), stored[column].astype(float), equal_nan=True), column
        right = frame.tile_id.str.endswith("01").to_numpy()  # the two tiles of the right column are all white matter
        assert (frame.analysis_area_mm2[right] == 0).all() and (frame.analysis_area_mm2[~right] > 0).all()
        assert (frame.analysis_area_mm2 <= frame.whole_analysis_area_mm2 + 1e-12).all()
    assert len(grey_matter.run(project, log=lambda text: None)) == 2  # 6E10 and AT8 (NeuN needs no pass)
    assert grey_matter.todo(project) == []


def test_stage_five_leaves_the_white_matter_out_and_keeps_the_whole_core_numbers(detected):
    project = detected
    _white(project)
    grey_matter.run(project, log=lambda text: None)
    project.config.setdefault("v2", {})["white_matter"] = {"analysis": "exclude", "min_grey_mm2": 0.001}  # the tiny core is measured
    lines = write_measures(project)
    folder = store.root(project) / "measures"
    assert any("white matter left out" in line for line in lines)
    core = pd.read_csv(folder / "results_core.csv").iloc[0]
    whole = pd.read_csv(folder / "whole_core" / "results_core.csv").iloc[0]
    found = store.read_core(project, "NeuN", CORE)
    accepted = found[found.decision == "accepted"]
    assert core.neun_positive_profile_count == (accepted.x_core < HALF * 8).sum() < whole.neun_positive_profile_count
    assert core.neun_positive_profile_count_whole_core == whole.neun_positive_profile_count
    assert core.neun_tissue_area_mm2 < whole.neun_tissue_area_mm2 == core.neun_tissue_area_mm2_whole_core
    wide = pd.read_csv(folder / "cohort_morphology_donor_region_wide.csv").iloc[0]
    assert "amyloid_positive_area_fraction_whole_core" in wide and "tau_positive_area_fraction_whole_core" in wide
    assert core.white_matter_status == "white" and core.white_matter_share == 0.5 and core.neun_pale == 0
    # the views shade it as left out: what they draw as counted is what the measures divide by
    tiles = pd.read_csv(folder / "tiles_NeuN.csv")
    exclusions.forget()
    shown = exclusions.view(project, CORE, "NeuN")
    counted_mm2 = float((shown.codes == exclusions.COUNTED).sum()) * (8 * PS) ** 2 / 1e6
    assert (shown.codes == exclusions.WHITE_MATTER).any() and counted_mm2 == pytest.approx(tiles.tissue_area_mm2.sum(), rel=0.03)
    from stainid.api import store_view

    kind = next(k for k in store_view.excluded_info(project, CORE, "NeuN")["kinds"] if k["id"] == "white_matter")
    assert not kind["counted"] and "left out" in kind["label"] and kind["fraction"] > 0


def test_an_outdated_pass_stops_stage_five(detected):
    project = detected
    _white(project)
    grey_matter.run(project, log=lambda text: None)
    path = white_matter.mask_path(project, CORE, "AT8")
    cv2.imwrite(str(path), np.zeros_like(cv2.imread(str(path), cv2.IMREAD_UNCHANGED)) + 1)  # the area changed since
    project.config.setdefault("v2", {})["white_matter"] = {"analysis": "exclude"}
    with pytest.raises(RuntimeError, match="grey-matter"):
        write_measures(project)


def test_off_counts_everything_and_a_pale_core_is_flagged_not_left_out(detected):
    project = detected
    write_measures(project)
    before = pd.read_csv(store.root(project) / "measures" / "results_core.csv")
    _white(project, pale=True)
    write_measures(project)
    after = pd.read_csv(store.root(project) / "measures" / "results_core.csv")
    assert not any(c.endswith("_whole_core") for c in after.columns)
    assert after.neun_pale.iloc[0] == 1 and after.white_matter_status.iloc[0] == "undetermined"
    pd.testing.assert_frame_equal(after[before.columns], before)
    region = pd.read_csv(store.root(project) / "measures" / "results_donor_region.csv").iloc[0]
    assert region.neun_pale == 1 and region.neun_pale_cores == 1


def test_a_sliver_of_grey_matter_is_not_measured(detected):
    """A core image left with less grey matter than v2.white_matter.min_grey_mm2 (here the whole tiny core, 0.03 mm² of
    grey matter against the default 1 mm²) is NA in the grey-matter tables, flagged, its whole-core numbers kept; its
    donor-region, with no other core, is NA too (not 0)."""
    project = detected
    _white(project)
    grey_matter.run(project, log=lambda text: None)
    project.config.setdefault("v2", {})["white_matter"] = {"analysis": "exclude"}
    lines = write_measures(project)
    folder = store.root(project) / "measures"
    assert any("not measured (NA)" in line for line in lines)
    core = pd.read_csv(folder / "results_core.csv")
    whole = pd.read_csv(folder / "whole_core" / "results_core.csv")
    assert len(core) == len(whole) == 1 and core.grey_too_small.iloc[0] == 1
    assert np.isnan(core.neun_positive_profile_count.iloc[0]) and np.isnan(core.neun_tissue_area_mm2.iloc[0])
    assert core.neun_positive_profile_count_whole_core.iloc[0] == whole.neun_positive_profile_count.iloc[0] > 0
    assert core.disease_group.iloc[0] == whole.disease_group.iloc[0]
    region = pd.read_csv(folder / "results_donor_region.csv")
    assert len(region) == 1 and region.grey_too_small.iloc[0] == 1 and region.grey_too_small_cores.iloc[0] == 1
    assert np.isnan(region.neun_profile_density_mm2.iloc[0]) and region.neun_profile_density_mm2_whole_core.iloc[0] > 0
    features = pd.read_csv(folder / "cohort_morphology_core_features.csv")
    assert set(features.grey_too_small) == {1} and features.tissue_area_mm2.isna().all()


def test_the_steps_run_their_outputs_track_what_they_are_made_from_and_people_change_nothing(detected, monkeypatch):
    """`white_matter.run` / `grey_matter.run` as the Run page runs them: done once, nothing to redo; a repeat leaves the
    areas and the summary alone, so the costly pass stays current; people's labels change no input of either."""
    project = detected
    assert white_matter.progress(project)["done"] == 0 and white_matter.progress(project)["total"] == 1
    white_matter.run(project, log=lambda text: None)
    assert white_matter.todo(project) == [] and white_matter.progress(project)["done"] == 1
    mask, table = white_matter.mask_path(project, CORE, "NeuN"), white_matter.summary_path(project)
    stamps = mask.stat().st_mtime_ns, table.stat().st_mtime_ns
    digest = white_matter.digest(project, CORE, "AT8")
    white_matter.run(project, redo=True, log=lambda text: None)
    assert (mask.stat().st_mtime_ns, table.stat().st_mtime_ns) == stamps and white_matter.digest(project, CORE, "AT8") == digest

    _white(project)  # a core with white matter, for the pass
    assert [s for _, s in grey_matter.todo(project)] == ["6E10", "AT8"]
    assert grey_matter.progress(project) == {"done": 0, "outdated": 0, "total": 2, "written": 0.0}
    grey_matter.run(project, log=lambda text: None)
    found = grey_matter.progress(project)
    assert (found["done"], found["outdated"], found["total"]) == (2, 0, 2) and found["written"] > 0

    inputs = white_matter._inputs(project, CORE, ["NeuN", "6E10", "AT8"])
    from stainid.labels import matter, store as labels

    stamp = {"source": "a", "session": "s", "tile_id": CORE, "x": None, "y": None}
    labels.append(project.output("reviews"), matter.records(CORE, "NeuN", [[[0, 0], [99, 0], [99, 99]]], [], stamp, None, False, None))
    assert white_matter._inputs(project, CORE, ["NeuN", "6E10", "AT8"]) == inputs
    assert grey_matter.todo(project) == []


def test_a_failed_core_does_not_stop_the_step_and_the_run_says_so(detected, monkeypatch):
    project = detected
    monkeypatch.setattr(white_matter, "run_core", lambda *a, **k: (_ for _ in ()).throw(ValueError("unreadable")))
    said = []
    with pytest.raises(RuntimeError, match="1 of 1 cores failed"):
        white_matter.run(project, log=said.append)
    assert any("failed: ValueError" in line for line in said) and white_matter.summary_path(project).exists()


def test_a_paused_job_stops_between_two_cores(detected, monkeypatch, tmp_path):
    from stainid.jobs import units

    control = tmp_path / "control"
    control.write_text("pause")
    monkeypatch.setenv(units.CONTROL_ENV, str(control))
    units.forget()
    with pytest.raises(units.Paused):
        white_matter.run(detected, log=lambda text: None)
    assert not white_matter.manifest_path(detected, CORE).exists()


def test_a_project_without_neun_images_has_nothing_to_do(detected):
    project = detected
    stored = exclusions.masks_folder(project, "NeuN")
    stored.rename(stored.with_name("NeuN_gone"))
    found = white_matter.run(project, log=lambda text: None)
    assert found.empty and not white_matter.root(project).exists()  # nothing written: no flags, no empty columns in the tables
    assert white_matter.progress(project)["total"] == 0 and grey_matter.todo(project) == [] and grey_matter.run(project, log=lambda text: None) == []


def _measured(project, folder: str, **flag) -> dict:
    project.config.setdefault("v2", {})["boundary_flag"] = flag
    target = store.root(project) / folder
    write_measures(project, target)
    return {name: pd.read_csv(target / name, low_memory=False) for name in ("results_core.csv", "results_donor_region.csv", "objects_AT8.csv")}


def test_the_boundary_flag_only_adds_columns(detected):
    """Whatever the flag decides, every other column of every table stays the same."""
    project = detected
    none, every = _measured(project, "none", min_nontissue_um2=1e12), _measured(project, "every", min_nontissue_um2=0.0)
    new = {"tau_neuron_boundary_fraction", "tau_neuron_density_no_boundary_mm2", "tissue_boundary", "nearby_nontissue_um2"}
    for name, table in none.items():
        old = [c for c in table.columns if c not in new]
        pd.testing.assert_frame_equal(table[old], every[name][old])
    accepted = store.read_core(project, "AT8", CORE)
    accepted = accepted[accepted.decision == "accepted"]
    assert len(accepted) and set(none["objects_AT8.csv"].candidate_id) == set(accepted.candidate_id)
    assert (none["objects_AT8.csv"].tissue_boundary == 0).all() and (every["objects_AT8.csv"].tissue_boundary == 1).all()
    plain, flagged = none["results_core.csv"].iloc[0], every["results_core.csv"].iloc[0]
    assert plain.tau_neuron_density_no_boundary_mm2 == pytest.approx(plain.tau_neuron_density_mm2)  # nothing flagged
    assert flagged.tau_neuron_density_no_boundary_mm2 == 0 and flagged.tau_neuron_density_mm2 == plain.tau_neuron_density_mm2


def test_the_boundary_columns_are_grey_matter_and_whole_core_and_say_why_they_are_missing(detected):
    project = detected
    _white(project)
    grey_matter.run(project, log=lambda text: None)
    project.config.setdefault("v2", {})["white_matter"] = {"analysis": "exclude", "min_grey_mm2": 0.001}
    tables = _measured(project, "grey", min_nontissue_um2=0.0)
    core = tables["results_core.csv"].iloc[0]
    assert core.tau_neuron_density_no_boundary_mm2 == 0 and "tau_neuron_density_no_boundary_mm2_whole_core" in core
    assert core.tau_neuron_density_no_boundary_mm2_whole_core == 0
    folder = store.root(project) / "grey"
    grey, whole = pd.read_csv(folder / "objects_AT8.csv"), pd.read_csv(folder / "whole_core" / "objects_AT8.csv")
    assert set(grey.candidate_id) <= set(whole.candidate_id)
    missing = pd.read_csv(folder / "results_missing_core.csv")
    if core.tau_neuron_count < 5:
        assert missing[missing.measure == "tau_neuron_boundary_fraction"].reason.tolist() == ["too_few_objects"]
