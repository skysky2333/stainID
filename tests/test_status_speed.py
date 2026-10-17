"""The v2 status counts (Home, Set up, the Run page) are the same as the per-core reading they replaced, and are made once
per stain (parameters, models) and per manifest folder, not once per core."""
from __future__ import annotations

import yaml
from test_v2_pipeline import _proposals, run_calls_inline, v2_project  # noqa: F401 - the synthetic three-stain project

from stainid import cache
from stainid.project import load_project
from stainid.v2 import classify, manifest, outlines, params, proposals, run
from stainid.v2.params import STAINS


def _status_before(project) -> dict:
    """`run.status` as it was: every core's expected manifests from the files, each with its own parameter merge."""
    out = {stage: {} for stage in ("proposals", "calls", "outlines", "candidates")}
    if not project.analysis_tiles().exists():
        return {stage: {stain: {"done": 0, "outdated": 0, "total": 0} for stain in STAINS} for stage in out}
    found = run.groups(project, STAINS)
    calibration, nuclei = run.proposal_inputs(project, fresh=False)
    for stain in STAINS:
        keys = sorted(core for core, s in found if s == stain)
        recorded = {stage: manifest.read_all(project, stage, stain) for stage in out}
        model = None
        counts = {stage: {"done": 0, "outdated": 0, "total": len(keys)} for stage in out}
        for core in keys:
            tiles = found[(core, stain)]
            threshold = calibration.get((tiles[0]["tma"], stain))
            stale = False
            for stage in ("proposals", "calls", "outlines", "candidates"):
                mine = recorded[stage].get(core)
                if mine is None:
                    if stage != "outlines":
                        stale = True
                    continue
                if stage == "proposals":
                    made = None if threshold is None else proposals.expected(project, stain, core, tiles, threshold, nuclei)
                elif stage == "calls":
                    if model is None and project.model(classify.MODEL_KEY[stain]).exists():
                        model = classify.load_model(project, stain)
                    made = classify.expected(project, stain, core, model) if model is not None else None
                elif stage == "outlines":
                    made = outlines.expected(project, stain, core)
                else:
                    made = classify.assembled_expected(project, stain, core)
                stale = stale or made is None or not manifest.up_to_date(mine, made)
                counts[stage]["outdated" if stale else "done"] += 1
        for stage in out:
            out[stage][stain] = counts[stage]
    return out


def _same(project) -> dict:
    cache.forget()
    now = run.status(project)
    assert now == _status_before(project)
    return now


def test_status_is_the_same_as_before_and_merges_the_parameters_once_per_stain(v2_project, monkeypatch):  # noqa: F811
    project = load_project(v2_project)
    assert _same(project)["proposals"]["NeuN"] == {"done": 0, "outdated": 0, "total": 1}
    _proposals(project)
    assert _same(project)["proposals"]["AT8"] == {"done": 1, "outdated": 0, "total": 1}
    run_calls_inline(project)
    found = _same(project)
    assert found["calls"]["NeuN"] == {"done": 1, "outdated": 0, "total": 1}
    assert found["candidates"]["6E10"] == {"done": 1, "outdated": 0, "total": 1}
    # a stage 3 manifest whose weights changed since, a missing calls manifest, a new rule and a new proposal setting
    made = outlines.expected(project, "NeuN", "LIP-1_B-2")
    manifest.write(project, "outlines", "NeuN", "LIP-1_B-2", {**made, "digest": "older"})
    manifest.path(project, "calls", "AT8", "LIP-1_B-2").unlink()
    found = _same(project)
    assert found["outlines"]["NeuN"]["outdated"] == 1 and found["calls"]["AT8"]["total"] == 1
    config = yaml.safe_load((v2_project / "stainid.yaml").read_text())
    config["v2"] = {"rules": {"6E10": {"lumen_contact": 0.5}}, "proposals": {"NeuN": {"levels": [0.6, 1.0]}}}
    (v2_project / "stainid.yaml").write_text(yaml.safe_dump(config))
    project = load_project(v2_project)
    found = _same(project)
    assert found["proposals"]["NeuN"]["outdated"] == 1 and found["calls"]["6E10"]["outdated"] == 1
    # 40 copies of the core: the same counts, and the parameters are merged once per stain and stage, not per core
    import shutil

    real_groups = run.groups

    def many(project, stains, cores=None):
        found = real_groups(project, stains, cores)
        return {(f"{core}{suffix}", stain): tiles for (core, stain), tiles in found.items() for suffix in [""] + [f"-{k}" for k in range(39)]}

    for stage in manifest.STAGES:
        for stain in STAINS:
            source = manifest.path(project, stage, stain, "LIP-1_B-2")
            if source.exists():
                for k in range(39):
                    shutil.copy(source, manifest.path(project, stage, stain, f"LIP-1_B-2-{k}"))
    monkeypatch.setattr(run, "groups", many)
    found = _same(project)
    assert found["proposals"]["NeuN"] == {"done": 0, "outdated": 40, "total": 40}
    calls = []
    real = params.stage_params
    monkeypatch.setattr(params, "stage_params", lambda *a: calls.append(a) or real(*a))
    cache.forget()
    run.status(project)
    assert len(calls) == 4 * len(STAINS)

def test_cached_status_follows_a_new_manifest(v2_project):  # noqa: F811
    project = load_project(v2_project)
    cache.forget()
    _proposals(project)
    before = run.cached_status(project)
    assert before["proposals"]["NeuN"]["done"] == 1
    manifest.path(project, "proposals", "NeuN", "LIP-1_B-2").unlink()
    cache.forget()  # folder signatures are coarse within a second; a fresh look sees the change
    assert run.cached_status(project)["proposals"]["NeuN"] == {"done": 0, "outdated": 0, "total": 1}
