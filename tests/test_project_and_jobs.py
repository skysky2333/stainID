from __future__ import annotations

import yaml

from stainid.project import DEFAULTS, load_project, save_project, write_default_config
from stainid.workflows.steps import to_argv


def test_defaults_and_overrides(tmp_path):
    (tmp_path / "stainid.yaml").write_text("pixel_size_um: 0.5\noutputs:\n  neun: out/neun\n")
    project = load_project(tmp_path)
    assert project.pixel_size_um == 0.5
    assert project.output("neun") == tmp_path.resolve() / "out/neun"
    assert project.output("fields") == tmp_path.resolve() / DEFAULTS["outputs"]["fields"]


def test_write_default_config_roundtrip(tmp_path):
    write_default_config(tmp_path)
    assert load_project(tmp_path).config == DEFAULTS


def test_relative_paths_fold_dot_dot(tmp_path):
    """A folder given as ../scans names files the same way the slides table does (one spelling per file)."""
    (tmp_path / "study").mkdir()
    (tmp_path / "study" / "stainid.yaml").write_text("inputs:\n  slides_dir: ../scans\n")
    assert load_project(tmp_path / "study").input("slides_dir") == tmp_path.resolve() / "scans"


def test_the_run_buttons_run_the_staged_detection_and_keep_the_first_one_as_legacy():
    from stainid.workflows.steps import BY_ID, LEGACY, MAIN_STAGES, depends_on

    main = [s for s in BY_ID.values() if s.stage in MAIN_STAGES]
    assert {"v2_proposals", "v2_calls", "v2_outlines", "v2_measures", "nuclei"} <= {s.id for s in main}
    assert not {"neun", "fields", "masks", "summarize"} & {s.id for s in main}  # "Run what's left" never queues them
    assert all(BY_ID[s].stage == LEGACY and BY_ID[s].command for s in ("neun", "fields", "masks", "summarize"))  # still runnable
    assert "legacy" in BY_ID["neun"].title and "v2" not in BY_ID["v2_proposals"].title
    assert depends_on(BY_ID["v2_measures"]) == ["v2_calls", "v2_outlines"]  # the tables wait for the outlines


def test_job_argv():
    assert to_argv("nuclei", {"stain": ["AT8", "6E10"], "device": "mps", "shard_index": 0, "reverse": False}) == [
        "nuclei", "--stain", "AT8", "--stain", "6E10", "--device", "mps", "--shard-index", "0"]
    assert to_argv("summarize", {"level": "core_id"}) == ["summarize", "--level", "core_id"]
    assert to_argv("prepare_labels", {"stain": "AT8", "fields": 5, "session": ""}) == ["labels", "prepare", "--stain", "AT8", "--fields", "5"]


def test_save_project_writes_only_changes(tmp_path):
    project = save_project(load_project(tmp_path), {"name": "Study", "tma": {"rows": 8}})
    assert load_project(tmp_path).config["tma"] == {"prefix": "TMA-", "rows": 8, "columns": 6}
    assert (tmp_path / "stainid.yaml").read_text() == "name: Study\ntma:\n  rows: 8\n"
    assert project.tma_name("3") == "TMA-3"


def _queue(tmp_path, monkeypatch, budget=None):
    from stainid.jobs import queue

    (tmp_path / "stainid.yaml").write_text("name: t\n")
    monkeypatch.setattr(queue.Scheduler, "_start", lambda self, job, gpus: (setattr(job, "status", "running"), setattr(job, "gpus", gpus)))
    scheduler = queue.Scheduler(tmp_path / "queue", budget=budget or queue.Budget(32, 2, 32))
    return scheduler, load_project(tmp_path)


def test_queue_respects_step_order_and_the_budget(tmp_path, monkeypatch):
    from stainid.jobs import queue

    scheduler, project = _queue(tmp_path, monkeypatch, queue.Budget(8, 1, 32))
    monkeypatch.setattr(queue.Scheduler, "_missing", lambda self, job, step: "")
    nuclei, neun, fields, masks = (scheduler.submit_step(project, step, {}) for step in ("nuclei", "neun", "fields", "masks"))
    for job, when in zip((nuclei, neun, fields, masks), range(4)):
        job.created = job.order = when
    scheduler._schedule()
    assert [j.status for j in (nuclei, neun, fields, masks)] == ["running", "queued", "queued", "queued"]
    assert "Find nuclei" in neun.waiting and "Find nuclei" in fields.waiting and "Nuclei" in fields.waiting
    assert nuclei.gpus == [0]
    nuclei.status = "finished"
    scheduler._schedule()  # the GPU is free: plaques and tau (a light GPU job) go first; NeuN needs all 8 processors and waits
    assert fields.status == "running" and neun.status == "queued" and "processors" in neun.waiting
    assert masks.status == "queued" and "Detect NeuN" in masks.waiting
    fields.status = "finished"
    scheduler._schedule()
    assert neun.status == "running"


def test_failed_step_skips_what_waits_for_it_and_missing_inputs_are_reported(tmp_path, monkeypatch):
    scheduler, project = _queue(tmp_path, monkeypatch)
    export, qc, tiles, summarize = (scheduler.submit_step(project, step, {}) for step in ("export", "qc", "tiles", "summarize"))
    for job, when in zip((export, qc, tiles, summarize), range(4)):
        job.created = job.order = when
    scheduler._schedule()
    assert export.status == "skipped" and "Core table" in export.outcome["error"] and "Find cores" in export.outcome["error"]
    assert qc.status == tiles.status == "skipped" and "Export cores" in qc.outcome["error"]
    assert summarize.status == "running"

    calibrate, neun, nuclei = (scheduler.submit_step(project, step, {}) for step in ("calibrate", "neun", "nuclei"))
    calibrate.created, neun.created, nuclei.created, calibrate.status = 10, 11, 12, "running"
    scheduler._ended(calibrate, 1)
    assert calibrate.status == "failed" and neun.status == "skipped" and "Calibrate stain thresholds” failed" in neun.outcome["error"]
    assert nuclei.status == "queued"


def test_project_models_never_go_to_a_shell_wide_hugging_face_cache(tmp_path, monkeypatch):
    """A shell that points HF_HOME / HF_HUB_CACHE elsewhere (e.g. an LLM cache) must not receive the project's models."""
    from stainid.project import load_project
    (tmp_path / "stainid.yaml").write_text("name: t\n")
    monkeypatch.setenv("HF_HOME", "/elsewhere"); monkeypatch.setenv("HF_HUB_CACHE", "/elsewhere/hub")
    project = load_project(tmp_path)
    project.apply_environment()
    import os
    assert os.environ["HF_HUB_CACHE"].startswith(str(project.model("huggingface_home"))) and os.environ["HF_HOME"] == str(project.model("huggingface_home"))


def _steps_project(tmp_path, analysis=None, tile_stains=("NeuN", "AT8")):
    v2 = {"v2": {"white_matter": {"analysis": analysis}}} if analysis else {}
    (tmp_path / "stainid.yaml").write_text(yaml.safe_dump({"name": "x", **v2}))
    project = load_project(tmp_path)
    tiles = project.analysis_tiles()
    tiles.parent.mkdir(parents=True, exist_ok=True)
    tiles.write_text("tile_id,core_id,tma,stain,selection_order\n" + "".join(f"t{i},c,1,{s},{i}\n" for i, s in enumerate(tile_stains)))
    return project


def test_white_matter_runs_between_the_outlines_and_the_tables():
    from stainid.workflows.steps import STEPS, depends_on, BY_ID

    order = [s.id for s in STEPS]
    assert order.index("v2_calls") < order.index("v2_outlines") < order.index("v2_white_matter") < order.index("v2_grey_matter") < order.index("v2_measures")
    assert depends_on(BY_ID["v2_white_matter"]) == ["v2_proposals", "v2_calls", "v2_outlines"]
    assert "v2_white_matter" in depends_on(BY_ID["v2_grey_matter"])
    assert to_argv("v2_grey_matter", {"workers": 6, "redo": False}) == ["v2", "grey-matter", "--workers", "6"]
    assert to_argv("v2_white_matter", {"redo": True}) == ["v2", "white-matter", "--redo"]


def test_the_tables_use_the_white_matter_steps_only_when_the_results_leave_it_out(tmp_path):
    from stainid.workflows.steps import BY_ID, depends_on, for_project_at, steps_for, wait_reason

    def measures(project):
        steps = steps_for(project)
        return next(s for s in steps if s.id == "v2_measures"), steps

    (tmp_path / "off").mkdir()
    (tmp_path / "exclude").mkdir()
    off, exclude = _steps_project(tmp_path / "off"), _steps_project(tmp_path / "exclude", "exclude")
    step, steps = measures(off)
    assert step == BY_ID["v2_measures"] and depends_on(step, steps) == ["v2_calls", "v2_outlines"]  # exactly as without the steps
    step, steps = measures(exclude)
    assert {"v2_white_matter", "v2_grey_matter"} <= set(step.uses) and {"v2_white_matter", "v2_grey_matter"} <= set(depends_on(step, steps))
    for earlier in ("v2_white_matter", "v2_grey_matter"):
        assert wait_reason(BY_ID[earlier], for_project_at(BY_ID["v2_measures"], str(exclude.root)))
        assert not wait_reason(BY_ID[earlier], for_project_at(BY_ID["v2_measures"], str(off.root)))
    assert for_project_at(BY_ID["v2_measures"], str(tmp_path / "gone")) == BY_ID["v2_measures"]


def test_the_white_matter_steps_say_when_they_are_not_needed_and_then_block_nothing(tmp_path):
    from stainid import cache
    from stainid.workflows.steps import BY_ID, _measures_status, describe, steps_for

    cache.forget()
    for folder, analysis, stains, word in (("off", None, ("NeuN", "AT8"), "off"), ("jog", "exclude", ("6E10", "AT8"), "no NeuN"), ("jog_off", None, ("AT8",), "no NeuN")):
        (tmp_path / folder).mkdir()
        project = _steps_project(tmp_path / folder, analysis, stains)
        for step in ("v2_white_matter", "v2_grey_matter"):
            found = BY_ID[step].status(project)
            assert found["state"] == "done" and not found["outdated"] and found["total"] == 0 and word in found["note"]
        assert _measures_status(project)["state"] == "todo"  # only the tables are left
    (tmp_path / "exclude").mkdir()
    project = _steps_project(tmp_path / "exclude", "exclude")
    assert BY_ID["v2_white_matter"].status(project)["state"] == "todo" and BY_ID["v2_grey_matter"].status(project)["state"] == "todo"
    shown = {s["id"]: s for s in describe(project)}
    assert "uses_when" not in shown["v2_measures"] and [u["id"] for u in shown["v2_measures"]["uses"]] == ["v2_outlines", "v2_white_matter", "v2_grey_matter"]
    assert shown["v2_white_matter"]["needs"][1]["id"] == "v2_calls" and shown["v2_grey_matter"]["needs"][0]["made_by"]["id"] == "v2_white_matter"
    assert [s.id for s in steps_for(project)].count("v2_grey_matter") == 1


def test_the_tables_go_out_of_date_with_the_white_matter_when_they_depend_on_it(tmp_path, monkeypatch):
    import os

    from stainid import cache
    from stainid.v2 import grey_matter, white_matter
    from stainid.workflows.steps import _grey_matter_status, _measures_status, _white_matter_status

    cache.forget()
    project = _steps_project(tmp_path, "exclude")
    table = project.output("v2") / "measures" / "results_donor_region.csv"
    table.parent.mkdir(parents=True)
    table.write_text("x\n1\n")
    os.utime(table, (1000, 1000))
    summary = white_matter.summary_path(project)
    summary.parent.mkdir(parents=True)
    summary.write_text("core_id\n")

    def progress(white, grey):
        monkeypatch.setattr(white_matter, "cached_progress", lambda p: white)
        monkeypatch.setattr(grey_matter, "cached_progress", lambda p: grey)

    fine = {"done": 2, "outdated": 0, "total": 2, "written": 500.0}
    progress(fine, fine)
    assert not _measures_status(project)["outdated"] and _white_matter_status(project)["state"] == "done"
    progress(fine, {**fine, "written": 2000.0})  # the burdens were made after the tables
    assert _measures_status(project)["outdated"]
    progress({"done": 1, "outdated": 1, "total": 2, "written": 500.0}, fine)  # NeuN changed: the white matter has to be made again
    assert _white_matter_status(project)["outdated"] and _grey_matter_status(project)["outdated"] and _measures_status(project)["outdated"]
    progress(fine, {"done": 0, "outdated": 0, "total": 0, "written": 0.0})  # no core has white matter
    assert _grey_matter_status(project)["state"] == "done" and not _measures_status(project)["outdated"]
