from __future__ import annotations

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


def test_queue_respects_step_order_and_heavy_slots(tmp_path, monkeypatch):
    from stainid.api import jobs

    started = []
    monkeypatch.setattr(jobs.JobManager, "_start", lambda self, job: (started.append(job.workflow), setattr(job, "status", "running")))
    monkeypatch.setattr(jobs.JobManager, "_loop", lambda self: None)
    monkeypatch.setattr(jobs.JobManager, "_missing", lambda self, step: "")
    manager = jobs.JobManager(tmp_path, tmp_path / "jobs", max_heavy=2)
    nuclei, neun, fields, masks = (manager.submit(step, {}) for step in ("nuclei", "neun", "fields", "masks"))
    for job, when in zip((nuclei, neun, fields, masks), range(4)):
        job.created = when
    manager._schedule()
    assert started == ["nuclei"]
    assert "Find nuclei" in neun.waiting and "Find nuclei" in fields.waiting and "Nuclei" in fields.waiting
    nuclei.status = "finished"
    manager._schedule()
    assert started == ["nuclei", "neun", "fields"] and masks.status == "queued" and "Detect NeuN" in masks.waiting


def test_failed_step_skips_what_waits_for_it_and_missing_inputs_are_reported(tmp_path, monkeypatch):
    from stainid.api import jobs

    monkeypatch.setattr(jobs.JobManager, "_start", lambda self, job: setattr(job, "status", "running"))
    monkeypatch.setattr(jobs.JobManager, "_loop", lambda self: None)
    manager = jobs.JobManager(tmp_path, tmp_path / "jobs")
    export, qc, tiles, summarize = (manager.submit(step, {}) for step in ("export", "qc", "tiles", "summarize"))
    for job, when in zip((export, qc, tiles, summarize), range(4)):
        job.created = when
    manager._schedule()
    assert export.status == "skipped" and "Core table" in export.outcome["error"] and "Find cores" in export.outcome["error"]
    assert qc.status == tiles.status == "skipped" and "Export cores" in qc.outcome["error"]
    assert summarize.status == "running"

    calibrate, neun, nuclei = manager.submit("calibrate", {}), manager.submit("neun", {}), manager.submit("nuclei", {})
    calibrate.created, neun.created, nuclei.created, calibrate.status = 10, 11, 12, "running"
    manager._finish(calibrate, 1)
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
