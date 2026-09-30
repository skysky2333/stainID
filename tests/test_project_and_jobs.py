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


def test_job_argv():
    assert to_argv("nuclei", {"stain": ["AT8", "6E10"], "device": "mps", "shard_index": 0, "reverse": False}) == [
        "nuclei", "--stain", "AT8", "--stain", "6E10", "--device", "mps", "--shard-index", "0"]
    assert to_argv("summarize", {"level": "core_id"}) == ["summarize", "--level", "core_id"]
    assert to_argv("training_set", {"stain": "AT8", "name": "r1", "enrich": False}) == ["training-set", "--stain", "AT8", "--name", "r1", "--no-enrich"]


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
    assert started == ["nuclei", "neun"]
    assert "Find nuclei" in fields.waiting and "Nuclei" in fields.waiting and "Detect NeuN" in masks.waiting
    nuclei.status = "finished"
    manager._schedule()
    assert started == ["nuclei", "neun", "fields"] and masks.status == "queued"


def test_failed_step_skips_what_waits_for_it_and_missing_inputs_are_reported(tmp_path, monkeypatch):
    from stainid.api import jobs

    monkeypatch.setattr(jobs.JobManager, "_start", lambda self, job: setattr(job, "status", "running"))
    monkeypatch.setattr(jobs.JobManager, "_loop", lambda self: None)
    manager = jobs.JobManager(tmp_path, tmp_path / "jobs")
    export, qc, select, summarize = (manager.submit(step, {}) for step in ("export", "qc", "select", "summarize"))
    for job, when in zip((export, qc, select, summarize), range(4)):
        job.created = when
    manager._schedule()
    assert export.status == "skipped" and "Core table" in export.outcome["error"] and "Find cores" in export.outcome["error"]
    assert qc.status == select.status == "skipped" and "Export cores" in qc.outcome["error"]
    assert summarize.status == "running"

    calibrate, neun, nuclei = manager.submit("calibrate", {}), manager.submit("neun", {}), manager.submit("nuclei", {})
    calibrate.created, neun.created, nuclei.created, calibrate.status = 10, 11, 12, "running"
    manager._finish(calibrate, 1)
    assert calibrate.status == "failed" and neun.status == "skipped" and "Calibrate stain thresholds” failed" in neun.outcome["error"]
    assert nuclei.status == "queued"
