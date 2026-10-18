"""The shared job queue: priority classes, the budget, pre-emption at unit boundaries, pause / resume, and restarts."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

import stainid
from stainid.jobs import JOB_ENV, Job, queue, units
from stainid.project import load_project

SRC = str(Path(stainid.__file__).parents[1])

# A job of `n` units: each unit is skipped when its output exists, else made (and recorded in `made.log`); between units
# it calls `units.boundary()`, so a pause stops it there.
WORKER = """
import sys, time
from pathlib import Path
from stainid.jobs import units
folder, n, seconds = Path(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
folder.mkdir(exist_ok=True)
for i in range(n):
    out = folder / f"unit{i}.done"
    if not out.exists():
        time.sleep(seconds)
        out.write_text("ok")
        with (folder / "made.log").open("a") as log:
            log.write(f"{i}\\n")
    print(f"[{i + 1}/{n}] unit {i}", flush=True)
    units.boundary(f"unit {i}")
"""


def _env() -> dict:
    return {**{k: v for k, v in os.environ.items() if k != JOB_ENV}, "PYTHONPATH": SRC}


def _wait(condition, seconds: float = 30.0, tick=None) -> None:
    end = time.time() + seconds
    while True:
        if tick is not None:
            tick()
        if condition():
            return
        assert time.time() < end, "timed out"
        time.sleep(0.1)


def _worker_job(scheduler, folder: Path, klass: str, n: int, seconds: float, cpus: float, title: str):
    return scheduler.submit_command([sys.executable, "-c", WORKER, str(folder), str(n), str(seconds)], str(folder.parent), title, klass,
                                    {"cpus": cpus, "ram_gb": 1}, env={"PYTHONPATH": SRC})


def test_classes_start_highest_first_and_first_in_first_out_within_one(tmp_path, monkeypatch):
    started = []
    monkeypatch.setattr(queue.Scheduler, "_start", lambda self, job, gpus: (started.append(job.title), setattr(job, "status", "running")))
    scheduler = queue.Scheduler(tmp_path / "q", budget=queue.Budget(4, 0, 32))
    jobs = {}
    for title, klass in (("public 1", "public"), ("training", "models"), ("public 2", "public"), ("pipeline", "pipeline"),
                         ("annotation", "annotation"), ("download", "downloads")):
        jobs[title] = scheduler.submit_command(["true"], str(tmp_path), title, klass, {"cpus": 4, "ram_gb": 1, "network": klass == "downloads"})
    scheduler._schedule()
    assert started == ["annotation", "download"]  # one 4-processor job fits; the network-only download runs alongside
    assert "processors" in jobs["pipeline"].waiting
    info = scheduler.describe(jobs["public 2"])
    assert info["class"] == "public" and info["queue"]["position"] == 2 and info["queue"]["in_class"] == 2
    for title in ("annotation",):
        jobs[title].status = "finished"
    scheduler._schedule()
    assert started[-1] == "pipeline"
    jobs["pipeline"].status = "finished"
    scheduler.request("move", jobs["public 2"].id, where="up")  # Move up: public 2 now before public 1 (training still first)
    scheduler._schedule()
    jobs["training"].status = "finished"
    scheduler._schedule()
    assert started == ["annotation", "download", "pipeline", "training", "public 2"]


def test_a_higher_class_pre_empts_a_lower_one_at_a_unit_boundary_and_it_resumes_without_redoing_units(tmp_path):
    scheduler = queue.Scheduler(tmp_path / "q", budget=queue.Budget(4, 0, 32), launcher=queue.ProcessLauncher())
    low_dir, high_dir = tmp_path / "low", tmp_path / "high"
    low = _worker_job(scheduler, low_dir, "public", 8, 0.4, 4, "public dataset")
    scheduler.tick()
    assert low.status == "running"
    _wait(lambda: (low_dir / "made.log").exists() and len((low_dir / "made.log").read_text().split()) >= 2, tick=scheduler.tick)
    high = _worker_job(scheduler, high_dir, "pipeline", 2, 0.1, 4, "main study")
    scheduler.tick()
    assert low.request == "preempt" and high.status == "queued" and "lower-priority work" in high.waiting
    _wait(lambda: high.status == "running", tick=scheduler.tick)
    assert low.status == "queued" and low.paused_by == "preempt" and low.returncode == units.PAUSED  # stopped between two units
    made_before = (low_dir / "made.log").read_text().split()
    assert 2 <= len(made_before) < 8
    _wait(lambda: high.status == "finished", tick=scheduler.tick)
    _wait(lambda: low.status == "finished", seconds=60, tick=scheduler.tick)
    made = (low_dir / "made.log").read_text().split()
    assert sorted(made, key=int) == [str(i) for i in range(8)]  # every unit once: none redone after the pause
    assert len(low.runs) == 2


def test_a_job_that_does_not_reach_a_boundary_in_time_is_stopped_and_queued_again(tmp_path, monkeypatch):
    monkeypatch.setitem(queue.GRACE, "public", 0.5)
    scheduler = queue.Scheduler(tmp_path / "q", budget=queue.Budget(4, 0, 32), launcher=queue.ProcessLauncher())
    low = _worker_job(scheduler, tmp_path / "low", "public", 1, 60, 4, "one long unit")
    scheduler.tick()
    _wait(lambda: scheduler._pids(low) and len(scheduler._pids(low)) == 2, tick=scheduler.tick)
    _worker_job(scheduler, tmp_path / "high", "annotation", 1, 0.1, 4, "annotation")
    _wait(lambda: low.status == "queued", seconds=20, tick=scheduler.tick)
    assert low.paused_by == "preempt" and not (tmp_path / "low" / "unit0.done").exists()


def test_pause_resume_cancel_and_pause_all(tmp_path):
    scheduler = queue.Scheduler(tmp_path / "q", budget=queue.Budget(8, 0, 32), launcher=queue.ProcessLauncher())
    job = _worker_job(scheduler, tmp_path / "a", "pipeline", 6, 0.3, 2, "a")
    waiting = scheduler.submit_command(["sleep", "30"], str(tmp_path), "b", "models", {"cpus": 2, "ram_gb": 1})
    scheduler.request("pause", waiting.id)
    assert waiting.status == "paused"
    scheduler.tick()
    assert job.status == "running" and waiting.status == "paused"
    _wait(lambda: (tmp_path / "a" / "made.log").exists(), tick=scheduler.tick)
    scheduler.request("pause", job.id)
    _wait(lambda: job.status == "paused", tick=scheduler.tick)
    done = len((tmp_path / "a" / "made.log").read_text().split())
    assert done < 6
    scheduler.request("cancel", waiting.id)
    assert waiting.status == "cancelled"
    scheduler.request("pause_all")
    scheduler.request("resume", job.id)
    scheduler.tick()
    assert job.status == "queued" and job.waiting == "All background work is paused"
    scheduler.request("resume_all")
    _wait(lambda: job.status == "finished", tick=scheduler.tick)
    assert sorted((tmp_path / "a" / "made.log").read_text().split(), key=int) == [str(i) for i in range(6)]


def test_a_restarted_queue_picks_up_queued_paused_and_running_jobs(tmp_path):
    folder = tmp_path / "q"
    first = queue.Scheduler(folder, budget=queue.Budget(4, 0, 32), launcher=queue.ProcessLauncher())
    running = _worker_job(first, tmp_path / "r", "pipeline", 3, 0.5, 2, "running")
    paused = _worker_job(first, tmp_path / "p", "pipeline", 1, 0.1, 2, "paused")
    queued = _worker_job(first, tmp_path / "w", "public", 1, 0.1, 4, "queued")
    first.request("pause", paused.id)
    first.tick()
    assert running.status == "running" and paused.status == "paused" and queued.status == "queued"
    first.close()  # the app stops; the running job carries on (it is not the app's child in its own session)

    second = queue.Scheduler(folder, budget=queue.Budget(4, 0, 32), launcher=queue.ProcessLauncher())
    assert second.leader
    assert second.jobs[running.id].status == "running"  # adopted
    assert second.jobs[paused.id].status == "paused" and second.jobs[queued.id].status == "queued"
    _wait(lambda: second.jobs[running.id].status == "finished" and second.jobs[queued.id].status == "finished", tick=second.tick)
    assert second.jobs[paused.id].status == "paused"
    second.request("resume", paused.id)
    _wait(lambda: second.jobs[paused.id].status == "finished", tick=second.tick)


def test_only_one_process_leads_and_the_others_leave_requests(tmp_path):
    leader = queue.Scheduler(tmp_path / "q", budget=queue.Budget(4, 0, 32), launcher=queue.ProcessLauncher())
    other = queue.Scheduler(tmp_path / "q", budget=queue.Budget(4, 0, 32))
    assert leader.leader and not other.leader and other.led_elsewhere()
    job = other.submit_command(["sleep", "30"], str(tmp_path), "from the command line", "models", {"cpus": 4, "ram_gb": 1})
    other.request("pause", job.id)
    assert other.jobs[job.id].status == "queued"  # left for the leader
    leader.tick()
    assert leader.jobs[job.id].status == "paused"
    other.tick()
    assert other.jobs[job.id].status == "paused"


def test_stainid_jobs_command_line(tmp_path):
    (tmp_path / "stainid.yaml").write_text(yaml.safe_dump({"name": "cli", "dataset": {"id": "x"}}))
    run = lambda *words: subprocess.run([sys.executable, "-m", "stainid.cli", "--project", str(tmp_path), "jobs", *words],  # noqa: E731
                                        capture_output=True, text=True, env=_env())
    done = run("submit", "--", "nuclei", "--stain", "AT8")
    assert done.returncode == 0, done.stderr
    assert "Queued" in done.stdout and "Public datasets" in done.stdout  # a public dataset's steps go in the public class
    done = run("submit", "--exec", "--class", "downloads", "--network", "--title", "Big downloads", "--", "bash", "-c", "echo hi")
    assert done.returncode == 0 and "Downloads" in done.stdout
    listing = run("list", "--json")
    overview = json.loads(listing.stdout)
    by_class = {c["id"]: c["jobs"] for c in overview["classes"]}
    [nuclei] = by_class["public"]
    [download] = by_class["downloads"]
    assert nuclei["workflow"] == "nuclei" and nuclei["options"]["stain"] == ["AT8"] and nuclei["needs"]["gpus"] == 1
    assert download["title"] == "Big downloads" and download["needs"]["network"]
    assert run("pause", nuclei["id"]).returncode == 0
    assert run("priority", nuclei["id"], "--class", "pipeline").returncode == 0
    leader = queue.Scheduler(queue.queue_dir(), budget=queue.Budget(0, 0, 0))  # takes the requests, starts nothing
    leader.tick()
    assert leader.jobs[nuclei["id"]].status == "paused" and leader.jobs[nuclei["id"]].klass == "pipeline"
    assert "Main-study pipeline" in run("list").stdout


def test_the_class_of_a_step_follows_the_project(tmp_path):
    (tmp_path / "stainid.yaml").write_text(yaml.safe_dump({"name": "main"}))
    main = load_project(tmp_path)
    assert queue.class_of(main, "v2_proposals") == "pipeline" and queue.class_of(main, "prepare_labels") == "annotation"
    assert queue.class_of(main, "train") == "models" and queue.class_of(main, "download_models") == "downloads"
    (tmp_path / "stainid.yaml").write_text(yaml.safe_dump({"name": "public", "dataset": {"id": "x"}}))
    assert queue.class_of(load_project(tmp_path), "v2_proposals") == "public"
    assert queue.needs_of("v2_proposals", {"workers": 6})["cpus"] == 12 and queue.needs_of("nuclei", {"device": "cpu"})["gpus"] == 0
    assert isinstance(Job(id="x", workflow="w", options={}, argv=[]).needs, dict)


def test_a_download_continues_a_partial_file_with_a_range_request(tmp_path):
    import http.server
    import threading

    from stainid.workflows import downloads

    body = bytes(range(256)) * 4096  # 1 MB
    asked = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            start = int(self.headers.get("Range", "bytes=0-")[6:].split("-")[0])
            asked.append(start)
            self.send_response(206 if start else 200)
            self.send_header("Content-Length", str(len(body) - start))
            self.end_headers()
            self.wfile.write(body[start:])

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    target = tmp_path / "weights.pth"
    target.with_name("weights.pth.part").write_bytes(body[:300_000])  # what a paused or broken run left
    downloads._fetch(f"http://127.0.0.1:{server.server_port}/w", target)
    server.shutdown()
    assert asked == [300_000] and target.read_bytes() == body and not target.with_name("weights.pth.part").exists()


def test_calibration_keeps_each_slide_as_it_is_done_and_stops_between_slides(tmp_path, monkeypatch):
    import numpy as np

    from stainid.workflows import calibrate

    (tmp_path / "stainid.yaml").write_text(yaml.safe_dump({"name": "c"}))
    project = load_project(tmp_path)
    project.analysis_tiles().parent.mkdir(parents=True, exist_ok=True)
    project.analysis_tiles().write_text("tile_id,core_id,tma,stain,image_path,x_px,y_px,width_px,height_px\n" + "".join(
        f"t{tma}{s},c{tma},{tma},{s},x.png,0,0,10,10\n" for tma in (1, 2) for s in ("NeuN", "AT8")))
    monkeypatch.setattr(calibrate, "sample_dab", lambda *a, **k: {"values": np.linspace(0, 0.3, 1000, dtype=np.float32), "excluded_px": 0, "tissue_px": 1})
    control = tmp_path / "control"
    monkeypatch.setenv(units.CONTROL_ENV, str(control))
    calls = []
    real = calibrate.calibrate_threshold

    def counted(values):
        calls.append(1)
        if len(calls) == 2:
            control.write_text("pause")  # the queue asks it to stop while it works on the second slide
        return real(values)

    monkeypatch.setattr(calibrate, "calibrate_threshold", counted)
    units.forget()
    with pytest.raises(units.Paused):
        calibrate.calibrate_slides(project)
    assert len(project.input("calibration").read_text().splitlines()) == 3  # the header and the two slides done before it stopped
    control.unlink()
    units.forget()
    calibrate.calibrate_slides(project)
    assert len(project.input("calibration").read_text().splitlines()) == 5 and len(calls) == 4  # only the other two were done


def test_a_job_queued_after_another_waits_for_it_and_is_skipped_when_it_fails(tmp_path, monkeypatch):
    started = []
    monkeypatch.setattr(queue.Scheduler, "_start", lambda self, job, gpus: (started.append(job.title), setattr(job, "status", "running")))
    scheduler = queue.Scheduler(tmp_path / "q", budget=queue.Budget(32, 0, 32))
    first = scheduler.submit_command(["true"], str(tmp_path), "proposals", "public", {"cpus": 2})
    then = scheduler.submit_command(["true"], str(tmp_path), "score", "public", {"cpus": 1}, after=[first.id])
    other = scheduler.submit_command(["true"], str(tmp_path), "check", "public", {"cpus": 1}, after=[then.id])
    scheduler._schedule()
    assert started == ["proposals"] and then.waiting == "Waits for “proposals” to finish"
    first.status = "finished"
    scheduler._schedule()
    assert started == ["proposals", "score"]
    scheduler._ended(then, 1)
    assert then.status == "failed" and other.status == "skipped" and "“score” failed" in other.outcome["error"]


# An --exec script that reports through $STAINID_PROGRESS_FILE: "total 4", then one "unit done" line per unit.
REPORTER = """
import os, time
path = os.environ["STAINID_PROGRESS_FILE"]
with open(path, "a") as out:
    out.write("total 4\\n")
for i in range(4):
    time.sleep(0.3 if i < 2 else 30)
    with open(path, "a") as out:
        out.write(f"unit done {i}\\n")
"""


def test_a_command_without_progress_shows_its_time_so_far_and_never_a_made_up_estimate(tmp_path, monkeypatch):
    monkeypatch.setattr(queue.Scheduler, "_start", lambda self, job, gpus: setattr(job, "status", "running"))
    scheduler = queue.Scheduler(tmp_path / "q", budget=queue.Budget(8, 0, 32))
    unknown = scheduler.submit_command(["true"], str(tmp_path), "variants", "public", {"cpus": 1, "ram_gb": 1})
    expected = scheduler.submit_command(["true"], str(tmp_path), "timed", "public", {"cpus": 1, "ram_gb": 1}, time_s=600)
    scheduler._schedule()
    for job in (unknown, expected):
        job.started = time.time() - 6300  # 1 h 45 min in
    info = scheduler.describe(unknown)  # nothing to go by: its time so far, no estimate
    assert info["status"] == "running" and info["eta_s"] is None and info["overdue"] is False and 6290 <= info["elapsed_s"] <= 6310
    info = scheduler.describe(expected)  # expected to take 10 min: taking longer than expected, not "about 30 s left"
    assert info["eta_s"] is None and info["overdue"] is True
    expected.started = time.time() - 590
    info = scheduler.describe(expected)
    assert 5 <= info["eta_s"] <= 15 and info["overdue"] is False  # and no 30-second floor


def test_an_exec_job_reports_progress_through_its_progress_file(tmp_path):
    from stainid.jobs import progress, progress_file

    scheduler = queue.Scheduler(tmp_path / "q", budget=queue.Budget(4, 0, 32), launcher=queue.ProcessLauncher())
    job = scheduler.submit_command([sys.executable, "-c", REPORTER], str(tmp_path), "variants", "public", {"cpus": 1, "ram_gb": 1},
                                   env={"PYTHONPATH": SRC})
    scheduler.tick()
    _wait(lambda: scheduler.describe(job)["progress"]["done"] >= 2, tick=scheduler.tick)
    info = scheduler.describe(job)
    assert info["status"] == "running" and info["progress"]["total"] == 4 and info["eta_s"] is not None and info["overdue"] is False
    scheduler.request("cancel", job.id)
    _wait(lambda: job.status == "cancelled", tick=scheduler.tick)

    progress_file(tmp_path / "x.log").write_text("3/72\n12/72\n")  # "done/total" lines: the last one counts
    assert (progress(tmp_path / "x.log")["done"], progress(tmp_path / "x.log")["total"]) == (12, 72)


def test_units_done_before_a_pause_still_count_and_a_job_being_paused_shows_no_time_left(tmp_path, monkeypatch):
    from stainid.jobs import cli as jobs_cli

    monkeypatch.setattr(queue.Scheduler, "_start", lambda self, job, gpus: setattr(job, "status", "running"))
    scheduler = queue.Scheduler(tmp_path / "q", budget=queue.Budget(8, 0, 32))
    job = scheduler.submit_command(["true"], str(tmp_path), "proposals", "public", {"cpus": 1, "ram_gb": 1})
    scheduler._schedule()
    log = tmp_path / "job.log"
    job.log_file, job.started = str(log), time.time() - 600
    log.write_text("".join(f"[{i}/72] core {i}\n" for i in range(1, 43)))
    job.request = "preempt"  # being paused for higher-priority work: "pausing…", not its grace period as time left
    info = scheduler.describe(job)
    assert info["eta_s"] is None and "pausing…" in jobs_cli._line(info)
    scheduler._ended(job, units.PAUSED)
    assert job.status == "queued" and job.outcome["done"] == 42
    assert scheduler.describe(job)["progress"]["done"] == 42  # waiting: the units done are still shown
    scheduler._schedule()
    job.started = time.time() - 60
    assert scheduler.describe(job)["progress"]["done"] == 42  # resumed, nothing new said yet
    with log.open("a") as out:
        out.write("[1/30] core 43\n[2/30] core 44\n")  # the resumed step counts only what is left
    progress = scheduler.describe(job)["progress"]
    assert (progress["done"], progress["total"]) == (44, 72)
    left = scheduler.describe(job)["eta_s"]
    assert 60 * 28 / 2 * 0.9 <= left <= 60 * 28 / 2 * 1.1  # the pace of this run: 2 units a minute, 28 left
    scheduler._ended(job, 0)
    assert job.status == "finished" and (job.outcome["done"], job.outcome["total"]) == (44, 72)
