"""The shared job store: app jobs, their exit codes across a server restart, and steps run from the command line."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

import stainid
from stainid.api import jobs as server_jobs
from stainid.jobs import COMMAND_LINE, JOB_ENV, Job, JobStore, alive, command_line_job

SRC = str(Path(stainid.__file__).parents[1])
ENV = {**{k: v for k, v in os.environ.items() if k != JOB_ENV}, "PYTHONPATH": SRC}


def _project(tmp_path: Path) -> Path:
    (tmp_path / "stainid.yaml").write_text(yaml.safe_dump({"name": "jobs test"}))
    return tmp_path


def _manager(root: Path, monkeypatch, started: list | None = None) -> server_jobs.JobManager:
    monkeypatch.setattr(server_jobs.JobManager, "_loop", lambda self: None)
    monkeypatch.setattr(server_jobs.JobManager, "_missing", lambda self, step: "")
    if started is not None:
        monkeypatch.setattr(server_jobs.JobManager, "_start",
                            lambda self, job: (started.append(job.workflow), setattr(job, "status", "running")))
    return server_jobs.JobManager(root, root / "data" / "jobs")


def _wait(condition, seconds: float = 20.0) -> None:
    end = time.time() + seconds
    while not condition():
        assert time.time() < end, "timed out"
        time.sleep(0.05)


def test_store_round_trip_and_tolerates_old_and_broken_records(tmp_path):
    store = JobStore(tmp_path / "jobs")
    job = Job(id="20260101-000000-abcdef", workflow="summarize", options={"level": "core_id"}, argv=["summarize"], source=COMMAND_LINE)
    store.save(job)
    old = {"id": "20250101-000000-000000", "workflow": "qc", "options": {}, "argv": ["qc"], "status": "finished", "future_field": 1}
    (store.dir / f"{old['id']}.json").write_text(json.dumps(old))
    (store.dir / "broken.json").write_text("{not json")
    (store.dir / "._20260101-000000-abcdef.json").write_bytes(b"\0\1")  # macOS resource fork on exFAT
    loaded = {j.id: j for j in store.all()}
    assert set(loaded) == {job.id, old["id"]}
    assert loaded[job.id] == job and loaded[old["id"]].source == "app" and loaded[old["id"]].log_file == ""
    assert store.log_path(loaded[old["id"]]) == store.dir / f"{old['id']}.log" and store.log_path(job) is None
    assert not list(store.dir.glob("*.tmp"))


def test_runner_records_exit_code_and_stops_the_whole_group(tmp_path):
    exit_file = tmp_path / "a.exit"
    done = subprocess.run([sys.executable, "-m", "stainid.jobs.run", str(exit_file), "--", sys.executable, "-c", "import sys; sys.exit(3)"],
                          env=ENV)
    assert done.returncode == 3 and exit_file.read_text() == "3"

    exit_file, marker = tmp_path / "b.exit", tmp_path / "grandchild.pid"
    script = (f"import subprocess, sys, time; p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
              f"open({str(marker)!r}, 'w').write(str(p.pid)); time.sleep(60)")
    runner = subprocess.Popen([sys.executable, "-m", "stainid.jobs.run", str(exit_file), "--", sys.executable, "-c", script],
                              env=ENV, start_new_session=True)
    _wait(lambda: marker.exists() and marker.read_text())
    os.killpg(runner.pid, signal.SIGTERM)  # what Stop in the app does
    assert runner.wait(20) == 128 + signal.SIGTERM and exit_file.read_text() == str(-signal.SIGTERM)
    grandchild = int(marker.read_text())
    _wait(lambda: not alive(grandchild) or _zombie(grandchild))


def _zombie(pid: int) -> bool:
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] == "Z"
    except OSError:
        return True


def test_restarted_server_finishes_adopted_jobs_with_their_real_exit_code(tmp_path, monkeypatch):
    root = _project(tmp_path)
    store = JobStore(root / "data" / "jobs")
    ended, running = Job(id="1-ended", workflow="summarize", options={}, argv=["summarize"]), Job(id="2-running", workflow="qc", options={}, argv=["qc"])
    gone = Job(id="0-gone", workflow="export", options={}, argv=["export"])
    script = "import sys, time; time.sleep(float(sys.argv[1])); sys.exit(int(sys.argv[2]))"
    processes = {}
    for job, (seconds, code) in ((ended, (0, 0)), (running, (1.5, 5))):
        processes[job.id] = subprocess.Popen([sys.executable, "-m", "stainid.jobs.run", str(store.exit_path(job.id)), "--", sys.executable,
                                              "-c", script, str(seconds), str(code)], env=ENV, start_new_session=True)
        job.status, job.pid, job.started = "running", processes[job.id].pid, time.time()
        store.save(job)
    processes[ended.id].wait()
    gone.status, gone.pid, gone.started = "running", processes[ended.id].pid, time.time()  # its process ended without an exit file
    store.save(gone)

    manager = _manager(root, monkeypatch)  # the "restarted" server: these processes are not its children
    assert manager.jobs[ended.id].status == "finished" and manager.jobs[ended.id].returncode == 0
    assert manager.jobs[gone.id].status == "interrupted"
    assert manager.jobs[running.id].status == "running"
    processes[running.id].wait()
    manager._reap()
    assert manager.jobs[running.id].status == "failed" and manager.jobs[running.id].returncode == 5
    assert json.loads(store.path(running.id).read_text())["returncode"] == 5


def test_app_job_runs_through_the_runner(tmp_path, monkeypatch):
    root = _project(tmp_path)
    monkeypatch.setenv("PYTHONPATH", SRC)
    manager = _manager(root, monkeypatch)
    job = manager.submit("summarize", {})
    manager._schedule()
    assert job.status == "running" and job.log_file
    _wait(lambda: (manager._reap(), job.status != "running")[1])
    assert job.status == "failed" and job.returncode == 1 and "No detection results yet" in job.outcome["error"]
    assert manager.store.exit_code(job.id) == 1


def test_command_line_run_registers_in_the_shared_list(tmp_path):
    root = _project(tmp_path)
    log = tmp_path / "run.log"
    with log.open("w") as out:
        done = subprocess.run([sys.executable, "-m", "stainid.cli", "--project", str(root), "summarize", "--level", "core_id"],
                              stdout=out, stderr=subprocess.STDOUT, env=ENV)
    assert done.returncode == 1
    [job] = JobStore(root / "data" / "jobs").all()
    assert job.source == COMMAND_LINE and job.workflow == "summarize" and job.argv == ["summarize", "--level", "core_id"]
    assert job.options == {"level": "core_id"} and job.status == "failed" and job.returncode == 1
    assert Path(job.log_file) == log and "No detection results yet" in job.outcome["error"]

    subprocess.run([sys.executable, "-m", "stainid.cli", "--project", str(root), "summarize"], capture_output=True, env={**ENV, JOB_ENV: "x"})
    subprocess.run([sys.executable, "-m", "stainid.cli", "--project", str(root), "info"], capture_output=True, env=ENV)
    assert len(JobStore(root / "data" / "jobs").all()) == 1  # started by the app, or not a workflow step: not registered


def test_command_line_run_holds_back_conflicting_app_jobs(tmp_path, monkeypatch):
    root = _project(tmp_path)
    started = []
    manager = _manager(root, monkeypatch, started)
    store = manager.store
    queued = manager.submit("neun", {})
    with command_line_job(store, "nuclei", {}, ["nuclei", "--stain", "NeuN"]) as outside:
        manager._rescan()
        assert manager.jobs[outside.id].source == COMMAND_LINE
        manager._schedule()
        assert started == [] and queued.waiting.startswith("Waits for “Find nuclei”, running from the command line, to finish")
        other = manager.submit("download_models", {})
        manager.jobs[outside.id].workflow = "export"  # two heavy steps outside the app fill both slots
        manager.jobs["x"] = Job(id="x", workflow="dearray", options={}, argv=["dearray"], status="running", pid=os.getpid(), source=COMMAND_LINE)
        manager._schedule()
        assert "2 heavy steps are already running, 2 of them from the command line" in other.waiting
        del manager.jobs["x"]
    assert store.load(store.path(outside.id)).status == "finished"
    manager._rescan()
    manager._schedule()
    assert manager.jobs[outside.id].status == "finished" and started == ["neun", "download_models"]


def test_command_line_runs_that_die_or_are_stopped_from_the_app(tmp_path, monkeypatch):
    root = _project(tmp_path)
    store = JobStore(root / "data" / "jobs")
    finished = subprocess.Popen([sys.executable, "-c", "pass"])
    finished.wait()
    dead = Job(id="dead", workflow="nuclei", options={}, argv=["nuclei"], status="running", pid=finished.pid, started=time.time(), source=COMMAND_LINE)
    store.save(dead)
    script = ("import sys, time; from pathlib import Path; from stainid.jobs import JobStore, command_line_job\n"
              "with command_line_job(JobStore(Path(sys.argv[1])), 'nuclei', {}, ['nuclei']):\n    print('ready', flush=True); time.sleep(60)")
    terminal = subprocess.Popen([sys.executable, "-c", script, str(store.dir)], env=ENV, stdout=subprocess.PIPE, start_new_session=True)
    assert terminal.stdout.readline().strip() == b"ready"

    manager = _manager(root, monkeypatch)
    assert manager.jobs["dead"].status == "interrupted"
    [running] = [j for j in manager.jobs.values() if j.pid == terminal.pid]
    manager.cancel(running.id)
    assert terminal.wait(20) == -signal.SIGTERM
    record = store.load(store.path(running.id))
    assert record.status == "cancelled" and record.returncode == 128 + signal.SIGTERM
    manager._rescan()
    manager._reap()
    assert manager.jobs[running.id].status == "cancelled"


@pytest.mark.skipif(not Path("/proc/self/stat").exists(), reason="process start times come from /proc")
def test_a_reused_process_number_is_not_taken_for_the_job():
    assert alive(os.getpid(), time.time()) and not alive(os.getpid(), time.time() - 3600 * 24 * 365 * 30)


def test_preparing_labels_does_not_wait_for_a_long_detection_run():
    """Label preparation is light and only waits for what it needs, not for steps making inputs it merely uses."""
    from stainid.workflows.steps import BY_ID, wait_reason
    prepare = BY_ID["prepare_labels"]
    assert not prepare.heavy
    assert wait_reason(BY_ID["nuclei"], prepare) == ""
    assert wait_reason(BY_ID["nuclei"], BY_ID["neun"]) != ""  # detection steps still wait for the nuclei they use
