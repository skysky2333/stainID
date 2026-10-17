"""How the queue schedules (`stainid.jobs.policy`): EASY backfill, conflict keys, job arrays, fair share, caps, sizes."""
from __future__ import annotations

import time
from pathlib import Path

import pytest
import yaml

from stainid.jobs import queue
from stainid.jobs.units import PAUSED
from stainid.project import load_project


@pytest.fixture()
def started(monkeypatch):
    """Jobs "start" without a process: the order they started in."""
    order = []

    def start(self, job, gpus):
        order.append(job.title)
        job.status, job.started, job.gpus = "running", time.time(), gpus

    monkeypatch.setattr(queue.Scheduler, "_start", start)
    monkeypatch.setattr(queue.Scheduler, "_missing", lambda self, job, step: "")
    return order


def _project(root: Path, name: str = "study", **config):
    root.mkdir(parents=True, exist_ok=True)
    (root / "stainid.yaml").write_text(yaml.safe_dump({"name": name, **config}))
    return load_project(root)


def _command(scheduler, title, klass, cpus, seconds=0.0, project=None, **needs):
    return scheduler.submit_command(["true"], "/tmp", title, klass, {"cpus": cpus, "ram_gb": 1, **needs}, project=project, time_s=seconds)


def test_backfill_starts_small_jobs_around_a_big_one_but_never_delays_it(tmp_path, started):
    scheduler = queue.Scheduler(tmp_path / "q", budget=queue.Budget(12, 0, 64))
    running = _command(scheduler, "running", "pipeline", 4, 100)
    scheduler._schedule()
    assert started == ["running"]
    top = _command(scheduler, "big main-study job", "pipeline", 10, 1000)
    short = _command(scheduler, "short", "public", 2, 50)  # ends before the big job can start
    beside = _command(scheduler, "beside", "public", 2, 500)  # long, but uses only what the big job leaves over (12 - 10)
    long = _command(scheduler, "long", "public", 2, 500)  # long, and would take what the big job needs
    quick = _command(scheduler, "quick", "public", 2, 30)
    scheduler._schedule()
    assert started == ["running", "short", "beside", "quick"]
    assert top.status == "queued" and "expected to start in about" in top.waiting
    assert long.status == "queued" and long.waiting.startswith("Waits so as not to delay “big main-study job”")
    for job in (running, short, quick):  # what ends before the big job's start ended
        job.status = "finished"
    scheduler._schedule()
    assert started[-1] == "big main-study job"  # it starts on time: nothing that started meanwhile is in its way
    assert long.status == "queued" and beside.status == "running"


def test_a_second_writer_of_the_same_outputs_waits_and_readers_and_other_cores_do_not(tmp_path, started):
    project = _project(tmp_path / "p")
    lists = {}
    for name, cores in (("ab", "a\nb\n"), ("bc", "b\nc\n"), ("de", "d\ne\n")):
        lists[name] = tmp_path / f"{name}.txt"
        lists[name].write_text(cores)
    scheduler = queue.Scheduler(tmp_path / "q", budget=queue.Budget(64, 0, 64, io_per_disk=10))

    def proposals(title, stain, cores):
        return scheduler.submit_step(project, "v2_proposals", {"stain": [stain]}, needs={"cpus": 2, "ram_gb": 1}, array=False, title=title,
                                     argv=["v2", "proposals", "--stain", stain, "--core", str(lists[cores])])

    first = proposals("AT8 a b", "AT8", "ab")
    overlapping = proposals("AT8 b c", "AT8", "bc")
    other_cores = proposals("AT8 d e", "AT8", "de")  # (c would wait too: "AT8 b c" writes it and was queued first)
    other_stain = proposals("NeuN a b", "NeuN", "ab")
    keyed = scheduler.submit_command(["true"], "/tmp", "writes x", "models", {"cpus": 1}, conflicts=["x"])
    same_key = scheduler.submit_command(["true"], "/tmp", "also writes x", "models", {"cpus": 1}, conflicts=["x"])
    scheduler._schedule()
    assert started == ["AT8 a b", "AT8 d e", "NeuN a b", "writes x"]
    assert overlapping.waiting == "Waits for “AT8 a b”: both write the same outputs"
    assert same_key.waiting == "Waits for “writes x”: both write the same outputs"
    first.status = keyed.status = "finished"
    other_cores.status = "finished"
    scheduler._schedule()
    assert started[-2:] == ["AT8 b c", "also writes x"]
    assert other_stain.status == "running"


def _tma_project(root: Path, tmas: int = 4, cores: int = 3):
    project = _project(root)
    project.analysis_tiles().parent.mkdir(parents=True, exist_ok=True)
    rows = ["tile_id,core_id,tma,stain"]
    for tma in range(1, tmas + 1):
        for core in range(cores):
            for tile in range(2):
                rows.append(f"t{tma}{core}{tile},LIP-{tma}_A-{core},{tma},AT8")
    project.analysis_tiles().write_text("\n".join(rows) + "\n")
    return project


def test_a_big_step_runs_as_an_array_of_chunks_and_is_pre_empted_one_chunk_at_a_time(tmp_path, started):
    project = _tma_project(tmp_path / "p", tmas=4)
    scheduler = queue.Scheduler(tmp_path / "q", budget=queue.Budget(8, 0, 64))
    parent = scheduler.submit_step(project, "v2_proposals", {"stain": ["AT8"], "workers": 2}, klass="public")
    chunks = [scheduler.jobs[c] for c in parent.children]
    assert len(chunks) == 4 and [c.title.split(" · ")[1] for c in chunks] == ["TMA 1", "TMA 2", "TMA 3", "TMA 4"]
    listing = Path(chunks[1].argv[chunks[1].argv.index("--core") + 1])
    assert listing.read_text().split() == ["LIP-2_A-0", "LIP-2_A-1", "LIP-2_A-2"] and chunks[1].units == 3
    assert parent.units == 12 and chunks[0].needs["cpus"] == 4
    scheduler._schedule()
    scheduler._update_parents()
    assert [c.status for c in chunks] == ["running", "running", "queued", "queued"] and parent.status == "running"
    urgent = _command(scheduler, "main study", "pipeline", 4, 60)
    scheduler._schedule()
    asked = [c for c in chunks if c.request == "preempt"]
    assert len(asked) == 1 and urgent.status == "queued"  # one chunk makes room; the other goes on
    scheduler._ended(asked[0], PAUSED)
    assert asked[0].status == "queued" and asked[0].paused_by == "preempt"
    scheduler._schedule()
    assert urgent.status == "running"
    info = scheduler.describe(parent)
    assert info["chunks"]["running"] == 1 and info["chunks"]["queued"] == 3 and info["chunks"]["total"] == 4
    # Pause and Cancel act on the parent's chunks
    scheduler.request("pause", parent.id)
    assert {c.status for c in chunks if c.status != "running"} == {"paused"} and all(c.request == "pause" for c in chunks if c.status == "running")
    scheduler.request("cancel", parent.id)
    for chunk in chunks:
        if chunk.status == "running":
            scheduler._ended(chunk, -15)
    scheduler._update_parents()
    assert parent.status == "cancelled" and {c.status for c in chunks} == {"cancelled"}


def test_an_array_finishes_when_its_chunks_have_and_the_next_stage_waits_for_all_of_them(tmp_path, started):
    project = _tma_project(tmp_path / "p", tmas=2)
    scheduler = queue.Scheduler(tmp_path / "q", budget=queue.Budget(64, 0, 64))
    proposals = scheduler.submit_step(project, "v2_proposals", {"stain": ["AT8"], "workers": 2})
    calls = scheduler.submit_step(project, "v2_calls", {"stain": ["AT8"]})
    assert len(proposals.children) == 2 and len(calls.children) == 2
    scheduler.tick()
    first = [scheduler.jobs[c] for c in proposals.children]
    later = [scheduler.jobs[c] for c in calls.children]
    assert all(c.status == "running" for c in first) and all(c.status == "queued" for c in later)
    scheduler._ended(first[0], 0)
    scheduler.tick()
    assert all(c.status == "queued" for c in later) and "Find candidates" in later[0].waiting
    scheduler._ended(first[1], 0)
    scheduler.tick()
    assert proposals.status == "finished" and all(c.status == "running" for c in later)


def test_fair_share_takes_turns_between_projects(tmp_path, started):
    a, b = _project(tmp_path / "a", "A"), _project(tmp_path / "b", "B")
    scheduler = queue.Scheduler(tmp_path / "q", budget=queue.Budget(4, 0, 64))
    for n in range(6):
        _command(scheduler, f"A{n}", "public", 1, 60, project=a)
    _command(scheduler, "B0", "public", 1, 60, project=b)
    _command(scheduler, "B1", "public", 1, 60, project=b)
    scheduler._schedule()
    assert started == ["A0", "B0", "A1", "B1"]  # not A0-A3: the other project gets its turn


def test_at_most_one_big_public_job_and_two_io_heavy_jobs_per_disk(tmp_path, started):
    scheduler = queue.Scheduler(tmp_path / "q", budget=queue.Budget(64, 0, 64))
    _command(scheduler, "big 1", "public", 2, 7200)
    second = _command(scheduler, "big 2", "public", 2, 7200)
    small = _command(scheduler, "small public", "public", 1, 60)
    heavy = [_command(scheduler, f"io {n}", "pipeline", 1, 60, io="heavy") for n in range(3)]
    for job in heavy:
        job.cwd = str(tmp_path)  # one disk
    scheduler._schedule()
    assert "big 1" in started and "big 2" not in started and "small public" in started
    assert second.waiting == "At most 1 big public datasets job at a time"
    assert started.count("io 0") + started.count("io 1") + started.count("io 2") == 2
    assert heavy[2].waiting == "Waits for its disk: 2 io-heavy jobs use it already"
    assert small.status == "running"


def test_sizes_and_estimates_come_from_resources_and_earlier_runs(tmp_path):
    scheduler = queue.Scheduler(tmp_path / "q", budget=queue.Budget(32, 2, 64))
    project = _project(tmp_path / "p")
    done = scheduler.submit_step(project, "nuclei", {}, array=False)
    done.status, done.runs, done.outcome = "finished", [[1000.0, 1400.0]], {"done": 100, "total": 100}  # 4 s a tile
    scheduler._history_cache = (0.0, None)
    chunk = scheduler.submit_step(project, "nuclei", {}, array=False)
    chunk.units = 30
    assert scheduler.estimate(chunk) == pytest.approx(120.0)
    assert scheduler.describe(chunk)["size"] == "big"  # it needs a GPU
    small = scheduler.submit_command(["true"], "/tmp", "tidy", "models", {"cpus": 1}, time_s=120)
    long = scheduler.submit_command(["true"], "/tmp", "long", "models", {"cpus": 1}, time_s=5 * 3600)
    assert scheduler.describe(small)["size"] == "small" and scheduler.describe(small)["estimate_s"] == 120
    assert scheduler.describe(long)["size"] == "big"
    from stainid.jobs import policy

    assert policy.duration_text(120) == "2 min" and policy.duration_text(5 * 3600) == "5 h"
