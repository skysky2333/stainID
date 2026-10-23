"""How the queue schedules: work-conserving starts, pre-emption, measured memory, conflict keys, job arrays, fair share,
ageing, the watchdog, and that any mix of jobs drains."""
from __future__ import annotations

import random
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


def test_a_restart_of_the_app_leaves_a_running_array_and_the_steps_after_it_waiting(tmp_path, started, monkeypatch):
    import os

    project = _tma_project(tmp_path / "p", tmas=2)
    first = queue.Scheduler(tmp_path / "q", budget=queue.Budget(64, 0, 64), launcher=queue.ProcessLauncher())
    proposals = first.submit_step(project, "v2_proposals", {"stain": ["AT8"], "workers": 2})
    calls = first.submit_step(project, "v2_calls", {"stain": ["AT8"]})
    first.tick()
    chunks = [first.jobs[c] for c in proposals.children]
    assert all(c.status == "running" for c in chunks)
    for chunk in chunks:  # running in their own units, started a while ago
        chunk.started = time.time() - 600
        first._save(chunk)
        Path(first.files(chunk)["pid"]).write_text(f"{os.getpid()} {os.getpid()}")
    first.tick()
    first.close()
    monkeypatch.setattr(queue, "alive", lambda pid, since=None: True)
    second = queue.Scheduler(tmp_path / "q", budget=queue.Budget(64, 0, 64), launcher=queue.ProcessLauncher())
    assert second.leader
    for _ in range(10):
        second.tick()
    parent = second.jobs[proposals.id]
    assert parent.status == "running" and parent.attempts == 0 and not parent.waiting  # its state is its chunks'
    assert {second.jobs[c].status for c in calls.children} == {"queued"}  # still waiting, not skipped
    for chunk_id in proposals.children:
        second._ended(second.jobs[chunk_id], 0)
    second.tick()
    assert second.jobs[proposals.id].status == "finished" and {second.jobs[c].status for c in calls.children} == {"running"}


def test_fair_share_takes_turns_between_projects(tmp_path, started):
    a, b = _project(tmp_path / "a", "A"), _project(tmp_path / "b", "B")
    scheduler = queue.Scheduler(tmp_path / "q", budget=queue.Budget(4, 0, 64))
    for n in range(6):
        _command(scheduler, f"A{n}", "public", 1, 60, project=a)
    _command(scheduler, "B0", "public", 1, 60, project=b)
    _command(scheduler, "B1", "public", 1, 60, project=b)
    scheduler._schedule()
    assert started == ["A0", "B0", "A1", "B1"]  # not A0-A3: the other project gets its turn


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


def _memory(scheduler, monkeypatch, total=53.0, available=40.0, app=3.0):
    """The scheduler looks at a computer with this memory (GB) and no GPUs used outside the queue."""
    found = {"total_gb": total, "available_gb": available, "app_gb": app}
    scheduler.watch_outside = True
    monkeypatch.setattr(scheduler, "memory", lambda: found)
    monkeypatch.setattr(scheduler, "outside", lambda: {"ram_gb": found["available_gb"], "gpus": set()})
    return found


def _measured(scheduler, job, gb, minutes=10):
    job.started = time.time() - minutes * 60
    scheduler._ram_seen[job.id] = (job.started, gb, gb)


def test_nothing_stays_idle_while_a_job_that_fits_waits(tmp_path, started):
    scheduler = queue.Scheduler(tmp_path / "q", budget=queue.Budget(12, 0, 64))
    _command(scheduler, "running", "pipeline", 4, 100)
    scheduler._schedule()
    big = _command(scheduler, "big main-study job", "pipeline", 10, 1000)  # does not fit, and nothing lower runs to pause
    later = [_command(scheduler, f"public {n}", "public", 2, 500) for n in range(4)]
    scheduler._schedule()
    assert started == ["running", "public 0", "public 1", "public 2", "public 3"]  # the free processors are used, whatever the class
    assert big.waiting.startswith("Waits for room: needs 10 processors") and all(j.status == "running" for j in later)
    first = next(j for j in scheduler.jobs.values() if j.title == "running")
    first.status = "finished"  # the pipeline job ends: the big one pauses as few public jobs as make room
    scheduler._schedule()
    asked = [j for j in later if j.request == "preempt"]
    assert big.status == "queued" and len(asked) == 3  # 12 processors, 4 free: 3 of the 4 public jobs (2 each) pause
    for job in asked:
        scheduler._ended(job, PAUSED)
    scheduler._schedule()
    assert big.status == "running" and started[-1] == "big main-study job"
    assert {j.status for j in later} == {"running", "queued"} and sum(j.status == "running" for j in later) == 1


def test_an_idle_gpu_gets_the_waiting_gpu_job_even_a_public_one(tmp_path, started):
    scheduler = queue.Scheduler(tmp_path / "q", budget=queue.Budget(32, 2, 48))
    _command(scheduler, "SEA-AD calibrate", "public", 8, 7200)
    _command(scheduler, "fusion check", "models", 6, 3600)
    _command(scheduler, "Tangle-Tracer find candidates", "public", 12, 7200, ram_gb=40)  # queued first, takes the memory
    nuclei = scheduler.submit_command(["true"], "/tmp", "SEA-AD find nuclei", "public", {"cpus": 4, "gpus": 1, "ram_gb": 8}, time_s=7200)
    second = scheduler.submit_command(["true"], "/tmp", "SEA-AD find nuclei 2", "public", {"cpus": 4, "gpus": 1, "ram_gb": 8}, time_s=7200)
    scheduler._schedule()
    assert nuclei.status == "running" and second.status == "running" and sorted(nuclei.gpus + second.gpus) == [0, 1]
    assert [j.status for j in scheduler.jobs.values() if j.title == "Tangle-Tracer find candidates"] == ["queued"]


def test_a_high_job_needing_memory_pauses_the_right_low_jobs_and_they_resume_afterwards(tmp_path, started, monkeypatch):
    scheduler = queue.Scheduler(tmp_path / "q", budget=queue.Budget(32, 0, 999))
    _memory(scheduler, monkeypatch, total=53, available=40, app=3)  # 53 - 3 - 4 headroom - 3 reserve = 43 GB for jobs
    low = {name: _command(scheduler, name, klass, 2, ram_gb=gb) for name, klass, gb in (("public small", "public", 3), ("public large", "public", 12),
                                                                                       ("models large", "models", 14))}
    scheduler._schedule()
    for name, gb in (("public small", 3), ("public large", 12), ("models large", 14)):
        _measured(scheduler, low[name], gb)
    scheduler._memory = (0.0, {})
    found = scheduler.memory()
    found["available_gb"] = 53 - 3 - 3 - 12 - 14 - 3  # what they use, measured
    assert scheduler.current_budget().ram_gb == 43
    main = scheduler.submit_command(["true"], "/tmp", "main study", "pipeline", {"cpus": 4, "ram_gb": 20})
    scheduler._schedule()
    assert main.status == "queued" and main.waiting == "Starts as soon as lower-priority work has paused"
    assert [n for n, j in low.items() if j.request == "preempt"] == ["public large"]  # lowest class, largest holder, one pause
    scheduler._ended(low["public large"], PAUSED)
    found["available_gb"] += 12
    scheduler._schedule()
    assert main.status == "running" and low["public large"].status == "queued"
    assert low["public small"].status == "running" and low["models large"].status == "running"
    main.status = "finished"  # once the main study's job is done, the paused one goes on where it stopped
    scheduler._schedule()
    assert low["public large"].status == "running" and low["public large"].paused_by == "preempt"


def test_memory_is_measured_so_declared_but_unused_memory_does_not_block_work(tmp_path, started, monkeypatch):
    scheduler = queue.Scheduler(tmp_path / "q", budget=queue.Budget(32, 0, 999))
    found = _memory(scheduler, monkeypatch, total=53, available=45, app=3)
    checks = [scheduler.submit_command(["true"], "/tmp", f"fusion check {n}", "models", {"cpus": 2, "ram_gb": 9}) for n in range(4)]
    scheduler._schedule()
    assert all(j.status == "running" for j in checks)  # 36 of 43 GB declared
    for job in checks:
        _measured(scheduler, job, 2.0)  # they use 2 GB each
    found["available_gb"] = 37.0
    other = scheduler.submit_command(["true"], "/tmp", "SEA-AD calibrate", "public", {"cpus": 4, "ram_gb": 16})
    scheduler._schedule()
    assert other.status == "running"  # counted: 4 x 2.2 GB, not 4 x 9 GB
    warm = scheduler.submit_command(["true"], "/tmp", "just started", "public", {"cpus": 1, "ram_gb": 8})
    scheduler._schedule()  # available 37 - 4 headroom - 2 margin - 16 (calibrate, not measured yet) = 15 GB: it fits
    assert warm.status == "running"
    too_much = scheduler.submit_command(["true"], "/tmp", "one more", "public", {"cpus": 1, "ram_gb": 8})
    scheduler._schedule()  # 15 - 8 = 7 GB left of what is available
    assert too_much.status == "queued" and "GB of memory" in too_much.waiting


def test_memory_pressure_pauses_the_lowest_class_jobs_at_once(tmp_path, started, monkeypatch):
    scheduler = queue.Scheduler(tmp_path / "q", budget=queue.Budget(32, 0, 999))
    found = _memory(scheduler, monkeypatch, total=53, available=30, app=3)
    jobs = {name: _command(scheduler, name, klass, 2) for name, klass in (("annotation", "annotation"), ("pipeline", "pipeline"),
                                                                          ("public a", "public"), ("public b", "public"))}
    scheduler._schedule()
    for name, gb in (("annotation", 2), ("pipeline", 6), ("public a", 1), ("public b", 5)):
        _measured(scheduler, jobs[name], gb)
    found["available_gb"], found["app_gb"] = 3.0, 14.0  # many annotators: the app grew, less than its headroom is left
    scheduler._schedule()
    assert [n for n, j in jobs.items() if j.request] == ["public b"]  # 3 GB short (6 - 3): the largest public job is enough
    assert "public b" in [j.title for j in jobs.values() if j.id in scheduler._urgent]
    scheduler._ended(jobs["public b"], PAUSED)
    assert jobs["public b"].status == "queued" and jobs["public b"].waiting.startswith("Paused to keep memory free")
    assert scheduler.current_budget().ram_gb == 53 - 14 - 4 - 3


def test_jobs_that_wait_in_a_circle_are_refused_and_too_big_ones_capped(tmp_path, started):
    scheduler = queue.Scheduler(tmp_path / "q", budget=queue.Budget(8, 1, 32))
    two = scheduler.submit_command(["true"], "/tmp", "two GPUs", "models", {"gpus": 2})
    assert two.needs["gpus"] == 1 and two.warning == "asks for 2 GPUs; at most 1 GPUs can be given to jobs on this computer: it runs with 1 GPUs"
    with pytest.raises(ValueError, match="no job nowhere"):
        scheduler.submit_command(["true"], "/tmp", "after nothing", "models", {"cpus": 1}, after=["nowhere"])
    a = scheduler.submit_command(["true"], "/tmp", "a", "models", {"cpus": 1})
    b = scheduler.submit_command(["true"], "/tmp", "b", "models", {"cpus": 1}, after=[a.id])
    a.after = [b.id]  # a record edited by hand into a circle
    with pytest.raises(ValueError, match="cycle"):
        scheduler.check(queue.Job(id=a.id, workflow="command", options={}, argv=["true"], after=[b.id]))
    c = scheduler.submit_command(["true"], "/tmp", "c", "models", {"cpus": 1}, after=[b.id])
    a.after = []
    scheduler._schedule()
    scheduler._ended(a, 1)  # a fails: b and what waits for b are not started, with the reason
    scheduler._schedule()
    scheduler._schedule()
    assert b.status == "skipped" and "“a” failed" in b.outcome["error"] and c.status == "skipped"


def test_a_job_that_asks_for_more_memory_than_jobs_can_ever_have_is_capped_and_runs(tmp_path, started, monkeypatch):
    """54 GB, the app 3.3 and 45 available: at most 45 - 4 headroom - 2 margin = 39 GB for jobs. A job declaring 40 GB
    used to wait for ever ("held by work of the same or a higher priority") on an idle computer."""
    scheduler = queue.Scheduler(tmp_path / "q", budget=queue.Budget(32, 2, 999))
    found = _memory(scheduler, monkeypatch, total=54, available=45, app=3.3)
    assert scheduler.machine()["ram_gb"] == pytest.approx(39.0)
    big = scheduler.submit_command(["true"], "/tmp", "v2b stage 2 proposals", "pipeline", {"cpus": 16, "ram_gb": 40})
    assert big.needs["ram_gb"] == 39.0
    assert big.warning == "asks for 40 GB of memory; at most 39 GB can be given to jobs on this computer: it runs with 39 GB of memory"
    assert scheduler.store.load(scheduler.store.path(big.id)).warning == big.warning  # kept with the record
    from stainid.jobs.cli import _line

    shown = scheduler.describe(big)  # what `jobs list` and the Run page show (app-started jobs alike)
    assert shown["warning"] == big.warning and "Warning: it asks for 40 GB of memory" in _line(shown)
    scheduler._schedule()
    assert big.status == "running"
    # queued while the computer had more, then other programs took memory: capped again once the queue is idle
    scheduler2 = queue.Scheduler(tmp_path / "q2", budget=queue.Budget(32, 2, 999))
    found = _memory(scheduler2, monkeypatch, total=54, available=45, app=3.3)
    later = scheduler2.submit_command(["true"], "/tmp", "proposals", "pipeline", {"cpus": 16, "ram_gb": 38})
    assert not later.warning
    found["available_gb"] = 43.0  # 37 GB for jobs now
    scheduler2._schedule()
    assert later.status == "running" and later.needs["ram_gb"] == 37.0 and "at most 37 GB can be given to jobs" in later.warning
    # a running job that is not measured yet counts with what it declared: a submit meanwhile is not capped by it
    found["available_gb"] = 20.0
    other = scheduler2.submit_command(["true"], "/tmp", "another", "pipeline", {"cpus": 1, "ram_gb": 30})
    assert not other.warning and other.needs["ram_gb"] == 30


def test_a_wait_for_room_names_who_holds_it(tmp_path, started, monkeypatch):
    scheduler = queue.Scheduler(tmp_path / "q", budget=queue.Budget(8, 0, 999))
    found = _memory(scheduler, monkeypatch, total=54, available=45, app=3.3)
    first = scheduler.submit_command(["true"], "/tmp", "first", "pipeline", {"cpus": 6, "ram_gb": 2})
    scheduler._schedule()
    second = scheduler.submit_command(["true"], "/tmp", "second", "pipeline", {"cpus": 4, "ram_gb": 2})
    scheduler._schedule()
    assert first.status == "running" and second.waiting == "Waits for room: needs 4 processors (2 free) (held by work of the same or a higher priority)"
    idle = queue.Scheduler(tmp_path / "q2", budget=queue.Budget(8, 0, 999))
    found = _memory(idle, monkeypatch, total=54, available=45, app=3.3)
    third = idle.submit_command(["true"], "/tmp", "third", "pipeline", {"cpus": 1, "ram_gb": 30})
    found["available_gb"] = 12.0  # nothing of the queue runs; other programs took the memory
    monkeypatch.setattr(idle, "cap", lambda job: "")  # (the idle cap is tested above: here only the wording)
    idle._schedule()
    assert third.status == "queued" and "held by" not in third.waiting
    assert third.waiting == ("Waits for room: needs 30 GB of memory (6 free) "
                             "(the rest of the memory is used by the app and other programs on this computer, not by queued work)")


def test_a_job_that_waited_long_goes_first_and_may_pause_newer_jobs_of_its_class(tmp_path, started):
    scheduler = queue.Scheduler(tmp_path / "q", budget=queue.Budget(8, 0, 64))
    big = _command(scheduler, "big", "public", 8, 3600)
    big.created -= queue.AGE_S + 60
    scheduler._ready_since[big.id] = time.time() - queue.AGE_S - 60  # it has waited half an hour while small ones kept coming
    smalls = []
    for n in range(4):
        job = _command(scheduler, f"small {n}", "public", 2, 600)
        smalls.append(job)
    for job in smalls:
        job.status, job.started = "running", time.time()  # started after the big job was queued
    scheduler._schedule()
    assert big.status == "queued" and sum(j.request == "preempt" for j in smalls) == 4
    for job in smalls:
        scheduler._ended(job, PAUSED)
    newer = _command(scheduler, "newest small", "public", 2, 60)
    scheduler._schedule()
    assert big.status == "running" and newer.status == "queued"  # the old job first; the new small one waits its turn


def test_room_being_made_is_not_taken_by_lower_jobs_and_the_watchdog_reports_a_long_wait(tmp_path, started):
    scheduler = queue.Scheduler(tmp_path / "q", budget=queue.Budget(8, 0, 64))
    low = _command(scheduler, "public", "public", 6, 3600)
    scheduler._schedule()
    main = _command(scheduler, "main", "pipeline", 8, 600)
    scheduler._schedule()
    assert low.request == "preempt"
    other = _command(scheduler, "another public", "public", 2, 60)
    scheduler._schedule()
    assert other.status == "queued"  # the 2 free processors are the main job's while the public one stops
    assert scheduler.overview()["alerts"] == []
    scheduler._making_room[main.id] -= queue.WATCHDOG_S + 1  # the public job never stops
    scheduler._schedule()
    assert [a["id"] for a in scheduler.overview()["alerts"]] == [main.id]


@pytest.mark.parametrize("seed", range(8))
def test_any_mix_of_jobs_with_dependencies_drains_completely(tmp_path, started, seed):
    rng = random.Random(seed)
    scheduler = queue.Scheduler(tmp_path / "q", budget=queue.Budget(8, 2, 32, io_per_disk=1))
    work, made = {}, []
    for n in range(40):
        after = rng.sample([j.id for j in made], k=min(len(made), rng.choice([0, 0, 1, 2])))
        needs = {"cpus": rng.choice([0.5, 1, 2, 4, 8]), "gpus": rng.choice([0, 0, 0, 1, 2]), "ram_gb": rng.choice([1, 4, 8, 16, 32]),
                 "io": rng.choice(["light", "heavy"])}
        job = scheduler.submit_command(["true"], str(tmp_path), f"job {n}", rng.choice(queue.CLASSES), needs, after=after,
                                       conflicts=[rng.choice(["x", "y", "z"])] if rng.random() < 0.3 else [], time_s=rng.choice([0, 60, 3600]))
        made.append(job)
        work[job.id] = rng.randint(1, 6)  # rounds of work
    for round_ in range(3000):
        for job in [j for j in scheduler.jobs.values() if j.status == "running"]:
            if job.request:
                scheduler._ended(job, PAUSED)  # stops at its unit boundary; its done work is kept
                continue
            work[job.id] -= 1
            if work[job.id] <= 0:
                scheduler._ended(job, 1 if rng.random() < 0.05 else 0)
        scheduler._schedule()
        if all(j.status in queue.FINAL for j in scheduler.jobs.values()):
            break
        if rng.random() < 0.02:  # now and then a job comes in late, of any class
            late = scheduler.submit_command(["true"], str(tmp_path), "late", rng.choice(queue.CLASSES), {"cpus": rng.choice([1, 8]), "ram_gb": 4})
            work[late.id] = rng.randint(1, 3)
    states = {j.status for j in scheduler.jobs.values()}
    assert states <= {"finished", "failed", "skipped"}, (round_, [(j.title, j.status, j.waiting) for j in scheduler.jobs.values() if j.status not in queue.FINAL])


def _using(scheduler, job, cpus, minutes=10):
    """The job has run `minutes` and used `cpus` processors on average over the last minute."""
    now = time.time()
    job.started = now - minutes * 60
    scheduler._cpu_seen[job.id] = (job.started, [(now - 60, 1000.0), (now, 1000.0 + 60 * cpus)])


def test_processors_are_counted_as_used_so_a_job_that_declares_many_and_uses_few_blocks_no_one(tmp_path, started, monkeypatch):
    scheduler = queue.Scheduler(tmp_path / "q", budget=queue.Budget(32, 0, 999))
    _memory(scheduler, monkeypatch, total=200, available=180, app=1)
    check = _command(scheduler, "cohort check", "models", 8)
    rebuild = _command(scheduler, "rebuild", "pipeline", 24)
    scheduler._schedule()
    assert check.status == "running" and rebuild.status == "running"
    test = _command(scheduler, "scaling test", "models", 28)
    scheduler._schedule()
    assert test.status == "queued" and "28 processors (0 free)" in test.waiting  # both just started: they count what they declared
    _using(scheduler, check, 1.0)
    _using(scheduler, rebuild, 1.0)
    scheduler._schedule()
    assert test.status == "running"  # 32 - 2 x max(0.5, 1.25) = 29.5 free
    used = scheduler.overview()["used"]
    assert used["cpus_declared"] == 60 and used["cpus_now"] == 2.0 and used["cpus"] == 28 + 2.5
    from stainid.jobs.cli import _resources

    assert _resources(scheduler.overview()).startswith("Processors: 2 in use, 60 declared, 30.5 counted of 32")
    # a job never counts more than it declared, however much it uses
    _using(scheduler, check, 20.0)
    assert scheduler._holds(check)["cpus"] == 8 and scheduler._holds(check)["cpus_now"] == 20.0


def test_when_a_job_grows_into_what_it_declared_the_lowest_class_pauses(tmp_path, started, monkeypatch):
    scheduler = queue.Scheduler(tmp_path / "q", budget=queue.Budget(32, 0, 999))
    _memory(scheduler, monkeypatch, total=200, available=180, app=1)
    main = _command(scheduler, "main study", "pipeline", 24)
    scheduler._schedule()
    _using(scheduler, main, 2.0)
    models = _command(scheduler, "model check", "models", 20)
    public = _command(scheduler, "public run", "public", 8)
    scheduler._schedule()
    assert models.status == "running" and public.status == "running"
    _using(scheduler, models, 20.0, minutes=5)
    _using(scheduler, public, 8.0, minutes=4)
    _using(scheduler, main, 24.0)  # the main study grows into its 24: 52 processors wanted of 32
    scheduler._schedule()
    assert not any(j.request for j in (main, models, public))  # not at once: it has to last a minute
    scheduler._cpu_over_since = time.time() - queue.CPU_OVER_S - 1
    scheduler._schedule()
    assert public.request == "preempt" and models.request == "preempt" and not main.request  # lowest class first, until 24 fit
    scheduler._ended(public, PAUSED)
    scheduler._ended(models, PAUSED)
    _using(scheduler, main, 24.0)
    scheduler._schedule()
    assert public.status == "running" and models.status == "queued"  # 8 fit next to the 24 again; 20 do not
