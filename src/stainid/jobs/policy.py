"""How the queue decides (`stainid.jobs.queue.Scheduler._schedule`): job sizes and run-time estimates, which jobs may
run side by side, fair share within a class, and the EASY backfill reservation.

- **Size** comes from what a job needs and how long it is expected to take, not from a label: a job is *small* when it
  needs at most 4 processors, no GPU and 8 GB, and is expected to end within 15 minutes; else *big*.
- **Estimates**: the time the job declared (`--time`), else its units (tiles, core images × stains) times the time per
  unit its step took in earlier runs (this project's first, then any project's, then a default), else how long the
  same step took when it last ran to the end, else a default per step.
- **Side by side**: jobs run in parallel whenever their resources fit. Two jobs that write the same outputs (same
  project, an output resource both make, overlapping stains and core images, or one explicit conflict key) never run
  at once; readers overlap freely. At most `io_per_disk` io-heavy jobs use one disk at a time.
- **Fair share**: within a class the next job comes from the project (for annotation work: the project and the person)
  with the fewest jobs running, so one project's 1,000-chunk array does not starve another's small job.
- **EASY backfill**: the first job of each class that waits for room gets a reservation: the time it is expected to
  start (its "shadow time", from the running jobs' expected ends) and what will be left over then. A job further down
  may start now only if it fits now and either ends before that shadow time or uses only what is left over, so it
  never delays a reserved job.
"""
from __future__ import annotations

import math
import os
from pathlib import Path

INF = math.inf
SMALL = {"cpus": 4, "gpus": 0, "ram_gb": 8, "seconds": 15 * 60}
IO_HEAVY = {"dearray", "export", "qc", "tiles", "images", "calibrate", "nuclei", "neun", "fields", "masks", "v2_proposals", "v2_outlines"}
# wall seconds per unit of a whole job, until earlier runs say better: tiles for nuclei, core images × stains for v2
UNIT_SECONDS = {"nuclei": 5.0, "v2_proposals": 45.0, "v2_calls": 2.0, "v2_outlines": 40.0}
STEP_SECONDS = {"slides": 10, "layout": 10, "summarize": 120, "v2_measures": 300, "train": 60, "download_models": 600,
                "prepare_labels": 300, "score_segmentation": 300, "calibrate": 1800, "tiles": 600, "images": 1800}
DEFAULT_CAPS = {"public": {"big": 1}}  # at most this many big jobs (an array counts once) of a class at a time
IO_PER_DISK = 2


def counted(need: dict) -> dict:
    """What a job takes from the budget: a network-only job takes no processors."""
    return {"cpus": 0 if need.get("network") else need.get("cpus", 0), "gpus": need.get("gpus", 0), "ram_gb": need.get("ram_gb", 0)}


KEYS = ("cpus", "gpus", "ram_gb")


def fits(need: dict, free: dict) -> bool:
    return all(need.get(k, 0) <= free.get(k, 0) + 1e-9 for k in KEYS if need.get(k, 0) > 0)


def minus(a: dict, b: dict) -> dict:
    return {k: a.get(k, 0) - b.get(k, 0) for k in KEYS}


def plus(a: dict, b: dict) -> dict:
    return {k: a.get(k, 0) + b.get(k, 0) for k in KEYS}


def size_of(need: dict, seconds: float | None) -> str:
    small = (need.get("cpus", 0) <= SMALL["cpus"] and need.get("gpus", 0) <= SMALL["gpus"] and need.get("ram_gb", 0) <= SMALL["ram_gb"]
             and (seconds is None or seconds <= SMALL["seconds"]))
    return "small" if small else "big"


def io_of(job, need: dict) -> str:
    if need.get("io") in ("light", "heavy"):
        return need["io"]
    return "heavy" if not job.command and job.workflow in IO_HEAVY else "light"


_DISKS: dict[str, int | None] = {}


def disk_of(job) -> int | None:
    """The device the job mostly reads and writes: its project's (else where it runs)."""
    path = job.project or job.cwd
    if not path:
        return None
    if path not in _DISKS:
        try:
            _DISKS[path] = os.stat(path).st_dev
        except OSError:
            _DISKS[path] = None
    return _DISKS[path]


# ---------------------------------------------------------------- what a job writes

_SCOPES: dict[tuple, dict] = {}


def scope_of(job) -> dict:
    """The stains, core images and shard a step job works on (None: all), from its arguments."""
    key = (job.id, tuple(job.argv))
    if key in _SCOPES:
        return _SCOPES[key]
    stains, cores, shard = set(), set(), [0, 1]
    words = list(job.argv)
    for i, word in enumerate(words[:-1]):
        value = words[i + 1]
        if word == "--stain":
            stains.add(value)
        elif word == "--core":
            path = Path(value)
            if not path.is_absolute() and job.project:
                path = Path(job.project) / path
            try:
                cores |= {line.strip() for line in path.read_text().splitlines() if line.strip()} if path.is_file() else {value}
            except OSError:
                cores.add(value)
        elif word == "--shard-index":
            shard[0] = int(value) if value.isdigit() else 0
        elif word == "--shard-count":
            shard[1] = int(value) if value.isdigit() else 1
    option = job.options.get("stain") if isinstance(job.options, dict) else None
    if not stains and option:
        stains = set(option) if isinstance(option, list) else {option}
    if job.options.get("stain") == "all":
        stains = set()
    found = {"stains": frozenset(stains) or None, "cores": frozenset(cores) or None, "shard": tuple(shard) if shard[1] > 1 else None}
    if len(_SCOPES) > 20000:
        _SCOPES.clear()
    _SCOPES[key] = found
    return found


def _overlap(a, b) -> bool:
    return a is None or b is None or bool(a & b)


def scopes_overlap(a: dict, b: dict) -> bool:
    if not _overlap(a["stains"], b["stains"]) or not _overlap(a["cores"], b["cores"]):
        return False
    if a["shard"] and b["shard"] and a["shard"][1] == b["shard"][1] and a["shard"][0] != b["shard"][0]:
        return False  # two parts of one step split the same way
    return True


def writes_of(job) -> set[str]:
    from stainid.workflows.steps import BY_ID

    step = BY_ID.get(job.workflow) if not job.command else None
    return set(step.produces) | set(step.writes) if step is not None else set()


def conflict(job, other) -> bool:
    """Whether the two jobs write the same outputs (they must not run at once)."""
    if set(job.conflicts or []) & set(other.conflicts or []):
        return True
    if job.project != other.project or job.command or other.command:
        return False
    if not writes_of(job) & writes_of(other):
        return False
    return scopes_overlap(scope_of(job), scope_of(other))


# ---------------------------------------------------------------- fair share

def share_key(job) -> tuple:
    return (job.project, job.owner) if job.klass == "annotation" else (job.project,)


def fair_order(queued: list, running: list, key, rank) -> list:
    """The waiting jobs in the order they may start: by class, and within a class taking turns between projects (the
    one with the fewest jobs running first; ties go to the job that has waited longest)."""
    out = []
    by_class: dict[int, dict[tuple, list]] = {}
    for job in sorted(queued, key=key):
        by_class.setdefault(rank(job), {}).setdefault(share_key(job), []).append(job)
    for klass in sorted(by_class):
        groups = by_class[klass]
        load = {share: sum(1 for j in running if rank(j) == klass and share_key(j) == share) for share in groups}
        heads = {share: 0 for share in groups}
        while groups:
            share = min(groups, key=lambda s: (load[s], key(groups[s][heads[s]])))
            out.append(groups[share][heads[share]])
            heads[share] += 1
            load[share] += 1
            if heads[share] == len(groups[share]):
                del groups[share], heads[share]
    return out


# ---------------------------------------------------------------- EASY backfill

def shadow(need: dict, free: dict, ends: list[tuple[float, dict]], now: float) -> tuple[float, dict]:
    """When a job needing `need` could start, given what is free now and when the running jobs release theirs
    (expected end, what it holds), and what would be left over then."""
    avail = dict(free)
    if fits(need, avail):
        return now, minus(avail, need)
    for end, held in sorted(ends, key=lambda e: e[0]):
        avail = plus(avail, held)
        if fits(need, avail):
            return end, {k: max(0.0, v) for k, v in minus(avail, need).items()}
    return INF, {k: 0.0 for k in KEYS}


def duration_text(seconds: float | None) -> str:
    if seconds is None or seconds == INF:
        return ""
    seconds = max(1, int(seconds))
    if seconds < 90:
        return f"{seconds} s"
    if seconds < 5400:
        return f"{round(seconds / 60)} min"
    return f"{seconds / 3600:.1f} h".replace(".0 h", " h")
