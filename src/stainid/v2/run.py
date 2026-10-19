"""Running the v2 stages: which core images each stage still has to do (missing or outdated), spread over processes.

Every stage is resumable per core image (a core is done once its manifest is written) and redoes a core whose manifest no
longer matches what it would be made from now (`stainid.v2.manifest`). Stage 2 runs `workers` core images at once, one
process each (a core image is decoded whole: about 0.8 GB, plus about 1 GB while a tile is worked on), stage 3 one
process per GPU (`--shard-index i --shard-count 2` with CUDA_VISIBLE_DEVICES), stages 4 and 5 are quick.
"""
from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context

from stainid import cache
from stainid.jobs import units
from stainid.tables import claim_field
from stainid.v2 import manifest
from stainid.v2.params import STAINS
from stainid.workflows import calibration_table, tiles_for

STAGES = ("proposals", "calls", "outlines", "measures")  # in the order a full run does them
GB = 2**30


def default_workers(per_worker_gb: float = 2.2, reserve_gb: float = 4.0) -> int:
    """Core images at once: bounded by the processors and by the memory free now."""
    try:
        with open("/proc/meminfo") as handle:
            info = {line.split(":")[0]: int(line.split()[1]) * 1024 for line in handle}
        available = info.get("MemAvailable", 8 * GB) / GB
    except OSError:
        available = 8.0
    return max(1, min((os.cpu_count() or 2) // 2, int((available - reserve_gb) / per_worker_gb)))


def groups(project, stains, cores: set[str] | None = None) -> dict[tuple[str, str], list[dict]]:
    return tiles_for(project.analysis_tiles(), set(stains), core_ids=cores)


def _claim(project, stage: str, stain: str, core_id: str):
    folder = project.output("v2") / "claims"
    folder.mkdir(parents=True, exist_ok=True)
    claim = folder / f"{stage}_{stain}_{core_id}.claim"
    return claim if claim_field(claim) else None


# ---------------------------------------------------------------- what is to do

def proposal_inputs(project, fresh: bool = True):
    """The slide thresholds and the names of the nuclei files (empty before they exist); `fresh` lists the nuclei folder
    now (a run), else at most every 30 s (the app's status: the folder holds a file per tile). No exclusions: stage 2
    proposes on all tissue and the excluded areas are applied downstream."""
    calibration = calibration_table(project.input("calibration")) if project.input("calibration").exists() else {}
    nuclei = cache.names(project.output("nuclei"), every=0 if fresh else 30)
    return calibration, nuclei


def todo(project, stage: str, stains, cores: set[str] | None = None, redo: bool = False) -> list[tuple[str, str, list[dict], str]]:
    """(core_id, stain, tiles, why) of every core image the stage has to (re)do: why is "missing" or "outdated"."""
    from stainid.v2 import classify, outlines, proposals

    out = []
    found = groups(project, stains, cores)
    if stage == "proposals":
        calibration, nuclei = proposal_inputs(project)
    models = {}
    since = units.since() if redo else None  # a redo that was paused: what it redid already stays
    for (core_id, stain), tiles in found.items():
        recorded = manifest.read_all(project, stage, stain).get(core_id)
        if stage == "proposals":
            threshold = calibration.get((tiles[0]["tma"], stain))
            if threshold is None:
                continue
            made = proposals.expected(project, stain, core_id, tiles, threshold, nuclei)
        elif stage == "calls":
            if stain not in models:
                models[stain] = classify.load_model(project, stain)
            made = classify.expected(project, stain, core_id, models[stain])
        else:
            made = outlines.expected(project, stain, core_id)
        if made is None:  # its upstream stage has not run
            continue
        if since is not None and recorded is not None and _written(recorded) >= since and manifest.up_to_date(recorded, made):
            continue
        if redo or recorded is None:
            out.append((core_id, stain, tiles, "missing" if recorded is None else "redo"))
        elif not manifest.up_to_date(recorded, made):
            out.append((core_id, stain, tiles, "outdated"))
    return out


def _written(recorded: dict) -> float:
    try:
        return time.mktime(time.strptime(recorded.get("written", ""), "%Y-%m-%d %H:%M:%S"))
    except (TypeError, ValueError):
        return 0.0


def status(project) -> dict:
    """Per stage and stain: core images done and up to date, outdated (made, but an input, a parameter or an upstream stage
    changed since, also when that upstream stage has not been re-run yet), and the total. The parameters, the model and
    the weights are looked at once per stain and the manifests read in one pass per folder (`manifest.read_all`), not
    once per core: a project of 15 000 core images is counted in seconds."""
    from stainid.v2 import classify, outlines, proposals
    from stainid.v2.params import stage_params

    out = {stage: {} for stage in ("proposals", "calls", "outlines", "candidates")}
    if not project.analysis_tiles().exists():
        return {stage: {stain: {"done": 0, "outdated": 0, "total": 0} for stain in STAINS} for stage in out}
    found = groups(project, STAINS)
    calibration, nuclei = proposal_inputs(project, fresh=False)
    sam = manifest.file_signature(project.model("sam"))
    for stain in STAINS:
        keys = sorted(core for core, s in found if s == stain)
        recorded = {stage: manifest.read_all(project, stage, stain) for stage in out}
        settings = {stage: stage_params(project, stage, stain) for stage in out} if keys else {}
        model, model_looked = None, False
        counts = {stage: {"done": 0, "outdated": 0, "total": len(keys)} for stage in out}
        for core in keys:
            tiles = found[(core, stain)]
            threshold = calibration.get((tiles[0]["tma"], stain))
            mine_all = {stage: recorded[stage].get(core) for stage in out}
            stale = False
            for stage in ("proposals", "calls", "outlines", "candidates"):
                mine = mine_all[stage]
                if mine is None:
                    if stage != "outlines":
                        stale = True  # what depends on a missing stage is not up to date either
                    continue
                if stage == "proposals":
                    made = None if threshold is None else proposals.expected(project, stain, core, tiles, threshold, nuclei, settings[stage])
                elif stage == "calls":
                    if not model_looked:
                        model_looked = True
                        model = classify.load_model(project, stain) if project.model(classify.MODEL_KEY[stain]).exists() else None
                    made = classify.expected(project, stain, core, model, settings[stage], mine_all["proposals"]) if model is not None else None
                elif stage == "outlines":
                    made = outlines.expected(project, stain, core, settings[stage], mine_all["proposals"], mine_all["calls"], sam)
                else:
                    made = classify.assembled_expected(project, stain, core, settings[stage], mine_all)
                stale = stale or made is None or not manifest.up_to_date(mine, made)
                counts[stage]["outdated" if stale else "done"] += 1
        for stage in out:
            out[stage][stain] = counts[stage]
    return out


def seconds_per_core(project) -> dict[str, dict[str, float]]:
    """Per stage and stain: the median seconds a core image took (the manifests record it), where at least 3 were
    made; for the Home page's estimate of what is left. Kept on disk while the manifests stay the same."""
    from stainid import disk_cache  # noqa: PLC0415

    def make() -> dict:
        out: dict[str, dict[str, float]] = {}
        for stage in ("proposals", "calls", "outlines"):
            for stain in STAINS:
                spent = sorted(float(m["seconds"]) for m in manifest.read_all(project, stage, stain).values()
                               if m and isinstance(m.get("seconds"), (int, float)))
                if len(spent) >= 3:
                    out.setdefault(stage, {})[stain] = round(spent[len(spent) // 2], 2)
        return out

    folders = [manifest.root(project) / "manifests" / stage / stain for stage in ("proposals", "calls", "outlines") for stain in STAINS]
    return cache.memo(("v2 seconds per core", str(project.root)), folders, make, every=60.0, folders=True,
                      persist=disk_cache.answer_file(project.root, "v2-seconds.pkl"), background=True)


def status_inputs(project) -> list:
    """What `status` is made from, for keeping it until one of them changes: every manifest folder, the tiles, the slide
    thresholds, the settings, the models and the Segment Anything weights."""
    from stainid.v2 import classify

    folders = [manifest.root(project) / "manifests" / stage / stain for stage in manifest.STAGES for stain in STAINS]
    models = [project.model(key) for key in sorted(set(classify.MODEL_KEY.values()))] + [project.model("sam")]
    config = project.config_file
    return [*folders, project.analysis_tiles(), project.input("calibration"), config, *models,
            project.output("nuclei")]


def cached_status(project) -> dict:
    """`status`, made once while its inputs stay the same (looked at no more often than every 10 s while it is slow to
    make); parallel callers wait for the one making it."""
    from stainid import disk_cache  # noqa: PLC0415

    return cache.memo(("v2 status", str(project.root)), status_inputs(project), lambda: status(project), every=10.0, folders=True,
                      persist=disk_cache.answer_file(project.root, "v2-status.pkl"), background=True)  # a server started again: no 5,000 manifests read


# ---------------------------------------------------------------- stage 2

def default_threads(workers: int) -> int:
    """Filter threads per stage-2 process: the processors the processes leave idle, shared out (at most 4)."""
    return max(1, min(4, (os.cpu_count() or 2) // max(1, workers)))


def _propose_worker(root: str, stain: str, core_id: str, tiles: list[dict], threshold: float, nuclei: frozenset,
                    threads: int = 1, tiles_at_once: int = 1) -> dict:
    import cv2

    from stainid.imaging import parallel
    from stainid.project import load_project
    from stainid.v2.proposals import propose_core

    cv2.setNumThreads(2)
    parallel.threads(threads)
    project = load_project(root)
    claim = _claim(project, "proposals", stain, core_id)
    if claim is None:
        return {"skipped": True}
    try:
        return propose_core(project, stain, core_id, tiles, threshold, nuclei, tiles_at_once)
    finally:
        claim.unlink(missing_ok=True)


def _limited_threads() -> None:
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(key, "2")


PROCESS_GB = 1.3  # a tile process (its fields and pictures; up to about 2 GB on a 6E10 tile with local windows)
CORE_GB = 1.4  # a core image in flight: its picture and core-wide masks in shared memory
# tile processes per logical processor: stage 2 is bound by memory bandwidth, which 8 processes fill on the 32-processor
# study machine (5950X; the same throughput from 8 to 16 processes, less beyond; 2 per physical core no faster; see
# docs/pipeline_v2.md, "Speed"), so more would only take processors from other work
PROCESSES_PER_CPU = 0.25


def default_processes(cores_at_once: int, threads: int = 1, reserve_gb: float = 4.0) -> int:
    """Tile processes for stage 2 (`threads` threads each): bounded by the processors and the memory a queue job was
    given (STAINID_JOB_CPUS, STAINID_JOB_RAM_GB), else by the processors and the memory free now."""
    cpus = int(os.environ.get("STAINID_JOB_CPUS") or 0) or round(PROCESSES_PER_CPU * (os.cpu_count() or 2))
    ram = float(os.environ.get("STAINID_JOB_RAM_GB") or 0)
    if not ram:
        try:
            with open("/proc/meminfo") as handle:
                info = {line.split(":")[0]: int(line.split()[1]) * 1024 for line in handle}
            available = info.get("MemAvailable", 8 * GB) / GB
        except OSError:
            available = 8.0
        ram = available - reserve_gb
    return max(1, min(cpus // max(1, threads), int((ram - 1.0 - cores_at_once * CORE_GB) / PROCESS_GB)))


def run_proposals(project, stains=STAINS, cores: set[str] | None = None, workers: int | None = None, shard_index: int = 0,
                  shard_count: int = 1, redo: bool = False, threads: int | None = None, tiles_at_once: int = 1,
                  processes: int | None = None) -> None:
    """Stage 2 on every core image to do. With `processes` (> 0; None: `default_processes`), one pool of that many tile
    processes works on the tiles of `workers` core images at once (default 3: one is decoded or written while the
    others' tiles keep every process busy), each process with `threads` threads (default 1). With `processes=0`, the older layout: `workers`
    core images at once in a process each, with `threads` filter threads (`stainid.imaging.parallel`) and `tiles_at_once`
    tiles in threads (sharing the decoded core image). The results depend on none of them."""
    jobs = todo(project, "proposals", stains, cores, redo)[shard_index::shard_count]
    calibration, nuclei = proposal_inputs(project)
    if processes != 0:
        _run_proposals_pooled(project, jobs, calibration, nuclei, workers or 3, processes, threads or 1)
        return
    workers = workers or default_workers(2.2 + 1.0 * (tiles_at_once - 1))
    threads = threads or default_threads(workers * tiles_at_once)
    print(f"[0/{len(jobs)}] stage 2 (proposals): {len(jobs)} core images to do with {workers} at once ({threads} filter threads and "
          f"{tiles_at_once} tile(s) at once each)", flush=True)
    if not jobs:
        return
    _limited_threads()
    started = time.time()
    with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as pool:
        futures = {pool.submit(_propose_worker, str(project.root), stain, core, tiles, calibration[(tiles[0]["tma"], stain)],
                               nuclei, threads, tiles_at_once): (core, stain, why)
                   for core, stain, tiles, why in jobs}
        for number, future in enumerate(as_completed(futures), start=1):
            core, stain, why = futures[future]
            units.drain(futures)  # asked to stop: the core images not started yet wait for the next run
            if future.cancelled():
                continue
            try:
                result = future.result()
            except Exception as error:  # noqa: BLE001 - one bad core must not stop the others
                print(f"[{number}/{len(jobs)}] {core} {stain}: failed: {type(error).__name__}: {error}", flush=True)
                continue
            if result.get("skipped"):
                print(f"[{number}/{len(jobs)}] {core} {stain}: taken by another process", flush=True)
            else:
                print(f"[{number}/{len(jobs)}] {core} {stain} ({why}): {result['candidates']} candidates, {result['unproposed']} unproposed, "
                      f"{result['seconds']:.0f} s ({(time.time() - started) / number:.0f} s per core image so far)", flush=True)
    units.boundary("stage 2")


class _TilePool:
    """The tile processes of a run, made again if one dies (killed for memory: the pool is then broken)."""

    def __init__(self, processes: int, threads: int = 1):
        from stainid.v2.proposals import tile_pool

        self.processes, self.threads, self.lock = processes, threads, threading.Lock()
        self.pool = tile_pool(processes, threads)

    def renew(self, broken) -> None:
        from stainid.v2.proposals import tile_pool

        with self.lock:
            if self.pool is broken:
                broken.shutdown(wait=False, cancel_futures=True)
                self.pool = tile_pool(self.processes, self.threads)

    def shutdown(self) -> None:
        self.pool.shutdown()


def _run_proposals_pooled(project, jobs: list, calibration: dict, nuclei: frozenset, cores_at_once: int, processes: int | None,
                          threads: int = 1) -> None:
    from concurrent.futures import ThreadPoolExecutor
    from concurrent.futures.process import BrokenProcessPool

    from stainid.imaging import shared_arrays
    from stainid.v2.proposals import propose_core

    processes = processes or default_processes(cores_at_once, threads)
    print(f"[0/{len(jobs)}] stage 2 (proposals): {len(jobs)} core images to do, {cores_at_once} at once, their tiles in {processes} "
          f"tile processes ({threads} thread{'s' if threads > 1 else ''} each)", flush=True)
    if not jobs:
        return
    shared_arrays.clean_stale()
    tiles_pool = _TilePool(processes, threads)

    def one(core: str, stain: str, tiles: list[dict]) -> dict:
        claim = _claim(project, "proposals", stain, core)
        if claim is None:
            return {"skipped": True}
        try:
            for attempt in range(2):
                pool = tiles_pool.pool
                try:
                    return propose_core(project, stain, core, tiles, calibration[(tiles[0]["tma"], stain)], nuclei, pool=pool)
                except BrokenProcessPool:
                    if attempt:
                        raise
                    tiles_pool.renew(pool)  # a tile process died (memory): once more with new ones
        finally:
            claim.unlink(missing_ok=True)

    started = time.time()
    try:
        with ThreadPoolExecutor(cores_at_once, thread_name_prefix="stainid-core") as coordinators:
            futures = {coordinators.submit(one, core, stain, tiles): (core, stain, why) for core, stain, tiles, why in jobs}
            for number, future in enumerate(as_completed(futures), start=1):
                core, stain, why = futures[future]
                units.drain(futures)  # asked to stop: the core images not started yet wait for the next run
                if future.cancelled():
                    continue
                try:
                    result = future.result()
                except Exception as error:  # noqa: BLE001 - one bad core must not stop the others
                    print(f"[{number}/{len(jobs)}] {core} {stain}: failed: {type(error).__name__}: {error}", flush=True)
                    continue
                if result.get("skipped"):
                    print(f"[{number}/{len(jobs)}] {core} {stain}: taken by another process", flush=True)
                else:
                    print(f"[{number}/{len(jobs)}] {core} {stain} ({why}): {result['candidates']} candidates, {result['unproposed']} unproposed, "
                          f"{result['seconds']:.0f} s ({(time.time() - started) / number:.0f} s per core image so far)", flush=True)
    finally:
        tiles_pool.shutdown()
    units.boundary("stage 2")


# ---------------------------------------------------------------- stage 4

_MODELS: dict = {}


def _calls_worker(root: str, stain: str, core_id: str) -> dict:
    from stainid.project import load_project
    from stainid.v2.classify import classify_core, load_model

    project = load_project(root)
    if stain not in _MODELS:
        _MODELS[stain] = load_model(project, stain)
    return classify_core(project, stain, core_id, _MODELS[stain])


def run_calls(project, stains=STAINS, cores: set[str] | None = None, workers: int | None = None, redo: bool = False) -> None:
    jobs = todo(project, "calls", stains, cores, redo)
    workers = workers or max(1, min(8, (os.cpu_count() or 2) // 4))
    print(f"[0/{len(jobs)}] stage 4 (calls): {len(jobs)} core images to do", flush=True)
    if not jobs:
        return
    _limited_threads()
    with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as pool:
        futures = {pool.submit(_calls_worker, str(project.root), stain, core): (core, stain) for core, stain, _, _ in jobs}
        for number, future in enumerate(as_completed(futures), start=1):
            core, stain = futures[future]
            units.drain(futures)
            if future.cancelled():
                continue
            try:
                result = future.result()
                print(f"[{number}/{len(jobs)}] {core} {stain}: {result['accepted']} of {result['candidates']} accepted", flush=True)
            except Exception as error:  # noqa: BLE001
                print(f"[{number}/{len(jobs)}] {core} {stain}: failed: {type(error).__name__}: {error}", flush=True)
    units.boundary("stage 4")


# ---------------------------------------------------------------- stage 3

def run_outlines(project, stains=STAINS, cores: set[str] | None = None, device: str | None = None, shard_index: int = 0,
                 shard_count: int = 1, redo: bool = False) -> None:
    from stainid.v2.outlines import BatchOutliner, outline_core

    jobs = todo(project, "outlines", stains, cores, redo)[shard_index::shard_count]
    print(f"[0/{len(jobs)}] stage 3 (outlines): {len(jobs)} core images to do", flush=True)
    if not jobs:
        return
    project.apply_environment()
    outliner = BatchOutliner(project.model("sam"), device)
    calibration = calibration_table(project.input("calibration"))
    for number, (core, stain, tiles, why) in enumerate(jobs, start=1):
        units.boundary(f"{number - 1} core images outlined")
        claim = _claim(project, "outlines", stain, core)
        if claim is None:
            continue
        try:
            result = outline_core(project, stain, core, tiles, outliner, calibration[(tiles[0]["tma"], stain)], redo)
            print(f"[{number}/{len(jobs)}] {core} {stain} ({why}): {result['new']} outlined ({result['outlined']} in all), "
                  f"{result['seconds']:.0f} s", flush=True)
        except Exception as error:  # noqa: BLE001
            print(f"[{number}/{len(jobs)}] {core} {stain}: failed: {type(error).__name__}: {error}", flush=True)
        finally:
            claim.unlink(missing_ok=True)


# ---------------------------------------------------------------- stage 5

def run_measures(project) -> None:
    from stainid.v2.measures import write_measures

    for line in write_measures(project):
        print(line, flush=True)
