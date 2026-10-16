"""Running the v2 stages: which core images each stage still has to do (missing or outdated), spread over processes.

Every stage is resumable per core image (a core is done once its manifest is written) and redoes a core whose manifest no
longer matches what it would be made from now (`stainid.v2.manifest`). Stage 2 runs `workers` core images at once, one
process each (a core image is decoded whole: about 0.8 GB, plus about 1 GB while a tile is worked on), stage 3 one
process per GPU (`--shard-index i --shard-count 2` with CUDA_VISIBLE_DEVICES), stages 4 and 5 are quick.
"""
from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context

from stainid import cache
from stainid.qc.exclusions import read_manual_exclusions
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
    """The slide thresholds, hand-drawn exclusions and the names of the nuclei files (empty before they exist); `fresh`
    lists the nuclei folder now (a run), else at most every 30 s (the app's status: the folder holds a file per tile)."""
    calibration = calibration_table(project.input("calibration")) if project.input("calibration").exists() else {}
    manual = read_manual_exclusions(project.input("manual_exclusions"))
    nuclei = cache.names(project.output("nuclei"), every=0 if fresh else 30)
    return calibration, manual, nuclei


def todo(project, stage: str, stains, cores: set[str] | None = None, redo: bool = False) -> list[tuple[str, str, list[dict], str]]:
    """(core_id, stain, tiles, why) of every core image the stage has to (re)do: why is "missing" or "outdated"."""
    from stainid.v2 import classify, outlines, proposals

    out = []
    found = groups(project, stains, cores)
    if stage == "proposals":
        calibration, manual, nuclei = proposal_inputs(project)
    models = {}
    for (core_id, stain), tiles in found.items():
        recorded = manifest.read_all(project, stage, stain).get(core_id)
        if stage == "proposals":
            threshold = calibration.get((tiles[0]["tma"], stain))
            if threshold is None:
                continue
            made = proposals.expected(project, stain, core_id, tiles, threshold, manual.get(tiles[0]["image_path"], []), nuclei)
        elif stage == "calls":
            if stain not in models:
                models[stain] = classify.load_model(project, stain)
            made = classify.expected(project, stain, core_id, models[stain])
        else:
            made = outlines.expected(project, stain, core_id)
        if made is None:  # its upstream stage has not run
            continue
        if redo or recorded is None:
            out.append((core_id, stain, tiles, "missing" if recorded is None else "redo"))
        elif not manifest.up_to_date(recorded, made):
            out.append((core_id, stain, tiles, "outdated"))
    return out


def status(project) -> dict:
    """Per stage and stain: core images done and up to date, outdated (made, but an input, a parameter or an upstream stage
    changed since, also when that upstream stage has not been re-run yet), and the total."""
    from stainid.v2 import classify, outlines, proposals

    out = {stage: {} for stage in ("proposals", "calls", "outlines", "candidates")}
    if not project.analysis_tiles().exists():
        return {stage: {stain: {"done": 0, "outdated": 0, "total": 0} for stain in STAINS} for stage in out}
    found = groups(project, STAINS)
    calibration, manual, nuclei = proposal_inputs(project, fresh=False)
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
                        stale = True  # what depends on a missing stage is not up to date either
                    continue
                if stage == "proposals":
                    made = None if threshold is None else proposals.expected(project, stain, core, tiles, threshold,
                                                                              manual.get(tiles[0]["image_path"], []), nuclei)
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


# ---------------------------------------------------------------- stage 2

def _propose_worker(root: str, stain: str, core_id: str, tiles: list[dict], threshold: float, polygons: list, nuclei: frozenset) -> dict:
    import cv2

    from stainid.project import load_project
    from stainid.v2.proposals import propose_core

    cv2.setNumThreads(2)
    project = load_project(root)
    claim = _claim(project, "proposals", stain, core_id)
    if claim is None:
        return {"skipped": True}
    try:
        return propose_core(project, stain, core_id, tiles, threshold, polygons, nuclei)
    finally:
        claim.unlink(missing_ok=True)


def _limited_threads() -> None:
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(key, "2")


def run_proposals(project, stains=STAINS, cores: set[str] | None = None, workers: int | None = None, shard_index: int = 0,
                  shard_count: int = 1, redo: bool = False) -> None:
    jobs = todo(project, "proposals", stains, cores, redo)[shard_index::shard_count]
    calibration, manual, nuclei = proposal_inputs(project)
    workers = workers or default_workers()
    print(f"[0/{len(jobs)}] stage 2 (proposals): {len(jobs)} core images to do with {workers} at once", flush=True)
    if not jobs:
        return
    _limited_threads()
    started = time.time()
    with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as pool:
        futures = {pool.submit(_propose_worker, str(project.root), stain, core, tiles, calibration[(tiles[0]["tma"], stain)],
                               manual.get(tiles[0]["image_path"], []), nuclei): (core, stain, why) for core, stain, tiles, why in jobs}
        for number, future in enumerate(as_completed(futures), start=1):
            core, stain, why = futures[future]
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
            try:
                result = future.result()
                print(f"[{number}/{len(jobs)}] {core} {stain}: {result['accepted']} of {result['candidates']} accepted", flush=True)
            except Exception as error:  # noqa: BLE001
                print(f"[{number}/{len(jobs)}] {core} {stain}: failed: {type(error).__name__}: {error}", flush=True)


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
