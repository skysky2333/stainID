"""Neuropil threads of one AT8 core image, for the core viewer.

The "Detect plaques and tau" step (`stainid fields`) traces AT8 threads per tile (`thread_network`) but keeps only the
totals (`thread_ridge_*` in `<fields>/parts/<core>_AT8_features.csv`), not where the threads are. The viewer traces
them again exactly as the step did — the same crop of the core image, exclusions, nuclei and slide threshold, so the
traced skeleton is pixel for pixel the one the step measured — the first time someone turns the layer on for a core.
The core image is decoded once and its tiles are traced a few at a time in separate processes (about 15 s of CPU per
tile; tracing is mostly Python-bound work that would hold up the server's own threads), one core at a time, and the result is cached in `data/cache/core_threads/`, rebuilt when the step rewrites the core.
Tiles finished so far are served while the rest are being traced.

Each thread piece (the skeleton between junctions) is a simplified open polyline in core-image pixel indices with its
mean width in µm.
"""
from __future__ import annotations

import json
import multiprocessing
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage

from stainid import disk_cache
from stainid.api import pyramid
from stainid.api.render import crop_origin
from stainid.api.state import ProjectState
from stainid.imaging.core_images import open_image
from stainid.nuclei import clean_nuclei
from stainid.pipelines.cohort_v1 import context_crop
from stainid.qc.exclusions import read_manual_exclusions
from stainid.stains.tau.network import segment_at8_network
from stainid.stains.tau.threads import thread_network
from stainid.workflows import calibration_table
from stainid.workflows.fields import exclusion_mask

Image.MAX_IMAGE_PIXELS = None
CACHE_VERSION = 1
EPSILON_PX = 0.75  # polyline simplification at full resolution
WORKERS = 3  # tiles traced at once, each in its own process (~0.6 GB while it runs)
STALE_SECONDS = 60  # a core nobody has asked about for this long is no longer waited for
RETRY_SECONDS = 60  # a failed core is reported for this long, then tried again
COLUMNS = ["width_um", "points"]
_NEIGHBOURS = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]  # straight steps first


# ---------------------------------------------------------------- tracing one tile

def trace_tile(rgb: np.ndarray, inner: tuple[slice, slice], nuclei: np.ndarray, dab_threshold: float, pixel_size_um: float,
               exclusion: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The thread skeleton of one field crop inside its tile (the pixels the step counted as `thread_ridge_skeleton_px`)
    and the thread width (µm) at every pixel, as `stainid.stains.tau.field_pipeline.analyze_at8_v2` computes them."""
    nuclei = clean_nuclei(nuclei, rgb)
    valid = segment_at8_network(rgb, dab_threshold, pixel_size_um, pixel_size_um, exclusion)[2]["valid"]
    network = thread_network(rgb, valid, dab_threshold, pixel_size_um, nuclei=nuclei)
    inside = np.zeros(valid.shape, bool)
    inside[inner] = True
    return network["skeleton"] & valid & inside, network["width_um"]


def _walk(pixels: set[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    """Order the pixels of a thin piece into paths, starting from its ends (straight steps before diagonal ones)."""
    left = set(pixels)
    degree = {p: sum((p[0] + dy, p[1] + dx) in pixels for dy, dx in _NEIGHBOURS) for p in pixels}
    paths = []
    while left:
        start = min(left, key=lambda p: (degree[p], p))
        path, at = [start], start
        left.discard(start)
        while True:
            step = next(((at[0] + dy, at[1] + dx) for dy, dx in _NEIGHBOURS if (at[0] + dy, at[1] + dx) in left), None)
            if step is None:
                break
            path.append(step)
            left.discard(step)
            at = step
        paths.append(path)
    return paths


def polylines(skeleton: np.ndarray, width_um: np.ndarray, epsilon: float = EPSILON_PX) -> list[tuple[float, list[int]]]:
    """The skeleton as pieces between junctions: (mean width in µm, flat [x0, y0, x1, y1, …] pixel indices). Each piece
    reaches into the junction pixels next to its ends, so the drawn network has no gaps."""
    if not skeleton.any():
        return []
    count = cv2.filter2D(skeleton.astype(np.uint8), -1, np.ones((3, 3), np.uint8), borderType=cv2.BORDER_CONSTANT)
    junction = skeleton & (count >= 4)  # itself and three or more neighbours
    labels, _ = ndimage.label(skeleton & ~junction, np.ones((3, 3)))
    junctions = set(zip(*map(np.ndarray.tolist, np.nonzero(junction))))
    out = []
    for index, box in enumerate(ndimage.find_objects(labels), start=1):
        if box is None:
            continue
        ys, xs = np.nonzero(labels[box] == index)
        ys, xs = ys + box[0].start, xs + box[1].start
        width = round(float(width_um[ys, xs].mean()), 2)
        for path in _walk(set(zip(ys.tolist(), xs.tolist()))):
            for end, at in ((0, path[0]), (len(path), path[-1])):
                near = next(((at[0] + dy, at[1] + dx) for dy, dx in _NEIGHBOURS if (at[0] + dy, at[1] + dx) in junctions), None)
                if near is not None:
                    path.insert(end, near)
            points = np.array([(x, y) for y, x in path], np.int32)
            if len(points) > 2:
                points = cv2.approxPolyDP(points.reshape(-1, 1, 2), epsilon, False)[:, 0, :]
            if len(points) == 1:
                points = np.repeat(points, 2, axis=0)  # a lone pixel: drawn as a dot
            out.append((width, points.ravel().tolist()))
    return out


# ---------------------------------------------------------------- one core

def features_path(state: ProjectState, core_id: str) -> Path:
    return state.project.output("fields") / "parts" / f"{core_id}_AT8_features.csv"


def measured(state: ProjectState, core_id: str) -> dict | None:
    """What the step measured on this core's threads (all tiles), or None before it has run."""
    try:
        features = pd.read_csv(features_path(state, core_id), usecols=["thread_ridge_length_mm", "thread_ridge_skeleton_px"])
    except (FileNotFoundError, ValueError):
        return None
    return {"length_mm": round(float(features.thread_ridge_length_mm.sum()), 2), "skeleton_px": int(features.thread_ridge_skeleton_px.sum())}


def _trace_field(rgb: np.ndarray, inner: tuple[slice, slice], tile: dict, manual: dict, dab_threshold: float, nuclei_path: Path,
                 context_px: int) -> dict:
    """Trace one tile's field crop as the step did (in a worker process): its skeleton size, length and pieces in core pixels."""
    pixel_size = float(np.sqrt(float(tile["pixel_width_um"]) * float(tile["pixel_height_um"])))
    nuclei = np.load(nuclei_path)["masks"] if nuclei_path.exists() else np.zeros(rgb.shape[:2], dtype=np.uint32)
    exclusion = exclusion_mask(rgb, tile, inner, pixel_size, manual)
    skeleton, width = trace_tile(rgb, inner, nuclei, dab_threshold, pixel_size, exclusion)
    x0, y0 = crop_origin(tile, context_px)
    pieces = [[w, [v + (x0 if k % 2 == 0 else y0) for k, v in enumerate(points)]] for w, points in polylines(skeleton, width)]
    return {"tile_id": tile["tile_id"], "skeleton_px": int(skeleton.sum()), "length_mm": float(skeleton.sum()) * pixel_size / 1000, "rows": pieces}


def _trace_core(state: ProjectState, tiles: list[dict], job: _Job) -> list[dict]:
    """Trace every tile of one core image, `WORKERS` at a time, adding each finished tile to `job.done`."""
    project = state.project
    calibration = calibration_table(project.input("calibration"))
    image_path = tiles[0]["image_path"]
    manual = {image_path: read_manual_exclusions(project.input("manual_exclusions")).get(image_path, [])}
    nuclei_dir = project.output("nuclei")
    with pyramid.DECODE, open_image(project.root / image_path) as image:
        image.load()
        crops = [context_crop(image, int(t["x_px"]), int(t["y_px"]), int(t["width_px"]), int(t["height_px"]), project.context_px) for t in tiles]
    with ProcessPoolExecutor(WORKERS, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = []
        for index, tile in enumerate(tiles):
            rgb, inner = crops[index]
            crops[index] = None  # the crop is freed once its tile is traced
            futures.append(pool.submit(_trace_field, rgb, inner, tile, manual, calibration[(str(tile["tma"]), "AT8")],
                                       nuclei_dir / f"{tile['tile_id']}.npz", project.context_px))
            del rgb
        for future in as_completed(futures):
            with _GUARD:
                job.done.append(future.result())
    return list(job.done)


@dataclass
class _Job:
    key: list
    total: int
    asked: float = field(default_factory=time.monotonic)
    running: bool = False
    done: list[dict] = field(default_factory=list)


_GUARD = threading.Lock()
_ONE_CORE = threading.Lock()  # one core traced at a time, across all projects
_JOBS: dict[str, _Job] = {}
_FAILED: dict[str, tuple[list, str, float]] = {}  # target -> (key, message, when)


def _run(state: ProjectState, target: Path, tiles: list[dict], job: _Job) -> None:
    name = str(target)
    try:
        while not _ONE_CORE.acquire(timeout=2):
            if time.monotonic() - job.asked > STALE_SECONDS:
                return  # nobody is looking any more
        try:
            if time.monotonic() - job.asked > STALE_SECONDS:
                return
            job.running = True
            results = _trace_core(state, tiles, job)
            disk_cache.write(target, json.dumps({"source": job.key, "tiles": results}, separators=(",", ":")).encode())
        finally:
            _ONE_CORE.release()
    except Exception as error:  # noqa: BLE001 - reported to the page that asked
        _FAILED[name] = (job.key, f"{type(error).__name__}: {error}", time.monotonic())
    finally:
        with _GUARD:
            if _JOBS.get(name) is job:
                del _JOBS[name]


def _summary(status: str, tiles: list[dict], total: int, **extra) -> dict:
    rows = [row for tile in tiles for row in tile["rows"]]
    return {"status": status, "tiles_done": len(tiles), "tiles_total": total, "pieces": len(rows),
            "length_mm": round(sum(t["length_mm"] for t in tiles), 2), "columns": COLUMNS, "rows": rows, **extra}


def core_threads(state: ProjectState, core_id: str, tiles: pd.DataFrame, prepare: bool = True) -> dict:
    """The threads of one AT8 core image: `status` "done" (all tiles traced), "waiting" / "tracing" (the tiles traced
    so far; ask again), "failed" (with `error`) or "not_run" (the step has not measured this core yet), the pieces as
    rows of `columns` (core-image pixels), the traced `length_mm` and what the step measured (`measured`)."""
    source = features_path(state, core_id)
    try:
        stat = source.stat()
    except FileNotFoundError:
        return _summary("not_run", [], len(tiles))
    target = state.cache_dir("core_threads") / f"{core_id}_AT8.json"
    records = tiles.to_dict("records")
    thresholds = calibration_table(state.project.input("calibration"))
    key = [CACHE_VERSION, stat.st_mtime_ns, stat.st_size, sorted({thresholds.get((str(t["tma"]), "AT8")) for t in records}, key=str)]
    extra = {"measured": measured(state, core_id)}
    try:
        cached = json.loads(target.read_text())
        if cached.get("source") == key:
            disk_cache.used(target)
            return _summary("done", cached["tiles"], len(records), **extra)
    except (FileNotFoundError, ValueError, KeyError, AttributeError):
        pass
    name = str(target)
    failed = _FAILED.get(name)
    if failed and failed[0] == key and time.monotonic() - failed[2] < RETRY_SECONDS:
        return _summary("failed", [], len(records), error=failed[1], **extra)
    with _GUARD:
        job = _JOBS.get(name)
        if job is None and prepare and records:
            job = _JOBS[name] = _Job(key, len(records))
            threading.Thread(target=_run, args=(state, target, records, job), name="stainid-threads", daemon=True).start()
        if job is None:
            return _summary("not_started", [], len(records), **extra)
        job.asked = time.monotonic()
        ahead = 0 if job.running else sum(1 for other in _JOBS.values() if other.running)
        return _summary("tracing" if job.running else "waiting", list(job.done), job.total, ahead=ahead, **extra)
