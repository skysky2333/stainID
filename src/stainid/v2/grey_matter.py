"""Grey-matter burdens: stage 2's per-tile stain areas again, with the algorithm's white matter (`v2.white_matter`)
counted out, for stage 5 when `v2.white_matter.analysis` is "exclude".

Stage 2 keeps each tile's stained areas as totals (the amyloid extent, the AT8 positive, non-compact and thread areas,
the DAB areas at the threshold's multiples, the nuclei), not per pixel, so leaving the white matter out of a burden needs
the pixels again. This pass reads each 6E10 and AT8 core image that has white matter once and works out exactly what
stage 2 works out for every tile the white matter reaches (`proposals.tile_summary`, with its cells left out: the same
crops, slide thresholds, nuclei, linear-artifact and fold masks, the same levels), without proposing, scoring or
outlining anything. The other tiles' values are stage 2's. Counts and areas of objects need no pass: stage 5 drops the
objects whose centre lies in the white matter and takes the area from the masks. NeuN needs none either (its measures
are counts and areas only).

Outputs, under `<outputs.v2>/white_matter/grey_tiles/`: `<stain>/<core_id>.csv` (the redone tiles' columns, `COLUMNS`)
and `manifests/<stain>/<core_id>.json` (the white matter's digest, stage 2's manifest digest, the version): a core image
is redone when its white matter or its stage-2 outputs change. `stainid v2 grey-matter` runs it; CPU only.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from stainid.tables import write_text_atomic

VERSION = 1
STAINS = ("6E10", "AT8")
SCALE = 8  # `proposals.MASK_SCALE`
COLUMNS = {  # the tile columns the pass redoes, per stain (the rest stay stage 2's)
    "6E10": ["analysis_area_mm2", "valid_area_mm2", "dab_area_mm2_at_0.5", "dab_area_mm2_at_0.75", "dab_area_mm2_at_1",
             "dab_area_mm2_at_1.5", "dab_area_mm2_at_2", "nucleus_count", "amyloid_extent_area_mm2"],
    "AT8": ["analysis_area_mm2", "valid_area_mm2", "dab_area_mm2_at_0.5", "dab_area_mm2_at_0.75", "dab_area_mm2_at_1",
            "dab_area_mm2_at_1.5", "dab_area_mm2_at_2", "nucleus_count", "at8_valid_area_mm2", "at8_positive_area_mm2",
            "at8_noncompact_area_mm2", "at8_compact_profile_count", "at8_legacy_skeleton_px", "thread_line_area_mm2",
            "thread_skeleton_px", "thread_width_sum_um", "thread_branchpoint_count", "thread_endpoint_count"],
}


def root(project) -> Path:
    from stainid.v2 import white_matter  # noqa: PLC0415

    return white_matter.root(project) / "grey_tiles"


def table_path(project, stain: str, core_id: str) -> Path:
    return root(project) / stain / f"{core_id}.csv"


def manifest_path(project, stain: str, core_id: str) -> Path:
    return root(project) / "manifests" / stain / f"{core_id}.json"


def leave_out(white: np.ndarray, tile: dict) -> np.ndarray:
    """The white matter on a tile's own grid (the cells of its mask piece: `proposals.mask_piece`)."""
    x, y = int(tile["x_px"]) // SCALE, int(tile["y_px"]) // SCALE
    w, h = max(1, int(tile["width_px"]) // SCALE), max(1, int(tile["height_px"]) // SCALE)
    out = np.zeros((h, w), bool)
    piece = white[y: y + h, x: x + w]
    out[: piece.shape[0], : piece.shape[1]] = piece
    return out


def expected(project, stain: str, core_id: str) -> dict | None:
    """What the core image's outputs are made from now; None when nothing is to do (no white matter, or stage 2 has not
    run)."""
    from stainid.v2 import manifest, white_matter  # noqa: PLC0415

    white = white_matter.read(project, core_id, stain)
    made = manifest.read(project, "proposals", stain, core_id)
    if white is None or not white.any() or made is None:
        return None
    return {"version": VERSION, "white_matter": white_matter.digest(project, core_id, stain), "proposals": made.get("digest")}


def current(project, stain: str, core_id: str) -> pd.DataFrame | None:
    """The redone tiles of a core image (`tile_id` + `COLUMNS`) when they are up to date; None when the core image has no
    white matter. Raises when it has and the pass has not run since its inputs changed: stage 5 would mix whole-core
    burdens into grey-matter measures."""
    wanted = expected(project, stain, core_id)
    if wanted is None:
        return None
    try:
        recorded = json.loads(manifest_path(project, stain, core_id).read_text()).get("inputs")
    except (OSError, ValueError):
        recorded = None
    if recorded != wanted:
        raise RuntimeError(f"{stain} {core_id}: the grey-matter burdens are missing or out of date; run `stainid v2 grey-matter` "
                           "before stage 5 (v2.white_matter.analysis is exclude)")
    return pd.read_csv(table_path(project, stain, core_id))


def todo(project, cores: set[str] | None = None, redo: bool = False) -> list[tuple[str, str]]:
    """(core_id, stain) of every 6E10 / AT8 core image with white matter whose grey-matter burdens are missing or out
    of date."""
    from stainid.v2 import white_matter  # noqa: PLC0415

    table = white_matter.summary(project)
    out = []
    for row in table[(table.status == "white") & table.stain.isin(STAINS)].itertuples():
        if cores and row.core_id not in cores:
            continue
        wanted = expected(project, row.stain, row.core_id)
        if wanted is None:
            continue
        try:
            recorded = json.loads(manifest_path(project, row.stain, row.core_id).read_text()).get("inputs")
        except (OSError, ValueError):
            recorded = None
        if redo or recorded != wanted:
            out.append((row.core_id, row.stain))
    return out


def _fields(project, stain: str, core_id: str, tiles: list[dict], wanted: set[str]):
    """The core image's tiles worked on as stage 2 works on them (`proposals.CoreWork`), without the proposals: yields
    (tile, field) for every tile in `wanted`; the field is released after each. With a detection window wider than the
    standard crop, detection's linear-artifact and fold masks are made from the standard crops of every tile near one."""
    from stainid.imaging.shared import shared_pictures  # noqa: PLC0415
    from stainid.v2 import proposals  # noqa: PLC0415
    from stainid.v2.run import proposal_inputs  # noqa: PLC0415

    calibration, nuclei = proposal_inputs(project)
    threshold = calibration[(tiles[0]["tma"], stain)]
    spec = proposals.core_spec(project, stain, tiles, threshold, nuclei)
    image, glass, _ = proposals.decode_core(project.root / tiles[0]["image_path"])
    size = image.shape[:2]
    automatic = (np.zeros(size, bool), np.zeros(size, bool)) if spec["window"] > spec["context"] else None
    work = proposals.CoreWork(spec, image, glass, automatic)
    first = min(spec["window"], max(spec["context"], proposals.FIRST_WINDOW)) if spec["whole_objects"] else spec["window"]
    mine = [t for t in tiles if t["tile_id"] in wanted]
    if automatic is not None:
        boxes = [work.window_box(t, first) for t in mine]
        for tile in tiles:  # the masks a wanted tile's crop reads: of the tiles it overlaps
            x0, y0 = int(tile["x_px"]), int(tile["y_px"])
            x1, y1 = x0 + int(tile["width_px"]), y0 + int(tile["height_px"])
            if any(x0 < b[2] and b[0] < x1 and y0 < b[3] and b[1] < y1 for b in boxes):
                work.standard_masks(tile)
    for tile in mine:
        field = work.cut_field(tile, work.window_box(tile, first))
        with shared_pictures(field.rgb):
            yield tile, field
        field.release()


def _extent(field) -> None:
    """6E10: the legacy deposits' extent stage 2 measures the amyloid burden on (`proposals.tile_regions`)."""
    from stainid.v2 import amyloid  # noqa: PLC0415

    _, objects, labels, _, _ = amyloid.segment_amyloid_candidates(
        field.rgb, field.threshold, field.pixel_width_um, field.pixel_height_um, field.exclusion, split_touching=True,
        statistics_mask=field.standard if field.reference is None else None,
        background=amyloid.background(field) if field.reference is not None else None)
    kept = sorted({int(row["label"]) for row in objects})
    field.memo["amyloid_extent"] = np.isin(labels, kept) if kept else np.zeros(field.shape, bool)


def _compact_profiles(field, out: np.ndarray) -> int:
    """AT8: the tile's compact profiles (`tau.maps`) whose centre is not in `out` (the tile's white-matter cells)."""
    from stainid.v2 import tau  # noqa: PLC0415

    count = 0
    for o in tau.maps(field)["objects"]:
        if o["source_kind"] != "compact_profile" or not field.inside(o["centroid_x_px"], o["centroid_y_px"]):
            continue
        cx = int((o["centroid_x_px"] - field.inner[1].start) // SCALE)
        cy = int((o["centroid_y_px"] - field.inner[0].start) // SCALE)
        if not (0 <= cy < out.shape[0] and 0 <= cx < out.shape[1] and out[cy, cx]):
            count += 1
    return count


def measure_core(project, stain: str, core_id: str, white: np.ndarray, tiles: list[dict], check: bool = False) -> pd.DataFrame:
    """The grey-matter values of every tile of a core image the white matter reaches (`check`: also every tile's
    values without leaving anything out, `whole_` columns, to compare with stage 2's)."""
    from stainid.v2 import proposals  # noqa: PLC0415

    rows = []
    wanted = {t["tile_id"] for t in tiles if check or leave_out(white, t).any()}
    for tile, field in _fields(project, stain, core_id, tiles, wanted):
        out = leave_out(white, tile)
        if stain == "6E10":
            _extent(field)
        found = proposals.tile_summary(field, stain, [], 0, leave_out=out)
        row = {"tile_id": tile["tile_id"], "white_cells": int(out.sum()), **{c: found.get(c) for c in COLUMNS[stain]}}
        if stain == "AT8":
            row["at8_compact_profile_count"] = _compact_profiles(field, out)
        if check:
            whole = proposals.tile_summary(field, stain, [], 0)
            row |= {f"whole_{c}": whole.get(c) for c in COLUMNS[stain]}
        rows.append(row)
    return pd.DataFrame(rows, columns=["tile_id", "white_cells", *COLUMNS[stain]] if not rows else None)


def _tile_rows(project, stain: str, core_id: str) -> list[dict]:
    """The core image's tiles as stage 2 got them (the tile table's rows)."""
    from stainid.v2.run import groups  # noqa: PLC0415

    found = groups(project, [stain], {core_id}).get((core_id, stain))
    if not found:
        raise KeyError(f"{stain} {core_id}: not in the tile table")
    return found


def run_core(project, core_id: str, stain: str) -> dict:
    """The pass on one core image: its redone tiles and its manifest."""
    from stainid.v2 import white_matter  # noqa: PLC0415

    started = time.time()
    wanted = expected(project, stain, core_id)
    white = white_matter.read(project, core_id, stain)
    frame = measure_core(project, stain, core_id, white, _tile_rows(project, stain, core_id))
    target = table_path(project, stain, core_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    write_text_atomic(target, frame.to_csv(index=False))
    place = manifest_path(project, stain, core_id)
    place.parent.mkdir(parents=True, exist_ok=True)
    seconds = round(time.time() - started, 1)
    write_text_atomic(place, json.dumps({"inputs": wanted, "tiles": len(frame), "seconds": seconds,
                                         "written": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=1))
    return {"core_id": core_id, "stain": stain, "tiles": len(frame), "seconds": seconds}


def _worker(root: str, core_id: str, stain: str) -> dict:
    import os  # noqa: PLC0415

    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(name, "1")
    from stainid.project import load_project  # noqa: PLC0415

    return run_core(load_project(root), core_id, stain)


def run(project, cores: set[str] | None = None, redo: bool = False, workers: int = 1, log=print) -> list[dict]:
    """The pass on every core image that needs it (`todo`), `workers` core images at once (about 2.5 GB each); returns
    one row per core image."""
    jobs = todo(project, cores, redo)
    log(f"grey-matter burdens: {len(jobs)} 6E10 / AT8 core images with white matter to (re)do, {workers} at once")
    done = []
    if workers <= 1:
        for core_id, stain in jobs:
            done.append(run_core(project, core_id, stain))
            log(f"{len(done)}/{len(jobs)} {stain} {core_id}: {done[-1]['tiles']} tiles in {done[-1]['seconds']:.0f} s")
        return done
    from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: PLC0415
    from multiprocessing import get_context  # noqa: PLC0415

    with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as pool:
        futures = {pool.submit(_worker, str(project.root), core_id, stain): (core_id, stain) for core_id, stain in jobs}
        for future in as_completed(futures):
            core_id, stain = futures[future]
            done.append(future.result())
            log(f"{len(done)}/{len(jobs)} {stain} {core_id}: {done[-1]['tiles']} tiles in {done[-1]['seconds']:.0f} s")
    return done
