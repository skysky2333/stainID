"""The AT8 thread threshold of "Fix the field": one slider per field instead of editing threads one by one.

In threads mode (H, AT8 fields only) the page draws the detected thread network of the field, and a slider redraws it
live at another thread threshold: a factor on the slide's DAB threshold the thread filter cuts at (both of its cuts,
the stain's and the ridge's: `stainid.stains.tau.threads.thread_ladder`), from 0.5x to 2x of detection's own (1.0,
marked on the track), in `FACTORS` steps. The person moves it until the lines match the threads they see and presses
Enter ("Threads look right"); left untouched, that says detection's threshold is right.

`field_ladder` makes the network at every step once per field, the same way stage 2 does (the tile's stage-2 field:
its detection window, detection's masks of the standard crops, the nuclei; `stainid.v2.tau.maps`, then the skeleton
in the tile's own area, `stainid.v2.threads.segments`), so the step at 1.0 is stage 2's thread output. It is kept on
disk (`thread_ladder` cache) until the core's stage-2 output changes, and made ahead with the field's pictures.

One label per field and person (`TASK`, id `thread_threshold:<tile_id>`): `value` the chosen factor, `points` what it
gives ({"factor", "detection": 1.0, "thread_threshold" (the DAB level of the ridge cut), "detection_threshold",
"length_um" and "detection_length_um" in the tile}). AI labels are never shown here (`store.is_ai_label`), and the
tutorial saves nothing. The Models page's "Thread detector" panel (`summary`) reads them: they only tune; the analysis
keeps detection's threshold until a new setting is validated and the pipeline run again.

The simple mode's neurite brush reads a wider ladder (`WIDE_FACTORS`: 0.25x to 4x, the same steps of 2^(1/8), 33 of
them, `wide=True`): its own cache file, made only for simple sessions' fields (ahead with their pictures, like the
detailed one). The ridge response is the whole picture's at every value, so a step of the wide ladder at a factor of
`FACTORS` is that step of the standard one to the pixel: step i of the standard ladder is step i + `WIDE_OFFSET` here.
"""
from __future__ import annotations

import json
import threading
import time
from collections import OrderedDict

import numpy as np
import pandas as pd

TASK = "thread_threshold"
FACTORS = tuple(round(2 ** (k / 8), 4) for k in range(-8, 9))  # 0.5 … 2, 17 steps, 1.0 in the middle
DETECTION = FACTORS.index(1.0)
WIDE_FACTORS = tuple(round(2 ** (k / 8), 4) for k in range(-16, 17))  # 0.25 … 4, 33 steps: the simple neurite brush
WIDE_DETECTION = WIDE_FACTORS.index(1.0)
WIDE_OFFSET = WIDE_DETECTION - DETECTION  # a standard step's index in the wide ladder
WIDE = "wide"
RIDGE = 0.25  # the ridge cut of `thread_network`, as a share of the slide's DAB threshold
VERSION = 1  # of the cached ladder
_KEPT: OrderedDict[tuple, dict] = OrderedDict()
_LOCK = threading.Lock()


def label_id(tile_id: str) -> str:
    return f"{TASK}:{tile_id}"


def _stage2_tiles(project, core_id: str) -> tuple[list[dict], float | None, frozenset]:
    from stainid.v2 import run  # noqa: PLC0415

    found = run.groups(project, ["AT8"], {str(core_id)}).get((str(core_id), "AT8"), [])
    calibration, nuclei = run.proposal_inputs(project, fresh=False)
    threshold = calibration.get((found[0]["tma"], "AT8")) if found else None
    return found, threshold, nuclei


def factors_of(wide: bool) -> tuple[float, ...]:
    return WIDE_FACTORS if wide else FACTORS


def _ladder_networks(project, tile_id: str, core_id: str, factors=FACTORS):
    """(stage 2's field of the tile, its `tau.maps`, the network at every step) or None when stage 2 cannot run there."""
    from stainid.imaging.shared import shared_pictures  # noqa: PLC0415
    from stainid.stains.tau.threads import thread_ladder  # noqa: PLC0415
    from stainid.v2 import proposals, tau  # noqa: PLC0415

    from stainid.api import render  # noqa: PLC0415

    render.yield_to_people()  # a core decode: after the pictures people wait for
    tiles, threshold, nuclei = _stage2_tiles(project, core_id)
    tile = next((t for t in tiles if str(t["tile_id"]) == str(tile_id)), None)
    if tile is None or threshold is None:
        return None
    spec = proposals.core_spec(project, "AT8", tiles, threshold, nuclei)
    image, glass, _ = proposals.decode_core(project.root / tile["image_path"])
    window = spec["window"]
    first = min(window, max(spec["context"], proposals.FIRST_WINDOW)) if spec["whole_objects"] else window
    automatic = (np.zeros(image.shape[:2], bool), np.zeros(image.shape[:2], bool)) if window > spec["context"] else None
    work = proposals.CoreWork(spec, image, glass, automatic)
    box = work.window_box(tile, first)
    if automatic is not None:  # detection's masks of the standard crops, of the tiles the window reaches into
        for other in tiles:
            if proposals._overlap(box, proposals._crop_box(other, 0, work.size)):
                work.standard_masks(other)
    field = work.cut_field(tile, box)
    with shared_pictures(field.rgb):
        found = tau.maps(field)
        networks = thread_ladder(field.rgb, found["maps"]["valid"], field.threshold, field.pixel_size_um, factors, nuclei=field.nuclei)
    return field, found, networks


def compute(project, tile: dict, wide: bool = False) -> dict:
    """The thread network of one AT8 field at every step of `FACTORS` (`wide`: `WIDE_FACTORS`), in the field picture's
    pixels: per step the factor, the thread length in the tile (µm) and the lines (flat [x0, y0, x1, y1, …] polylines,
    stage 2's segments)."""
    from stainid.labels import frames  # noqa: PLC0415
    from stainid.v2.threads import segments  # noqa: PLC0415

    started = time.time()
    factors = factors_of(wide)
    detection = factors.index(1.0)
    kind = {"ladder": WIDE} if wide else {}
    made = _ladder_networks(project, str(tile["tile_id"]), str(tile["core_id"]), factors)
    if made is None:
        return {"status": "unavailable", "factors": list(factors), "detection": detection, "steps": [], **kind}
    field, found, networks = made
    inner = np.zeros(field.shape, bool)
    inner[field.inner] = True
    keep = found["maps"]["valid"] & inner
    ox, oy = frames.field_origin(tile, project.context_px)
    shift = np.array([field.origin[0] - ox, field.origin[1] - oy])
    owner = np.zeros(field.shape, np.int32)
    steps = []
    for factor, network in zip(factors, networks):
        pieces = segments(network["skeleton"] & keep, network["width_um"], owner, field.pixel_size_um)
        steps.append({"factor": factor, "length_um": round(float(sum(p["length_um"] for p in pieces)), 1),
                      "lines": [(np.asarray(p["points"]) + shift).reshape(-1).astype(int).tolist() for p in pieces]})
    field.release()
    return {"status": "done", "factors": list(factors), "detection": detection, "dab_threshold": float(field.threshold),
            "ridge_threshold": RIDGE * float(field.threshold), "steps": steps, "seconds": round(time.time() - started, 2), **kind}


def _stamp(project, core_id: str) -> str | None:
    from stainid.v2 import manifest  # noqa: PLC0415

    made = manifest.read(project, "proposals", "AT8", str(core_id))
    return None if made is None else f"{VERSION}:{made.get('digest', '')}"


def _ladder_file(project, tile: dict, wide: bool):
    from stainid import disk_cache  # noqa: PLC0415

    return disk_cache.path(project.root, "thread_ladder", f"{tile['tile_id']}{'.wide' if wide else ''}.json")


def field_ladder(project, tile: dict, wide: bool = False) -> dict:
    """`compute`, kept in memory and on disk (`thread_ladder` cache) until the core's stage-2 output changes."""
    from stainid import disk_cache  # noqa: PLC0415

    stamp = _stamp(project, str(tile["core_id"]))
    key = (str(project.root), str(tile["tile_id"]), wide)
    with _LOCK:
        kept = _KEPT.get(key)
        if kept is not None and kept.get("stamp") == stamp:
            _KEPT.move_to_end(key)
            return kept
    target = _ladder_file(project, tile, wide)
    out = None
    try:
        saved = json.loads(target.read_text())
        if saved.get("stamp") == stamp:
            out = saved
            disk_cache.used(target)
    except (OSError, ValueError):
        pass
    if out is None:
        out = {**compute(project, tile, wide), "stamp": stamp}
        if out["status"] == "done":
            try:
                disk_cache.write(target, json.dumps(out).encode())
            except OSError:
                pass
    with _LOCK:
        _KEPT[key] = out
        while len(_KEPT) > 16:
            _KEPT.popitem(last=False)
    return out


def ready(project, tile: dict, wide: bool = False) -> bool:
    """Whether the field's ladder is made (in memory or on disk) for the core's current stage-2 output."""
    stamp = _stamp(project, str(tile["core_id"]))
    kept = _KEPT.get((str(project.root), str(tile["tile_id"]), wide))
    if kept is not None and kept.get("stamp") == stamp:
        return True
    target = _ladder_file(project, tile, wide)
    try:
        return json.loads(target.read_text()).get("stamp") == stamp
    except (OSError, ValueError):
        return False


def nearest_step(factor: float, factors=FACTORS) -> int:
    return int(np.argmin([abs(np.log(f) - np.log(max(float(factor), 1e-6))) for f in factors]))


def record(ladder: dict, factor: float) -> dict:
    """The label fields of a chosen thread threshold: `value` the factor, `points` what it gives (see the module)."""
    factors = tuple(ladder.get("factors") or FACTORS)
    step = nearest_step(factor, factors)
    detection = factors.index(1.0)
    steps = ladder.get("steps") or []
    dab = ladder.get("dab_threshold")
    length = steps[step]["length_um"] if steps else None
    detected = steps[detection]["length_um"] if steps else None
    return {"task": TASK, "value": f"{factors[step]:.4f}", "confirmed": step == detection,
            "points": {"factor": factors[step], "detection": 1.0, "thread_threshold": None if dab is None else round(RIDGE * dab * factors[step], 5),
                       "detection_threshold": None if dab is None else round(RIDGE * dab, 5), "length_um": length, "detection_length_um": detected}}


def summary(labels: pd.DataFrame, tiles: pd.DataFrame) -> dict:
    """The Models page's "Thread detector" panel: people's chosen thread thresholds relative to detection's (AI labels
    left out), per TMA and slide, the thread length each gives against detection's, and a suggested setting for the
    stain (the median factor), per TMA where they differ consistently (a TMA's median at least a step away, with three
    fields or more)."""
    from stainid.labels.store import is_ai_label  # noqa: PLC0415

    rows = labels[(labels.task == TASK)] if "task" in labels and len(labels) else labels.iloc[0:0]
    if len(rows):
        rows = rows[~rows.source.map(is_ai_label).astype(bool)]
    if rows.empty:
        return {"fields": 0, "people": 0, "rows": [], "by_tma": [], "by_slide": [], "suggested": None, "per_tma": []}
    rows = rows.assign(factor=pd.to_numeric(rows.value, errors="coerce")).dropna(subset=["factor"])
    info = tiles.drop_duplicates("tile_id").set_index("tile_id") if len(tiles) and "tile_id" in tiles else pd.DataFrame()

    def of(tile_id, column):
        return str(info.loc[tile_id, column]) if column in info and tile_id in info.index else ""

    def lengths(points, key):
        return float(points.get(key)) if isinstance(points, dict) and points.get(key) is not None else np.nan

    rows = rows.assign(tma=[of(t, "tma") for t in rows.tile_id], slide=[of(t, "slide_id") or of(t, "image_id") or f"TMA {of(t, 'tma')} AT8" for t in rows.tile_id],  # a TMA's AT8 glass slide when no slide is named
                       length_um=[lengths(p, "length_um") for p in rows.points], detection_length_um=[lengths(p, "detection_length_um") for p in rows.points])
    step = np.log(FACTORS[1] / FACTORS[0])

    def group(by: str) -> list[dict]:
        out = []
        for name, g in rows.groupby(by):
            detected = g.detection_length_um.sum()
            out.append({by: name, "fields": int(g.tile_id.nunique()), "labels": len(g), "median": round(float(g.factor.median()), 4),
                        "q25": round(float(g.factor.quantile(0.25)), 4), "q75": round(float(g.factor.quantile(0.75)), 4),
                        "length_ratio": round(float(g.length_um.sum() / detected), 3) if detected else None,
                        "factors": [round(float(f), 4) for f in g.factor]})
        return out

    overall = float(rows.factor.median())
    per_tma = [g for g in group("tma") if g["fields"] >= 3 and abs(np.log(g["median"]) - np.log(overall)) >= step - 1e-9]
    detected = rows.detection_length_um.sum()
    return {"fields": int(rows.tile_id.nunique()), "people": int(rows.source.nunique()),
            "rows": rows[["tile_id", "source", "session", "factor", "tma", "slide", "length_um", "detection_length_um"]].replace({np.nan: None}).to_dict("records"),
            "by_tma": group("tma"), "by_slide": group("slide"),
            "length_ratio": round(float(rows.length_um.sum() / detected), 3) if detected else None,
            "suggested": FACTORS[nearest_step(overall)], "per_tma": [{"tma": g["tma"], "suggested": FACTORS[nearest_step(g["median"])], "fields": g["fields"]} for g in per_tma]}
