"""Object texture (pyradiomics): texture and intensity features of every accepted object's Segment Anything outline on
its DAB stain, for the results tables (stage 5). Replaces the standalone `pyradiomics_feature_extract.py` of the older
workflow, whose plots by group are now the Results page's.

Two units, as stage 5 reports them:
- per object: pyradiomics on each accepted object with a kept outline (`decision` accepted, `sam_status` ok: the objects
  of the outline shape measures), on a crop around it; stage 5 takes the median per core and per donor-region
  (`<object>_rad_<feature>_median`, at least `mask_aggregation.MINIMUM_OBJECTS` objects);
- per field: pyradiomics once per tile on one mask holding all the tile's objects, as the older workflow did; stage 5
  takes the mean over the fields that have objects, weighted by their mask area (`<object>_rad_field_<feature>`). Where
  the core image has white matter (`v2.white_matter`) the field mask is also made without the objects whose centre lies
  in it (`scope` grey), for the grey-matter tables.

The image is the DAB optical density (`imaging.color.rgb_to_hed`, as stage 3 measures it) mapped onto 0–255 with a fixed
ceiling (`v2.radiomics.od_ceiling`, default 0.3: the darkest brown is about 0.25), the same for every slide, so the
parameter file's `binWidth: 25` gives about ten grey levels. The features and settings are those of the parameter file
(`radiomics_params.yaml` beside this module, or `v2.radiomics.params_file`); 2D is forced. An object pyradiomics refuses
(a mask of one pixel, a flat crop) gets no values: empty, never 0.

Outputs, under `<outputs.v2>/radiomics/`: `objects/<stain>/<core_id>.csv` (`candidate_id`, `tile_id`, a column per
feature), `fields/<stain>/<core_id>.csv` (`tile_id`, `scope`, `n_objects`, `mask_px`, a column per feature; every tile,
those without objects too) and `manifests/<stain>/<core_id>.json`. A core image is redone when its candidates (calls,
outlines), its white matter, the parameter file or the settings change. `stainid v2 radiomics` runs it; CPU only. Needs
the `radiomics` extra (`pip install -e ".[radiomics]"`).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

VERSION = 1
STAINS = ("NeuN", "6E10", "AT8")
PREFIX = {"NeuN": "neun_soma", "6E10": "plaque", "AT8": "tau_neuron"}  # the outline measures' column prefixes
PARAMS_FILE = Path(__file__).with_name("radiomics_params.yaml")
DEFAULTS = {"enabled": False, "params_file": None, "od_ceiling": 0.3, "margin_px": 4}
SCOPES = ("whole", "grey")
# pyradiomics key: (column stem, label, unit, what it means). A feature the parameter file enables that is not here gets
# a stem made from its name (`stem`) and a plain label.
FEATURES = {
    "original_firstorder_Mean": ("firstorder_mean", "mean DAB level", "grey level, 0–255",
                                 "the mean DAB optical density inside the outline, on the 0–255 scale (0 = none, 255 = the ceiling)"),
    "original_glcm_JointEntropy": ("glcm_joint_entropy", "joint entropy (GLCM)", "bits",
                                   "how unpredictable the DAB levels of neighbouring pixels are (higher = more heterogeneous texture)"),
    "original_glrlm_LongRunHighGrayLevelEmphasis": ("glrlm_long_run_high_grey", "long dark runs (GLRLM)", "",
                                                    "how much of the stain lies in long runs of dark pixels (higher = long, strongly stained streaks)"),
    "original_glszm_LargeAreaEmphasis": ("glszm_large_area_emphasis", "large uniform zones (GLSZM)", "",
                                         "how much of the outline is made of large zones of one DAB level (higher = coarser, more uniform patches)"),
    "original_gldm_LargeDependenceEmphasis": ("gldm_large_dependence_emphasis", "large dependence (GLDM)", "",
                                              "how often pixels sit among many neighbours of a similar DAB level (higher = more homogeneous)"),
}


def stem(key: str) -> str:
    """The column stem of a pyradiomics feature key (`original_glcm_JointEntropy` -> `glcm_joint_entropy`)."""
    if key in FEATURES:
        return FEATURES[key][0]
    name = key[len("original_"):] if key.startswith("original_") else key
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name).lower()


def available() -> bool:
    """Whether pyradiomics and SimpleITK can be imported (without importing them)."""
    from importlib.util import find_spec  # noqa: PLC0415

    return find_spec("radiomics") is not None and find_spec("SimpleITK") is not None


def settings(project) -> dict:
    from stainid.v2.params import params  # noqa: PLC0415

    return {**DEFAULTS, **(params(project).get("radiomics") or {})}


def params_file(project) -> Path:
    found = settings(project)["params_file"]
    if not found:
        return PARAMS_FILE
    path = Path(found)
    return path if path.is_absolute() else project.root / path


def root(project) -> Path:
    return project.output("v2") / "radiomics"


def table_path(project, kind: str, stain: str, core_id: str) -> Path:
    return root(project) / kind / stain / f"{core_id}.csv"


def manifest_path(project, stain: str, core_id: str) -> Path:
    return root(project) / "manifests" / stain / f"{core_id}.json"


# ---------------------------------------------------------------- what is to do

def expected(project, stain: str, core_id: str) -> dict | None:
    """What the core image's outputs are made from now; None while its candidates have no outlines (stage 3 not run)."""
    from stainid.v2 import manifest, white_matter  # noqa: PLC0415

    made = manifest.read(project, "candidates", stain, core_id)
    if made is None or manifest.read(project, "outlines", stain, core_id) is None:
        return None
    found = settings(project)
    out = {"version": VERSION, "candidates": made.get("digest"), "od_ceiling": float(found["od_ceiling"]),
           "margin_px": int(found["margin_px"]), "params_file": hashlib.sha256(params_file(project).read_bytes()).hexdigest()[:16]}
    if white_matter.read(project, core_id, stain) is not None:
        out["white_matter"] = white_matter.digest(project, core_id, stain)
    return out


def _recorded(project, stain: str, core_id: str) -> dict | None:
    try:
        return json.loads(manifest_path(project, stain, core_id).read_text())
    except (OSError, ValueError):
        return None


def _standing(project, stain: str, core_id: str) -> str:
    """Where a core image stands: "done", "outdated" (made from other inputs), "missing", or "" (nothing to do yet)."""
    wanted = expected(project, stain, core_id)
    if wanted is None:
        return ""
    recorded = _recorded(project, stain, core_id)
    if recorded is None:
        return "missing"
    return "done" if recorded.get("inputs") == wanted else "outdated"


def _images(project, stains=STAINS) -> list[tuple[str, str]]:
    """(core_id, stain) of every core image with outlines."""
    from stainid.v2 import manifest  # noqa: PLC0415

    return [(core_id, stain) for stain in stains for core_id in sorted(manifest.read_all(project, "outlines", stain))]


def todo(project, stains=STAINS, cores: set[str] | None = None, redo: bool = False) -> list[tuple[str, str]]:
    """(core_id, stain) of every core image whose features are missing or out of date (all of them with `redo`)."""
    out = []
    for core_id, stain in _images(project, stains):
        if cores and core_id not in cores:
            continue
        standing = _standing(project, stain, core_id)
        if standing and (redo or standing != "done"):
            out.append((core_id, stain))
    return out


def progress(project) -> dict:
    """How far the step has got, for the Run page: core images up to date (`done`), made from other inputs (`outdated`),
    in all (`total`), and when one was last made (`written`, 0: never)."""
    standing = [s for s in (_standing(project, stain, core_id) for core_id, stain in _images(project)) if s]
    written = 0.0
    for stain in STAINS:
        folder = root(project) / "manifests" / stain
        if folder.is_dir():
            with os.scandir(folder) as entries:
                written = max(written, *(entry.stat().st_mtime for entry in entries), 0.0)
    return {"done": standing.count("done"), "outdated": standing.count("outdated"), "total": len(standing), "written": written}


def cached_progress(project) -> dict:
    """`progress`, made once while its inputs stay the same."""
    from stainid import cache, disk_cache  # noqa: PLC0415
    from stainid.v2 import manifest  # noqa: PLC0415

    upstream = manifest.root(project) / "manifests"
    inputs = [*(root(project) / "manifests" / s for s in STAINS), *(upstream / "candidates" / s for s in STAINS),
              *(upstream / "outlines" / s for s in STAINS), params_file(project)]
    return cache.memo(("radiomics progress", str(project.root)), inputs, lambda: progress(project), every=10.0, folders=True,
                      persist=disk_cache.answer_file(project.root, "v2-radiomics.pkl"), background=True)


def current(project, stain: str, core_id: str) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """The core image's (objects, fields) tables when they are up to date; None when missing or out of date (stage 5
    then leaves its texture columns empty, `not_measured`)."""
    wanted = expected(project, stain, core_id)
    recorded = _recorded(project, stain, core_id)
    if wanted is None or recorded is None or recorded.get("inputs") != wanted:
        return None
    try:
        objects = pd.read_csv(table_path(project, "objects", stain, core_id), dtype={"candidate_id": str, "tile_id": str})
        fields = pd.read_csv(table_path(project, "fields", stain, core_id), dtype={"tile_id": str, "scope": str})
    except (OSError, pd.errors.EmptyDataError):
        return None
    return objects, fields


# ---------------------------------------------------------------- measuring

def extractor(project):
    """A pyradiomics feature extractor from the parameter file, 2D forced, its log quiet."""
    import logging  # noqa: PLC0415

    from radiomics import featureextractor  # noqa: PLC0415

    logging.getLogger("radiomics").setLevel(logging.ERROR)
    found = featureextractor.RadiomicsFeatureExtractor(str(params_file(project)))
    found.settings["force2D"] = True
    found.settings["force2Ddimension"] = 0
    return found


def grey_levels(rgb: np.ndarray, ceiling: float) -> np.ndarray:
    """The DAB optical density of an RGB crop on 0–255 (`ceiling` OD and above: 255), as float32."""
    from stainid.imaging.color import rgb_to_hed  # noqa: PLC0415

    dab = np.maximum(rgb_to_hed(rgb)[..., 2], 0)
    return (np.clip(dab / ceiling, 0.0, 1.0) * 255.0).astype(np.float32)


def features(engine, grey: np.ndarray, mask: np.ndarray, spacing: tuple[float, float]) -> dict[str, float]:
    """pyradiomics on one 2D image and mask (`spacing`: µm per pixel, x then y): {stem: value}; {} when it refuses."""
    import SimpleITK as sitk  # noqa: PLC0415, N813

    if int(mask.sum()) < 2:
        return {}
    image, label = sitk.GetImageFromArray(np.ascontiguousarray(grey)), sitk.GetImageFromArray(mask.astype(np.uint8))
    for picture in (image, label):
        picture.SetSpacing(spacing)
    try:
        found = engine.execute(image, label, label=1)
    except Exception:  # noqa: BLE001 - pyradiomics refuses masks it cannot measure (too small, one grey level) in many ways
        return {}
    return {stem(k): float(v) for k, v in found.items() if not k.startswith("diagnostics_")}


def _polygons(text, origin: tuple[int, int]) -> list[np.ndarray]:
    """A `sam_outline` cell's rings in crop pixels (`origin`: the crop's top-left in core pixels)."""
    from stainid import candidates as store  # noqa: PLC0415

    out = []
    for ring in store.rings(text):
        pts = np.asarray(ring, np.int64).reshape(-1, 2) - np.asarray(origin, np.int64)
        if len(pts) >= 3:
            out.append(pts.astype(np.int32))
    return out


def measure_field(engine, grey: np.ndarray, origin: tuple[int, int], rows: pd.DataFrame, spacing: tuple[float, float], margin: int,
                  grey_only: np.ndarray | None = None) -> tuple[list[dict], list[dict]]:
    """One tile: a row per object (`rows`: its accepted, outlined objects) and a row per field scope ("whole"; "grey"
    too when `grey_only`, which of `rows` lie outside the white matter, is given)."""
    import cv2  # noqa: PLC0415

    height, width = grey.shape
    whole = np.zeros(grey.shape, np.uint8)
    grey_mask = np.zeros(grey.shape, np.uint8) if grey_only is not None else None
    objects = []
    for position, row in enumerate(rows.itertuples(index=False)):
        polygons = _polygons(row.sam_outline, origin)
        values: dict[str, float] = {}
        if polygons:
            every = np.concatenate(polygons)
            x0, y0 = (int(v) for v in np.maximum(every.min(axis=0) - margin, 0))
            x1, y1 = (int(v) for v in np.minimum(every.max(axis=0) + margin + 1, (width, height)))
            if x1 > x0 and y1 > y0:
                mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
                cv2.fillPoly(mask, [p - [x0, y0] for p in polygons], 1)
                values = features(engine, grey[y0:y1, x0:x1], mask, spacing)
                whole[y0:y1, x0:x1] |= mask
                if grey_mask is not None and grey_only[position]:
                    grey_mask[y0:y1, x0:x1] |= mask
        objects.append({"candidate_id": str(row.candidate_id), "tile_id": str(row.tile_id), **values})
    fields = []
    for scope, mask, count in (("whole", whole, len(rows)), ("grey", grey_mask, int(grey_only.sum()) if grey_only is not None else 0)):
        if mask is None:
            continue
        fields.append({"scope": scope, "n_objects": count, "mask_px": int(mask.sum()), **(features(engine, grey, mask, spacing) if count else {})})
    return objects, fields


def _outlined(project, stain: str, core_id: str) -> pd.DataFrame:
    """The core image's accepted objects with a kept Segment Anything outline (the outline shape measures' objects)."""
    from stainid import candidates as store  # noqa: PLC0415

    found = store.read_core(project, stain, core_id, ["candidate_id", "tile_id", "x_core", "y_core", "decision", "sam_status", "sam_outline"])
    if found.empty or "decision" not in found or "sam_status" not in found:
        return pd.DataFrame(columns=["candidate_id", "tile_id", "x_core", "y_core", "sam_outline"])
    return found[(found.decision == "accepted") & (found.sam_status == "ok")]


def _outside_white(white: np.ndarray, rows: pd.DataFrame) -> np.ndarray:
    """Which objects' centres lie outside the white matter (the rule of stage 5's grey-matter tables)."""
    from stainid.v2 import grey_matter  # noqa: PLC0415

    x, y = rows.x_core.to_numpy(float), rows.y_core.to_numpy(float)
    ok = np.isfinite(x) & np.isfinite(y)
    cx, cy = np.where(ok, x // grey_matter.SCALE, -1).astype(int), np.where(ok, y // grey_matter.SCALE, -1).astype(int)
    inside = (cx >= 0) & (cy >= 0) & (cx < white.shape[1]) & (cy < white.shape[0])
    out = np.zeros(len(rows), bool)
    out[inside] = white[cy[inside], cx[inside]]
    return ~out


def measure_core(project, stain: str, core_id: str, tiles: list[dict], engine=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(objects, fields) of one core image; the image is read only when it has objects."""
    from stainid.imaging.core_images import open_image  # noqa: PLC0415
    from stainid.pipelines.cohort_v1 import context_crop  # noqa: PLC0415
    from stainid.v2 import white_matter  # noqa: PLC0415

    found = settings(project)
    ceiling, margin = float(found["od_ceiling"]), int(found["margin_px"])
    outlined = _outlined(project, stain, core_id)
    white = white_matter.read(project, core_id, stain)
    by_tile = {tile_id: rows for tile_id, rows in outlined.groupby("tile_id", sort=False)} if len(outlined) else {}
    objects, fields = [], []
    image = open_image(project.root / tiles[0]["image_path"]) if by_tile else None
    engine = engine if engine is not None or not by_tile else extractor(project)
    try:
        for tile in tiles:
            tile_id = str(tile["tile_id"])
            rows = by_tile.get(tile_id)
            if rows is None:
                fields += [{"tile_id": tile_id, "scope": s, "n_objects": 0, "mask_px": 0} for s in (SCOPES if white is not None else SCOPES[:1])]
                continue
            x, y = int(tile["x_px"]), int(tile["y_px"])
            rgb, inner = context_crop(image, x, y, int(tile["width_px"]), int(tile["height_px"]), project.context_px)
            spacing = (float(tile.get("pixel_width_um") or project.pixel_size_um), float(tile.get("pixel_height_um") or project.pixel_size_um))
            o, f = measure_field(engine, grey_levels(rgb, ceiling), (x - inner[1].start, y - inner[0].start), rows, spacing, margin,
                                 _outside_white(white, rows) if white is not None else None)
            objects += o
            fields += [{"tile_id": tile_id, **row} for row in f]
    finally:
        if image is not None:
            image.close()
    return pd.DataFrame(objects, columns=None if objects else ["candidate_id", "tile_id"]), pd.DataFrame(fields)


def run_core(project, core_id: str, stain: str, engine=None) -> dict:
    """The step on one core image: its two tables and its manifest."""
    from stainid.tables import write_text_atomic  # noqa: PLC0415
    from stainid.v2.run import groups  # noqa: PLC0415

    started = time.time()
    wanted = expected(project, stain, core_id)
    tiles = groups(project, [stain], {core_id}).get((core_id, stain))
    if not tiles:
        raise KeyError(f"{stain} {core_id}: not in the tile table")
    objects, fields = measure_core(project, stain, core_id, tiles, engine)
    for kind, frame in (("objects", objects), ("fields", fields)):
        target = table_path(project, kind, stain, core_id)
        target.parent.mkdir(parents=True, exist_ok=True)
        write_text_atomic(target, frame.to_csv(index=False))
    measured = int(objects.drop(columns=["candidate_id", "tile_id"]).notna().any(axis=1).sum()) if len(objects) else 0
    seconds = round(time.time() - started, 1)
    place = manifest_path(project, stain, core_id)
    place.parent.mkdir(parents=True, exist_ok=True)
    write_text_atomic(place, json.dumps({"inputs": wanted, "objects": len(objects), "measured": measured, "fields": len(fields),
                                         "seconds": seconds, "written": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=1))
    return {"core_id": core_id, "stain": stain, "objects": len(objects), "measured": measured, "seconds": seconds}


_ENGINE = None


def _worker(root_folder: str, core_id: str, stain: str) -> dict:
    global _ENGINE  # noqa: PLW0603 - one extractor per worker process
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(name, "1")
    from stainid.project import load_project  # noqa: PLC0415

    project = load_project(root_folder)
    if _ENGINE is None:
        _ENGINE = extractor(project)
    return run_core(project, core_id, stain, _ENGINE)


def run(project, stains=STAINS, cores: set[str] | None = None, redo: bool = False, workers: int = 1, log=print) -> list[dict]:
    """The step on every core image that needs it (`todo`), `workers` at once; returns one row per core image. Stops
    between two core images when the queue asks; a core image that fails does not stop the others, and the run ends with
    an error naming how many failed."""
    from stainid.jobs import units  # noqa: PLC0415

    if not available():
        raise RuntimeError('object texture needs pyradiomics and SimpleITK: pip install -e ".[radiomics]"')
    jobs = todo(project, stains, cores, redo)
    log(f"[0/{len(jobs)}] object texture (pyradiomics, {params_file(project).name}): {len(jobs)} core images to (re)do, {workers} at once")
    done, failed = [], 0

    def report(core_id: str, stain: str, get) -> None:
        nonlocal failed
        try:
            row = get()
        except Exception as error:  # noqa: BLE001 - one bad core image must not stop the others
            failed += 1
            log(f"[{len(done) + failed}/{len(jobs)}] {stain} {core_id}: failed: {type(error).__name__}: {error}")
            return
        done.append(row)
        log(f"[{len(done) + failed}/{len(jobs)}] {stain} {core_id}: {row['measured']} of {row['objects']} objects measured in {row['seconds']:.0f} s")

    if workers <= 1:
        engine = extractor(project) if jobs else None
        for core_id, stain in jobs:
            units.boundary(f"{len(done)} core images done")
            report(core_id, stain, lambda c=core_id, s=stain: run_core(project, c, s, engine))
    elif jobs:
        from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: PLC0415
        from multiprocessing import get_context  # noqa: PLC0415

        with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as pool:
            futures = {pool.submit(_worker, str(project.root), core_id, stain): (core_id, stain) for core_id, stain in jobs}
            for future in as_completed(futures):
                core_id, stain = futures[future]
                units.drain(futures)  # asked to stop: the core images not started yet wait for the next run
                if not future.cancelled():
                    report(core_id, stain, future.result)
        units.boundary(f"{len(done)} core images done")
    if failed:
        raise RuntimeError(f"object texture: {failed} of {len(jobs)} core images failed; see the lines above")
    return done


# ---------------------------------------------------------------- stage 5

def field_means(rows: list[dict], stems: list[str]) -> dict[str, float]:
    """The mean of each feature over the fields with objects and values, weighted by their mask area (NaN: none)."""
    out = {}
    for name in stems:
        pairs = [(float(r["mask_px"]), float(r[name])) for r in rows
                 if float(r.get("n_objects") or 0) > 0 and r.get(name) not in ("", None) and np.isfinite(float(r[name])) and float(r["mask_px"]) > 0]
        weight = sum(w for w, _ in pairs)
        out[name] = sum(w * v for w, v in pairs) / weight if weight else float("nan")
    return out
