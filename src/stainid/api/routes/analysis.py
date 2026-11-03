from __future__ import annotations

import os
import threading
import time
from collections import OrderedDict
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException

from stainid.analysis import measure_stats as ms
from stainid.analysis.dictionary import TOPICS, describe, measure
from stainid.api.state import get_state
from stainid.api.util import records
from stainid.setup.naming import Groups

router = APIRouter(tags=["analysis"])
GROUP_COLUMNS = ("disease_group", "region", "tma")
ID_COLUMNS = ("sample_region_id", "core_id", "donor_id", "tma", "region", "disease_group")
V2_MEASURES = "v2/measures"  # where "Make results tables" (the staged detection, `stainid v2 measures`) writes, under the tables folder
TABLES = {  # name: (title, description)
    f"{V2_MEASURES}/results_donor_region.csv": ("All measures, per donor-region", "Main results: one row per donor and brain region, every stain."),
    f"{V2_MEASURES}/results_core.csv": ("All measures, per core", "Main results, one row per core."),
    "results_donor_region.csv": ("All measures, per donor-region", "Main results: one row per donor and brain region, every stain."),
    "results_core.csv": ("All measures, per core", "Main results, one row per core."),
    "neun_donor_region.csv": ("Neurons, per donor-region", "NeuN neuron measures per donor-region."),
    "fields_donor_region.csv": ("Plaques and tau, per donor-region", "6E10 plaque and AT8 tau measures per donor-region."),
    "masks_donor_region.csv": ("Object shapes, per donor-region", "Object-outline shape measures per donor-region."),
    "neun_core.csv": ("Neurons, per core", "NeuN neuron measures per core."),
    "fields_core.csv": ("Plaques and tau, per core", "6E10 plaque and AT8 tau measures per core."),
    "masks_core.csv": ("Object shapes, per core", "Object-outline shape measures per core."),
}
NOT_MEASURES = {"tma", "technical_replicate", "selection_order", "label"}

# The tables change only when a step writes them, so what is read from them is kept: folder listings by the folder's
# modification time (and read again at least every LISTING_FRESH seconds), headers and whole tables by each file's
# modification time and size. Listing ~250 tables or plotting a measure then needs no folder scans or file reads, which
# are slow on USB and network drives, above all while a long run writes to the same drive.
_lock = threading.Lock()
_folders: dict[Path, tuple[int, float, list[Path], list[Path]]] = {}
_headers: dict[Path, tuple[tuple[int, int], list[str]]] = {}
_frames: OrderedDict[Path, tuple[tuple[int, int], pd.DataFrame]] = OrderedDict()
FRAMES_KEPT = 12
LISTING_FRESH = 60.0


def _folder(folder: Path) -> tuple[list[Path], list[Path]]:
    """The CSV files and the subfolders of a folder."""
    try:
        mtime = folder.stat().st_mtime_ns
    except FileNotFoundError:
        return [], []
    with _lock:
        known = _folders.get(folder)
    if known and known[0] == mtime and time.monotonic() - known[1] < LISTING_FRESH:
        return known[2], known[3]
    files, folders = [], []
    with os.scandir(folder) as entries:
        for entry in entries:
            if entry.is_dir():
                folders.append(Path(entry.path))
            elif entry.name.endswith(".csv") and entry.is_file():
                files.append(Path(entry.path))
    with _lock:
        _folders[folder] = (mtime, time.monotonic(), sorted(files), sorted(folders))
    return sorted(files), sorted(folders)


def _csv_tree(folder: Path) -> list[Path]:
    files, folders = _folder(folder)
    return files + [path for sub in folders for path in _csv_tree(sub)]


def _stamp(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return stat.st_mtime_ns, stat.st_size


def _header(path: Path, stamp: tuple[int, int]) -> list[str]:
    with _lock:
        known = _headers.get(path)
    if known and known[0] == stamp:
        return known[1]
    with path.open() as handle:
        header = handle.readline().strip().split(",")
    with _lock:
        _headers[path] = (stamp, header)
    return header


def read_table(path: Path) -> pd.DataFrame:
    """The whole table (kept in memory until the file changes). Callers must not modify it."""
    stamp = _stamp(path)
    with _lock:
        known = _frames.get(path)
        if known and known[0] == stamp:
            _frames.move_to_end(path)
            return known[1]
    frame = pd.read_csv(path, dtype={"donor_id": str, "tma": str})
    with _lock:
        _frames[path] = (stamp, frame)
        while len(_frames) > FRAMES_KEPT:
            _frames.popitem(last=False)
    return frame


def _whole_cores(frame: pd.DataFrame) -> pd.DataFrame:
    """Tables with one row per tile subset: the whole cores."""
    if "sampling_extent" not in frame or frame.empty:
        return frame
    return frame[frame.sampling_extent == ("whole_core" if "whole_core" in set(frame.sampling_extent) else frame.sampling_extent.iloc[0])]


def tables_root() -> Path:
    return get_state().project.output("tables")


def resolve(name: str) -> Path:
    path = (tables_root() / name).resolve()
    if tables_root().resolve() not in path.parents or path.suffix != ".csv" or not path.exists():
        raise HTTPException(404, name)
    return path


# ---------------------------------------------------------------- values a results table does not have, and why

MISSING_TABLES = {"results_core.csv": ("results_missing_core.csv", "core_id"),
                  "results_donor_region.csv": ("results_missing_donor_region.csv", "sample_region_id")}
NOT_MEASURED = "not_measured"  # also the reason of an empty cell the table of reasons does not list
OBJECT_WORDS = {"neurons": "neurons", "plaques": "plaques", "tau": "tau objects"}  # by the measure's topic
REASON_WORDS = {"no_grey_matter": ("no grey matter", "This tissue has no grey matter, so there is nothing to measure here."),
                "no_image": ("no image", "There is no image of this stain here, so nothing could be measured."),
                NOT_MEASURED: ("not measured", "This was not measured here.")}


def _missing(path: Path) -> tuple[pd.DataFrame, str] | None:
    """The table of reasons beside a main results table (one row per value it lacks: the row's id columns, `measure`,
    `reason`, `n_objects`) and the column naming a row; None when the project has no such table."""
    name, key = MISSING_TABLES.get(path.name, (None, None))
    file = path.with_name(name) if name else None
    if file is None or not file.exists():
        return None
    frame = _whole_cores(read_table(file))
    if not {key, "measure", "reason"} <= set(frame.columns):
        return None
    frame = frame.assign(**{key: frame[key].astype(str), "measure": frame.measure.astype(str), "reason": frame.reason.fillna(NOT_MEASURED).astype(str)})
    if "n_objects" not in frame:
        frame = frame.assign(n_objects=float("nan"))
    return frame.drop_duplicates([key, "measure"]), key


def _reason_text(reason: str, column: str) -> str:
    """A reason in plain words, for this measure ("too few plaques")."""
    if reason == "too_few_objects":
        return f"too few {OBJECT_WORDS.get(measure(column).get('topic'), 'objects')}"
    return REASON_WORDS[reason][0] if reason in REASON_WORDS else reason.replace("_", " ")


def _reason_why(reason: str, column: str, n_objects) -> str:
    """The same as a sentence, for hovering."""
    if reason == "too_few_objects":
        found = f" ({int(n_objects):,} found)" if pd.notna(n_objects) else ""
        return f"Too few {OBJECT_WORDS.get(measure(column).get('topic'), 'objects')} here to measure this{found}."
    return REASON_WORDS[reason][1] if reason in REASON_WORDS else f"No value here: {reason.replace('_', ' ')}."


def _empty_cells(frame: pd.DataFrame, measures: list[str], missing: pd.DataFrame, key: str) -> pd.DataFrame:
    """Every empty cell of these measures as a row (`key`, `measure`, `reason`, `n_objects`): a value the table has is
    never counted as missing, an empty cell without a listed reason is "not measured"."""
    cells = frame[[key, *measures]].astype({key: str}).melt(id_vars=key, var_name="measure", value_name="value")
    cells = cells[cells.value.isna()].drop(columns="value").astype({"measure": str})
    cells = cells.merge(missing[[key, "measure", "reason", "n_objects"]], on=[key, "measure"], how="left")
    return cells.assign(reason=cells.reason.fillna(NOT_MEASURED))


def _counted(texts: pd.Series) -> dict[str, int]:
    return {str(text): int(n) for text, n in texts.value_counts().items()}


def _table_gaps(frame: pd.DataFrame, shown: pd.DataFrame, page: pd.DataFrame, columns: list[str], found: tuple[pd.DataFrame, str]) -> dict:
    """What a page of a results table lacks: the reason of each empty cell of each row of the page (`reasons`) and over
    all rows asked for (`cells`), how many rows the table has, and the rows that have none in it (`absent`, by reason)."""
    missing, key = found
    numeric = set(measure_columns(frame))
    measures = [c for c in columns if c in numeric]
    cells = _empty_cells(shown, measures, missing, key)
    texts = {pair: _reason_text(pair[1], pair[0]) for pair in set(zip(cells.measure, cells.reason, strict=True))}
    on_page: dict[str, dict[str, dict]] = {}
    keys = set(page[key].astype(str))
    for cell in cells[cells[key].isin(keys)].itertuples(index=False):
        on_page.setdefault(getattr(cell, key), {})[cell.measure] = {"text": texts[cell.measure, cell.reason], "why": _reason_why(cell.reason, cell.measure, cell.n_objects)}
    absent = missing[~missing[key].isin(set(frame[key].astype(str)))].drop_duplicates(key)
    return {"reasons": [on_page.get(k, {}) for k in page[key].astype(str)],
            "gaps": {"rows": len(frame), "cells": _counted(pd.Series([texts[pair] for pair in zip(cells.measure, cells.reason, strict=True)], dtype=object)),
                     "absent": _counted(pd.Series([_reason_text(r, m) for r, m in zip(absent.reason, absent.measure, strict=True)], dtype=object))}}


def _feature_gaps(frame: pd.DataFrame, column: str, keep: list[str], found: tuple[pd.DataFrame, str]) -> list[dict]:
    """The rows with no value of this measure, with their id columns and why: those of the table and those it has no row for."""
    missing, key = found
    listed = missing[missing.measure == column]
    empty = frame[frame[column].isna()].astype({key: str})
    rows = empty[[c for c in keep if c != column]].merge(listed[[key, "reason", "n_objects"]], on=key, how="left")
    absent = listed[~listed[key].isin(set(frame[key].astype(str)))]
    rows = pd.concat([rows, absent[[c for c in [*keep, "reason", "n_objects"] if c in absent and c != column]]], ignore_index=True)
    rows = rows.assign(reason=rows.reason.fillna(NOT_MEASURED))
    rows = rows.assign(text=[_reason_text(r, column) for r in rows.reason], why=[_reason_why(r, column, n) for r, n in zip(rows.reason, rows.n_objects, strict=True)])
    return records(rows)


@router.get("/tables")
def tables() -> list[dict]:
    base = tables_root()
    project = get_state().project
    out = []
    staged = (base / V2_MEASURES / "results_donor_region.csv").exists()
    for path in _folder(base / V2_MEASURES)[0] + _folder(base)[0] + sorted(_csv_tree(base / "biology")):
        stamp = _stamp(path)
        header = _header(path, stamp)
        name = str(path.relative_to(base))
        title, description = TABLES.get(name, (name, ""))
        if staged and name in TABLES and "/" not in name:  # the staged detection made the main tables: these are the first detection's
            title, description = f"{title} (legacy v1)", f"The first detection's results (legacy v1). {description}"
        out.append({"name": name, "title": title, "description": description, "main": name in TABLES, "path": project.relative(path),
                    "columns": len(header), "size_kb": round(stamp[1] / 1024, 1), "modified": stamp[0] / 1e9,
                    "donor_region": "sample_region_id" in header, "contrasts": "std_effect" in header or "effect_sd" in header})
    return sorted(out, key=lambda t: (not t["main"], list(TABLES).index(t["name"]) if t["main"] else 0, t["name"]))


def measure_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c]) and c not in NOT_MEASURES and not c.endswith("_id")]


@router.get("/tables/{name:path}/columns")
def columns(name: str) -> dict:
    frame = read_table(resolve(name))
    numeric = measure_columns(frame)
    return {"columns": list(frame.columns), "numeric": numeric, "groupable": [c for c in GROUP_COLUMNS if c in frame],
            "descriptions": {c: describe(c) for c in frame.columns if describe(c)},
            "measures": [measure(c) for c in numeric], "topics": TOPICS, "rows": len(frame)}


@router.get("/tables/{name:path}/rows")
def rows(name: str, offset: int = 0, limit: int = 100, sort: str = "", desc: bool = False, q: str = "", columns: str = "") -> dict:
    """One page of a table, optionally sorted by a column and filtered to rows whose names contain `q`."""
    path = resolve(name)
    whole = frame = read_table(path)
    if q:
        text = [c for c in frame.columns if c in ID_COLUMNS or not pd.api.types.is_numeric_dtype(frame[c])]
        hit = pd.Series(False, index=frame.index)
        for column in text:
            hit |= frame[column].astype(str).str.contains(q, case=False, regex=False, na=False)
        frame = frame[hit]
    if sort in frame:
        frame = frame.sort_values(sort, ascending=not desc, na_position="last", kind="stable")
    keep = [c for c in dict.fromkeys(columns.split(",")) if c in frame] if columns else list(frame.columns)
    limit = max(1, min(limit, 5000))
    page = frame.iloc[offset: offset + limit]
    found = _missing(path)
    gaps = _table_gaps(whole, frame, page, keep, found) if found else {}
    return {"total": len(frame), "offset": offset, "columns": keep, "rows": records(page[keep]), **gaps}


@router.get("/tables/{name:path}/feature")
def feature(name: str, column: str, group: str = "disease_group", facet: str | None = None) -> dict:
    path = resolve(name)
    frame = _whole_cores(read_table(path))
    if column not in frame or group not in frame:
        raise HTTPException(400, "unknown column")
    keep = [c for c in dict.fromkeys(["sample_region_id", "core_id", "donor_id", group, facet, column]) if c and c in frame]
    found = _missing(path)  # `gaps`: every row with no value and why, when the project has the table of reasons; else null
    return {"column": column, "group": group, "facet": facet, "points": records(frame[keep].dropna(subset=[column])),
            "gaps": _feature_gaps(frame, column, keep, found) if found else None}


# ---------------------------------------------------------------- reproducibility and group differences per measure

_stats: dict[str, tuple[tuple, dict]] = {}


def _sibling(path: Path, part: str, other: str) -> Path | None:
    """The same table at the other level (results_donor_region.csv <-> results_core.csv)."""
    if f"_{part}" not in path.stem:
        return None
    sibling = path.with_name(path.name.replace(f"_{part}", f"_{other}", 1))
    return sibling if sibling.exists() else None


def _one_extent(frame: pd.DataFrame, extent: str | None = None) -> pd.DataFrame:
    """Tables with several tile sets (sampling_extent) hold one row per set: keep one, `extent` when the table has it."""
    if "sampling_extent" not in frame or frame.empty:
        return frame
    keep = extent if extent is not None and (frame.sampling_extent == extent).any() else frame.sampling_extent.iloc[0]
    return frame[frame.sampling_extent == keep]


def _reliability_table(path: Path, biology: Path, since: float) -> Path | None:
    """Stage 5's measure_reliability.csv: beside the shown table (written in the same run, so taken as it is), else a
    copy in the study's biology folder made from the current results."""
    beside = path.with_name(ms.RELIABILITY_TABLE)
    if beside.exists():
        return beside
    other = biology / ms.RELIABILITY_TABLE
    return other if _fresh(other, since) else None


def _fresh(path: Path, since: float) -> bool:
    """A study table made from the current results (not written before the table on screen)."""
    return path.exists() and path.stat().st_mtime >= since


@router.get("/tables/{name:path}/stats")
def table_stats(name: str) -> dict:
    """For every measure of a table: its replicate-core ICC, and its difference between pairs of diagnosis groups
    (effect in SD, p and FDR), from the study's tables where they hold the same numbers, else computed here. Kept until
    one of the files it is made from changes."""
    path = resolve(name)
    project = get_state().project
    base = tables_root()
    shown_time = path.stat().st_mtime
    is_core = "_core" in path.stem
    donor_region = _sibling(path, "core", "donor_region") if is_core else path
    core = path if is_core else _sibling(path, "donor_region", "core")
    tiles = str(project.config.get("analysis", {}).get("tiles") or "whole_core")
    biology = base / "biology"
    primary = biology / "cohort_morphology" / tiles / "replicate_reliability.csv"  # the study's main reliability table
    icc_files = [primary] + [p for p in _folder(biology)[0] if p.name.endswith("_replicate_icc.csv")]
    icc_files = [p for p in icc_files if _fresh(p, shown_time)]
    given = _reliability_table(path, biology, shown_time)  # stage 5's own, made with these results: it wins
    specs = [s for s in ms.STUDY_CONTRASTS if _fresh(base / s.file, shown_time)]
    inputs = sorted({base / i for s in specs for i in s.inputs if (base / i).exists()})
    files = [p for p in [path, donor_region, core, *([given] if given else []), *icc_files, *(base / s.file for s in specs), *inputs] if p is not None]
    stamp = (tuple((str(p), _stamp(p)) for p in files), repr(project.config.get("groups")), tiles)
    with _lock:
        known = _stats.get(name)
    if known and known[0] == stamp:
        return known[1]

    frame = _one_extent(read_table(path), tiles)
    measures = measure_columns(frame)
    core_frame = _one_extent(read_table(core), tiles) if core is not None else None
    icc = ms.reliability(measures, core_frame, ms.study_icc(icc_files, project.relative, primary),
                         ms.results_icc(given, project.relative) if given else None)
    comparisons = []
    if donor_region is not None:
        shown = _one_extent(read_table(donor_region), tiles)
        if "disease_group" in shown:
            groups = Groups(project.config.get("groups"))
            kinds = {kind: groups.code(kind) for pair in ms.PREFERRED for kind in pair}
            order = [g["code"] for g in get_state().groups()]
            found = set(shown.disease_group.dropna().astype(str))
            present = [c for c in order if c in found] + sorted(found - set(order))
            inputs_read = {p: _one_extent(read_table(p), tiles) for p in inputs}
            tested = [m for m in measures if m in shown]
            for first, second in ms.comparisons(present, kinds):
                comparisons.append(_comparison(shown, tested, (first, second), kinds, specs, inputs_read))
    result = {"icc": icc, "comparisons": comparisons, "icc_files": [project.relative(p) for p in [*([given] if given else []), *icc_files]],
              "contrast_files": [project.relative(base / s.file) for s in specs]}
    with _lock:
        _stats[name] = (stamp, result)
    return result


def _comparison(shown: pd.DataFrame, measures: list[str], pair: tuple[str, str], kinds: dict[str, str],
                specs: list[ms.StudyContrasts], inputs: dict[Path, pd.DataFrame]) -> dict:
    """One pair of groups: the study's tests of the measures it tested, a quick test of the others, and the models."""
    base, project = tables_root(), get_state().project
    tests: dict[str, dict[str, dict]] = {}
    within = ""
    for spec in specs:
        if (kinds.get(spec.groups[0]), kinds.get(spec.groups[1])) != pair:
            continue
        study = ms.study_tests(shown, spec, read_table(base / spec.file), [inputs.get(base / i) for i in spec.inputs])
        for column, by_model in study.items():
            for model, test in by_model.items():
                tests.setdefault(column, {}).setdefault(model, {**test, "file": project.relative(base / spec.file)})
        within = within or spec.within
    quick = ms.quick_tests(shown, [m for m in measures if m not in tests], *pair)
    for column, test in quick.items():
        tests.setdefault(column, {})[ms.QUICK] = test
    models = sorted({m for by_model in tests.values() for m in by_model if m != ms.QUICK}, key=lambda m: len(ms.model_terms(m)))
    listed = [ms.model_info(m, ms.model_terms(m), "study", i == len(models) - 1) for i, m in enumerate(models)]
    if quick:
        listed.append(ms.model_info(ms.QUICK, list(ms.QUICK_TERMS), "quick", not models))
    return {"key": "|".join(pair), "first": pair[0], "second": pair[1], "within": within or "Within TMAs that hold both groups.",
            "models": listed, "tests": tests}
