"""Objects people added where the detection step proposed nothing, measured so that they can train the stain model.

Someone labelling a field can add an object the AI never circled ("Add a missed object"). Such a label has no candidate
in the pool (none within `pool.MATCH_PX`): it shows where the AI is blind, but has no features to learn from. `measure`
segments the object at the clicked position with the stain's own candidate segmentation, restricted to a window around
the click (`training.candidates.measure_at`), and stores it next to the pool the way the pool stores its rows, one CSV
per field: `pool/<stain>/added/<tile_id>.csv`, in the pool's columns plus

- `added_by_person` (always True), `measured` (False when nothing could be segmented there; `reason` says why in plain
  words, and the label stays: it still counts as an object the AI missed), `measured_how`, `measured_x`, `measured_y`;
- `candidate_x`, `candidate_y`: set when the segmented object is a pool candidate after all (the click was off its
  centre, e.g. in a session that shows nothing from the AI); the label then belongs to that candidate.

Only labels made in labelling sessions are measured (the AI reference labels were made on pool candidates), and only on
fields whose pool is built: elsewhere "no candidate" only means the field is not prepared yet. The label log is never
changed.
"""
from __future__ import annotations

import fcntl
from pathlib import Path

import numpy as np
import pandas as pd

from stainid.labels import frames, pool, store
from stainid.labels.sessions import REFERENCE, session_of
from stainid.project import Project
from stainid.sampling.tiles import known_tile_rows, known_tiles
from stainid.tables import write_text_atomic

FLAGS = ["added_by_person", "measured", "measured_how", "reason", "measured_x", "measured_y", "candidate_x", "candidate_y"]
_ALL: dict[tuple[str, str], tuple[tuple, pd.DataFrame]] = {}


def folder(project: Project, stain: str) -> Path:
    return pool.folder(project, stain) / "added"


def path(project: Project, stain: str, tile_id: str) -> Path:
    return folder(project, stain) / f"{tile_id}.csv"


def load_field(project: Project, stain: str, tile_id: str) -> pd.DataFrame | None:
    target = path(project, stain, tile_id)
    return pd.read_csv(target, dtype={"tma": str, "measured_how": str, "reason": str}) if target.exists() else None


def _files(project: Project, stain: str) -> list[Path]:
    return sorted(folder(project, stain).glob("*.csv")) if folder(project, stain).exists() else []


def signature(project: Project, stain: str) -> tuple:
    return tuple((f.name, f.stat().st_mtime_ns) for f in _files(project, stain))


def all_added(project: Project, stain: str) -> pd.DataFrame:
    """Every measured (or unmeasurable) added object of a stain: position, flags and the AI call of the candidate it turned
    out to be (if any), reloaded only when a file changes."""
    stamp = signature(project, stain)
    key = (str(project.root), stain)
    if key in _ALL and _ALL[key][0] == stamp:
        return _ALL[key][1]
    columns = ["x", "y", "radius_px", "ai_value", "tma", *FLAGS]
    parts = [pd.read_csv(f, usecols=lambda c: c in columns, dtype={"tma": str, "measured_how": str, "reason": str}).assign(tile_id=f.stem)
             for f in _files(project, stain)]
    frame = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=[*columns, "tile_id"])
    for column in columns:
        if column not in frame:
            frame[column] = np.nan
    _ALL[key] = (stamp, frame)
    return frame


def additions(project: Project, stain: str, labels: pd.DataFrame | None = None) -> pd.DataFrame:
    """Object labels people made in labelling sessions on prepared fields, at positions with no pool candidate: one row
    per position (tile_id, x, y)."""
    if labels is None:
        labels = frames.in_field_frame(store.current(project.output("reviews")), known_tiles(project), project.context_px)
    mine = labels[(labels.stain == stain) & (labels.task == "class") & labels.x.notna() & (session_of(labels) != REFERENCE)]
    mine = mine[mine.tile_id.isin(pool.prepared(project, stain))]
    if mine.empty:
        return pd.DataFrame(columns=["tile_id", "x", "y"])
    mine = mine.assign(x=mine.x.astype(float), y=mine.y.astype(float), rx=mine.x.astype(float).round(), ry=mine.y.astype(float).round())
    mine = mine.drop_duplicates(["tile_id", "rx", "ry"])
    candidates = {t: g for t, g in pool.all_objects(project, stain).groupby("tile_id")}
    lonely = [pool.nearest(candidates.get(r.tile_id), r.x, r.y) is None for r in mine.itertuples()]
    return mine[lonely][["tile_id", "x", "y"]].reset_index(drop=True)


def pending(project: Project, stain: str, labels: pd.DataFrame | None = None) -> pd.DataFrame:
    """The additions not measured yet."""
    todo = additions(project, stain, labels)
    done = {t: g for t, g in all_added(project, stain).groupby("tile_id")}
    keep = [pool.nearest(done.get(r.tile_id), r.x, r.y) is None for r in todo.itertuples()]
    return todo[keep].reset_index(drop=True)


def _measured_row(project: Project, stain: str, tile: dict, field, loader, x: float, y: float, candidates: pd.DataFrame | None) -> dict:
    from stainid.training.candidates import measure_at

    base = {"x": round(x, 1), "y": round(y, 1), "tma": str(tile["tma"]), "complete": True, "added_by_person": True}
    try:
        row, features, names, how = measure_at(stain, loader, field, x, y)
    except Exception as error:  # noqa: BLE001 - one unmeasurable object must not stop the others
        return {**base, "measured": False, "reason": f"measuring failed ({type(error).__name__}: {error})"}
    if row is None:
        return {**base, "measured": False, "reason": how}
    known = pool.feature_names(project, stain)
    if known is not None and list(known) != list(names):
        return {**base, "measured": False, "reason": "its measurements differ from the field's candidates (the pool was built with other features)"}
    mx, my = float(row["centroid_x_px"]), float(row["centroid_y_px"])
    out = {**base, "radius_px": round(pool._radius(row, field.pixel_size_um), 1), "ai_value": "", "ai_probability": np.nan,
           **{k: row.get(k) for k in pool.KEEP if k not in names}, **dict(zip(names, features)),
           "measured": True, "measured_how": how, "reason": "", "measured_x": round(mx, 1), "measured_y": round(my, 1)}
    same = pool.nearest(candidates, mx, my)
    if same is not None:
        hit = candidates.iloc[same]
        out |= {"candidate_x": float(hit.x), "candidate_y": float(hit.y), "ai_value": hit.ai_value}
    return out


def _write(project: Project, stain: str, tile_id: str, rows: list[dict]) -> None:
    """Adds rows to the field's file (one row per rounded position; a newer measurement replaces an older one)."""
    target = path(project, stain, tile_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    with (target.parent / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        old = load_field(project, stain, tile_id)
        frame = pd.DataFrame(rows) if old is None else pd.concat([old, pd.DataFrame(rows)], ignore_index=True)
        frame = frame.assign(rx=frame.x.round(), ry=frame.y.round()).drop_duplicates(["rx", "ry"], keep="last").drop(columns=["rx", "ry"])
        write_text_atomic(target, frame.to_csv(index=False))


def measure(project: Project, stain: str, tile_ids: set[str] | None = None, loader=None) -> dict[str, int]:
    """Measure every added object not measured yet (only on these fields when given). Returns how many were measured and
    how many could not be."""
    from stainid.training import candidates as candidate_code

    todo = pending(project, stain)
    if tile_ids is not None:
        todo = todo[todo.tile_id.isin(tile_ids)]
    counts = {"measured": 0, "could_not_be_measured": 0}
    if todo.empty:
        return counts
    tiles = known_tile_rows(project)
    loader = loader or candidate_code.FieldLoader(project)
    for tile_id, group in todo.groupby("tile_id"):
        tile = tiles.get(tile_id)
        if tile is None:
            continue
        try:
            field = loader.load(tile)
        except (OSError, KeyError) as error:
            rows = [{"x": round(r.x, 1), "y": round(r.y, 1), "tma": str(tile["tma"]), "complete": True, "added_by_person": True, "measured": False,
                     "reason": f"the field picture could not be read ({error})"} for r in group.itertuples()]
        else:
            candidates = pool.load_field(project, stain, tile_id)
            rows = [_measured_row(project, stain, tile, field, loader, r.x, r.y, candidates) for r in group.itertuples()]
        _write(project, stain, tile_id, rows)
        for row in rows:
            counts["measured" if row["measured"] else "could_not_be_measured"] += 1
    return counts


def resolve(found: pd.DataFrame | None, x: float, y: float) -> pd.Series | None:
    """The added-object row at a label's position (None when it is not measured yet)."""
    i = pool.nearest(found, x, y)
    return None if i is None else found.iloc[i]


def summary(project: Project, stain: str, labels: pd.DataFrame) -> dict[str, int]:
    """How the additions of a stain stand: measured, could not be measured, still waiting to be measured (additions that
    turned out to be candidates are left out)."""
    todo = additions(project, stain, labels)
    found = {t: g for t, g in all_added(project, stain).groupby("tile_id")}
    states = []
    for r in todo.itertuples():
        hit = resolve(found.get(r.tile_id), r.x, r.y)
        if hit is None or pd.isna(hit.candidate_x):
            states.append("waiting" if hit is None else "measured" if is_true(hit.measured) else "could_not_be_measured")
    return {"objects": len(states), **{k: states.count(k) for k in ("measured", "could_not_be_measured", "waiting")}}


def is_true(value) -> bool:
    return str(value) == "True" or value is True
