"""The candidate pool: every object the detection step considers in a field, with its features and the AI's call.

Candidates and features come from exactly the code the detection pipeline runs (training/candidates.py), so a label on
a pool object can train the stain model directly. The AI's call is the published result for that field (the
pipeline output at the same position); a candidate the pipeline did not keep is a negative. One CSV per field:
`pool/<stain>/<tile_id>.csv`. Objects imported from earlier label sets on images outside the project's tiles live in
the same place, marked incomplete.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load

from stainid.labels.tasks import MODEL_KEY, NEGATIVE, ai_value
from stainid.outputs import objects as pipeline_objects
from stainid.project import Project
from stainid.sampling.tiles import known_tile_rows
from stainid.tables import write_text_atomic
from stainid.training import candidates

MATCH_PX = 4.0
KEEP = ("candidate_source", "equivalent_diameter_um", "normalized_inner_dab", "soma_dab_p90", "soma_area_um2")
GENERIC_POSITIVE = {"NeuN": "NeuN+ neuron", "6E10": "plaque", "AT8": "tau+ neuron"}


def folder(project: Project, stain: str) -> Path:
    return project.output("reviews") / "pool" / stain


def path(project: Project, stain: str, tile_id: str) -> Path:
    return folder(project, stain) / f"{tile_id}.csv"


def feature_names(project: Project, stain: str) -> list[str] | None:
    names = folder(project, stain) / "features.json"
    return json.loads(names.read_text())["feature_names"] if names.exists() else None


def _check_names(project: Project, stain: str, names: list[str]) -> None:
    known = feature_names(project, stain)
    if known is None:
        folder(project, stain).mkdir(parents=True, exist_ok=True)
        write_text_atomic(folder(project, stain) / "features.json", json.dumps({"feature_names": names}))
    elif known != names:
        raise ValueError(f"The {stain} candidate features changed since the pool was built; move {folder(project, stain)} aside and prepare again")


def model_probability(project: Project, stain: str, matrix: np.ndarray, names: list[str]) -> tuple[np.ndarray, float]:
    """The active model's probability for each candidate, and its decision threshold."""
    bundle = load(project.model(MODEL_KEY[stain]))
    if stain == "NeuN":
        return bundle["classifier"].predict_proba(matrix[:, bundle["usable_features"]])[:, 1], float(bundle["context_threshold"])
    if stain == "6E10":
        return bundle["identity_classifier"].predict_proba(matrix)[:, 1], float(bundle["identity_threshold"])
    return bundle["classifier"].predict_proba(matrix)[:, 1], float(bundle["probability_threshold"])


def _number(value) -> float:
    return np.nan if value is None or value == "" else float(value)


def _radius(row: dict, pixel_size_um: float) -> float:
    diameter = _number(row.get("equivalent_diameter_um"))
    if not np.isfinite(diameter):
        area = next((a for a in (_number(row.get("soma_area_um2")), _number(row.get("area_um2"))) if np.isfinite(a)), 50.0)
        diameter = 2 * np.sqrt(area / np.pi)
    return 0.5 * diameter / pixel_size_um


def _ai_calls(project: Project, tile: dict, x: np.ndarray, y: np.ndarray) -> list[str]:
    stain = tile["stain"]
    found = pipeline_objects(project, pd.DataFrame([tile]), stain)
    if found.empty:
        return [NEGATIVE[stain]] * len(x)
    fx, fy = found.x.to_numpy(float), found.y.to_numpy(float)
    calls = []
    for cx, cy in zip(x, y):
        d = np.hypot(fx - cx, fy - cy)
        hit = found.iloc[int(d.argmin())] if d.min() <= MATCH_PX else None
        calls.append(ai_value(stain, None if hit is None else hit.model_class, None if hit is None else bool(hit.get("mature", True))))
    return calls


def build(project: Project, tile: dict, loader: candidates.FieldLoader) -> pd.DataFrame:
    stain = tile["stain"]
    field = loader.load(tile)
    rows, matrix, names = candidates.EXTRACT[stain](loader, field)
    if not rows:
        frame = pd.DataFrame(columns=["x", "y", "radius_px", "ai_value", "ai_probability", "tma", "complete"])
    else:
        _check_names(project, stain, names)
        probability, _ = model_probability(project, stain, matrix, names)
        x = np.array([float(r["centroid_x_px"]) for r in rows])
        y = np.array([float(r["centroid_y_px"]) for r in rows])
        frame = pd.DataFrame({"x": x.round(1), "y": y.round(1), "radius_px": [round(_radius(r, field.pixel_size_um), 1) for r in rows],
                              "ai_value": _ai_calls(project, tile, x, y), "ai_probability": probability.round(4), "tma": str(tile["tma"]), "complete": True})
        frame = pd.concat([frame, pd.DataFrame([{k: r.get(k) for k in KEEP if k not in names} for r in rows]), pd.DataFrame(matrix, columns=names)], axis=1)
    target = path(project, stain, tile["tile_id"])
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=False)
    return frame


def load_field(project: Project, stain: str, tile_id: str) -> pd.DataFrame | None:
    target = path(project, stain, tile_id)
    return pd.read_csv(target, dtype={"tma": str}) if target.exists() else None


def prepared(project: Project, stain: str) -> set[str]:
    """Fields whose full pool is built (not just objects imported from earlier label sets)."""
    out = set()
    for csv in folder(project, stain).glob("*.csv") if folder(project, stain).exists() else []:
        with csv.open() as handle:
            header = handle.readline().rstrip("\n").split(",")
            first = handle.readline().split(",")
        if "complete" in header and (len(first) < len(header) or first[header.index("complete")] == "True"):
            out.add(csv.stem)
    return out


def prepare(project: Project, stain: str, tile_ids: list[str], redo: bool = False) -> None:
    """Build the pool for these fields (skipping finished ones unless redo)."""
    tiles = known_tile_rows(project)
    done = set() if redo else prepared(project, stain)
    todo = [t for t in tile_ids if t not in done]
    loader = candidates.FieldLoader(project)
    for number, tile_id in enumerate(todo, start=1):
        frame = build(project, tiles[tile_id], loader)
        print(f"[{number}/{len(todo)}] {tile_id}: {len(frame)} candidates", flush=True)
    print(f"{len(tile_ids) - len(todo)} of {len(tile_ids)} fields were already prepared", flush=True)


def add_imported(project: Project, stain: str, tile_id: str, rows: pd.DataFrame) -> None:
    """Objects from an earlier label set (with their saved features), unless the field's full pool already exists."""
    target = path(project, stain, tile_id)
    old = load_field(project, stain, tile_id)
    if old is not None and old.complete.astype(str).eq("True").all() and len(old):
        return
    frame = rows if old is None else pd.concat([old, rows], ignore_index=True)
    frame = frame.assign(rx=frame.x.round(), ry=frame.y.round()).drop_duplicates(["rx", "ry"]).drop(columns=["rx", "ry"])
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=False)


def generic_calls(stain: str, probability: np.ndarray, threshold: float) -> list[str]:
    return [GENERIC_POSITIVE[stain] if p >= threshold else NEGATIVE[stain] for p in probability]


def nearest(pool: pd.DataFrame, x: float, y: float, tolerance: float = MATCH_PX) -> int | None:
    if pool is None or pool.empty:
        return None
    d = np.hypot(pool.x.to_numpy(float) - x, pool.y.to_numpy(float) - y)
    return int(d.argmin()) if d.min() <= tolerance else None


_ALL: dict[tuple[str, str], tuple[tuple, pd.DataFrame]] = {}


def _files(project: Project, stain: str) -> list[Path]:
    return sorted(folder(project, stain).glob("*.csv")) if folder(project, stain).exists() else []


def signature(project: Project, stain: str) -> tuple:
    """Changes whenever a pool file of this stain is added or rewritten."""
    return tuple((f.name, f.stat().st_mtime_ns) for f in _files(project, stain))


def all_objects(project: Project, stain: str) -> pd.DataFrame:
    """Every pool object of a stain (position, size, AI call), reloaded only when a pool file changes."""
    files = _files(project, stain)
    stamp = tuple((f.name, f.stat().st_mtime_ns) for f in files)
    cache_key = (str(project.root), stain)
    if cache_key in _ALL and _ALL[cache_key][0] == stamp:
        return _ALL[cache_key][1]
    columns = ["x", "y", "radius_px", "ai_value", "ai_probability", "tma", "complete"]
    frames = [pd.read_csv(f, usecols=lambda c: c in columns, dtype={"tma": str}).assign(tile_id=f.stem) for f in files]
    frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=[*columns, "tile_id"])
    _ALL[cache_key] = (stamp, frame)
    return frame
