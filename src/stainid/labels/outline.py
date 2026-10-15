"""Outline sessions ("Outline every object"): people outline every object of the stain in whole fields, whatever the
pipeline proposed, so that `stainid.segmentation` can measure how well detection finds and outlines objects.

A session of kind `outline` lists fields like a whole-field review. Its labels are kept apart from the class labels
that models learn from, so the check stays independent of training:
- `object`: one outlined object: `points` (its polygon in field pixels), `value` (its class), `x`, `y` (the polygon's
  centre). Its id is `object:<session>:<field>:<ref>`, `ref` chosen by the page, so an object keeps its id while its
  outline is edited.
- `complete`: the person confirmed that every object in the field is outlined (`complete:<session>:<field>`). Only
  complete fields are scored.

Fields are chosen to be ready (`labels.readiness`), on the chosen TMAs, spread over TMA x diagnosis group x region and
over how dense the stain is (`density_levels`: sparse, medium, dense thirds of the AI's objects per mm² of tissue).
"""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from stainid import cache
from stainid.labels.tasks import CLASSES, positive
from stainid.project import Project

KIND = "outline"
UNSURE = "unsure"
DEFAULT_FIELDS = 40
DEFAULT_TMAS = tuple(str(n) for n in range(1, 8))  # the study's LIP-1 to LIP-7
LEVELS = ("sparse", "medium", "dense")
# where the AI's object count and the tissue area of each field are (written by the detection steps)
COUNTS = {"NeuN": ("neun_positive_profile_count", "tissue_area_mm2"), "6E10": ("v2_plaque_count", "v2_tissue_area_mm2"),
          "AT8": ("tau_neuron_count", "v2_tissue_area_mm2")}


def title(stain: str) -> str:
    return f"Segmentation check – {stain}"


def classes(stain: str) -> list[str]:
    """What an outlined object can be: the stain's object classes, then "unsure" (left out of the scores)."""
    return [c for c in CLASSES[stain] if positive(stain, c)] + [UNSURE]


def object_id(session: str, tile_id: str, ref: str) -> str:
    return f"object:{session}:{tile_id}:{ref}"


def complete_id(session: str, tile_id: str) -> str:
    return f"complete:{session}:{tile_id}"


def centre(points: list[list[float]]) -> tuple[float, float]:
    """The centroid of a polygon's area (its corners' mean when it has none)."""
    xy = np.asarray(points, float)
    x, y = xy[:, 0], xy[:, 1]
    cross = x * np.roll(y, -1) - np.roll(x, -1) * y
    area = cross.sum() / 2
    if abs(area) < 1e-9:
        return float(x.mean()), float(y.mean())
    return float(((x + np.roll(x, -1)) * cross).sum() / (6 * area)), float(((y + np.roll(y, -1)) * cross).sum() / (6 * area))


def object_record(session: dict, person: str, tile_id: str, ref: str, value: str, points: list[list[float]] | None, deleted: bool) -> dict:
    """The label of one outlined object (or its deletion), checked."""
    if not ref or len(ref) > 40 or not ref.replace("-", "").replace("_", "").isalnum():
        raise ValueError("an object needs a short id (letters, digits, - and _)")
    record = {"id": object_id(session["name"], tile_id, ref), "source": person, "stain": session["stain"], "task": "object",
              "tile_id": tile_id, "session": session["name"], "deleted": deleted, "value": value}
    if deleted:
        return record
    if value not in classes(session["stain"]):
        raise ValueError(f"unknown class {value!r}")
    if not points or len(points) < 3 or any(len(p) != 2 for p in points):
        raise ValueError("an outline needs at least three [x, y] corners")
    points = [[round(float(px), 1), round(float(py), 1)] for px, py in points]
    x, y = centre(points)
    return {**record, "points": points, "x": round(x, 1), "y": round(y, 1)}


def complete_record(session: dict, person: str, tile_id: str, complete: bool) -> dict:
    return {"id": complete_id(session["name"], tile_id), "source": person, "stain": session["stain"], "task": "complete",
            "tile_id": tile_id, "session": session["name"], "deleted": not complete}


def objects(labels: pd.DataFrame, session: str, tile_id: str | None = None, person: str | None = None) -> pd.DataFrame:
    """The outlined objects of a session (of one field, of one person)."""
    mine = labels[(labels.task == "object") & (labels.session == session)]
    if tile_id is not None:
        mine = mine[mine.tile_id == tile_id]
    if person is not None:
        mine = mine[mine.source == person]
    return mine


def finished(labels: pd.DataFrame, session: str) -> dict[str, set[str]]:
    """The fields of a session each person marked complete."""
    done = labels[(labels.task == "complete") & (labels.session == session)]
    return {tile: set(group.source) for tile, group in done.groupby("tile_id")}


def started(labels: pd.DataFrame, session: str) -> dict[str, set[str]]:
    """Who has outlined anything on each field of a session."""
    return {tile: set(group.source) for tile, group in objects(labels, session).groupby("tile_id")}


# ---------- choosing the fields ----------

def _features(project: Project, stain: str, cores: Iterable[str]) -> pd.DataFrame:
    """The AI's object count and tissue area of every field of these cores (from the detection steps' field tables)."""
    count, area = COUNTS[stain]
    if stain == "NeuN":
        path = project.output("neun") / "tile_features.csv"
        rows = cache.rows(path, ("tile_id", count, area))
    else:
        parts = project.output("fields") / "parts"
        rows = [r for core in sorted(set(cores)) for r in cache.rows(parts / f"{core}_{stain}_features.csv", ("tile_id", count, area))]
    frame = pd.DataFrame(rows, columns=["tile_id", count, area])
    return pd.DataFrame({"tile_id": frame.tile_id, "count": pd.to_numeric(frame[count], errors="coerce"),
                         "area": pd.to_numeric(frame[area], errors="coerce")}).drop_duplicates("tile_id")


def field_density(project: Project, stain: str, tiles: pd.DataFrame) -> pd.Series:
    """The AI's objects per mm² of tissue in each of these fields (by tile id; NaN where the detection step wrote no
    count)."""
    found = _features(project, stain, tiles.core_id).set_index("tile_id")
    density = found["count"] / found["area"].where(found["area"] > 0)
    return density.reindex(tiles.tile_id)


def density_levels(density: pd.Series, seed: int = 0) -> pd.Series:
    """Each field's third of the stain's density: sparse, medium or dense ("unknown" without a count). Ties (many
    fields without any object) are broken at random, so each third holds the same number of fields."""
    known = density.dropna()
    out = pd.Series("unknown", index=density.index, dtype=object)
    if known.empty:
        return out
    shuffled = known.sample(frac=1, random_state=seed)
    ranks = shuffled.rank(method="first").to_numpy()
    out.loc[shuffled.index] = [LEVELS[min(2, int(3 * (r - 1) / len(shuffled)))] for r in ranks]
    return out


def default_tmas(tmas: Iterable[str]) -> list[str]:
    """The TMAs a check uses unless told otherwise: TMAs 1 to 7 where the project has them, else every TMA."""
    present = sorted(set(map(str, tmas)), key=lambda t: (len(t), t))
    chosen = [t for t in present if t in DEFAULT_TMAS]
    return chosen or present


def _interleave(groups: list[list]) -> list:
    """[a1, b1, c1, a2, b2, …]: the first of every group, then the second, …"""
    out, depth = [], max((len(g) for g in groups), default=0)
    for i in range(depth):
        out += [g[i] for g in groups if i < len(g)]
    return out


def pick_fields(tiles: pd.DataFrame, levels: pd.Series, n: int, seed: int, avoid: set[str] = frozenset()) -> list[str]:
    """n fields spread over TMA x diagnosis group x region (cells taken in turn, TMAs and groups interleaved) and over
    the density thirds (each pick takes the third picked least so far that the cell has), one core at a time where
    possible, fields in `avoid` (labelled already) last. Returned in shuffled order."""
    if tiles.empty or n <= 0:
        return []
    rng = np.random.default_rng(seed)
    frame = tiles.assign(level=levels.reindex(tiles.tile_id).fillna("unknown").to_numpy(), avoided=tiles.tile_id.isin(avoid).to_numpy())
    frame = frame.sample(frac=1, random_state=seed).fillna({"disease_group": "", "region": ""})
    by_tma: dict[str, list[list[tuple]]] = {}
    for tma, part in frame.groupby("tma", sort=False):
        by_group = [[(tma, group, region) for region in part[part.disease_group == group].region.unique()]
                    for group in part.disease_group.unique()]
        by_tma[tma] = _interleave(by_group)
    order = _interleave([by_tma[t] for t in sorted(by_tma, key=lambda t: (len(str(t)), str(t)))])
    left = {cell: part for cell, part in frame.groupby(["tma", "disease_group", "region"], sort=False)}
    counts = dict.fromkeys([*LEVELS, "unknown"], 0)
    cores: set[str] = set()
    picked: list[str] = []
    while len(picked) < n and any(len(part) for part in left.values()):
        for cell in order:
            part = left[cell]
            if len(picked) >= n or part.empty:
                continue
            present = [lv for lv in (*LEVELS, "unknown") if (part.level == lv).any()]
            level = min(present, key=lambda lv: (lv == "unknown", counts[lv]))
            options = part[part.level == level]
            options = options.assign(core_used=options.core_id.isin(cores)).sort_values(["avoided", "core_used"], kind="stable")
            choice = options.iloc[0]
            picked.append(choice.tile_id)
            counts[level] += 1
            cores.add(choice.core_id)
            left[cell] = part[part.tile_id != choice.tile_id]
    return [str(t) for t in rng.permutation(picked)]
