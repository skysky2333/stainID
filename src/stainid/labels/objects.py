"""The labelled objects of a set of labels, one row per object: what people said, the AI's call and the object's size.

Labels from several people about one object (same field, same rounded position, as in `store.consensus`) make one
object. Positions are in each label's own frame (`labels.frames`), tile-frame imports already moved into the field
picture. The AI's call and the object's radius come from the candidate pool at that position, else from the
pipeline's published objects of the field; neither is known for an object nobody detected (a person added it).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stainid import cache
from stainid.labels import frames, pool
from stainid.labels.tasks import SKIP, ai_value
from stainid.outputs import objects as pipeline_objects
from stainid.project import Project

COLUMNS = ["tile_id", "x", "y", "value", "values", "labels", "n", "sources", "time", "ai_value", "ai_probability", "radius_px"]


def group(labels: pd.DataFrame) -> pd.DataFrame:
    """One row per object of these class labels: majority value ("" when people are split), every label, newest time."""
    if labels.empty:
        return pd.DataFrame(columns=COLUMNS[:9])
    rows = labels.assign(rx=labels.x.astype(float).round().astype(int), ry=labels.y.astype(float).round().astype(int),
                         t=pd.to_numeric(labels.time, errors="coerce"))
    out = []
    for (tile, _, _), g in rows.groupby(["tile_id", "rx", "ry"], sort=True):
        counts = g.value[~g.value.isin(SKIP)].value_counts()
        top = "" if counts.empty or (len(counts) > 1 and counts.iloc[0] == counts.iloc[1]) else str(counts.index[0])
        if counts.empty:
            top = str(g.value.iloc[-1])  # only "unsure"
        out.append({"tile_id": tile, "x": float(g.x.iloc[0]), "y": float(g.y.iloc[0]), "value": top, "values": sorted(set(g.value)),
                    "labels": [{"value": r.value, "source": r.source, "time": None if pd.isna(r.t) else float(r.t)} for r in g.itertuples()],
                    "n": len(g), "sources": sorted(set(g.source)), "time": float(g.t.max()) if g.t.notna().any() else None})
    return pd.DataFrame(out, columns=COLUMNS[:9])


def _published(project: Project, tiles: pd.DataFrame, tile_id: str, stain: str) -> tuple[pd.DataFrame, bool]:
    """The pipeline's objects of one field (field-picture pixels) and whether detection has looked at the field at all,
    kept until its output file changes."""
    tile = tiles[tiles.tile_id == tile_id]
    if tile.empty:
        return pd.DataFrame(), False
    source = (project.output("neun") / "tiles" / f"{tile_id}_objects.csv" if stain == "NeuN"
              else project.output("fields") / "parts" / f"{tile.iloc[0].core_id}_{stain}_objects.csv")
    return cache.memo(("published objects", str(project.root), tile_id, stain), [source],
                      lambda: (pipeline_objects(project, tile, stain), source.exists()))


def with_ai(project: Project, tiles: pd.DataFrame, objects: pd.DataFrame, stain: str) -> pd.DataFrame:
    """The objects with the AI's call, its probability and the object's radius. An object in a field detection has
    looked at but where it found nothing is the AI's negative; None where detection has not run (or for a picture
    outside the project's tiles with no candidate there)."""
    if objects.empty:
        return objects.assign(ai_value=[], ai_probability=[], radius_px=[])
    candidates = pool.all_objects(project, stain)
    by_tile = {t: g.reset_index(drop=True) for t, g in candidates.groupby("tile_id")}
    calls, probabilities, radii = [], [], []
    for row in objects.itertuples():
        found = by_tile.get(row.tile_id)
        i = pool.nearest(found, row.x, row.y)
        if i is not None:
            hit = found.iloc[i]
            calls.append(str(hit.ai_value))
            probabilities.append(None if pd.isna(hit.ai_probability) else float(hit.ai_probability))
            radii.append(None if pd.isna(hit.radius_px) else float(hit.radius_px))
            continue
        published, analysed = _published(project, tiles, row.tile_id, stain)
        j = pool.nearest(published, row.x, row.y) if not published.empty else None
        if j is None:
            calls.append(ai_value(stain, None) if analysed else None)
            probabilities.append(None)
            radii.append(None)
            continue
        hit = published.iloc[j]
        mature = hit.get("mature", True)
        calls.append(ai_value(stain, hit.model_class, bool(mature) if pd.notna(mature) else True))
        probabilities.append(None if pd.isna(hit.probability) else float(hit.probability))
        radii.append(float(hit.radius_px) if np.isfinite(hit.radius_px) else None)
    return objects.assign(ai_value=calls, ai_probability=probabilities, radius_px=radii)


def labelled(project: Project, tiles: pd.DataFrame, labels: pd.DataFrame, stain: str) -> pd.DataFrame:
    """Every object these labels (of any task) give a class of this stain, in their own frames, with the AI's call."""
    mine = labels[(labels.task == "class") & (labels.stain == stain) & labels.x.notna()]
    mine = frames.in_field_frame(mine, tiles, project.context_px)
    return with_ai(project, tiles, group(mine), stain)
