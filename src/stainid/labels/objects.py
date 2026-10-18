"""The labelled objects of a set of labels, one row per object: what people said, the AI's call and the object's size.

Labels about one object make one row, keyed as every reader keys objects (`store.object_keys` through
`decisions.keys`: the candidate the label names, else the candidate at its position, else its field and rounded
position), and counted by the one rule (each person's latest label, `store.latest_say`; the majority, `store.tally`). Positions are in each label's own frame (`labels.frames`), tile-frame imports already moved into the field
picture. The AI's call and the object's radius come from the candidate pool at that position, else from the
pipeline's published objects of the field; neither is known for an object nobody detected (a person added it).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stainid import cache
from stainid.labels import frames, pool, store
from stainid.labels import vocabulary as vocab
from stainid.labels.tasks import ai_value
from stainid.labels.vocabulary import Vocabulary
from stainid.outputs import objects as pipeline_objects
from stainid.project import Project

COLUMNS = ["tile_id", "x", "y", "value", "values", "labels", "n", "sources", "time", "ai_value", "ai_probability", "radius_px"]
GROUPED = [*COLUMNS[:9], "identity", "subtype", "object"]


def group(labels: pd.DataFrame, vocabulary: Vocabulary, keys: pd.Series | None = None) -> pd.DataFrame:
    """One row per object of these class labels (of one stain), the objects keyed by `keys` (by label index; default
    `store.object_keys` of the labels alone): the majority class of each person's latest label (`store.tally`) as
    `Vocabulary.display` names it (identity, and the subtype most of the people who typed it gave, or
    `Vocabulary.disputed` when they are split on the type; "" when people are split on the identity; the class said
    when everyone said "unsure"), its `identity` and `subtype` ids, every label (the history), the number of people,
    newest time."""
    if labels.empty:
        return pd.DataFrame(columns=GROUPED)
    rows = store.classes(labels, vocabulary.stain, vocabulary)
    keys = store.object_keys(rows) if keys is None else keys.reindex(rows.index)
    rows = rows.assign(object=keys.to_numpy(object), t=pd.to_numeric(rows.time, errors="coerce"),
                       shown=[vocabulary.display(i, s) for i, s in zip(rows.identity, rows.subtype)])
    latest = store.latest_say(rows, keys)
    counted = store.tally(latest).set_index("object") if len(latest) else pd.DataFrame()
    out = []
    for key, g in rows.groupby("object", sort=False):
        mine = latest[latest.object == key]
        if key in counted.index:
            identity = str(counted.at[key, "value"])
            typed = mine.subtype[(mine.identity == identity) & (mine.subtype != "") & (mine.role != vocab.SKIP)] if identity else mine.subtype.iloc[:0]
            subtype = str(counted.at[key, "subtype"]) if identity else ""
        else:  # only "unsure": that
            identity, subtype, typed = str(mine.identity.iloc[-1]), "", mine.subtype.iloc[:0]
        shown = "" if not identity else vocabulary.disputed(identity) if len(typed) and not subtype else vocabulary.display(identity, subtype)
        out.append({"tile_id": g.tile_id.iloc[0], "x": float(g.x.iloc[0]), "y": float(g.y.iloc[0]), "value": shown,
                    "values": sorted(set(g.shown)), "identity": identity, "subtype": subtype, "object": key,
                    "labels": [{"value": r.shown, "identity": r.identity, "subtype": r.subtype, "source": r.source, "time": None if pd.isna(r.t) else float(r.t)}
                               for r in g.itertuples()],
                    "n": int(mine.source.nunique()), "sources": sorted(set(g.source)), "time": float(g.t.max()) if g.t.notna().any() else None})
    return pd.DataFrame(out, columns=GROUPED).sort_values(["tile_id", "x", "y"], kind="stable").reset_index(drop=True)


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
    from stainid.labels import decisions  # noqa: PLC0415

    mine = labels[(labels.task == "class") & (labels.stain == stain) & labels.x.notna()]
    mine = frames.in_field_frame(mine, tiles, project.context_px)
    return with_ai(project, tiles, group(mine, vocab.of(project, stain), decisions.keys(project, stain, mine, tiles)), stain)
