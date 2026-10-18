"""What people said against what detection called: the one comparison the Cores view's "Annotations" layer and the
Labels overview read, built on the decisions scoring uses (`labels.decisions.state`: the latest class of each person on
each candidate, objects in excluded areas left out).

Per candidate someone decided: detection's call (identity, subtype), the people's consensus and each person's answer,
and how they compare (`kind`): "agree", "changed" (another class or type, also a candidate detection rejected that
people call an object), "rejected" (people say it is not an object; detection accepted it) or "unsure" (only "unsure"
answers). Also the outlines people fixed (`outline` labels on a candidate) and the objects people added that detection
missed (class labels on no candidate). Labels only train and evaluate models: nothing here changes what the analysis
counts.
"""
from __future__ import annotations

import pandas as pd

from stainid import candidates
from stainid import exclusions as excluded_areas
from stainid.labels import candidate_store, decisions, frames, objectedit, store
from stainid.labels import vocabulary as vocab

KINDS = ("agree", "changed", "rejected", "unsure")
COLUMNS = ["candidate_id", "core_id", "x_core", "y_core", "kind", "detection_identity", "detection_subtype", "identity", "subtype", "people"]


def _core_of(candidate_id: str) -> str:
    try:
        return candidates.parse_id(candidate_id)[1]
    except ValueError:
        return ""


def versus_detection(project, stain: str, tiles: pd.DataFrame, core_id: str | None = None) -> pd.DataFrame:
    """One row per candidate people decided (of one core image, or every one): `COLUMNS`; `people` a list of
    {person, identity, subtype, role, time}."""
    state = decisions.state(project, stain, tiles)
    rows = state.rows
    if rows.empty:
        return pd.DataFrame(columns=COLUMNS)
    rows = rows.assign(core_id=rows.candidate.map(_core_of))
    if core_id is not None:
        rows = rows[rows.core_id == str(core_id)]
    if rows.empty:
        return pd.DataFrame(columns=COLUMNS)
    if core_id is None:  # every core: the compact copies matching keeps, not each core's whole table
        found = candidate_store.core_calls(project, stain, set(rows.core_id))
    else:
        found = candidate_store.for_cores(project, stain, set(rows.core_id), ["candidate_id", "x_core", "y_core", "decision", "class", "subtype"])
    found = found.drop_duplicates("candidate_id").set_index("candidate_id") if len(found) else found
    words = vocab.of(project, stain)
    out = []
    for candidate, group in rows.groupby("candidate", sort=False):
        if candidate not in found.index:
            continue
        row = found.loc[candidate]
        identity, subtype = candidate_store.ai_call(words, row.decision, row["class"], row.subtype)
        said = group[group.role != vocab.SKIP]
        people = [{"person": r.source, "identity": r.identity, "subtype": r.subtype or "", "role": r.role, "time": float(r.time)}
                  for r in group.sort_values("time").itertuples()]
        if said.empty:
            kind, call = "unsure", ("", "")
        else:
            call = state.calls.get(candidate, (said.identity.iloc[-1], said.subtype.iloc[-1] or ""))
            if (call[0], call[1] or "") == (identity, subtype or ""):
                kind = "agree"
            elif candidate_store.accepted(row.decision) and words.role(call[0]) == vocab.NEGATIVE:
                kind = "rejected"
            else:
                kind = "changed"
        out.append({"candidate_id": candidate, "core_id": str(group.core_id.iloc[0]), "x_core": float(row.x_core), "y_core": float(row.y_core),
                    "kind": kind, "detection_identity": identity, "detection_subtype": subtype or "", "identity": call[0], "subtype": call[1] or "",
                    "people": people})
    return pd.DataFrame(out, columns=COLUMNS)


def _object_labels(project, stain: str, task: str) -> pd.DataFrame:
    labels = excluded_areas.drop_excluded(project, store.current(project.output("reviews")))
    if labels.empty:
        return labels
    return labels[(labels.stain == stain) & (labels.task == task)]


def fixed_outlines(project, stain: str, tiles: pd.DataFrame, core_id: str | None = None) -> pd.DataFrame:
    """Outlines people fixed (precise editing, another mask, drawing), the latest per person and candidate: candidate_id,
    person, time, points (core pixels) or mask (the index of the Segment Anything mask chosen)."""
    rows = _object_labels(project, stain, "outline")
    if rows.empty or "candidate" not in rows:
        return pd.DataFrame(columns=["candidate_id", "person", "time", "points", "mask"])
    rows = rows[(rows.candidate.fillna("").astype(str).str.len() > 0) & ~rows.confirmed.fillna(False).astype(bool)]
    if core_id is not None:
        rows = rows[rows.candidate.map(_core_of) == str(core_id)]
    rows = rows.sort_values("time").drop_duplicates(["source", "candidate"], keep="last")
    boxes = tiles.drop_duplicates("tile_id").set_index("tile_id") if not tiles.empty else pd.DataFrame()
    out = []
    for r in rows.itertuples():
        value = str(r.value or "")
        mask = int(value[5:]) if value.startswith("mask ") and value[5:].isdigit() else None
        points = None
        if mask is None and r.points is not None and r.tile_id in boxes.index:
            x0, y0 = frames.field_origin(boxes.loc[r.tile_id], project.context_px)
            points = [[round(float(x) + x0, 1), round(float(y) + y0, 1)] for x, y in r.points]
        out.append({"candidate_id": r.candidate, "person": r.source, "time": float(r.time or 0), "points": points, "mask": mask})
    return pd.DataFrame(out, columns=["candidate_id", "person", "time", "points", "mask"])


def added_objects(project, stain: str, tiles: pd.DataFrame, core_id: str | None = None) -> pd.DataFrame:
    """Objects people added where detection had no candidate (class labels naming none), the latest per person:
    person, time, identity, subtype, x_core, y_core, core_id and their outline (core pixels) when they drew or chose one."""
    columns = ["person", "time", "identity", "subtype", "x_core", "y_core", "core_id", "points"]
    rows = _object_labels(project, stain, "class")
    if rows.empty:
        return pd.DataFrame(columns=columns)
    named = rows.candidate.fillna("").astype(str).str.len() > 0 if "candidate" in rows else pd.Series(False, index=rows.index)
    rows = rows[~named & rows.x.notna() & ~objectedit.pieces_mask(rows)]  # pieces of a split or merge: `object_edits`
    boxes = tiles[tiles.stain == stain].drop_duplicates("tile_id").set_index("tile_id") if not tiles.empty else pd.DataFrame()
    rows = rows[rows.tile_id.isin(set(boxes.index))] if len(boxes) else rows.iloc[:0]
    if core_id is not None and len(rows):
        rows = rows[[str(boxes.loc[t, "core_id"]) == str(core_id) for t in rows.tile_id]]
    if rows.empty:
        return pd.DataFrame(columns=columns)
    words = vocab.of(project, stain)
    classes = store.classes(rows, stain, words)
    outlines = _object_labels(project, stain, "outline")
    shapes = {(r.source, r.tile_id, round(float(r.x)), round(float(r.y))): r.points for r in outlines.itertuples() if r.x == r.x and r.points is not None}
    out = []
    for r in classes.itertuples():
        if words.role(r.identity) != vocab.POSITIVE:
            continue
        tile = boxes.loc[r.tile_id]
        x0, y0 = frames.field_origin(tile, project.context_px)
        drawn = shapes.get((r.source, r.tile_id, round(float(r.x)), round(float(r.y))))
        out.append({"person": r.source, "time": float(r.time or 0), "identity": r.identity, "subtype": r.subtype or "",
                    "x_core": float(r.x) + x0, "y_core": float(r.y) + y0, "core_id": str(tile["core_id"]),
                    "points": [[round(float(x) + x0, 1), round(float(y) + y0, 1)] for x, y in drawn] if drawn else None})
    return pd.DataFrame(out, columns=columns)


def object_edits(project, stain: str, tiles: pd.DataFrame, core_id: str | None = None) -> list[dict]:
    """The splits and merges people made (`labels.objectedit`), each with who and when, the objects it replaced
    (candidate ids where they were candidates; positions in core pixels) and the new objects (type and outline in core
    pixels). Object labels only: detection's objects stay what the analysis counts."""
    labels = excluded_areas.drop_excluded(project, store.current(project.output("reviews")))
    if labels.empty or "meta" not in labels:
        return []
    mine = labels[labels.stain == stain]
    replaced = mine[mine.task == objectedit.TASK]
    if replaced.empty:
        return []
    boxes = tiles[tiles.stain == stain].drop_duplicates("tile_id").set_index("tile_id") if not tiles.empty else pd.DataFrame()

    def origin(tile_id):
        if tile_id not in boxes.index:
            return None
        return frames.field_origin(boxes.loc[tile_id], project.context_px), str(boxes.loc[tile_id, "core_id"])

    pieces = mine[(mine.task.isin(["class", "outline"])) & objectedit.pieces_mask(mine)]
    words = vocab.of(project, stain)
    out = {}
    for r in replaced.itertuples():
        meta = r.meta if isinstance(r.meta, dict) else {}
        found = origin(r.tile_id)
        if found is None or r.x != r.x:
            continue
        (x0, y0), core = found
        if core_id is not None and core != str(core_id):
            continue
        key = (r.source, str(meta.get("group", r.id)))
        edit = out.setdefault(key, {"kind": str(r.value), "person": r.source, "time": float(r.time or 0), "core_id": core, "originals": [], "pieces": []})
        edit["originals"].append({"candidate_id": r.candidate if isinstance(r.candidate, str) and r.candidate else None,
                                  "x_core": float(r.x) + x0, "y_core": float(r.y) + y0})
    for (person, group), edit in out.items():
        mine_pieces = pieces[(pieces.source == person) & (pieces.meta.map(lambda m: str(m.get("group")) if isinstance(m, dict) else "") == group)]
        shapes = {(r.tile_id, round(float(r.x)), round(float(r.y))): r.points for r in mine_pieces[mine_pieces.task == "outline"].itertuples()}
        for r in store.classes(mine_pieces, stain, words).itertuples():
            found = origin(r.tile_id)
            if found is None:
                continue
            (x0, y0), _ = found
            drawn = shapes.get((r.tile_id, round(float(r.x)), round(float(r.y))))
            edit["pieces"].append({"identity": r.identity, "subtype": r.subtype or "", "x_core": float(r.x) + x0, "y_core": float(r.y) + y0,
                                   "points": [[round(float(x) + x0, 1), round(float(y) + y0, 1)] for x, y in drawn] if drawn else None})
    return sorted(out.values(), key=lambda e: e["time"])


def summary(project, stain: str, tiles: pd.DataFrame, core_id: str | None = None) -> dict:
    """How people's labels compare with detection (of one core image, or every one): candidates they agree on, changed
    the class of, rejected, were unsure about, the objects they added and the outlines they fixed."""
    found = versus_detection(project, stain, tiles, core_id)
    counts = {k: int((found.kind == k).sum()) if len(found) else 0 for k in KINDS}
    fixed = fixed_outlines(project, stain, tiles, core_id)
    return {**counts, "added": int(len(added_objects(project, stain, tiles, core_id))), "outline_fixed": int(fixed.candidate_id.nunique()) if len(fixed) else 0}
