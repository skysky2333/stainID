"""Score the models on imported labels: the external test of stainID's detection on another lab's annotations.

The AI's objects (the v2 calls, `stainid.candidates`, where the v2 detection has run; else the published detections,
`stainid.outputs`) are compared with every rater's annotations and with
the consensus, per stain, on the images that rater annotated:

- points: an accepted object within `MATCH_UM` (or the object's own radius, if larger) of a mark finds it, one object
  per mark (closest first);
- boxes: an accepted object whose centre lies in the box finds it (also given: boxes whose best object overlaps them by
  IoU >= 0.5, the object taken as a square of its diameter);
- image labels: an image said to contain a class is found when the AI accepted any object of that class in the image.

Recall is the share of marked objects (or positive images) the AI found, precision the share of the AI's accepted
objects (or positive images) that someone marked, F1 their harmonic mean. Marks of a negative class ("not a plaque")
are not objects to find: an accepted object on one counts against precision. Where the AI gives probabilities, the AUC
says how well they rank the marked objects (or positive images) above the negative ones. The raters' agreement with each
other is given the same way (each pair, F1 for objects, Cohen's kappa for image labels), so the AI's figures can be read
against how much people agree.
"""
from __future__ import annotations

import itertools
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from stainid.imageset.annotations import CONSENSUS, MATCH_UM, folder, imported
from stainid.labels import vocabulary as vocab
from stainid.labels.tasks import ai_value, positive
from stainid.outputs import objects as pipeline_objects
from stainid.project import Project
from stainid.sampling.tiles import table
from stainid.tables import write_records, write_text_atomic

COLUMNS = ["stain", "rater", "kind", "class", "images", "marks", "found", "missed", "ai_objects", "ai_unmarked", "precision", "recall", "f1",
           "auc", "candidate_auc", "iou_recall", "same_type", "specificity", "scope"]
WHOLE_SET = 0.5  # a rater who marked something on at least this share of the annotated images read all of them
GENERIC = {"plaque", "tau+ neuron", "NeuN+ neuron"}
OBJECT = {"6E10": "plaque", "AT8": "tau+ neuron", "NeuN": "NeuN+ neuron"}


def _auc(scores: list[float], truth: list[bool]) -> float | None:
    """Area under the ROC curve (ties count half); None without both classes."""
    s, t = np.asarray(scores, float), np.asarray(truth, bool)
    if t.all() or not t.any():
        return None
    order = s.argsort(kind="mergesort")
    ranks = np.empty(len(s))
    sorted_scores = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    positives = t.sum()
    return float((ranks[t].sum() - positives * (positives + 1) / 2) / (positives * (len(t) - positives)))


def _ratio(a: float, b: float) -> float | None:
    return round(a / b, 4) if b else None


def _f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None or precision + recall == 0:
        return 0.0 if precision == 0 or recall == 0 else None
    return round(2 * precision * recall / (precision + recall), 4)


DETECTION_COLUMNS = ["core_id", "tile_id", "cx", "cy", "radius_px", "value", "accepted", "probability", "model_class"]
V2_COLUMNS = ["tile_id", "x_core", "y_core", "radius_px", "area_um2", "probability", "decision", "reason", "class", "subtype"]


def v2_detections(project: Project, stain: str, core_ids: set[str]) -> pd.DataFrame | None:
    """The v2 detection's candidates of these cores (`stainid.candidates`, after stage 4), in the frame `detections`
    gives: accepted = the call `accepted`, `value` its class and type ("plaque/compact"), `model_class` the type, or
    `vascular_or_edge` for deposits the vessel / edge rule set aside. None when no core has a v2 store."""
    from stainid import candidates as store

    frames = []
    for core in sorted(core_ids):
        if store.path(project, stain, core).exists():
            found = store.read_core(project, stain, core, V2_COLUMNS)
            if len(found):
                frames.append(found.assign(core_id=core))
    if not frames:
        return None
    found = pd.concat(frames, ignore_index=True)
    text = {c: found[c].fillna("").astype(str) if c in found else pd.Series("", index=found.index) for c in ("decision", "reason", "class", "subtype")}
    accepted = text["decision"].eq("accepted").to_numpy()
    values = [f"{c}/{t}" if a and t else c for c, t, a in zip(text["class"], text["subtype"], accepted)]
    radius = pd.to_numeric(found.radius_px, errors="coerce") if "radius_px" in found else pd.Series(np.nan, index=found.index)
    area = pd.to_numeric(found.area_um2, errors="coerce") if "area_um2" in found else pd.Series(np.nan, index=found.index)
    radius = radius.fillna(np.sqrt(area / np.pi) / project.pixel_size_um)
    model_class = np.where(text["reason"].eq("rule:vascular_or_edge"), "vascular_or_edge", np.where(text["subtype"].ne(""), text["subtype"], text["class"]))
    return pd.DataFrame({"core_id": found.core_id.to_numpy(), "tile_id": text_column(found, "tile_id"), "cx": found.x_core.to_numpy(float),
                         "cy": found.y_core.to_numpy(float), "radius_px": radius.to_numpy(float), "value": values, "accepted": accepted,
                         "probability": pd.to_numeric(found.probability, errors="coerce").to_numpy(), "model_class": model_class})


def text_column(frame: pd.DataFrame, column: str) -> np.ndarray:
    return frame[column].fillna("").astype(str).to_numpy() if column in frame else np.full(len(frame), "", dtype=object)


def detections(project: Project, stain: str, core_ids: set[str]) -> pd.DataFrame:
    """The AI's objects of these cores in core-image pixels, with whether each was accepted and its class: the v2
    detection's calls where it has run (`v2_detections`), else the published v1 detections."""
    found = v2_detections(project, stain, core_ids)
    if found is not None:
        return found
    tiles = table(project.analysis_tiles())
    tiles = tiles[(tiles.stain == stain) & tiles.core_id.isin(core_ids)]
    found = pipeline_objects(project, tiles, stain) if len(tiles) else pd.DataFrame()
    if found.empty:
        return pd.DataFrame(columns=DETECTION_COLUMNS)
    origin = tiles.drop_duplicates("tile_id").set_index("tile_id")
    ox = (origin.x_px - project.context_px).clip(lower=0)
    oy = (origin.y_px - project.context_px).clip(lower=0)
    mature = found["mature"] if "mature" in found else pd.Series(True, index=found.index)
    values = [ai_value(stain, c, bool(m) if pd.notna(m) else True) for c, m in zip(found.model_class, mature)]
    return pd.DataFrame({"core_id": origin.core_id.reindex(found.tile_id).to_numpy(), "tile_id": found.tile_id,
                         "cx": found.x.to_numpy(float) + ox.reindex(found.tile_id).to_numpy(), "cy": found.y.to_numpy(float) + oy.reindex(found.tile_id).to_numpy(),
                         "radius_px": found.radius_px.to_numpy(float), "value": values, "accepted": [positive(stain, v) for v in values],
                         "probability": pd.to_numeric(found.probability, errors="coerce").to_numpy(), "model_class": found.model_class.astype(str).to_numpy()})


def _kind(stain: str, value: str) -> str:
    """A class (a rater's, through the class map, or the AI's) as "identity/subtype" ("identity" without a type), so the
    rater's "plaque/compact" and the AI's "compact plaque" are the same kind (`labels.vocabulary`)."""
    identity, subtype = vocab.defaults(stain).resolve(value)
    return f"{identity}/{subtype}" if subtype else identity


def _typed(stain: str, value: str) -> bool:
    return bool(vocab.defaults(stain).resolve(value)[1])


def _same_class(label: str, value: str, stain: str) -> bool:
    return positive(stain, value) and (not _typed(stain, label) or _kind(stain, label) == _kind(stain, value) or not positive(stain, label))


def _match_marks(marks: pd.DataFrame, found: pd.DataFrame, project: Project, stain: str) -> tuple[list[int | None], list[int | None]]:
    """For each mark the accepted object that finds it (index into `found`) and the nearest object of any kind (for the
    AUC); one accepted object per mark, closest pairs first."""
    near = MATCH_UM[stain] / project.pixel_size_um
    accepted = {core: group for core, group in found[found.accepted].groupby("core_id")}
    pairs = []
    for i, (mark, cls) in enumerate(zip(marks.itertuples(), marks["class"])):
        if not positive(stain, cls) and cls not in GENERIC:
            continue
        candidates = accepted.get(mark.core_id)
        for j, obj in zip(candidates.index, candidates.itertuples()) if candidates is not None else ():
            if mark.kind == "box" and pd.notna(mark.x0):
                inside = mark.x0 <= obj.cx <= mark.x1 and mark.y0 <= obj.cy <= mark.y1
                distance = math.hypot(obj.cx - mark.core_x, obj.cy - mark.core_y)
                if inside:
                    pairs.append((distance, i, j))
            else:
                distance = math.hypot(obj.cx - mark.core_x, obj.cy - mark.core_y)
                if distance <= max(near, obj.radius_px if np.isfinite(obj.radius_px) else 0):
                    pairs.append((distance, i, j))
    hit: list[int | None] = [None] * len(marks)
    taken: set[int] = set()
    for _, i, j in sorted(pairs):
        if hit[i] is None and j not in taken:
            hit[i], taken = j, taken | {j}
    nearest: list[int | None] = []
    by_core = {core: group for core, group in found.groupby("core_id")}
    for mark in marks.itertuples():
        same = by_core.get(mark.core_id)
        if same is None:
            nearest.append(None)
            continue
        d = np.hypot(same.cx - mark.core_x, same.cy - mark.core_y)
        k = int(d.argmin())
        nearest.append(int(same.index[k]) if d.iloc[k] <= max(near, same.radius_px.iloc[k] if np.isfinite(same.radius_px.iloc[k]) else 0) else None)
    return hit, nearest


def _iou_recall(marks: pd.DataFrame, found: pd.DataFrame) -> float | None:
    boxes = marks[(marks.kind == "box") & marks.x0.notna()]
    if boxes.empty:
        return None
    hits = 0
    for box in boxes.itertuples():
        same = found[(found.core_id == box.core_id) & found.accepted]
        best = 0.0
        for obj in same.itertuples():
            r = obj.radius_px if np.isfinite(obj.radius_px) else 0
            ix = max(0, min(box.x1, obj.cx + r) - max(box.x0, obj.cx - r))
            iy = max(0, min(box.y1, obj.cy + r) - max(box.y0, obj.cy - r))
            union = (box.x1 - box.x0) * (box.y1 - box.y0) + 4 * r * r - ix * iy
            best = max(best, ix * iy / union if union > 0 else 0)
        hits += best >= 0.5
    return round(hits / len(boxes), 4)


def _candidate_auc(project: Project, marks: pd.DataFrame, found: pd.DataFrame, stain: str, reviewed: set[str], is_object: list[bool]) -> float | None:
    """How well the model's probability ranks the candidates a rater marked above those they did not, over every
    candidate (accepted or not) in the images the rater read: a marked candidate is the most probable one whose centre
    lies in the mark's box, or within the match distance of its point. Works with marks of objects only (no negative
    marks), where the mark AUC cannot be made."""
    pool = found[found.core_id.isin(reviewed)]
    if pool.empty or pool.probability.isna().all():
        return None
    near = MATCH_UM[stain] / project.pixel_size_um
    probability = pool.probability.fillna(0.0)
    by_core = {core: group for core, group in pool.groupby("core_id")}
    marked: set[int] = set()
    for mark, wanted in zip(marks.itertuples(), is_object):
        same = by_core.get(mark.core_id)
        if not wanted or same is None:
            continue
        if mark.kind == "box" and pd.notna(mark.x0):
            inside = same[(same.cx >= mark.x0) & (same.cx <= mark.x1) & (same.cy >= mark.y0) & (same.cy <= mark.y1)]
        else:
            inside = same[np.hypot(same.cx - mark.core_x, same.cy - mark.core_y) <= np.maximum(near, same.radius_px.fillna(0))]
        if len(inside):
            marked.add(int(probability.loc[inside.index].idxmax()))
    return _auc(probability.to_list(), [i in marked for i in pool.index])


def reviewed_images(mine: pd.DataFrame, rater: str) -> set[str]:
    """The core images a rater read, as far as the labels tell: those they marked anything on, or every annotated one
    when they marked at least `WHOLE_SET` of them (sets where every rater reads every image and an image with nothing
    marked means "nothing here"). Sets that give each rater their own images (and a few shared ones) count only the
    rater's own for precision."""
    every = set(mine.core_id)
    own = set(mine.core_id[mine.rater == rater])
    return every if len(own) >= WHOLE_SET * len(every) else own


def score_objects(project: Project, marks: pd.DataFrame, found: pd.DataFrame, stain: str, reviewed: set[str] | None = None) -> dict:
    """Recall, precision, F1 and AUC of the AI's objects against one rater's point / box / polygon marks."""
    marks = marks.reset_index(drop=True)
    reviewed = reviewed if reviewed is not None else set(marks.core_id)
    hit, nearest = _match_marks(marks, found, project, stain)
    is_object = [positive(stain, c) or c in GENERIC for c in marks["class"]]
    targets = int(sum(is_object))
    found_n = int(sum(h is not None for h, o in zip(hit, is_object) if o))
    # Raters who marked only some types (e.g. only cored plaques) say nothing about the others: precision counts the AI's
    # objects of the types they marked.
    types = {_kind(stain, c) for c, o in zip(marks["class"], is_object) if o}
    untyped = any(o and not _typed(stain, c) for c, o in zip(marks["class"], is_object))  # marked without a type: every type counts
    scope = (found.accepted & found.core_id.isin(reviewed)) & (True if untyped else found.value.map(lambda v: _kind(stain, v)).isin(types))
    in_scope = found[scope]
    matched = {h for h in hit if h is not None}
    marked = int(sum(i in matched for i in in_scope.index))
    precision, recall = _ratio(marked, len(in_scope)), _ratio(found_n, targets)
    scores = [float(found.probability.loc[n]) if n is not None and np.isfinite(found.probability.loc[n]) else 0.0 for n in nearest]
    typed = [(_kind(stain, c), _kind(stain, found.value.loc[h])) for c, h, o in zip(marks["class"], hit, is_object)
             if o and h is not None and _typed(stain, c)]
    return {"images": int(marks.image_id.nunique()), "marks": targets, "found": found_n, "missed": targets - found_n, "ai_objects": int(len(in_scope)),
            "ai_unmarked": int(len(in_scope) - marked), "precision": precision, "recall": recall, "f1": _f1(precision, recall),
            "auc": None if found.probability.isna().all() else _auc(scores, is_object),
            "candidate_auc": _candidate_auc(project, marks, found, stain, reviewed, is_object), "iou_recall": _iou_recall(marks, found),
            "same_type": _ratio(sum(c == v for c, v in typed), len(typed)), "scope": "all accepted" if untyped or not types else ", ".join(sorted(types))}


def score_negatives(project: Project, marks: pd.DataFrame, found: pd.DataFrame, stain: str) -> list[dict]:
    """Marks of things that are not the stain's object (e.g. AT8 neuritic plaques, which are not tau+ neurons), from a
    rater who marked nothing else: per class, on how many of them the AI accepted an object (a false alarm), and the
    specificity (the share it rightly left alone)."""
    out = []
    for cls, group in marks.groupby("class"):
        group = group.reset_index(drop=True)  # an outline is matched as its box: any accepted object inside it is a false alarm
        kinds = np.where(group.x0.notna(), "box", group.kind) if "x0" in group else group.kind
        as_objects = group.assign(**{"class": OBJECT[stain], "kind": kinds})
        hit, _ = _match_marks(as_objects, found, project, stain)
        alarms = sum(h is not None for h in hit)
        out.append({"class": cls, "images": int(group.image_id.nunique()), "marks": len(group), "found": alarms, "missed": len(group) - alarms,
                    "ai_objects": None, "ai_unmarked": None, "precision": None, "recall": None, "f1": None, "auc": None, "candidate_auc": None,
                    "iou_recall": None, "same_type": None, "specificity": _ratio(len(group) - alarms, len(group)),
                    "scope": f"not {OBJECT[stain]}: detection's accepted objects on them are false alarms"})
    return out


def score_vessels(project: Project, marks: pd.DataFrame, found: pd.DataFrame, reviewed: set[str]) -> dict | None:
    """Vessel (CAA) marks against the deposits the AI set aside as on a vessel or the tissue edge."""
    vessels = marks[[vocab.defaults("6E10").resolve(c)[0] == "CAA" for c in marks["class"]]].reset_index(drop=True)  # "vascular" too
    if vessels.empty or "model_class" not in found:
        return None
    flagged = found[found.model_class.astype(str).eq("vascular_or_edge")].assign(accepted=True, value="plaque")
    as_plaques = vessels.assign(**{"class": "plaque"})
    hit, _ = _match_marks(as_plaques, flagged, project, "6E10")
    found_n = sum(h is not None for h in hit)
    in_scope = flagged[flagged.core_id.isin(reviewed)]
    marked = len({h for h in hit if h is not None} & set(in_scope.index))
    precision, recall = _ratio(marked, len(in_scope)), _ratio(found_n, len(vessels))
    return {"class": "vascular", "images": int(vessels.image_id.nunique()), "marks": len(vessels), "found": found_n, "missed": len(vessels) - found_n,
            "ai_objects": int(len(in_scope)), "ai_unmarked": int(len(in_scope) - marked), "precision": precision, "recall": recall,
            "f1": _f1(precision, recall), "auc": None, "iou_recall": None, "same_type": None, "scope": "vascular or edge"}


def score_images(labels: pd.DataFrame, found: pd.DataFrame, stain: str) -> list[dict]:
    """Per class of image label: is that class in the image (rater) against any accepted object of it (AI)."""
    out = []
    images = labels.groupby("image_id")
    cores_of = labels.groupby("image_id").core_id.agg(set)
    classes = sorted({c for c in labels["class"] if positive(stain, c) or c in GENERIC})
    for cls in classes:
        truth, said, scores = [], [], []
        for image, group in images:
            in_image = found[found.core_id.isin(cores_of[image])]
            truth.append(cls in set(group["class"]))
            said.append(bool(any(_same_class(cls, v, stain) for v in in_image.value[in_image.accepted])))
            probability = in_image.probability.max() if len(in_image) else 0.0
            scores.append(float(probability) if pd.notna(probability) else 0.0)
        t, s = np.array(truth), np.array(said)
        tp, fp, fn = int((t & s).sum()), int((~t & s).sum()), int((t & ~s).sum())
        precision, recall = _ratio(tp, tp + fp), _ratio(tp, tp + fn)
        out.append({"class": cls, "images": len(t), "marks": int(t.sum()), "found": tp, "missed": fn, "ai_objects": tp + fp, "ai_unmarked": fp,
                    "precision": precision, "recall": recall, "f1": _f1(precision, recall), "auc": _auc(scores, list(t)), "iou_recall": None})
    return out


def _pair_f1(project: Project, a: pd.DataFrame, b: pd.DataFrame, stain: str) -> float | None:
    """How well two raters' marks of objects agree (F1, marks matched as for the AI) on the images both annotated."""
    shared = set(a.image_id) & set(b.image_id)
    a, b = a[a.image_id.isin(shared)], b[b.image_id.isin(shared)]
    pa = a[[positive(stain, c) or c in GENERIC for c in a["class"]]]
    pb = b[[positive(stain, c) or c in GENERIC for c in b["class"]]]
    if pa.empty and pb.empty:
        return None
    as_found = pd.DataFrame({"core_id": pb.core_id.to_numpy(), "cx": pb.core_x.to_numpy(float), "cy": pb.core_y.to_numpy(float),
                             "radius_px": [0.5 * min(r.x1 - r.x0, r.y1 - r.y0) if pd.notna(r.x0) else np.nan for r in pb.itertuples()],
                             "value": pb["class"].to_numpy(), "accepted": True, "probability": np.nan})
    hit, _ = _match_marks(pa.reset_index(drop=True), as_found, project, stain)
    matched = sum(h is not None for h in hit)
    precision, recall = _ratio(matched, len(pb)), _ratio(matched, len(pa))
    return _f1(precision, recall)


def _kappa(a: list[bool], b: list[bool]) -> float | None:
    a, b = np.array(a), np.array(b)
    if not len(a):
        return None
    observed = float((a == b).mean())
    expected = float(a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean()))
    return round((observed - expected) / (1 - expected), 4) if expected < 1 else None


def agreement(project: Project, placed: pd.DataFrame) -> list[dict]:
    """Every pair of raters, per stain: F1 between their object marks, Cohen's kappa between their image labels."""
    out = []
    people = placed[placed.rater != CONSENSUS]
    for stain, mine in people.groupby("stain"):
        raters = sorted(set(mine.rater))
        for first, second in itertools.combinations(raters, 2):
            a, b = mine[mine.rater == first], mine[mine.rater == second]
            objects_a, objects_b = a[a.kind != "image_label"], b[b.kind != "image_label"]
            row = {"stain": stain, "rater": first, "other": second, "shared_images": len(set(a.image_id) & set(b.image_id)),
                   "object_f1": _pair_f1(project, objects_a, objects_b, stain) if len(objects_a) and len(objects_b) else None, "image_kappa": None}
            ia, ib = a[a.kind == "image_label"], b[b.kind == "image_label"]
            shared = sorted(set(ia.image_id) & set(ib.image_id))
            if shared:
                classes = sorted(set(ia["class"]) | set(ib["class"]))
                said_a = [c in set(ia["class"][ia.image_id == i]) for i in shared for c in classes]
                said_b = [c in set(ib["class"][ib.image_id == i]) for i in shared for c in classes]
                row["image_kappa"] = _kappa(said_a, said_b)
            out.append(row)
    return out


def _plural(kind: str) -> str:
    """A kind in words: "plaque/compact" -> "compact plaques", "tau+ neuron/mature tangle" -> "mature tangle tau+ neurons"."""
    identity, _, subtype = kind.partition("/")
    return f"{subtype} {identity}s" if subtype else f"{identity}s"


def summary_text(stain: str, row: dict, people: list[dict]) -> str:
    """One sentence in plain words for the Models page."""
    who = "the consensus of the raters" if row["rater"] == CONSENSUS else row["rater"]
    if row.get("kind") == "negatives":
        return (f"{stain}: {who} marked {row['marks']:,} {row['class']}s, which are not {OBJECT[stain]}s; stainID detection accepted a {OBJECT[stain]} "
                f"on {row['found']:,} of them ({1 - row['specificity']:.0%}).") if row.get("specificity") is not None else ""
    if row.get("recall") is None:
        return ""
    scope = row.get("scope") or "all accepted"
    thing = ("images" if row["kind"] == "image" else "vessel deposits (CAA)" if row["class"] == "vascular"
             else _plural(scope) if scope != "all accepted" and "," not in scope else "objects")
    text = (f"{stain}: of the {row['marks']:,} {thing} {who} marked, stainID detection found {row['recall']:.0%}"
            + (f"; {row['precision']:.0%} of what detection found was marked (F1 {row['f1']:.2f})" if row.get("precision") is not None else "") + ".")
    values = [p["object_f1"] if row["kind"] != "image" else p["image_kappa"] for p in people if p["stain"] == stain]
    values = [v for v in values if v is not None]
    if values:
        text += f" The raters agreed with each other at {'F1' if row['kind'] != 'image' else 'kappa'} {np.mean(values):.2f} on average."
    return text


def score(project: Project) -> dict:
    """Score the active models against every imported rater and the consensus; saved next to the imported labels."""
    placed = imported(project)
    if placed.empty:
        raise ValueError("No imported labels in this project: import a dataset package with annotations first")
    rows = []
    for stain, mine in placed.groupby("stain"):
        found = detections(project, stain, set(mine.core_id))
        if found.empty:
            continue
        for rater, theirs in mine.groupby("rater"):
            reviewed = reviewed_images(mine, rater)  # the images this rater read: the AI's objects there count for precision
            marks = theirs[theirs.kind != "image_label"]
            if len(marks) and any(positive(stain, c) or c in GENERIC for c in marks["class"]):
                rows.append({"stain": stain, "rater": rater, "kind": "objects", "class": "all", **score_objects(project, marks, found, stain, reviewed)})
            elif len(marks) and not (stain == "6E10" and any(vocab.defaults(stain).resolve(c)[0] == "CAA" for c in marks["class"])):
                rows += [{"stain": stain, "rater": rater, "kind": "negatives", **r} for r in score_negatives(project, marks, found, stain)]
            vessels = score_vessels(project, marks, found, reviewed) if stain == "6E10" and len(marks) else None
            if vessels:
                rows.append({"stain": stain, "rater": rater, "kind": "objects", **vessels})
            labels = theirs[theirs.kind == "image_label"]
            if len(labels):
                rows += [{"stain": stain, "rater": rater, "kind": "image", **r} for r in score_images(labels, found, stain)]
    if not rows:
        raise ValueError("Detection has not looked at these images yet: run detection (Run → Run everything) first, then score again")
    people = agreement(project, placed)
    main = [r for r in rows if r["rater"] == CONSENSUS] or rows
    result = {"time": time.time(), "dataset": project.config.get("dataset", {}).get("id", ""), "rows": rows, "agreement": people,
              "models": {k: project.relative(project.model(k)) for k in ("neun", "amyloid", "tau")},
              "summary": [s for s in (summary_text(r["stain"], r, people) for r in main) if s]}
    out = folder(project)
    write_text_atomic(out / "scores.json", json.dumps(result, indent=1, default=float))
    write_records(out / "scores.csv", [{c: r.get(c) if r.get(c) is not None else "" for c in COLUMNS} for r in rows], COLUMNS)
    return result


def saved(project: Project) -> dict | None:
    path = folder(project) / "scores.json"
    return json.loads(path.read_text()) if path.exists() else None


def scores_csv(project: Project) -> Path:
    return folder(project) / "scores.csv"
