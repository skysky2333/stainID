"""How well detection's outlines match the outlines people fixed in "Fix the field": the evidence for each stain's
default outline (`v2.outlines.default_outline`, the Run page's setting).

The objects whose outline people changed, from the one label-log reader of the Annotations layer (`labels.compare`):
a precise edit (an `outline` label "edited" on a candidate), a split or a merge (`labels.objectedit`: the pieces or the
merged object against the candidates they replace). Each is compared with every outline detection had for it: the
stain outline and Segment Anything's three masks (small, medium, large), and the one it showed first (`shown`,
`candidates.outline_options`). For a split or merge the union of the originals' outlines is compared with the union
of what people drew. Per outline: the intersection over union (IoU) and the area ratio (detection's area over the
person's), and which outline was closest (highest IoU) for each object. An object whose outline people switched (Tab)
to another of its outlines (`candidate_store.CHOICES`: the stain outline, Segment Anything small, medium, large; e.g. a
6E10 plaque from the stain outline to SAM medium) counts as a fix too (kind "switched"), its chosen outline compared
like a drawn one.

An indication, not an accuracy measure: only objects people chose to fix are counted (where the shown outline was
wrong enough to fix), and a precise edit starts from the shown outline. Where people only switched to another of
detection's outlines (`mask N`), that choice is counted apart (`picked`). Nothing here changes the default; the rows
(`rows`) keep each person's outline with its comparison, for training a better outline later. Labels train and
evaluate models only; the analysis never reads them."""
from __future__ import annotations

import cv2
import numpy as np
import pandas as pd

from stainid import cache, candidates, disk_cache
from stainid.exclusions import signature as excluded_signature
from stainid.labels import candidate_store, compare, store

OPTIONS = ("stain", "sam_small", "sam_medium", "sam_large")
NAMES = {"stain": "stain outline", "sam_small": "SAM small", "sam_medium": "SAM medium", "sam_large": "SAM large", "shown": "the outline shown first",
         "sam": "Segment Anything"}
COLUMNS = ["candidate_id", "x_core", "y_core", "outline", "sam_outline", "sam_status", "sam_masks", "sam_scores", "sam_default", "outline_default"]
HINT_GAIN = 0.05  # a hint when one outline's median IoU is at least this much above the shown outline's ...
HINT_MIN = 5  # ... over at least this many fixed objects
MASK_OPTION = {0: "sam_small", 1: "sam_medium", 2: "sam_large", 3: "stain"}  # pages of earlier releases: "mask N"
KINDS = ("edited", "switched", "split", "merge")
ROW_COLUMNS = ["stain", "kind", "person", "time", "candidates", "outline", *[f"{m}_{n}" for n in (*OPTIONS, "shown") for m in ("iou", "ratio")], "closest"]


def _ring(points) -> np.ndarray | None:
    if points is None:
        return None
    ring = np.asarray(points, np.float64).reshape(-1, 2)
    return ring if len(ring) >= 3 else None


def overlap(drawn: list, theirs: list) -> tuple[float, float] | None:
    """The IoU of two sets of polygons (each the union of its polygons) and the area ratio `theirs` / `drawn`; None when
    either is empty."""
    a, b = [r for r in map(_ring, drawn) if r is not None], [r for r in map(_ring, theirs) if r is not None]
    if not a or not b:
        return None
    every = np.vstack(a + b)
    x0, y0 = np.floor(every.min(axis=0)) - 1
    w, h = (np.ceil(every.max(axis=0)) - (x0, y0) + 2).astype(int)
    if w * h > 4096 * 4096:
        return None

    def mask(rings) -> np.ndarray:
        out = np.zeros((h, w), np.uint8)
        cv2.fillPoly(out, [np.rint(r - (x0, y0)).astype(np.int32) for r in rings], 1)
        return out.astype(bool)

    ma, mb = mask(a), mask(b)
    union = int((ma | mb).sum())
    if not union or not ma.any():
        return None
    return round(int((ma & mb).sum()) / union, 4), round(int(mb.sum()) / int(ma.sum()), 4)


def _options(row) -> dict[str, list | None]:
    """Every outline detection had for a candidate (core px), and the one it showed first."""
    masks = list(row.sam_masks) if isinstance(row.sam_masks, list) else []
    out = {"stain": row.outline if isinstance(row.outline, list) else None}
    for i, name in enumerate(("sam_small", "sam_medium", "sam_large")):
        out[name] = masks[i] if i < len(masks) and isinstance(masks[i], list) else None
    shown, default = candidates.outline_options(row)
    out["shown"] = shown[default] if shown and default < len(shown) and isinstance(shown[default], list) else None
    return out


def _fixed(project, stain: str, tiles: pd.DataFrame) -> tuple[list[dict], dict[str, int]]:
    """The objects people fixed: (kind, person, candidate ids, their outlines in core px), and the outlines people
    switched to without drawing (option → count)."""
    layer = compare.layer(project, stain, tiles)
    objects, picked = [], {}
    for r in layer["fixed"].itertuples():
        option = getattr(r, "option", "") or ""
        if option:  # another of its outlines, chosen with Tab: a changed option is an outline fix
            picked[option] = picked.get(option, 0) + 1
            if r.points:
                objects.append({"kind": "switched", "person": r.person, "time": r.time, "candidates": [r.candidate_id], "drawn": [r.points]})
        elif r.points:
            objects.append({"kind": "edited", "person": r.person, "time": r.time, "candidates": [r.candidate_id], "drawn": [r.points]})
        elif r.mask is not None and r.mask == r.mask and int(r.mask) in MASK_OPTION:
            picked[MASK_OPTION[int(r.mask)]] = picked.get(MASK_OPTION[int(r.mask)], 0) + 1
    for e in layer["edits"]:
        ids = [o["candidate_id"] for o in e["originals"] if o["candidate_id"]]
        drawn = [p["points"] for p in e["pieces"] if p["points"]]
        if ids and drawn and len(ids) == len(e["originals"]):
            objects.append({"kind": e["kind"], "person": e["person"], "time": e["time"], "candidates": ids, "drawn": drawn})
    return objects, picked


def _rows(project, stain: str, tiles: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    objects, picked = _fixed(project, stain, tiles)
    by_core: dict[str, set[str]] = {}
    for o in objects:
        for c in o["candidates"]:
            by_core.setdefault(compare._core_of(c), set()).add(c)
    found = {}
    for core, ids in by_core.items():
        frame = candidate_store.read_core(project, stain, core, COLUMNS)
        if frame is not None:
            for row in frame[frame.candidate_id.isin(ids)].itertuples():
                found[row.candidate_id] = _options(row)
    out = []
    for o in objects:
        if not all(c in found for c in o["candidates"]):
            continue
        row = {"stain": stain, "kind": o["kind"], "person": o["person"], "time": o["time"], "candidates": " ".join(o["candidates"]),
               "outline": " | ".join(candidate_store.encode(p) for p in o["drawn"])}
        for name in (*OPTIONS, "shown"):
            theirs = [found[c][name] for c in o["candidates"]]
            got = overlap(o["drawn"], theirs) if all(t is not None for t in theirs) else None
            row[f"iou_{name}"], row[f"ratio_{name}"] = got if got else (np.nan, np.nan)
        ious = {n: row[f"iou_{n}"] for n in OPTIONS if row[f"iou_{n}"] == row[f"iou_{n}"]}
        row["closest"] = max(ious, key=ious.get) if ious else ""
        out.append(row)
    return pd.DataFrame(out, columns=ROW_COLUMNS), picked


def rows(project, stain: str, tiles: pd.DataFrame) -> pd.DataFrame:
    """One row per object people fixed, with their outline ("x0 y0 x1 y1 …" in core px, pieces separated by " | ") and
    its IoU and area ratio against each of detection's outlines: what a better outline could be trained on."""
    return _cached(project, stain, tiles)[0]


def _cached(project, stain: str, tiles: pd.DataFrame):
    inputs = [store.log_path(project.output("reviews")), candidate_store.folder(project, stain)]
    key = ("outline fixes", 2, str(project.root), stain, len(tiles), excluded_signature(project, stain))
    return cache.memo(key, inputs, lambda: _rows(project, stain, tiles), folders=True,
                      persist=disk_cache.answer_file(project.root, f"outline-fixes-{stain}.pkl"))


def summary(project, stain: str, tiles: pd.DataFrame) -> dict:
    """The stain's outline fixes: how many objects (by kind), per outline the median IoU and area ratio and how often it
    was the closest, the outline people switched to without drawing, and a hint when one outline is clearly closer
    than the one shown first (the stain's default; never changed here)."""
    frame, picked = _cached(project, stain, tiles)
    from stainid.v2.params import default_outline  # noqa: PLC0415

    default = default_outline(project, stain)
    closest = frame.closest.value_counts().to_dict() if len(frame) else {}
    options = []
    for name in (*OPTIONS, "shown"):
        iou, ratio = frame[f"iou_{name}"].dropna(), frame[f"ratio_{name}"].dropna()
        options.append({"id": name, "name": NAMES[name], "compared": int(len(iou)), "iou_median": round(float(iou.median()), 3) if len(iou) else None,
                        "iou_mean": round(float(iou.mean()), 3) if len(iou) else None, "area_ratio_median": round(float(ratio.median()), 3) if len(ratio) else None,
                        "closest": int(closest.get(name, 0)) if name != "shown" else None})
    hint = None
    shown = next(o for o in options if o["id"] == "shown")
    ranked = sorted((o for o in options if o["id"] != "shown" and o["compared"] >= HINT_MIN and o["iou_median"] is not None), key=lambda o: -o["iou_median"])
    if ranked and shown["iou_median"] is not None and ranked[0]["iou_median"] - shown["iou_median"] >= HINT_GAIN and shown["compared"] >= HINT_MIN:
        best = ranked[0]
        hint = {"option": best["id"], "text": f"{best['name']} matches people's fixes better for {stain} (IoU {best['iou_median']:.2f} vs "
                                              f"{shown['iou_median']:.2f} for {NAMES['shown']}, over {best['compared']} objects)"}
    kinds = frame.kind.value_counts().to_dict() if len(frame) else {}
    return {"stain": stain, "objects": int(len(frame)), "edited": int(kinds.get("edited", 0)), "switched": int(kinds.get("switched", 0)),
            "split": int(kinds.get("split", 0)), "merge": int(kinds.get("merge", 0)),
            "people": sorted(set(frame.person)) if len(frame) else [], "options": options, "picked": picked, "default": default, "hint": hint}
