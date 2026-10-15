"""`stainid score-segmentation`: how well detection finds and outlines objects, measured on the fields people outlined
completely (outline sessions, `labels.outline`).

Every object people outlined is matched against (a) every candidate the detection step's segmentation proposes
(*proposal recall*: is the object proposed at all, with a fitting outline?) and (b) the candidates the pipeline counted
(*end-to-end recall and precision*), with the rules of `segmentation.scoring`. Classification scores measured only on
candidates (the model's AUC) cannot see objects that were never proposed or were cut up or merged; these can.

Writes into `<tables>/segmentation_benchmark/`: `report.json` (per stain: the figures in plain numbers, by TMA, group,
density third and object size), `objects.csv` (one row per outlined object), `detections.csv` (one row per counted
object) and `fields.csv` (one row per field and person). Candidates are found first where missing (`proposals`).
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from stainid.labels import outline, sessions, store
from stainid.labels.tasks import STAINS
from stainid.project import Project
from stainid.sampling.tiles import known_tile_rows, table
from stainid.segmentation import proposals, scoring
from stainid.tables import write_records, write_text_atomic

# object diameters (µm) at which the size bins of each stain start
SIZE_BINS = {"NeuN": (0, 10, 15, 20), "6E10": (0, 15, 30, 60), "AT8": (0, 10, 15, 20)}
FACTS = ("core_id", "tma", "disease_group", "region")
WORDS = {"NeuN": "neurons", "6E10": "plaques", "AT8": "tau+ neurons"}


def report_path(project: Project):
    return proposals.folder(project) / "report.json"


def outline_sessions(project: Project, stains: list[str] | None = None, names: list[str] | None = None) -> list[dict]:
    root = project.output("reviews")
    metas = [sessions.meta(root, n) for n in sessions.names(root)]
    return [m for m in metas if m["kind"] == outline.KIND and (not stains or m["stain"] in stains) and (not names or m["name"] in names)]


def find_proposals(project: Project, metas: list[dict], redo: bool = False) -> list[str]:
    """Find the candidates of every field of these sessions that has none (or old ones); returns the fields that failed,
    with the reason."""
    from stainid.training.candidates import FieldLoader

    tiles = known_tile_rows(project)
    todo = [(m["stain"], t) for m in metas for t in m["fields"]]
    todo = list(dict.fromkeys(todo))
    loader = FieldLoader(project)
    waiting = [(s, t) for s, t in todo if t in tiles and (redo or not proposals.current(project, tiles[t], loader))]
    failed = []
    print(f"[0/{len(waiting)}] {len(todo) - len(waiting)} of {len(todo)} fields have their candidates already", flush=True)
    for number, (stain, tile_id) in enumerate(waiting, start=1):
        try:
            count = proposals.compute(project, tiles[tile_id], loader)
            print(f"[{number}/{len(waiting)}] {tile_id}: {count} candidates", flush=True)
        except (LookupError, ValueError, OSError) as error:
            failed.append(f"{tile_id}: {error}")
            print(f"[{number}/{len(waiting)}] {tile_id}: could not find its candidates ({error})", flush=True)
    return failed


def _truths(rows: pd.DataFrame, shape: tuple[int, int]) -> list[scoring.Truth]:
    out = []
    for row in rows.itertuples():
        truth = scoring.Truth(row.points, row.value)
        truth.rasterize(shape)
        if truth.area:
            out.append(truth)
    return out


def _disk_layer(shape: tuple[int, int], found: pd.DataFrame) -> np.ndarray:
    import cv2

    layer = np.zeros(shape, np.uint32)
    for i, row in enumerate(found.itertuples(), start=1):
        cv2.circle(layer, (int(round(row.x)), int(round(row.y))), max(1, int(round(row.radius_px))), i, -1)
    return layer


def score_field(project: Project, tile: dict, core_tiles: pd.DataFrame, rows: pd.DataFrame) -> tuple[list[dict], list[dict], dict] | None:
    """One field outlined by one person: a row per object, a row per counted object, and the field's summary."""
    saved = proposals.load(project, tile["stain"], tile["tile_id"])
    if saved is None:
        return None
    layers = saved["layers"]
    shape = layers.shape[1:]
    box = saved["box"]
    px = float(saved["pixel_size_um"])
    found = proposals.detections(project, tile, core_tiles, shape)
    chosen, unexplained = proposals.counted(saved, found)
    keys = list(zip(saved["layer"].astype(int).tolist(), saved["label"].astype(int).tolist()))
    if len(unexplained):
        layers = np.concatenate([layers.astype(np.uint32), _disk_layer(shape, unexplained)[None]])
        extra = [(len(layers) - 1, i) for i in range(1, len(unexplained) + 1)]
    else:
        extra = []
    candidates = scoring.Candidates(layers, keys + extra)
    truths = _truths(rows, shape)
    inter = scoring.intersections(truths, candidates)
    all_columns = list(range(len(keys)))
    counted_columns = chosen + list(range(len(keys), len(keys) + len(extra)))
    proposed = scoring.score_set(truths, candidates, inter, all_columns)
    detected = scoring.score_set(truths, candidates, inter, counted_columns)
    sam = proposals.outlines(project, tile)
    sam_iou = np.zeros(len(truths))
    if sam is not None and sam.shape == tuple(shape) and sam.any():
        sam_set = scoring.Candidates(sam[None].astype(np.uint32), [(0, int(v)) for v in np.unique(sam[sam > 0])])
        sam_inter = scoring.intersections(truths, sam_set)
        scores = scoring.iou(sam_inter, np.array([t.area for t in truths], float), sam_set.area)
        sam_iou = scores.max(axis=1) if scores.shape[1] else sam_iou
    facts = {k: tile.get(k, "") for k in FACTS}
    objects = []
    for t, truth in enumerate(truths):
        x, y = truth.centre
        diameter = 2 * np.sqrt(truth.area / np.pi) * px
        objects.append({"tile_id": tile["tile_id"], **facts, "x": round(x, 1), "y": round(y, 1), "class": truth.value,
                        "counted_in_scores": truth.value != scoring.UNSURE and scoring.inside_box(x, y, box),
                        "area_um2": round(truth.area * px * px, 1), "diameter_um": round(diameter, 2),
                        "proposal_iou": round(float(proposed.best_iou[t]), 3), "proposal_outcome": proposed.outcome[t],
                        "proposal_found_at_low_iou": bool(proposed.found_low[t]), "split_into": proposed.pieces[t],
                        "boundary_iou": round(float(proposed.boundary[t]), 3) if np.isfinite(proposed.boundary[t]) else None,
                        "detection_iou": round(float(detected.best_iou[t]), 3), "detection_outcome": detected.outcome[t],
                        "detection_found_at_low_iou": bool(detected.found_low[t]),
                        "outline_iou": round(float(sam_iou[t]), 3) if sam is not None else None,
                        "outcome": scoring.end_to_end(proposed.outcome[t], detected.outcome[t])})
    counted_rows = []
    for j, column in enumerate(counted_columns):
        x, y = candidates.centre(column)
        if not scoring.inside_box(x, y, box):
            continue
        counted_rows.append({"tile_id": tile["tile_id"], **facts, "x": round(x, 1), "y": round(y, 1),
                             "area_um2": round(float(candidates.area[column]) * px * px, 1),
                             "made_again": column < len(keys), "outcome": detected.extra[j], "right_at_low_iou": bool(detected.extra_low[j])})
    inside = [o for o in objects if o["counted_in_scores"]]
    summary = {"tile_id": tile["tile_id"], **facts, "objects": len(inside), "counted_right": sum(o["outcome"] == "counted" for o in inside),
               "proposed": sum(o["proposal_outcome"] == "found" for o in inside), "counted": len(counted_rows),
               "counted_wrong": sum(r["outcome"] not in ("right", "unsure") for r in counted_rows),
               "candidates": int(sum(scoring.inside_box(float(cx), float(cy), box) for cx, cy in zip(saved["x"], saved["y"]))),
               "counted_not_found_again": int(len(unexplained))}
    return objects, counted_rows, summary


def _share(part: int, whole: int) -> float | None:
    return round(part / whole, 4) if whole else None


def _median(values) -> float | None:
    values = [v for v in values if v is not None and np.isfinite(v)]
    return round(float(np.median(values)), 3) if values else None


def summarize(objects: pd.DataFrame, counted: pd.DataFrame) -> dict:
    """The figures of one group of objects and counted objects (a stain, a TMA, …)."""
    real = objects[objects.counted_in_scores] if len(objects) else objects
    judged = counted[counted.outcome != "unsure"] if len(counted) else counted
    n, m = len(real), len(judged)
    outcomes = real.outcome.value_counts().to_dict() if n else {}
    proposal = real.proposal_outcome.value_counts().to_dict() if n else {}
    wrong = judged.outcome.value_counts().to_dict() if m else {}
    overlapped = real[real.proposal_outcome != "missed"] if n else real
    right = int(wrong.get("right", 0))
    recall, precision = _share(int(outcomes.get("counted", 0)), n), _share(right, m)
    return {
        "objects": n, "counted": m,
        "proposal_recall": _share(int(proposal.get("found", 0)), n),
        "proposal_recall_low": _share(int(real.proposal_found_at_low_iou.sum()), n) if n else None,
        "detection_recall": recall, "detection_recall_low": _share(int(real.detection_found_at_low_iou.sum()), n) if n else None,
        "detection_precision": precision, "detection_precision_low": _share(int(judged.right_at_low_iou.sum()), m) if m else None,
        "f1": round(2 * recall * precision / (recall + precision), 4) if recall and precision else None,
        "outcomes": {k: int(outcomes.get(k, 0)) for k in ("counted", "rejected", "missed", "merged", "split", "poor outline")},
        "proposal_outcomes": {k: int(proposal.get(k, 0)) for k in scoring.OUTCOMES},
        "counted_outcomes": {k: int(wrong.get(k, 0)) for k in scoring.EXTRA},
        "proposal_iou_median": _median(overlapped.proposal_iou) if n else None,
        "boundary_iou_median": _median(overlapped.boundary_iou) if n else None,
        "outline_iou_median": _median(real[real.outcome == "counted"].outline_iou) if n else None,
    }


def _by(objects: pd.DataFrame, counted: pd.DataFrame, column: str, order: list[str] | None = None) -> list[dict]:
    keys = order or sorted({str(v) for v in objects[column]} | ({str(v) for v in counted[column]} if len(counted) else set()), key=lambda v: (len(v), v))
    out = []
    for key in keys:
        part = objects[objects[column].astype(str) == key]
        part_counted = counted[counted[column].astype(str) == key] if len(counted) else counted
        out.append({"key": key, "fields": int(part.tile_id.nunique()), **summarize(part, part_counted)})
    return out


def _size_bins(stain: str, objects: pd.DataFrame) -> list[dict]:
    starts = SIZE_BINS[stain]
    real = objects[objects.counted_in_scores]
    out = []
    for i, start in enumerate(starts):
        stop = starts[i + 1] if i + 1 < len(starts) else np.inf
        part = real[(real.diameter_um >= start) & (real.diameter_um < stop)]
        name = f"under {stop:g} µm" if start == 0 else f"{start:g} µm and larger" if not np.isfinite(stop) else f"{start:g}–{stop:g} µm"
        out.append({"key": name, "objects": int(len(part)), "proposal_recall": _share(int((part.proposal_outcome == "found").sum()), len(part)),
                    "detection_recall": _share(int((part.outcome == "counted").sum()), len(part)),
                    "outcomes": {k: int((part.outcome == k).sum()) for k in ("counted", "rejected", "missed", "merged", "split", "poor outline")}})
    return out


def _histogram(values: pd.Series) -> list[int]:
    return np.histogram(values.clip(0, 1), bins=np.linspace(0, 1, 11))[0].astype(int).tolist() if len(values) else [0] * 10


def score(project: Project, metas: list[dict], failed: list[str] | None = None) -> dict:
    """Score every complete field of these outline sessions and write the report and tables."""
    labels = store.current(project.output("reviews"))
    tiles = known_tile_rows(project)
    analysis = table(project.analysis_tiles())
    all_objects, all_counted, all_fields = [], [], []
    stains: dict[str, dict] = {}
    for meta in metas:
        stain, name = meta["stain"], meta["name"]
        entry = stains.setdefault(stain, {"stain": stain, "sessions": [], "fields_total": 0, "fields_complete": 0, "people": set(),
                                          "waiting": 0, "not_found_again": 0})
        entry["sessions"].append({"name": name, "title": meta.get("title", name)})
        entry["fields_total"] += len(meta["fields"])
        done = outline.finished(labels, name)
        density = meta.get("benchmark", {}).get("density", {})
        for tile_id in meta["fields"]:
            if tile_id not in tiles or not done.get(tile_id):
                continue
            tile = tiles[tile_id]
            core_tiles = analysis[(analysis.core_id == tile["core_id"]) & (analysis.stain == stain)] if not analysis.empty else analysis
            entry["fields_complete"] += 1
            for person in sorted(done[tile_id]):
                rows = outline.objects(labels, name, tile_id, person)
                result = score_field(project, tile, core_tiles, rows)
                if result is None:
                    entry["waiting"] += 1
                    continue
                objects, counted, field = result
                extra = {"stain": stain, "session": name, "person": person, "density": density.get(tile_id, "unknown")}
                all_objects += [{**extra, **o} for o in objects]
                all_counted += [{**extra, **c} for c in counted]
                all_fields.append({**extra, **field})
                entry["people"].add(person)
                entry["not_found_again"] += field["counted_not_found_again"]
    objects = pd.DataFrame(all_objects)
    counted = pd.DataFrame(all_counted)
    out = []
    for stain in [s for s in STAINS if s in stains]:
        entry = stains[stain]
        mine = objects[objects.stain == stain] if len(objects) else pd.DataFrame(columns=["stain", "counted_in_scores", "outcome"])
        mine_counted = counted[counted.stain == stain] if len(counted) else pd.DataFrame(columns=["stain", "outcome"])
        scored = bool(len(mine))
        out.append({**{k: v for k, v in entry.items() if k != "people"}, "people": sorted(entry["people"]),
                    "fields_scored": int(mine.tile_id.nunique()) if scored else 0,
                    **(summarize(mine, mine_counted) if scored else {"objects": 0}),
                    "by_tma": _by(mine, mine_counted, "tma") if scored else [],
                    "by_group": _by(mine, mine_counted, "disease_group") if scored else [],
                    "by_density": _by(mine, mine_counted, "density", [lv for lv in (*outline.LEVELS, "unknown") if (mine.density == lv).any()]) if scored else [],
                    "by_size": _size_bins(stain, mine) if scored else [],
                    "proposal_iou_histogram": _histogram(mine[mine.counted_in_scores].proposal_iou) if scored else [],
                    "detection_iou_histogram": _histogram(mine[mine.counted_in_scores].detection_iou) if scored else []})
    report = {"made": time.time(), "iou": scoring.IOU, "iou_low": scoring.IOU_LOW, "stains": out, "failed": failed or [],
              "rules": {"overlap_of_object": scoring.OVERLAP_OF_OBJECT, "overlap_of_candidate": scoring.OVERLAP_OF_CANDIDATE,
                        "merge_cover": scoring.MERGE_COVER, "split_part": scoring.SPLIT_PART, "boundary_band": scoring.BAND,
                        "match_px": proposals.MATCH_PX, "size_bins_um": SIZE_BINS}}
    target = proposals.folder(project)
    target.mkdir(parents=True, exist_ok=True)
    write_records(target / "objects.csv", all_objects, ["stain", "session", "person", "tile_id"])
    write_records(target / "detections.csv", all_counted, ["stain", "session", "person", "tile_id"])
    write_records(target / "fields.csv", all_fields, ["stain", "session", "person", "tile_id"])
    write_text_atomic(report_path(project), json.dumps(report, indent=1, default=lambda v: v.item() if hasattr(v, "item") else str(v)))
    return report


def plain(stain_report: dict) -> str:
    """The report of one stain in a sentence or two."""
    n = stain_report.get("objects", 0)
    if not n:
        return f"{stain_report['stain']}: no field is outlined completely yet."
    words = WORDS.get(stain_report["stain"], "objects")
    o = stain_report["outcomes"]
    pct = lambda v: "—" if v is None else f"{100 * v:.0f}%"  # noqa: E731
    return (f"{stain_report['stain']}: of {n} {words} people outlined in {stain_report['fields_scored']} fields, the AI proposed "
            f"{pct(stain_report['proposal_recall'])} with a fitting outline and counted {pct(stain_report['detection_recall'])}. "
            f"{o['missed']} were never proposed, {o['split']} were cut into pieces, {o['merged']} were merged with a neighbour, "
            f"{o['poor outline']} had a poor outline and {o['rejected']} were proposed but not counted. "
            f"{pct(stain_report['detection_precision'])} of what it counted is a real object with a fitting outline.")


def run(project: Project, stain: str = "all", session: str = "", proposals_only: bool = False, redo: bool = False) -> dict | None:
    metas = outline_sessions(project, None if stain in ("", "all") else [stain], [s for s in session.split(",") if s] or None)
    if not metas:
        raise ValueError("There is no segmentation check yet: set one up under Models → How well does detection outline objects?")
    failed = find_proposals(project, metas, redo)
    if proposals_only:
        return None
    report = score(project, metas, failed)
    for entry in report["stains"]:
        print(plain(entry), flush=True)
    print(f"Report: {report_path(project)}", flush=True)
    return report
