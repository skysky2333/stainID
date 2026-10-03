"""Shared field annotation: several people annotate whole fields, each in their own layer, optionally seeing the AI output.

An annotation project is a folder with `meta.json` (stain, fields, labels, enabled tools, what annotators see of the AI)
and `annotations.jsonl`, one append-only log written by every annotator (and by every server process, under a file lock).
Each line is one version of one annotation: {id, annotator, tile_id, type, label, points, ai, note, time, deleted}.
The latest version of (annotator, id) wins, so edits and deletions keep the full history.

Types: `object` labels an AI detection, `point` marks an object the AI missed, `polygon` outlines one, `tag` is a
field-level tag, `note` a field-level comment and `done` marks the field finished by that annotator.
"""
from __future__ import annotations

import fcntl
import json
import time
from collections import Counter
from itertools import combinations
from pathlib import Path

import pandas as pd

TYPES = ("object", "point", "polygon", "tag", "note", "done")
TOOLS = ("object", "point", "polygon", "tag")
AI_VIEWS = ("full", "positions", "none")
FOLDER = "shared"
_CACHE: dict[Path, tuple[tuple[int, int], list[dict]]] = {}


def folder(root: Path, name: str) -> Path:
    return root / FOLDER / name


def create(root: Path, name: str, title: str, stain: str, fields: list[str], labels: list[str], tags: list[str], tools: list[str],
           ai_view: str, per_field: int, instructions: str) -> dict:
    target = folder(root, name)
    if target.exists():
        raise FileExistsError(target)
    if ai_view not in AI_VIEWS or not set(tools) <= set(TOOLS) or not tools:
        raise ValueError(f"tools must come from {TOOLS} and ai_view from {AI_VIEWS}")
    target.mkdir(parents=True)
    meta = {"name": name, "title": title or name, "stain": stain, "fields": fields, "labels": labels, "tags": tags, "tools": tools,
            "ai_view": ai_view, "per_field": per_field, "instructions": instructions, "created": time.time()}
    (target / "meta.json").write_text(json.dumps(meta, indent=1))
    (target / "annotations.jsonl").touch()
    return meta


def meta(root: Path, name: str) -> dict:
    return json.loads((folder(root, name) / "meta.json").read_text())


def names(root: Path) -> list[str]:
    return sorted(p.parent.name for p in (root / FOLDER).glob("*/meta.json"))


def append(root: Path, name: str, records: list[dict]) -> list[dict]:
    stamped = [{**r, "time": time.time()} for r in records]
    with (folder(root, name) / "annotations.jsonl").open("a", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.write("".join(json.dumps(r) + "\n" for r in stamped))
        handle.flush()
        fcntl.flock(handle, fcntl.LOCK_UN)
    return stamped


def history(root: Path, name: str) -> list[dict]:
    path = folder(root, name) / "annotations.jsonl"
    stat = path.stat()
    key = (stat.st_mtime_ns, stat.st_size)
    if path in _CACHE and _CACHE[path][0] == key:
        return _CACHE[path][1]
    with path.open(encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_SH)
        lines = handle.read().splitlines()
        fcntl.flock(handle, fcntl.LOCK_UN)
    records = [json.loads(line) for line in lines if line]
    _CACHE[path] = (key, records)
    return records


def current(root: Path, name: str) -> list[dict]:
    """Latest version of every annotation that is not deleted."""
    latest = {(r["annotator"], r["id"]): r for r in history(root, name)}
    return [r for r in latest.values() if not r.get("deleted")]


def done_by(annotations: list[dict]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for r in annotations:
        if r["type"] == "done":
            out.setdefault(r["tile_id"], set()).add(r["annotator"])
    return out


def next_field(project: dict, annotations: list[dict], annotator: str, busy: dict[str, set[str]]) -> str | None:
    """The next field for this annotator: fields that still need annotators first, then fields nobody else has open."""
    done = done_by(annotations)
    todo = [t for t in project["fields"] if annotator not in done.get(t, set())]
    order = {t: i for i, t in enumerate(project["fields"])}

    def rank(tile: str) -> tuple:
        n = len(done.get(tile, set()))
        return n >= project["per_field"], bool(busy.get(tile, set()) - {annotator}), n, order[tile]

    return min(todo, key=rank) if todo else None


def summary(project: dict, annotations: list[dict]) -> dict:
    """Progress per annotator, AI class against human labels, and agreement between annotators on the same AI objects."""
    done = done_by(annotations)
    people = sorted({r["annotator"] for r in annotations})
    per_person = []
    for person in people:
        mine = [r for r in annotations if r["annotator"] == person]
        per_person.append({"annotator": person, "fields_done": sum(person in s for s in done.values()),
                           **{t: sum(r["type"] == t for r in mine) for t in ("object", "point", "polygon", "tag")}})
    objects = [r for r in annotations if r["type"] == "object"]
    versus = Counter((r["ai"].get("model_class") or "?", r["label"]) for r in objects)
    by_object: dict[tuple, dict[str, str]] = {}
    for r in objects:
        by_object.setdefault((r["tile_id"], r["id"]), {})[r["annotator"]] = r["label"]
    pairs = Counter()
    for labels in by_object.values():
        for a, b in combinations(sorted(labels), 2):
            pairs[(a, b, "n")] += 1
            pairs[(a, b, "same")] += labels[a] == labels[b]
    agreement = [{"a": a, "b": b, "objects": pairs[(a, b, "n")], "agreement": pairs[(a, b, "same")] / pairs[(a, b, "n")]}
                 for a, b, kind in pairs if kind == "n"]
    covered = sum(len(done.get(t, set())) >= project["per_field"] for t in project["fields"])
    return {"fields": len(project["fields"]), "fields_covered": covered, "annotators": per_person, "agreement": agreement,
            "ai_vs_human": [{"ai_class": ai, "label": label, "n": n} for (ai, label), n in sorted(versus.items())]}


def export(annotations: list[dict], tiles: pd.DataFrame) -> pd.DataFrame:
    """One row per annotation; points and polygons as field-crop pixel coordinates (centroid in x, y)."""
    rows = []
    for r in sorted(annotations, key=lambda r: (r["tile_id"], r["annotator"], r["time"])):
        points = r.get("points") or []
        ai = r.get("ai") or {}
        x, y = (ai.get("x"), ai.get("y")) if r["type"] == "object" else (
            (sum(p[0] for p in points) / len(points), sum(p[1] for p in points) / len(points)) if points else (None, None))
        rows.append({"annotator": r["annotator"], "tile_id": r["tile_id"], "type": r["type"], "label": r.get("label", ""),
                     "x": None if x is None else round(x, 1), "y": None if y is None else round(y, 1),
                     "polygon": json.dumps(points) if r["type"] == "polygon" else "", "ai_class": ai.get("model_class", ""),
                     "ai_probability": ai.get("probability"), "note": r.get("note", ""),
                     "time": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(r["time"]))})
    frame = pd.DataFrame(rows, columns=["annotator", "tile_id", "type", "label", "x", "y", "polygon", "ai_class", "ai_probability", "note", "time"])
    context = tiles[["tile_id", "core_id", "tma", "disease_group", "region"]].drop_duplicates("tile_id") if not tiles.empty else None
    return frame.merge(context, on="tile_id", how="left") if context is not None else frame
