"""A dataset package's annotations, placed on the project's tiles and kept as labelling sessions.

Each annotation (a point, a box, a polygon or a label for the whole image, in the image's own pixels) is moved to the
core image it lies on (the image itself, or the region of a whole slide; enlarged to the analysis scale) and to the tile
it falls in. Then:

- `<reviews>/imported/<dataset>/annotations.csv` keeps every annotation in the project's frame (core-image pixels and
  field-picture pixels, boxes as corners) with its rater and class: the scores against the models are worked out from it;
- every rater becomes one read-only labelling session (`import-<dataset>-<rater>`), and the consensus one more:
  the package's own consensus rater if it has one, else the objects that more than half of the raters marked
  (`consensus_objects`), with the class most of them gave. The labels go into the label store like any other, so they
  can be looked at on the Labels page and train models.

The class of an annotation is the project's class map entry for its label (from the package, editable), else the
package's `stainid_class`; "ignore" leaves it out.
"""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from stainid.imageset.package import IGNORE, read_package
from stainid.labels import sessions, store
from stainid.project import Project
from stainid.sampling.tiles import table
from stainid.tables import read_csv, write_records, write_text_atomic

CONSENSUS = "consensus"
MATCH_UM = {"NeuN": 6.0, "6E10": 12.0, "AT8": 8.0}  # two marks this close are one object (larger objects: their own size)
COLUMNS = ["rater", "kind", "label", "class", "stain", "image_id", "core_id", "tile_id", "x", "y", "core_x", "core_y",
           "x0", "y0", "x1", "y1", "points", "confidence", "votes"]


def folder(project: Project) -> Path:
    dataset = project.config.get("dataset", {}).get("id") or "package"
    return project.output("reviews") / "imported" / _safe(dataset)


def _safe(text: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "-" for c in str(text)).strip("-") or "x"


def session_name(dataset: str, rater: str) -> str:
    return f"import-{_safe(dataset)}-{_safe(rater)}"


def class_of(row: dict, class_map: dict[str, str]) -> str:
    label = row.get("label", "")
    value = class_map.get(label) or row.get("stainid_class") or ""
    return "" if value == IGNORE else value


def _cores_by_image(project: Project) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for row in read_csv(project.input("core_images")) if project.input("core_images").exists() else []:
        out[row.get("image_id", row["core_id"])].append(row)
    return out


def _box(core: dict) -> tuple[float, float, float, float]:
    x, y, w, h = (float(v) for v in core["source_box"].split())
    return x, y, w, h


def _core_at(cores: list[dict], x: float, y: float) -> dict | None:
    for core in cores:
        bx, by, bw, bh = _box(core)
        if bx <= x < bx + bw and by <= y < by + bh:
            return core
    return None


def _to_core(core: dict, x: float, y: float) -> tuple[float, float]:
    bx, by, _, _ = _box(core)
    scale = float(core["scale"])
    return (x - bx) * scale, (y - by) * scale


def _tile_at(tiles: pd.DataFrame, x: float, y: float) -> dict | None:
    hit = tiles[(tiles.x_px <= x) & (x < tiles.x_px + tiles.width_px) & (tiles.y_px <= y) & (y < tiles.y_px + tiles.height_px)]
    return hit.iloc[0].to_dict() if len(hit) else None


def _number(value) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except ValueError:
        return None


def _shape(row: dict) -> tuple[str, list[tuple[float, float]]] | None:
    """The annotation's kind and its points in source pixels (a box as its two corners); None if it has no position."""
    kind = (row.get("kind") or "point").strip().lower()
    if kind == "image_label":
        return kind, []
    if kind == "polygon" and row.get("polygon"):
        points = [(float(p[0]), float(p[1])) for p in json.loads(row["polygon"])]
        return (kind, points) if points else None
    x, y = _number(row.get("x")), _number(row.get("y"))
    if x is None or y is None:
        return None
    if kind == "box":
        x2, y2 = _number(row.get("x2")), _number(row.get("y2"))
        if x2 is None or y2 is None:
            return None
        return kind, [(min(x, x2), min(y, y2)), (max(x, x2), max(y, y2))]
    return "point", [(x, y)]


def place(project: Project, rows: list[dict], class_map: dict[str, str]) -> tuple[pd.DataFrame, list[str]]:
    """The annotations in the project's frame (see `COLUMNS`), and notes on those that could not be placed."""
    cores = _cores_by_image(project)
    tiles = table(project.analysis_tiles())
    by_core = {c: g for c, g in tiles.groupby("core_id")} if not tiles.empty else {}
    context = project.context_px
    out, unplaced = [], Counter()
    for row in rows:
        cls = class_of(row, class_map)
        image = row.get("image_id", "")
        if not cls:
            unplaced["no class (ignored)"] += 1
            continue
        found = cores.get(image)
        if not found:
            unplaced["image not prepared"] += 1
            continue
        shape = _shape(row)
        if shape is None:
            unplaced["no position"] += 1
            continue
        kind, points = shape
        base = {"rater": row.get("rater") or "rater", "kind": kind, "label": row.get("label", ""), "class": cls, "image_id": image,
                "confidence": row.get("confidence", ""), "votes": ""}
        if kind == "image_label":
            for core in found:
                out.append({**base, "stain": core["stain"], "core_id": core["core_id"]})
            continue
        cx, cy = np.mean([p[0] for p in points]), np.mean([p[1] for p in points])
        core = _core_at(found, cx, cy)
        if core is None:
            unplaced["outside the tissue regions"] += 1
            continue
        mapped = [_to_core(core, px, py) for px, py in points]
        x, y = _to_core(core, cx, cy)
        tile = _tile_at(by_core.get(core["core_id"], tiles.iloc[:0]), x, y)
        if tile is None:
            unplaced["not on a tile (no tissue there)"] += 1
            continue
        ox, oy = max(0, int(tile["x_px"]) - context), max(0, int(tile["y_px"]) - context)
        xs, ys = [p[0] for p in mapped], [p[1] for p in mapped]
        entry = {**base, "stain": core["stain"], "core_id": core["core_id"], "tile_id": tile["tile_id"], "x": round(x - ox, 2), "y": round(y - oy, 2),
                 "core_x": round(x, 2), "core_y": round(y, 2)}
        if kind in ("box", "polygon"):
            entry |= {"x0": round(min(xs), 2), "y0": round(min(ys), 2), "x1": round(max(xs), 2), "y1": round(max(ys), 2)}
        if kind == "polygon":
            entry["points"] = json.dumps([[round(px - ox, 1), round(py - oy, 1)] for px, py in mapped])
        out.append(entry)
    notes = [f"{n:,} annotations left out: {why}" for why, n in unplaced.most_common()]
    return pd.DataFrame(out, columns=COLUMNS), notes


def _radius_px(project: Project, stain: str, row) -> float:
    own = MATCH_UM.get(stain, 8.0) / project.pixel_size_um
    if pd.notna(row.get("x0")) and row.get("x0") != "":
        own = max(own, 0.5 * min(float(row["x1"]) - float(row["x0"]), float(row["y1"]) - float(row["y0"])))
    return own


def consensus_objects(project: Project, placed: pd.DataFrame, min_votes: int | None = None) -> pd.DataFrame:
    """The consensus of the raters: marks of different raters within an object's size of each other are one object,
    kept when at least `min_votes` raters marked it (default: more than half of the raters of that image), with the class
    most of them gave and the mean position; image labels kept the same way per image and class."""
    raters_of = placed.groupby("image_id").rater.agg(lambda r: set(r)) if len(placed) else {}
    out = []
    for (core_id, stain), group in placed[placed.kind != "image_label"].groupby(["core_id", "stain"]):
        rows = group.to_dict("records")
        need = min_votes or len(raters_of[rows[0]["image_id"]]) // 2 + 1
        used = [False] * len(rows)
        for i, row in enumerate(rows):
            if used[i]:
                continue
            members, raters = [i], {row["rater"]}
            radius = _radius_px(project, stain, row)
            for j in range(i + 1, len(rows)):
                other = rows[j]
                if used[j] or other["rater"] in raters:
                    continue
                if math.hypot(other["core_x"] - row["core_x"], other["core_y"] - row["core_y"]) <= max(radius, _radius_px(project, stain, other)):
                    members.append(j)
                    raters.add(other["rater"])
            if len(raters) < need:
                continue
            for j in members:
                used[j] = True
            chosen = [rows[j] for j in members]
            top = Counter(r["class"] for r in chosen).most_common()
            if len(top) > 1 and top[0][1] == top[1][1]:
                continue  # split between classes: no consensus
            first = min(chosen, key=lambda r: math.hypot(r["core_x"] - np.mean([c["core_x"] for c in chosen]),
                                                         r["core_y"] - np.mean([c["core_y"] for c in chosen])))
            entry = {**first, "rater": CONSENSUS, "class": top[0][0], "label": "", "votes": len(raters), "confidence": ""}
            for key in ("x0", "y0", "x1", "y1"):
                values = [float(c[key]) for c in chosen if c.get(key) not in (None, "") and pd.notna(c.get(key))]
                if values:
                    entry[key] = round(float(np.median(values)), 2)
            out.append(entry)
    images = placed[placed.kind == "image_label"]
    for (core_id, stain), group in images.groupby(["core_id", "stain"]):
        voters = set(group.rater)
        need = min_votes or len(voters) // 2 + 1
        for cls, votes in group.groupby("class").rater.agg(lambda r: set(r)).items():
            if len(votes) >= need:
                out.append({**group.iloc[0].to_dict(), "rater": CONSENSUS, "class": cls, "label": "", "votes": len(votes)})
    return pd.DataFrame(out, columns=COLUMNS)


def _label_records(project: Project, frame: pd.DataFrame, source: str, session: str) -> list[dict]:
    records = []
    for row in frame[frame.kind != "image_label"].to_dict("records"):
        x, y = float(row["x"]), float(row["y"])
        records.append({"id": store.object_id("class", row["tile_id"], x, y), "source": source, "stain": row["stain"], "task": "class",
                        "tile_id": row["tile_id"], "x": x, "y": y, "value": row["class"], "session": session})
        corners = _outline(row)
        if corners:
            records.append({"id": store.object_id("outline", row["tile_id"], x, y), "source": source, "stain": row["stain"], "task": "outline",
                            "tile_id": row["tile_id"], "x": x, "y": y, "points": corners, "session": session})
    return records


def _outline(row: dict) -> list[list[float]] | None:
    """A polygon's points, or a box's corners, in field-picture pixels."""
    if isinstance(row.get("points"), str) and row["points"]:
        return json.loads(row["points"])
    if row.get("x0") in (None, "") or pd.isna(row.get("x0")):
        return None
    dx, dy = float(row["x"]) - float(row["core_x"]), float(row["y"]) - float(row["core_y"])
    x0, y0, x1, y1 = (float(row[k]) for k in ("x0", "y0", "x1", "y1"))
    return [[x0 + dx, y0 + dy], [x1 + dx, y0 + dy], [x1 + dx, y1 + dy], [x0 + dx, y1 + dy]]


def _replace_session(project: Project, name: str, title: str, stain: str, fields: list[str], rater: str, dataset: str, count: int) -> None:
    root = project.output("reviews")
    if name in sessions.names(root):
        sessions.update(root, name, fields=fields, stain=stain, title=title, read_only=True, labels=count)
        return
    sessions.create(root, name, title=title, stain=stain, kind="fields", tools=["class", "outline"], ai_view="full", per_field=1, rule="uncertain",
                    fields=fields, instructions=f"Labels by {rater} from the dataset {dataset}, imported read-only.", read_only=True,
                    imported={"dataset": dataset, "rater": rater}, labels=count)


def import_package_labels(project: Project, redo: bool = True, log=print) -> Path:
    """Place the package's annotations on the tiles, write the imported table and one read-only session per rater (and
    the consensus). Run again (after the tiles change or the class map is edited) it replaces what it made before."""
    dataset = project.config.get("dataset", {})
    package = read_package(dataset["package"])
    class_map = dict(dataset.get("class_map") or {})
    placed, notes = place(project, package.annotations(), class_map)
    raters = sorted(set(placed.rater) - {CONSENSUS})
    if CONSENSUS not in set(placed.rater) and len(raters) > 1:
        placed = pd.concat([placed, consensus_objects(project, placed)], ignore_index=True)
    out = folder(project)
    out.mkdir(parents=True, exist_ok=True)
    write_records(out / "annotations.csv", placed.astype(object).where(placed.notna(), "").to_dict("records"), COLUMNS)
    root = project.output("reviews")
    name_of = {rater: session_name(package.id, rater) for rater in sorted(set(placed.rater))}
    if redo:  # earlier imports of these sessions are replaced: their labels are marked deleted (the history keeps them)
        current = store.current(root)
        old = current[current.session.isin(set(name_of.values()) | {n for n in sessions.names(root) if n.startswith(f"import-{_safe(package.id)}-")})]
        if len(old):
            store.append(root, [{**r, "deleted": True, "time": None} for r in old.astype(object).where(old.notna(), None).to_dict("records")])
    made = []
    for rater, name in name_of.items():
        mine = placed[placed.rater == rater]
        for stain, part in mine.groupby("stain"):
            session = name if mine.stain.nunique() == 1 else f"{name}-{_safe(stain)}"
            source = f"import: {package.id} {rater}"
            records = _label_records(project, part, source, session)
            if records:
                store.append(root, records)
            fields = list(dict.fromkeys(part.tile_id.dropna().astype(str)))
            fields = [f for f in fields if f]
            title = f"{package.title}: {'consensus' if rater == CONSENSUS else rater}" + (f" ({stain})" if mine.stain.nunique() > 1 else "")
            if fields:
                _replace_session(project, session, title, stain, fields, rater, package.id, int((part.kind != "image_label").sum()))
                made.append(session)
    summary = {"dataset": package.id, "annotations": int(len(placed)), "raters": raters, "consensus": CONSENSUS in set(placed.rater),
               "sessions": made, "notes": notes, "by_rater": {r: int((placed.rater == r).sum()) for r in sorted(set(placed.rater))}}
    write_text_atomic(out / "import.json", json.dumps(summary, indent=1))
    for note in notes:
        log(note, flush=True)
    log(f"{len(placed):,} annotations placed; {len(made)} read-only labelling {'session' if len(made) == 1 else 'sessions'}", flush=True)
    return out / "annotations.csv"


def imported(project: Project) -> pd.DataFrame:
    """The imported annotations in the project's frame (empty when there are none)."""
    path = folder(project) / "annotations.csv"
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(path, dtype={"rater": str, "class": str, "stain": str, "image_id": str, "core_id": str, "tile_id": str, "label": str,
                                    "kind": str, "points": str})
