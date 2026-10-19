"""An "AI agent review" labelling batch: whole fields of the v2 candidates, as numbered picture sheets that AI reviewer
agents label by looking (their labels come back through `review.ai_ingest`).

`stainid review ai-batch build` writes, per stain, under `<reviews>/ai_review_round<N>/<stain>/`:

- `fields.csv`: the fields chosen and why. Fields are drawn as "Fix the field" draws them (the labelling tiles whose
  core has candidates), `FIELDS` per stain from different TMAs; fields people or earlier AI rounds labelled, fields of
  any session and the tutorial's are left out (`taken`); cores whose bad areas a person checked come first. About half
  are rich in the stain's object, a quarter sparse, a quarter with many candidates near the model's threshold.
  Every own candidate of a field is asked about (those in people's or detection's excluded areas are not, as in "Fix
  the field"). A field with more than `MOST` candidates is cut to a sub-field (the tile split n × n, the cell with
  `FEWEST`–`MOST` candidates that best serves its category; `window`) and, where even that has too many, capped by a
  stratified sample (`stratified`); both are recorded (`subfield`, `capped`).
- `manifest.csv`: one row per object, its `review_id` (`<stain>-F03-017`), candidate, position in the field picture,
  detection's call and the outline the reviewers judge (`outline_choice`, `outline_points`: field-picture pixels).
- `sheets/`: numbered panels, `PANELS` per JPEG about 2000 px wide: the raw crop (the object about 60 % of it, at least
  `CONTEXT_UM` across) and the same crop with the stain's default outline and the centre, with the id, area,
  detection's call and probability and its rule.
- `overviews/`: each field (or sub-field) whole, raw and with every object numbered, on a grid of field-picture pixels,
  for reporting missed objects by position.
- `assignment_A.*`, `assignment_B.*`: the fields split in two halves (balanced by category), about `OVERLAP` of the
  objects in both halves, for agreement; `review_labels_<half>.csv` the template each reviewer fills in.
- `REVIEW_INSTRUCTIONS.md`: the stain's allowed labels (from `labels.vocabulary`) and the CSV columns.

Labels (people's or AI reviewers') only train and evaluate models; the analysis never reads them.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from stainid import exclusions as excluded_areas
from stainid.labels import candidate_store, frames, sessions, status, store
from stainid.labels import vocabulary as vocab
from stainid.project import Project

FIELDS = 8  # fields per stain
FEWEST, MOST = 20, 80  # objects per field aimed at
GRIDS = (1, 2, 3, 4, 5, 6)  # a field is cut n × n to reach `MOST`
MIX = (("rich", 0.5), ("sparse", 0.25), ("threshold", 0.25))
NEAR = 0.15  # "near the threshold": |probability - threshold| at most this, among the candidates the model decided
OVERLAP = 0.10  # share of a stain's objects labelled in both halves
HALVES = ("A", "B")
CONTEXT_UM = 100.0  # a panel shows at least this much around the object
FILL = 0.6  # the object fills about this share of its panel
PANEL_PX = 236  # each of a panel's two pictures
COLUMNS, ROWS = 4, 5
PANELS = COLUMNS * ROWS
OVERVIEW_PX = 2000
READ_MARGIN = 640  # core pixels read around a field, so crops at its edge keep their context
CONFIDENCE = ("high", "moderate", "low")
OUTLINE = ("ok", "too big", "too small", "wrong object")
CSV_COLUMNS = ["review_id", "label", "confidence", "outline_ok", "rationale", "reviewer", "field_x", "field_y"]
STATS = ["candidate_id", "tile_id", "x_core", "y_core", "probability", "threshold", "decision", "reason", "excluded_fraction"]
ROUND = 4


def folder(project: Project, round_: int = ROUND) -> Path:
    return project.output("reviews") / f"ai_review_round{round_}"


def stain_folder(project: Project, stain: str, round_: int = ROUND) -> Path:
    return folder(project, round_) / stain


# ---------------------------------------------------------------- the allowed labels

@dataclass(frozen=True)
class Choice:
    key: str  # as the pages print it: 1, Q, Del, `
    label: str  # what reviewers write: "plaque / compact", "CAA", "not a plaque", "unsure"
    identity: str
    subtype: str
    group: str  # counted, lookalike, reject, unsure
    name: str
    description: str


def choices(vocabulary: vocab.Vocabulary) -> list[Choice]:
    """The stain's labels in the order of their keys (`vocabulary.assign_keys`, types asked)."""
    return [Choice(vocab.key_label(c.key), f"{c.identity} / {c.subtype}" if c.subtype else c.identity, c.identity, c.subtype or "", c.group,
                   c.name, c.description) for c in vocab.assign_keys(vocabulary, True)]


def resolve_label(vocabulary: vocab.Vocabulary, text) -> tuple[str, str] | None:
    """(identity, subtype) of a reviewer's label: the label as the instructions write it, its key, a class or type
    name, "identity / type", or an older value (`Vocabulary.resolve`); None when it is none of the stain's shown classes."""
    said = vocab.text(text)
    if not said:
        return None
    options = choices(vocabulary)
    low = said.lower()
    for c in options:
        names = {c.label.lower(), c.key.lower(), c.name.lower(), vocabulary.display(c.identity, c.subtype).lower()}
        if c.group in ("reject", "unsure"):
            names |= {"del" if c.group == "reject" else "0", "reject" if c.group == "reject" else "unsure"}
        if low in names:
            return c.identity, c.subtype
    identity, subtype = vocabulary.resolve(said)
    found = vocabulary.identity(identity) if identity else None
    if found is None or found.hidden:
        return None
    allowed = {(c.identity, c.subtype) for c in options}
    if (identity, subtype) in allowed:
        return identity, subtype
    if (identity, "") in allowed:  # a type that is not asked (hidden): the identity
        return identity, ""
    if found.role == vocab.POSITIVE and not subtype:  # the counted object without a type: valid, "type not recorded"
        return identity, ""
    return None


# ---------------------------------------------------------------- which fields

def _box(tile) -> tuple[int, int, int, int]:
    return int(tile["x_px"]), int(tile["y_px"]), int(tile["x_px"]) + int(tile["width_px"]), int(tile["y_px"]) + int(tile["height_px"])


def _overlaps(a, b) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _earlier_ai_cores(project: Project) -> dict[str, set[str]]:
    """stain -> cores of the earlier AI review rounds and adjudications (their images are whole cores, `M0xx_<stain>`,
    mapped to cores by the annotation manifest)."""
    root = project.output("reviews")
    manifest = root / "annotation_manifest.csv"
    if not manifest.exists():
        return {}
    images = pd.read_csv(manifest, dtype=str)
    core_of = {r.image_id: "_".join(Path(r.image_path).parts[-3:-1]) for r in images.itertuples() if isinstance(r.image_path, str)}
    used: set[str] = set()
    for path in [*root.glob("*ai_review_manifest.csv"), *root.glob("adjudication_*_manifest.csv"), *root.glob("*_candidate_review_round*/selection_key.csv")]:
        try:
            frame = pd.read_csv(path, dtype=str)
        except (OSError, ValueError):
            continue
        for column in ("image_id", "source_image_id"):
            if column in frame:
                used |= set(frame[column].dropna())
    out: dict[str, set[str]] = {}
    for image in used:
        if image in core_of and "_" in image:
            out.setdefault(image.rsplit("_", 1)[-1], set()).add(core_of[image])
    return out


def taken(project: Project, stain: str, tiles: pd.DataFrame, labels: pd.DataFrame) -> tuple[set[str], set[str]]:
    """The fields (and whole cores) this batch leaves out: fields with any label of the stain (or overlapping a sampled
    field that has one), cores labelled as whole cores or by the earlier AI rounds (their images were whole cores),
    the fields of every labelling session and the tutorial's field."""
    mine = tiles[tiles.stain == stain]
    boxes = mine.drop_duplicates("tile_id").set_index("tile_id")
    fields: set[str] = set()
    cores = set(_earlier_ai_cores(project).get(stain, set()))
    said = labels[labels.stain == stain] if len(labels) else labels
    core_ids = set(mine.core_id.astype(str))
    for tile_id in set(said.tile_id.dropna().astype(str)) if len(said) else set():
        if tile_id in core_ids:
            cores.add(tile_id)
        elif tile_id in boxes.index:
            fields.add(tile_id)
            row = boxes.loc[tile_id]
            same = mine[mine.core_id == row.core_id]
            fields |= {t for t, r in zip(same.tile_id, same.to_dict("records")) if _overlaps(_box(r), _box(row))}
    root = project.output("reviews")
    for name in sessions.names(root):
        meta = sessions.meta(root, name)
        fields |= set(meta.get("fields") or []) | set(meta.get("scope") or [])
    try:
        from stainid import tutorial  # noqa: PLC0415

        practice = tutorial.practice()
    except Exception:  # noqa: BLE001 - no tutorial built
        practice = None
    if practice and practice.get("tile_id"):
        fields.add(str(practice["tile_id"]))
    return fields, cores


def field_stats(project: Project, stain: str, tiles: pd.DataFrame) -> pd.DataFrame:
    """Per labelling field of the stain whose core has candidates: its own candidates (outside detection's excluded
    areas), how many detection accepted and how many lie near the model's threshold."""
    from stainid.sampling.tiles import labelling_tiles  # noqa: PLC0415

    pool = labelling_tiles(project)
    pool = pool[(pool.stain == stain) & pool.core_id.astype(str).isin(candidate_store.cores(project, stain))]
    if pool.empty:
        return pd.DataFrame(columns=["tile_id", "core_id", "tma", "disease_group", "n", "accepted", "near"])
    found = candidate_store.for_cores(project, stain, set(pool.core_id.astype(str)), STATS)
    found = found[found.tile_id.isin(set(pool.tile_id))]
    if "excluded_fraction" in found:
        found = found[pd.to_numeric(found.excluded_fraction, errors="coerce").fillna(0) < excluded_areas.EXCLUDED_AT]
    probability = pd.to_numeric(found.probability, errors="coerce")
    threshold = pd.to_numeric(found.threshold, errors="coerce") if "threshold" in found else pd.Series(0.5, index=found.index)
    near = candidate_store.by_model(found) & ((probability - threshold.fillna(0.5)).abs() <= NEAR)
    grouped = found.assign(accepted=found.decision == candidate_store.ACCEPTED, near=near).groupby("tile_id")
    counts = pd.DataFrame({"n": grouped.size(), "accepted": grouped.accepted.sum(), "near": grouped.near.sum()})
    keep = [c for c in ("tile_id", "core_id", "tma", "disease_group", "region") if c in pool]
    out = pool[keep].drop_duplicates("tile_id").merge(counts, left_on="tile_id", right_index=True, how="left").fillna({"n": 0, "accepted": 0, "near": 0})
    return out.astype({"n": int, "accepted": int, "near": int}).reset_index(drop=True)


def choose_fields(stats: pd.DataFrame, n: int, seed: int, checked: set[str], mix=MIX) -> pd.DataFrame:
    """`n` fields from different TMAs (while there are TMAs left): about half rich in the stain's object, a quarter
    sparse, a quarter with many candidates near the threshold. Within each category the best fields (the top sixth,
    at least 12) are drawn at random, those whose core's bad areas a person checked first. Returns the rows with
    `category` and `areas_checked`."""
    rng = np.random.default_rng(seed)
    stats = stats[stats.n >= 5].assign(areas_checked=lambda f: f.core_id.astype(str).isin(checked))
    if stats.empty:
        return stats.assign(category=[])
    wanted = {name: int(round(share * n)) for name, share in mix}
    wanted[mix[0][0]] += n - sum(wanted.values())
    top = lambda frame, column: frame.sort_values(column, ascending=False).head(max(12, len(frame) // 6))  # noqa: E731
    pools = {"rich": top(stats[stats.accepted >= 3], "accepted"),
             "threshold": top(stats[stats.near >= 3], "near"),
             "sparse": stats[(stats.accepted <= max(1, int(stats.accepted.quantile(0.2)))) & (stats.n >= 8)]}
    if len(pools["sparse"][pools["sparse"].n >= FEWEST]) >= 12:
        pools["sparse"] = pools["sparse"][pools["sparse"].n >= FEWEST]
    picked: list[dict] = []
    tmas: set[str] = set()
    for name, _ in mix:
        pool = pools[name]
        pool = pool.iloc[rng.permutation(len(pool))]
        pool = pool.sort_values("areas_checked", ascending=False, kind="stable")
        rows = pool.to_dict("records")
        for distinct in (True, False):  # different TMAs first; repeat a TMA only when every one is used
            for row in rows:
                if sum(p["category"] == name for p in picked) >= wanted[name]:
                    break
                if any(p["tile_id"] == row["tile_id"] for p in picked) or (distinct and str(row["tma"]) in tmas):
                    continue
                if any(p["core_id"] == row["core_id"] for p in picked):
                    continue
                picked.append({**row, "category": name})
                tmas.add(str(row["tma"]))
    return pd.DataFrame(picked)


def window(objects: pd.DataFrame, box: tuple[float, float, float, float], category: str, most: int = MOST, fewest: int = FEWEST
           ) -> tuple[tuple[float, float, float, float], int]:
    """The part of a field (field-picture pixels) whose objects are asked: the whole field when it has at most `most`,
    else the cell of the tile cut n × n (n from `GRIDS`, the smallest that works) with `fewest`–`most` objects that
    best serves the category (rich: most accepted; threshold: most near the threshold; sparse: most objects). Where no
    cell has that many, the fullest cell with at most `most`; where every cell of the finest grid has more, its first
    cell (`stratified` then caps it). Returns the box and n."""
    if len(objects) <= most:
        return box, 1
    x0, y0, x1, y1 = box
    best = None
    for n in GRIDS[1:]:
        w, h = (x1 - x0) / n, (y1 - y0) / n
        cells = []
        for i in range(n):
            for j in range(n):
                cell = (x0 + i * w, y0 + j * h, x0 + (i + 1) * w, y0 + (j + 1) * h)
                inside = objects[(objects.x >= cell[0]) & (objects.x < cell[2]) & (objects.y >= cell[1]) & (objects.y < cell[3])]
                score = {"rich": inside.accepted.sum(), "threshold": inside.near.sum()}.get(category, len(inside))
                cells.append((cell, len(inside), float(score)))
        fits = [c for c in cells if fewest <= c[1] <= most]
        if fits:
            cell = max(fits, key=lambda c: (c[2], c[1]))
            return cell[0], n
        under = [c for c in cells if c[1] <= most]
        if under:
            best = (max(under, key=lambda c: (c[1], c[2]))[0], n)
            if best and max(c[1] for c in under) >= fewest // 2:
                return best
    if best is not None:
        return best
    n = GRIDS[-1]
    w, h = (x1 - x0) / n, (y1 - y0) / n
    return (x0, y0, x0 + w, y0 + h), n


def stratified(objects: pd.DataFrame, most: int, seed: int) -> pd.DataFrame:
    """At most `most` objects, sampled in proportion from strata of detection's call (accepted, rejected by the model,
    rejected by a rule) and probability (fifths), each stratum keeping at least one."""
    if len(objects) <= most:
        return objects
    kind = np.where(objects.decision == candidate_store.ACCEPTED, "accepted", np.where(objects.reason.astype(str).str.startswith("rule:"), "rule", "model"))
    bins = np.clip((pd.to_numeric(objects.probability, errors="coerce").fillna(0).to_numpy() * 5).astype(int), 0, 4)
    strata = pd.Series([f"{k}{b}" for k, b in zip(kind, bins)], index=objects.index)
    rng = np.random.default_rng(seed)
    sizes = strata.value_counts()
    share = {s: max(1, int(round(most * c / len(objects)))) for s, c in sizes.items()}
    while sum(share.values()) > most:
        biggest = max(share, key=lambda s: share[s])
        share[biggest] -= 1
    keep = []
    for s, k in share.items():
        members = strata.index[strata == s].to_numpy()
        keep.extend(rng.choice(members, size=min(k, len(members)), replace=False).tolist())
    return objects.loc[sorted(keep)]


# ---------------------------------------------------------------- the objects of a field

def field_objects(project: Project, stain: str, tile: dict) -> pd.DataFrame:
    """Every own candidate of a field not in an excluded area (people's final say over detection's, as "Fix the field"
    asks), with its position, call and default outline in field-picture pixels."""
    found = candidate_store.in_field(project, stain, tile, project.context_px)
    own = found[found.own].copy() if len(found) else found
    if own.empty:
        return own.assign(accepted=[], near=[])
    shares = excluded_areas.shares(project, stain, str(tile["core_id"]), list(own.candidate_id))
    own = own[[shares.get(c, 0.0) < excluded_areas.EXCLUDED_AT for c in own.candidate_id]]
    words = vocab.of(project, stain)
    calls = [candidate_store.ai_call(words, d, k, s, r) for d, k, s, r in zip(own.decision, own["class"], own.subtype, own.reason)]
    outlines, chosen = [], []
    for row in own.itertuples():
        options, shown = candidate_store.choices(row)
        points = options[shown] if options and 0 <= shown < len(options) else None
        if points is None:  # no outline of the default kind: the first there is
            shown, points = next(((i, p) for i, p in enumerate(options or []) if p), (shown, None))
        outlines.append(points)
        chosen.append(candidate_store.CHOICES[shown] if points is not None and 0 <= shown < len(candidate_store.CHOICES) else "")
    probability = pd.to_numeric(own.probability, errors="coerce")
    threshold = pd.to_numeric(own.threshold, errors="coerce").fillna(0.5) if "threshold" in own else pd.Series(0.5, index=own.index)
    return own.assign(ai_identity=[c[0] for c in calls], ai_subtype=[c[1] for c in calls], outline_points=outlines, outline_choice=chosen,
                      accepted=own.decision == candidate_store.ACCEPTED, threshold=threshold,
                      near=candidate_store.by_model(own) & ((probability - threshold).abs() <= NEAR)).reset_index(drop=True)


def why(row) -> str:
    """Detection's reason in a few words: "accepted", "model", "rule: shape"."""
    reason = str(row.reason or "")
    if reason.startswith("rule:"):
        return "rule: " + reason[5:].replace("_", " ")
    return {"accepted": "accepted", "rejected_by_model": "model"}.get(reason, reason)


# ---------------------------------------------------------------- pictures

def _font(size: int):
    from PIL import ImageFont  # noqa: PLC0415

    for name in ("DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def read_region(path: Path, box: tuple[int, int, int, int]) -> tuple[np.ndarray, tuple[int, int]]:
    """RGB pixels of a box of a core image (clipped to it) and where they start."""
    from stainid.imaging.core_images import open_image  # noqa: PLC0415

    with open_image(path) as image:
        x0, y0 = max(0, box[0]), max(0, box[1])
        x1, y1 = min(image.width, box[2]), min(image.height, box[3])
        return np.asarray(image.crop((x0, y0, x1, y1)).convert("RGB")), (x0, y0)


def crop(region: np.ndarray, origin: tuple[float, float], cx: float, cy: float, side: float, out: int) -> np.ndarray:
    """A square of `side` pixels centred on (cx, cy) (region coordinates + origin), padded grey where the image ends,
    resized to `out` pixels."""
    half = side / 2
    x0, y0 = int(round(cx - half - origin[0])), int(round(cy - half - origin[1]))
    size = int(round(side))
    canvas = np.full((size, size, 3), 200, np.uint8)
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(region.shape[1], x0 + size), min(region.shape[0], y0 + size)
    if sx1 > sx0 and sy1 > sy0:
        canvas[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = region[sy0:sy1, sx0:sx1]
    return cv2.resize(canvas, (out, out), interpolation=cv2.INTER_AREA if size > out else cv2.INTER_LINEAR)


def crop_side(points, radius_px: float, pixel_um: float) -> float:
    """A panel's side in core pixels: the object about `FILL` of it, at least `CONTEXT_UM`, at most 1000 px."""
    if points:
        ring = np.asarray(points, float)
        extent = float(max(np.ptp(ring[:, 0]), np.ptp(ring[:, 1])))
    else:
        extent = 2 * float(radius_px or 8)
    return float(min(1000.0, max(extent / FILL, CONTEXT_UM / pixel_um)))


def panel(raw: np.ndarray, points_px, centre_px: tuple[float, float], lines: list[str], number: str) -> np.ndarray:
    """One numbered panel: the raw crop, the same with the outline (green) and the centre (red cross), text below."""
    from PIL import Image, ImageDraw  # noqa: PLC0415

    size = raw.shape[0]
    drawn = raw.copy()
    if points_px is not None and len(points_px) >= 3:
        cv2.polylines(drawn, [np.round(np.asarray(points_px, float)).astype(np.int32).reshape(-1, 1, 2)], True, (0, 230, 0), 2, cv2.LINE_AA)
    cx, cy = int(round(centre_px[0])), int(round(centre_px[1]))
    cv2.line(drawn, (cx - 5, cy), (cx + 5, cy), (255, 0, 0), 2)
    cv2.line(drawn, (cx, cy - 5), (cx, cy + 5), (255, 0, 0), 2)
    pad, head, foot = 6, 26, 42
    canvas = Image.new("RGB", (2 * size + 3 * pad, head + size + foot), (255, 255, 255))
    canvas.paste(Image.fromarray(raw), (pad, head))
    canvas.paste(Image.fromarray(drawn), (2 * pad + size, head))
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, 3), number, fill=(0, 0, 0), font=_font(19))
    for k, line in enumerate(lines[:2]):
        draw.text((pad, head + size + 3 + 19 * k), line, fill=(30, 30, 30), font=_font(15 if k == 0 else 14))
    return np.asarray(canvas)


def sheet(panels: list[np.ndarray], title: str) -> np.ndarray:
    """Up to `PANELS` panels, `COLUMNS` across, under a title."""
    from PIL import Image, ImageDraw  # noqa: PLC0415

    h, w = panels[0].shape[:2]
    gap, top = 8, 40
    rows = (len(panels) + COLUMNS - 1) // COLUMNS
    canvas = Image.new("RGB", (COLUMNS * w + (COLUMNS + 1) * gap, top + rows * (h + gap)), (235, 235, 235))
    ImageDraw.Draw(canvas).text((gap, 8), title, fill=(0, 0, 0), font=_font(22))
    for n, p in enumerate(panels):
        canvas.paste(Image.fromarray(p), (gap + (n % COLUMNS) * (w + gap), top + (n // COLUMNS) * (h + gap)))
    return np.asarray(canvas)


def overview(region: np.ndarray, origin: tuple[int, int], field_origin: tuple[int, int], box: tuple[float, float, float, float],
             objects: pd.DataFrame | None, title: str) -> np.ndarray:
    """The asked part of a field (`box`, field-picture pixels, drawn blue) with some context, on a grid labelled in
    field-picture pixels; with `objects`, each one's outline (green: detection accepted it, yellow: rejected) and its
    number."""
    from PIL import Image, ImageDraw  # noqa: PLC0415

    pad = 60
    fx0, fy0 = field_origin
    side = max(box[2] - box[0], box[3] - box[1]) + 2 * pad
    mx, my = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    view = (mx - side / 2, my - side / 2, mx + side / 2, my + side / 2)  # field px, square
    scale = min(OVERVIEW_PX / side, 2.0)
    picture = crop(region, origin, fx0 + (view[0] + view[2]) / 2, fy0 + (view[1] + view[3]) / 2, side, int(round(side * scale)))
    image = Image.fromarray(picture)
    draw = ImageDraw.Draw(image, "RGBA")
    to = lambda x, y: ((x - view[0]) * scale, (y - view[1]) * scale)  # noqa: E731  field px -> picture px
    step = next(s for s in (50, 100, 200, 250, 500) if (view[2] - view[0]) / s <= 12)
    font = _font(15)
    for v in np.arange(np.ceil(view[0] / step) * step, view[2], step):
        x = to(v, 0)[0]
        draw.line([(x, 0), (x, image.height)], fill=(255, 255, 255, 70), width=1)
        draw.text((x + 2, 2), f"x {int(v)}", fill=(0, 0, 0, 255), font=font, stroke_width=2, stroke_fill=(255, 255, 255))
    for v in np.arange(np.ceil(view[1] / step) * step, view[3], step):
        y = to(0, v)[1]
        draw.line([(0, y), (image.width, y)], fill=(255, 255, 255, 70), width=1)
        draw.text((2, y + 2), f"y {int(v)}", fill=(0, 0, 0, 255), font=font, stroke_width=2, stroke_fill=(255, 255, 255))
    draw.rectangle([to(box[0], box[1]), to(box[2], box[3])], outline=(40, 90, 255, 255), width=3)
    if objects is not None:
        label = _font(17)
        for r in objects.itertuples():
            colour = (0, 220, 0, 255) if r.accepted else (255, 210, 0, 255)
            if r.outline_points:
                draw.line([to(*p) for p in [*r.outline_points, r.outline_points[0]]], fill=colour, width=2)
            px, py = to(r.x, r.y)
            draw.text((px + 6, py - 20), str(r.number), fill=(0, 0, 0, 255), font=label, stroke_width=3, stroke_fill=(255, 255, 255))
    banner = Image.new("RGB", (image.width, 34), (255, 255, 255))
    ImageDraw.Draw(banner).text((6, 6), title, fill=(0, 0, 0), font=_font(18))
    out = Image.new("RGB", (image.width, image.height + 34), (255, 255, 255))
    out.paste(banner, (0, 0))
    out.paste(image.convert("RGB"), (0, 34))
    return np.asarray(out)


def _jpeg(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])


# ---------------------------------------------------------------- halves

def split_halves(fields: pd.DataFrame, counts: dict[str, int], seed: int) -> dict[str, str]:
    """field code -> half: the fields of each category dealt to the half with fewer objects so far."""
    rng = np.random.default_rng(seed)
    total = {h: 0 for h in HALVES}
    size = {h: 0 for h in HALVES}
    out = {}
    order = fields.iloc[rng.permutation(len(fields))]
    for category, _ in MIX:
        for code in order[order.category == category].field:
            half = min(HALVES, key=lambda h: (size[h], total[h], h))
            out[code] = half
            size[half] += 1
            total[half] += counts.get(code, 0)
    for code in order.field:
        out.setdefault(code, min(HALVES, key=lambda h: (size[h], total[h], h)))
    return out


def overlap(objects: pd.DataFrame, share: float, seed: int) -> pd.Series:
    """The other half each object is also labelled in ("" for none): about `share` of all objects, half of them from
    each half, drawn evenly over detection's calls (accepted or not) and fields."""
    rng = np.random.default_rng(seed)
    out = pd.Series("", index=objects.index, dtype=object)
    wanted = int(round(share * len(objects)))
    for half in HALVES:
        mine = objects[objects.half == half]
        k = min(len(mine), wanted // 2 + (wanted % 2 if half == HALVES[0] else 0))
        if not k:
            continue
        order = mine.iloc[rng.permutation(len(mine))].sort_values(["accepted", "field"], kind="stable")
        picked = order.iloc[np.linspace(0, len(order) - 1, k).round().astype(int)].index if k < len(order) else order.index
        out.loc[picked] = HALVES[1] if half == HALVES[0] else HALVES[0]
    return out


# ---------------------------------------------------------------- instructions

def instructions(vocabulary: vocab.Vocabulary, stain: str, round_: int) -> str:
    """REVIEW_INSTRUCTIONS.md of one stain."""
    rows = "\n".join(f"| `{c.label}` | {c.key} | {c.name} | {c.description} |" for c in choices(vocabulary))
    reject = next((c for c in choices(vocabulary) if c.group == "reject"), None)
    lookalikes = [c for c in choices(vocabulary) if c.group == "lookalike"]
    named = ", ".join(f"`{c.label}`" for c in lookalikes) or "(none for this stain)"
    positive = " / ".join(sorted({c.identity for c in choices(vocabulary) if c.group == "counted"}))
    return f"""# AI review round {round_}: {stain}

You label every numbered object of a few fields of {stain}-stained human brain tissue (a TMA core, 0.27 µm per pixel).
Your labels train and evaluate stainID's detection models; nothing you write changes the analysis directly.

## What you get

- `sheets/<half>_NN.jpg`: numbered panels, 20 per sheet. Each panel: **left** the raw crop (at least 100 µm across,
  the object about 60 % of it), **right** the same crop with detection's outline (green) and the object's centre (red
  cross). Below: the area (µm²), detection's call and probability (`p`, with the model's threshold) and why detection
  decided (`accepted`, `model`, or a rule such as `rule: shape`). Detection is often wrong: judge the picture.
- `overviews/F0N.jpg`: each field (or the part of it asked about, the **blue box**) with every object's number and
  outline (green: detection accepted it, yellow: rejected), on a grid labelled in field pixels (`x`, `y`).
  `overviews/F0N_raw.jpg` is the same without drawings. Use them for context and to find **missed objects**.
- `assignment_<half>.md`: your sheets, fields and objects; `review_labels_<half>.csv`: the file you fill in.

## Labels for {stain} (write the `label` column exactly as in the first column)

| label | key | class | what it looks like |
| --- | --- | --- | --- |
{rows}

- **Counted ({positive})**: the object stainID counts. Give its type when you can see it.
- **Named look-alike** ({named}): something real and recognisable that is *not* the counted object but is easily
  taken for it. Name it rather than rejecting it: these teach the model what to leave out.
- **Reject** (`{reject.label if reject else ''}`, Del): not a {stain} object and none of the look-alikes: background,
  a speck or fibre, a blue-only nucleus, debris, a fold, an edge, stain precipitate.
- **Unsure** (`unsure`, `): you cannot decide from the picture. It is left out of training; use it rarely, and say
  why in the rationale. Prefer a label with `confidence` low over `unsure` when you lean one way.

## Outline (`outline_ok`)

Judge the green outline of the right picture for counted objects and look-alikes (leave it empty for reject and
unsure):

- `ok`: it follows the object's edge closely enough to measure its size (within about a fifth of its area);
- `too big`: it takes in clearly more than the object (background, a neighbour's part);
- `too small`: it misses a clear part of the object;
- `wrong object`: it outlines something else (a neighbour, a fragment) rather than the object under the red cross.

## Missed objects

On the overview of each of *your* fields (not the extra objects from the other half), look for clear {positive}
objects inside the blue box that have **no number**. Add one row per missed object: `review_id` = `{stain}-F0N-M1`,
`M2`, … (the field's code), `label` its class, `field_x`, `field_y` its centre read from the grid (field pixels;
within about 20 px is fine), `confidence`, a short rationale; leave `outline_ok` empty. Report only clear ones.

## The CSV

`review_labels_<half>.csv`, one row per numbered object of your assignment (the ids are filled in) plus any missed
objects. Columns:

| column | value |
| --- | --- |
| `review_id` | the panel's id, e.g. `{stain}-F03-017` (missed objects: `{stain}-F03-M1`) |
| `label` | one label from the table above, written exactly |
| `confidence` | `high`, `moderate` or `low` |
| `outline_ok` | `ok`, `too big`, `too small`, `wrong object`, or empty (reject, unsure, missed objects) |
| `rationale` | at most 12 words: what you see |
| `reviewer` | your reviewer name (the same on every row; it becomes the account `ai:<name>`) |
| `field_x`, `field_y` | only for missed objects: their centre in field pixels |

Label every numbered object of your assignment; do not leave rows empty. Judge each object on its own picture; the
same object may be shown to another reviewer too (that is how agreement is measured), so do not coordinate.
"""


# ---------------------------------------------------------------- building

def _safe_code(stain: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", stain)


def build_stain(project: Project, stain: str, round_: int = ROUND, n_fields: int = FIELDS, seed: int = 0, log=print,
                most: int = MOST, fewest: int = FEWEST) -> dict:
    """Choose a stain's fields and write its batch (`stain_folder`); returns its summary."""
    from stainid.sampling.tiles import known_tiles  # noqa: PLC0415

    tiles = known_tiles(project)
    labels = store.current(project.output("reviews"))
    out = stain_folder(project, stain, round_)
    if (out / "manifest.csv").exists():
        raise FileExistsError(f"{out} has a batch already: remove it (or choose another round) to build it again")
    skip_fields, skip_cores = taken(project, stain, tiles, labels)
    stats = field_stats(project, stain, tiles)
    stats = stats[~stats.tile_id.isin(skip_fields) & ~stats.core_id.astype(str).isin(skip_cores)]
    checked = {c for (c, s) in status.areas_checked(labels, stain)}
    chosen = choose_fields(stats, n_fields, seed, checked)
    if chosen.empty:
        raise ValueError(f"no {stain} field can be chosen (none is ready and free)")
    chosen = chosen.iloc[np.random.default_rng(seed + 1).permutation(len(chosen))].reset_index(drop=True)
    chosen["field"] = [f"F{i + 1:02d}" for i in range(len(chosen))]
    boxes = tiles.drop_duplicates("tile_id").set_index("tile_id")
    vocabulary = vocab.of(project, stain)
    rows, field_rows, regions = [], [], {}
    for f in chosen.itertuples():
        tile = {**boxes.loc[f.tile_id].to_dict(), "tile_id": f.tile_id}
        objects = field_objects(project, stain, tile)
        fx0, fy0 = frames.field_origin(tile, project.context_px)
        own_box = (int(tile["x_px"]) - fx0, int(tile["y_px"]) - fy0, int(tile["x_px"]) - fx0 + int(tile["width_px"]),
                   int(tile["y_px"]) - fy0 + int(tile["height_px"]))
        box, grid = window(objects, own_box, f.category, most, fewest)
        inside = objects[(objects.x >= box[0]) & (objects.x < box[2]) & (objects.y >= box[1]) & (objects.y < box[3])]
        asked = stratified(inside, most, seed + len(rows))
        capped = len(asked) < len(inside)
        asked = asked.sort_values(["y", "x"]).reset_index(drop=True).assign(number=lambda a: np.arange(1, len(a) + 1))
        field_rows.append({"field": f.field, "tile_id": f.tile_id, "core_id": tile["core_id"], "tma": tile["tma"],
                           "disease_group": tile.get("disease_group", ""), "region": tile.get("region", ""), "category": f.category,
                           "areas_checked": bool(f.areas_checked), "field_candidates": len(objects), "field_accepted": int(objects.accepted.sum()),
                           "field_near_threshold": int(objects.near.sum()), "grid": grid, "subfield": grid > 1,
                           "box_x0": round(box[0], 1), "box_y0": round(box[1], 1), "box_x1": round(box[2], 1), "box_y1": round(box[3], 1),
                           "in_box": len(inside), "asked": len(asked), "capped": capped,
                           "sampling": ("whole field" if grid == 1 else f"sub-field: cell of the tile cut {grid} x {grid}")
                           + (f"; stratified sample of {len(asked)} of {len(inside)}" if capped else "")})
        regions[f.field] = (tile, (fx0, fy0), box)
        for r in asked.itertuples():
            rows.append({"review_id": f"{_safe_code(stain)}-{f.field}-{r.number:03d}", "stain": stain, "field": f.field, "number": int(r.number),
                         "tile_id": f.tile_id, "core_id": tile["core_id"], "tma": tile["tma"], "candidate_id": r.candidate_id,
                         "x": round(float(r.x), 1), "y": round(float(r.y), 1), "x_core": float(r.x_core), "y_core": float(r.y_core),
                         "area_um2": round(float(r.area_um2), 1) if pd.notna(r.area_um2) else np.nan, "radius_px": float(candidate_store.radius(r, project.pixel_size_um)),
                         "probability": round(float(r.probability), 4) if pd.notna(r.probability) else np.nan, "threshold": float(r.threshold),
                         "decision": r.decision, "reason": r.reason, "why": why(r), "accepted": bool(r.accepted), "near_threshold": bool(r.near),
                         "ai_identity": r.ai_identity, "ai_subtype": r.ai_subtype, "ai_label": f"{r.ai_identity} / {r.ai_subtype}" if r.ai_subtype else r.ai_identity,
                         "outline_choice": r.outline_choice, "outline_points": json.dumps(np.round(np.asarray(r.outline_points, float), 1).tolist())
                         if r.outline_points else "", "category": f.category})
        log(f"{stain} {f.field} {f.tile_id} ({f.category}): {len(asked)} objects" + (f", {field_rows[-1]['sampling']}" if grid > 1 or capped else ""), flush=True)
    manifest = pd.DataFrame(rows)
    fields = pd.DataFrame(field_rows)
    halves = split_halves(fields, manifest.groupby("field").size().to_dict(), seed + 2)
    fields["half"] = fields.field.map(halves)
    manifest["half"] = manifest.field.map(halves)
    manifest["also_in"] = overlap(manifest, OVERLAP, seed + 3)
    out.mkdir(parents=True, exist_ok=True)
    _pictures(project, stain, out, manifest, regions, vocabulary, log)
    manifest.to_csv(out / "manifest.csv", index=False)
    fields.to_csv(out / "fields.csv", index=False)
    (out / "REVIEW_INSTRUCTIONS.md").write_text(instructions(vocabulary, stain, round_))
    assignments = _assignments(stain, out, manifest, fields, round_)
    summary = {"stain": stain, "round": round_, "seed": seed, "created": time.strftime("%Y-%m-%d %H:%M:%S"), "fields": len(fields),
               "objects": int(len(manifest)), "overlap": int((manifest.also_in != "").sum()), "assignments": assignments,
               "by_category": fields.groupby("category").asked.sum().astype(int).to_dict(), "subfields": int(fields.subfield.sum()),
               "capped": int(fields.capped.sum()), "fields_left_out": len(skip_fields), "cores_left_out": len(skip_cores),
               "labels": [c.label for c in choices(vocabulary)]}
    (out / "batch.json").write_text(json.dumps(summary, indent=1, default=str))
    return summary


def _pictures(project: Project, stain: str, out: Path, manifest: pd.DataFrame, regions: dict, vocabulary: vocab.Vocabulary, log) -> None:
    """The panels (as sheets per half) and the overviews of every field: each core image read once per field."""
    panels: dict[str, np.ndarray] = {}
    pixel_um = project.pixel_size_um
    for code, (tile, (fx0, fy0), box) in regions.items():
        mine = manifest[manifest.field == code]
        side = max(box[2] - box[0], box[3] - box[1])
        read = (int(fx0 + box[0] - READ_MARGIN), int(fy0 + box[1] - READ_MARGIN), int(fx0 + box[0] + side + READ_MARGIN), int(fy0 + box[1] + side + READ_MARGIN))
        region, origin = read_region(project.root / tile["image_path"], read)
        for r in mine.itertuples():
            points = json.loads(r.outline_points) if r.outline_points else None
            size = crop_side(points, r.radius_px, pixel_um)
            raw = crop(region, origin, fx0 + r.x, fy0 + r.y, size, PANEL_PX)
            k = PANEL_PX / size
            shown = [[(px - r.x) * k + PANEL_PX / 2, (py - r.y) * k + PANEL_PX / 2] for px, py in points] if points else None
            call = vocabulary.display(r.ai_identity, r.ai_subtype if isinstance(r.ai_subtype, str) else "")
            p = "" if pd.isna(r.probability) else f"p {r.probability:.2f} (thr {r.threshold:.2f})"
            lines = [f"{r.area_um2:.0f} µm² · {size * pixel_um:.0f} µm wide", f"det: {call} · {p} · {r.why}"]
            panels[r.review_id] = panel(raw, shown, (PANEL_PX / 2, PANEL_PX / 2), lines, r.review_id)
        objects = mine.assign(outline_points=[json.loads(t) if t else None for t in mine.outline_points])
        title = f"{stain} {code} · blue box: the objects asked (numbered) · green: detection accepted · yellow: rejected · grid in field px"
        _jpeg(out / "overviews" / f"{code}.jpg", overview(region, origin, (fx0, fy0), box, objects, title))
        _jpeg(out / "overviews" / f"{code}_raw.jpg", overview(region, origin, (fx0, fy0), box, None, f"{stain} {code} (raw) · grid in field px"))
        log(f"  {code}: {len(mine)} panels and the overview", flush=True)
    for half in HALVES:
        own = manifest[manifest.half == half].sort_values(["field", "number"])
        extra = manifest[manifest.also_in == half].sort_values(["field", "number"])
        sheets = []
        for kind, part in (("", own), ("x", extra)):
            ids = list(part.review_id)
            for n in range(0, len(ids), PANELS):
                chunk = ids[n:n + PANELS]
                name = f"{half}_{kind}{len([s for s in sheets if s[1] == kind]) + 1:02d}"
                what = "extra objects from the other half" if kind else "fields " + ", ".join(dict.fromkeys(manifest.set_index("review_id").field[chunk]))
                _jpeg(out / "sheets" / f"{name}.jpg", sheet([panels[i] for i in chunk], f"{stain} · half {half} · sheet {name} · {what}"))
                sheets.append((name, kind, chunk))
        for name, _, chunk in sheets:
            for i, review_id in enumerate(chunk, start=1):
                column = "sheet" if manifest.loc[manifest.review_id == review_id, "half"].iloc[0] == half else "extra_sheet"
                manifest.loc[manifest.review_id == review_id, column] = name
                manifest.loc[manifest.review_id == review_id, column + "_panel"] = i


def _assignments(stain: str, out: Path, manifest: pd.DataFrame, fields: pd.DataFrame, round_: int) -> list[str]:
    made = []
    for half in HALVES:
        mine = manifest[(manifest.half == half) | (manifest.also_in == half)].copy()
        mine["sheet_here"] = np.where(mine.half == half, mine.get("sheet"), mine.get("extra_sheet"))
        mine["panel_here"] = np.where(mine.half == half, mine.get("sheet_panel"), mine.get("extra_sheet_panel"))
        mine["extra"] = mine.half != half
        mine = mine.sort_values(["extra", "sheet_here", "panel_here"])
        mine[["review_id", "field", "sheet_here", "panel_here", "extra"]].rename(columns={"sheet_here": "sheet", "panel_here": "panel"}).to_csv(
            out / f"assignment_{half}.csv", index=False)
        template = pd.DataFrame({c: (mine.review_id.to_numpy() if c == "review_id" else "") for c in CSV_COLUMNS})
        target = out / f"review_labels_{half}.csv"
        if not target.exists():
            template.to_csv(target, index=False)
        own_fields = fields[fields.half == half]
        sheets = sorted(set(mine.sheet_here.dropna()))
        lines = [f"# AI review round {round_} · {stain} · half {half}", "",
                 f"Folder: `{out}`", "",
                 "1. Read `REVIEW_INSTRUCTIONS.md` (the labels, outline judgement and CSV columns).",
                 "2. For each of your fields, look at its overview (`overviews/<field>.jpg`, raw: `<field>_raw.jpg`), then its sheets.",
                 f"3. Fill in `review_labels_{half}.csv` (the review ids are listed; keep the header), one row per object, plus "
                 "missed-object rows for your fields.", "",
                 f"Objects: {len(mine)} ({int((~mine.extra).sum())} of your fields, {int(mine.extra.sum())} extra from the other half: "
                 "label those like any other, no missed objects for them).", "",
                 "| field | overview | asked | sheets |", "| --- | --- | --- | --- |"]
        for f in own_fields.itertuples():
            here = sorted(set(mine[(mine.field == f.field) & ~mine.extra].sheet_here))
            lines.append(f"| {f.field} | overviews/{f.field}.jpg | {f.asked} objects ({'the blue box: part of the field' if f.subfield else 'the whole field'}) | "
                         + ", ".join(f"sheets/{s}.jpg" for s in here) + " |")
        extra = sorted(set(mine[mine.extra].sheet_here))
        if extra:
            lines.append(f"| extra | (none) | {int(mine.extra.sum())} objects | " + ", ".join(f"sheets/{s}.jpg" for s in extra) + " |")
        lines += ["", f"All sheets in order: {', '.join(f'sheets/{s}.jpg' for s in sheets)}", "",
                  f"Write: `{out / f'review_labels_{half}.csv'}`"]
        (out / f"assignment_{half}.md").write_text("\n".join(lines) + "\n")
        made.append(str(out / f"assignment_{half}.md"))
    return made


def build(project: Project, stains: list[str], round_: int = ROUND, n_fields: int = FIELDS, seed: int = 0, log=print) -> dict:
    """Every stain's batch, and the list of the reviewers' assignments (`ASSIGNMENTS.md`)."""
    summaries = {}
    for stain in stains:
        summaries[stain] = build_stain(project, stain, round_, n_fields, seed, log)
        log(f"{stain}: {summaries[stain]['objects']} objects in {summaries[stain]['fields']} fields", flush=True)
    top = folder(project, round_)
    rows = ["# AI review round %d: assignments" % round_, "",
            "One reviewer per line. Each reads its stain's REVIEW_INSTRUCTIONS.md and its assignment, and writes the CSV named there.", "",
            "| reviewer | stain | half | assignment | objects |", "| --- | --- | --- | --- | --- |"]
    n = 0
    for stain in sorted(set(summaries) | {p.name for p in top.iterdir() if (p / "batch.json").exists()}):
        manifest = pd.read_csv(top / stain / "manifest.csv")
        for half in HALVES:
            n += 1
            count = int(((manifest.half == half) | (manifest.also_in == half)).sum())
            rows.append(f"| {n} | {stain} | {half} | {top / stain / f'assignment_{half}.md'} | {count} |")
    rows += ["", f"Ingest when done: `stainid --project {project.root} review ai-batch ingest --round {round_} --stain <stain>`"]
    (top / "ASSIGNMENTS.md").write_text("\n".join(rows) + "\n")
    return summaries
