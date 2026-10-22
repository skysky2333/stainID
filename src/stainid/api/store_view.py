"""The core viewer's layers from the v2 candidate store (`stainid.candidates`), read only: every candidate with its call
(accepted and rejected, every stain), its exact outline, the AT8 threads as saved (never traced again) and the stained
regions no candidate covers.

Which detection the viewer shows is the project setting `v2.viewer`: "auto" (the default: the v2 store where it has the
core image, else the current outputs), "v2" or "current".
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stainid import candidates, palette
from stainid import exclusions as excluded_areas

COLUMNS = ["candidate_id", "x_core", "y_core", "radius_px", "area_um2", "probability", "decision", "reason", "class", "subtype",
           "m_from_ring", "outline", "sam_outline", "sam_status", "outline_default", "sam_area_um2"]


def setting(project) -> str:
    return str((project.config.get("v2") or {}).get("viewer", "auto"))


def available(project, core_id: str, stain: str) -> bool:
    """Whether the viewer shows the v2 store for this core image."""
    if setting(project) == "current":
        return False
    return candidates.path(project, stain, core_id, "candidates").exists()


LOOKALIKE = "lookalike"  # the viewer's class of a candidate detection called a named look-alike (CAA, an AT8 thread)


def model_class(stain: str, frame: pd.DataFrame) -> np.ndarray:
    """The viewer's class of each candidate (its colours and legend): the same names as for the current outputs
    ("rejected" for detection's rejects; a look-alike is set apart by `found`)."""
    accepted = frame.decision.eq("accepted").to_numpy()
    out = np.full(len(frame), "rejected", dtype=object)
    if stain == "NeuN":
        out[accepted] = "neuron"
    elif stain == "6E10":
        subtype = np.array([{"small": "small_plaque"}.get(s, s) for s in frame.subtype.fillna("").astype(str)], dtype=object)
        out[accepted] = subtype[accepted]
    else:
        ring = pd.to_numeric(frame.get("m_from_ring", pd.Series(0, index=frame.index)), errors="coerce").fillna(0).to_numpy() == 1
        out[accepted] = np.where(ring[accepted], "tau_neuron_ring", "tau_neuron_dense")
    return out


def found(project, core_id: str, stain: str) -> pd.DataFrame:
    """Every candidate of the core image as the viewer's object rows (x, y in core pixels), with its label class (`class`,
    `subtype`: label-vocabulary ids, empty for rejected ones), so the viewer colours it as every other page does. A
    candidate detection called a named look-alike (`candidate_store.lookalike`: a deposit on a vessel wall is CAA) is
    that class, `model_class` "lookalike": drawn solid in the class's colour, not dotted as a reject; one that is a
    reject reason (`Vocabulary.rejecting`: a thread-like stain) the page draws dotted grey with its name."""
    from stainid.labels import candidate_store  # noqa: PLC0415
    from stainid.labels import vocabulary as vocab  # noqa: PLC0415

    frame = candidates.read_core(project, stain, core_id, COLUMNS)
    if frame.empty:
        return pd.DataFrame(columns=["x", "y", "radius_px", "model_class", "probability", "area_um2", "reason", "candidate_id", "class", "subtype"])
    accepted = frame.decision.eq("accepted")
    text = lambda column: frame[column].fillna("").astype(str).where(accepted, "") if column in frame else ""  # noqa: E731
    classes, subtypes, shown = text("class"), text("subtype"), model_class(stain, frame)
    if "class" in frame and (~accepted).any():
        words = vocab.of(project, stain)
        reasons = frame.reason.fillna("").astype(str) if "reason" in frame else pd.Series("", index=frame.index)
        pairs = pd.Series(list(zip(frame["class"].fillna("").astype(str), reasons)), index=frame.index)[~accepted]
        named = {pair: candidate_store.lookalike(words, *pair) for pair in set(pairs)}
        found_named = pairs.map(named)
        found_named = found_named[found_named != ""]
        if len(found_named):
            classes = classes.copy()
            classes.loc[found_named.index] = found_named
            shown[frame.index.get_indexer(found_named.index)] = LOOKALIKE
    return pd.DataFrame({"x": frame.x_core.round(1), "y": frame.y_core.round(1), "radius_px": frame.radius_px.round(1),
                         "model_class": shown, "probability": pd.to_numeric(frame.probability, errors="coerce").round(3),
                         "area_um2": frame.area_um2.round(1), "reason": frame.reason.fillna(""), "candidate_id": frame.candidate_id,
                         "class": classes, "subtype": subtypes})


OUTLINE_CHOICES = ("default", "stain", "sam", "sam0", "sam1", "sam2")  # which outline the core viewer draws for every candidate


def _flat_points(value) -> list[int]:
    """An outline cell (store text) or point list as flat whole pixels ([] for none)."""
    if value is None:
        return []
    if isinstance(value, str):
        return candidates.points(value) if value.strip() else []
    flat = np.rint(np.asarray(value, float).reshape(-1)).astype(np.int64).tolist()
    return flat if len(flat) >= 6 else []


def _polygon_um2(flat: list[int], pixel_um: float) -> float:
    xs, ys = np.asarray(flat[0::2], float) + 0.5, np.asarray(flat[1::2], float) + 0.5
    return float(abs(np.dot(xs, np.roll(ys, -1)) - np.dot(ys, np.roll(xs, -1))) / 2) * pixel_um ** 2


def outline_rows(project, core_id: str, stain: str, outline: str = "default") -> list[list]:
    """Each candidate's outline: [x, y, area µm², rings]. `outline`: "default" the outline the stain draws
    (`candidates.outline_rings`, with the area the measures use: `candidates.shown_area_um2`), "stain" the candidate's own outline (as the stain's segmentation found it), "sam0" /
    "sam1" / "sam2" the small / medium / large Segment Anything output (its own area), else the candidate's own outline
    where Segment Anything gave none."""
    if outline not in OUTLINE_CHOICES:
        raise ValueError(f"outline must be one of {', '.join(OUTLINE_CHOICES)}")
    sam = outline.startswith("sam")
    frame = candidates.read_core(project, stain, core_id, [*COLUMNS, "sam_masks", "sam_default"] if sam else COLUMNS)
    shown = candidates.shown_area_um2(frame).to_numpy(float) if outline == "default" and not frame.empty else None
    rows = []
    for number, row in enumerate(frame.to_dict("records")):
        area = float(row["area_um2"])
        own = candidates.points(row.get("outline"))
        if outline == "default":
            rings = candidates.outline_rings(row)
            area = float(shown[number]) if np.isfinite(shown[number]) else area  # the drawn outline's, as the measures
        elif outline == "stain":
            rings = [own] if own else []
        elif outline == "sam":  # the Segment Anything output stage 3 kept (`candidates.choice_index`); the stain's where it has none
            options = candidates.outline_choices(row)[0]
            index = candidates.choice_index("sam", row)
            chosen = _flat_points(options[index]) if index else []
            rings = [chosen] if chosen else ([own] if own else [])
            if chosen:
                area = _polygon_um2(chosen, project.pixel_size_um)
        else:
            masks = candidates.sam_masks(row.get("sam_masks"))
            index = int(outline[3:])
            chosen = masks[index] if index < len(masks) and len(masks[index]) >= 6 else []
            rings = [chosen] if chosen else ([own] if own else [])
            if chosen:
                area = _polygon_um2(chosen, project.pixel_size_um)
        if rings:
            rows.append([round(float(row["x_core"]), 1), round(float(row["y_core"]), 1), round(area, 1), rings])
    return rows


def threads(project, core_id: str) -> dict | None:
    """The saved AT8 thread skeleton (status "done"), or None when stage 2 has not saved it for this core."""
    if not candidates.threads_path(project, core_id).exists():
        return None
    frame = candidates.threads(project, core_id)
    accepted = set()
    store = candidates.read_core(project, "AT8", core_id, ["candidate_id", "decision"])
    if "decision" in store:
        accepted = set(store.candidate_id[store.decision == "accepted"])
    free = frame[~frame.candidate_id.isin(accepted)] if not frame.empty else frame
    rows = [[float(w), candidates.points(p)] for w, p in zip(free.width_um, free.points)] if not free.empty else []
    length = float(free.length_um.sum()) / 1000 if not free.empty else 0.0
    measured = {"length_mm": round(length, 2), "skeleton_px": int(free.n_px.sum()) if not free.empty else 0}
    return {"status": "done", "tiles_done": int(frame.tile_id.nunique()) if not frame.empty else 0, "tiles_total": int(frame.tile_id.nunique()) if not frame.empty else 0,
            "pieces": len(rows), "length_mm": round(length, 2), "columns": ["width_um", "points"], "rows": rows, "measured": measured,
            "saved": True}


def unproposed(project, core_id: str, stain: str) -> dict:
    """The stained regions no candidate covers (status "not_run" before stage 2)."""
    path = candidates.path(project, stain, core_id, "unproposed")
    if not path.exists():
        return {"status": "not_run", "columns": [], "rows": []}
    frame = candidates.unproposed(project, stain, core_id)
    rows = [[round(float(r["x_core"]), 1), round(float(r["y_core"]), 1), round(float(r["area_um2"]), 1), round(float(r["dab_over_threshold"]), 2),
             [candidates.points(r["outline"])]] for r in frame.to_dict("records")]
    return {"status": "done", "columns": ["x", "y", "area_um2", "dab_over_threshold", "rings"], "rows": rows}


def unproposed_count(project, core_id: str, stain: str) -> int | None:
    path = candidates.path(project, stain, core_id, "unproposed")
    return len(candidates.unproposed(project, stain, core_id)) if path.exists() else None


# ---------------------------------------------------------------- excluded areas

class ExcludedKind:
    """One type of area left out of the analysis as the "Excluded from analysis" layer draws it: its code in the layer
    (`stainid.exclusions.CODES`), colour, hatching (a backslash, "/", "|", "-", "." fine dots, or "" for none), the brush
    type it is (`labels.exclusions.KINDS`; "" for the AT8 edge guard, which only detection makes) and how strongly it is
    drawn (`fill`, `line`: alpha of the wash and of the hatching)."""

    def __init__(self, id: str, code: int, color: tuple[int, int, int], hatch: str, type: str = "", fill: int = 70, line: int = 210):
        self.id, self.code, self.color, self.hatch, self.type, self.fill, self.line = id, code, color, hatch, type, fill, line
        self.label = excluded_areas.CODES[code]


EXCLUDED_KINDS = (  # the colours are the one palette's (stainid/palette.json "areas"), the brush's too
    ExcludedKind("fold", 1, palette.area("1"), "/", "1"),
    ExcludedKind("artifact", 2, palette.area("2"), "\\", "2"),
    ExcludedKind("no_tissue", 3, palette.area("3"), ".", "3", fill=40, line=120),  # lighter: glass is much of an edge tile
    ExcludedKind("focus", 4, palette.area("4"), "|", "4"),
    ExcludedKind("edge_guard", 5, palette.area("edge_guard"), "-"),
    ExcludedKind("not_analysed", 6, palette.area("not_analysed"), "/"),
)
EXCLUDED_SCALE = excluded_areas.SCALE  # core-image px per layer px
HATCH_PX = 4  # layer px between hatching lines (32 core-image px)


def excluded_path(project, core_id: str, stain: str):
    return excluded_areas.mask_path(project, core_id, stain)


def kind_info(kind: ExcludedKind, fraction: float | None) -> dict:
    return {"id": kind.id, "label": kind.label, "color": "#%02x%02x%02x" % kind.color, "hatch": kind.hatch, "type": kind.type, "fraction": fraction}


def _shares(codes: np.ndarray) -> list[dict]:
    """Each type's share of the core's area (every cell but those outside the image's tissue and tiles)."""
    analysed = max(1, int((codes != excluded_areas.OUTSIDE).sum()))
    counts = np.bincount(codes.ravel(), minlength=256)
    return [kind_info(kind, round(float(counts[kind.code]) / analysed, 4)) for kind in EXCLUDED_KINDS]


def excluded_info(project, core_id: str, stain: str) -> dict:
    """The "Excluded from analysis" layer of one core image as the views draw it (`stainid.exclusions.view`: detection's
    automatic areas, what the analysis uses, and the tissue in no analysed tile; people's areas are not in it): status
    ("done", or "not_made" before the v2 proposals), its size in layer px and scale (core-image px per layer px),
    `version` (changes with the layer) and each type with its colour, hatching and share of the core's area."""
    layer = excluded_areas.view(project, core_id, stain)
    if layer is None:
        return {"status": "not_made", "kinds": [kind_info(k, None) for k in EXCLUDED_KINDS]}
    height, width = layer.codes.shape
    return {"status": "done", "scale": layer.scale, "width": int(width), "height": int(height), "version": layer.version,
            "kinds": _shares(layer.codes)}


def _rgba(codes: np.ndarray, rows: int = 0, columns: int = 0) -> bytes:
    """Codes of the layer as an RGBA PNG: each type a translucent wash in its colour with stronger hatching lines ("no
    tissue" lighter: a faint grey wash with fine dots), counted tissue transparent; (`rows`, `columns`) where `codes` start in the core's layer (so a field's
    hatching lines up with its core's)."""
    import cv2  # noqa: PLC0415

    height, width = codes.shape
    yy, xx = np.mgrid[rows:rows + height, columns:columns + width]
    lines = {"\\": (xx - yy) % HATCH_PX == 0, "/": (xx + yy) % HATCH_PX == 0, "|": xx % HATCH_PX == 0, "-": yy % HATCH_PX == 0,
             ".": (xx % 3 == 0) & (yy % 3 == 0)}
    out = np.zeros((height, width, 4), np.uint8)  # cv2 writes BGRA
    for kind in EXCLUDED_KINDS:
        pixels = codes == kind.code
        if not pixels.any():
            continue
        r, g, b = kind.color
        out[pixels] = (b, g, r, kind.fill)
        if kind.hatch:
            out[pixels & lines[kind.hatch]] = (b, g, r, kind.line)
    ok, encoded = cv2.imencode(".png", out)
    if not ok:
        raise RuntimeError("could not encode the excluded areas")
    return encoded.tobytes()


_PICTURES: dict[tuple, bytes] = {}  # (project, core, stain, layer digest, window) -> PNG; a few hundred at most


def _picture(project, core_id: str, stain: str, layer, window: tuple[slice, slice] | None = None) -> bytes:
    where = None if window is None else (window[0].start, window[0].stop, window[1].start, window[1].stop)
    key = (str(project.root), core_id, stain, layer.version, where)
    found = _PICTURES.get(key)
    if found is None:
        codes = layer.codes if window is None else layer.codes[window]
        found = _rgba(codes, *((0, 0) if window is None else (window[0].start, window[1].start)))
        if len(_PICTURES) >= 512:
            _PICTURES.pop(next(iter(_PICTURES)))
        _PICTURES[key] = found
    return found


def _cells(layer, x0: float, y0: float, x1: float, y1: float) -> tuple[slice, slice]:
    """The rows and columns of the layer under a core-pixel box (clipped to it)."""
    h, w = layer.codes.shape
    c0, r0 = max(0, int(x0) // layer.scale), max(0, int(y0) // layer.scale)
    c1, r1 = min(w, -(-int(x1) // layer.scale)), min(h, -(-int(y1) // layer.scale))
    return slice(r0, max(r0, r1)), slice(c0, max(c0, c1))


def excluded_png(project, core_id: str, stain: str) -> bytes | None:
    """The automatic layer as an RGBA PNG at its own resolution (`_rgba`); None before the v2 proposals."""
    layer = excluded_areas.view(project, core_id, stain)
    return None if layer is None else _picture(project, core_id, stain, layer)


def field_excluded(project, tile: dict) -> tuple[dict, bytes | None]:
    """The excluded areas of one field for the annotation pages: its part of the core's automatic layer (the same pixels
    as the Cores view), where that part lies in the field picture (`x`, `y`, `width`, `height` in field px, `scale` field
    px per pixel of the picture), the types present in it, and the PNG."""
    from stainid.labels import frames  # noqa: PLC0415

    core_id, stain = str(tile["core_id"]), str(tile["stain"])
    layer = excluded_areas.view(project, core_id, stain)
    if layer is None:
        return {"status": "not_made", "kinds": [kind_info(k, None) for k in EXCLUDED_KINDS]}, None
    x0, y0 = frames.field_origin(tile, project.context_px)
    x1, y1 = int(tile["x_px"]) + int(tile["width_px"]) + project.context_px, int(tile["y_px"]) + int(tile["height_px"]) + project.context_px
    window = _cells(layer, x0, y0, x1, y1)
    codes = layer.codes[window]
    scale = layer.scale
    info = {"status": "done", "scale": scale, "x": window[1].start * scale - x0, "y": window[0].start * scale - y0,
            "width": codes.shape[1] * scale, "height": codes.shape[0] * scale, "version": layer.version,
            "kinds": [k for k in _shares(codes) if k["fraction"]]}
    return info, _picture(project, core_id, stain, layer, window)


PAINTED_TYPES = ("1", "2", "3", "4")  # the types the brush paints (labels.exclusions.KINDS)


def field_areas(project, tile: dict, session: excluded_areas.Scope = None) -> dict:
    """Each brush type's current area over a field (`current_areas`), placed in the field picture's pixels. `session`:
    detection's areas with that labelling session's edits only."""
    from stainid.labels import frames  # noqa: PLC0415

    x0, y0 = frames.field_origin(tile, project.context_px)
    x1, y1 = int(tile["x_px"]) + int(tile["width_px"]) + project.context_px, int(tile["y_px"]) + int(tile["height_px"]) + project.context_px
    return current_areas(project, str(tile["core_id"]), str(tile["stain"]), (x0, y0, x1, y1), ai=False, session=session)


def not_counted(project, tile: dict, session: excluded_areas.Scope = None) -> dict:
    """AT8 threads mode: where the field is not counted, so the page shades it and fades the thread lines there (people
    judge the thread threshold on counted tissue only). Detection's automatic areas (fold, artifact, no tissue, focus,
    the edge guard, …) with people's saved edits on top (`exclusions.final_say`, without AI areas), as one
    white-on-transparent PNG (data URL, one pixel per layer cell) placed in field-picture px, and `labels`: a point well
    inside each sizeable region (field px) for a small "not counted" note."""
    import base64  # noqa: PLC0415

    import cv2  # noqa: PLC0415

    from stainid.labels import frames  # noqa: PLC0415

    layer = excluded_areas.final_say(project, str(tile["core_id"]), str(tile["stain"]), ai=False, session=session)
    if layer is None:
        return {"status": "not_made"}
    x0, y0 = frames.field_origin(tile, project.context_px)
    x1, y1 = int(tile["x_px"]) + int(tile["width_px"]) + project.context_px, int(tile["y_px"]) + int(tile["height_px"]) + project.context_px
    window = _cells(layer, x0, y0, x1, y1)
    codes = layer.codes[window]
    cells = (codes != excluded_areas.COUNTED) & (codes != excluded_areas.OUTSIDE)
    scale = layer.scale
    ox, oy = window[1].start * scale - x0, window[0].start * scale - y0
    out = {"status": "done", "scale": scale, "x": ox, "y": oy, "width": codes.shape[1] * scale, "height": codes.shape[0] * scale,
           "version": layer.digest, "png": None, "labels": []}
    if not cells.any():
        return out
    rgba = np.zeros((*codes.shape, 4), np.uint8)
    rgba[cells] = 255
    ok, png = cv2.imencode(".png", rgba)
    out["png"] = "data:image/png;base64," + base64.b64encode(png.tobytes()).decode() if ok else None
    inside = cv2.distanceTransform(np.pad(cells, 1).astype(np.uint8), cv2.DIST_L2, 3)[1:-1, 1:-1]
    count, regions, stats, _ = cv2.connectedComponentsWithStats(cells.astype(np.uint8), connectivity=8)
    big = [r for r in range(1, count) if stats[r, cv2.CC_STAT_AREA] * scale * scale >= 160 * 160]
    for r in sorted(big, key=lambda r: -stats[r, cv2.CC_STAT_AREA])[:8]:
        depth = np.where(regions == r, inside, 0)
        row, col = np.unravel_index(int(np.argmax(depth)), depth.shape)
        out["labels"].append([int(ox + (col + 0.5) * scale), int(oy + (row + 0.5) * scale)])
    return out


def current_areas(project, core_id: str, stain: str, box: tuple[int, int, int, int] | None = None, ai: bool = True,
                  session: excluded_areas.Scope = None) -> dict:
    """Each brush type's current area, as the one brush (field tool and "Mark bad areas") starts from it: detection's
    area of that type with every edit people saved merged in (`exclusions.final_say`, one region set per type), a
    white-on-transparent PNG per type (data URL, one pixel per layer cell) and where it lies (`x`, `y`, `width`,
    `height`, in px from the box's corner: the whole core image without a box). The brush adds to and takes from these
    areas; nothing here is an analysis input. `ai` False (the annotation pages): without the AI reviewer's areas
    (`labels.store.is_ai_label`), detection's areas and people's edits only."""
    import base64  # noqa: PLC0415

    import cv2  # noqa: PLC0415

    layer = excluded_areas.final_say(project, core_id, stain, ai=ai, session=session)
    if layer is None:
        return {"status": "not_made", "kinds": {}}
    x0, y0, x1, y1 = box if box is not None else (0, 0, layer.codes.shape[1] * layer.scale, layer.codes.shape[0] * layer.scale)
    window = _cells(layer, x0, y0, x1, y1)
    codes = layer.codes[window]
    kinds = {}
    for kind in PAINTED_TYPES:
        cells = codes == int(kind)
        if not cells.any():
            continue
        rgba = np.zeros((*codes.shape, 4), np.uint8)
        rgba[cells] = 255
        ok, png = cv2.imencode(".png", rgba)
        if ok:
            kinds[kind] = "data:image/png;base64," + base64.b64encode(png.tobytes()).decode()
    return {"status": "done", "scale": layer.scale, "x": window[1].start * layer.scale - x0, "y": window[0].start * layer.scale - y0,
            "width": codes.shape[1] * layer.scale, "height": codes.shape[0] * layer.scale, "version": layer.digest, "kinds": kinds}


def people_areas(project, core_id: str, stain: str, origin: tuple[int, int] = (0, 0),
                 box: tuple[float, float, float, float] | None = None, ai: bool = True, session: excluded_areas.Scope = None) -> dict:
    """People's areas of a core image for the Annotations layers (`stainid.exclusions.annotated`; they train and evaluate
    models, the analysis never reads them): each painted or cleared area as polygons (core px less `origin`; only those
    reaching into `box`, x0 y0 x1 y1 core px), how they compare with detection's areas, and the edits (who, when, type,
    `ai`: the AI reviewer's). `ai` False (the annotation pages): people's own only (`labels.store.is_ai_label`)."""
    from stainid.labels.store import is_ai_label  # noqa: PLC0415

    found = excluded_areas.annotated(project, core_id, stain, ai=ai, session=session)
    if found is None:
        return {"regions": [], "agreement": None, "edits": []}
    regions = found.regions()
    if box is not None:
        x0, y0, x1, y1 = box

        def reaches(ring) -> bool:
            xs, ys = [p[0] for p in ring], [p[1] for p in ring]
            return min(xs) < x1 and max(xs) > x0 and min(ys) < y1 and max(ys) > y0

        regions = [r for r in regions if reaches(r["rings"][0])]
    ox, oy = origin
    if ox or oy:
        regions = [{"kind": r["kind"], "rings": [[[round(x - ox, 1), round(y - oy, 1)] for x, y in ring] for ring in r["rings"]]} for r in regions]
    auto = excluded_areas.automatic(project, core_id, stain)
    return {"regions": regions, "agreement": excluded_areas.agreement(auto, found) if auto is not None else None,
            "edits": [{"person": e.source, "time": e.time, "kinds": sorted(set(e.kinds)), "ai": is_ai_label(e.source)} for e in found.edits]}
