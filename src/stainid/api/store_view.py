"""The core viewer's layers from the v2 candidate store (`stainid.candidates`), read only: every candidate with its call
(accepted and rejected, every stain), its exact outline, the AT8 threads as saved (never traced again) and the stained
regions no candidate covers.

Which detection the viewer shows is the project setting `v2.viewer`: "auto" (the default: the v2 store where it has the
core image, else the current outputs), "v2" or "current".
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stainid import candidates
from stainid import exclusions as excluded_areas

COLUMNS = ["candidate_id", "x_core", "y_core", "radius_px", "area_um2", "probability", "decision", "reason", "class", "subtype",
           "m_from_ring", "outline", "sam_outline", "sam_status", "outline_default"]


def setting(project) -> str:
    return str((project.config.get("v2") or {}).get("viewer", "auto"))


def available(project, core_id: str, stain: str) -> bool:
    """Whether the viewer shows the v2 store for this core image."""
    if setting(project) == "current":
        return False
    return candidates.path(project, stain, core_id, "candidates").exists()


def model_class(stain: str, frame: pd.DataFrame) -> np.ndarray:
    """The viewer's class of each candidate (its colours and legend): the same names as for the current outputs."""
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
    `subtype`: label-vocabulary ids, empty for rejected ones), so the viewer colours it as every other page does."""
    frame = candidates.read_core(project, stain, core_id, COLUMNS)
    if frame.empty:
        return pd.DataFrame(columns=["x", "y", "radius_px", "model_class", "probability", "area_um2", "reason", "candidate_id", "class", "subtype"])
    accepted = frame.decision.eq("accepted")
    text = lambda column: frame[column].fillna("").astype(str).where(accepted, "") if column in frame else ""  # noqa: E731
    return pd.DataFrame({"x": frame.x_core.round(1), "y": frame.y_core.round(1), "radius_px": frame.radius_px.round(1),
                         "model_class": model_class(stain, frame), "probability": pd.to_numeric(frame.probability, errors="coerce").round(3),
                         "area_um2": frame.area_um2.round(1), "reason": frame.reason.fillna(""), "candidate_id": frame.candidate_id,
                         "class": text("class"), "subtype": text("subtype")})


OUTLINE_CHOICES = ("default", "stain", "sam0", "sam1", "sam2")  # which outline the core viewer draws for every candidate


def _polygon_um2(flat: list[int], pixel_um: float) -> float:
    xs, ys = np.asarray(flat[0::2], float) + 0.5, np.asarray(flat[1::2], float) + 0.5
    return float(abs(np.dot(xs, np.roll(ys, -1)) - np.dot(ys, np.roll(xs, -1))) / 2) * pixel_um ** 2


def outline_rows(project, core_id: str, stain: str, outline: str = "default") -> list[list]:
    """Each candidate's outline: [x, y, area µm², rings]. `outline`: "default" the outline the stain draws
    (`candidates.outline_rings`), "stain" the candidate's own outline (as the stain's segmentation found it), "sam0" /
    "sam1" / "sam2" the small / medium / large Segment Anything output (its own area), else the candidate's own outline
    where Segment Anything gave none."""
    if outline not in OUTLINE_CHOICES:
        raise ValueError(f"outline must be one of {', '.join(OUTLINE_CHOICES)}")
    sam = outline.startswith("sam")
    frame = candidates.read_core(project, stain, core_id, [*COLUMNS, "sam_masks"] if sam else COLUMNS)
    rows = []
    for row in frame.to_dict("records"):
        area = float(row["area_um2"])
        own = candidates.points(row.get("outline"))
        if outline == "default":
            rings = candidates.outline_rings(row)
        elif outline == "stain":
            rings = [own] if own else []
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
    (`stainid.exclusions.CODES`), colour, hatching (a backslash, "/", "|", "-", or "" for none) and the brush type it is
    (`labels.exclusions.KINDS`; "" for the AT8 edge guard, which only detection makes)."""

    def __init__(self, id: str, code: int, color: tuple[int, int, int], hatch: str, type: str = ""):
        self.id, self.code, self.color, self.hatch, self.type = id, code, color, hatch, type
        self.label = excluded_areas.CODES[code]


EXCLUDED_KINDS = (  # the colours are the brush's (webapp labels/brush.ts EXCLUDE_TYPES)
    ExcludedKind("fold", 1, (255, 176, 0), "/", "1"),
    ExcludedKind("artifact", 2, (168, 85, 247), "\\", "2"),
    ExcludedKind("no_tissue", 3, (156, 163, 175), "", "3"),
    ExcludedKind("focus", 4, (0, 229, 255), "|", "4"),
    ExcludedKind("edge_guard", 5, (20, 184, 166), "-"),
    ExcludedKind("not_analysed", 6, (100, 116, 139), "/"),
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
    """Codes of the layer as an RGBA PNG: each type a translucent fill in its colour with stronger hatching lines (none
    for "no tissue"), the rest transparent; (`rows`, `columns`) where `codes` start in the core's layer (so a field's
    hatching lines up with its core's)."""
    import cv2  # noqa: PLC0415

    height, width = codes.shape
    yy, xx = np.mgrid[rows:rows + height, columns:columns + width]
    lines = {"\\": (xx - yy) % HATCH_PX == 0, "/": (xx + yy) % HATCH_PX == 0, "|": xx % HATCH_PX == 0, "-": yy % HATCH_PX == 0}
    out = np.zeros((height, width, 4), np.uint8)  # cv2 writes BGRA
    for kind in EXCLUDED_KINDS:
        pixels = codes == kind.code
        if not pixels.any():
            continue
        r, g, b = kind.color
        out[pixels] = (b, g, r, 70)
        if kind.hatch:
            out[pixels & lines[kind.hatch]] = (b, g, r, 210)
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


def people_areas(project, core_id: str, stain: str, origin: tuple[int, int] = (0, 0),
                 box: tuple[float, float, float, float] | None = None) -> dict:
    """People's areas of a core image for the Annotations layers (`stainid.exclusions.annotated`; they train and evaluate
    models, the analysis never reads them): each painted or cleared area as polygons (core px less `origin`; only those
    reaching into `box`, x0 y0 x1 y1 core px), how they compare with detection's areas, and the edits (who, when, type)."""
    found = excluded_areas.annotated(project, core_id, stain)
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
            "edits": [{"person": e.source, "time": e.time, "kinds": sorted(set(e.kinds))} for e in found.edits]}
