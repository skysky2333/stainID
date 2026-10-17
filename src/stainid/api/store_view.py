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

COLUMNS = ["candidate_id", "x_core", "y_core", "radius_px", "area_um2", "probability", "decision", "reason", "class", "subtype",
           "m_from_ring", "outline", "sam_outline", "sam_status"]


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
    """One kind of area left out of the analysis, as the "Excluded areas" layer draws it: which pixels of the stage-2
    mask (`stainid.v2.proposals.BITS`, one byte per 8 × 8 core-image px) are of this kind (`pick`), its colour and its
    hatching ("\\\\", "/", "|", "-", or "" for none)."""

    def __init__(self, id: str, label: str, color: tuple[int, int, int], hatch: str, pick):
        self.id, self.label, self.color, self.hatch, self.pick = id, label, color, hatch, pick


TISSUE, LINEAR, FOLD, MANUAL, EDGE_GUARD, TILE = 1, 2, 4, 8, 16, 32  # stainid.v2.proposals.BITS
ANNOTATOR = 64  # not in the stage-2 mask: the areas annotators marked as bad on the field page, painted on here (`_annotated`)
# Drawn in this order, each pixel as the first kind it is: what a person drew comes first. A further kind (e.g. the areas
# annotators mark on the field page) is one more entry here; `pick` gets the mask and returns that kind's pixels.
EXCLUDED_KINDS = (
    ExcludedKind("annotators", "marked by annotators (the field tool's brush)", (255, 77, 210), "/", lambda m: (m & ANNOTATOR) > 0),
    ExcludedKind("manual", "drawn by hand (manual exclusion)", (255, 59, 59), "\\", lambda m: (m & MANUAL) > 0),
    ExcludedKind("fold", "tissue fold", (255, 176, 0), "/", lambda m: (m & FOLD) > 0),
    ExcludedKind("linear", "linear artifact (scratch, tear, edge line)", (168, 85, 247), "|", lambda m: (m & LINEAR) > 0),
    ExcludedKind("edge_guard", "AT8 edge guard (brown rim along an edge)", (20, 184, 166), "-", lambda m: (m & EDGE_GUARD) > 0),
    ExcludedKind("no_tissue", "no tissue (empty glass, holes)", (156, 163, 175), "", lambda m: ((m & TILE) > 0) & ((m & TISSUE) == 0)),
)
EXCLUDED_SCALE = 8  # stainid.v2.proposals.MASK_SCALE: core-image px per mask px
HATCH_PX = 4  # mask px between hatching lines (32 core-image px)


def excluded_path(project, core_id: str, stain: str):
    return candidates.root(project) / "masks" / stain / f"{core_id}.png"


def _annotated(project, core_id: str, stain: str, shape: tuple[int, int]) -> np.ndarray:
    """The areas annotators marked as bad on this core image's fields (`labels.exclusions`), on the mask's grid."""
    import cv2  # noqa: PLC0415

    from stainid.api.state import get_state  # noqa: PLC0415
    from stainid.labels import exclusions, store  # noqa: PLC0415

    out = np.zeros(shape, np.uint8)
    for area in exclusions.in_core(store.current(project.output("reviews")), get_state().tiles, project.context_px, core_id, stain):
        for ring, hole in zip(area["rings"], area["holes"]):
            cv2.fillPoly(out, [np.round(np.asarray(ring) / EXCLUDED_SCALE).astype(np.int32)], 0 if hole else 1)
    return out > 0


def _deps(project, path) -> list:
    """What the layer of a core image is made from: its stage-2 mask and the label log (the annotators' areas)."""
    from stainid.labels import store  # noqa: PLC0415

    return [path, store.log_path(project.output("reviews"))]


def _excluded(path, project=None, core_id: str = "", stain: str = "") -> dict:
    """The excluded areas of one mask (with the annotators' areas when `project` is given): each kind's pixels (a pixel
    counts as the first kind it is) and the analysed area."""
    import cv2  # noqa: PLC0415

    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(path)
    mask = mask if mask.ndim == 2 else mask[..., 0]
    if project is not None:
        mask = mask.astype(np.uint8) | np.where(_annotated(project, core_id, stain, mask.shape), ANNOTATOR, 0).astype(np.uint8)
    left = np.ones(mask.shape, bool)
    kinds = []
    for kind in EXCLUDED_KINDS:
        pixels = kind.pick(mask) & left
        left &= ~pixels
        kinds.append((kind, pixels))
    return {"shape": mask.shape, "kinds": kinds, "analysed": int(((mask & TILE) > 0).sum())}


def kind_info(kind: ExcludedKind, fraction: float | None) -> dict:
    return {"id": kind.id, "label": kind.label, "color": "#%02x%02x%02x" % kind.color, "hatch": kind.hatch, "fraction": fraction}


def excluded_info(project, core_id: str, stain: str) -> dict:
    """The "Excluded areas" layer of one core image: status ("done", or "not_made" before the v2 proposals), the mask's
    size in mask px and its scale (core-image px per mask px), and each kind with its colour, hatching and share of the
    analysed area."""
    from stainid import cache  # noqa: PLC0415

    path = excluded_path(project, core_id, stain)
    if not path.exists():
        return {"status": "not_made", "kinds": [kind_info(k, None) for k in EXCLUDED_KINDS]}
    found = cache.memo(("excluded", str(path)), _deps(project, path), lambda: _excluded(path, project, core_id, stain))
    height, width = found["shape"]
    analysed = max(1, found["analysed"])
    return {"status": "done", "scale": EXCLUDED_SCALE, "width": int(width), "height": int(height), "version": sum(p.stat().st_mtime_ns for p in _deps(project, path) if p.exists()),  # changes with the mask and the labels
            "kinds": [kind_info(kind, round(float(pixels.sum()) / analysed, 4)) for kind, pixels in found["kinds"]]}


def excluded_png(project, core_id: str, stain: str) -> bytes | None:
    """The layer as an RGBA PNG at the mask's resolution: each kind a translucent fill in its colour with stronger
    hatching lines (no lines for "no tissue"), the rest transparent; kept until the mask changes. None without a mask."""
    import cv2  # noqa: PLC0415

    from stainid import cache  # noqa: PLC0415

    path = excluded_path(project, core_id, stain)
    if not path.exists():
        return None

    def make() -> bytes:
        found = cache.memo(("excluded", str(path)), _deps(project, path), lambda: _excluded(path, project, core_id, stain))
        height, width = found["shape"]
        yy, xx = np.mgrid[0:height, 0:width]
        lines = {"\\": (xx - yy) % HATCH_PX == 0, "/": (xx + yy) % HATCH_PX == 0, "|": xx % HATCH_PX == 0, "-": yy % HATCH_PX == 0}
        out = np.zeros((height, width, 4), np.uint8)  # cv2 writes BGRA
        for kind, pixels in found["kinds"]:
            r, g, b = kind.color
            out[pixels] = (b, g, r, 70)
            if kind.hatch:
                out[pixels & lines[kind.hatch]] = (b, g, r, 210)
        ok, encoded = cv2.imencode(".png", out)
        if not ok:
            raise RuntimeError(f"could not encode the excluded areas of {path}")
        return encoded.tobytes()

    return cache.memo(("excluded png", str(path)), _deps(project, path), make)
