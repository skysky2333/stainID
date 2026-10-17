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
    """Every candidate of the core image as the viewer's object rows (x, y in core pixels)."""
    frame = candidates.read_core(project, stain, core_id, COLUMNS)
    if frame.empty:
        return pd.DataFrame(columns=["x", "y", "radius_px", "model_class", "probability", "area_um2", "reason", "candidate_id"])
    return pd.DataFrame({"x": frame.x_core.round(1), "y": frame.y_core.round(1), "radius_px": frame.radius_px.round(1),
                         "model_class": model_class(stain, frame), "probability": pd.to_numeric(frame.probability, errors="coerce").round(3),
                         "area_um2": frame.area_um2.round(1), "reason": frame.reason.fillna(""), "candidate_id": frame.candidate_id})


def outline_rows(project, core_id: str, stain: str) -> list[list]:
    """Each candidate's outline: [x, y, area µm², rings] (the stain's display outline, `candidates.outline_rings`)."""
    frame = candidates.read_core(project, stain, core_id, COLUMNS)
    rows = []
    for row in frame.to_dict("records"):
        rings = candidates.outline_rings(row)
        if rings:
            rows.append([round(float(row["x_core"]), 1), round(float(row["y_core"]), 1), round(float(row["area_um2"]), 1), rings])
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
