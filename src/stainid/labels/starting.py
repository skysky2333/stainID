"""Where a field starts in "Fix the field": the acceptance threshold an annotator sets before editing.

Step 1 of every field is a slider, starting at detection's own threshold (the store's `threshold`, the model's
cut-off). It only changes the starting point: candidates the model scored (not dropped by a rule) start accepted, in the
type detection would give them, where their probability is at least the chosen threshold, and rejected below it; the
others keep detection's call. The slider locks before the first edit, and the locked value is saved with the field
(a `threshold` label, `labels.store`) as a training and evaluation signal (tuning each stain's cut-off later). Nothing in
the analysis reads it: stage 4 keeps detection's threshold.

`if_accepted` is what each candidate would be if it passed: stage 4's own rules run with every scored candidate passing
(`v2.classify.call` at a threshold below every score, with the stored probabilities), so a plaque promoted by a lower
threshold gets detection's type for it (compact, diffuse, small …) and one a later rule drops (a speck, a vessel) does
not start accepted. Kept per core image until its store changes.
"""
from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
import pandas as pd

from stainid import cache
from stainid import candidates as v2store

TASK = "threshold"  # the label of a field's locked threshold: id "threshold:<tile_id>", value the number
SCORED = ("accepted", "rejected_by_model")  # the reasons of candidates the model decided (a rule decided the others)
_KEPT: OrderedDict[tuple, dict] = OrderedDict()
_LOCK = threading.Lock()


def label_id(tile_id: str) -> str:
    return f"{TASK}:{tile_id}"


def _would_pass(project, stain: str, frame: pd.DataFrame) -> pd.DataFrame | None:
    """Stage 4's calls with every scored candidate passing (None when the stain's model cannot be loaded)."""
    from stainid.v2 import classify  # noqa: PLC0415
    from stainid.v2.params import stage_params  # noqa: PLC0415

    try:
        model = classify.load_model(project, stain, jobs=1)
    except Exception:  # noqa: BLE001 - no model here (a test project, a moved file): the fallback type
        return None
    stored = pd.to_numeric(frame.get("probability", pd.Series(np.nan, index=frame.index)), errors="coerce").to_numpy(float)

    @dataclass
    class Stored(classify.Model):
        def probability(self, rows: pd.DataFrame) -> np.ndarray:  # the scores detection gave, not scored again
            return stored

    fixed = Stored(model.stain, model.bundle, model.path, model.model_id, -np.inf, model.names)
    try:
        return classify.call(project, stain, frame, fixed, stage_params(project, "calls", stain))
    except Exception:  # noqa: BLE001 - a store from an older stage 2 without a rule's inputs
        return None


def if_accepted(project, stain: str, core_id: str) -> dict[str, tuple[str, str] | None]:
    """Per candidate of a core image: its call (class, subtype as the store words them) if the threshold let it pass,
    None when a rule decides it whatever the threshold. Candidates detection accepted keep their own call."""
    path = v2store.stored(project, stain, core_id)
    if path is None:
        return {}
    key = (str(project.root), stain, str(core_id))
    stamp = cache.signature(path)
    with _LOCK:
        kept = _KEPT.get(key)
        if kept is not None and kept["stamp"] == stamp:
            _KEPT.move_to_end(key)
            return kept["calls"]
    frame = v2store.read_core(project, stain, core_id)
    out: dict[str, tuple[str, str] | None] = {}
    if not frame.empty:
        passing = _would_pass(project, stain, frame)
        would = passing.set_index("candidate_id") if passing is not None else None
        from stainid.v2.classify import POSITIVE  # noqa: PLC0415

        for r in frame[["candidate_id", "reason", "class", "subtype"]].itertuples(index=False):
            if r.reason == "accepted":
                out[r.candidate_id] = (str(r[2]), str(r.subtype or ""))
            elif r.reason != "rejected_by_model":
                out[r.candidate_id] = None
            elif would is None:
                out[r.candidate_id] = (POSITIVE.get(stain, ""), "")
            else:
                w = would.loc[r.candidate_id]
                w = w.iloc[0] if isinstance(w, pd.DataFrame) else w
                out[r.candidate_id] = (str(w["class"]), str(w["subtype"] or "")) if w["decision"] == "accepted" else None
    with _LOCK:
        _KEPT[key] = {"stamp": stamp, "calls": out}
        while len(_KEPT) > 64:
            _KEPT.popitem(last=False)
    return out


def start_call(probability, if_passed, threshold: float, detection: tuple[str, str], reject: tuple[str, str]) -> tuple[str, str]:
    """Where one candidate starts at this threshold: `if_passed` (its call if accepted) at or above it, `reject` below;
    detection's own call when a rule decides it (`if_passed` None) or it has no score. The page draws the same rule
    (webapp/src/labels/fieldStart.ts)."""
    if if_passed is None or probability is None or probability != probability:
        return detection
    return tuple(if_passed) if float(probability) >= float(threshold) else reject
