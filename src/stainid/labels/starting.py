"""Where a field starts in "Fix the field": the acceptance threshold an annotator sets before editing.

Step 1 of every field is a slider, starting at detection's own threshold (the store's `threshold`, the model's
cut-off), where every candidate starts as detection called it. Moved, it re-decides only the model's part, as stage 4
would at that threshold: a candidate passes where its probability is at least the threshold (a missing score is 0) and
then gets stage 4's call, its rules after the model included (a speck stays rejected, a deposit on a vessel wall or
tissue edge is CAA); below it is rejected. A candidate a rule decided (any `rule:` reason) keeps detection's call at
every value, as does one in an excluded area; the page says which rule ("Why"). So raising the threshold never counts
more, and at 0 every candidate no rule decided (outside excluded areas) counts. A person still overrides any call
with a key. The slider locks before the first edit, and the locked value is saved with the field (a `threshold`
label, `labels.store`) as a training and evaluation signal (tuning each stain's cut-off later). Nothing in the
analysis reads it: stage 4 keeps detection's threshold.

`if_accepted` is each candidate's call once it passes the model: stage 4's own call with every scored candidate
passing (`v2.classify.call` at a threshold below every score, with the stored probabilities), so a plaque promoted by a
lower threshold gets detection's type for it (compact, diffuse, small …) and one a later rule decides gets that rule's
call. Kept per core image until its store changes.
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
    stored = pd.to_numeric(frame.get("probability", pd.Series(np.nan, index=frame.index)), errors="coerce").fillna(0.0).to_numpy(float)

    @dataclass
    class Stored(classify.Model):
        def probability(self, rows: pd.DataFrame) -> np.ndarray:  # the scores detection gave, not scored again (none: 0)
            return stored

    fixed = Stored(model.stain, model.bundle, model.path, model.model_id, -np.inf, model.names)
    try:
        return classify.call(project, stain, frame, fixed, stage_params(project, "calls", stain))
    except Exception:  # noqa: BLE001 - a store from an older stage 2 without a rule's inputs
        return None


def if_accepted(project, stain: str, core_id: str) -> dict[str, tuple[str, str, str, str] | None]:
    """Per candidate of a core image: its call once it passes the model, as (decision, class, subtype, rule) the store
    words them: accepted in detection's type, or what a rule after the model makes of it (`rule`: a speck rejected, a
    vascular deposit CAA); None where detection's call stays at every threshold: a rule decided it (any `rule:`
    reason) or it lies in an excluded area."""
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
    out: dict[str, tuple[str, str, str, str] | None] = {}
    if not frame.empty:
        from stainid.v2.classify import POSITIVE  # noqa: PLC0415

        passing = _would_pass(project, stain, frame)
        would = passing.set_index("candidate_id") if passing is not None else None
        for r in frame[["candidate_id", "reason", "class", "subtype", "decision"]].itertuples(index=False):
            reason = str(r.reason or "")
            if reason.startswith("rule:"):
                out[r.candidate_id] = None
            elif str(r.decision) == "accepted" or reason == "accepted":
                out[r.candidate_id] = ("accepted", str(r[2]), str(r.subtype or ""), "")
            elif would is None:
                out[r.candidate_id] = ("accepted", POSITIVE.get(stain, ""), "", "")
            else:
                w = would.loc[r.candidate_id]
                w = w.iloc[0] if isinstance(w, pd.DataFrame) else w
                rule = str(w["reason"]) if str(w["reason"]).startswith("rule:") else ""
                out[r.candidate_id] = (str(w["decision"]), str(w["class"]), str(w["subtype"] or ""), rule)
    with _LOCK:
        _KEPT[key] = {"stamp": stamp, "calls": out}
        while len(_KEPT) > 64:
            _KEPT.popitem(last=False)
    return out


def score(probability) -> float:
    """A candidate's score for the slider: a missing one is 0."""
    try:
        value = float(probability)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if value != value else value


def start_call(probability, if_passed, threshold: float, detection: tuple[str, str], reject: tuple[str, str]) -> tuple[str, str]:
    """Where one candidate starts at this threshold: `if_passed` (its call once it passes the model) where its score
    (`score`: 0 when missing) is at least the threshold, `reject` below; detection's own call where a rule decided it
    or it lies in an excluded area (`if_passed` None). The page draws the same rule (webapp/src/labels/fieldStart.ts)."""
    if if_passed is None:
        return detection
    return tuple(if_passed) if score(probability) >= float(threshold) - 1e-9 else reject  # the page's step: 0.29 may be 0.29000000000000004
