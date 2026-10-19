"""Where a field starts in "Fix the field": the acceptance threshold an annotator sets before editing.

Step 1 of every field is a slider, starting at detection's own threshold (the store's `threshold`, the model's
cut-off), where every candidate starts as detection called it. Moved, it re-decides only the model's part, as stage 4
would at that threshold: a candidate passes where its probability is at least the threshold (a missing score is 0) and
then gets stage 4's call, its rules after the model included (a speck stays rejected, a deposit on a vessel wall or
tissue edge is CAA); below it gets stage 4's call for a candidate the model does not count (rejected, or the look-alike
detection names: an AT8 candidate lying on threads is a thread). A candidate a rule decided (any `rule:` reason) keeps
detection's call at every value, as does one in an excluded area: CAA stays CAA, counted as CAA; the page says which
rule ("Why"). So raising the threshold never counts more, and at 0 every candidate no rule decided (outside excluded
areas) counts. A look-alike a rule names is a class, not a reject (`candidate_store.ai_call`). A person still overrides any call
with a key. The slider locks before the first edit, and the locked value is saved with the field (a `threshold`
label, `labels.store`) as a training and evaluation signal (tuning each stain's cut-off later). Nothing in the
analysis reads it: stage 4 keeps detection's threshold.

`if_accepted` is each candidate's call once it passes the model: stage 4's own call with every scored candidate
passing (`v2.classify.call` at a threshold below every score, with the stored probabilities), so a plaque promoted by a
lower threshold gets detection's type for it (compact, diffuse, small …) and one a later rule decides gets that rule's
call. `if_rejected` is the same below the threshold (`v2.classify.call` above every score). Kept per core image until
its store changes.
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


def _calls_at(project, stain: str, frame: pd.DataFrame, threshold: float) -> pd.DataFrame | None:
    """Stage 4's calls with every scored candidate passing (`threshold` -inf) or failing (+inf), from the stored scores
    (None when the stain's model cannot be loaded)."""
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

    fixed = Stored(model.stain, model.bundle, model.path, model.model_id, threshold, model.names)
    try:
        return classify.call(project, stain, frame, fixed, stage_params(project, "calls", stain))
    except Exception:  # noqa: BLE001 - a store from an older stage 2 without a rule's inputs
        return None


Call = tuple[str, str, str, str]  # (decision, class, subtype, rule) as the store words them


def _row(frame: pd.DataFrame | None, candidate_id: str):
    if frame is None:
        return None
    found = frame.loc[candidate_id]
    return found.iloc[0] if isinstance(found, pd.DataFrame) else found


def _calls(project, stain: str, core_id: str) -> dict:
    """{"passed": `if_accepted`, "failed": `if_rejected`} of a core image, kept until its store changes."""
    path = v2store.stored(project, stain, core_id)
    if path is None:
        return {"passed": {}, "failed": {}}
    key = (str(project.root), stain, str(core_id))
    stamp = cache.signature(path)
    with _LOCK:
        kept = _KEPT.get(key)
        if kept is not None and kept["stamp"] == stamp:
            _KEPT.move_to_end(key)
            return kept["calls"]
    frame = v2store.read_core(project, stain, core_id)
    passed: dict[str, Call | None] = {}
    failed: dict[str, Call | None] = {}
    if not frame.empty:
        from stainid.v2.classify import NEGATIVE, POSITIVE, THREAD  # noqa: PLC0415 (the fallback without a model: stage 4's thread rule)

        found = _calls_at(project, stain, frame, -np.inf)
        would = found.set_index("candidate_id") if found is not None else None
        found = _calls_at(project, stain, frame, np.inf)
        wont = found.set_index("candidate_id") if found is not None else None
        threads = pd.to_numeric(frame.get("thread_fraction", pd.Series(np.nan, index=frame.index)), errors="coerce").to_numpy(float) >= 0.5
        for n, r in enumerate(frame[["candidate_id", "reason", "class", "subtype", "decision"]].itertuples(index=False)):
            reason = str(r.reason or "")
            if reason.startswith("rule:"):
                passed[r.candidate_id] = failed[r.candidate_id] = None
                continue
            if str(r.decision) == "accepted" or reason == "accepted":
                passed[r.candidate_id] = ("accepted", str(r[2]), str(r.subtype or ""), "")
            elif would is None:
                passed[r.candidate_id] = ("accepted", POSITIVE.get(stain, ""), "", "")
            else:
                w = _row(would, r.candidate_id)
                rule = str(w["reason"]) if str(w["reason"]).startswith("rule:") else ""
                passed[r.candidate_id] = (str(w["decision"]), str(w["class"]), str(w["subtype"] or ""), rule)
            if str(r.decision) != "accepted" and reason == "rejected_by_model":
                failed[r.candidate_id] = ("rejected", str(r[2]), "", "")  # detection's own call below its threshold
            elif wont is not None:
                w = _row(wont, r.candidate_id)
                failed[r.candidate_id] = ("rejected", str(w["class"]), "", "")
            else:
                failed[r.candidate_id] = ("rejected", THREAD if stain == "AT8" and threads[n] else NEGATIVE.get(stain, ""), "", "")
    out = {"passed": passed, "failed": failed}
    with _LOCK:
        _KEPT[key] = {"stamp": stamp, "calls": out}
        while len(_KEPT) > 64:
            _KEPT.popitem(last=False)
    return out


def if_accepted(project, stain: str, core_id: str) -> dict[str, Call | None]:
    """Per candidate of a core image: its call once it passes the model, as (decision, class, subtype, rule) the store
    words them: accepted in detection's type, or what a rule after the model makes of it (`rule`: a speck rejected, a
    vascular deposit CAA); None where detection's call stays at every threshold: a rule decided it (any `rule:`
    reason) or it lies in an excluded area."""
    return _calls(project, stain, core_id)["passed"]


def if_rejected(project, stain: str, core_id: str) -> dict[str, Call | None]:
    """Per candidate of a core image: its call once the model does not count it (stage 4's: rejected, or the look-alike
    detection names, an AT8 candidate lying on threads a thread); None where a rule decided it, as `if_accepted`."""
    return _calls(project, stain, core_id)["failed"]


def score(probability) -> float:
    """A candidate's score for the slider: a missing one is 0."""
    try:
        value = float(probability)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if value != value else value


def start_call(probability, if_passed, threshold: float, detection: tuple[str, str], reject: tuple[str, str], if_failed=None) -> tuple[str, str]:
    """Where one candidate starts at this threshold: `if_passed` (its call once it passes the model) where its score
    (`score`: 0 when missing) is at least the threshold, below `if_failed` (the look-alike detection names once the model
    does not count it: an AT8 thread) or else `reject`; detection's own call where a rule decided it or it lies in an
    excluded area (`if_passed` None: CAA stays CAA). The page draws the same rule (webapp/src/labels/fieldStart.ts)."""
    if if_passed is None:
        return detection
    if score(probability) >= float(threshold) - 1e-9:
        return tuple(if_passed)
    return tuple(if_failed) if if_failed else reject  # the page's step: 0.29 may be 0.29000000000000004
