"""Where a field starts in "Fix the field": the acceptance threshold an annotator sets before editing.

Step 1 of every field is a slider, starting at detection's own threshold (the store's `threshold`, the model's
cut-off), where every candidate starts as detection called it. Moved, it decides by score alone: a candidate whose
probability is at least the chosen threshold starts accepted (counted, in the type detection would give it), one below
it rejected; a candidate with no score counts as 0. So at 0 every candidate counts and at 1 only those scoring 1, except
candidates in an excluded area, which keep detection's call (drawn faint, not asked about). Detection's rules (a speck
too small for a plaque, a deposit on a vessel wall called CAA …) decide its own calls, not the slider's; the page says
which rule left a counted candidate out of detection's count. The slider locks before the first edit, and the locked
value is saved with the field (a `threshold` label, `labels.store`) as a training and evaluation signal (tuning each
stain's cut-off later). Nothing in the analysis reads it: stage 4 keeps detection's threshold.

`if_accepted` is each candidate's call when it counts: detection's own for those it accepted, else the type detection's
type rule gives it (`v2.classify.types_if_passed`: compact, diffuse, small …). Kept per core image until its store
changes.
"""
from __future__ import annotations

import threading
from collections import OrderedDict

import numpy as np
import pandas as pd

from stainid import cache
from stainid import candidates as v2store

TASK = "threshold"  # the label of a field's locked threshold: id "threshold:<tile_id>", value the number
EXCLUDED = ("rule:excluded",)  # in an area detection left out of the analysis: the slider does not move them
_KEPT: OrderedDict[tuple, dict] = OrderedDict()
_LOCK = threading.Lock()


def label_id(tile_id: str) -> str:
    return f"{TASK}:{tile_id}"


def _types(project, stain: str, frame: pd.DataFrame) -> np.ndarray | None:
    """Detection's type for each candidate if accepted (None when the stain's model cannot be loaded)."""
    from stainid.v2 import classify  # noqa: PLC0415

    try:
        model = classify.load_model(project, stain, jobs=1)
        return classify.types_if_passed(project, stain, frame, model)
    except Exception:  # noqa: BLE001 - no model here (a test project, a moved file), or an older store: no type
        return None


def if_accepted(project, stain: str, core_id: str) -> dict[str, tuple[str, str, str, str] | None]:
    """Per candidate of a core image: its call when the slider counts it, as (decision, class, subtype, rule) the store
    words them (decision "accepted"; `rule` the rule that kept it out of detection's own count, else ""); None for a
    candidate in an excluded area, which keeps detection's call whatever the threshold."""
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

        types = _types(project, stain, frame)
        for i, r in enumerate(frame[["candidate_id", "reason", "class", "subtype", "decision"]].itertuples(index=False)):
            reason = str(r.reason or "")
            if reason in EXCLUDED:
                out[r.candidate_id] = None
            elif str(r.decision) == "accepted":
                out[r.candidate_id] = ("accepted", str(r[2]), str(r.subtype or ""), "")
            else:
                out[r.candidate_id] = ("accepted", POSITIVE.get(stain, ""), str(types[i]) if types is not None else "",
                                       reason if reason.startswith("rule:") else "")
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
    """Where one candidate starts at this threshold: `if_passed` (its call when counted) where its score (`score`: 0
    when missing) is at least the threshold, `reject` below; detection's own call in an excluded area (`if_passed`
    None). The page draws the same rule (webapp/src/labels/fieldStart.ts)."""
    if if_passed is None:
        return detection
    return tuple(if_passed) if score(probability) >= float(threshold) - 1e-9 else reject  # the page's step: 0.29 may be 0.29000000000000004
