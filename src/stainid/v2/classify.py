"""Stage 4, calls: score every candidate with the active model from its stored features (no image is read), then apply
the stain's rules, and give each candidate a decision, a reason code, a class and a subtype.

Rules come before the model where the legacy pipeline dropped a candidate before scoring it (`rule:excluded`,
`rule:near_profile`, `rule:pale_nucleus`, `rule:shape`, `rule:low_contrast`, `rule:small`, `rule:artifact_shape`) or where a
candidate only the new sources found looks like an artifact (`rule:edge`, `rule:fold`, `rule:irregular`), and after it
where it re-labelled an accepted one (`rule:speck`, `rule:vascular_or_edge`, `rule:duplicate`); each can be switched off
or retuned under `v2.rules` in stainid.yaml. Re-running this stage after a new model, cut-off or rule takes minutes.

Also here: assembling the candidate store (proposals + outlines + calls) of a core image.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load

from stainid import candidates as store
from stainid.v2 import manifest
from stainid.v2.params import stage_params

MODEL_KEY = {"NeuN": "neun", "6E10": "amyloid", "AT8": "tau"}
POSITIVE = {"NeuN": "NeuN+ neuron", "6E10": "plaque", "AT8": "tau+ neuron"}
NEGATIVE = {"NeuN": "not a neuron", "6E10": "not a plaque", "AT8": "not tau"}
LEGACY_SOURCES = {"NeuN": ("profile", "cellpose"), "6E10": ("legacy",), "AT8": ("ring", "dense")}  # what the old pipeline proposed
CALL_COLUMNS = ["candidate_id", "probability", "threshold", "model_id", "decision", "reason", "class", "subtype"]


@dataclass
class Model:
    stain: str
    bundle: dict
    path: Path
    model_id: str
    threshold: float
    names: list[str]

    def probability(self, frame: pd.DataFrame) -> np.ndarray:
        if frame.empty:
            return np.zeros(0)
        matrix = store.model_matrix(frame, self.names)
        if self.stain == "NeuN":
            usable = np.asarray(self.bundle.get("usable_features", np.ones(len(self.names), bool)))
            return self.bundle["classifier"].predict_proba(matrix[:, usable])[:, 1]
        if self.stain == "6E10":
            return self.bundle["identity_classifier"].predict_proba(matrix)[:, 1]
        return self.bundle["classifier"].predict_proba(matrix)[:, 1]


def load_model(project, stain: str, jobs: int = 2) -> Model:
    """The project's active model for a stain and its cut-off (the project's acceptance setting, else the model's own);
    the bundle is read once per process until its file changes."""
    from stainid import cache

    path = project.model(MODEL_KEY[stain])
    bundle = cache.memo(("v2 bundle", str(path)), [path], lambda: load(path))
    for key in ("classifier", "identity_classifier"):
        estimator = bundle.get(key)
        if estimator is not None and hasattr(estimator, "steps"):
            estimator.steps[-1][1].set_params(n_jobs=jobs)
    if stain == "NeuN":
        threshold, names = project.acceptance("neun", bundle), list(bundle["feature_names"])
    elif stain == "AT8":
        threshold, names = project.acceptance("tau", bundle), list(bundle["feature_names"])
    else:
        threshold, names = float(bundle["identity_threshold"]), list(bundle["identity_feature_names"])
    return Model(stain, bundle, path, f"{path.parent.name}@{manifest.sha256(path)[:8]}", float(threshold), names)


def _column(frame: pd.DataFrame, name: str, default=np.nan) -> pd.Series:
    return frame[name] if name in frame else pd.Series(default, index=frame.index)


def _number(frame: pd.DataFrame, name: str) -> np.ndarray:
    return pd.to_numeric(_column(frame, name), errors="coerce").to_numpy(float)


def _flag(frame: pd.DataFrame, name: str) -> np.ndarray:
    values = _column(frame, name, "")
    return values.astype(str).str.lower().isin(["true", "1", "1.0"]).to_numpy()


def pre_rules(stain: str, frame: pd.DataFrame, rules: dict) -> np.ndarray:
    """The reason a rule rejects each candidate before the model ("" = none): the first rule that applies."""
    reason = np.full(len(frame), "", dtype=object)

    def apply(mask, name):
        mask = np.asarray(mask, bool) & (reason == "")
        reason[mask] = f"rule:{name}"

    drop = _column(frame, "flag_legacy_drop", "").fillna("").astype(str).to_numpy()
    new_source = ~_column(frame, "source", "").astype(str).isin(LEGACY_SOURCES[stain]).to_numpy()
    rim = new_source & (_number(frame, "edge_band_fraction") >= float(rules.get("edge_band", np.inf)))
    excluded = _number(frame, "excluded_fraction") >= float(rules.get("excluded_fraction", 0.5))
    if stain == "NeuN":
        apply(excluded | (drop == "excluded"), "excluded")
        if rules.get("near_profile", True):
            apply(drop == "near_profile", "near_profile")
        if rules.get("pale_nucleus", True):
            apply(drop == "pale_nucleus", "pale_nucleus")
        if rules.get("profile_shape", True):
            profile = _number(frame, "m_source_is_dab_profile") == 1
            area = _number(frame, "m_profile_area_um2")
            shape = (_number(frame, "m_stained_area_um2") < 4.0) | (area < 15.0) | (area > 650.0) | (_number(frame, "m_aspect_ratio") > 3.5)
            apply(profile & shape, "shape")
        apply(rim, "edge")
        # the new sources also find patches of dark fold bands, which the legacy NeuN detection never masked; a neuron on a
        # band keeps a sharp edge (DAB gradient on its boundary), a patch of the band does not
        on_fold = _number(frame, "fold_fraction") >= float(rules.get("fold_fraction", np.inf))
        apply(new_source & on_fold & (_number(frame, "edge_strength") < float(rules.get("fold_min_edge", 0.0))), "fold")
        apply(new_source & (_number(frame, "m_solidity") < float(rules.get("body_min_solidity", 0.0))), "irregular")
    elif stain == "6E10":
        apply(excluded, "excluded")
        if rules.get("artifact_shape", False):
            apply(_flag(frame, "flag_artifact_shape"), "artifact_shape")
    else:
        apply(excluded, "excluded")
        if rules.get("centroid_valid", True) and "centroid_valid" in frame:
            apply(~_flag(frame, "centroid_valid"), "excluded")
        apply(_number(frame, "m_normalized_contrast") < float(rules.get("min_contrast", 0.3)), "low_contrast")
        body = _column(frame, "legacy_class", "").astype(str).to_numpy() == "dab_body"
        apply(body & (_number(frame, "m_soma_area_um2") < float(rules.get("min_body_um2", 40.0))), "small")
        apply(rim, "edge")
        irregular = (_number(frame, "m_soma_solidity") < float(rules.get("body_min_solidity", 0.0))) | \
            (_number(frame, "m_soma_circularity") < float(rules.get("body_min_circularity", 0.0)))
        apply(new_source & irregular, "irregular")
        if rules.get("thread_like"):
            apply(_number(frame, "thread_fraction") >= 0.5, "thread_like")
    return reason


def suppress_near(frame: pd.DataFrame, probability: np.ndarray, candidates: np.ndarray, radius_px: float) -> np.ndarray:
    """Non-maximum suppression: of candidates closer than `radius_px`, only the most probable stays (True = suppressed)."""
    from scipy.spatial import cKDTree

    suppressed = np.zeros(len(frame), bool)
    index = np.flatnonzero(candidates)
    if not len(index):
        return suppressed
    xy = frame[["x_core", "y_core"]].to_numpy(float)[index]
    tree = cKDTree(xy)
    kept = np.zeros(len(index), bool)  # a lookup, not a list: thousands of passed AT8 candidates made the scan quadratic
    for position in np.argsort(-probability[index], kind="stable"):
        near = tree.query_ball_point(xy[position], radius_px)
        if any(kept[k] for k in near if k != position):
            suppressed[index[position]] = True
        else:
            kept[position] = True
    return suppressed


def call(project, stain: str, frame: pd.DataFrame, model: Model, rules: dict) -> pd.DataFrame:
    """The calls of one core image's candidates."""
    probability = model.probability(frame)
    reason = pre_rules(stain, frame, rules)
    scored = reason == ""
    passed = scored & (probability >= model.threshold)
    klass = np.full(len(frame), NEGATIVE[stain], dtype=object)
    subtype = np.full(len(frame), "", dtype=object)
    if stain == "AT8":
        radius = float(model.bundle.get("nms_radius_um", 12.0)) / project.pixel_size_um
        duplicate = suppress_near(frame, probability, passed, radius)
        reason[duplicate] = "rule:duplicate"
        passed &= ~duplicate
        feature = str(model.bundle.get("mature_feature", "soma_dab_p90"))  # s_soma_dab_p90 for a slide-scaled model
        mature = _number(frame, feature if feature.startswith("s_") else f"m_{feature}") >= float(model.bundle.get("mature_p90_threshold", np.inf))
        subtype[passed] = np.where(mature[passed], "mature tangle", "pretangle")
        threadish = _number(frame, "thread_fraction") >= 0.5
        klass[~passed & threadish] = "thread"
    elif stain == "6E10":
        diameter = _number(frame, "m_equivalent_diameter_um")
        speck = passed & (diameter <= float(rules.get("speck_diameter_um", 10.0)))
        reason[speck] = "rule:speck"
        vascular = passed & ~speck & (_number(frame, "lumen_contact_fraction") >= float(rules.get("lumen_contact", 0.25)))
        reason[vascular] = "rule:vascular_or_edge"
        klass[vascular] = "CAA"
        passed &= ~speck & ~vascular
        normalized = _number(frame, "normalized_inner_dab")
        small = diameter < float(model.bundle.get("morphotype_minimum_diameter_um", 15.0))
        compact = normalized >= float(model.bundle.get("morphotype_threshold", np.inf))
        subtype[passed] = np.where(small[passed], "small", np.where(compact[passed], "compact", "diffuse"))
    klass[passed] = POSITIVE[stain]
    reason[passed] = "accepted"
    reason[(reason == "") & scored] = "rejected_by_model"
    return pd.DataFrame({
        "candidate_id": frame.candidate_id.to_numpy(), "probability": probability, "threshold": model.threshold,
        "model_id": model.model_id, "decision": np.where(passed, "accepted", "rejected"), "reason": reason, "class": klass, "subtype": subtype,
    })


# ---------------------------------------------------------------- per core image: manifests, writing, assembling

def expected(project, stain: str, core_id: str, model: Model, settings: dict | None = None, upstream=manifest.READ) -> dict | None:
    """What stage 4's manifest of this core would be now (`settings`, `upstream`: the stain's `stage_params` and the
    core's proposals manifest, when known already)."""
    upstream = manifest.read(project, "proposals", stain, core_id) if upstream is manifest.READ else upstream
    if upstream is None:
        return None
    return manifest.make("calls", stage_params(project, "calls", stain) if settings is None else settings,
                         {"proposals": upstream["digest"], "model": model.model_id, "threshold": model.threshold})


def classify_core(project, stain: str, core_id: str, model: Model) -> dict:
    frame = store.read_table(store.path(project, stain, core_id, "proposals"))
    calls = call(project, stain, frame, model, stage_params(project, "calls", stain)) if not frame.empty else pd.DataFrame(columns=CALL_COLUMNS)
    store.write_table(store.root(project) / "calls" / stain / f"{core_id}.csv.gz", calls)
    made = expected(project, stain, core_id, model)
    manifest.write(project, "calls", stain, core_id, made, accepted=int((calls.decision == "accepted").sum()), candidates=len(calls))
    assemble(project, stain, core_id)
    return {"candidates": len(calls), "accepted": int((calls.decision == "accepted").sum())}


def assembled_expected(project, stain: str, core_id: str, settings: dict | None = None, recorded: dict | None = None) -> dict | None:
    """What the assembled store's manifest of this core would be now (`settings`: the stain's `stage_params`;
    `recorded`: the core's manifests by stage, when read already)."""
    inputs = {}
    for stage in ("proposals", "outlines", "calls"):
        mine = manifest.read(project, stage, stain, core_id) if recorded is None else recorded.get(stage)
        if mine is None and stage != "outlines":
            return None
        inputs[stage] = mine["digest"] if mine else ""
    return manifest.make("candidates", stage_params(project, "candidates", stain) if settings is None else settings, inputs)


def assemble(project, stain: str, core_id: str) -> Path | None:
    """Write the candidate store of one core image: proposals with their outlines (stage 3, when made) and calls, and
    which outline is the object's own for display (`outline_default`: "stain" or "sam")."""
    made = assembled_expected(project, stain, core_id)
    if made is None:
        return None
    frame = store.read_table(store.path(project, stain, core_id, "proposals"))
    calls = store.read_table(store.path(project, stain, core_id, "calls"))
    outlines = store.read_table(store.path(project, stain, core_id, "outlines"))
    if not outlines.empty:
        frame = frame.merge(outlines, on="candidate_id", how="left")
    for column in store.OUTLINE_COLUMNS:
        if column not in frame:
            frame[column] = ""
    if not calls.empty:
        frame = frame.merge(calls, on="candidate_id", how="left")
    frame["outline_default"] = made["params"]["default_outline"]
    target = store.root(project) / "candidates" / stain / f"{core_id}.csv.gz"
    store.write_table(target, frame)
    manifest.write(project, "candidates", stain, core_id, made)
    return target
