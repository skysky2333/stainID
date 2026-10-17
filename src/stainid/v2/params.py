"""Parameters of the v2 detection stages, with the project's overrides (`v2:` in stainid.yaml, same shape as DEFAULTS).

Each stage records the parameters it ran with (and a hash of them) in its manifests, so a change here marks exactly the
stage outputs it affects as outdated: proposal parameters re-run stage 2, rules and model cut-offs only stage 4.
"""
from __future__ import annotations

import copy
import hashlib
import json

STAINS = ("NeuN", "6E10", "AT8")

DEFAULTS: dict = {
    "proposals": {
        # component tree: smoothed DAB at these multiples of the slide threshold (per stain; the legacy sources run as well)
        "smooth_px": 1.0,
        "local_background_um": 60.0,
        # context_px: how far beyond its tile (core px) each tile's detection window reaches, so the tile owning an
        # object's centre sees all of it: the largest object's reach from its centre (accepted, whole objects of the cohort:
        # NeuN 221 px, AT8 168 px, 6E10 926 px = 253 µm) with a margin
        "NeuN": {"context_px": 256, "levels": [0.5, 0.75, 1.0, 1.5], "local_contrast": 0.5, "min_um2": 15.0, "max_um2": 650.0,
                 "significant_um2": 25.0, "split_max_um2": 6000.0, "mode": "split",
                 # merged components are cut into cells only when they look like touching cells, not a rim or band
                 "split": {"max_edge_fraction": 0.3, "max_aspect": 4.0, "min_radius_um": 3.0},
                 "unproposed": {"min_um2": 15.0}},
        # 6E10: the legacy segmentation finds almost every deposit; the faint levels add deposits below its seed level
        "6E10": {"context_px": 1024, "levels": [0.5, 0.75, 1.0, 1.5], "local_contrast": None, "min_um2": 20.0, "max_um2": 50000.0, "significant_um2": 20.0,
                 "split_max_um2": None, "mode": "roots",
                 # a faint node that merges deposits already kept (or is mostly covered by them) gives way to its
                 # children: on a dark-background core the faint field joins plaques the legacy seed level misses
                 "descend": True,
                 # pale diffuse plaques as one region: DAB smoothed over 4 µm at max(0.75 × threshold, tissue median + 0.25 ×
                 # threshold), ≥ 100 µm² (source `coarse`), offered after the legacy deposits and before the tree
                 "coarse": {"smooth_um": 4.0, "level": 0.75, "contrast": 0.25, "min_um2": 100.0},
                 # stained regions no candidate proposes: DAB smoothed over 4 µm (a pale diffuse plaque is one region, not
                 # specks) at 0.75 × the threshold or the tissue median + 0.25 × the threshold, ≥ 100 µm², that no single
                 # candidate covers a quarter of
                 "unproposed": {"smooth_um": 4.0, "level": 0.75, "contrast": 0.25, "min_um2": 100.0, "min_cover": 0.25}},
        # AT8: bodies off the nucleus rings and the legacy dense bodies (pretangle-like, threads' swellings)
        "AT8": {"context_px": 256, "levels": [1.0, 1.5, 2.0], "local_contrast": 1.0, "min_um2": 20.0, "max_um2": 1500.0, "significant_um2": 15.0,
                "split_max_um2": 4000.0, "mode": "split", "ring_contrast": 0.15,
                "split": {"max_edge_fraction": 0.3, "max_aspect": 4.0, "min_radius_um": 2.5},
                "unproposed": {"min_um2": 20.0}},
        # dedupe: a new geometry is the same object as a kept candidate when they overlap this much (IoU), or one holds
        # this share of the other
        "same_iou": 0.5,
        "same_inside": 0.8,
        "covered": 0.3,  # a new geometry overlapping kept candidates more than this (share of its area) is not added
    },
    "outlines": {
        "select": "probable",  # probable: accepted or probability >= min_probability; accepted; all
        "min_probability": 0.2,
        "runaway_factor": 5.0,  # a mask more than this many times its candidate's area falls back to the candidate outline
        # a mask under this share of its candidate's area (candidates of at least shrink_min_um2) is not a default either:
        # on a pale diffuse plaque Segment Anything often outlines one dark speck of it
        "shrink_factor": 0.25,
        "shrink_min_um2": 100.0,
        # advanced: the outline the viewer and the annotation pages show first for an object ("stain": the candidate's own
        # outline from the stain; "sam": Segment Anything's default output, the stain outline where it fell back). Both are
        # always stored; areas in the measures come from the stain outline either way. 6E10: SAM clips diffuse plaques to
        # their dense part; NeuN: SAM is as good or better; AT8: mixed.
        "default_outline": {"NeuN": "sam", "6E10": "stain", "AT8": "stain"},
    },
    # model training (not a detection stage: changing it re-runs nothing). slide_scaled: a new model takes its DAB
    # intensities in slide thresholds and the measures cut at absolute OD floors at levels relative to the threshold
    # (`s_` inputs, stainid.v2.scaled) instead of absolute OD, so pale slides (LIP-6) are not rejected for being pale
    "training": {"NeuN": {"slide_scaled": True}, "6E10": {"slide_scaled": True}, "AT8": {"slide_scaled": True}},
    "rules": {
        # edge_band: a candidate found only by the new sources lying mostly within 5 µm of a tissue edge or hole (brown rims)
        "NeuN": {"excluded_fraction": 0.5, "near_profile": True, "pale_nucleus": True, "profile_shape": True, "edge_band": 0.5,
                 "fold_fraction": 0.5, "fold_min_edge": 1.0, "body_min_solidity": 0.6},
        "6E10": {"excluded_fraction": 0.5, "artifact_shape": False, "speck_diameter_um": 10.0, "lumen_contact": 0.25},
        "AT8": {"excluded_fraction": 0.5, "min_contrast": 0.3, "min_body_um2": 40.0, "centroid_valid": True,
                "thread_like": False, "edge_band": 0.5,
                # bodies found only by the new sources must be as compact as the bodies the legacy sources propose
                "body_min_solidity": 0.7, "body_min_circularity": 0.35},
    },
}


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        out[key] = _merge(out[key], value) if isinstance(value, dict) and isinstance(out.get(key), dict) else value
    return out


def params(project) -> dict:
    """The v2 parameters of a project: DEFAULTS with the project's `v2:` section merged in."""
    return _merge(DEFAULTS, project.config.get("v2") or {})


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()[:16]


def default_outline(project, stain: str) -> str:
    """Which outline is an object's own for display ("stain" or "sam", `outlines.default_outline`)."""
    value = params(project)["outlines"].get("default_outline", {})
    value = value.get(stain, "sam") if isinstance(value, dict) else value
    if value not in ("stain", "sam"):
        raise ValueError(f"v2.outlines.default_outline for {stain} must be 'stain' or 'sam', not {value!r}")
    return value


def stage_params(project, stage: str, stain: str) -> dict:
    """What one stage's output for one stain depends on (besides its inputs): stage 2 the proposal settings, stage 3 the
    outline settings (without the display choice), stage 4 the stain's rules (the model and its cut-off are recorded as
    inputs), the assembled store the stain's default outline."""
    p = params(project)
    if stage == "proposals":
        shared = {k: v for k, v in p["proposals"].items() if k not in STAINS}
        return {"shared": shared, "stain": p["proposals"][stain], "context_px": project.context_px}
    if stage == "outlines":
        return {k: v for k, v in p["outlines"].items() if k != "default_outline"}
    if stage == "candidates":
        return {"default_outline": default_outline(project, stain)}
    if stage == "calls":
        return dict(p["rules"][stain])
    return {}
