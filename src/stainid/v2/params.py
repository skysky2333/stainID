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
        "NeuN": {"levels": [0.5, 0.75, 1.0, 1.5], "local_contrast": 0.5, "min_um2": 15.0, "max_um2": 650.0,
                 "significant_um2": 25.0, "split_max_um2": 6000.0, "mode": "split",
                 # merged components are cut into cells only when they look like touching cells, not a rim or band
                 "split": {"max_edge_fraction": 0.3, "max_aspect": 4.0, "min_radius_um": 3.0}},
        # 6E10: the legacy segmentation finds almost every deposit; the faint levels add deposits below its seed level
        "6E10": {"levels": [0.5, 0.75], "local_contrast": None, "min_um2": 20.0, "max_um2": 50000.0, "significant_um2": 20.0,
                 "split_max_um2": None, "mode": "roots"},
        # AT8: bodies off the nucleus rings and the legacy dense bodies (pretangle-like, threads' swellings)
        "AT8": {"levels": [1.0, 1.5, 2.0], "local_contrast": 1.0, "min_um2": 20.0, "max_um2": 1500.0, "significant_um2": 15.0,
                "split_max_um2": 4000.0, "mode": "split", "ring_contrast": 0.15,
                "split": {"max_edge_fraction": 0.3, "max_aspect": 4.0, "min_radius_um": 2.5}},
        # dedupe: a new geometry is the same object as a kept candidate when they overlap this much (IoU), or one holds
        # this share of the other
        "same_iou": 0.5,
        "same_inside": 0.8,
        "covered": 0.3,  # a new geometry overlapping kept candidates more than this (share of its area) is not added
        "unproposed_min_um2": {"NeuN": 15.0, "6E10": 20.0, "AT8": 20.0},
    },
    "outlines": {
        "select": "probable",  # probable: accepted or probability >= min_probability; accepted; all
        "min_probability": 0.2,
        "runaway_factor": 5.0,  # a mask more than this many times its candidate's area falls back to the candidate outline
    },
    "rules": {
        # edge_band: a candidate found only by the new sources lying mostly within 5 µm of a tissue edge or hole (brown rims)
        "NeuN": {"excluded_fraction": 0.5, "near_profile": True, "pale_nucleus": True, "profile_shape": True, "edge_band": 0.5},
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


def stage_params(project, stage: str, stain: str) -> dict:
    """What one stage's output for one stain depends on (besides its inputs): stage 2 the proposal settings, stage 3 the
    outline settings, stage 4 the stain's rules (the model and its cut-off are recorded as inputs)."""
    p = params(project)
    if stage == "proposals":
        shared = {k: v for k, v in p["proposals"].items() if k not in STAINS}
        return {"shared": shared, "stain": p["proposals"][stain], "context_px": project.context_px}
    if stage == "outlines":
        return dict(p["outlines"])
    if stage == "calls":
        return dict(p["rules"][stain])
    return {}
