from __future__ import annotations

import time
from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter

from stainid.api.state import get_state

router = APIRouter(tags=["models"])
ROLES = {
    "neun": "NeuN neuron identity (context random forest)",
    "amyloid": "6E10 plaque identity (RF + Phikon + Wong 2022 ensemble) and morphotype rule",
    "tau": "AT8 tau+ neuron identity (context random forest)",
    "cellpose": "Cellpose-SAM (cpsam) cell / nucleus segmentation",
    "sam": "Segment Anything ViT-B (object outlines)",
    "plaque_cnn_dir": "Published amyloid CNNs (Plaquebox 2019, Wong 2022 consensus)",
    "huggingface_home": "Foundation-model cache (Phikon)",
}


@lru_cache(maxsize=16)
def describe_bundle(path: str, mtime: float) -> dict:
    from joblib import load

    bundle = load(path)
    if not isinstance(bundle, dict):
        return {"type": type(bundle).__name__}
    info = {"keys": sorted(bundle)}
    for key in ("feature_names", "identity_feature_names", "usable_features"):
        if key in bundle:
            info["n_features"] = int(sum(bundle[key]) if key == "usable_features" else len(bundle[key]))
    for key in ("probability_threshold", "context_threshold", "identity_threshold", "morphotype_threshold", "morphotype_minimum_diameter_um",
                "nms_radius_um", "training_n", "training_positive_n", "morphotype_training_n"):
        if key in bundle:
            info[key] = float(bundle[key]) if not isinstance(bundle[key], (list, tuple)) else bundle[key]
    for key in ("training_candidate_ids", "identity_training_candidate_ids"):
        if key in bundle:
            info["training_labels"] = len(bundle[key])
    return info


def _size(path: Path) -> int:
    return path.stat().st_size if path.is_file() else sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


@router.get("/models")
def models() -> list[dict]:
    project = get_state().project
    out = []
    for key, role in ROLES.items():
        path = project.model(key)
        entry = {"key": key, "role": role, "path": str(path.relative_to(project.root)) if path.is_relative_to(project.root) else str(path), "exists": path.exists()}
        if path.exists():
            entry |= {"size_mb": round(_size(path) / 1e6, 1), "modified": time.strftime("%Y-%m-%d", time.localtime(path.stat().st_mtime))}
            if path.suffix == ".joblib":
                entry["bundle"] = describe_bundle(str(path), path.stat().st_mtime)
        out.append(entry)
    return out
