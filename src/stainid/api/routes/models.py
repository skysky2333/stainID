from __future__ import annotations

import time
from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from stainid.api.state import get_state, open_project
from stainid.training.train import activate, trained_models

router = APIRouter(tags=["models"])
ROLES = {
    "neun": "Decides which candidate objects on NeuN slides are neurons (random forest on shape, stain and 55 µm neighbourhood).",
    "amyloid": "Decides which 6E10 deposits are plaques (random forest + Phikon + Wong 2022 CNN) and sorts them into compact and diffuse.",
    "tau": "Decides which AT8 candidates are tau+ neurons (random forest on shape, stain and neighbourhood).",
    "cellpose": "Cellpose-SAM: finds cells and nuclei. Published weights, the same for every study.",
    "sam": "Segment Anything (ViT-B): draws precise object outlines. Published weights.",
    "plaque_cnn_dir": "Published amyloid CNNs (Plaquebox 2019, Wong 2022 consensus); optional member of the 6E10 model.",
    "huggingface_home": "Phikon pathology foundation model (owkin/phikon); used by the 6E10 model.",
}
SOURCE = {"neun": "train", "amyloid": "train", "tau": "train", "cellpose": "download", "sam": "download", "huggingface_home": "download",
          "plaque_cnn_dir": "manual"}
STAIN = {"neun": "NeuN", "amyloid": "6E10", "tau": "AT8"}


class ActivateRequest(BaseModel):
    path: str


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
        entry = {"key": key, "role": role, "path": project.relative(path), "exists": path.exists(), "source": SOURCE[key], "stain": STAIN.get(key)}
        if path.exists():
            entry |= {"size_mb": round(_size(path) / 1e6, 1), "modified": time.strftime("%Y-%m-%d", time.localtime(path.stat().st_mtime))}
            if path.suffix == ".joblib":
                entry["bundle"] = describe_bundle(str(path), path.stat().st_mtime)
        out.append(entry)
    return out


@router.get("/models/trained")
def trained() -> list[dict]:
    return trained_models(get_state().project)


@router.post("/models/activate")
def use_model(request: ActivateRequest) -> dict:
    project = get_state().project
    bundle = project.root / request.path
    if not (bundle.exists() and (bundle.parent / "report.json").exists()):
        raise HTTPException(404, request.path)
    open_project(activate(project, bundle).root)
    return {"ok": True}
