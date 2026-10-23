from __future__ import annotations

import time
from functools import lru_cache
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from stainid.api import work
from stainid.api.state import get_state, reload_project
from stainid.training.train import activate, trained_models

router = APIRouter(tags=["models"])
ROLES = {
    "neun": "Decides which candidate objects on NeuN slides are neurons (random forest on shape, stain and 55 µm neighbourhood).",
    "amyloid": "Decides which 6E10 deposits are plaques (random forest on shape, stain and surroundings) and sorts them into compact and diffuse.",
    "tau": "Decides which AT8 candidates are tau+ neurons (random forest on shape, stain and neighbourhood).",
    "cellpose": "Cellpose-SAM: finds cells and nuclei. Published weights, the same for every study.",
    "sam": "Segment Anything (ViT-B): draws precise object outlines. Published weights.",
    "plaque_cnn_dir": "Published amyloid CNNs (Plaquebox 2019, Wong 2022 consensus); optional, used to compare the 6E10 model with published ones.",
    "huggingface_home": "Phikon pathology foundation model (owkin/phikon); optional image features for analyses.",
}
SOURCE = {"neun": "train", "amyloid": "train", "tau": "train", "cellpose": "download", "sam": "download", "huggingface_home": "download",
          "plaque_cnn_dir": "manual"}
STAIN = {"neun": "NeuN", "amyloid": "6E10", "tau": "AT8"}


class ActivateRequest(BaseModel):
    path: str


class ExtraProjects(BaseModel):
    projects: list[str]  # folders of the other projects whose labels the models also learn from (all of them: the new list)


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


@lru_cache(maxsize=32)
def _size(path: Path, mtime: float) -> int:
    """Bytes in a model file or folder (a folder is walked once per change of its modification time)."""
    return path.stat().st_size if path.is_file() else sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


@router.get("/models")
def models() -> list[dict]:
    project = get_state().project
    out = []
    for key, role in ROLES.items():
        path = project.model(key)
        entry = {"key": key, "role": role, "path": project.relative(path), "exists": path.exists(), "source": SOURCE[key], "stain": STAIN.get(key)}
        if path.exists():
            entry |= {"size_mb": round(_size(path, path.stat().st_mtime) / 1e6, 1), "modified": time.strftime("%Y-%m-%d", time.localtime(path.stat().st_mtime))}
            if path.suffix == ".joblib":
                entry["bundle"] = describe_bundle(str(path), path.stat().st_mtime)
        out.append(entry)
    return out


@router.get("/models/trained")
def trained() -> list[dict]:
    return trained_models(get_state().project)


def _labels_of(root: Path) -> dict[str, int]:
    """How many objects people labelled in another project, per stain (what its training would add)."""
    from stainid.labels import sessions as label_sessions
    from stainid.labels import simple, store
    from stainid.project import load_project

    other = load_project(root)
    labels = store.current(other.output("reviews"))
    out = {}
    for stain in other.stains:
        sets = label_sessions.label_sets(labels, stain, {}, simple=simple.of(other, stain)) if len(labels) else []
        out[stain] = int(sum(s["objects"] for s in sets))
    return out


@router.get("/models/extra-projects")
def extra_projects() -> dict:
    """The other projects whose labels this project's models also learn from (training-only cases, never in the test),
    each with its labelled objects per stain, and the other projects on this server's list that could be added."""
    from stainid.api.registry import entries, project_id
    from stainid.project import CONFIG_NAME
    from stainid.training.train import extra_projects as chosen

    project = get_state().project
    on = chosen(project)
    out = []
    for root in on:
        entry = {"path": str(root), "id": project_id(root), "name": root.name, "enabled": True, "exists": (root / CONFIG_NAME).exists()}
        if entry["exists"]:
            try:
                from stainid.project import load_project

                other = load_project(root)
                entry |= {"name": other.name, "kind": other.kind, "stains": list(other.stains), "labels": _labels_of(root)}
            except Exception as error:  # noqa: BLE001 - a project that does not load is shown with why
                entry["error"] = str(error)
        out.append(entry)
    here = project.root.resolve()
    available = [{"path": e.path, "id": e.id, "name": e.name, "enabled": False, "exists": True} for e in entries()
                 if e.exists and Path(e.path).resolve() != here and Path(e.path).resolve() not in on]
    return {"projects": out + available, "stains": list(project.stains)}


@router.put("/models/extra-projects")
def set_extra_projects(request: ExtraProjects) -> dict:
    """Choose the other projects whose labels the models also learn from (`training.extra_projects`)."""
    from stainid.project import CONFIG_NAME, save_project

    project = get_state().project
    paths: list[str] = []
    for text in request.projects:
        path = Path(text).expanduser()
        path = (path if path.is_absolute() else project.root / path).resolve()
        if path == project.root.resolve():
            raise HTTPException(400, "A project cannot lend its labels to itself")
        if not (path / CONFIG_NAME).exists():
            raise HTTPException(404, f"{path} is not a stainID project (it has no {CONFIG_NAME})")
        if str(path) not in paths:
            paths.append(str(path))
    reload_project(save_project(project, {"training": {"extra_projects": paths}}).root)
    return extra_projects()


@router.post("/models/activate")
def use_model(request: ActivateRequest) -> dict:
    project = get_state().project
    bundle = project.root / request.path
    if not (bundle.exists() and (bundle.parent / "report.json").exists()):
        raise HTTPException(404, request.path)
    reload_project(activate(project, bundle).root)
    return {"ok": True}


def _fixed_stains() -> list[str]:
    from stainid.labels import candidate_store
    from stainid.labels.tasks import STAINS

    return [s for s in STAINS if candidate_store.cores(get_state().project, s)]


QUICK_WAIT = 0.5  # how long the outline-fixes page waits for a new comparison before it shows "preparing"


@router.get("/models/outline-fixes")
def outline_fixes():
    """Per stain with candidates: the outlines people fixed in "Fix the field" against each of detection's outlines
    (`labels.outline_fixes`), the evidence for the stain's default outline (never changed here). Worked out in the
    background (a whole cohort's comparison); a server started again answers with the last one meanwhile."""
    from stainid import disk_cache
    from stainid.labels import outline_fixes as fixes

    state = get_state()
    return work.answer("outline fixes", ("outline fixes", str(state.root)),
                       lambda task: {"stains": [fixes.summary(state.project, s, state.tiles) for s in _fixed_stains()]}, work.NOW,
                       "Comparing the outlines people fixed", fresh=work.within(10), saved=disk_cache.answer_file(state.root, "outline-fixes.json"),
                       seconds=QUICK_WAIT)  # a cohort's comparison: the page shows "preparing" at once instead of waiting


@router.get("/models/outline-fixes.csv")
def outline_fixes_csv() -> Response:
    """Every fixed outline with its comparison (one row per object), to train a better outline from."""
    from stainid.labels import outline_fixes as fixes

    state = get_state()
    parts = [p for p in (fixes.rows(state.project, s, state.tiles) for s in _fixed_stains()) if len(p)]
    frame = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=fixes.ROW_COLUMNS)
    return Response(frame.to_csv(index=False), media_type="text/csv", headers={"Content-Disposition": 'attachment; filename="outline_fixes.csv"'})
