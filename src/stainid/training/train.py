"""Train a stain model on the human labels in the label store and compare it with the current model on the same labels.

Each labelled object's class is the majority of the people who labelled it; its features are those of the matching
candidate in the pool (labels on positions no candidate covers, such as objects people added, cannot train the
classifier and are skipped).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold
from sklearn.pipeline import make_pipeline

from stainid.labels import pool, store
from stainid.labels.tasks import MODEL_KEY
from stainid.project import Project, save_project
from stainid.stains.amyloid.model import MORPHOTYPE_MINIMUM_DIAMETER_UM, morphotype_threshold

MINIMUM_PER_CLASS = 10


def forest():
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                         RandomForestClassifier(n_estimators=500, min_samples_leaf=2, class_weight="balanced", random_state=0, n_jobs=-1))


def labelled(project: Project, stain: str) -> tuple[pd.DataFrame, list[str]]:
    """Consensus-labelled objects with their candidate features (one row per object)."""
    truth = store.consensus(store.current(project.output("reviews")), stain)
    truth = truth[truth.positive.notna()]
    names = pool.feature_names(project, stain)
    rows = []
    for tile_id, group in truth.groupby("tile_id"):
        objects = pool.load_field(project, stain, tile_id)
        for r in group.itertuples():
            i = pool.nearest(objects, r.x, r.y)
            if i is not None:
                rows.append({**objects.iloc[i].to_dict(), "tile_id": tile_id, "label": r.value, "positive": bool(r.positive), "sources": r.sources})
    if not rows:
        raise ValueError(f"No labelled {stain} objects yet: label some on the Labels page first")
    data = pd.DataFrame(rows).drop_duplicates(["tile_id", "x", "y"]).reset_index(drop=True)
    return data, names


def out_of_fold(fit_predict, n: int, y: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, str]:
    """Leave-one-TMA-out when every fold still sees both classes, otherwise stratified 5-fold."""
    logo = LeaveOneGroupOut()
    if len(set(groups)) >= 3 and all(len(set(y[train])) == 2 for train, _ in logo.split(np.zeros(n), y, groups)):
        splits, scheme = logo.split(np.zeros(n), y, groups), "leave-one-TMA-out"
    else:
        splits, scheme = StratifiedKFold(min(5, int(min(y.sum(), (~y).sum()))), shuffle=True, random_state=0).split(np.zeros(n), y), "5-fold"
    oof = np.full(n, np.nan)
    for train, test in splits:
        oof[test] = fit_predict(train, test)
    return oof, scheme


def metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, float]:
    ok = np.isfinite(p)
    y, p = y[ok], p[ok]
    return {"n": int(len(y)), "positives": int(y.sum()), "auc": float(roc_auc_score(y, p)) if len(set(y)) == 2 else float("nan"),
            "precision": float(precision_score(y, p >= threshold, zero_division=0)), "recall": float(recall_score(y, p >= threshold, zero_division=0))}


def _current(project: Project, stain: str, x: np.ndarray, names: list[str]) -> tuple[np.ndarray | None, dict]:
    path = project.model(MODEL_KEY[stain])
    if not path.exists():
        return None, {}
    probability, _ = pool.model_probability(project, stain, x, names)
    return probability, load(path)


def train_model(project: Project, stain: str) -> Path:
    data, names = labelled(project, stain)
    y = data.positive.to_numpy(bool)
    if y.sum() < MINIMUM_PER_CLASS or (~y).sum() < MINIMUM_PER_CLASS:
        raise ValueError(f"Need at least {MINIMUM_PER_CLASS} positive and {MINIMUM_PER_CLASS} negative labels; have {int(y.sum())} and {int((~y).sum())}. "
                         "Label more objects and try again.")
    x = data[names].to_numpy(float)
    groups = data.tma.astype(str).to_numpy()
    ids = (data.tile_id + ":" + data.x.round().astype(int).astype(str) + ":" + data.y.round().astype(int).astype(str)).tolist()
    current, old = _current(project, stain, x, names)
    sources = data.sources.str.split(", ").explode().value_counts()
    print(f"[1/3] {len(y)} labelled objects ({int(y.sum())} positive) from {len(sources)} source(s)", flush=True)

    usable = np.any(np.isfinite(x), axis=0)
    threshold = float(old.get("probability_threshold", 0.4)) if stain == "AT8" else 0.5
    columns = usable if stain == "NeuN" else np.ones(len(names), bool)
    oof, scheme = out_of_fold(lambda tr, te: forest().fit(x[tr][:, columns], y[tr]).predict_proba(x[te][:, columns])[:, 1], len(y), y, groups)
    print(f"[2/3] cross-validated ({scheme})", flush=True)

    if stain == "NeuN":
        bundle = {"classifier": forest().fit(x[:, usable], y), "feature_names": names, "usable_features": usable, "decision_mode": "context_rf",
                  "context_threshold": threshold, "addition_threshold": 0.6, "training_candidate_ids": ids}
    elif stain == "AT8":
        positives = y & data.label.str.startswith("tau+ neuron:").to_numpy()
        mature = data.label.eq("tau+ neuron: tangle").to_numpy()
        p90 = data.soma_dab_p90.to_numpy(float)
        enough = (positives & mature).sum() >= 3 and (positives & ~mature).sum() >= 3
        maturity = float(max(np.unique(p90[positives]), key=lambda v: (p90[positives & mature] >= v).mean() + (p90[positives & ~mature] < v).mean())) \
            if enough else float(old.get("mature_p90_threshold", 0.16))
        bundle = {"classifier": forest().fit(x, y), "feature_names": names, "probability_threshold": threshold, "nms_radius_um": 12.0,
                  "mature_p90_threshold": maturity, "context_um": 55.0, "training_n": int(len(y)), "training_positive_n": int(y.sum())}
    else:
        plaques = y & (data.equivalent_diameter_um.to_numpy(float) >= MORPHOTYPE_MINIMUM_DIAMETER_UM)
        typed = data.label.isin(["compact plaque", "diffuse plaque"]).to_numpy()
        compact = data.label.eq("compact plaque").to_numpy()
        enough = (plaques & typed & compact).sum() >= 5 and (plaques & typed & ~compact).sum() >= 5
        morph = morphotype_threshold(data.normalized_inner_dab.to_numpy(float)[plaques & typed], compact[plaques & typed]) if enough \
            else float(old.get("morphotype_threshold", 1.5))
        bundle = {"identity_classifier": forest().fit(x, y), "identity_feature_names": names, "identity_threshold": threshold,
                  "identity_training_candidate_ids": ids, "morphotype_feature": "inner_mean_dab_od / slide 6E10 calibration threshold",
                  "morphotype_threshold": float(morph), "morphotype_minimum_diameter_um": MORPHOTYPE_MINIMUM_DIAMETER_UM,
                  "morphotype_training_n": int((plaques & typed).sum()) if enough else int(old.get("morphotype_training_n", 0))}

    folder = project.output("trained_models") / f"{stain}_{time.strftime('%Y%m%d-%H%M%S')}"
    folder.mkdir(parents=True)
    dump(bundle, folder / "bundle.joblib")
    report = {"stain": stain, "created": time.time(), "validation": scheme, "threshold": threshold, "sources": sources.to_dict(),
              "new_model": metrics(y, oof, threshold), "current_model": metrics(y, current, threshold) if current is not None else None,
              "current_model_path": project.relative(project.model(MODEL_KEY[stain]))}
    (folder / "report.json").write_text(json.dumps(report, indent=1))
    new = report["new_model"]
    print(f"[3/3] new model: AUC {new['auc']:.2f}, precision {new['precision']:.2f}, recall {new['recall']:.2f} -> {project.relative(folder)}", flush=True)
    return folder


def trained_models(project: Project) -> list[dict]:
    root = project.output("trained_models")
    reports = sorted(root.glob("*/report.json"), reverse=True) if root.exists() else []
    active = {project.model(k).resolve() for k in MODEL_KEY.values()}
    return [{**json.loads(r.read_text()), "path": project.relative(r.parent / "bundle.joblib"), "active": (r.parent / "bundle.joblib").resolve() in active}
            for r in reports]


def activate(project: Project, bundle: Path) -> Project:
    stain = json.loads((bundle.parent / "report.json").read_text())["stain"]
    return save_project(project, {"models": {MODEL_KEY[stain]: project.relative(bundle)}})
