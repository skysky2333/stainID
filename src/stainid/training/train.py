"""Train a stain model from labelled training sets and compare it with the current model on the same labels."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from stainid.project import Project, save_project
from stainid.review.store import read_labels
from stainid.stains.amyloid.model import MORPHOTYPE_MINIMUM_DIAMETER_UM, morphotype_threshold
from stainid.training.sets import MODEL_KEY, POSITIVE, SKIP

MINIMUM_PER_CLASS = 10


def forest():
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                         RandomForestClassifier(n_estimators=500, min_samples_leaf=2, class_weight="balanced", random_state=0, n_jobs=-1))


def probe():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=5000, class_weight="balanced"))


def training_folders(project: Project, stain: str) -> list[Path]:
    root = project.output("reviews") / "training"
    metas = sorted(root.glob("*/meta.json")) if root.exists() else []
    return [m.parent for m in metas if json.loads(m.read_text()).get("stain") == stain]


def labelled(project: Project, stain: str) -> tuple[pd.DataFrame, np.ndarray | None, list[str]]:
    frames, embeddings, names = [], [], None
    for folder in training_folders(project, stain):
        meta = json.loads((folder / "meta.json").read_text())
        if names is not None and meta["feature_names"] != names:
            raise ValueError(f"{folder.name} was made with different feature definitions than the other {stain} training sets")
        names = meta["feature_names"]
        labels = read_labels(folder)
        if labels.empty:
            continue
        features = pd.read_csv(folder / "features.csv", dtype={"review_id": str})
        key = pd.read_csv(folder / "key.csv", dtype={"review_id": str, "tma": str})[["review_id", "tma"]]
        frame = features.merge(key, on="review_id").merge(labels[["review_id", "label"]], on="review_id")
        frame = frame[~frame.label.isin(SKIP)]
        frame.insert(0, "set", folder.name)
        frames.append(frame)
        if (folder / "embeddings.npy").exists():
            embeddings.append(np.load(folder / "embeddings.npy")[features.review_id.isin(frame.review_id).to_numpy()])
    if not frames:
        raise ValueError(f"No labelled {stain} training candidates yet: create a training set and label it first")
    data = pd.concat(frames, ignore_index=True)
    stacked = np.concatenate(embeddings) if embeddings and sum(len(e) for e in embeddings) == len(data) else None
    return data, stacked, names


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


def _current(project: Project, stain: str, x: np.ndarray, names: list[str], emb: np.ndarray | None, wong: np.ndarray | None) -> tuple[np.ndarray | None, dict]:
    path = project.model(MODEL_KEY[stain])
    if not path.exists():
        return None, {}
    bundle = load(path)
    if stain == "NeuN" and names == bundle["feature_names"]:
        return bundle["classifier"].predict_proba(x[:, bundle["usable_features"]])[:, 1], bundle
    if stain == "AT8" and names == bundle["feature_names"]:
        return bundle["classifier"].predict_proba(x)[:, 1], bundle
    if stain == "6E10" and names == bundle["identity_feature_names"]:
        parts = [bundle["identity_classifier"].predict_proba(x)[:, 1]]
        if "phikon_classifier" in bundle and emb is not None:
            parts.append(bundle["phikon_classifier"].predict_proba(emb)[:, 1])
            if bundle.get("use_wong_consensus") and wong is not None:
                parts.append(wong)
        return np.mean(parts, axis=0), bundle
    return None, bundle


def train_model(project: Project, stain: str) -> Path:
    data, emb, names = labelled(project, stain)
    y = data.label.isin(POSITIVE[stain]).to_numpy()
    if y.sum() < MINIMUM_PER_CLASS or (~y).sum() < MINIMUM_PER_CLASS:
        raise ValueError(f"Need at least {MINIMUM_PER_CLASS} positive and {MINIMUM_PER_CLASS} negative labels; have {int(y.sum())} and {int((~y).sum())}. "
                         "Label more candidates (or create another training set) and try again.")
    x = data[names].to_numpy(float)
    groups = data.tma.astype(str).to_numpy()
    wong = data["wong_plaque_probability"].to_numpy(float) if "wong_plaque_probability" in data and data["wong_plaque_probability"].notna().all() else None
    ids = (data.set + "/" + data.review_id).tolist()
    current, old = _current(project, stain, x, names, emb, wong)
    print(f"[1/3] {len(y)} labels ({int(y.sum())} positive) from {data.set.nunique()} training set(s)", flush=True)

    usable = np.any(np.isfinite(x), axis=0)
    if stain == "NeuN":
        threshold = 0.5
        oof, scheme = out_of_fold(lambda tr, te: forest().fit(x[tr][:, usable], y[tr]).predict_proba(x[te][:, usable])[:, 1], len(y), y, groups)
    elif stain == "AT8":
        threshold = float(old.get("probability_threshold", 0.4))
        oof, scheme = out_of_fold(lambda tr, te: forest().fit(x[tr], y[tr]).predict_proba(x[te])[:, 1], len(y), y, groups)
    else:
        threshold = 0.5

        def ensemble(tr, te):
            parts = [forest().fit(x[tr], y[tr]).predict_proba(x[te])[:, 1]]
            if emb is not None:
                parts.append(probe().fit(emb[tr], y[tr]).predict_proba(emb[te])[:, 1])
                if wong is not None:
                    parts.append(wong[te])
            return np.mean(parts, axis=0)

        oof, scheme = out_of_fold(ensemble, len(y), y, groups)
    print(f"[2/3] cross-validated ({scheme})", flush=True)

    if stain == "NeuN":
        bundle = {"classifier": forest().fit(x[:, usable], y), "feature_names": names, "usable_features": usable, "decision_mode": "context_rf",
                  "context_threshold": threshold, "addition_threshold": 0.6, "training_candidate_ids": ids}
    elif stain == "AT8":
        positives = y & data.label.str.startswith("tau+ neuron").to_numpy()
        mature = data.label.eq("tau+ neuron: tangle").to_numpy()
        p90 = data.soma_dab_p90.to_numpy(float)
        enough = (positives & mature).sum() >= 3 and (positives & ~mature).sum() >= 3
        maturity = float(max(np.unique(p90[positives]), key=lambda v: (p90[positives & mature] >= v).mean() + (p90[positives & ~mature] < v).mean())) \
            if enough else float(old.get("mature_p90_threshold", 0.16))
        bundle = {"classifier": forest().fit(x, y), "feature_names": names, "probability_threshold": threshold, "nms_radius_um": 12.0,
                  "mature_p90_threshold": maturity, "context_um": 55.0, "training_n": int(len(y)), "training_positive_n": int(y.sum())}
    else:
        plaques = y & (data.equivalent_diameter_um.to_numpy(float) >= MORPHOTYPE_MINIMUM_DIAMETER_UM)
        compact = data.label.eq("compact plaque").to_numpy()
        enough = (plaques & compact).sum() >= 5 and (plaques & ~compact).sum() >= 5
        morph = morphotype_threshold(data.normalized_inner_dab.to_numpy(float)[plaques], compact[plaques]) if enough else float(old.get("morphotype_threshold", 1.5))
        bundle = {"identity_classifier": forest().fit(x, y), "identity_feature_names": names, "identity_threshold": threshold,
                  "identity_training_candidate_ids": ids, "morphotype_feature": "inner_mean_dab_od / slide 6E10 calibration threshold",
                  "morphotype_threshold": float(morph), "morphotype_minimum_diameter_um": MORPHOTYPE_MINIMUM_DIAMETER_UM,
                  "morphotype_training_n": int(plaques.sum()) if enough else int(old.get("morphotype_training_n", 0))}
        if emb is not None:
            bundle |= {"phikon_classifier": probe().fit(emb, y), "use_wong_consensus": wong is not None,
                       "identity_rule": "mean of morphology-context random forest, Phikon linear probe" + (" and Wong 2022 consensus CNN" if wong is not None else "")}

    folder = project.output("trained_models") / f"{stain}_{time.strftime('%Y%m%d-%H%M%S')}"
    folder.mkdir(parents=True)
    dump(bundle, folder / "bundle.joblib")
    report = {"stain": stain, "created": time.time(), "validation": scheme, "threshold": threshold, "sets": sorted(data.set.unique()),
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
