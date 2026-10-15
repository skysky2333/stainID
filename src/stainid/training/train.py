"""Train a stain model on the human labels in the label store and compare it with the current model on the same labels.

Each labelled object's class is the majority of the people who labelled it; its features are those of the matching
candidate in the pool. An object people added where the detection step proposed nothing is measured at the clicked
position first (`labels.added`) and trains the model like a candidate; the report counts these person-added objects and
those that could not be measured (their labels stay, but cannot train). Labels on positions with no candidate on fields
that are not prepared are skipped.
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
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline

from stainid.labels import added, frames, pool, store
from stainid.labels import sessions as label_sessions
from stainid.labels.tasks import MODEL_KEY
from stainid.project import Project, save_project
from stainid.sampling.tiles import known_tiles
from stainid.stains.amyloid.model import MORPHOTYPE_MINIMUM_DIAMETER_UM, morphotype_threshold

MINIMUM_PER_CLASS = 10


def forest():
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                         RandomForestClassifier(n_estimators=500, min_samples_leaf=2, class_weight="balanced", random_state=0, n_jobs=-1))


def labelled(project: Project, stain: str, sessions: list[str] | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Consensus-labelled objects with their candidate features (one row per object), from the labels of these
    sessions (`labels.sessions.in_sessions`; all labels when None)."""
    labels = frames.in_field_frame(store.current(project.output("reviews")), known_tiles(project), project.context_px)
    counts = added.measure(project, stain)
    if any(counts.values()):
        print(f"measured {counts['measured']} object(s) people added; {counts['could_not_be_measured']} could not be measured", flush=True)
    truth = store.consensus(label_sessions.in_sessions(labels, sessions), stain)
    truth = truth[truth.positive.notna()]
    names = pool.feature_names(project, stain)
    rows, unmeasured = [], 0
    for tile_id, group in truth.groupby("tile_id"):
        objects = pool.load_field(project, stain, tile_id)
        extra = added.load_field(project, stain, tile_id)
        for r in group.itertuples():
            i = pool.nearest(objects, r.x, r.y)
            hit = added.resolve(extra, r.x, r.y) if i is None else None
            if hit is not None and pd.notna(hit.get("candidate_x")):  # the click was off the centre of a candidate
                i, hit = pool.nearest(objects, float(hit.candidate_x), float(hit.candidate_y)), None
            label = {"tile_id": tile_id, "label": r.value, "positive": bool(r.positive), "sources": r.sources}
            if i is not None:
                rows.append({**objects.iloc[i].to_dict(), **label, "added_by_person": False})
            elif hit is not None and added.is_true(hit.measured):
                rows.append({**hit.to_dict(), **label, "added_by_person": True})
            elif hit is not None:
                unmeasured += 1
    if not rows:
        raise ValueError(f"No labelled {stain} objects {'in the chosen sessions ' if sessions else ''}yet: label some on the Labels page first")
    data = pd.DataFrame(rows).drop_duplicates(["tile_id", "x", "y"]).reset_index(drop=True)
    data["added_by_person"] = data.added_by_person.astype(bool)
    data.attrs["unmeasured"] = unmeasured
    return data, names


def person_added(data: pd.DataFrame, oof: np.ndarray, threshold: float) -> dict:
    """The objects people added (never proposed by the detection step) among the training labels, and the end-to-end
    reading of the new model: of the real objects, how many it finds, how many the detection step never proposes (no
    classifier can find those until candidate proposal changes) and how many are proposed but rejected by the model."""
    flag = data.added_by_person.to_numpy(bool)
    y = data.positive.to_numpy(bool)
    unmeasured = int(data.attrs.get("unmeasured", 0))
    proposed = y & ~flag & np.isfinite(oof)
    found = int((proposed & (oof >= threshold)).sum())
    rejected = int(proposed.sum()) - found
    never = int((y & flag).sum()) + unmeasured
    real = found + rejected + never
    share = (lambda k: k / real if real else None)  # noqa: E731
    return {"objects": int(flag.sum()), "positive": int((y & flag).sum()), "could_not_be_measured": unmeasured,
            "detection": {"real": real, "found": found, "never_proposed": never, "rejected": rejected, "recall": share(found),
                          "never_proposed_share": share(never), "rejected_share": share(rejected)}}


def out_of_fold(fit_predict, n: int, y: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, str]:
    """Out-of-fold predictions that never split a group (TMA) between training and testing. Each donor sits on one TMA,
    so every split is by donor: leave-one-TMA-out when every training fold still sees both classes, otherwise
    stratified folds of whole TMAs."""
    logo = LeaveOneGroupOut()
    if len(set(groups)) >= 3 and all(len(set(y[train])) == 2 for train, _ in logo.split(np.zeros(n), y, groups)):
        splits, scheme = logo.split(np.zeros(n), y, groups), "leave-one-TMA-out"
    else:
        folds = min(5, len(set(groups)))
        if folds < 2:
            raise ValueError("Validation needs labels from at least two TMAs, so that no donor is in both training and testing")
        splits = StratifiedGroupKFold(folds, shuffle=True, random_state=0).split(np.zeros(n), y, groups)
        scheme = f"{folds}-fold, whole TMAs per fold"
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


def train_model(project: Project, stain: str, sessions: list[str] | None = None) -> Path:
    """A new model from the labels of these sessions (all when None), checked by leaving one TMA out at a time."""
    data, names = labelled(project, stain, sessions)
    y = data.positive.to_numpy(bool)
    if y.sum() < MINIMUM_PER_CLASS or (~y).sum() < MINIMUM_PER_CLASS:
        raise ValueError(f"Need at least {MINIMUM_PER_CLASS} positive and {MINIMUM_PER_CLASS} negative labels; have {int(y.sum())} and {int((~y).sum())}. "
                         "Label more objects and try again.")
    x = data[names].to_numpy(float)
    groups = data.tma.astype(str).to_numpy()
    ids = (data.tile_id + ":" + data.x.round().astype(int).astype(str) + ":" + data.y.round().astype(int).astype(str)).tolist()
    current, old = _current(project, stain, x, names)
    sources = data.sources.str.split(", ").explode().value_counts()
    extra = int(data.added_by_person.sum())
    print(f"[1/3] {len(y)} labelled objects ({int(y.sum())} positive) from {len(sources)} source(s)"
          f"{f', including {extra} person-added object(s)' if extra else ''}", flush=True)

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
              "current_model_path": project.relative(project.model(MODEL_KEY[stain])),
              "sessions": trained_on(project, stain, sessions), "person_added": person_added(data, oof, threshold)}
    (folder / "report.json").write_text(json.dumps(report, indent=1))
    new = report["new_model"]
    print(f"[3/3] new model: AUC {new['auc']:.2f}, precision {new['precision']:.2f}, recall {new['recall']:.2f} -> {project.relative(folder)}", flush=True)
    return folder


def trained_on(project: Project, stain: str, sessions: list[str] | None) -> list[dict]:
    """The sessions whose labels a training used (every session with labels of this stain when None): name, title and
    how many objects each labelled, for the model's report."""
    root = project.output("reviews")
    titles = {name: label_sessions.meta(root, name).get("title", name) for name in label_sessions.names(root)}
    used = label_sessions.label_sets(label_sessions.in_sessions(store.current(root), sessions), stain, titles)
    return [{"name": s["name"], "title": s["title"], "objects": s["objects"]} for s in used]


def trained_models(project: Project) -> list[dict]:
    root = project.output("trained_models")
    reports = sorted(root.glob("*/report.json"), reverse=True) if root.exists() else []
    active = {project.model(k).resolve() for k in MODEL_KEY.values()}
    return [{**json.loads(r.read_text()), "path": project.relative(r.parent / "bundle.joblib"), "active": (r.parent / "bundle.joblib").resolve() in active}
            for r in reports]


def activate(project: Project, bundle: Path) -> Project:
    stain = json.loads((bundle.parent / "report.json").read_text())["stain"]
    return save_project(project, {"models": {MODEL_KEY[stain]: project.relative(bundle)}})
