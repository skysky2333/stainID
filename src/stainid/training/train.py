"""Train a stain model on the human labels in the label store and compare it with the current model on the same labels.

Each labelled object's class is the majority of the people who labelled it; its features are those of the matching
candidate in the pool, or, for objects of the annotation rounds, of the v2 candidate store (`labels.candidate_store`:
the decisions people made there are class labels on its candidates). An object people added where the detection step proposed nothing is measured at the clicked
position first (`labels.added`) and trains the model like a candidate; the report counts these person-added objects and
those that could not be measured (their labels stay, but cannot train). Labels on positions with no candidate on fields
that are not prepared are skipped.

Other stainID projects can lend their labels as extra training cases (`training.extra_projects` in stainid.yaml, the
toggle on the Models page; `extra_labelled`): their labelled candidates, with their own candidate features, join every
fit, and never the evaluation. The out-of-fold (leave-one-TMA-out) check holds out only the study's own TMAs and scores
only the study's own labels; an extra case of a donor in the held-out fold is left out of that fold's fit. The model in
use is compared on the study's labels alone, and nothing else in the study (results, the Labels page, scoring) reads
another project's labels. Simple labels of an extra project train through this project's mapping, as the study's own do.
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

from stainid import exclusions as excluded_areas
from stainid.labels import added, candidate_store, decisions, frames, pool, simple, store
from stainid.labels import sessions as label_sessions
from stainid.labels.tasks import MODEL_KEY
from stainid.project import CONFIG_NAME, Project, load_project, save_project
from stainid.sampling.tiles import known_tiles
from stainid.stains.amyloid.model import MORPHOTYPE_MINIMUM_DIAMETER_UM, morphotype_threshold
from stainid.v2 import scaled as v2_scaled
from stainid.v2.params import params as v2_params

MINIMUM_PER_CLASS = 10


def forest():
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                         RandomForestClassifier(n_estimators=500, min_samples_leaf=2, class_weight="balanced", random_state=0, n_jobs=-1))


def labelled(project: Project, stain: str, sessions: list[str] | None = None, mapping_from: Project | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Consensus-labelled objects with their candidate features (one row per object), from the labels of these
    sessions (`labels.sessions.in_sessions`; all labels when None). Simple labels train through the mapping of
    `mapping_from` (the project whose model is trained; default this one)."""
    labels = frames.in_field_frame(excluded_areas.drop_excluded(project, store.current(project.output("reviews"))), known_tiles(project),
                                  project.context_px)
    counts = added.measure(project, stain)
    if any(counts.values()):
        print(f"measured {counts['measured']} object(s) people added; {counts['could_not_be_measured']} could not be measured", flush=True)
    chosen = label_sessions.in_sessions(labels, sessions)
    mapper = mapping_from or project
    mapping = simple.used_mapping(chosen, mapper, stain)  # simple sessions' labels train through the mapping on the Models page
    chosen = simple.with_training_rows(chosen, mapper, stain)
    truth = decisions.consensus(project, stain, chosen, known_tiles(project))  # by candidate
    truth = truth[truth.positive.notna()]
    names = pool.feature_names(project, stain)
    tiles = known_tiles(project)
    stored, names = candidate_store.training_rows(project, stain, truth, tiles, names)
    rows, unmeasured = [], 0
    for tile_id, group in truth.groupby("tile_id"):
        objects = pool.load_field(project, stain, tile_id)
        extra = added.load_field(project, stain, tile_id)
        for r in group.itertuples():
            i = pool.nearest(objects, r.x, r.y)
            hit = added.resolve(extra, r.x, r.y) if i is None else None
            if hit is not None and pd.notna(hit.get("candidate_x")):  # the click was off the centre of a candidate
                i, hit = pool.nearest(objects, float(hit.candidate_x), float(hit.candidate_y)), None
            label = {"tile_id": tile_id, "label": r.value, "subtype": r.subtype, "positive": bool(r.positive), "sources": r.sources}
            if i is not None:
                rows.append({**objects.iloc[i].to_dict(), **label, "added_by_person": False})
            elif r.Index in stored:
                rows.append({**stored[r.Index], **label, "added_by_person": False})
            elif hit is not None and added.is_true(hit.measured):
                rows.append({**hit.to_dict(), **label, "added_by_person": True})
            elif hit is not None:
                unmeasured += 1
    if not rows:
        raise ValueError(f"No labelled {stain} objects {'in the chosen sessions ' if sessions else ''}yet: label some on the Labels page first")
    data = pd.DataFrame(rows).drop_duplicates(["tile_id", "x", "y"]).reset_index(drop=True)
    data["added_by_person"] = data.added_by_person.astype(bool)
    if "donor_id" in tiles and "tile_id" in tiles:  # the donor of each object, for the extra cases' fold guard
        donor = data.tile_id.map(tiles.drop_duplicates("tile_id").set_index("tile_id").donor_id)
        data["donor_id"] = donor.where(donor.notna(), data["donor_id"]) if "donor_id" in data else donor
    data.attrs["unmeasured"] = unmeasured
    data.attrs["simple_mapping"] = mapping
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


def out_of_fold(fit_predict, n: int, y: np.ndarray, groups: np.ndarray, fit_only: np.ndarray | None = None,
                donors: np.ndarray | None = None) -> tuple[np.ndarray, str]:
    """Out-of-fold predictions that never split a group (TMA) between training and testing. Each donor sits on one TMA,
    so every split is by donor: leave-one-TMA-out when every training fold still sees both classes, otherwise
    stratified folds of whole TMAs. Rows marked `fit_only` (other projects' cases) are never tested and never make a
    fold: they join the training of every fold, except those of a donor (`donors`) that the fold tests; their
    predictions stay NaN."""
    fit_only = np.zeros(n, bool) if fit_only is None else np.asarray(fit_only, bool)
    own, extra = np.flatnonzero(~fit_only), np.flatnonzero(fit_only)
    y_own, groups_own = y[own], groups[own]

    def fold(train, test):  # positions among the study's rows -> rows of all, with the extra cases in training
        train, test = own[train], own[test]
        if len(extra):
            keep = extra if donors is None else extra[~np.isin(donors[extra], [d for d in donors[test] if isinstance(d, str) and d])]
            train = np.concatenate([train, keep])
        return train, test

    m = len(own)
    logo = LeaveOneGroupOut()
    if len(set(groups_own)) >= 3 and all(len(set(y[fold(tr, te)[0]])) == 2 for tr, te in logo.split(np.zeros(m), y_own, groups_own)):
        splits, scheme = logo.split(np.zeros(m), y_own, groups_own), "leave-one-TMA-out"
    else:
        folds = min(5, len(set(groups_own)))
        if folds < 2:
            raise ValueError("Validation needs labels from at least two TMAs, so that no donor is in both training and testing")
        splits = StratifiedGroupKFold(folds, shuffle=True, random_state=0).split(np.zeros(m), y_own, groups_own)
        scheme = f"{folds}-fold, whole TMAs per fold"
    oof = np.full(n, np.nan)
    for train, test in splits:
        train, test = fold(train, test)
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


def slide_scaled(project: Project, stain: str) -> bool:
    """Whether a new model of this stain takes slide-scaled inputs (`v2.training.<stain>.slide_scaled`)."""
    return bool(((v2_params(project).get("training") or {}).get(stain) or {}).get("slide_scaled", False))


def with_scaled(project: Project, stain: str, data: pd.DataFrame, names: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """The labelled objects with the slide-scaled version of each input that has one (`s_<name>`, as stage 2 stores
    them), and those names. Where a row lacks one (a candidate of the old pool, a store made before the `s_` columns), an
    intensity is its OD over the slide threshold of the object's TMA; a recomputed measure (6E10 dense core / seed, NeuN
    stained area) stays empty (the model's imputer fills it)."""
    from stainid.workflows import calibration_table

    scaled = v2_scaled.names(stain, names)
    calibration = calibration_table(project.input("calibration")) if project.input("calibration").exists() else {}
    threshold = data.tma.astype(str).map(lambda t: calibration.get((t, stain), np.nan)).astype(float)
    out = data.copy()
    for name, base in zip(scaled, names):
        if name == base:
            continue
        values = pd.to_numeric(out[name], errors="coerce") if name in out else pd.Series(np.nan, index=out.index)
        if base in v2_scaled.INTENSITY.get(stain, ()) and base in out:
            values = values.fillna(pd.to_numeric(out[base], errors="coerce") / threshold)
        out[name] = values
    for base in v2_scaled.INTENSITY.get(stain, ()):  # also the scaled intensities the model does not take (the AT8 maturity cut)
        if base in out and f"s_{base}" not in out:
            out[f"s_{base}"] = pd.to_numeric(out[base], errors="coerce") / threshold
    return out, scaled


def extra_projects(project: Project) -> list[Path]:
    """The other projects whose labels this project's models also learn from (`training.extra_projects`: folders,
    relative to this project or absolute; this project itself and repeats are left out)."""
    out: list[Path] = []
    for value in (project.config.get("training") or {}).get("extra_projects") or []:
        text = str(value.get("path", "") if isinstance(value, dict) else value).strip()
        if not text:
            continue
        path = Path(text).expanduser()
        path = (path if path.is_absolute() else project.root / path).resolve()
        if path.name == CONFIG_NAME:
            path = path.parent
        if path != project.root.resolve() and path not in out:
            out.append(path)
    return out


EXTRA_GROUP = "extra:"  # the group (`tma`) of an extra case starts with this: never a fold of the out-of-fold check


def extra_labelled(project: Project, stain: str, names: list[str], scaled: bool, log=print) -> tuple[pd.DataFrame, list[dict]]:
    """The labelled objects of the extra projects (`extra_projects`) as training-only rows with this training's inputs
    (`names`; an input another project lacks is empty and the model's imputer fills it), and per project what it
    gave: its name, folder, objects and positives used, sessions and the simple mapping (this project's) its simple
    labels trained through, or a note on why it gave nothing."""
    frames_: list[pd.DataFrame] = []
    used: list[dict] = []
    for root in extra_projects(project):
        entry: dict = {"path": str(root), "name": root.name, "objects": 0, "positives": 0}
        used.append(entry)
        if not (root / CONFIG_NAME).exists():
            entry["note"] = "not found"
            continue
        try:
            other = load_project(root)
            entry["name"] = other.name
            if stain not in other.stains:
                entry["note"] = f"no {stain} slides"
                continue
            data, other_names = labelled(other, stain, None, mapping_from=project)
            if scaled:
                data, _ = with_scaled(other, stain, data, other_names)
        except ValueError as error:  # no labels of this stain yet, or a table it cannot read
            entry["note"] = "no labels yet" if str(error).startswith("No labelled") else str(error)
            continue
        data = data.copy()
        for name in names:
            if name not in data:
                data[name] = np.nan
        data["tma"] = EXTRA_GROUP + entry["name"] + ":" + data.tma.astype(str)
        data["extra_project"] = str(root)
        data["candidate_id"] = str(root.name) + "/" + data.tile_id.astype(str) + ":" + data.x.round().astype(int).astype(str) + ":" \
            + data.y.round().astype(int).astype(str)
        positives = data.positive.astype(bool)
        entry |= {"objects": int(len(data)), "positives": int(positives.sum()), "negatives": int((~positives).sum()),
                  "sessions": trained_on(other, stain, None, mapping_from=project), "simple_mapping": data.attrs.get("simple_mapping")}
        log(f"extra training cases: {len(data)} labelled {stain} objects ({int(positives.sum())} positive) from {entry['name']} ({root})", flush=True)
        frames_.append(data)
    return (pd.concat(frames_, ignore_index=True) if frames_ else pd.DataFrame()), used


def train_model(project: Project, stain: str, sessions: list[str] | None = None, scaled: bool | None = None, extra: bool = True) -> Path:
    """A new model from the labels of these sessions (all when None), checked by leaving one TMA out at a time. `scaled`
    (default: `slide_scaled`): slide-scaled inputs instead of absolute DAB OD (`stainid.v2.scaled`). `extra`: also fit
    on the labels of the extra projects (`extra_projects`), which are never tested."""
    data, names = labelled(project, stain, sessions)
    scaled = slide_scaled(project, stain) if scaled is None else bool(scaled)
    current_names = names
    if scaled:
        data, names = with_scaled(project, stain, data, names)
    y_own = data.positive.to_numpy(bool)
    if y_own.sum() < MINIMUM_PER_CLASS or (~y_own).sum() < MINIMUM_PER_CLASS:
        raise ValueError(f"Need at least {MINIMUM_PER_CLASS} positive and {MINIMUM_PER_CLASS} negative labels; have {int(y_own.sum())} and "
                         f"{int((~y_own).sum())}. Label more objects and try again.")
    own_ids = (data.tile_id + ":" + data.x.round().astype(int).astype(str) + ":" + data.y.round().astype(int).astype(str)).tolist()
    current, old = _current(project, stain, data[current_names].to_numpy(float), current_names)
    sources = data.sources.str.split(", ").explode().value_counts()
    added_n = int(data.added_by_person.sum())
    study = data
    extra_rows, extra_used = extra_labelled(project, stain, names, scaled) if extra else (pd.DataFrame(), [])
    if len(extra_rows):
        data = pd.concat([study, extra_rows], ignore_index=True)
    own = np.arange(len(data)) < len(study)
    y = data.positive.to_numpy(bool)
    x = data[names].to_numpy(float)
    groups = data.tma.astype(str).to_numpy()
    donors = data.donor_id.astype(object).where(data.donor_id.notna(), "").astype(str).to_numpy() if "donor_id" in data else None
    ids = own_ids + (extra_rows.candidate_id.tolist() if len(extra_rows) else [])
    print(f"[1/3] {len(study)} labelled objects ({int(y_own.sum())} positive) from {len(sources)} source(s)"
          f"{f', including {added_n} person-added object(s)' if added_n else ''}"
          f"{f'; {len(extra_rows)} more from other projects, for fitting only' if len(extra_rows) else ''}", flush=True)

    usable = np.any(np.isfinite(x), axis=0)
    threshold = float(old.get("probability_threshold", 0.4)) if stain == "AT8" else 0.5
    columns = usable if stain == "NeuN" else np.ones(len(names), bool)
    oof, scheme = out_of_fold(lambda tr, te: forest().fit(x[tr][:, columns], y[tr]).predict_proba(x[te][:, columns])[:, 1], len(y), y, groups,
                              fit_only=~own, donors=donors)
    print(f"[2/3] cross-validated ({scheme}{', on this project' + chr(39) + 's labels only' if len(extra_rows) else ''})", flush=True)

    if stain == "NeuN":
        bundle = {"classifier": forest().fit(x[:, usable], y), "feature_names": names, "usable_features": usable, "decision_mode": "context_rf",
                  "context_threshold": threshold, "addition_threshold": 0.6, "training_candidate_ids": ids, "slide_scaled": bool(scaled)}
    elif stain == "AT8":
        positives = y & data.subtype.isin(["pretangle", "mature tangle"]).to_numpy()  # typed as mature or not
        mature = data.subtype.eq("mature tangle").to_numpy()
        feature = v2_scaled.MATURE if scaled else "soma_dab_p90"  # mature tangle: the soma's DAB 90th percentile
        p90 = pd.to_numeric(data[feature], errors="coerce").to_numpy(float)
        positives &= np.isfinite(p90)  # (a scaled value needs the slide's threshold)
        enough = (positives & mature).sum() >= 3 and (positives & ~mature).sum() >= 3
        fallback = float(old.get("mature_p90_threshold", 0.16))
        if scaled and old.get("mature_feature", "soma_dab_p90") == "soma_dab_p90":  # the old cut-off in OD: in typical slide thresholds
            fallback /= float(np.nanmedian(pd.to_numeric(data.soma_dab_p90, errors="coerce") / p90)) if np.isfinite(p90).any() else 1.0
        maturity = float(max(np.unique(p90[positives & np.isfinite(p90)]), key=lambda v: (p90[positives & mature] >= v).mean()
                             + (p90[positives & ~mature] < v).mean())) if enough else fallback
        bundle = {"classifier": forest().fit(x, y), "feature_names": names, "probability_threshold": threshold, "nms_radius_um": 12.0,
                  "mature_p90_threshold": maturity, "mature_feature": feature, "context_um": 55.0, "training_n": int(len(y)),
                  "training_positive_n": int(y.sum()), "slide_scaled": bool(scaled)}
    else:
        plaques = y & (data.equivalent_diameter_um.to_numpy(float) >= MORPHOTYPE_MINIMUM_DIAMETER_UM)
        typed = data.subtype.isin(["compact", "diffuse"]).to_numpy()
        compact = data.subtype.eq("compact").to_numpy()
        enough = (plaques & typed & compact).sum() >= 5 and (plaques & typed & ~compact).sum() >= 5
        morph = morphotype_threshold(data.normalized_inner_dab.to_numpy(float)[plaques & typed], compact[plaques & typed]) if enough \
            else float(old.get("morphotype_threshold", 1.5))
        bundle = {"identity_classifier": forest().fit(x, y), "identity_feature_names": names, "identity_threshold": threshold,
                  "slide_scaled": bool(scaled),
                  "identity_training_candidate_ids": ids, "morphotype_feature": "inner_mean_dab_od / slide 6E10 calibration threshold",
                  "morphotype_threshold": float(morph), "morphotype_minimum_diameter_um": MORPHOTYPE_MINIMUM_DIAMETER_UM,
                  "morphotype_training_n": int((plaques & typed).sum()) if enough else int(old.get("morphotype_training_n", 0))}

    folder = project.output("trained_models") / f"{stain}_{time.strftime('%Y%m%d-%H%M%S')}"
    folder.mkdir(parents=True)
    dump(bundle, folder / "bundle.joblib")
    report = {"stain": stain, "created": time.time(), "validation": scheme, "threshold": threshold, "sources": sources.to_dict(),
              "slide_scaled": bool(scaled),
              "new_model": metrics(y[own], oof[own], threshold), "current_model": metrics(y_own, current, threshold) if current is not None else None,
              "current_model_path": project.relative(project.model(MODEL_KEY[stain])),
              "sessions": trained_on(project, stain, sessions), "person_added": person_added(study, oof[own], threshold),
              "simple_mapping": study.attrs.get("simple_mapping"),  # how simple labels mapped onto the classes (None: none used)
              # other projects' labels the model was also fitted on (never tested; the figures above are this project's)
              "extra_projects": extra_used, "fitted_on": {"objects": int(len(y)), "positives": int(y.sum()), "this_project": int(own.sum()),
                                                          "extra_projects": int((~own).sum())}}
    (folder / "report.json").write_text(json.dumps(report, indent=1))
    new = report["new_model"]
    print(f"[3/3] new model: AUC {new['auc']:.2f}, precision {new['precision']:.2f}, recall {new['recall']:.2f} -> {project.relative(folder)}", flush=True)
    return folder


def trained_on(project: Project, stain: str, sessions: list[str] | None, mapping_from: Project | None = None) -> list[dict]:
    """The sessions whose labels a training used (every session with labels of this stain when None): name, title and
    how many objects each labelled, for the model's report (simple labels counted through `mapping_from`'s mapping)."""
    root = project.output("reviews")
    titles = {name: label_sessions.meta(root, name).get("title", name) for name in label_sessions.names(root)}
    used = label_sessions.label_sets(label_sessions.in_sessions(excluded_areas.drop_excluded(project, store.current(root)), sessions), stain, titles,
                                     simple=simple.of(mapping_from or project, stain))
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
