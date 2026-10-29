"""Refit the continuous type scores (`stainid.v2.scores`) on a project's typed labels and print the terms and bin cuts
to put in the module (and `v2.scores.<stain>.bins`). Reads only: the label log, the candidate store and the active
models; no cache or table of the project is written.

    python tools/type_scores.py [--project .] [--labels ai|all]

The fit: the labelled tau+ neurons typed mature tangle or pretangle, and the labelled plaques of at least 15 µm typed
compact or diffuse, joined to their candidates; each feature of `scores.features` divided by its interquartile range
over the accepted (typed) objects and centred on their median (the DAB term on stage 4's cut, 0); a logistic regression
with non-negative weights (L2 penalty 1); a weight per unit of the feature = the fitted weight / the range. Leave-one-TMA-
out AUC and the share of accepted objects whose score falls on the same side of 0.5 as their stage 4 type are printed.
The bin cuts are the scores where the fitted model (its intercept added back) gives the classes 2:1 odds either way.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

POSITIVE = {"AT8": "mature tangle", "6E10": "compact"}


def fit(x: np.ndarray, y: np.ndarray, penalty: float = 1.0) -> np.ndarray:
    """Intercept and non-negative weights of an L2-penalised logistic regression."""
    from scipy.optimize import minimize

    n, k = x.shape

    def loss(p):
        z = p[0] + x @ p[1:]
        return (np.logaddexp(0, z) - y * z).sum() / n + penalty * (p[1:] ** 2).sum() / n

    def grad(p):
        r = 1 / (1 + np.exp(-(p[0] + x @ p[1:]))) - y
        return np.r_[r.sum() / n, x.T @ r / n + 2 * penalty * p[1:] / n]

    return minimize(loss, np.zeros(k + 1), jac=grad, method="L-BFGS-B", bounds=[(None, None)] + [(0, None)] * k).x


def labelled(project, stain: str, which: str) -> pd.DataFrame:
    from stainid import disk_cache, exclusions
    from stainid.exclusions import drop_excluded
    from stainid.labels import candidate_store, decisions, frames, store
    from stainid.sampling.tiles import known_tiles

    disk_cache.write = lambda *a, **k: None  # read only: no matching index nor dropped-label answer is saved
    candidate_store.save_index = lambda *a, **k: None
    exclusions._save_drops = lambda *a, **k: None
    labels = store.current(project.output("reviews"))
    if which == "ai":
        labels = labels[labels.source.map(store.is_ai_label)]
    tiles = known_tiles(project)
    labels = frames.in_field_frame(drop_excluded(project, labels), tiles, project.context_px)
    truth = decisions.consensus(project, stain, labels, tiles)
    return truth[truth.subtype.isin(TYPES[stain]) & truth.candidate.notna()]


TYPES = {"AT8": ("mature tangle", "pretangle"), "6E10": ("compact", "diffuse")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project", default=".")
    parser.add_argument("--labels", choices=("ai", "all"), default="ai", help="the AI-made labels only (default) or every label")
    args = parser.parse_args()
    from sklearn.metrics import roc_auc_score

    from stainid import candidates as store
    from stainid.project import load_project
    from stainid.v2 import scores
    from stainid.v2.classify import load_model

    project = load_project(args.project)
    for stain, score in scores.SCORES.items():
        bundle = load_model(project, stain).bundle
        names = [t.name for t in score.terms]
        columns = ["candidate_id", "core_id", "decision", "subtype", "equivalent_diameter_um", *scores.COLUMNS[stain]]
        found = pd.concat([store.read_core(project, stain, c, columns) for c in sorted(store.cores(project, stain))], ignore_index=True)
        accepted = found[(found.decision == "accepted") & found.subtype.isin(TYPES[stain])].reset_index(drop=True)
        truth = labelled(project, stain, args.labels)
        rows = truth[["candidate", "subtype"]].rename(columns={"subtype": "label"}).merge(found, left_on="candidate", right_on="candidate_id")
        if stain == "6E10":
            rows = rows[pd.to_numeric(rows.equivalent_diameter_um, errors="coerce") >= 15.0]
        rows = rows.reset_index(drop=True)
        y = (rows.label == POSITIVE[stain]).to_numpy(float)
        reference = scores.features(stain, accepted, bundle)
        centre = reference.median()
        centre[names[0]] = 0.0
        spread = reference.quantile(0.75) - reference.quantile(0.25)
        x = ((scores.features(stain, rows, bundle)[names] - centre[names]) / spread[names]).fillna(0.0).to_numpy()
        tma = rows.core_id.astype(str).str.split("_").str[0].to_numpy()
        held = np.zeros(len(y))
        for group in np.unique(tma):
            train = tma != group
            p = fit(x[train], y[train])
            held[~train] = p[0] + x[~train] @ p[1:]
        p = fit(x, y)
        weights = p[1:] / spread[names].to_numpy()
        logit = ((reference[names] - centre[names]).fillna(0.0).to_numpy() @ weights)
        agree = ((logit >= 0) == (accepted.subtype == POSITIVE[stain]).to_numpy()).mean()
        odds = np.log(2.0)
        low, high = (1 / (1 + np.exp(-(v - p[0]))) for v in (-odds, odds))
        print(f"{stain}: {len(rows)} labelled ({int(y.sum())} {POSITIVE[stain]}), leave-one-TMA-out AUC {roc_auc_score(y, held):.3f}, "
              f"same side of 0.5 as stage 4's type: {agree:.1%} of {len(accepted)} accepted")
        for name, w, c in zip(names, weights, centre[names]):
            print(f"    Term({name!r}, {w:.3g}, {c:.3g})")
        print(f"    intercept of the labels {p[0]:.3f}; bins at {low:.2f} / {high:.2f}")


if __name__ == "__main__":
    main()
