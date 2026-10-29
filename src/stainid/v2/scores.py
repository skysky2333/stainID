"""Continuous type scores: tau maturity (pretangle → mature tangle) and plaque compactness (diffuse → compact), each
from 0 to 1, with three bins, from the stored per-object features of stages 2–4 (no image is read, nothing is
re-detected). Stage 5 writes them per object (`measures/type_scores_<stain>.csv`) and summarises them per core and
donor-region (`results_<level>.csv`, `readout_names`). The two-class types of stage 4 (`subtype`) are not changed.

Each score is a logistic function of a few documented features, each oriented so that more means more mature (more
compact) and with a weight that is never negative, so the score is monotone in every feature:

    score = 1 / (1 + exp(-Σ weight × (feature - centre)))

The DAB term is the feature stage 4's type rule cuts, as the log of its ratio to the active model's cut-off (AT8
`mature_p90_threshold` on the soma's DAB 90th percentile, 6E10 `morphotype_threshold` on the inner mean DAB in slide
thresholds), so it is 0 at the cut; the other features are centred on the study's accepted objects' medians. An
object with those median features therefore scores exactly 0.5 at stage 4's cut, and 0.5 is where the score's two
classes meet the existing ones (on the study 94.8 % of accepted tau+ neurons and 95.2 % of typed plaques fall on the
same side). A retrained model moves the cut and the score with it.

The weights were fitted once on the study's typed labels (label set "ai": 412 tau+ neurons, 387 plaques of at least
15 µm) by a logistic regression with non-negative weights on features divided by their interquartile range over the
accepted objects (`tools/type_scores.py` refits them; docs/pipeline_v2.md has the validation); features the fit gave
no weight were left out (AT8: the soma's stained share and the share of it a nucleus covers; 6E10: solidity). A
missing feature counts as its centre.

Bins (`v2.scores.<stain>.bins`, two cut points): the score where the label set's own calibration (the fitted
intercept added back) gives the labelled classes 2:1 odds either way, rounded to 0.01; between them the labelled
classes are within 2:1 of each other, the intermediate bin.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Term:
    name: str
    weight: float  # per unit of the feature
    centre: float  # the accepted objects' median (0 for the DAB term: the cut)
    meaning: str


@dataclass(frozen=True)
class Score:
    stain: str
    column: str  # per-object column and readout prefix
    terms: tuple[Term, ...]
    bins: tuple[str, str, str]  # low, middle, high
    min_count: int  # the fewest scored objects a distribution or share is given for (as the existing fractions)


TAU = Score("AT8", "tau_maturity", (
    Term("dab", 3.70, 0.0, "ln(soma DAB 90th percentile / the model's mature-tangle cut-off)"),
    Term("contrast", 0.281, 2.83, "soma mean DAB minus its 2-10 µm ring's median, in slide thresholds (m_normalized_contrast)"),
    Term("texture", 1.43, 0.597, "the soma's DAB coefficient of variation: a fibrillar, uneven fill (m_soma_dab_cv)"),
), ("pretangle", "intermediate", "mature"), 5)

PLAQUE = Score("6E10", "plaque_compactness", (
    Term("core_density", 0.749, 0.0, "ln(inner mean DAB in slide thresholds / the model's morphotype cut-off)"),
    Term("core_contrast", 1.47, 0.777, "inner minus outer mean DAB (radial profile), in slide thresholds (s_radial_dab_contrast)"),
    Term("core_fraction", 5.18, 0.197, "share of the deposit at or above the morphotype cut-off (smoothed DAB): the dense-core fraction"),
    Term("edge_contrast", 1.39, 0.717, "deposit mean DAB minus its surrounding ring's, in slide thresholds (boundary_contrast): edge sharpness"),
), ("diffuse", "intermediate", "compact"), 10)

SCORES = {"AT8": TAU, "6E10": PLAQUE}
TYPED = {"AT8": ("mature tangle", "pretangle"), "6E10": ("compact", "diffuse")}  # the subtypes a bin is given for
COLUMNS = {"AT8": ["m_soma_dab_p90", "s_soma_dab_p90", "m_normalized_contrast", "m_soma_dab_cv"],
           "6E10": ["normalized_inner_dab", "core_dab_q", "s_radial_dab_contrast", "m_radial_dab_contrast", "m_inner_mean_dab_od",
                    "boundary_contrast"]}


def _number(frame: pd.DataFrame, name: str) -> np.ndarray:
    return pd.to_numeric(frame[name], errors="coerce").to_numpy(float) if name in frame else np.full(len(frame), np.nan)


def _log_ratio(values: np.ndarray, cut: float) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(np.where(values > 0, values, np.nan) / cut) if cut > 0 and np.isfinite(cut) else np.full(len(values), np.nan)


def core_fractions(quantiles, level: float) -> np.ndarray:
    """The share of each deposit at or above `level` from its stored smoothed-DAB quantiles (`measures.fraction_at_least`)."""
    from stainid.v2.measures import fraction_at_least  # noqa: PLC0415 (measures imports this module)

    return np.array([fraction_at_least(q, level) for q in quantiles], float)


def features(stain: str, frame: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """The score's features of each object (NaN where not stored)."""
    if stain == "AT8":
        feature = str(bundle.get("mature_feature", "soma_dab_p90"))
        p90 = _number(frame, feature if feature.startswith("s_") else f"m_{feature}")
        return pd.DataFrame({"dab": _log_ratio(p90, float(bundle.get("mature_p90_threshold", np.nan))),
                             "contrast": _number(frame, "m_normalized_contrast"), "texture": _number(frame, "m_soma_dab_cv")}, index=frame.index)
    if stain == "6E10":
        cut = float(bundle.get("morphotype_threshold", np.nan))
        inner = _number(frame, "normalized_inner_dab")
        contrast = _number(frame, "s_radial_dab_contrast")
        with np.errstate(divide="ignore", invalid="ignore"):  # a store without s_ inputs: the OD contrast over the slide threshold
            contrast = np.where(np.isfinite(contrast), contrast, _number(frame, "m_radial_dab_contrast") * inner / _number(frame, "m_inner_mean_dab_od"))
        quantiles = frame["core_dab_q"] if "core_dab_q" in frame else pd.Series("", index=frame.index)
        return pd.DataFrame({"core_density": _log_ratio(inner, cut), "core_contrast": contrast, "core_fraction": core_fractions(quantiles, cut),
                             "edge_contrast": _number(frame, "boundary_contrast")}, index=frame.index)
    raise ValueError(f"no type score for {stain}")


def logit(score: Score, values: pd.DataFrame) -> np.ndarray:
    total = np.zeros(len(values))
    for term in score.terms:
        part = term.weight * (values[term.name].to_numpy(float) - term.centre)
        total += np.where(np.isfinite(part), part, 0.0)  # a missing feature counts as its centre
    return total


def compute(stain: str, frame: pd.DataFrame, bundle: dict) -> np.ndarray:
    """The score (0-1) of each object; NaN where its DAB term is missing (no cut-off, no soma or core level)."""
    score = SCORES[stain]
    values = features(stain, frame, bundle)
    out = 1.0 / (1.0 + np.exp(-logit(score, values)))
    anchor = values[score.terms[0].name].to_numpy(float)
    return np.where(np.isfinite(anchor), out, np.nan)


def cuts(project, stain: str) -> tuple[float, float]:
    """The two bin cut points of a stain (`v2.scores.<stain>.bins`)."""
    from stainid.v2.params import params  # noqa: PLC0415

    value = params(project)["scores"][stain]["bins"]
    low, high = (float(v) for v in value)
    if not 0.0 <= low <= high <= 1.0:
        raise ValueError(f"v2.scores.{stain}.bins must be two scores 0 <= low <= high <= 1, not {value!r}")
    return low, high


def bins(stain: str, values: np.ndarray, between: tuple[float, float]) -> np.ndarray:
    """Each score's bin name ("" for no score): low below the first cut, high at or above the second."""
    names = SCORES[stain].bins
    values = np.asarray(values, float)
    out = np.where(values >= between[1], names[2], np.where(values >= between[0], names[1], names[0])).astype(object)
    out[~np.isfinite(values)] = ""
    return out


def object_rows(stain: str, found: pd.DataFrame, bundle: dict, between: tuple[float, float], meta: dict) -> list[dict]:
    """One row per accepted object: its ids, stage 4's type, the score and the bin (only for the typed subtypes: a 6E10
    plaque under 15 µm, `small`, has a score and no bin, as it has no compact / diffuse type)."""
    accepted = found[found.decision == "accepted"]
    if accepted.empty:
        return []
    score = SCORES[stain]
    values = compute(stain, accepted, bundle)
    named = bins(stain, values, between)
    typed = accepted.subtype.isin(TYPED[stain]).to_numpy()
    named[~typed] = ""
    rows = []
    for tile_id, candidate_id, subtype, value, name in zip(accepted.tile_id, accepted.candidate_id, accepted.subtype, values, named):
        rows.append({**meta.get(tile_id, {}), "tile_id": tile_id, "candidate_id": candidate_id, "subtype": subtype,
                     f"{score.column}_score": float(value), f"{score.column}_bin": name})
    return rows


def readout_names(stain: str) -> list[str]:
    """The per-core and per-donor-region columns of one stain's score, in order."""
    score = SCORES[stain]
    head = [f"{score.column}_scored_count", f"{score.column}_score_median", f"{score.column}_score_iqr", f"{score.column}_score_mean"]
    return head + [f"{score.column}_{b}_fraction" for b in score.bins] + [f"{score.column}_{b}_density_mm2" for b in score.bins]


def readouts(stain: str, rows: list[dict], area_mm2: float) -> dict:
    """One core's or donor-region's score readouts from its objects' rows (`object_rows`) and its analysed area: the
    binned objects' score median, interquartile range and mean and each bin's share (given for at least `min_count`
    objects, as the existing type fractions), and each bin's density per mm² of the area."""
    score = SCORES[stain]
    binned = [r for r in rows if r[f"{score.column}_bin"]]
    values = np.array([r[f"{score.column}_score"] for r in binned], float)
    enough = len(values) >= score.min_count
    nan = float("nan")
    out = {f"{score.column}_scored_count": len(values),
           f"{score.column}_score_median": float(np.median(values)) if enough else nan,
           f"{score.column}_score_iqr": float(np.subtract(*np.quantile(values, [0.75, 0.25]))) if enough else nan,
           f"{score.column}_score_mean": float(np.mean(values)) if enough else nan}
    counts = {b: sum(r[f"{score.column}_bin"] == b for r in binned) for b in score.bins}
    out |= {f"{score.column}_{b}_fraction": counts[b] / len(values) if enough else nan for b in score.bins}
    out |= {f"{score.column}_{b}_density_mm2": counts[b] / area_mm2 if area_mm2 else nan for b in score.bins}
    return out


__all__ = ["COLUMNS", "PLAQUE", "SCORES", "TAU", "TYPED", "bins", "compute", "cuts", "features", "object_rows", "readout_names", "readouts"]
