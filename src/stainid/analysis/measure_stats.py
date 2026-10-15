"""How reproducible each measure is and how it differs between diagnosis groups (the web app's Results page).

Reproducibility is the replicate-core ICC: how well the two cores of the same donor and region agree. Group differences
are effects in standard deviations with p values and false discovery rates (Benjamini-Hochberg). Both come from the
study's own tables when the project has them, checked against the numbers on screen; otherwise they are computed here
from the results tables, the group test as a quick test that is not the study's main analysis.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats

from stainid.analysis.burden_screen import icc_oneway

# Densities (and the plaque outline area fraction) are compared on log10(x + 1), as the study's ICC and contrast scripts do.
LOG_SCALE = re.compile(r"density|mask_area_fraction")
MIN_PAIRS = 10
ICC_AGREES = 0.03  # a table's ICC describes a column when it is this close to the ICC of the column's own core values
MIN_PER_GROUP = 5

# Log-scale entries of the whole-core reliability table and the column each one is the log of. The report quotes these.
LOG_ALIASES = {"log_neun_density": "neun_profile_density_mm2", "log_amyloid": "amyloid_positive_area_fraction",
               "log_tau": "tau_positive_area_fraction"}
TERMS = {"region": "brain region", "tma": "TMA", "la": "amyloid burden", "age": "age", "e4_count": "APOE ε4"}
FULL_MODEL = "is_ad + la + age + e4_count + region + tma"
QUICK = "quick"
QUICK_TERMS = ("region", "tma")
# Which pairs of groups to compare, in order, by the kind of group (see setup.naming.FAMILIES).
PREFERRED = (("AD", "ASYMP"), ("AD", "CT"), ("ASYMP", "CT"))


@dataclass(frozen=True)
class StudyContrasts:
    """A table of the study's group contrasts (written by its R scripts) and the tables its measures were read from, in
    the order the script merged them (a measure comes from the first that has it)."""
    file: str
    groups: tuple[str, str]  # kinds of group: effect = first − second
    inputs: tuple[str, ...]
    model: str = ""  # the model, for tables without a `model` column
    within: str = "Within TMAs that hold both groups."


STUDY_CONTRASTS = (
    StudyContrasts("cohort_morphology_donor_region_wide_within_tma_contrasts.csv", ("AD", "ASYMP"),
                   ("cohort_morphology_donor_region_wide.csv", "cohort_v3_donor_region.csv", "cohort_object_masks_donor_region.csv",
                    "spatial_patterning_donor_region.csv", "neuronal_nuclei_donor_region.csv", "neuronal_nuclear_size_donor_region.csv")),
    StudyContrasts("neuron_size_classes_donor_region_contrasts.csv", ("AD", "ASYMP"), ("neuron_size_classes_donor_region.csv",), FULL_MODEL),
    StudyContrasts("neun_large_cells_donor_region_contrasts.csv", ("AD", "ASYMP"), ("neun_large_cells_donor_region.csv",), FULL_MODEL),
)


def log_scale(column: str, values: pd.Series | np.ndarray) -> bool:
    return bool(LOG_SCALE.search(column)) and bool(np.nanmin(values) >= 0)


def scaled(column: str, values: pd.Series) -> pd.Series:
    return np.log10(values + 1) if log_scale(column, values) else values


# ---------------------------------------------------------------- reproducibility


def replicate_pairs(core: pd.DataFrame, column: str) -> np.ndarray:
    """The values of the first two cores of every donor-region that has two with a value."""
    if "sample_region_id" not in core or column not in core:
        return np.empty((0, 2))
    order = [c for c in ("sample_region_id", "technical_replicate", "core_id") if c in core]
    d = core[["sample_region_id", column] + [c for c in order[1:]]].copy()
    d[column] = pd.to_numeric(d[column], errors="coerce")
    d = d[np.isfinite(d[column])].sort_values(order, kind="stable")
    d = d.assign(k=d.groupby("sample_region_id").cumcount())
    wide = d[d.k < 2].pivot(index="sample_region_id", columns="k", values=column)
    return wide.dropna().to_numpy(dtype=float) if wide.shape[1] == 2 else np.empty((0, 2))


def computed_icc(core: pd.DataFrame, column: str) -> dict | None:
    """ICC(1) of a column from its replicate cores: on the raw values and on log10(x + 1) for non-negative ones."""
    pairs = replicate_pairs(core, column)
    if len(pairs) < MIN_PAIRS or np.var(pairs) == 0:
        return None
    raw = icc_oneway(pairs)
    log = icc_oneway(np.log10(pairs + 1)) if pairs.min() >= 0 and np.var(np.log10(pairs + 1)) > 0 else float("nan")
    use_log = log_scale(column, pairs) and np.isfinite(log)
    value = log if use_log else raw
    if not np.isfinite(value):
        return None
    return {"icc": float(value), "pairs": int(len(pairs)), "scale": "log" if use_log else "raw", "_raw": float(raw), "_log": float(log)}


def study_icc(paths: list[Path], relative: Callable[[Path], str], primary: Path | None = None) -> dict[str, list[dict]]:
    """Every ICC the project's reliability tables give, by column name: the log-scale entries first, then those of the
    `primary` table (the study's main reliability table, of its tile set), then the others in the order given."""
    found: dict[str, list[dict]] = {}
    for path in paths:
        try:
            table = pd.read_csv(path)
        except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
            continue
        if not {"feature", "icc"} <= set(table.columns):
            continue
        for row in table.itertuples():
            if not np.isfinite(row.icc):
                continue
            column = LOG_ALIASES.get(row.feature, row.feature)
            found.setdefault(column, []).append({"icc": float(row.icc), "pairs": int(getattr(row, "pairs", 0) or 0), "file": relative(path),
                                                 "scale": "log" if row.feature in LOG_ALIASES else None, "primary": path == primary})
    for entries in found.values():
        entries.sort(key=lambda e: (e["scale"] != "log", not e["primary"]))  # stable: otherwise in the order given
    return found


def reliability(columns: list[str], core: pd.DataFrame | None, tables: dict[str, list[dict]]) -> dict[str, dict]:
    """The ICC of every column: the study's value when one of its tables has the column and (when the core values are at
    hand) agrees with the column's own replicate cores, otherwise the ICC computed from those cores."""
    out = {}
    for column in columns:
        own = computed_icc(core, column) if core is not None else None
        candidates = tables.get(column, [])
        if own:
            candidates = [c for c in candidates if min(abs(c["icc"] - own["_raw"]), abs(c["icc"] - own["_log"]) if np.isfinite(own["_log"]) else 1) <= ICC_AGREES]
        elif candidates and not (candidates[0]["scale"] == "log" or candidates[0].get("primary")) \
                and max(c["icc"] for c in candidates) - min(c["icc"] for c in candidates) > ICC_AGREES:
            candidates = []  # tables disagree and nothing tells which one is this column
        if candidates:
            best = candidates[0]
            scale = best["scale"] or (own["scale"] if own else None)
            out[column] = {"icc": round(best["icc"], 3), "pairs": best["pairs"] or (own or {}).get("pairs", 0), "source": "study",
                           "file": best["file"], "scale": scale}
        elif own:
            out[column] = {"icc": round(own["icc"], 3), "pairs": own["pairs"], "source": "computed", "scale": own["scale"]}
    return out


# ---------------------------------------------------------------- group differences


def fdr(p: pd.Series) -> pd.Series:
    """Benjamini-Hochberg adjusted p values (NaN stays NaN)."""
    valid = p.dropna()
    if valid.empty:
        return p * np.nan
    order = valid.sort_values()
    ranked = order * len(order) / np.arange(1, len(order) + 1)
    adjusted = np.minimum.accumulate(ranked[::-1])[::-1].clip(upper=1)
    return adjusted.reindex(p.index)


def comparisons(present: list[str], kinds: dict[str, str]) -> list[tuple[str, str]]:
    """Pairs of groups to compare (effect = first − second): AD vs ASYMAD, AD vs control, ASYMAD vs control when the
    project has them (`kinds` gives the project's code for each kind), then any other pair, later group minus earlier."""
    pairs = [(kinds[a], kinds[b]) for a, b in PREFERRED if kinds.get(a) in present and kinds.get(b) in present]
    pairs += [(b, a) for a, b in combinations(present, 2) if (b, a) not in pairs and (a, b) not in pairs]
    return pairs


def within_tmas(frame: pd.DataFrame, first: str, second: str) -> pd.DataFrame:
    """Rows of the two groups, on the TMAs that hold both."""
    d = frame[frame.disease_group.isin([first, second])]
    if "tma" in d:
        both = d.groupby("tma").disease_group.nunique()
        d = d[d.tma.isin(both[both == 2].index)]
    return d


def quick_test(frame: pd.DataFrame, column: str, first: str, second: str) -> dict | None:
    """first − second in SD of the measure (log10(x + 1) for densities), from a linear model with brain region and TMA,
    on the TMAs that hold both groups. Standard errors allow for the two regions of a donor being alike (clustered by
    donor), p from a t distribution with donors − 1 degrees of freedom."""
    d = within_tmas(frame, first, second)
    d = d.assign(_y=pd.to_numeric(d[column], errors="coerce"))
    d = d[np.isfinite(d._y)]
    is_first = (d.disease_group == first).astype(float)
    if min(is_first.sum(), len(d) - is_first.sum()) < MIN_PER_GROUP:
        return None
    y = scaled(column, d._y)
    if y.std() == 0 or not np.isfinite(y.std()):
        return None
    y = ((y - y.mean()) / y.std()).to_numpy()
    covariates = [c for c in QUICK_TERMS if c in d and d[c].nunique() > 1]
    X = pd.concat([pd.Series(1.0, index=d.index, name="const"), is_first.rename("group")]
                  + [pd.get_dummies(d[c].astype(str), prefix=c, drop_first=True, dtype=float) for c in covariates], axis=1).to_numpy()
    n, k = X.shape
    if np.linalg.matrix_rank(X) < k or n <= k + 1:
        return None
    bread = np.linalg.inv(X.T @ X)
    beta = bread @ X.T @ y
    residual = y - X @ beta
    clusters = d.donor_id.astype(str).to_numpy() if "donor_id" in d else np.arange(n).astype(str)
    groups = pd.unique(clusters)
    if len(groups) < 3:
        return None
    meat = np.zeros((k, k))
    for g in groups:
        score = X[clusters == g].T @ residual[clusters == g]
        meat += np.outer(score, score)
    scale = len(groups) / (len(groups) - 1) * (n - 1) / (n - k)
    se = float(np.sqrt((scale * bread @ meat @ bread)[1, 1]))
    if not se > 0:
        return None
    t = float(beta[1] / se)
    p = float(2 * stats.t.sf(abs(t), df=len(groups) - 1))
    return {"effect": round(float(beta[1]), 3), "p": p, "n": int(n), "n_first": int(is_first.sum()), "n_second": int(n - is_first.sum()),
            "terms": covariates}


def model_terms(model: str) -> list[str]:
    """The adjustment terms of a model formula ("is_ad + la + region + tma" -> ["la", "region", "tma"])."""
    return [t.strip() for t in model.split("+") if t.strip() and not t.strip().startswith("is_")]


def adjustment(terms: list[str]) -> str:
    """"brain region, TMA and age" for ["region", "tma", "age"]."""
    words = [TERMS.get(t, t) for t in terms]
    if len(words) < 2:
        return words[0] if words else "nothing"
    return ", ".join(words[:-1]) + " and " + words[-1]


def same_values(shown: pd.DataFrame, source: pd.DataFrame, column: str) -> bool:
    """Whether a column holds the same numbers in two donor-region tables (the study measured what is on screen)."""
    if "sample_region_id" not in shown or "sample_region_id" not in source or column not in source:
        return False
    a = pd.to_numeric(shown.drop_duplicates("sample_region_id").set_index("sample_region_id")[column], errors="coerce")
    b = pd.to_numeric(source.drop_duplicates("sample_region_id").set_index("sample_region_id")[column], errors="coerce")
    a, b = a.align(b, join="inner")
    both = np.isfinite(a) & np.isfinite(b)
    return int(both.sum()) >= MIN_PER_GROUP and bool(np.allclose(a[both], b[both], rtol=1e-6, atol=1e-9))


def study_tests(shown: pd.DataFrame, spec: StudyContrasts, table: pd.DataFrame, inputs: list[pd.DataFrame | None]) -> dict[str, dict[str, dict]]:
    """The study's tests of the columns on screen, by column and model, each with its FDR over the study table's
    measures (for that model). A column is used only when the study read it from a table with the same numbers."""
    if not {"feature", "std_effect", "p"} <= set(table.columns):
        return {}
    table = table.assign(model=table["model"] if "model" in table else spec.model)
    table = table[np.isfinite(pd.to_numeric(table.p, errors="coerce"))]
    out: dict[str, dict[str, dict]] = {}
    for model, rows in table.groupby("model", sort=False):
        rows = rows.assign(fdr=fdr(rows.p.astype(float)))
        terms = model_terms(str(model))
        for row in rows.itertuples():
            column = row.feature
            if column not in shown:
                continue
            source = next((frame for frame in inputs if frame is not None and column in frame), None)
            if source is None or not same_values(shown, source, column):
                continue
            n = int(row.n) if hasattr(row, "n") and pd.notna(row.n) else None
            n_first = int(row.n_ad) if hasattr(row, "n_ad") and pd.notna(row.n_ad) else None
            out.setdefault(column, {})[str(model)] = {
                "effect": round(float(row.std_effect), 3), "p": float(row.p), "fdr": float(row.fdr), "n": n, "n_first": n_first,
                "n_second": n - n_first if n is not None and n_first is not None else None, "terms": terms}
    return out


def quick_tests(shown: pd.DataFrame, columns: list[str], first: str, second: str) -> dict[str, dict]:
    tests = {c: t for c in columns if (t := quick_test(shown, c, first, second))}
    adjusted = fdr(pd.Series({c: t["p"] for c, t in tests.items()}, dtype=float))
    for column, test in tests.items():
        test["fdr"] = float(adjusted[column])
    return tests


def model_info(key: str, terms: list[str], source: str, default: bool) -> dict:
    return {"key": key, "terms": terms, "adjusted": adjustment(terms), "source": source, "default": default}


__all__ = [
    "STUDY_CONTRASTS", "QUICK", "StudyContrasts", "adjustment", "comparisons", "computed_icc", "fdr", "model_info",
    "model_terms", "quick_test", "quick_tests", "reliability", "same_values", "study_icc", "study_tests", "within_tmas",
]
