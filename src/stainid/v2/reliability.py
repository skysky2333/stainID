"""How far each results column can be trusted (stage 5): `measure_reliability.csv`, one row per numeric measure of
`results_core.csv`, for the Results page to filter on.

- `icc`: the replicate-core intraclass correlation, the share of a measure's variance between donor-regions over the
  donor-regions with at least two cores measured (one-way random effects, ANOVA estimator with unequal group sizes,
  ICC(1); as `workspace/scripts/v2_reliability.R` fits it with `lmer(v ~ 1 + (1 | donor-region))`, floored at 0).
  Densities and counts are on the log scale, log10(x + 1). `pairs`: those donor-regions; missing under `MIN_PAIRS`.
- `unique_variance`: the share of the measure's rank variance across donor-regions that the ranks of the primary
  measures (`PRIMARY`, each but the measure itself) do not explain (least squares); low means it restates burden or
  neuron density. Missing under `MIN_REGIONS` donor-regions with every value.
- `reliable`: 1 when the ICC reaches `GATE`, the study's gate for confirmatory use (the compact-plaque fraction reaches
  0.71); 0 below, empty without an ICC. A measure under the gate still has its column: the gate only advises.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

GATE = 0.5
MIN_PAIRS = 10
MIN_REGIONS = 20
PRIMARY = ("v2_plaque_density_mm2", "plaque_area_fraction", "at8_positive_area_fraction", "tau_neuron_density_mm2",
           "thread_length_density_mm_per_mm2", "neun_profile_density_mm2")
IDS = ("core_id", "sample_region_id", "donor_id", "tma", "region", "disease_group", "technical_replicate")
NOT_MEASURES = ("white_matter_share", "neun_pale", "neun_pale_cores", "grey_too_small", "grey_too_small_cores")
COLUMNS = ["measure", "icc", "n_pairs", "unique_variance", "regions", "reliable"]


def logged(column: str) -> bool:
    return "density" in column or column.endswith("_count")


def icc(values: np.ndarray, groups: np.ndarray) -> tuple[float, int]:
    """ICC(1) of `values` grouped by `groups`, over the groups with at least two values, and how many groups."""
    frame = pd.DataFrame({"v": values, "g": groups}).dropna()
    sizes = frame.groupby("g").v.transform("size")
    frame = frame[sizes >= 2]
    k = frame.g.nunique()
    if k < 2:
        return np.nan, k
    n = len(frame)
    by = frame.groupby("g").v
    means, counts = by.mean(), by.size()
    grand = frame.v.mean()
    between = float((counts * (means - grand) ** 2).sum() / (k - 1))
    within = float(((frame.v - frame.g.map(means)) ** 2).sum() / (n - k))
    k0 = (n - (counts ** 2).sum() / n) / (k - 1)
    total = between + (k0 - 1) * within
    if total <= 0:
        return np.nan, k
    return float(max(0.0, (between - within) / total)), k


def unique_variance(frame: pd.DataFrame, column: str) -> tuple[float, int]:
    """The share of `column`'s rank variance not explained by the primary measures' ranks, and over how many rows."""
    covariates = [c for c in PRIMARY if c in frame and c != column]
    data = frame[[column, *covariates]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < MIN_REGIONS or data[column].nunique() < 3 or not covariates:
        return np.nan, len(data)
    ranks = np.column_stack([stats.rankdata(data[c]) for c in [column, *covariates]])
    y = ranks[:, 0]
    design = np.c_[np.ones(len(data)), ranks[:, 1:]]
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    total = float(np.var(y))
    return (float(np.var(y - design @ beta) / total) if total > 0 else np.nan), len(data)


def table(core: pd.DataFrame, region: pd.DataFrame | None) -> pd.DataFrame:
    rows = []
    groups = core.sample_region_id.astype(str).to_numpy() if "sample_region_id" in core else None
    for column in core.columns:
        if column in IDS or column in NOT_MEASURES or not pd.api.types.is_numeric_dtype(core[column]):
            continue
        values = pd.to_numeric(core[column], errors="coerce").replace([np.inf, -np.inf], np.nan).to_numpy(float)
        if logged(column):
            values = np.log10(np.clip(values, 0, None) + 1)
        value, pairs = icc(values, groups) if groups is not None else (np.nan, 0)
        if pairs < MIN_PAIRS or not np.isfinite(np.nanvar(values)) or np.nanvar(values) == 0:
            value = np.nan
        unique, n = unique_variance(region, column) if region is not None and column in region else (np.nan, 0)
        rows.append({"measure": column, "icc": round(value, 3) if np.isfinite(value) else np.nan, "n_pairs": pairs,
                     "unique_variance": round(unique, 3) if np.isfinite(unique) else np.nan, "regions": n,
                     "reliable": (int(value >= GATE) if np.isfinite(value) else np.nan)})
    return pd.DataFrame(rows, columns=COLUMNS)


def write(folder: Path) -> list[str]:
    """`measure_reliability.csv` beside the results tables in `folder`."""
    core_path, region_path = folder / "results_core.csv", folder / "results_donor_region.csv"
    if not core_path.exists():
        return []
    core = pd.read_csv(core_path, low_memory=False, dtype={"donor_id": str})
    region = pd.read_csv(region_path, low_memory=False, dtype={"donor_id": str}) if region_path.exists() else None
    out = table(core, region)
    out.to_csv(folder / "measure_reliability.csv", index=False)
    known = out.icc.notna().sum()
    return [f"measure_reliability: {len(out)} measures, {known} with a replicate-core ICC ({int((out.reliable == 1).sum())} at least {GATE})"]
