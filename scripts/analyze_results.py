#!/usr/bin/env python3
"""
Multi-level differential analysis for stain segmentation results.

Inputs
- Features (site-level): results_stainID/features/<TYPE>/*_sites.csv
- Image-level summary: results_stainID/summary/*_image_metrics.csv or all_types_image_metrics.csv
- Phenotypes: data/brain_summary.csv (columns: Region, Label, C, B, r, c)

Outputs
- Per-stain, per-phenotype tables with test stats and FDR q-values
- Plots for significant features after FDR filtering

Notes
- Excludes KC control at r1_c1 (first row in phenotype CSV) from all analyses and plots.
- Matches images to phenotypes using parsed r,c from image names like '<TYPE>_r2_c4'.
- Site-level aggregation per image supports: mean, sd, iqr, median.
"""

import argparse
import os
import re
import math
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats


PHENOTYPE_COLS = ["Region", "Label", "C", "B"]
EXCLUDE_RC = (1, 1)  # r1_c1 (KC control)


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Differential analysis for stain features and image-level metrics")
    p.add_argument("--out_root", default="results_stainID", help="Root directory with features/ and summary/")
    p.add_argument("--metadata", default="data/brain_summary.csv", help="Phenotype CSV path")
    p.add_argument("--types", nargs="*", help="Optional list of stain types to include (e.g., 212 213)")
    p.add_argument("--alpha", type=float, default=0.05, help="FDR threshold for significance")
    p.add_argument("--min_per_group", type=int, default=2, help="Minimum samples per group for a test to run")
    p.add_argument("--agg", nargs="*", default=["mean", "sd", "iqr", "median"],
                   help="Site-level aggregations per numeric column")
    p.add_argument("--plots_dir", default=None, help="Custom plots output dir; default: <out_root>/analysis/plots")
    return p.parse_args()


def read_metadata(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Normalize column names just in case
    # Ensure r,c are integers
    if "r" not in df.columns or "c" not in df.columns:
        raise ValueError("Metadata missing required columns 'r' and 'c'.")
    df = df.copy()
    df["r"] = pd.to_numeric(df["r"], errors="coerce").astype("Int64")
    df["c"] = pd.to_numeric(df["c"], errors="coerce").astype("Int64")
    # Drop the KC control (r1_c1)
    df = df[~((df["r"] == EXCLUDE_RC[0]) & (df["c"] == EXCLUDE_RC[1]))].copy()
    return df


def parse_image_to_rc(image: str) -> Tuple[int, int]:
    """Parse (row, col) indices from an image identifier.

    Supports multiple naming patterns:
    - '..._r2_c4...' (preferred if present)
    - '..._col02_row07...' or '..._row07_col02...'

    Returns (r, c) as 1-based integers.
    """
    # rX_cY pattern anywhere
    m = re.search(r"r(\d+)_c(\d+)", image, flags=re.IGNORECASE)
    if m:
        return int(m.group(1)), int(m.group(2))
    # Look for col/row tokens (order-agnostic)
    m_col_row = re.search(r"col(\d+).*row(\d+)", image, flags=re.IGNORECASE)
    if m_col_row:
        c = int(m_col_row.group(1))
        r = int(m_col_row.group(2))
        return r, c
    m_row_col = re.search(r"row(\d+).*col(\d+)", image, flags=re.IGNORECASE)
    if m_row_col:
        r = int(m_row_col.group(1))
        c = int(m_row_col.group(2))
        return r, c
    raise ValueError(f"Cannot parse row/col from image name: {image}")


def load_all_types_summary(out_root: str) -> pd.DataFrame:
    path = os.path.join(out_root, "summary", "all_types_image_metrics.csv")
    if not os.path.exists(path):
        # Fallback: concatenate per-type summaries
        summ_dir = os.path.join(out_root, "summary")
        parts = []
        for fn in os.listdir(summ_dir):
            if fn.endswith("_image_metrics.csv") and fn != "all_types_image_metrics.csv":
                parts.append(pd.read_csv(os.path.join(summ_dir, fn)))
        if not parts:
            raise FileNotFoundError("No summary CSVs found under summary/ directory")
        df = pd.concat(parts, ignore_index=True)
    else:
        df = pd.read_csv(path)
    # Parse r,c
    rc = df["image"].apply(parse_image_to_rc)
    df["r"] = rc.apply(lambda x: x[0])
    df["c"] = rc.apply(lambda x: x[1])
    # Drop KC control if present
    df = df[~((df["r"] == EXCLUDE_RC[0]) & (df["c"] == EXCLUDE_RC[1]))].copy()
    return df


def numeric_columns(df: pd.DataFrame, exclude: Iterable[str]) -> List[str]:
    num_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    return num_cols


def is_normalized_intensity_col(col: str) -> bool:
    """Return True if the column corresponds to normalized intensity (non-raw).

    Policy: any feature whose name contains 'intensity' but does NOT contain
    'intensity_raw' is treated as normalized and should be excluded from
    differential testing and plotting.
    """
    name = str(col).lower()
    return ("intensity" in name) and ("intensity_raw" not in name)


def aggregate_sites_for_image(df_sites: pd.DataFrame, aggs: List[str]) -> Dict[str, float]:
    # Compute aggregations for all numeric columns at site level
    out: Dict[str, float] = {}
    # Exclude identifiers and coordinates
    exclude_cols = {"image", "label", "centroid_row", "centroid_col"}
    cols = numeric_columns(df_sites, exclude_cols)
    for col in cols:
        vals = df_sites[col].values
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            for a in aggs:
                out[f"sites_{col}_{a}"] = np.nan
            continue
        for a in aggs:
            if a == "mean":
                out[f"sites_{col}_mean"] = float(np.mean(vals))
            elif a == "sd":
                out[f"sites_{col}_sd"] = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
            elif a == "iqr":
                out[f"sites_{col}_iqr"] = float(np.percentile(vals, 75) - np.percentile(vals, 25))
            elif a == "median":
                out[f"sites_{col}_median"] = float(np.median(vals))
            else:
                # Unknown aggregation: skip
                pass
    return out


def build_site_aggregates(out_root: str, t: str, aggs: List[str]) -> pd.DataFrame:
    """Return DataFrame with one row per image per channel and aggregated site-level features.
    Supports both legacy (features/<TYPE>/*.csv) and per-channel (features/<TYPE>/<channel>/*.csv).
    """
    feat_root = os.path.join(out_root, "features", str(t))
    if not os.path.isdir(feat_root):
        raise FileNotFoundError(f"Features directory not found: {feat_root}")
    rows = []
    # Detect subfolders as channels
    entries = [os.path.join(feat_root, e) for e in os.listdir(feat_root)]
    subdirs = [d for d in entries if os.path.isdir(d)]
    if subdirs:
        for ch_dir in subdirs:
            ch_name = os.path.basename(ch_dir)
            for fn in os.listdir(ch_dir):
                if not fn.endswith("_sites.csv"):
                    continue
                path = os.path.join(ch_dir, fn)
                try:
                    df_sites = pd.read_csv(path)
                except Exception:
                    continue
                if df_sites.empty or "image" not in df_sites.columns:
                    continue
                image_id = str(df_sites.loc[0, "image"])  # all rows share image id
                r, c = parse_image_to_rc(image_id)
                agg_map = aggregate_sites_for_image(df_sites, aggs)
                agg_map.update({"image": image_id, "r": r, "c": c, "channel": ch_name})
                rows.append(agg_map)
    else:
        # Legacy flat files (assume single channel)
        for fn in os.listdir(feat_root):
            if not fn.endswith("_sites.csv"):
                continue
            path = os.path.join(feat_root, fn)
            try:
                df_sites = pd.read_csv(path)
            except Exception:
                continue
            if df_sites.empty or "image" not in df_sites.columns:
                continue
            image_id = str(df_sites.loc[0, "image"])  # all rows share image id
            r, c = parse_image_to_rc(image_id)
            agg_map = aggregate_sites_for_image(df_sites, aggs)
            agg_map.update({"image": image_id, "r": r, "c": c})
            rows.append(agg_map)
    if not rows:
        return pd.DataFrame(columns=["image", "r", "c", "channel"])  # empty but mergeable
    return pd.DataFrame(rows)


def benjamini_hochberg(pvals: List[float]) -> List[float]:
    """Benjamini-Hochberg FDR correction. Returns q-values in original order.
    NaN p-values are preserved as NaN.
    """
    # Pair indices with p-values and exclude NaNs for ranking
    indexed = [(i, p) for i, p in enumerate(pvals)]
    finite = [(i, p) for i, p in indexed if p is not None and not math.isnan(p)]
    if not finite:
        return [math.nan for _ in pvals]
    finite.sort(key=lambda x: x[1])  # ascending p
    m = len(finite)
    q = [None] * len(pvals)
    prev = float("inf")
    for rank, (i, p) in enumerate(reversed(finite), start=1):
        # Iterate from largest to smallest p to enforce monotonicity
        k = m - rank + 1
        val = (p * m) / k
        if val > prev:
            val = prev
        prev = val
        q[i] = min(val, 1.0)
    # Restore NaNs
    for i, p in indexed:
        if p is None or (isinstance(p, float) and math.isnan(p)):
            q[i] = math.nan
    return q


def test_feature_by_group(values: pd.Series, groups: pd.Series, min_per_group: int = 2) -> Tuple[str, float]:
    """Return (test_name, p_value) using Welch t-test for 2 groups else Kruskal-Wallis.
    Returns ("insufficient", NaN) if requirements not met.
    """
    # Drop NaN pairs
    mask = values.notna() & groups.notna()
    if mask.sum() < 2:
        return "insufficient", float("nan")
    v = values[mask]
    g = groups[mask].astype(str)
    cats = sorted(g.unique())
    # Build lists per group
    samples = [v[g == c].values.astype(float) for c in cats]
    sizes = [arr.size for arr in samples]
    if len(samples) < 2:
        return "insufficient", float("nan")
    if any(s < min_per_group for s in sizes):
        return "insufficient", float("nan")
    try:
        if len(samples) == 2:
            p = stats.ttest_ind(samples[0], samples[1], equal_var=False, nan_policy="omit").pvalue
            return "welch_t", float(p)
        else:
            # Nonparametric across >2 groups
            p = stats.kruskal(*samples, nan_policy="omit").pvalue
            return "kruskal", float(p)
    except Exception:
        return "error", float("nan")


def compute_effect_size(values: pd.Series, groups: pd.Series) -> float:
    """Compute Cohen's d for two groups; NaN otherwise."""
    mask = values.notna() & groups.notna()
    v = values[mask]
    g = groups[mask].astype(str)
    cats = g.unique()
    if len(cats) != 2:
        return float("nan")
    a, b = cats[0], cats[1]
    x = v[g == a].values.astype(float)
    y = v[g == b].values.astype(float)
    if x.size < 2 or y.size < 2:
        return float("nan")
    # Pooled SD
    nx, ny = x.size, y.size
    sx2 = np.var(x, ddof=1)
    sy2 = np.var(y, ddof=1)
    sp = math.sqrt(((nx - 1) * sx2 + (ny - 1) * sy2) / (nx + ny - 2)) if (nx + ny - 2) > 0 else float("nan")
    if not np.isfinite(sp) or sp == 0:
        return float("nan")
    d = (np.mean(x) - np.mean(y)) / sp
    return float(d)


def sanitize(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", s)


def plot_feature(df: pd.DataFrame, x_col: str, y_col: str, title: str, out_path: str) -> None:
    ensure_dir(os.path.dirname(out_path))
    # Simple boxplot + jitter
    plt.figure(figsize=(6, 4))
    # Prepare groups
    groups = df[x_col].astype(str).values
    y = df[y_col].values.astype(float)
    cats = sorted(pd.unique(groups))
    data = [y[groups == c] for c in cats]
    plt.boxplot(data, labels=cats, showfliers=False)
    # Jitter points
    xs = []
    ys = []
    for i, cat in enumerate(cats, start=1):
        vals = y[groups == cat]
        jitter = (np.random.rand(vals.size) - 0.5) * 0.2
        xs.append(np.full_like(vals, i, dtype=float) + jitter)
        ys.append(vals)
    if xs:
        plt.plot(np.concatenate(xs), np.concatenate(ys), "o", alpha=0.6, markersize=3, color="#555555")
    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.tight_layout()
    try:
        plt.savefig(out_path, dpi=150)
    finally:
        plt.close()


def run_for_type(t: str, df_summary: pd.DataFrame, metadata: pd.DataFrame, out_root: str,
                 aggs: List[str], alpha: float, min_per_group: int, plots_dir: str) -> None:
    # Subset summary for this type
    sub = df_summary[df_summary["type"].astype(str) == str(t)].copy()
    if sub.empty:
        print(f"[warn] No summary rows for type {t}")
        return
    # Merge with metadata via r,c
    df = sub.merge(metadata, on=["r", "c"], how="left", suffixes=("", "_meta"))
    # Build site-level aggregates and merge (supports per-channel)
    df_sites = build_site_aggregates(out_root, str(t), aggs)
    if not df_sites.empty:
        on_cols = ["image", "r", "c"] + (["channel"] if "channel" in df_sites.columns and "channel" in df.columns else [])
        df = df.merge(df_sites, on=on_cols, how="left")

    # Identify feature columns to test
    exclude_cols = set(["image", "type", "threshold_used", "r", "c", "channel"] + PHENOTYPE_COLS)
    feat_cols_all = [c for c in df.columns if (c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c]))]
    # Exclude normalized intensity features; keep only raw intensity ones
    removed_intensity = [c for c in feat_cols_all if is_normalized_intensity_col(c)]
    feat_cols = [c for c in feat_cols_all if c not in removed_intensity]
    if not feat_cols:
        print(f"[warn] No numeric features to test for type {t}")
        return
    if removed_intensity:
        print(f"[info] Excluding {len(removed_intensity)} normalized intensity features for type {t}")

    # Run analyses per channel (if present), else once
    channels = sorted(df["channel"].dropna().astype(str).unique().tolist()) if "channel" in df.columns else [None]
    for ch in channels:
        if ch is not None:
            df_ch = df[df["channel"].astype(str) == ch].copy()
            tables_dir = os.path.join(out_root, "analysis", "tables", str(t), ch)
            plots_subbase = os.path.join(plots_dir, str(t), ch)
        else:
            df_ch = df.copy()
            tables_dir = os.path.join(out_root, "analysis", "tables", str(t))
            plots_subbase = os.path.join(plots_dir, str(t))
        ensure_dir(tables_dir)
        for pheno in PHENOTYPE_COLS:
            if pheno not in df_ch.columns:
                print(f"[warn] Phenotype '{pheno}' missing in metadata; skipping for type {t}")
                continue
            df_ph = df_ch[[pheno] + feat_cols + ["image"]].copy()
            df_ph = df_ph[df_ph[pheno].notna()]
            df_ph = df_ph[df_ph[pheno].astype(str) != "KC"]
            rows = []
            for col in feat_cols:
                test_name, p = test_feature_by_group(df_ph[col], df_ph[pheno], min_per_group=min_per_group)
                means = df_ph.groupby(pheno)[col].mean().to_dict()
                effect = compute_effect_size(df_ph[col], df_ph[pheno])
                rows.append({
                    "feature": col,
                    "test": test_name,
                    "p_value": p,
                    "effect_d": effect,
                    "n": int(df_ph[col].notna().sum()),
                    **{f"mean[{k}]": float(v) for k, v in means.items()},
                })
            res = pd.DataFrame(rows)
            res["q_value"] = benjamini_hochberg(res["p_value"].tolist())
            res = res.sort_values(["q_value", "p_value", "feature"], na_position="last")
            out_csv = os.path.join(tables_dir, f"{sanitize(pheno)}_diff.csv")
            res.to_csv(out_csv, index=False)
            # Plots
            sig = res[(res["q_value"].notna()) & (res["q_value"] <= alpha)]
            if sig.empty:
                continue
            plots_subdir = os.path.join(plots_subbase, sanitize(pheno))
            ensure_dir(plots_subdir)
            for _, r in sig.head(24).iterrows():
                feat = r["feature"]
                ttl = f"type {t}{' | ' + ch if ch else ''} | {pheno} | {feat} (q={r['q_value']:.3g})"
                out_png = os.path.join(plots_subdir, f"{sanitize(feat)}.png")
                try:
                    plot_feature(df_ph[[pheno, feat]].dropna(), pheno, feat, ttl, out_png)
                except Exception:
                    pass


def main() -> None:
    args = parse_args()
    out_root = args.out_root
    ensure_dir(out_root)

    # Load metadata and summary
    meta = read_metadata(args.metadata)
    summ = load_all_types_summary(out_root)

    # Select types
    all_types = sorted(summ["type"].astype(str).unique().tolist())
    sel_types = [str(t) for t in (args.types if args.types else all_types)]

    plots_dir = args.plots_dir or os.path.join(out_root, "analysis", "plots")
    ensure_dir(plots_dir)

    print(f"Running diff analysis for types: {', '.join(sel_types)}")
    for t in sel_types:
        print(f"  Processing type {t}")
        run_for_type(
            t=t,
            df_summary=summ,
            metadata=meta,
            out_root=out_root,
            aggs=args.agg,
            alpha=args.alpha,
            min_per_group=args.min_per_group,
            plots_dir=plots_dir,
        )
    print("Done. Tables under <out_root>/analysis/tables and plots under <out_root>/analysis/plots.")


if __name__ == "__main__":
    main()
