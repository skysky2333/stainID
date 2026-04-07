import math
import warnings
from typing import Dict, Iterable, Tuple

import cv2
import numpy as np
import pandas as pd
from scipy import stats


def finite_values(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).ravel()
    return arr[np.isfinite(arr)]


def safe_mean(values: Iterable[float]) -> float:
    arr = finite_values(values)
    return float(np.mean(arr)) if arr.size else float("nan")


def safe_std(values: Iterable[float], ddof: int = 1) -> float:
    arr = finite_values(values)
    if arr.size == 0:
        return float("nan")
    if arr.size == 1:
        return 0.0
    return float(np.std(arr, ddof=ddof))


def safe_median(values: Iterable[float]) -> float:
    arr = finite_values(values)
    return float(np.median(arr)) if arr.size else float("nan")


def safe_iqr(values: Iterable[float]) -> float:
    arr = finite_values(values)
    if not arr.size:
        return float("nan")
    return float(np.percentile(arr, 75) - np.percentile(arr, 25))


def safe_percentile(values: Iterable[float], q: float) -> float:
    arr = finite_values(values)
    return float(np.percentile(arr, q)) if arr.size else float("nan")


def safe_min(values: Iterable[float]) -> float:
    arr = finite_values(values)
    return float(np.min(arr)) if arr.size else float("nan")


def safe_max(values: Iterable[float]) -> float:
    arr = finite_values(values)
    return float(np.max(arr)) if arr.size else float("nan")


def safe_skew(values: Iterable[float]) -> float:
    arr = finite_values(values)
    if arr.size < 3:
        return float("nan")
    if np.ptp(arr) < 1e-12:
        return float("nan")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return float(stats.skew(arr, bias=False))
    except Exception:
        return float("nan")


def safe_kurtosis(values: Iterable[float]) -> float:
    arr = finite_values(values)
    if arr.size < 4:
        return float("nan")
    if np.ptp(arr) < 1e-12:
        return float("nan")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return float(stats.kurtosis(arr, fisher=True, bias=False))
    except Exception:
        return float("nan")


def safe_cv(values: Iterable[float]) -> float:
    arr = finite_values(values)
    if arr.size == 0:
        return float("nan")
    mean = float(np.mean(arr))
    if abs(mean) < 1e-12:
        return float("nan")
    return float(np.std(arr, ddof=1) / mean) if arr.size > 1 else 0.0


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    keep = np.isfinite(x) & np.isfinite(y)
    if keep.sum() < 3:
        return float("nan")
    xx = x[keep]
    yy = y[keep]
    if np.std(xx) < 1e-12 or np.std(yy) < 1e-12:
        return float("nan")
    return float(np.corrcoef(xx, yy)[0, 1])


def aggregate_numeric_columns(df: pd.DataFrame, exclude: Iterable[str] = ()) -> Dict[str, float]:
    out: Dict[str, float] = {}
    skip = set(exclude)
    for col in df.columns:
        if col in skip or not pd.api.types.is_numeric_dtype(df[col]):
            continue
        vals = finite_values(df[col].values)
        out[f"{col}_mean"] = safe_mean(vals)
        out[f"{col}_sd"] = safe_std(vals)
        out[f"{col}_iqr"] = safe_iqr(vals)
        out[f"{col}_median"] = safe_median(vals)
    return out


def robust_normalize(image: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    if mask is None or not np.any(mask):
        vals = arr.ravel()
    else:
        vals = arr[mask]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.zeros_like(arr, dtype=np.float32)
    lo = float(np.percentile(vals, 1))
    hi = float(np.percentile(vals, 99))
    if hi <= lo:
        lo = float(np.min(vals))
        hi = float(np.max(vals))
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    out = (arr - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def shannon_entropy(values: Iterable[float]) -> float:
    arr = finite_values(values)
    if arr.size == 0:
        return float("nan")
    arr = arr[arr > 0]
    if arr.size == 0:
        return 0.0
    probs = arr / float(np.sum(arr))
    return float(-np.sum(probs * np.log2(probs)))


def gini_coefficient(values: Iterable[float]) -> float:
    arr = finite_values(values)
    if arr.size == 0:
        return float("nan")
    arr = np.sort(arr)
    if np.allclose(arr, 0):
        return 0.0
    n = arr.size
    index = np.arange(1, n + 1, dtype=np.float64)
    return float(np.sum((2.0 * index - n - 1.0) * arr) / (n * np.sum(arr)))


def score_bin_features(values: Iterable[float], prefix: str) -> Dict[str, float]:
    arr = np.clip(finite_values(values), 0.0, 1.0)
    out: Dict[str, float] = {}
    if arr.size == 0:
        for label in ("0", "1plus", "2plus", "3plus"):
            out[f"{prefix}_frac_{label}"] = float("nan")
        out[f"{prefix}_hscore"] = float("nan")
        return out
    bins = np.array([0.0, 0.25, 0.50, 0.75, 1.000001], dtype=np.float64)
    counts, _ = np.histogram(arr, bins=bins)
    frac = counts.astype(np.float64) / float(arr.size)
    out[f"{prefix}_frac_0"] = float(frac[0])
    out[f"{prefix}_frac_1plus"] = float(frac[1])
    out[f"{prefix}_frac_2plus"] = float(frac[2])
    out[f"{prefix}_frac_3plus"] = float(frac[3])
    out[f"{prefix}_hscore"] = float(100.0 * (frac[1] + 2.0 * frac[2] + 3.0 * frac[3]))
    return out


def fractal_dimension_boxcount(mask: np.ndarray) -> float:
    mask = np.asarray(mask, dtype=bool)
    if not np.any(mask):
        return float("nan")
    h, w = mask.shape
    max_pow = int(math.floor(math.log2(min(h, w)))) if min(h, w) > 1 else 0
    sizes = [2 ** k for k in range(1, max_pow + 1)]
    if len(sizes) < 2:
        return float("nan")
    counts = []
    scales = []
    for size in sizes:
        pad_h = (-h) % size
        pad_w = (-w) % size
        if pad_h or pad_w:
            padded = np.pad(mask, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=False)
        else:
            padded = mask
        rows = padded.shape[0] // size
        cols = padded.shape[1] // size
        blocks = padded.reshape(rows, size, cols, size)
        total = int(np.count_nonzero(np.any(blocks, axis=(1, 3))))
        if total > 0:
            counts.append(total)
            scales.append(1.0 / size)
    if len(counts) < 2:
        return float("nan")
    slope, _ = np.polyfit(np.log(scales), np.log(counts), deg=1)
    return float(slope)


def hu_moments_from_mask(mask: np.ndarray, prefix: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    mask_u8 = (np.asarray(mask, dtype=np.uint8) * 255)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        for idx in range(1, 8):
            out[f"{prefix}_hu_{idx}"] = float("nan")
        return out
    contour = max(contours, key=cv2.contourArea)
    moments = cv2.moments(contour)
    hu = cv2.HuMoments(moments).flatten()
    for idx, value in enumerate(hu, start=1):
        if value == 0:
            mapped = 0.0
        else:
            mapped = -math.copysign(1.0, float(value)) * math.log10(abs(float(value)))
        out[f"{prefix}_hu_{idx}"] = float(mapped)
    return out


def format_band(a: float, b: float) -> str:
    def _fmt(v: float) -> str:
        text = f"{float(v):g}"
        return text.replace(".", "p")

    return f"{_fmt(a)}_{_fmt(b)}"


def format_length_um(value: float) -> str:
    text = f"{float(value):.3g}"
    return text.replace(".", "p").replace("-", "m")


def euclidean_lengths(points_a: np.ndarray, points_b: np.ndarray, pixel_width_um: float, pixel_height_um: float) -> np.ndarray:
    dy = (points_a[:, 0] - points_b[:, 0]) * float(pixel_height_um)
    dx = (points_a[:, 1] - points_b[:, 1]) * float(pixel_width_um)
    return np.sqrt(dx * dx + dy * dy)


def bounding_box(mask: np.ndarray) -> Tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return 0, 0, mask.shape[0], mask.shape[1]
    minr, minc = coords.min(axis=0)
    maxr, maxc = coords.max(axis=0) + 1
    return int(minr), int(minc), int(maxr), int(maxc)
