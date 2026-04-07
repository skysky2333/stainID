from typing import Dict, Sequence

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.measure import euler_number, label

from .common import (
    fractal_dimension_boxcount,
    gini_coefficient,
    safe_cv,
    safe_iqr,
    safe_kurtosis,
    safe_mean,
    safe_skew,
    safe_std,
    score_bin_features,
    shannon_entropy,
)


def compute_site_summary(df_sites) -> Dict[str, float]:
    out: Dict[str, float] = {}
    cols = [
        "area_um2",
        "perimeter_um",
        "equivalent_diameter_um",
        "major_axis_length_um",
        "minor_axis_length_um",
        "eccentricity",
        "solidity",
        "extent",
        "circularity",
        "roughness",
        "feret_diameter_um",
        "bbox_area_um2",
        "bbox_fill_ratio",
        "bbox_aspect_ratio",
        "elongation",
        "filled_area_um2",
        "convex_area_um2",
        "hole_area_um2",
        "hole_fraction",
        "euler_number",
        "intensity_mean",
        "intensity_raw_mean",
        "intensity_raw_median",
        "intensity_raw_iqr",
        "intensity_raw_integrated",
    ]
    for col in cols:
        if df_sites is None or df_sites.empty or col not in df_sites.columns:
            out[f"{col}_median"] = float("nan")
            out[f"{col}_iqr"] = float("nan")
            continue
        vals = np.asarray(df_sites[col].values, dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        out[f"{col}_median"] = float(np.median(vals)) if vals.size else float("nan")
        out[f"{col}_iqr"] = safe_iqr(vals)
    return out


def compute_pixel_intensity_features(mask: np.ndarray,
                                     valid_mask: np.ndarray,
                                     norm_image: np.ndarray,
                                     raw_image: np.ndarray) -> Dict[str, float]:
    out: Dict[str, float] = {}
    mask = np.asarray(mask, dtype=bool)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    if mask.shape != valid_mask.shape:
        raise ValueError("mask and valid_mask must have the same shape")
    tissue_pixels = valid_mask
    positive_pixels = mask & tissue_pixels
    out["percent_area_stained"] = float(np.mean(mask))
    out["percent_tissue_area"] = float(np.mean(tissue_pixels))
    out["positive_fraction_in_tissue"] = float(np.sum(positive_pixels) / np.sum(tissue_pixels)) if np.any(tissue_pixels) else float("nan")
    for prefix, arr, use_mask in (
        ("pixel_norm_tissue", norm_image, tissue_pixels),
        ("pixel_norm_positive", norm_image, positive_pixels),
        ("pixel_raw_tissue", raw_image, tissue_pixels),
        ("pixel_raw_positive", raw_image, positive_pixels),
    ):
        vals = np.asarray(arr)[use_mask]
        out[f"{prefix}_mean"] = safe_mean(vals)
        out[f"{prefix}_sd"] = safe_std(vals)
        out[f"{prefix}_iqr"] = safe_iqr(vals)
        out[f"{prefix}_skew"] = safe_skew(vals)
        out[f"{prefix}_kurtosis"] = safe_kurtosis(vals)
    out.update(score_bin_features(np.asarray(norm_image)[tissue_pixels], prefix="pixel_norm_tissue"))
    out.update(score_bin_features(np.asarray(norm_image)[positive_pixels], prefix="pixel_norm_positive"))
    return out


def compute_object_score_features(df_sites) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if df_sites is None or df_sites.empty:
        out.update(score_bin_features([], prefix="object_norm_mean"))
        out.update(score_bin_features([], prefix="object_norm_integrated"))
        return out
    out.update(score_bin_features(df_sites["intensity_mean"].values, prefix="object_norm_mean"))
    if "intensity_integrated" in df_sites.columns:
        integrated = np.asarray(df_sites["intensity_integrated"].values, dtype=np.float64)
        finite = integrated[np.isfinite(integrated)]
        max_val = float(np.max(finite)) if finite.size else float("nan")
        if np.isfinite(max_val) and max_val > 0:
            integrated = integrated / max_val
        out.update(score_bin_features(integrated, prefix="object_norm_integrated"))
    else:
        out.update(score_bin_features([], prefix="object_norm_integrated"))
    return out


def compute_mask_topology_features(mask: np.ndarray,
                                   valid_mask: np.ndarray,
                                   pixel_width_um: float,
                                   pixel_height_um: float,
                                   labeled: np.ndarray = None,
                                   prefix: str = "topology") -> Dict[str, float]:
    out: Dict[str, float] = {}
    mask = np.asarray(mask, dtype=bool)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    px_area_um2 = float(pixel_width_um) * float(pixel_height_um)
    labeled = label(mask) if labeled is None else np.asarray(labeled)
    area_counts = np.bincount(labeled.ravel()) if labeled.size else np.zeros(0, dtype=np.int64)
    areas_px = area_counts[1:].astype(np.float64, copy=False)
    areas_px = areas_px[areas_px > 0]
    tissue_area_px = float(np.sum(valid_mask))
    tissue_area_mm2 = tissue_area_px * px_area_um2 / 1_000_000.0 if tissue_area_px > 0 else float("nan")

    component_count = float(areas_px.size)
    out[f"{prefix}_component_count"] = component_count
    out[f"{prefix}_component_density_per_mm2"] = (component_count / tissue_area_mm2) if tissue_area_mm2 and tissue_area_mm2 > 0 else float("nan")
    out[f"{prefix}_largest_component_frac"] = float(np.max(areas_px) / np.sum(areas_px)) if areas_px.size and np.sum(areas_px) > 0 else float("nan")
    out[f"{prefix}_component_area_median_um2"] = float(np.median(areas_px) * px_area_um2) if areas_px.size else float("nan")
    out[f"{prefix}_component_area_iqr_um2"] = safe_iqr(areas_px * px_area_um2) if areas_px.size else float("nan")
    euler = float(euler_number(mask.astype(np.uint8), connectivity=1))
    out[f"{prefix}_euler_number"] = euler
    hole_mask = ndi.binary_fill_holes(mask) & ~mask
    hole_area_px = float(np.sum(hole_mask))
    out[f"{prefix}_hole_area_frac"] = float(hole_area_px / np.sum(mask)) if np.any(mask) else float("nan")
    out[f"{prefix}_hole_count"] = max(0.0, component_count - euler)
    mask_u8 = (mask.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    perimeter_total = float(sum(cv2.arcLength(cnt, True) for cnt in contours))
    out[f"{prefix}_boundary_length_um"] = perimeter_total * float(np.sqrt(px_area_um2))
    out[f"{prefix}_fractal_dimension"] = fractal_dimension_boxcount(mask)
    return out


def compute_tile_heterogeneity_features(mask: np.ndarray,
                                        valid_mask: np.ndarray,
                                        norm_image: np.ndarray,
                                        grids: Sequence[int] = (4, 8),
                                        prefix: str = "tile") -> Dict[str, float]:
    out: Dict[str, float] = {}
    mask = np.asarray(mask, dtype=bool)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    norm_image = np.asarray(norm_image)
    h, w = mask.shape
    for grid in grids:
        mass = []
        positive_fraction = []
        intensity_mean = []
        row_edges = np.linspace(0, h, int(grid) + 1, dtype=int)
        col_edges = np.linspace(0, w, int(grid) + 1, dtype=int)
        for r0, r1 in zip(row_edges[:-1], row_edges[1:]):
            for c0, c1 in zip(col_edges[:-1], col_edges[1:]):
                valid_tile = valid_mask[r0:r1, c0:c1]
                if not np.any(valid_tile):
                    continue
                mask_tile = mask[r0:r1, c0:c1] & valid_tile
                vals_tile = norm_image[r0:r1, c0:c1][valid_tile]
                mass.append(float(np.sum(mask_tile)))
                positive_fraction.append(float(np.sum(mask_tile) / np.sum(valid_tile)))
                intensity_mean.append(float(np.mean(vals_tile)) if vals_tile.size else float("nan"))
        base = f"{prefix}_g{int(grid)}"
        frac_mean = safe_mean(positive_fraction)
        out[f"{base}_mass_entropy"] = shannon_entropy(mass)
        out[f"{base}_mass_gini"] = gini_coefficient(mass)
        out[f"{base}_positive_frac_mean"] = frac_mean
        out[f"{base}_positive_frac_sd"] = safe_std(positive_fraction)
        out[f"{base}_positive_frac_cv"] = safe_cv(positive_fraction)
        out[f"{base}_positive_frac_iqr"] = safe_iqr(positive_fraction)
        out[f"{base}_intensity_mean_sd"] = safe_std(intensity_mean)
        out[f"{base}_intensity_mean_iqr"] = safe_iqr(intensity_mean)
    return out
