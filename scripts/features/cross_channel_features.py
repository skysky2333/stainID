from typing import Dict, Mapping

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.morphology import disk, dilation

from .common import format_band, safe_corr


def compute_cross_channel_features(channel_name: str,
                                   masks_by_channel: Mapping[str, np.ndarray],
                                   labels_by_channel: Mapping[str, np.ndarray],
                                   site_tables: Mapping[str, object],
                                   norm_by_channel: Mapping[str, np.ndarray],
                                   raw_by_channel: Mapping[str, np.ndarray],
                                   valid_mask: np.ndarray,
                                   pixel_width_um: float,
                                   pixel_height_um: float) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if channel_name not in masks_by_channel:
        return out
    this_mask = np.asarray(masks_by_channel.get(channel_name), dtype=bool)
    this_labels = labels_by_channel.get(channel_name)
    this_sites = site_tables.get(channel_name)
    if this_mask is None or this_labels is None or this_sites is None:
        return out
    this_points = _site_points(this_sites)
    for other_name, other_mask in masks_by_channel.items():
        if other_name == channel_name:
            continue
        other_mask = np.asarray(other_mask, dtype=bool)
        other_labels = labels_by_channel.get(other_name)
        other_sites = site_tables.get(other_name)
        if other_labels is None or other_sites is None:
            continue
        other_points = _site_points(other_sites)
        prefix = f"cross_with_{other_name}"
        intersection = float(np.sum(this_mask & other_mask))
        union = float(np.sum(this_mask | other_mask))
        this_area = float(np.sum(this_mask))
        other_area = float(np.sum(other_mask))
        out[f"{prefix}_jaccard"] = (intersection / union) if union > 0 else float("nan")
        out[f"{prefix}_dice"] = (2.0 * intersection / (this_area + other_area)) if (this_area + other_area) > 0 else float("nan")
        out[f"{prefix}_frac_self_overlap"] = (intersection / this_area) if this_area > 0 else float("nan")
        out[f"{prefix}_frac_other_overlap"] = (intersection / other_area) if other_area > 0 else float("nan")
        out[f"{prefix}_positive_area_ratio"] = (this_area / other_area) if other_area > 0 else float("nan")

        dilated_other = dilation(other_mask, footprint=disk(2))
        this_object_count = int(np.unique(this_labels[this_labels > 0]).size)
        if this_object_count > 0:
            touching_labels = np.unique(this_labels[dilated_other & (this_labels > 0)])
            overlapping_labels = np.unique(this_labels[other_mask & (this_labels > 0)])
            out[f"{prefix}_touching_object_fraction"] = float(touching_labels.size / this_object_count)
            out[f"{prefix}_overlapping_object_fraction"] = float(overlapping_labels.size / this_object_count)
        else:
            out[f"{prefix}_touching_object_fraction"] = float("nan")
            out[f"{prefix}_overlapping_object_fraction"] = float("nan")

        out.update(_nearest_distance_features(prefix, this_points, other_points))
        out.update(_correlation_features(prefix, channel_name, other_name, raw_by_channel, norm_by_channel, valid_mask, this_mask, other_mask))
        out.update(_radial_features(prefix, norm_by_channel[channel_name], other_mask, valid_mask, pixel_width_um, pixel_height_um))
    return out


def _site_points(df_sites) -> np.ndarray:
    if df_sites is None or getattr(df_sites, "empty", True):
        return np.zeros((0, 2), dtype=np.float64)
    return df_sites[["centroid_row_um", "centroid_col_um"]].values.astype(np.float64)


def _nearest_distance_features(prefix: str,
                               src_points: np.ndarray,
                               dst_points: np.ndarray) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if src_points.shape[0] == 0 or dst_points.shape[0] == 0:
        out[f"{prefix}_nn_other_mean_um"] = float("nan")
        return out
    tree = cKDTree(dst_points)
    dists_um, _ = tree.query(src_points, k=1)
    out[f"{prefix}_nn_other_mean_um"] = float(np.mean(dists_um))
    return out


def _correlation_features(prefix: str,
                          this_name: str,
                          other_name: str,
                          raw_by_channel,
                          norm_by_channel,
                          valid_mask: np.ndarray,
                          this_mask: np.ndarray,
                          other_mask: np.ndarray) -> Dict[str, float]:
    out: Dict[str, float] = {}
    tissue = np.asarray(valid_mask, dtype=bool)
    union = tissue & (this_mask | other_mask)
    raw_a = np.asarray(raw_by_channel[this_name])
    raw_b = np.asarray(raw_by_channel[other_name])
    norm_a = np.asarray(norm_by_channel[this_name])
    norm_b = np.asarray(norm_by_channel[other_name])
    out[f"{prefix}_corr_raw_tissue"] = safe_corr(raw_a[tissue], raw_b[tissue])
    out[f"{prefix}_corr_raw_union"] = safe_corr(raw_a[union], raw_b[union])
    out[f"{prefix}_corr_norm_tissue"] = safe_corr(norm_a[tissue], norm_b[tissue])
    return out


def _radial_features(prefix: str,
                     source_norm: np.ndarray,
                     reference_mask: np.ndarray,
                     valid_mask: np.ndarray,
                     pixel_width_um: float,
                     pixel_height_um: float) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if not np.any(reference_mask):
        for a, b in ((0, 5), (5, 15), (15, 30)):
            band = format_band(a, b)
            out[f"{prefix}_annulus_{band}um_norm_mean"] = float("nan")
        out[f"{prefix}_annulus_near_far_ratio"] = float("nan")
        return out
    dist_um = ndi.distance_transform_edt(~reference_mask, sampling=(float(pixel_height_um), float(pixel_width_um)))
    valid = np.asarray(valid_mask, dtype=bool)
    bands = []
    for a, b in ((0, 5), (5, 15), (15, 30)):
        keep = valid & (dist_um >= float(a)) & (dist_um < float(b))
        val = float(np.mean(source_norm[keep])) if np.any(keep) else float("nan")
        out[f"{prefix}_annulus_{format_band(a, b)}um_norm_mean"] = val
        bands.append(val)
    near = bands[0]
    far = bands[-1]
    out[f"{prefix}_annulus_near_far_ratio"] = (near / far) if np.isfinite(near) and np.isfinite(far) and abs(far) > 1e-12 else float("nan")
    return out
