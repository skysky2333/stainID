from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage
from skimage.filters import sato

from stainid.imaging.color import rgb_to_hed
from stainid.imaging.tissue import field_tissue_mask


def neuritic_plaque_candidates(
    rgb: np.ndarray,
    dab_threshold: float,
    pixel_size_um: float,
    exclusion: np.ndarray,
    soma_labels: np.ndarray | None = None,
    radius_um: float = 40.0,
    min_contrast: float = 0.2,
) -> list[dict[str, float]]:
    """Round clusters of dystrophic-neurite AT8 signal that are not neuronal somata or straight threads."""
    raw_tissue = field_tissue_mask(rgb)
    tissue = raw_tissue & ~exclusion
    hed = rgb_to_hed(rgb)
    dab = np.maximum(hed[..., 2], 0).astype(np.float32)
    hematoxylin = np.maximum(hed[..., 0], 0).astype(np.float32)
    edge_band = cv2.dilate((~raw_tissue).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * round(15 / pixel_size_um) + 1,) * 2)).astype(bool)
    smooth = cv2.GaussianBlur(dab, (0, 0), 0.8)
    positive = (smooth >= 0.6 * dab_threshold) & tissue
    ridge = sato(dab, sigmas=[1, 1.5, 2], black_ridges=False)
    linear = positive & (ridge >= 0.4 * dab_threshold)
    blobby = positive & ~linear
    if soma_labels is not None:
        blobby &= soma_labels == 0
    haze = np.where(tissue & ~linear & ((soma_labels == 0) if soma_labels is not None else True), dab, 0).astype(np.float32)
    weight = (tissue & ~linear).astype(np.float32)
    scale = 4
    small = lambda a: cv2.resize(a, (a.shape[1] // scale, a.shape[0] // scale), interpolation=cv2.INTER_AREA)
    sigma = radius_um / 2 / pixel_size_um / scale
    local = cv2.GaussianBlur(small(haze), (0, 0), sigma) / np.maximum(cv2.GaussianBlur(small(weight), (0, 0), sigma), 1e-3)
    wide = cv2.GaussianBlur(small(haze), (0, 0), 4 * sigma) / np.maximum(cv2.GaussianBlur(small(weight), (0, 0), 4 * sigma), 1e-3)
    contrast_small = (local - wide) / dab_threshold * (small(tissue.astype(np.float32)) > 0.5)
    peaks_small = (contrast_small == ndimage.maximum_filter(contrast_small, size=round(2 * radius_um / pixel_size_um / scale))) & (contrast_small >= min_contrast)
    peaks = np.zeros(dab.shape, dtype=bool)
    ys_small, xs_small = np.nonzero(peaks_small)
    peaks[np.minimum(ys_small * scale + scale // 2, dab.shape[0] - 1), np.minimum(xs_small * scale + scale // 2, dab.shape[1] - 1)] = True
    density = cv2.resize(local, (dab.shape[1], dab.shape[0])) / dab_threshold
    contrast = cv2.resize(contrast_small, (dab.shape[1], dab.shape[0]))
    radius_px = round(radius_um / pixel_size_um)
    yy, xx = np.ogrid[-radius_px : radius_px + 1, -radius_px : radius_px + 1]
    disk = xx**2 + yy**2 <= radius_px**2
    rows = []
    height, width = dab.shape
    for y, x in zip(*np.nonzero(peaks)):
        y0, y1, x0, x1 = max(0, y - radius_px), min(height, y + radius_px + 1), max(0, x - radius_px), min(width, x + radius_px + 1)
        local_disk = disk[y0 - (y - radius_px) : y1 - (y - radius_px), x0 - (x - radius_px) : x1 - (x - radius_px)]
        local_blob = blobby[y0:y1, x0:x1] & local_disk
        local_line = linear[y0:y1, x0:x1] & local_disk
        count, _ = cv2.connectedComponents(local_blob.astype(np.uint8), connectivity=8)
        ys, xs = np.nonzero(local_blob)
        spread = np.hypot(ys + y0 - y, xs + x0 - x) if len(ys) else np.array([0.0])
        rows.append({
            "centroid_x_px": float(x), "centroid_y_px": float(y),
            "haze_level": float(density[y, x]), "haze_contrast": float(contrast[y, x]),
            "blob_fraction": float(local_blob.sum() / max(local_disk.sum(), 1)),
            "line_fraction": float(local_line.sum() / max((local_blob | local_line).sum(), 1)),
            "fragment_count": int(count - 1),
            "radial_spread_um": float(np.median(spread) * pixel_size_um),
            "normalized_dab": float(smooth[y0:y1, x0:x1][local_blob].mean() / dab_threshold) if local_blob.any() else 0.0,
            "tissue_fraction": float(tissue[y0:y1, x0:x1][local_disk].mean()),
            "window_fraction": float(local_disk.sum() / disk.sum()),
            "edge_fraction": float(edge_band[y0:y1, x0:x1][local_disk].mean()),
            "hematoxylin_mean": float(hematoxylin[y0:y1, x0:x1][local_disk].mean()),
            "haze_dab_p50": float(np.median(dab[y0:y1, x0:x1][local_disk & ~local_line]) / dab_threshold),
        })
    return rows


__all__ = [
    "neuritic_plaque_candidates",
]
