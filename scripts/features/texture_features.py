import math
from typing import Dict, Sequence, Tuple

import cv2
import numpy as np
from skimage.morphology import disk

from .common import robust_normalize, safe_mean, safe_std


def compute_texture_features(image: np.ndarray,
                             mask: np.ndarray,
                             prefix: str,
                             distances: Sequence[int] = (1, 2, 4),
                             distance_labels: Sequence[str] = None,
                             angles: Sequence[float] = (0.0, math.pi / 4.0, math.pi / 2.0, 3.0 * math.pi / 4.0),
                             levels: int = 32) -> Dict[str, float]:
    out: Dict[str, float] = {}
    mask = np.asarray(mask, dtype=bool)
    if not np.any(mask):
        for prop in _texture_property_names():
            out[f"{prefix}_{prop}_mean"] = float("nan")
            out[f"{prefix}_{prop}_sd"] = float("nan")
            for distance in distances:
                out[f"{prefix}_{_scale_label(distance, distances, distance_labels)}_{prop}_mean"] = float("nan")
        return out

    quantized = _quantize_to_levels(image, mask, levels)
    property_values = {name: [] for name in _texture_property_names()}

    for distance in distances:
        distance_values = {name: [] for name in _texture_property_names()}
        for angle in angles:
            dy = int(round(math.sin(angle) * float(distance)))
            dx = int(round(math.cos(angle) * float(distance)))
            if dy == 0 and dx == 0:
                continue
            glcm = _masked_glcm(quantized, mask, dy=dy, dx=dx, levels=levels)
            if glcm is None:
                continue
            props = _glcm_properties(glcm)
            for name, value in props.items():
                property_values[name].append(value)
                distance_values[name].append(value)
        for name in _texture_property_names():
            vals = distance_values[name]
            out[f"{prefix}_{_scale_label(distance, distances, distance_labels)}_{name}_mean"] = safe_mean(vals)
    for name in _texture_property_names():
        vals = property_values[name]
        out[f"{prefix}_{name}_mean"] = safe_mean(vals)
        out[f"{prefix}_{name}_sd"] = safe_std(vals)
    return out


def compute_granularity_features(image: np.ndarray,
                                 mask: np.ndarray,
                                 prefix: str,
                                 radii: Sequence[int] = (1, 2, 4, 8),
                                 radius_labels: Sequence[str] = None) -> Dict[str, float]:
    out: Dict[str, float] = {}
    mask = np.asarray(mask, dtype=bool)
    if not np.any(mask):
        for radius in radii:
            out[f"{prefix}_{_scale_label(radius, radii, radius_labels)}"] = float("nan")
        return out
    norm = robust_normalize(image, mask=mask)
    work = np.where(mask, norm, 0.0).astype(np.float32)
    base_sum = float(np.sum(work[mask]))
    if base_sum <= 0:
        for radius in radii:
            out[f"{prefix}_{_scale_label(radius, radii, radius_labels)}"] = 0.0
        return out
    current = work.copy()
    previous_sum = base_sum
    for radius in radii:
        opened = cv2.morphologyEx(
            current,
            cv2.MORPH_OPEN,
            disk(int(radius)).astype(np.uint8),
            borderType=cv2.BORDER_REFLECT,
        )
        opened[~mask] = 0.0
        opened_sum = float(np.sum(opened[mask]))
        out[f"{prefix}_{_scale_label(radius, radii, radius_labels)}"] = max(0.0, previous_sum - opened_sum) / base_sum
        current = opened
        previous_sum = opened_sum
    return out


def _scale_label(value: int, values: Sequence[int], labels: Sequence[str]) -> str:
    if labels is None:
        return f"d{int(value)}"
    idx = list(values).index(value)
    return str(labels[idx])


def _texture_property_names() -> Tuple[str, ...]:
    return ("contrast", "dissimilarity", "homogeneity", "asm", "energy", "correlation", "entropy")


def _quantize_to_levels(image: np.ndarray, mask: np.ndarray, levels: int) -> np.ndarray:
    norm = robust_normalize(image, mask=mask)
    quantized = np.clip(np.floor(norm * float(levels - 1)), 0, levels - 1).astype(np.int32)
    quantized[~mask] = 0
    return quantized


def _masked_glcm(quantized: np.ndarray, mask: np.ndarray, dy: int, dx: int, levels: int) -> np.ndarray:
    h, w = quantized.shape
    if abs(dy) >= h or abs(dx) >= w:
        return None
    src_r0 = max(0, -dy)
    src_r1 = h - max(0, dy)
    src_c0 = max(0, -dx)
    src_c1 = w - max(0, dx)
    dst_r0 = src_r0 + dy
    dst_r1 = src_r1 + dy
    dst_c0 = src_c0 + dx
    dst_c1 = src_c1 + dx

    src_mask = mask[src_r0:src_r1, src_c0:src_c1]
    dst_mask = mask[dst_r0:dst_r1, dst_c0:dst_c1]
    keep = src_mask & dst_mask
    if not np.any(keep):
        return None

    src = quantized[src_r0:src_r1, src_c0:src_c1][keep]
    dst = quantized[dst_r0:dst_r1, dst_c0:dst_c1][keep]
    flat_size = int(levels) * int(levels)
    forward = np.bincount(src * levels + dst, minlength=flat_size)
    reverse = np.bincount(dst * levels + src, minlength=flat_size)
    counts = (forward + reverse).astype(np.float64, copy=False).reshape((levels, levels))
    total = float(np.sum(counts))
    if total <= 0:
        return None
    return counts / total


def _glcm_properties(glcm: np.ndarray) -> Dict[str, float]:
    levels = glcm.shape[0]
    i, j = np.indices((levels, levels))
    diff = i - j
    abs_diff = np.abs(diff)
    mu_i = float(np.sum(i * glcm))
    mu_j = float(np.sum(j * glcm))
    sigma_i = math.sqrt(max(0.0, float(np.sum(((i - mu_i) ** 2) * glcm))))
    sigma_j = math.sqrt(max(0.0, float(np.sum(((j - mu_j) ** 2) * glcm))))
    if sigma_i > 0 and sigma_j > 0:
        corr = float(np.sum(((i - mu_i) * (j - mu_j) * glcm)) / (sigma_i * sigma_j))
    else:
        corr = float("nan")
    nz = glcm[glcm > 0]
    entropy = float(-np.sum(nz * np.log2(nz))) if nz.size else 0.0
    asm = float(np.sum(glcm ** 2))
    return {
        "contrast": float(np.sum((diff ** 2) * glcm)),
        "dissimilarity": float(np.sum(abs_diff * glcm)),
        "homogeneity": float(np.sum(glcm / (1.0 + diff ** 2))),
        "asm": asm,
        "energy": float(math.sqrt(asm)),
        "correlation": corr,
        "entropy": entropy,
    }
