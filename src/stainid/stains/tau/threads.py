from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage
from skimage.filters import sato
from skimage.morphology import remove_small_objects, skeletonize

from stainid.imaging.color import rgb_to_hed
from stainid.imaging.tissue import field_tissue_mask


def _disk(radius_px: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius_px + 1, 2 * radius_px + 1))


def thread_network(
    rgb: np.ndarray,
    valid: np.ndarray,
    dab_threshold: float,
    pixel_size_um: float,
    nuclei: np.ndarray | None = None,
    contrast: float = 0.6,
    background_um: float = 8.0,
    rim_um: float = 4.0,
    minimum_length_um: float = 4.0,
) -> dict[str, np.ndarray | float]:
    """Neuropil threads: ridges that exceed the slide threshold and stand out from the local AT8 background (haze),
    away from tissue/hole rims and hematoxylin nuclei, and at least `minimum_length_um` long."""
    dab = np.maximum(rgb_to_hed(rgb)[..., 2], 0).astype(np.float32)
    smooth = cv2.GaussianBlur(dab, (0, 0), 0.8)
    ridge = sato(dab, sigmas=[1, 1.5, 2, 3], black_ridges=False)
    valid = valid & (cv2.distanceTransform(field_tissue_mask(rgb).astype(np.uint8), cv2.DIST_L2, 5) * pixel_size_um > rim_um)
    if nuclei is not None:
        valid = valid & ~cv2.dilate((nuclei > 0).astype(np.uint8), _disk(max(1, round(1.0 / pixel_size_um)))).astype(bool)
    background = cv2.GaussianBlur(dab, (0, 0), background_um / pixel_size_um)
    line = (ridge >= 0.25 * dab_threshold) & (smooth >= 0.6 * dab_threshold) & (smooth >= (1 + contrast) * background) & valid
    line = remove_small_objects(line, 12)
    labels, count = ndimage.label(line, np.ones((3, 3)))
    length = np.bincount(labels[skeletonize(line)], minlength=count + 1) * pixel_size_um
    line = np.isin(labels, np.flatnonzero(length >= minimum_length_um)) & (labels > 0)
    skeleton = skeletonize(line)
    neighbors = cv2.filter2D(skeleton.astype(np.uint8), cv2.CV_16S, np.ones((3, 3), np.uint8), borderType=cv2.BORDER_CONSTANT) - skeleton
    distance = cv2.distanceTransform(line.astype(np.uint8), cv2.DIST_L2, 5)
    return {
        "line": line,
        "skeleton": skeleton,
        "branchpoints": skeleton & (neighbors >= 3),
        "endpoints": skeleton & (neighbors == 1),
        "width_um": 2 * distance * pixel_size_um,
    }
