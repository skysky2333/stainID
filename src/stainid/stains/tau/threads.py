from __future__ import annotations

import math
from itertools import combinations_with_replacement

import cv2
import numpy as np
from scipy import ndimage
from scipy.ndimage._filters import _gaussian_kernel1d
from skimage.feature import hessian_matrix_eigvals
from skimage.morphology import remove_small_objects, skeletonize

from stainid.imaging.color import rgb_to_hed
from stainid.imaging.parallel import map_all, strips
from stainid.imaging.tissue import field_tissue_mask


def _disk(radius_px: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius_px + 1, 2 * radius_px + 1))


# A Gaussian tap smaller than this share of the centre tap adds nothing a float32 picture can hold (values below 1e-45)
NEGLIGIBLE_TAP = 1e-60


def _gaussian(image: np.ndarray, sigma: float, order: tuple[int, int], truncate: float) -> np.ndarray:
    """`scipy.ndimage.gaussian_filter(image, sigma, order, mode="reflect", truncate=truncate)` of a 2-D float32 picture,
    with each kernel cut where its taps fall below NEGLIGIBLE_TAP. The taps kept are scipy's own (computed over the full
    radius, then cut), applied in the same order, so the result is scipy's up to contributions below 1e-60 of a pixel
    (in float32: subnormal values where the picture is zero nearby). skimage asks for 100 standard deviations at the
    smallest scale (about 140 taps, 9 of them not negligible), which made the ridge filter the slowest step of AT8."""
    output = np.zeros(image.shape, dtype=image.dtype)
    source = image
    keep = int(np.ceil(sigma * np.sqrt(2.0 * np.log(1.0 / NEGLIGIBLE_TAP)))) + 1
    for axis in range(2):
        radius = int(truncate * float(sigma) + 0.5)
        weights = _gaussian_kernel1d(sigma, order[axis], radius)[::-1]
        if radius > keep:
            weights = weights[radius - keep: radius + keep + 1]

        def strip(part, axis=axis, weights=weights, source=source):  # the lines across `part`, each filtered on its own
            at = (slice(None), part) if axis == 0 else (part, slice(None))
            ndimage.correlate1d(source[at], weights, axis, output[at], "reflect", 0.0, 0)

        map_all(strip, strips(image.shape[1 - axis]))
        source = output
    return output


def ridge_filter(image: np.ndarray, sigmas=(1, 1.5, 2, 3)) -> np.ndarray:
    """`skimage.filters.sato(image, sigmas, black_ridges=False)` (skimage 0.25) of a 2-D float32 picture, with the
    Gaussian derivatives of `_gaussian` (the same values; see there)."""
    image = -np.asarray(image, dtype=np.float32)
    filtered_max = np.zeros_like(image)
    for sigma in sigmas:
        truncate = 8 if sigma > 1 else 100  # skimage's choice, from the unscaled sigma
        scaled = (1 / math.sqrt(2)) * sigma  # as skimage scales it
        gradients = [_gaussian(image, scaled, (1, 0), truncate), _gaussian(image, scaled, (0, 1), truncate)]
        orders = ((1, 0), (0, 1))
        elements = [_gaussian(gradients[a], scaled, orders[b], truncate) for a, b in combinations_with_replacement(range(2), 2)]
        eigvals = hessian_matrix_eigvals(elements)[:-1]
        vals = sigma**2 * np.prod(np.maximum(eigvals, 0), 0) ** (1 / len(eigvals))
        filtered_max = np.maximum(filtered_max, vals)
    return filtered_max


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
    ridge = ridge_filter(dab, sigmas=[1, 1.5, 2, 3])
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
