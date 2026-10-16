from __future__ import annotations

import numpy as np

from stainid.imaging.shared import remembered

HED_FROM_RGB = np.array(
    [
        [1.87798274, -1.00767869, -0.55611582],
        [-0.06590806, 1.13473037, -0.13552180],
        [-0.60190736, -0.48041419, 1.57358807],
    ],
    dtype=np.float32,
)


def rgb_to_hed(rgb: np.ndarray) -> np.ndarray:
    return remembered("hed", rgb, lambda: _rgb_to_hed(rgb))


def _optical_density(levels: np.ndarray) -> np.ndarray:
    scaled = np.maximum(levels.astype(np.float32) / 255.0, 1e-6)
    return np.log(scaled) / np.log(1e-6)


# the optical density of each 8-bit level, computed by the same steps as for a picture (a pixel's value depends on its
# level alone, so looking it up gives the same numbers in a fraction of the time)
_DENSITY = _optical_density(np.arange(256, dtype=np.uint8))


def _rgb_to_hed(rgb: np.ndarray) -> np.ndarray:
    density = _DENSITY[rgb] if rgb.dtype == np.uint8 else _optical_density(rgb)
    stains = density @ HED_FROM_RGB
    return np.maximum(stains, 0.0)
