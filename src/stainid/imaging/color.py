from __future__ import annotations

import numpy as np

HED_FROM_RGB = np.array(
    [
        [1.87798274, -1.00767869, -0.55611582],
        [-0.06590806, 1.13473037, -0.13552180],
        [-0.60190736, -0.48041419, 1.57358807],
    ],
    dtype=np.float32,
)


def rgb_to_hed(rgb: np.ndarray) -> np.ndarray:
    scaled = np.maximum(rgb.astype(np.float32) / 255.0, 1e-6)
    stains = (np.log(scaled) / np.log(1e-6)) @ HED_FROM_RGB
    return np.maximum(stains, 0.0)
