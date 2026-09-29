from __future__ import annotations

import cv2
import numpy as np


def calibrate_threshold(dab_values: np.ndarray) -> tuple[float, tuple[float, ...]]:
    values = np.asarray(dab_values, dtype=np.float32)
    values = values[np.isfinite(values)]
    if values.size < 100:
        raise ValueError("At least 100 tissue pixels are required for calibration")
    quantiles = tuple(float(value) for value in np.quantile(values, [0.5, 0.75, 0.9, 0.95, 0.99, 0.999]))
    upper = max(0.05, quantiles[-1])
    scaled = np.rint(np.clip(values, 0.0, upper) * (255.0 / upper)).astype(np.uint8)
    otsu, _ = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    median = quantiles[0]
    mad = float(np.median(np.abs(values - median)))
    threshold = max(float(otsu) * upper / 255.0, median + 6.0 * mad, 0.01)
    return threshold, quantiles
