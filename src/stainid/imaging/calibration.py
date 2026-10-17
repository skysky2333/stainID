from __future__ import annotations

import cv2
import numpy as np

SATURATED_OD = 1.0  # DAB optical density no unsaturated pixel reaches (a colour channel clipped at 0 in 8-bit RGB)


def calibrate_threshold(dab_values: np.ndarray) -> tuple[float, tuple[float, ...]]:
    """The DAB threshold of a slide: the largest of Otsu's split (between 0 and the 99.9th percentile), the median plus
    6 MADs, and 0.01 OD. Saturated pixels (DAB OD >= `SATURATED_OD`: a channel clipped to 0, as in strongly stained
    JPEG fields) are stained for certain but say nothing about where the stain begins; with them in the range, a few
    per thousand pull Otsu's split into the empty gap below them (0.6 OD instead of ~0.1), so Otsu's split is made on
    the unsaturated pixels. The quantiles reported are of all pixels."""
    values = np.asarray(dab_values, dtype=np.float32)
    values = values[np.isfinite(values)]
    if values.size < 100:
        raise ValueError("At least 100 tissue pixels are required for calibration")
    quantiles = tuple(float(value) for value in np.quantile(values, [0.5, 0.75, 0.9, 0.95, 0.99, 0.999]))
    unsaturated = values[values < SATURATED_OD]
    pool = values if unsaturated.size == values.size or unsaturated.size < 100 else unsaturated
    upper = max(0.05, quantiles[-1] if pool is values else float(np.quantile(pool, 0.999)))
    scaled = np.rint(np.clip(pool, 0.0, upper) * (255.0 / upper)).astype(np.uint8)
    otsu, _ = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    median = quantiles[0]
    mad = float(np.median(np.abs(values - median)))
    threshold = max(float(otsu) * upper / 255.0, median + 6.0 * mad, 0.01)
    return threshold, quantiles
