import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from skimage.measure import label, regionprops

from .common import (
    safe_iqr,
    safe_kurtosis,
    safe_max,
    safe_median,
    safe_min,
    safe_skew,
    safe_std,
)


def measure_objects(mask: np.ndarray,
                    norm_image: np.ndarray,
                    raw_image: np.ndarray,
                    pixel_width_um: float,
                    pixel_height_um: float,
                    labeled_in: Optional[np.ndarray] = None) -> Tuple[np.ndarray, pd.DataFrame]:
    labeled = label(mask) if labeled_in is None else labeled_in
    props = regionprops(labeled)
    rows: List[Dict] = []
    px_area_um2 = float(pixel_width_um) * float(pixel_height_um)
    mean_pixel_um = math.sqrt(px_area_um2)

    for region in props:
        coords = region.coords
        vals_norm = norm_image[coords[:, 0], coords[:, 1]]
        vals_raw = raw_image[coords[:, 0], coords[:, 1]]

        area_px = float(region.area)
        perimeter_px = float(region.perimeter)
        filled_area_px = float(getattr(region, "filled_area", region.area) or region.area)
        convex_area_px = float(getattr(region, "convex_area", region.area) or region.area)
        bbox_minr, bbox_minc, bbox_maxr, bbox_maxc = region.bbox
        bbox_height_px = float(bbox_maxr - bbox_minr)
        bbox_width_px = float(bbox_maxc - bbox_minc)
        bbox_area_px = bbox_height_px * bbox_width_px
        major_axis_px = float(getattr(region, "major_axis_length", 0.0) or 0.0)
        minor_axis_px = float(getattr(region, "minor_axis_length", 0.0) or 0.0)
        aspect_ratio = (bbox_width_px / bbox_height_px) if bbox_height_px > 0 else float("nan")
        elongation = (major_axis_px / minor_axis_px) if minor_axis_px > 0 else float("nan")
        circularity = float(4.0 * math.pi * area_px / (perimeter_px ** 2)) if perimeter_px > 0 else 0.0
        roughness = float((perimeter_px ** 2) / (4.0 * math.pi * area_px)) if area_px > 0 else float("nan")
        hole_area_px = max(0.0, filled_area_px - area_px)
        hole_fraction = (hole_area_px / filled_area_px) if filled_area_px > 0 else 0.0
        area_um2 = area_px * px_area_um2
        perimeter_um = perimeter_px * mean_pixel_um
        bbox_area_um2 = bbox_area_px * px_area_um2

        row = {
            "label": int(region.label),
            "centroid_row_um": float(region.centroid[0]) * float(pixel_height_um),
            "centroid_col_um": float(region.centroid[1]) * float(pixel_width_um),
            "area_um2": area_um2,
            "perimeter_um": perimeter_um,
            "equivalent_diameter_um": float(getattr(region, "equivalent_diameter", 0.0) or 0.0) * mean_pixel_um,
            "major_axis_length_um": major_axis_px * mean_pixel_um,
            "minor_axis_length_um": minor_axis_px * mean_pixel_um,
            "eccentricity": float(getattr(region, "eccentricity", 0.0) or 0.0),
            "solidity": float(getattr(region, "solidity", 0.0) or 0.0),
            "extent": float(getattr(region, "extent", 0.0) or 0.0),
            "orientation_deg": float(math.degrees(float(getattr(region, "orientation", 0.0) or 0.0))),
            "circularity": circularity,
            "roughness": roughness,
            "feret_diameter_um": float(getattr(region, "feret_diameter_max", major_axis_px) or major_axis_px) * mean_pixel_um,
            "bbox_area_um2": bbox_area_um2,
            "bbox_fill_ratio": (area_px / bbox_area_px) if bbox_area_px > 0 else float("nan"),
            "bbox_aspect_ratio": aspect_ratio,
            "elongation": elongation,
            "filled_area_um2": filled_area_px * px_area_um2,
            "convex_area_um2": convex_area_px * px_area_um2,
            "hole_area_um2": hole_area_px * px_area_um2,
            "hole_fraction": hole_fraction,
            "euler_number": float(getattr(region, "euler_number", 1.0) or 1.0),
        }

        row.update(_intensity_features(vals_norm, prefix="intensity"))
        row.update(_intensity_features(vals_raw, prefix="intensity_raw"))
        rows.append(row)

    return labeled, pd.DataFrame(rows)


def _intensity_features(values: np.ndarray, prefix: str) -> Dict[str, float]:
    vals = np.asarray(values, dtype=np.float64)
    if vals.size == 0:
        keys = (
            "mean", "std", "median", "iqr", "integrated",
            "min", "max", "skew", "kurtosis",
        )
        return {f"{prefix}_{key}": float("nan") for key in keys}
    return {
        f"{prefix}_mean": float(np.mean(vals)),
        f"{prefix}_std": safe_std(vals, ddof=0),
        f"{prefix}_median": safe_median(vals),
        f"{prefix}_iqr": safe_iqr(vals),
        f"{prefix}_integrated": float(np.sum(vals)),
        f"{prefix}_min": safe_min(vals),
        f"{prefix}_max": safe_max(vals),
        f"{prefix}_skew": safe_skew(vals),
        f"{prefix}_kurtosis": safe_kurtosis(vals),
    }
