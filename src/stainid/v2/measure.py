"""Generic measurements of any candidate region (the superset every current or later model can draw on).

Plain-named columns of the candidate store. Raw values are kept (OD units, µm, fractions) together with the DAB deciles
of the region, so a new feature can be derived later without reading the images.
"""
from __future__ import annotations

import cv2
import numpy as np

from stainid.v2.geometry import Region, disk

QUANTILES = np.linspace(0, 1, 11)


def shape_features(mask: np.ndarray, pixel_size_um: float) -> dict[str, float]:
    pixel_area = pixel_size_um ** 2
    contours, _ = cv2.findContours(np.pad(mask.astype(np.uint8), 1), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    perimeter = float(cv2.arcLength(contour, True))
    contour_area = float(cv2.contourArea(contour))
    hull = float(cv2.contourArea(cv2.convexHull(contour)))
    _, _, w, h = cv2.boundingRect(contour)
    ys, xs = np.nonzero(mask)
    area_px = len(ys)
    if area_px >= 3:
        eigen = np.sort(np.linalg.eigvalsh(np.cov(np.column_stack((xs, ys)).astype(float), rowvar=False)))[::-1]
    else:
        eigen = np.array([0.0, 0.0])
    major = 4.0 * np.sqrt(max(eigen[0], 0.0) * pixel_area)
    minor = 4.0 * np.sqrt(max(eigen[-1], 0.0) * pixel_area)
    return {
        "perimeter_um": perimeter * pixel_size_um,
        "equivalent_diameter_um": 2.0 * np.sqrt(area_px * pixel_area / np.pi),
        "major_axis_um": float(major),
        "minor_axis_um": float(minor),
        "elongation": float(major / max(minor, 1e-6)) if minor else float("nan"),
        "eccentricity": float(np.sqrt(max(0.0, 1.0 - eigen[-1] / eigen[0]))) if eigen[0] > 0 else 0.0,
        "aspect_ratio": float(max(w / max(h, 1), h / max(w, 1))),
        "solidity": contour_area / hull if hull else 0.0,
        "circularity": 4.0 * np.pi * contour_area / perimeter ** 2 if perimeter else 0.0,
        "boundary_irregularity": perimeter ** 2 / (4.0 * np.pi * area_px) if area_px else 0.0,
    }


def generic_features(field, region: Region) -> dict[str, object]:
    ps = field.pixel_size_um
    threshold = field.threshold
    ring_px = max(1, round(2.0 / ps))
    box, local = region.padded(ring_px + 1, field.shape)
    dab = field.dab[box]
    values = dab[local]
    hematoxylin = field.hematoxylin[box][local]
    ring = cv2.dilate(local.astype(np.uint8), disk(ring_px)).astype(bool) & ~local & field.tissue[box]
    ring_mean = float(dab[ring].mean()) if ring.any() else float("nan")
    eroded = cv2.erode(local.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    boundary = local & ~eroded
    x, y = region.centroid()
    xi, yi = min(field.shape[1] - 1, max(0, round(x))), min(field.shape[0] - 1, max(0, round(y)))
    raw = field.raw_nuclei[box][local] if field.raw_nuclei is not None else np.zeros(0, np.uint32)
    nucleus_ids = np.unique(raw[raw > 0])
    deciles = np.quantile(values, QUANTILES)
    mean_dab = float(values.mean())
    out = {
        "area_um2": float(local.sum() * ps * ps),
        **shape_features(region.mask, ps),
        "mean_dab_od": mean_dab,
        "median_dab_od": float(deciles[5]),
        "p10_dab_od": float(deciles[1]),
        "p90_dab_od": float(deciles[9]),
        "max_dab_od": float(deciles[10]),
        "std_dab_od": float(values.std()),
        "dab_q": " ".join(f"{v:.5f}" for v in deciles),
        "mean_hematoxylin_od": float(hematoxylin.mean()),
        "p90_hematoxylin_od": float(np.quantile(hematoxylin, 0.9)),
        "mean_chromogen_fraction": float(field.chromogen_fraction[box][local].mean()),
        "normalized_mean_dab": mean_dab / threshold,
        "normalized_p90_dab": float(deciles[9]) / threshold,
        "positive_fraction": float((values >= threshold).mean()),
        "ring_dab_od": ring_mean,
        "boundary_contrast": (mean_dab - ring_mean) / threshold if ring.any() else float("nan"),
        "edge_strength": float(field.gradient[box][boundary].mean() / threshold) if boundary.any() else 0.0,
        "local_background_dab_od": float(field.local_background[yi, xi]),
        "max_thickness_um": float(cv2.distanceTransform(np.pad(region.mask, 1).astype(np.uint8), cv2.DIST_L2, 5).max() * ps),
        "tissue_fraction": float(field.tissue[box][local].mean()),
        "linear_artifact_fraction": float(field.linear[box][local].mean()),
        "fold_fraction": float(field.fold[box][local].mean()),
        "manual_exclusion_fraction": float(field.manual[box][local].mean()),
        "edge_band_fraction": float(field.edge_band[box][local].mean()),
        "nucleus_count": int(nucleus_ids.size),
        "nucleus_overlap_fraction": float((raw > 0).mean()) if raw.size else 0.0,
        "touches_crop_edge": bool(region.y0 == 0 or region.x0 == 0 or region.y0 + region.mask.shape[0] >= field.shape[0]
                                  or region.x0 + region.mask.shape[1] >= field.shape[1]),
    }
    return out
