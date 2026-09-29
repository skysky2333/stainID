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


def glial_cells(
    rgb: np.ndarray,
    dab_threshold: float,
    pixel_size_um: float,
    exclusion: np.ndarray,
    soma_thickness_um: float = 1.5,
    territory_um: float = 25.0,
) -> tuple[list[dict[str, float]], dict[str, np.ndarray]]:
    """DAB-positive glial somata with their assigned process skeleton (Iba1 microglia or GFAP astrocytes)."""
    tissue = field_tissue_mask(rgb) & ~exclusion
    dab = np.maximum(rgb_to_hed(rgb)[..., 2], 0).astype(np.float32)
    smooth = cv2.GaussianBlur(dab, (0, 0), 0.8)
    positive = remove_small_objects((smooth >= dab_threshold) & tissue, 6)
    thickness = cv2.distanceTransform(positive.astype(np.uint8), cv2.DIST_L2, 5) * pixel_size_um
    seeds = remove_small_objects(thickness >= soma_thickness_um, 4)
    soma = cv2.dilate(seeds.astype(np.uint8), _disk(max(1, round(2.0 / pixel_size_um)))).astype(bool) & positive
    count, soma_labels, stats, centroids = cv2.connectedComponentsWithStats(soma.astype(np.uint8), 8)
    keep = np.zeros(count, dtype=bool)
    area_um2 = stats[:, cv2.CC_STAT_AREA] * pixel_size_um**2
    keep[1:] = (area_um2[1:] >= 12.0) & (area_um2[1:] <= 500.0)
    soma_labels = np.where(keep[soma_labels], soma_labels, 0)
    ridge = sato(dab, sigmas=[1, 1.5, 2], black_ridges=False)
    processes = (smooth >= 0.5 * dab_threshold) & tissue & (ridge >= 0.15 * dab_threshold) & (soma_labels == 0)
    skeleton = skeletonize(remove_small_objects(processes, 8))
    distance, (iy, ix) = ndimage.distance_transform_edt(soma_labels == 0, return_indices=True)
    owner = soma_labels[iy, ix]
    owner[distance * pixel_size_um > territory_um] = 0
    neighbors = cv2.filter2D(skeleton.astype(np.uint8), cv2.CV_16S, np.ones((3, 3), np.uint8), borderType=cv2.BORDER_CONSTANT) - skeleton
    branch = skeleton & (neighbors >= 3)
    endpoints = skeleton & (neighbors == 1)
    length = np.bincount(owner[skeleton], minlength=count) * pixel_size_um
    branches = np.bincount(owner[branch], minlength=count)
    ends = np.bincount(owner[endpoints], minlength=count)
    mean_dab = ndimage.mean(dab, soma_labels, index=np.arange(count))
    rows = []
    for label in np.nonzero(keep)[0]:
        x, y, w, h, area = stats[label]
        mask = soma_labels[y : y + h, x : x + w] == label
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        perimeter = cv2.arcLength(max(contours, key=cv2.contourArea), True) * pixel_size_um
        area = area * pixel_size_um**2
        diameter = 2 * np.sqrt(area / np.pi)
        rows.append({
            "label": int(label), "centroid_x_px": float(centroids[label][0]), "centroid_y_px": float(centroids[label][1]),
            "soma_area_um2": float(area), "soma_circularity": float(4 * np.pi * area / perimeter**2) if perimeter else float("nan"),
            "soma_mean_dab": float(mean_dab[label]), "process_length_um": float(length[label]),
            "branch_points": int(branches[label]), "end_points": int(ends[label]),
            "ramification_index": float(length[label] / diameter),
        })
    return rows, {"tissue": tissue, "positive": positive, "soma_labels": soma_labels, "skeleton": skeleton}


def summarize_glia(rows: list[dict[str, float]], maps: dict[str, np.ndarray], inner: tuple[slice, slice], pixel_size_um: float) -> dict[str, float]:
    tissue = maps["tissue"][inner]
    area_mm2 = tissue.sum() * pixel_size_um**2 / 1e6
    inside = [r for r in rows if inner[1].start <= r["centroid_x_px"] < inner[1].stop and inner[0].start <= r["centroid_y_px"] < inner[0].stop]
    median = lambda key: float(np.median([r[key] for r in inside])) if len(inside) >= 5 else float("nan")
    points = np.array([(r["centroid_x_px"], r["centroid_y_px"]) for r in inside])
    if len(points) >= 5:
        from scipy.spatial import cKDTree

        nn = cKDTree(points).query(points, k=2)[0][:, 1] * pixel_size_um
        expected = 0.5 / np.sqrt(len(points) / (area_mm2 * 1e6))
        clark_evans = float(nn.mean() / expected)
    else:
        clark_evans = float("nan")
    return {
        "glia_tissue_area_mm2": area_mm2,
        "glia_cell_count": len(inside),
        "glia_cell_density_mm2": len(inside) / area_mm2 if area_mm2 else float("nan"),
        "glia_positive_area_fraction": float(maps["positive"][inner][tissue].mean()) if tissue.any() else float("nan"),
        "glia_process_length_density_mm_per_mm2": float(maps["skeleton"][inner][tissue].sum() * pixel_size_um / 1000 / area_mm2) if area_mm2 else float("nan"),
        "glia_soma_area_median_um2": median("soma_area_um2"),
        "glia_process_length_median_um": median("process_length_um"),
        "glia_ramification_median": median("ramification_index"),
        "glia_branch_points_median": median("branch_points"),
        "glia_clark_evans": clark_evans,
    }


def ring_enrichment(
    plaque_mask: np.ndarray,
    cell_points: np.ndarray,
    tissue: np.ndarray,
    pixel_size_um: float,
    rings_um: tuple[tuple[float, float], ...] = ((0.0, 25.0), (25.0, 50.0)),
    far_um: float = 100.0,
) -> dict[str, float]:
    distance = ndimage.distance_transform_edt(~plaque_mask) * pixel_size_um
    point_map = np.zeros(tissue.shape, dtype=np.int32)
    if len(cell_points):
        xs = np.clip(np.round(cell_points[:, 0]).astype(int), 0, tissue.shape[1] - 1)
        ys = np.clip(np.round(cell_points[:, 1]).astype(int), 0, tissue.shape[0] - 1)
        np.add.at(point_map, (ys, xs), 1)
    def density(region):
        area = (region & tissue).sum() * pixel_size_um**2 / 1e6
        return point_map[region & tissue].sum() / area if area > 0.002 else float("nan"), area
    far, far_area = density(distance > far_um)
    out = {"far_density_mm2": far, "far_area_mm2": far_area}
    for low, high in rings_um:
        value, area = density((distance > low) & (distance <= high)) if low > 0 else density(plaque_mask | ((distance > 0) & (distance <= high)))
        out[f"ring_{int(low)}_{int(high)}_density_mm2"] = value
        out[f"ring_{int(low)}_{int(high)}_area_mm2"] = area
        out[f"ring_{int(low)}_{int(high)}_enrichment"] = value / far if far and np.isfinite(far) and far > 0 else float("nan")
    return out


__all__ = [
    "glial_cells",
    "ring_enrichment",
    "summarize_glia",
]
