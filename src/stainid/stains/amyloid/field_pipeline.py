from __future__ import annotations

import cv2
import numpy as np

from stainid.imaging.color import rgb_to_hed
from stainid.imaging.tissue import field_tissue_mask
from stainid.nuclei import clean_nuclei, nucleus_centroids
from stainid.pipelines.cohort_v1 import centered_objects
from stainid.stains.amyloid.model import classify_amyloid_objects, flag_vascular_or_edge
from stainid.stains.amyloid.segmentation import segment_amyloid_candidates


def _disk(radius_px: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius_px + 1, 2 * radius_px + 1))


def analyze_6e10_v2(
    rgb: np.ndarray,
    inner: tuple[slice, slice],
    nuclei: np.ndarray,
    dab_threshold: float,
    pixel_size_um: float,
    exclusion: np.ndarray,
    amyloid_bundle: dict[str, object],
    ring_um: float = 15.0,
) -> tuple[dict[str, float | int], list[dict[str, object]]]:
    nuclei = clean_nuclei(nuclei, rgb)
    _, objects, plaque_labels, _, excluded = segment_amyloid_candidates(rgb, dab_threshold, pixel_size_um, pixel_size_um, exclusion, split_touching=True)
    objects = classify_amyloid_objects(rgb, objects, dab_threshold, pixel_size_um, amyloid_bundle)
    objects = flag_vascular_or_edge(objects, plaque_labels, ~field_tissue_mask(rgb))
    valid = field_tissue_mask(rgb) & ~excluded
    accepted_all = [row for row in objects if row["candidate_class"] in {"compact", "diffuse", "small_plaque"}]
    accepted_mask = np.isin(plaque_labels, [int(row["label"]) for row in accepted_all])
    ring_px = max(1, round(ring_um / pixel_size_um))
    near_any = cv2.dilate(accepted_mask.astype(np.uint8), _disk(ring_px)).astype(bool)
    points = nucleus_centroids(nuclei, valid, pixel_size_um)
    point_map = np.zeros(valid.shape, dtype=np.int32)
    if len(points):
        np.add.at(point_map, (points[:, 1], points[:, 0]), 1)
    far = valid & ~cv2.dilate(accepted_mask.astype(np.uint8), _disk(3 * ring_px)).astype(bool)
    background_density = point_map[far].sum() / max(far.sum() * pixel_size_um**2 / 1e6, 1e-9)
    dab = np.maximum(rgb_to_hed(rgb)[..., 2], 0).astype(np.float32)
    core_threshold = float(amyloid_bundle["morphotype_threshold"]) * dab_threshold
    plaques = []
    for row in centered_objects(accepted_all, inner):
        label = int(row["label"])
        rows_, cols_ = np.nonzero(plaque_labels == label)
        y0, y1 = max(0, rows_.min() - ring_px - 1), min(valid.shape[0], rows_.max() + ring_px + 2)
        x0, x1 = max(0, cols_.min() - ring_px - 1), min(valid.shape[1], cols_.max() + ring_px + 2)
        mask = plaque_labels[y0:y1, x0:x1] == label
        ring = cv2.dilate(mask.astype(np.uint8), _disk(ring_px)).astype(bool) & ~mask & valid[y0:y1, x0:x1] & ~accepted_mask[y0:y1, x0:x1]
        core = mask & (cv2.GaussianBlur(dab[y0:y1, x0:x1], (0, 0), 1.0) >= core_threshold)
        area = mask.sum() * pixel_size_um**2
        ring_area = ring.sum() * pixel_size_um**2
        plaques.append({
            "centroid_x_px": row["centroid_x_px"], "centroid_y_px": row["centroid_y_px"],
            "plaque_class": row["candidate_class"], "plaque_probability": row["plaque_probability"],
            "plaque_area_um2": float(area), "dense_core_area_um2": float(core.sum() * pixel_size_um**2),
            "dense_core_fraction": float(core.sum() / max(mask.sum(), 1)),
            "equivalent_diameter_um": float(row["equivalent_diameter_um"]),
            "circularity": float(row["circularity"]), "solidity": float(row["solidity"]),
            "boundary_irregularity": float(row["boundary_irregularity"]),
            "normalized_inner_dab": float(row["normalized_inner_dab"]),
            "nuclei_in_plaque": int(point_map[y0:y1, x0:x1][mask].sum()),
            "nuclei_in_ring": int(point_map[y0:y1, x0:x1][ring].sum()),
            "ring_area_um2": float(ring_area),
        })
    inner_valid = valid[inner]
    area_mm2 = inner_valid.sum() * pixel_size_um**2 / 1e6
    summary = {
        "v2_tissue_area_mm2": area_mm2,
        "v2_plaque_count": len(plaques),
        "v2_vascular_or_edge_count": sum(
            row["candidate_class"] == "vascular_or_edge" for row in centered_objects(objects, inner)
        ),
        "v2_plaque_area_sum_um2": float(sum(p["plaque_area_um2"] for p in plaques)),
        "v2_dense_core_area_sum_um2": float(sum(p["dense_core_area_um2"] for p in plaques)),
        "v2_ring_nuclei": int(sum(p["nuclei_in_ring"] for p in plaques)),
        "v2_ring_area_um2": float(sum(p["ring_area_um2"] for p in plaques)),
        "v2_plaque_nuclei": int(sum(p["nuclei_in_plaque"] for p in plaques)),
        "v2_background_nucleus_density_mm2": float(background_density),
        "v2_far_area_mm2": float(far.sum() * pixel_size_um**2 / 1e6),
        "v2_far_nuclei": int(point_map[far].sum()),
        "nucleus_count": int(point_map[inner][inner_valid].sum()),
        "nucleus_density_mm2": float(point_map[inner][inner_valid].sum() / area_mm2) if area_mm2 else float("nan"),
        "plaque_near_area_fraction": float((near_any[inner] & inner_valid).sum() / max(inner_valid.sum(), 1)),
    }
    return summary, plaques
