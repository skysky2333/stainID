from __future__ import annotations

import cv2
import numpy as np
from skimage.measure import regionprops

from stainid.imaging.color import rgb_to_hed
from stainid.masks.sam import WindowedSam

MASK_RULES = {"NeuN": 2, "6E10": 1, "AT8": "contrast"}
AREA_LIMITS_UM2 = {"NeuN": (15.0, 1500.0), "AT8": (15.0, 1500.0), "6E10": (15.0, 40000.0)}


def ring_contrast(mask: np.ndarray, dab: np.ndarray, ring_px: int = 4) -> float:
    ring = cv2.dilate(mask.astype(np.uint8), np.ones((2 * ring_px + 1,) * 2, np.uint8)).astype(bool) & ~mask
    return float(dab[mask].mean() - (dab[ring].mean() if ring.any() else 0.0))


def select_mask(masks: list[np.ndarray], dab: np.ndarray, stain: str) -> tuple[np.ndarray, int]:
    rule = MASK_RULES[stain]
    if rule == "contrast":
        index = max((1, 2), key=lambda i: ring_contrast(masks[i], dab) if masks[i].sum() >= 20 else -1e9)
        return masks[index], index
    return masks[rule], rule


def shape_features(mask: np.ndarray, dab: np.ndarray, pixel_size_um: float, dab_threshold: float,
                   core_threshold: float | None = None) -> dict[str, float]:
    props = regionprops(mask.astype(np.uint8))[0]
    area = props.area * pixel_size_um**2
    perimeter = props.perimeter * pixel_size_um
    values = dab[mask]
    features = {
        "area_um2": area,
        "equivalent_diameter_um": props.equivalent_diameter_area * pixel_size_um,
        "perimeter_um": perimeter,
        "major_axis_um": props.axis_major_length * pixel_size_um,
        "minor_axis_um": props.axis_minor_length * pixel_size_um,
        "elongation": props.axis_major_length / max(props.axis_minor_length, 1e-6),
        "eccentricity": props.eccentricity,
        "solidity": props.solidity,
        "circularity": 4 * np.pi * area / perimeter**2 if perimeter else float("nan"),
        "mean_dab_od": float(values.mean()),
        "p90_dab_od": float(np.quantile(values, 0.9)),
        "normalized_mean_dab": float(values.mean() / dab_threshold),
        "ring_contrast": ring_contrast(mask, dab),
    }
    if core_threshold is not None:
        smooth = cv2.GaussianBlur(dab, (0, 0), 1.0)
        core = mask & (smooth >= core_threshold)
        count, _ = cv2.connectedComponents(core.astype(np.uint8), connectivity=8)
        features |= {
            "dense_core_area_um2": float(core.sum() * pixel_size_um**2),
            "dense_core_fraction": float(core.sum() / mask.sum()),
            "dense_core_count": int(count - 1),
        }
    return features


def segment_objects(
    sam: WindowedSam,
    rgb: np.ndarray,
    seeds: list[dict[str, object]],
    stain: str,
    dab_threshold: float,
    pixel_size_um: float,
    core_threshold: float | None = None,
) -> tuple[list[dict[str, object]], np.ndarray]:
    dab = np.maximum(rgb_to_hed(rgb)[..., 2], 0).astype(np.float32)
    labels = np.zeros(rgb.shape[:2], dtype=np.int32)
    lower, upper = AREA_LIMITS_UM2[stain]
    sam.set_field(rgb)
    rows = []
    for seed in sorted(seeds, key=lambda s: (sam._origin(float(s["x"]), float(s["y"])), s["y"], s["x"])):
        masks, scores, origin = sam.predict(float(seed["x"]), float(seed["y"]))
        x0, y0 = origin
        local_dab = dab[y0 : y0 + masks[0].shape[0], x0 : x0 + masks[0].shape[1]]
        mask, index = select_mask(masks, local_dab, stain)
        area = mask.sum() * pixel_size_um**2
        status = "ok"
        if not lower <= area <= upper:
            status = "size_rejected"
        full = np.zeros(labels.shape, dtype=bool)
        full[y0 : y0 + mask.shape[0], x0 : x0 + mask.shape[1]] = mask
        overlap = labels[full]
        if status == "ok" and overlap.size and (overlap > 0).mean() > 0.5:
            status = "duplicate"
        row = {**seed, "sam_output": index, "sam_score": scores[index], "mask_status": status}
        if status == "ok":
            full &= labels == 0
            label = int(labels.max()) + 1
            labels[full] = label
            ys, xs = np.nonzero(full)
            pad = 8
            by0, by1 = max(0, ys.min() - pad), min(labels.shape[0], ys.max() + pad + 1)
            bx0, bx1 = max(0, xs.min() - pad), min(labels.shape[1], xs.max() + pad + 1)
            row |= {"label": label, "mask_centroid_x_px": float(xs.mean()), "mask_centroid_y_px": float(ys.mean()),
                    **shape_features(full[by0:by1, bx0:bx1], dab[by0:by1, bx0:bx1], pixel_size_um, dab_threshold, core_threshold)}
        rows.append(row)
    return rows, labels


__all__ = [
    "MASK_RULES",
    "segment_objects",
    "select_mask",
    "shape_features",
]
