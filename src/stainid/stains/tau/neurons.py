from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage

from stainid.imaging.color import rgb_to_hed
from stainid.imaging.tissue import field_tissue_mask
from stainid.stains.amyloid.classifier import context_features
from stainid.stains.neun.audit import crop_with_padding

TAU_FEATURES = (
    "normalized_contrast", "normalized_soma_dab", "soma_area_um2", "soma_solidity", "soma_circularity",
    "soma_elongation", "soma_dab_cv", "soma_dab_p90", "nucleus_count", "nucleus_overlap_fraction",
    "positive_fraction", "soma_hematoxylin_mean", "edge_fraction", "background_dab_median",
)


def _disk(radius_px: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius_px + 1, 2 * radius_px + 1))


def _shape(mask: np.ndarray, pixel_size_um: float) -> dict[str, float]:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv2.contourArea)
    area = float(mask.sum()) * pixel_size_um**2
    perimeter = cv2.arcLength(contour, True) * pixel_size_um
    hull = cv2.contourArea(cv2.convexHull(contour)) * pixel_size_um**2
    if len(contour) >= 5:
        _, (a, b), _ = cv2.fitEllipse(contour)
        major, minor = max(a, b) * pixel_size_um, max(min(a, b), 1e-6) * pixel_size_um
    else:
        major = minor = 2 * np.sqrt(area / np.pi)
    return {
        "area_um2": area,
        "solidity": area / hull if hull else 0.0,
        "circularity": 4 * np.pi * area / perimeter**2 if perimeter else 0.0,
        "elongation": major / minor,
    }


def _local_features(
    dab: np.ndarray,
    hematoxylin: np.ndarray,
    tissue: np.ndarray,
    nuclei: np.ndarray,
    center: tuple[float, float],
    soma: np.ndarray,
    origin: tuple[int, int],
    dab_threshold: float,
    pixel_size_um: float,
) -> dict[str, float]:
    x0, y0 = origin
    height, width = soma.shape
    local_dab = dab[y0 : y0 + height, x0 : x0 + width]
    local_tissue = tissue[y0 : y0 + height, x0 : x0 + width]
    inner = cv2.dilate(soma.astype(np.uint8), _disk(max(1, round(2 / pixel_size_um)))).astype(bool)
    outer = cv2.dilate(soma.astype(np.uint8), _disk(max(1, round(10 / pixel_size_um)))).astype(bool)
    background = outer & ~inner & local_tissue
    background_median = float(np.median(local_dab[background])) if background.any() else float("nan")
    values = local_dab[soma]
    local_nuclei = nuclei[y0 : y0 + height, x0 : x0 + width]
    nucleus_ids = np.unique(local_nuclei[soma])
    nucleus_ids = nucleus_ids[nucleus_ids > 0]
    nucleus_overlap = float(np.isin(local_nuclei, nucleus_ids).sum() / max(soma.sum(), 1)) if nucleus_ids.size else 0.0
    return {
        **{f"soma_{k}": v for k, v in _shape(soma, pixel_size_um).items()},
        "soma_dab_mean": float(values.mean()),
        "soma_dab_p90": float(np.quantile(values, 0.9)),
        "soma_dab_cv": float(values.std() / max(values.mean(), 1e-6)),
        "background_dab_median": background_median,
        "normalized_soma_dab": float(values.mean() / dab_threshold),
        "normalized_contrast": float((values.mean() - background_median) / dab_threshold),
        "soma_hematoxylin_mean": float(hematoxylin[y0 : y0 + height, x0 : x0 + width][soma].mean()),
        "nucleus_count": int(nucleus_ids.size),
        "nucleus_overlap_fraction": nucleus_overlap,
        "positive_fraction": float((values >= dab_threshold).mean()),
    }


def tau_neuron_candidates(
    rgb: np.ndarray,
    nuclei: np.ndarray,
    dab_threshold: float,
    pixel_size_um: float,
    exclusion: np.ndarray | None = None,
    ring_um: float = 5.0,
    minimum_contrast: float = 0.3,
) -> tuple[list[dict[str, object]], np.ndarray]:
    tissue = field_tissue_mask(rgb)
    if exclusion is not None:
        tissue &= ~exclusion
    hed = rgb_to_hed(rgb)
    hematoxylin = np.maximum(hed[..., 0], 0).astype(np.float32)
    dab = np.maximum(hed[..., 2], 0).astype(np.float32)
    smooth = cv2.GaussianBlur(dab, (0, 0), 1.0 / pixel_size_um)
    height, width = dab.shape
    margin = max(1, round(14 / pixel_size_um))
    candidates: list[tuple[str, tuple[float, float], np.ndarray, tuple[int, int]]] = []

    ring_px = max(1, round(ring_um / pixel_size_um))
    for label, bounds in enumerate(ndimage.find_objects(nuclei), start=1):
        if bounds is None:
            continue
        y0 = max(0, bounds[0].start - margin)
        y1 = min(height, bounds[0].stop + margin)
        x0 = max(0, bounds[1].start - margin)
        x1 = min(width, bounds[1].stop + margin)
        nucleus = nuclei[y0:y1, x0:x1] == label
        area = nucleus.sum() * pixel_size_um**2
        local_tissue = tissue[y0:y1, x0:x1]
        if not 10.0 <= area <= 200.0 or local_tissue[nucleus].mean() < 0.5:
            continue
        ring = cv2.dilate(nucleus.astype(np.uint8), _disk(ring_px)).astype(bool) & ~nucleus & local_tissue
        far = cv2.dilate(nucleus.astype(np.uint8), _disk(ring_px + round(8 / pixel_size_um))).astype(bool) & ~cv2.dilate(nucleus.astype(np.uint8), _disk(ring_px + round(2 / pixel_size_um))).astype(bool) & local_tissue
        if ring.sum() < 20 or far.sum() < 20:
            continue
        local = smooth[y0:y1, x0:x1]
        background = float(np.median(local[far]))
        if (float(local[ring].mean()) - background) / dab_threshold < minimum_contrast:
            continue
        stained = (local >= background + 0.5 * dab_threshold) & (ring | nucleus)
        soma = cv2.morphologyEx((stained | nucleus).astype(np.uint8), cv2.MORPH_CLOSE, _disk(2)).astype(bool)
        ys, xs = np.nonzero(nucleus)
        candidates.append(("nucleus_ring", (x0 + xs.mean(), y0 + ys.mean()), soma, (x0, y0)))

    taken = np.zeros(dab.shape, dtype=bool)
    for _, _, soma, (x0, y0) in candidates:
        taken[y0 : y0 + soma.shape[0], x0 : x0 + soma.shape[1]] |= soma
    local_background = cv2.blur(np.where(tissue, smooth, 0), (round(60 / pixel_size_um),) * 2) / np.maximum(
        cv2.blur(tissue.astype(np.float32), (round(60 / pixel_size_um),) * 2), 1e-3
    )
    strong = ((smooth - local_background) >= max(1.0 * dab_threshold, 0.03)) & tissue
    thick = cv2.distanceTransform(strong.astype(np.uint8), cv2.DIST_L2, 5) * pixel_size_um >= 2.5
    body = cv2.dilate(thick.astype(np.uint8), _disk(max(1, round(2 / pixel_size_um)))).astype(bool) & strong
    count, components, stats, centroids = cv2.connectedComponentsWithStats(body.astype(np.uint8), 8)
    for component in range(1, count):
        x, y, w, h, area = stats[component]
        if not 40.0 <= area * pixel_size_um**2 <= 1500.0:
            continue
        x0, y0 = max(0, x - margin), max(0, y - margin)
        x1, y1 = min(width, x + w + margin), min(height, y + h + margin)
        soma = components[y0:y1, x0:x1] == component
        if taken[y0:y1, x0:x1][soma].mean() > 0.3:
            continue
        candidates.append(("dense_body", tuple(centroids[component]), soma, (x0, y0)))

    non_tissue = ~field_tissue_mask(rgb)
    edge_band = cv2.dilate(non_tissue.astype(np.uint8), _disk(max(1, round(5 / pixel_size_um)))).astype(bool)
    objects: list[dict[str, object]] = []
    labels = np.zeros(dab.shape, dtype=np.int32)
    for kind, (cx, cy), soma, (x0, y0) in candidates:
        features = _local_features(dab, hematoxylin, tissue, nuclei, (cx, cy), soma, (x0, y0), dab_threshold, pixel_size_um)
        if features["normalized_contrast"] < minimum_contrast:
            continue
        features["edge_fraction"] = float(edge_band[y0 : y0 + soma.shape[0], x0 : x0 + soma.shape[1]][soma].mean())
        number = len(objects) + 1
        target = labels[y0 : y0 + soma.shape[0], x0 : x0 + soma.shape[1]]
        target[soma & (target == 0)] = number
        objects.append({"label": number, "candidate_source": kind, "centroid_x_px": float(cx), "centroid_y_px": float(cy), **features})
    return objects, labels


def add_context_features(rgb: np.ndarray, objects: list[dict[str, object]], pixel_size_um: float, context_um: float = 55.0) -> list[dict[str, object]]:
    size = max(64, round(context_um / pixel_size_um))
    rows = []
    for obj in objects:
        crop = crop_with_padding(rgb, float(obj["centroid_x_px"]), float(obj["centroid_y_px"]), size)
        diameter = 2 * np.sqrt(float(obj["soma_area_um2"]) / np.pi)
        rows.append({**obj, **{f"ctx_{k}": v for k, v in context_features(crop, diameter, context_um=context_um).items()}})
    return rows


def tau_feature_matrix(rows: list[dict[str, object]]) -> tuple[np.ndarray, list[str]]:
    """Shape, stain and context features of tau+ neuron candidates (the tau random forest's input)."""
    context = [k for k in rows[0] if str(k).startswith("ctx_")]
    names = list(TAU_FEATURES) + context + ["from_ring"]
    matrix = np.asarray([[float(row.get(n, np.nan)) for n in names[:-1]] + [float(row["candidate_source"] == "nucleus_ring")] for row in rows])
    return matrix, names


def classify_tau_neurons(rows: list[dict[str, object]], bundle: dict[str, object], pixel_size_um: float) -> list[dict[str, object]]:
    if not rows:
        return []
    names = bundle["feature_names"]
    matrix = np.asarray(
        [[float(row[name]) for name in names[:-1]] + [float(row["candidate_source"] == "nucleus_ring")] for row in rows]
    )
    probability = bundle["classifier"].predict_proba(matrix)[:, 1]
    order = np.argsort(-probability)
    radius_px = float(bundle["nms_radius_um"]) / pixel_size_um
    kept: list[int] = []
    for index in order:
        if probability[index] < float(bundle["probability_threshold"]):
            break
        x, y = float(rows[index]["centroid_x_px"]), float(rows[index]["centroid_y_px"])
        if all((x - float(rows[k]["centroid_x_px"])) ** 2 + (y - float(rows[k]["centroid_y_px"])) ** 2 > radius_px**2 for k in kept):
            kept.append(int(index))
    kept_set = set(kept)
    return [
        {
            **row,
            "tau_neuron_probability": float(value),
            "tau_neuron": index in kept_set,
            "tau_neuron_mature": index in kept_set and float(row["soma_dab_p90"]) >= float(bundle["mature_p90_threshold"]),
        }
        for index, (row, value) in enumerate(zip(rows, probability))
    ]


__all__ = [
    "TAU_FEATURES",
    "add_context_features",
    "classify_tau_neurons",
    "tau_feature_matrix",
    "tau_neuron_candidates",
]
