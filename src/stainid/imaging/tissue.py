from __future__ import annotations

import cv2
import numpy as np

from stainid.imaging.color import rgb_to_hed


def linear_artifact_mask(rgb: np.ndarray) -> np.ndarray:
    height, width = rgb.shape[:2]
    minimum_dimension = min(height, width)
    kernel_size = min(81, max(31, round(minimum_dimension * 0.035)))
    kernel_size += 1 - kernel_size % 2
    center = kernel_size // 2
    radius = center
    thickness = max(1, kernel_size // 25)
    darkness = 1.0 - cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    non_brown = rgb[..., 0].astype(np.int16) <= rgb[..., 2].astype(np.int16) + 15
    tissue = field_tissue_mask(rgb)
    distance = cv2.distanceTransform(tissue.astype(np.uint8), cv2.DIST_L2, 5)
    internal = tissue & (distance >= max(5, kernel_size // 10))
    darkness = cv2.GaussianBlur(darkness * non_brown * internal, (0, 0), 2.0)
    candidates = np.zeros((height, width), dtype=np.uint8)
    for angle in range(0, 180, 15):
        radians = np.deg2rad(angle)
        dx = round(radius * np.cos(radians))
        dy = round(radius * np.sin(radians))
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.uint8)
        cv2.line(
            kernel,
            (center - dx, center - dy),
            (center + dx, center + dy),
            1,
            thickness,
        )
        opened = cv2.morphologyEx(darkness, cv2.MORPH_OPEN, kernel)
        candidates |= (opened >= 0.35).astype(np.uint8)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidates, 8)
    artifact = np.zeros((height, width), dtype=np.uint8)
    minimum_area = 0.002 * height * width
    minimum_span = 0.35 * max(height, width)
    for index in range(1, count):
        area = stats[index, cv2.CC_STAT_AREA]
        span = max(
            stats[index, cv2.CC_STAT_WIDTH],
            stats[index, cv2.CC_STAT_HEIGHT],
        )
        if area >= minimum_area and span >= minimum_span:
            artifact[labels == index] = 1
    dilation = max(5, round(kernel_size * 0.2))
    if dilation % 2 == 0:
        dilation += 1
    return cv2.dilate(
        artifact,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation, dilation)),
    ).astype(bool)


def field_tissue_mask(rgb: np.ndarray) -> np.ndarray:
    scaled = np.maximum(rgb.astype(np.float32) / 255.0, 1.0 / 255.0)
    darkness = np.max(-np.log(scaled), axis=2)
    mask = (darkness >= 0.10).astype(np.uint8)
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
    )
    return mask.astype(bool)


def fold_mask(rgb: np.ndarray, pixel_size_um: float) -> np.ndarray:
    tissue = field_tissue_mask(rgb)
    hematoxylin = np.maximum(rgb_to_hed(rgb)[..., 0], 0.0).astype(np.float32)
    blurred = cv2.GaussianBlur(hematoxylin * tissue, (0, 0), 5.0 / pixel_size_um)
    support = cv2.GaussianBlur(tissue.astype(np.float32), (0, 0), 5.0 / pixel_size_um)
    blurred = np.divide(blurred, support, out=np.zeros_like(blurred), where=support > 0.5)
    values = blurred[tissue & (support > 0.5)]
    if values.size < 1000:
        return np.zeros(tissue.shape, dtype=bool)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    candidate = (blurred >= median + max(0.02, 5.0 * 1.4826 * mad)) & tissue
    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate.astype(np.uint8), 8)
    mask = np.zeros(tissue.shape, dtype=bool)
    for index in range(1, count):
        component = labels == index
        area_um2 = stats[index, cv2.CC_STAT_AREA] * pixel_size_um**2
        if area_um2 < 1000.0:
            continue
        contours, _ = cv2.findContours(component.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        (_, _), (w, h), _ = cv2.minAreaRect(max(contours, key=cv2.contourArea))
        length_um = max(w, h) * pixel_size_um
        width_um = area_um2 / max(length_um, 1e-6)
        if length_um >= 150.0 and length_um / max(width_um, 1e-6) >= 4.0:
            mask |= component
    radius = max(1, round(5.0 / pixel_size_um))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)
