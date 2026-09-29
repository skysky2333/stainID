from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes
from scipy.spatial import cKDTree

from stainid.qc.core_qc import tissue_mask


@dataclass(frozen=True)
class VoidLandmark:
    x: float
    y: float
    area_um2: float
    width_um: float
    height_um: float


@dataclass(frozen=True)
class VoidMatch:
    reference_index: int
    moving_index: int
    error_um: float
    area_ratio: float


def detect_void_landmarks(
    rgb: np.ndarray,
    microns_per_pixel: float,
    minimum_area_um2: float = 80.0,
    maximum_area_um2: float = 30000.0,
    maximum_aspect_ratio: float = 6.0,
) -> list[VoidLandmark]:
    tissue, _, _ = tissue_mask(rgb)
    holes = (binary_fill_holes(tissue) & ~tissue).astype(np.uint8)
    count, _, stats, centers = cv2.connectedComponentsWithStats(holes, 8)
    landmarks = []
    for index in range(1, count):
        area = float(stats[index, cv2.CC_STAT_AREA] * microns_per_pixel**2)
        width = float(stats[index, cv2.CC_STAT_WIDTH] * microns_per_pixel)
        height = float(stats[index, cv2.CC_STAT_HEIGHT] * microns_per_pixel)
        aspect = max(width, height) / max(min(width, height), microns_per_pixel)
        if (
            minimum_area_um2 <= area <= maximum_area_um2
            and aspect <= maximum_aspect_ratio
        ):
            landmarks.append(
                VoidLandmark(
                    float(centers[index, 0]),
                    float(centers[index, 1]),
                    area,
                    width,
                    height,
                )
            )
    return landmarks


def match_void_landmarks(
    reference: list[VoidLandmark],
    moving: list[VoidLandmark],
    microns_per_pixel: float,
    maximum_distance_um: float = 400.0,
    minimum_area_ratio: float = 0.25,
    maximum_area_ratio: float = 4.0,
) -> list[VoidMatch]:
    if not reference or not moving:
        return []
    reference_points = np.array([(item.x, item.y) for item in reference])
    moving_points = np.array([(item.x, item.y) for item in moving])
    reference_tree = cKDTree(reference_points)
    moving_tree = cKDTree(moving_points)
    distances, reference_indices = reference_tree.query(moving_points)
    _, moving_indices = moving_tree.query(reference_points)
    matches = []
    for moving_index, (distance, reference_index) in enumerate(
        zip(distances, reference_indices)
    ):
        area_ratio = moving[moving_index].area_um2 / reference[reference_index].area_um2
        if (
            moving_indices[reference_index] == moving_index
            and distance * microns_per_pixel <= maximum_distance_um
            and minimum_area_ratio <= area_ratio <= maximum_area_ratio
        ):
            matches.append(
                VoidMatch(
                    int(reference_index),
                    moving_index,
                    float(distance * microns_per_pixel),
                    float(area_ratio),
                )
            )
    return sorted(matches, key=lambda item: item.reference_index)


def crop_centered(rgb: np.ndarray, center: tuple[float, float], size: int) -> np.ndarray:
    half = size // 2
    x = round(center[0]) - half
    y = round(center[1]) - half
    output = np.full((size, size, 3), 255, dtype=np.uint8)
    left = max(0, x)
    top = max(0, y)
    right = min(rgb.shape[1], x + size)
    bottom = min(rgb.shape[0], y + size)
    if right > left and bottom > top:
        output[top - y : bottom - y, left - x : right - x] = rgb[top:bottom, left:right]
    return output


def render_landmark_pages(
    reference_rgb: np.ndarray,
    moving_rgb: np.ndarray,
    warped_rgb: np.ndarray,
    rows: list[dict[str, object]],
    output_prefix: str,
    crop_size: int = 220,
    rows_per_page: int = 10,
) -> list[tuple[str, np.ndarray]]:
    pages = []
    for page_index, start in enumerate(range(0, len(rows), rows_per_page), start=1):
        page_rows = rows[start : start + rows_per_page]
        canvas = np.full((64 + len(page_rows) * (crop_size + 38), crop_size * 3, 3), 245, np.uint8)
        cv2.putText(
            canvas,
            "NeuN reference                 coarse moving                 refined moving",
            (12, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (25, 25, 25),
            2,
            cv2.LINE_AA,
        )
        for row_index, row in enumerate(page_rows):
            y = 64 + row_index * (crop_size + 38)
            reference_center = (float(row["reference_x_px"]), float(row["reference_y_px"]))
            moving_center = (float(row["coarse_x_px"]), float(row["coarse_y_px"]))
            refined_center = (float(row["refined_x_px"]), float(row["refined_y_px"]))
            crops = [
                crop_centered(reference_rgb, reference_center, crop_size),
                crop_centered(moving_rgb, moving_center, crop_size),
                crop_centered(warped_rgb, refined_center, crop_size),
            ]
            for column, crop in enumerate(crops):
                center = (crop_size // 2, crop_size // 2)
                cv2.circle(crop, center, 12, (20, 190, 30), 3, cv2.LINE_AA)
                canvas[y : y + crop_size, column * crop_size : (column + 1) * crop_size] = cv2.cvtColor(
                    crop, cv2.COLOR_RGB2BGR
                )
            label = (
                f"{row['candidate_id']}  coarse={float(row['coarse_error_um']):.1f} um  "
                f"refined={float(row['refined_error_um']):.1f} um  review: accept / reject"
            )
            cv2.putText(
                canvas,
                label,
                (8, y + crop_size + 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (25, 25, 25),
                1,
                cv2.LINE_AA,
            )
        pages.append((f"{output_prefix}_page{page_index}.jpg", canvas))
    return pages


__all__ = [
    "VoidLandmark",
    "VoidMatch",
    "crop_centered",
    "detect_void_landmarks",
    "match_void_landmarks",
    "render_landmark_pages",
]
