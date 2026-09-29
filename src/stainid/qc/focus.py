from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class FocusTile:
    x: int
    y: int
    width: int
    height: int
    tissue_pixels: int
    score: float
    relative_score: float
    low_focus: bool


@dataclass(frozen=True)
class FocusQc:
    reference: float
    p10: float
    tile_count: int
    low_focus_tile_count: int
    low_focus_tissue_fraction: float


def measure_focus(
    rgb: np.ndarray,
    tissue: np.ndarray,
    tile_size: int = 128,
    minimum_tissue_fraction: float = 0.50,
    low_focus_ratio: float = 0.15,
) -> tuple[FocusQc, list[FocusTile]]:
    if rgb.shape[:2] != tissue.shape:
        raise ValueError("RGB image and tissue mask dimensions disagree")
    if tile_size < 16:
        raise ValueError("Focus tiles must be at least 16 pixels wide")
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    candidates = []
    for y in range(0, gray.shape[0], tile_size):
        for x in range(0, gray.shape[1], tile_size):
            height = min(tile_size, gray.shape[0] - y)
            width = min(tile_size, gray.shape[1] - x)
            local_tissue = tissue[y : y + height, x : x + width]
            tissue_pixels = int(local_tissue.sum())
            if tissue_pixels < minimum_tissue_fraction * width * height:
                continue
            local_gray = gray[y : y + height, x : x + width]
            score = float(cv2.Laplacian(local_gray, cv2.CV_32F).var())
            candidates.append((x, y, width, height, tissue_pixels, score))
    if not candidates:
        return FocusQc(float("nan"), float("nan"), 0, 0, 0.0), []

    scores = np.array([candidate[-1] for candidate in candidates])
    reference = float(np.quantile(scores, 0.75))
    threshold = low_focus_ratio * reference
    tiles = [
        FocusTile(
            x=x,
            y=y,
            width=width,
            height=height,
            tissue_pixels=tissue_pixels,
            score=score,
            relative_score=score / reference if reference else 0.0,
            low_focus=score < threshold,
        )
        for x, y, width, height, tissue_pixels, score in candidates
    ]
    total_tissue = sum(tile.tissue_pixels for tile in tiles)
    low_focus_tissue = sum(tile.tissue_pixels for tile in tiles if tile.low_focus)
    qc = FocusQc(
        reference=reference,
        p10=float(np.quantile(scores, 0.10)),
        tile_count=len(tiles),
        low_focus_tile_count=sum(tile.low_focus for tile in tiles),
        low_focus_tissue_fraction=low_focus_tissue / total_tissue,
    )
    return qc, tiles


def low_focus_mask(shape: tuple[int, int], tiles: list[FocusTile]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for tile in tiles:
        if tile.low_focus:
            mask[tile.y : tile.y + tile.height, tile.x : tile.x + tile.width] = True
    return mask


__all__ = [
    "FocusQc",
    "FocusTile",
    "low_focus_mask",
    "measure_focus",
]
