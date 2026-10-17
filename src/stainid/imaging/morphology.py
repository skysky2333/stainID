"""Exact fast binary dilation of small masks by OpenCV's elliptic (disk) structuring element.

`cv2.dilate` with a large elliptic kernel visits every kernel pixel for every image pixel; the per-object loops of the
detection (a ring of 5 µm and 13 µm around each of thousands of nuclei, the 10 µm background of each tau candidate)
spent most of their time there. A disk is a stack of centred horizontal segments, so the dilation of a mask is, row by
row, the union of its pixel runs widened by each segment: one interval per (run, kernel row), painted with a difference
array. The result is the same mask as `cv2.dilate(mask, disk)` (pixels beyond the mask's own array are not touched, as
OpenCV's default border does), in a fraction of the time when the kernel is big compared with the object.
"""
from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np

SMALL_RADIUS = 10  # below this OpenCV's own dilation is as quick


@lru_cache(maxsize=None)
def disk(radius_px: int) -> np.ndarray:
    radius_px = max(0, int(radius_px))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius_px + 1, 2 * radius_px + 1))


@lru_cache(maxsize=None)
def _disk_rows(radius_px: int) -> tuple[np.ndarray, np.ndarray]:
    """(row offsets, half widths) of the disk kernel: row dy holds the columns -w … w (checked)."""
    kernel = disk(radius_px)
    offsets, halves = [], []
    for row, values in enumerate(kernel):
        columns = np.flatnonzero(values)
        if not columns.size:
            continue
        half = radius_px - int(columns[0])
        if int(columns[-1]) != radius_px + half or columns.size != 2 * half + 1:
            raise ValueError(f"disk({radius_px}) row {row} is not a centred segment")
        offsets.append(row - radius_px)
        halves.append(half)
    return np.asarray(offsets, np.int64), np.asarray(halves, np.int64)


def runs(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(row, first column, column after the last) of every horizontal run of True pixels, row by row."""
    height, width = mask.shape
    padded = np.zeros((height, width + 2), np.int8)
    padded[:, 1:-1] = mask
    steps = np.diff(padded, axis=1)
    rows, starts = np.nonzero(steps == 1)
    _, stops = np.nonzero(steps == -1)
    return rows, starts, stops


def dilate_disk(mask: np.ndarray, radius_px: int, found_runs=None) -> np.ndarray:
    """`cv2.dilate(mask.astype(np.uint8), disk(radius_px)).astype(bool)`, computed from the mask's runs (pass
    `found_runs=runs(mask)` to dilate one mask by several radii)."""
    height, width = mask.shape
    if radius_px < SMALL_RADIUS and found_runs is None:
        return cv2.dilate(mask.astype(np.uint8), disk(radius_px)).astype(bool)
    rows, starts, stops = runs(mask) if found_runs is None else found_runs
    if not rows.size:
        return np.zeros((height, width), bool)
    offsets, halves = _disk_rows(int(radius_px))
    out_rows = rows[:, None] + offsets[None, :]
    inside = (out_rows >= 0) & (out_rows < height)
    first = np.clip(starts[:, None] - halves[None, :], 0, width)
    after = np.clip(stops[:, None] + halves[None, :], 0, width)
    base = out_rows * (width + 1)
    size = height * (width + 1)
    marks = np.bincount((base + first)[inside], minlength=size) - np.bincount((base + after)[inside], minlength=size)
    return (np.cumsum(marks.reshape(height, width + 1), axis=1)[:, :width] > 0)


__all__ = ["dilate_disk", "disk", "runs"]
