"""Candidate geometry: a region is a boolean mask in a box of the field crop, plus where it came from.

Regions are kept as small local masks (never field-sized), so a tile can hold thousands of them. `Occupancy` paints kept
regions into one label image of the field so the overlap of a new region with every kept one is a lookup in its box.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy import ndimage


@dataclass
class Region:
    y0: int
    x0: int
    mask: np.ndarray  # bool, local to the box
    source: str  # the source whose geometry this is: profile, cellpose, dab@0.75, split@0.5, ring, dense, legacy, …
    level: float | None = None  # the DAB level (multiple of the slide threshold) of a threshold geometry
    sources: list[str] = field(default_factory=list)
    style: str = ""  # which legacy feature set describes it (profile, cellpose, amyloid, tau)
    legacy: dict = field(default_factory=dict)  # the legacy row (features as the old pipeline computed them), if any
    flags: dict = field(default_factory=dict)  # proposal-time facts the rules use (legacy filters, …)

    def __post_init__(self):
        if not self.sources:
            self.sources = [self.source]

    @property
    def area_px(self) -> int:
        return int(self.mask.sum())

    @property
    def box(self) -> tuple[slice, slice]:
        return slice(self.y0, self.y0 + self.mask.shape[0]), slice(self.x0, self.x0 + self.mask.shape[1])

    def centroid(self) -> tuple[float, float]:
        ys, xs = np.nonzero(self.mask)
        return float(self.x0 + xs.mean()), float(self.y0 + ys.mean())

    def add_source(self, source: str) -> None:
        if source not in self.sources:
            self.sources.append(source)

    def full(self, shape: tuple[int, int]) -> np.ndarray:
        out = np.zeros(shape, bool)
        out[self.box] = self.mask
        return out

    def padded(self, pad: int, shape: tuple[int, int]) -> tuple[tuple[slice, slice], np.ndarray]:
        """The region's mask in a box grown by `pad` pixels (clipped to the field)."""
        y0, x0 = max(0, self.y0 - pad), max(0, self.x0 - pad)
        y1, x1 = min(shape[0], self.y0 + self.mask.shape[0] + pad), min(shape[1], self.x0 + self.mask.shape[1] + pad)
        local = np.zeros((y1 - y0, x1 - x0), bool)
        local[self.y0 - y0: self.y0 - y0 + self.mask.shape[0], self.x0 - x0: self.x0 - x0 + self.mask.shape[1]] = self.mask
        return (slice(y0, y1), slice(x0, x1)), local


def from_full(mask: np.ndarray, source: str, **kwargs) -> Region | None:
    """A region from a field-sized mask (cropped to its box); None when the mask is empty."""
    ys, xs = np.nonzero(mask)
    if not len(ys):
        return None
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    return Region(int(y0), int(x0), mask[y0:y1, x0:x1].copy(), source, **kwargs)


def label_regions(labels: np.ndarray, source: str, **kwargs) -> dict[int, Region]:
    """Every label of a label image as a region, by label."""
    out = {}
    for index, box in enumerate(ndimage.find_objects(labels), start=1):
        if box is None:
            continue
        local = labels[box] == index
        if local.any():
            out[index] = Region(box[0].start, box[1].start, local, source, **kwargs)
    return out


def components(mask: np.ndarray, connectivity: int = 8) -> list[tuple[tuple[slice, slice], np.ndarray, int]]:
    """Connected components of a mask: (box, local mask, area in px)."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity)
    out = []
    for index in range(1, count):
        x, y, w, h, area = (int(v) for v in stats[index])
        out.append(((slice(y, y + h), slice(x, x + w)), labels[y: y + h, x: x + w] == index, area))
    return out


class Occupancy:
    """The kept regions of a field painted into one label image (a pixel keeps the first region that took it)."""

    def __init__(self, shape: tuple[int, int]):
        self.labels = np.zeros(shape, np.int32)
        self.regions: list[Region] = []
        self.areas: list[int] = []

    def add(self, region: Region) -> int:
        self.regions.append(region)
        self.areas.append(region.area_px)
        index = len(self.regions)
        view = self.labels[region.box]
        view[region.mask & (view == 0)] = index
        return index - 1

    def overlaps(self, region: Region) -> dict[int, int]:
        """{index of a kept region: overlapping pixels} for every kept region the given one touches."""
        values = self.labels[region.box][region.mask]
        values = values[values > 0]
        if not values.size:
            return {}
        found, counts = np.unique(values, return_counts=True)
        return {int(f) - 1: int(c) for f, c in zip(found, counts)}


def same_object(region: Region, occupancy: Occupancy, same_iou: float, same_inside: float, covered: float,
                significant_px: int = 0, max_ratio: float = 3.0) -> tuple[str, int | None]:
    """How a new region relates to the kept ones: ("same", index) when it is one of them (IoU, or one holds the other),
    ("merge", None) when it spans two or more kept regions, ("covered", None) when kept regions already cover it, else
    ("new", None). A region holding a kept one counts as the same object up to `max_ratio` times its area (a NeuN soma
    around a Cellpose nucleus)."""
    overlaps = occupancy.overlaps(region)
    if not overlaps:
        return "new", None
    area = region.area_px
    held = [k for k, inter in overlaps.items() if inter >= 0.5 * occupancy.areas[k] and occupancy.areas[k] >= significant_px]
    if len(held) >= 2:
        return "merge", None
    best = max(overlaps, key=lambda k: overlaps[k] / (area + occupancy.areas[k] - overlaps[k]))
    inter = overlaps[best]
    iou = inter / (area + occupancy.areas[best] - inter)
    if iou >= same_iou or inter >= same_inside * area or (inter >= same_inside * occupancy.areas[best] and area <= max_ratio * occupancy.areas[best]):
        return "same", best
    if sum(overlaps.values()) > covered * area:
        return "covered", None
    return "new", None


def outline(region: Region, offset: tuple[int, int] = (0, 0), epsilon: float = 1.0) -> np.ndarray:
    """The region's outer boundary (largest piece) as an (n, 2) array of x, y pixel indices, simplified."""
    piece = np.pad(region.mask.astype(np.uint8), 1)
    contours, _ = cv2.findContours(piece, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros((0, 2), int)
    contour = max(contours, key=cv2.contourArea)
    if len(contour) > 3:
        contour = cv2.approxPolyDP(contour, epsilon, True)
    points = contour[:, 0, :] + np.array([region.x0 - 1 + offset[0], region.y0 - 1 + offset[1]])
    return points.astype(int)


def disk(radius_px: int) -> np.ndarray:
    radius_px = max(0, int(radius_px))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius_px + 1, 2 * radius_px + 1))
