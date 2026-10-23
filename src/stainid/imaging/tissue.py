from __future__ import annotations

import cv2
import numpy as np

from stainid.imaging.color import rgb_to_hed
from stainid.imaging.parallel import map_all
from stainid.imaging.shared import remembered


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
    dark = (darkness >= 0.35).astype(np.uint8)

    def opened(angle: int) -> np.ndarray:
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
        # a flat opening commutes with a threshold: open(darkness) >= 0.35 is the opening of (darkness >= 0.35), which
        # is the same mask computed on bytes (several times faster than on floats)
        return cv2.morphologyEx(dark, cv2.MORPH_OPEN, kernel)

    for found in map_all(opened, range(0, 180, 15)):
        candidates |= found

    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidates, 8)
    minimum_area = 0.002 * height * width
    minimum_span = 0.35 * max(height, width)
    span = np.maximum(stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT])
    keep = (stats[:, cv2.CC_STAT_AREA] >= minimum_area) & (span >= minimum_span)
    keep[0] = False
    artifact = keep.astype(np.uint8)[labels]
    dilation = max(5, round(kernel_size * 0.2))
    if dilation % 2 == 0:
        dilation += 1
    return cv2.dilate(
        artifact,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation, dilation)),
    ).astype(bool)


GLASS_MAX_CHROMA = 8
GLASS_MIN_LEVEL = 180
GLASS_MIN_FRACTION = 0.02
TISSUE_DARKNESS = 0.10
GLASS_MARGIN = 0.04
MAX_HOLE_PX = 130_000


def core_glass_darkness(image) -> float | None:
    """How dark the bare slide glass is around one core image (PIL image), from bright, colourless pixels in the
    outer 4% of the image, which is slide margin. Some scanners render glass light grey (about 230/255), darker than
    the fixed tissue cut-off. Estimated once per core image: an estimate from a single field mistakes pale tissue
    (white matter, neuropil, vacuoles) for glass whenever the field shows no glass, and then undercounts tissue."""
    small = np.asarray(image.reduce(8).convert("RGB")).astype(np.int16)
    border = max(4, round(0.04 * min(small.shape[:2])))
    ring = np.concatenate([small[:border].reshape(-1, 3), small[-border:].reshape(-1, 3),
                           small[:, :border].reshape(-1, 3), small[:, -border:].reshape(-1, 3)])
    return _glass_level(ring, len(ring))


def slide_glass_darkness(rgb: np.ndarray) -> float | None:
    """How dark the bare glass of a whole slide is, from a small picture of the whole slide (RGB, analysis colours):
    the bright, colourless pixels around its tissue (outside `field_tissue_mask`). A region cut out of a slide has
    little glass at its edge, often none, so `core_glass_darkness` would take the pale tissue at its edge for glass;
    the slide's own glass is measured once for all its regions instead. Pure white (outside the scanned area) is left
    out."""
    pixels = rgb[~field_tissue_mask(rgb)].astype(np.int16)
    return _glass_level(pixels[pixels.min(axis=1) < 254], rgb.shape[0] * rgb.shape[1])


def _glass_level(pixels: np.ndarray, out_of: int) -> float | None:
    """The darkness of the glass among `pixels` (N x 3): the median of the bright, colourless ones, when at least
    GLASS_MIN_FRACTION of `out_of` are."""
    glass = pixels[(pixels.max(axis=1) - pixels.min(axis=1) <= GLASS_MAX_CHROMA) & (pixels.min(axis=1) >= GLASS_MIN_LEVEL)]
    if len(glass) < GLASS_MIN_FRACTION * out_of:
        return None
    return float(np.max(-np.log(np.maximum(np.median(glass, axis=0), 1.0) / 255.0)))


def field_tissue_mask(rgb: np.ndarray, glass: float | None = None) -> np.ndarray:
    """Pixels darker than bare glass. The detection steps use the fixed cut-off they were trained with (glass=None);
    tissue-area measurements pass the core's glass darkness (core_glass_darkness, see tissue_area_mask)."""
    if glass is None:
        return remembered("tissue", rgb, lambda: _field_tissue_mask(rgb, None))
    return _field_tissue_mask(rgb, glass)


def _darkness(levels: np.ndarray) -> np.ndarray:
    return -np.log(np.maximum(levels.astype(np.float32) / 255.0, 1.0 / 255.0))


# the darkness of each 8-bit level; it falls with the level, so a pixel's darkest channel is its lowest level
_DARKNESS = _darkness(np.arange(256, dtype=np.uint8))
_DARKNESS_FALLS = bool(np.all(np.diff(_DARKNESS) <= 0))


def _field_tissue_mask(rgb: np.ndarray, glass: float | None) -> np.ndarray:
    if rgb.dtype == np.uint8 and _DARKNESS_FALLS:
        darkness = _DARKNESS[rgb.min(axis=2)]
    else:
        darkness = np.max(_darkness(rgb), axis=2)
    threshold = TISSUE_DARKNESS if glass is None else max(TISSUE_DARKNESS, glass + GLASS_MARGIN)
    mask = (darkness >= threshold).astype(np.uint8)
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


def tissue_area_mask(rgb: np.ndarray, glass: float | None = None) -> np.ndarray:
    """Tissue for area measurements (the denominator of every density): darker than the core's glass (`glass`, from
    core_glass_darkness), so a light-grey slide background is not counted as tissue, and with enclosed holes up to
    MAX_HOLE_PX (vacuoles, small vessel lumens) counted as tissue."""
    solid = field_tissue_mask(rgb, glass)
    count, labels, stats, _ = cv2.connectedComponentsWithStats((~solid).astype(np.uint8), connectivity=4)
    border = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    fill = stats[:, cv2.CC_STAT_AREA] <= MAX_HOLE_PX
    fill[border] = False
    return solid | (fill[labels] & ~solid)


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
    keep = np.zeros(count, dtype=bool)
    for index in range(1, count):
        area_um2 = stats[index, cv2.CC_STAT_AREA] * pixel_size_um**2
        if area_um2 < 1000.0:
            continue
        # the component's own box (one pixel wider where the picture allows): the same contour as in the whole picture
        x, y, w, h = (int(v) for v in stats[index, :4])
        x0, y0 = max(0, x - 1), max(0, y - 1)
        component = labels[y0: y + h + 1, x0: x + w + 1] == index
        contours, _ = cv2.findContours(component.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contour = max(contours, key=cv2.contourArea) + np.array([x0, y0], dtype=np.int32)
        (_, _), (w, h), _ = cv2.minAreaRect(contour)
        length_um = max(w, h) * pixel_size_um
        width_um = area_um2 / max(length_um, 1e-6)
        if length_um >= 150.0 and length_um / max(width_um, 1e-6) >= 4.0:
            keep[index] = True
    mask = keep[labels]
    radius = max(1, round(5.0 / pixel_size_um))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)
