"""One tile's field crop (the tile and its context margin) with every derived picture the proposal stage needs, each
computed once (the legacy detection recomputed stain separation and the artifact masks several times per tile)."""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

import cv2
import numpy as np

from stainid.imaging.color import rgb_to_hed
from stainid.imaging.tissue import field_tissue_mask, fold_mask, linear_artifact_mask, tissue_area_mask
from stainid.v2.geometry import disk


@dataclass
class Field:
    tile: dict
    rgb: np.ndarray
    inner: tuple[slice, slice]
    origin: tuple[int, int]  # the crop's top-left corner in core pixels (x, y)
    threshold: float  # the slide's DAB threshold
    pixel_size_um: float
    manual: np.ndarray  # extra excluded areas (bool, crop); stage 2 passes none: people's areas only train and evaluate
    raw_nuclei: np.ndarray | None  # Cellpose labels of the crop (Find nuclei), None when missing
    glass: float | None = None
    smooth_px: float = 1.0
    background_um: float = 60.0
    memo: dict = field(default_factory=dict)  # values derived once per field by the stain modules
    # the tile's own field, for a local window cut around an object of it: its crop-wide levels are the tile's (none for
    # the tile's own field)
    reference: Field | None = None

    @property
    def shape(self) -> tuple[int, int]:
        return self.rgb.shape[:2]

    @property
    def pixel_width_um(self) -> float:
        return float(self.tile["pixel_width_um"])

    @property
    def pixel_height_um(self) -> float:
        return float(self.tile["pixel_height_um"])

    def inside(self, x: float, y: float, margin: float = 0) -> bool:
        """Whether a point (field pixels) lies in the tile's own area, grown by `margin` pixels on every side."""
        return self.inner[1].start - margin <= x < self.inner[1].stop + margin and self.inner[0].start - margin <= y < self.inner[0].stop + margin

    @cached_property
    def standard(self) -> np.ndarray:
        """The standard crop within this field: the tile and the project's context margin (`tile["context_px"]`). The
        crop-wide levels (the legacy segmentations' background median and spread, the AT8 edge-guard trigger, the stained
        levels) are taken over it, so a wider detection window shows more of each object at the levels the standard
        crop sets, and does not move them (a lower extent level joins neighbouring plaques). All of the field for a
        field cut with the standard margin."""
        out = np.zeros(self.shape, bool)
        out[self.standard_box] = True
        return out

    @property
    def standard_box(self) -> tuple[slice, slice]:
        """`standard` as the slices of the field it covers."""
        context = int(self.tile.get("context_px", 128))
        return (slice(min(self.shape[0], max(0, self.inner[0].start - context)), max(0, min(self.shape[0], self.inner[0].stop + context))),
                slice(min(self.shape[1], max(0, self.inner[1].start - context)), max(0, min(self.shape[1], self.inner[1].stop + context))))

    @cached_property
    def hed(self) -> np.ndarray:
        return rgb_to_hed(self.rgb)

    @cached_property
    def hematoxylin(self) -> np.ndarray:
        return np.maximum(self.hed[..., 0], 0.0).astype(np.float32)

    @cached_property
    def dab(self) -> np.ndarray:
        return np.maximum(self.hed[..., 2], 0.0).astype(np.float32)

    @cached_property
    def chromogen_fraction(self) -> np.ndarray:
        return self.dab / (self.hematoxylin + self.dab + 1e-6)

    @cached_property
    def smooth(self) -> np.ndarray:
        return cv2.GaussianBlur(self.dab, (0, 0), self.smooth_px)

    @cached_property
    def tissue(self) -> np.ndarray:
        """Tissue as the detection steps see it (fixed cut-off)."""
        return field_tissue_mask(self.rgb)

    @cached_property
    def tissue_area(self) -> np.ndarray:
        """Tissue for areas (the core's glass level, enclosed holes filled), made on the standard crop (`standard`): a
        hole is filled when the standard crop encloses it, so the tile's areas do not depend on the detection window
        (only the tile's own area is used)."""
        box = self.standard_box
        if box[0].stop <= box[0].start or box[1].stop <= box[1].start:
            return tissue_area_mask(self.rgb, self.glass)
        out = np.zeros(self.shape, bool)
        out[box] = tissue_area_mask(np.ascontiguousarray(self.rgb[box]), self.glass)
        return out

    @cached_property
    def linear(self) -> np.ndarray:
        return linear_artifact_mask(self.rgb)

    @cached_property
    def fold(self) -> np.ndarray:
        return fold_mask(self.rgb, self.pixel_size_um)

    @cached_property
    def exclusion(self) -> np.ndarray:
        """What 6E10 and AT8 leave out: detection's linear artifacts and folds (and `manual`)."""
        return self.linear | self.fold | self.manual

    @cached_property
    def neun_exclusion(self) -> np.ndarray:
        """What NeuN leaves out: detection's linear artifacts (no fold mask; and `manual`)."""
        return self.linear | self.manual

    @property
    def excluded(self) -> np.ndarray:
        """The excluded pixels of this field's stain: they count in no measure (tissue, stain, background, context)."""
        return self.neun_exclusion if self.tile.get("stain") == "NeuN" else self.exclusion

    @cached_property
    def valid(self) -> np.ndarray:
        """Tissue outside the excluded areas of this field's stain."""
        return self.tissue & ~self.excluded

    @cached_property
    def local_background(self) -> np.ndarray:
        """Tissue-weighted mean of the smoothed DAB over `background_um` (the legacy dense-body background)."""
        size = max(3, round(self.background_um / self.pixel_size_um))
        tissue = self.valid.astype(np.float32)  # excluded pixels are not background
        return cv2.blur(np.where(self.valid, self.smooth, 0), (size, size)) / np.maximum(cv2.blur(tissue, (size, size)), 1e-3)

    @cached_property
    def gradient(self) -> np.ndarray:
        gx = cv2.Sobel(self.smooth, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(self.smooth, cv2.CV_32F, 0, 1, ksize=3)
        return np.sqrt(gx * gx + gy * gy)

    @cached_property
    def edge_band(self) -> np.ndarray:
        """Within 5 µm of non-tissue (tissue edges, holes)."""
        return cv2.dilate((~self.tissue).astype(np.uint8), disk(max(1, round(5 / self.pixel_size_um)))).astype(bool)

    @cached_property
    def nuclei(self) -> np.ndarray:
        """Cellpose nuclei with enough hematoxylin (vacuoles dropped), as AT8 and 6E10 use them."""
        from stainid.nuclei import clean_nuclei

        if self.raw_nuclei is None:
            return np.zeros(self.shape, np.uint32)
        return clean_nuclei(self.raw_nuclei, self.rgb)

    # ------------------------------------------------ pictures for the context features (skimage colour maths, exact)

    @cached_property
    def skimage_hed(self) -> np.ndarray:
        from skimage.color import rgb2hed

        return rgb2hed(self.rgb)

    @cached_property
    def saturation(self) -> np.ndarray:
        """HSV saturation, computed as skimage's rgb2hsv does (same floating-point steps), without its hue."""
        scaled = self.rgb.astype(np.float64) / 255.0
        value = scaled.max(axis=2)
        delta = value - scaled.min(axis=2)
        with np.errstate(invalid="ignore", divide="ignore"):
            saturation = delta / value
        saturation[delta == 0.0] = 0.0
        return saturation

    @cached_property
    def gray(self) -> np.ndarray:
        return cv2.cvtColor(self.rgb, cv2.COLOR_RGB2GRAY)

    def release(self) -> None:
        """Free the big derived pictures (the field is done)."""
        for name in list(self.__dict__):
            if name not in ("tile", "rgb", "inner", "origin", "threshold", "pixel_size_um", "manual", "raw_nuclei", "glass",
                            "smooth_px", "background_um", "memo"):
                del self.__dict__[name]
        self.memo.clear()
