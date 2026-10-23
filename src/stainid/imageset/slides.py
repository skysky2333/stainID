"""Whole slides put back together for viewing: a display and layout layer over the regions of a whole-slide project.

*Prepare images* cuts each slide into regions of tissue (`stainid.imageset.prepare`, a grid of `REGION_PX` squares at
the analysis scale, each cut down to its tissue); the regions stay the units of every pipeline step, label and table.
This module only says where each region lies on its slide, so the Cores view and the annotation pages can show the
whole section as one picture (`stainid.api.slide_pyramid`) and draw everything in slide coordinates:

- a slide's coordinates are analysis pixels (`pixel_size_um`) of the whole scan, from an `origin` on the regions' grid
  (the grid cell of the top-left region), so every region lies inside one cell of `CELL` px and the cells line up with
  the 512 px tiles of the slide picture;
- a region's offset is its `source_box` (level-0 pixels of the scan) times its `scale`, less the origin;
- `to_slide` and `to_region` map points between the two.

Nothing here reads an image or writes a file: it reads the core image table only.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from stainid.imageset.prepare import REGION_PX

CELL = REGION_PX  # slide px per grid cell: a region never crosses a cell's edge


@dataclass(frozen=True)
class Region:
    core_id: str
    x: int  # the region's top-left corner on the slide (slide px)
    y: int
    width: int
    height: int

    def contains(self, x: float, y: float) -> bool:
        return self.x <= x < self.x + self.width and self.y <= y < self.y + self.height


@dataclass(frozen=True)
class Slide:
    slide_id: str
    stain: str
    name: str  # the image id as the image list has it
    donor_id: str
    region: str
    tma: str  # the batch its stain threshold is set for (`prepare.batch_of`)
    origin: tuple[int, int]  # slide px (0, 0) in the scan's analysis pixels
    width: int
    height: int
    regions: tuple[Region, ...] = field(default=())

    @property
    def cores(self) -> list[str]:
        return [r.core_id for r in self.regions]

    def find(self, core_id: str) -> Region | None:
        return next((r for r in self.regions if r.core_id == core_id), None)

    def at(self, x: float, y: float) -> Region | None:
        """The region holding a slide point (None on glass between regions)."""
        return next((r for r in self.regions if r.contains(x, y)), None)

    def to_slide(self, core_id: str, x: float, y: float) -> tuple[float, float]:
        region = self.find(core_id)
        if region is None:
            raise KeyError(f"{core_id} is not a region of {self.slide_id}")
        return x + region.x, y + region.y

    def to_region(self, x: float, y: float) -> tuple[str, float, float] | None:
        region = self.at(x, y)
        return None if region is None else (region.core_id, x - region.x, y - region.y)

    def overlapping(self, x0: float, y0: float, x1: float, y1: float) -> list[Region]:
        """The regions a slide box (x0, y0)-(x1, y1) touches."""
        return [r for r in self.regions if r.x < x1 and r.x + r.width > x0 and r.y < y1 and r.y + r.height > y0]

    def to_api(self) -> dict:
        return {"slide_id": self.slide_id, "stain": self.stain, "name": self.name, "donor_id": self.donor_id, "region": self.region,
                "tma": self.tma, "width": self.width, "height": self.height, "origin": list(self.origin),
                "regions": [[r.core_id, r.x, r.y, r.width, r.height] for r in self.regions]}


def _text(value) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(int(value)) if isinstance(value, float) and value.is_integer() else str(value)


def slide_id_of(image_id: str, stain: str, stains_of_image: int = 1) -> str:
    """A slide's id: its image id (a slide is one scan, one stain); with the stain added when an image id has several."""
    from stainid.imaging.core_images import safe_name  # noqa: PLC0415

    return safe_name(image_id if stains_of_image <= 1 else f"{image_id}_{stain}")


def layouts(core_images: pd.DataFrame) -> dict[str, Slide]:
    """Every slide of a whole-slide project's core image table, by id, with its regions' places (empty for a table
    without `source_box`, e.g. a TMA study or an image set)."""
    if core_images is None or core_images.empty or not {"image_id", "source_box", "core_id", "stain"} <= set(core_images):
        return {}
    rows = core_images.dropna(subset=["source_box"]).drop_duplicates(["core_id", "stain"])
    stains = rows.groupby(rows.image_id.astype(str)).stain.nunique().to_dict()
    out: dict[str, Slide] = {}
    for (image_id, stain), group in rows.groupby([rows.image_id.astype(str), rows.stain.astype(str)], sort=True):
        places = []
        for r in group.to_dict("records"):
            box = [float(v) for v in str(r["source_box"]).split()]
            if len(box) != 4:
                continue
            scale = float(r.get("scale") or 1.0)
            places.append((str(r["core_id"]), int(round(box[0] * scale)), int(round(box[1] * scale)),
                           int(r.get("native_width_px") or round(box[2] * scale)), int(r.get("native_height_px") or round(box[3] * scale))))
        if not places:
            continue
        ox = min(p[1] for p in places) // CELL * CELL
        oy = min(p[2] for p in places) // CELL * CELL
        regions = tuple(sorted((Region(c, x - ox, y - oy, w, h) for c, x, y, w, h in places), key=lambda r: (r.y, r.x)))
        first = group.iloc[0]
        slide_id = slide_id_of(image_id, stain, int(stains.get(image_id, 1)))
        out[slide_id] = Slide(slide_id, stain, image_id, _text(first.get("donor_id")), _text(first.get("region")), _text(first.get("tma")), (ox, oy),
                              max(r.x + r.width for r in regions), max(r.y + r.height for r in regions), regions)
    return out


def by_core(slides: dict[str, Slide]) -> dict[str, Slide]:
    """The slide of every region (core id -> slide)."""
    return {core: slide for slide in slides.values() for core in slide.cores}


# ---------------------------------------------------------------- moving drawings between region and slide pixels

def shift_flat(flat, dx: float, dy: float, digits: int | None = None) -> list:
    """A flat ring or line [x0, y0, x1, y1, …] moved by (dx, dy)."""
    out = np.asarray(flat, float).reshape(-1)
    if out.size % 2:
        return list(flat)
    out = out.reshape(-1, 2) + (dx, dy)
    values = out.reshape(-1)
    return [round(float(v), digits) if digits is not None else (int(v) if float(v).is_integer() else float(v)) for v in values]


def shift_ring(ring, dx: float, dy: float) -> list[list[float]]:
    """A ring [[x, y], …] moved by (dx, dy)."""
    return [[float(p[0]) + dx, float(p[1]) + dy, *p[2:]] for p in ring]


def split_mask(mask: np.ndarray, scale: float, slide: Slide) -> dict[str, tuple[np.ndarray, float, float]]:
    """A mask painted over the whole slide (1 mask px = `scale` slide px) cut into one mask per region it touches: the
    region's part of the mask on the mask's own grid, and the (dx, dy) in region px to add to rings found in it (the
    grid's cells do not start at the region's corner, so this keeps the rings exactly where they were painted)."""
    out = {}
    height, width = mask.shape[:2]
    for region in slide.regions:
        mx0, my0 = int(region.x // scale), int(region.y // scale)
        mx1 = min(width, int(math.ceil((region.x + region.width) / scale)))
        my1 = min(height, int(math.ceil((region.y + region.height) / scale)))
        if mx1 <= mx0 or my1 <= my0:
            continue
        part = mask[my0:my1, mx0:mx1]
        if not part.any():
            continue
        out[region.core_id] = (part, mx0 * scale - region.x, my0 * scale - region.y)
    return out


def clip_ring(ring, region: Region) -> list[list[float]]:
    """A ring (region px) kept inside its region's box (points beyond it moved onto the edge)."""
    return [[min(max(float(p[0]), 0.0), float(region.width)), min(max(float(p[1]), 0.0), float(region.height))] for p in ring]
