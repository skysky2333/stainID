"""NeuN proposals: the legacy DAB profiles and Cellpose cells (with every legacy filter recorded instead of applied),
merged profiles above 650 µm² split by a marker watershed, and the DAB component tree / local-background components.

Order (the first geometry of an object is kept, later ones only add their source tag):
1. legacy profiles (the old candidate outlines, kept unchanged so the current model scores them as before);
2. Cellpose cells the legacy combination kept, then the profiles the legacy size / shape filter dropped (flagged);
3. component-tree and local-background components (stain-visible profiles the chromogen-fraction rule broke up, dark
   neurons in brown bands, merged neurons the legacy filter dropped above 650 µm², split at higher DAB levels or by
   the tree's watershed, …);
4. the Cellpose cells the legacy combination dropped (beside a profile, pale, excluded): merged into the DAB region they
   lie in, else kept as their own candidates with the legacy reason flagged.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage

from stainid.v2 import scaled
from stainid.v2.geometry import Occupancy, Region, same_object
from stainid.v2.tree import local_background_regions, tree_regions

MEASUREMENT = ("selection_area_um2", "stained_area_um2", "profile_area_um2", "area_um2", "aspect_ratio", "solidity", "circularity",
               "mean_dab_od", "mean_hematoxylin_od", "neighborhood_dab_p80", "chromogen_fraction", "mean_chromogen_fraction",
               "tissue_fraction")


def _largest(mask: np.ndarray) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if count <= 1:
        return np.zeros(mask.shape, bool)
    return labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))


def stained_mask(field) -> np.ndarray:
    """The legacy profile stain mask: tissue, not excluded, DAB above 0.35 × threshold and at least half chromogen."""
    if "neun_stained" not in field.memo:
        minimum = max(0.012, 0.35 * field.threshold)
        field.memo["neun_stained"] = field.tissue & ~field.neun_exclusion & (field.dab >= minimum) & (field.chromogen_fraction >= 0.50)
    return field.memo["neun_stained"]


def profile_row(field, component: np.ndarray, box: tuple[slice, slice]) -> tuple[dict, np.ndarray] | None:
    """`stainid.stains.neun.profiles.segment_dab_profiles` for one component (local mask in `box`): the legacy row and the
    profile mask (contour filled, tissue, not excluded), or None when the profile has no pixel left."""
    y, x = box[0].start, box[1].start
    height, width = component.shape
    pixel_area = field.pixel_width_um * field.pixel_height_um
    contours, _ = cv2.findContours(component.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    stained_area = int(stained_mask(field)[box][component].sum()) * pixel_area
    profile_area = float(cv2.contourArea(contour)) * pixel_area
    aspect = max(width / height, height / width)
    filled = np.zeros(component.shape, np.uint8)
    cv2.drawContours(filled, [contour], -1, 1, -1)
    profile = filled.astype(bool) & field.tissue[box] & ~field.neun_exclusion[box]
    if not profile.any():
        return None
    ys, xs = np.nonzero(profile)
    perimeter = float(cv2.arcLength(contour, True))
    contour_area = float(cv2.contourArea(contour))
    hull = float(cv2.contourArea(cv2.convexHull(contour)))
    solidity = contour_area / hull if hull else 0.0
    circularity = 4.0 * np.pi * contour_area / perimeter ** 2 if perimeter else 0.0
    image_height, image_width = field.shape
    touches = bool(x == 0 or y == 0 or x + width == image_width or y + height == image_height)
    accepted = bool(not touches and 25.0 <= profile_area <= 350.0 and aspect <= 2.5 and solidity >= 0.55 and circularity >= 0.15)
    row = {
        "centroid_x_px": float(x + xs.mean()), "centroid_y_px": float(y + ys.mean()),
        "stained_area_um2": stained_area, "profile_area_um2": profile_area, "aspect_ratio": aspect, "solidity": solidity,
        "circularity": circularity, "mean_dab_od": float(field.dab[box][profile].mean()),
        "mean_chromogen_fraction": float(field.chromogen_fraction[box][profile].mean()),
        "candidate_class": "positive_profile" if accepted else "review", "touches_edge": touches,
        "source_kind": "dab_profile", "source_candidate_class": "", "selection_area_um2": profile_area,
        "legacy_kept": bool(stained_area >= 4.0 and 15.0 <= profile_area <= 650.0 and aspect <= 3.5),
    }
    return row, profile


def legacy_profiles(field) -> list[tuple[dict, Region, np.ndarray, tuple[slice, slice]]]:
    """Every closed stain component as (legacy row, profile region, component mask, component box)."""
    closed = cv2.morphologyEx(stained_mask(field).astype(np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(closed, 8)
    out = []
    for index in range(1, count):
        x, y, w, h, _ = (int(v) for v in stats[index])
        box = (slice(y, y + h), slice(x, x + w))
        component = labels[box] == index
        made = profile_row(field, component, box)
        if made is None:
            continue
        row, profile = made
        out.append((row, Region(y, x, profile, "profile", style="profile", legacy=row), component, box))
    return out


def cellpose_cells(field) -> list[tuple[dict, Region]]:
    """`stainid.stains.neun.cellpose.classify_cellpose_masks`: each Cellpose cell (at least half tissue, 8–650 µm²) with
    its legacy row (largest piece inside valid tissue, DAB neighbourhood within 5 px)."""
    if field.raw_nuclei is None:
        return []
    valid = field.tissue & ~field.neun_exclusion  # a cell's pixels in an excluded area are not there (half or more: no cell)
    pixel_area = field.pixel_width_um * field.pixel_height_um
    height, width = field.shape
    out = []
    for raw_label, box in enumerate(ndimage.find_objects(field.raw_nuclei), start=1):
        if box is None:
            continue
        ys_, xs_ = box
        raw_local = field.raw_nuclei[box] == raw_label
        tissue_fraction = float(valid[box][raw_local].mean())
        if tissue_fraction < 0.5:
            continue
        local = _largest(raw_local & valid[box])
        area = int(local.sum()) * pixel_area
        if not 8.0 <= area <= 650.0:
            continue
        y0, y1 = max(0, ys_.start - 5), min(height, ys_.stop + 5)
        x0, x1 = max(0, xs_.start - 5), min(width, xs_.stop + 5)
        grown = np.zeros((y1 - y0, x1 - x0), bool)
        grown[ys_.start - y0: ys_.stop - y0, xs_.start - x0: xs_.stop - x0] = local
        neighbourhood = cv2.dilate(grown.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))).astype(bool)
        neighbourhood &= valid[y0:y1, x0:x1]
        values = field.dab[box][local]
        mean_dab = float(values.mean())
        p80 = float(np.quantile(field.dab[y0:y1, x0:x1][neighbourhood], 0.8))
        positive = mean_dab >= 0.5 * field.threshold or p80 >= field.threshold
        review = not positive and (mean_dab >= 0.4 * field.threshold or p80 >= 0.75 * field.threshold)
        ys, xs = np.nonzero(local)
        hematoxylin = float(field.hematoxylin[box][local].mean())
        contours, _ = cv2.findContours(local.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=cv2.contourArea)
        contour_area, perimeter = float(cv2.contourArea(contour)), float(cv2.arcLength(contour, True))
        hull = float(cv2.contourArea(cv2.convexHull(contour)))
        _, _, wb, hb = cv2.boundingRect(contour)
        source_class = "positive" if positive else "review" if review else "negative"
        chromogen = mean_dab / (hematoxylin + mean_dab + 1e-6)
        row = {
            "centroid_x_px": float(xs_.start + xs.mean()), "centroid_y_px": float(ys_.start + ys.mean()),
            "area_um2": area, "tissue_fraction": tissue_fraction, "mean_hematoxylin_od": hematoxylin, "mean_dab_od": mean_dab,
            "neighborhood_dab_p80": p80, "source_candidate_class": source_class, "chromogen_fraction": chromogen,
            "equivalent_diameter_um": 2.0 * np.sqrt(area / np.pi), "aspect_ratio": max(wb / max(hb, 1), hb / max(wb, 1)),
            "solidity": contour_area / hull if hull else 0.0,
            "circularity": 4.0 * np.pi * contour_area / perimeter ** 2 if perimeter else 0.0,
            "touches_edge": bool(xs_.start == 0 or ys_.start == 0 or xs_.stop == width or ys_.stop == height),
            "source_kind": "cellpose", "selection_area_um2": area,
            "candidate_class": "review" if (mean_dab >= max(0.010, 0.25 * field.threshold) and chromogen >= 0.24)
            or source_class in {"positive", "review"} else "negative",
        }
        out.append((row, Region(ys_.start, xs_.start, local, "cellpose", style="cellpose", legacy=row)))
    return out


def profile_style_row(field, region: Region) -> dict:
    """Legacy profile features of a region found by another source (component tree, watershed piece)."""
    box = region.box
    pixel_area = field.pixel_width_um * field.pixel_height_um
    padded = np.pad(region.mask.astype(np.uint8), 1)
    contours, _ = cv2.findContours(padded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    height, width = region.mask.shape
    perimeter, contour_area = float(cv2.arcLength(contour, True)), float(cv2.contourArea(contour))
    hull = float(cv2.contourArea(cv2.convexHull(contour)))
    aspect = max(width / height, height / width)
    profile_area = contour_area * pixel_area
    solidity = contour_area / hull if hull else 0.0
    circularity = 4.0 * np.pi * contour_area / perimeter ** 2 if perimeter else 0.0
    image_height, image_width = field.shape
    touches = bool(region.x0 == 0 or region.y0 == 0 or region.x0 + width == image_width or region.y0 + height == image_height)
    accepted = bool(not touches and 25.0 <= profile_area <= 350.0 and aspect <= 2.5 and solidity >= 0.55 and circularity >= 0.15)
    x, y = region.centroid()
    return {
        "centroid_x_px": x, "centroid_y_px": y,
        "stained_area_um2": int(stained_mask(field)[box][region.mask].sum()) * pixel_area, "profile_area_um2": profile_area,
        "aspect_ratio": aspect, "solidity": solidity, "circularity": circularity,
        "mean_dab_od": float(field.dab[box][region.mask].mean()),
        "mean_chromogen_fraction": float(field.chromogen_fraction[box][region.mask].mean()),
        "candidate_class": "positive_profile" if accepted else "review", "touches_edge": touches,
        "source_kind": "dab_profile", "source_candidate_class": "", "selection_area_um2": profile_area,
        "legacy_kept": bool(15.0 <= profile_area <= 650.0 and aspect <= 3.5),
    }


def propose(field, settings: dict, shared: dict) -> list[Region]:
    """Every NeuN candidate region of the field (context margin included)."""
    pixel_area = field.pixel_size_um ** 2
    occupancy = Occupancy(field.shape)
    same = dict(same_iou=shared["same_iou"], same_inside=shared["same_inside"], covered=shared["covered"],
                significant_px=round(settings["significant_um2"] / pixel_area))

    def offer(region: Region) -> Region | None:
        """Keep a region unless it is an object already kept (then it only adds its source), spans several, or is covered."""
        how, index = same_object(region, occupancy, **same, max_ratio=settings.get("same_max_ratio", 8.0))
        if how == "same":
            occupancy.regions[index].add_source(region.source)
            return None
        if how != "new":
            return None
        occupancy.add(region)
        return region

    shape_failed = []
    for row, region, component, box in legacy_profiles(field):
        if row["legacy_kept"]:
            region.flags["legacy_kept"] = True
            occupancy.add(region)
        elif settings["min_um2"] <= row["profile_area_um2"] <= settings["max_um2"]:
            region.flags["legacy_kept"] = False
            region.flags["legacy_drop"] = "shape"
            shape_failed.append(region)
        # a merged profile (above max_um2) is left to the component tree: its higher levels separate the cells, and a
        # merge no level separates is cut into cells (`stainid.v2.tree.cell_pieces`)

    cells = cellpose_cells(field)
    supported = cv2.dilate(np.isin(occupancy.labels, [i + 1 for i, r in enumerate(occupancy.regions) if r.flags.get("legacy_kept")]).astype(np.uint8),
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (49, 49))).astype(bool)
    height, width = field.shape
    later = []
    for row, region in cells:
        xi = min(width - 1, max(0, round(row["centroid_x_px"])))
        yi = min(height - 1, max(0, round(row["centroid_y_px"])))
        reason = ""
        if field.manual[yi, xi] or field.manual[region.box][region.mask].mean() >= 0.25:
            reason = "excluded"
        elif supported[yi, xi]:
            reason = "near_profile"
        elif row["source_candidate_class"] == "negative" and row["mean_hematoxylin_od"] < 0.025:
            reason = "pale_nucleus"
        region.flags["legacy_drop"] = reason
        region.flags["legacy_kept"] = not reason
        if reason:
            later.append(region)
        else:
            offer(region)

    for region in shape_failed:  # after the legacy cells, so a cell the legacy pipeline kept stays the candidate there
        offer(region)
    allowed = field.tissue & ~field.neun_exclusion
    for region in tree_regions(field, settings["levels"], allowed, settings["min_um2"], settings["max_um2"], settings["significant_um2"],
                               settings["split_max_um2"], settings["mode"], settings.get("split")):
        region.style = "profile"
        offer(region)
    if settings.get("local_contrast"):
        for region in local_background_regions(field, settings["local_contrast"], allowed, settings["min_um2"], settings["max_um2"]):
            region.style = "profile"
            offer(region)
    for region in later:
        overlaps = occupancy.overlaps(region)
        holder = max(overlaps, key=overlaps.get) if overlaps else None
        if holder is not None and overlaps[holder] >= 0.5 * region.area_px:
            occupancy.regions[holder].add_source("cellpose")
        else:
            occupancy.add(region)
    for region in occupancy.regions:
        if not region.legacy:
            region.legacy = profile_style_row(field, region)
    return occupancy.regions


def model_inputs(field, regions: list[Region], contexts: list[dict]) -> list[dict]:
    """The current NeuN model's inputs (`stainid.stains.neun.classifier.build_matrix_from_contexts`), as m_ columns, and
    their slide-scaled versions (`s_`, `stainid.v2.scaled`)."""
    out = []
    for region, context in zip(regions, contexts):
        row = region.legacy
        values = {f"m_{name}": float(row[name]) if row.get(name) not in (None, "") else float("nan") for name in MEASUREMENT}
        values |= {
            "m_source_is_dab_profile": float(row["source_kind"] == "dab_profile"),
            "m_proposal_is_positive_profile": float(row["candidate_class"] == "positive_profile"),
            "m_source_candidate_is_positive": float(row["source_candidate_class"] == "positive"),
            "m_source_candidate_is_review": float(row["source_candidate_class"] == "review"),
        }
        values |= {f"m_{k}": v for k, v in context.items()}
        values |= scaled.intensities("NeuN", values, field.threshold)  # slide-scaled inputs (stainid.v2.scaled)
        values["s_stained_area_um2"] = scaled.neun_stained_area(field, region, values["m_stained_area_um2"])
        out.append(values)
    return out


def context_diameters(regions: list[Region]) -> list[float]:
    return [2.0 * np.sqrt(float(r.legacy["selection_area_um2"]) / np.pi) for r in regions]

