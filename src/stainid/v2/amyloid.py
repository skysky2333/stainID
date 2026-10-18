"""6E10 proposals: every deposit of the legacy segmentation (touching plaques split at h-maxima, as the detection step
does), including those the legacy shape rule called artifacts and never showed the classifier, plus faint deposits of
the component tree below the legacy seed level (roots at 0.5 × and 0.75 × the slide threshold)."""
from __future__ import annotations

import cv2
import numpy as np

from stainid.nuclei import nucleus_centroids
from stainid.stains.amyloid.classifier import MORPHOLOGY_FEATURES
from stainid.stains.amyloid.model import identity_features, outer_boundary_contact
from stainid.stains.amyloid.segmentation import _radial_features, _remove_small_components, _shape_features, segment_amyloid_candidates
from stainid.v2 import scaled
from stainid.v2.geometry import Occupancy, Region, label_regions, same_object
from stainid.v2.tree import coarse_regions, tree_regions, walk_roots

CORE_QUANTILES = np.linspace(0, 1, 21)


def _levels(field) -> dict:
    """The legacy segmentation's background statistics, seed and extent thresholds and smoothed DAB for this field (the
    statistics over its standard crop, `Field.standard`, as the segmentation takes them)."""
    if "amyloid" not in field.memo:
        valid = field.tissue & ~field.exclusion
        dab = field.dab
        sample = valid & field.standard
        values = dab[sample] if sample.sum() >= 100 else dab[valid] if valid.sum() >= 100 else dab[field.tissue] if field.tissue.any() else dab.ravel()
        median = float(np.median(values))
        sigma = max(1.4826 * float(np.median(np.abs(values - median))), 0.001)
        seed_threshold = max(0.035, float(field.threshold), median + 6.0 * sigma)
        extent_threshold = max(0.010, 0.30 * seed_threshold, median + 2.5 * sigma)
        local_dab = cv2.GaussianBlur(dab, (0, 0), 1.2)
        pixel_area = field.pixel_width_um * field.pixel_height_um
        seed = _remove_small_components(valid & (local_dab >= seed_threshold), max(1, round(3.0 / pixel_area)))
        field.memo["amyloid"] = {"valid": valid, "seed_threshold": seed_threshold, "extent_threshold": extent_threshold,
                                  "local_dab": local_dab, "seed": seed, "background_median": median, "background_sigma": sigma}
    return field.memo["amyloid"]


def deposit_row(field, region: Region) -> dict:
    """The legacy per-deposit measurements (`segment_amyloid_candidates`' loop) of any region."""
    levels = _levels(field)
    pixel_area = field.pixel_width_um * field.pixel_height_um
    box, mask = region.box, region.mask
    dab, local_dab = field.dab[box], levels["local_dab"][box]
    area = int(mask.sum()) * pixel_area
    shape = _shape_features(mask, pixel_area)
    x, y = region.x0 + shape["bbox_x_px"], region.y0 + shape["bbox_y_px"]
    width, height = shape["bbox_width_px"], shape["bbox_height_px"]
    image_height, image_width = field.shape
    touches = bool(x == 0 or y == 0 or x + width == image_width or y + height == image_height)
    values = dab[mask]
    core_threshold = max(0.10, 1.20 * levels["seed_threshold"], float(np.quantile(values, 0.85)))
    core = _remove_small_components(mask & (local_dab >= core_threshold), max(1, round(4.0 / pixel_area)))
    count, core_labels, core_stats, _ = cv2.connectedComponentsWithStats(core.astype(np.uint8), 8)
    largest = core_labels == 1 + int(np.argmax(core_stats[1:, cv2.CC_STAT_AREA])) if count > 1 else np.zeros(core.shape, bool)
    core_area = int(largest.sum()) * pixel_area
    radial = _radial_features(mask, dab)
    if largest.any():
        cy, cx = np.nonzero(largest)
        offset = np.hypot(float(cx.mean()) - radial["centroid_x_px"], float(cy.mean()) - radial["centroid_y_px"]) * np.sqrt(pixel_area)
    else:
        offset = float("nan")
    radius = np.sqrt(area / np.pi)
    stained = int((mask & (dab >= levels["extent_threshold"])).sum()) * pixel_area
    return {
        "centroid_x_px": region.x0 + radial["centroid_x_px"], "centroid_y_px": region.y0 + radial["centroid_y_px"],
        "deposit_area_um2": area, "equivalent_diameter_um": 2.0 * radius, "stained_area_um2": stained,
        "stain_fill_fraction": stained / area, "seed_area_um2": int((mask & levels["seed"][box]).sum()) * pixel_area,
        "core_area_um2": core_area, "core_fraction": core_area / area, "core_threshold_od": core_threshold,
        "normalized_core_offset": offset / radius if largest.any() and radius else float("nan"),
        "mean_dab_od": float(values.mean()), "median_dab_od": float(np.median(values)), "p90_dab_od": float(np.quantile(values, 0.9)),
        "max_dab_od": float(values.max()), "inner_mean_dab_od": radial["inner_mean_dab_od"], "outer_mean_dab_od": radial["outer_mean_dab_od"],
        "radial_dab_contrast": radial["radial_dab_contrast"], **{k: shape[k] for k in ("perimeter_um", "circularity", "solidity", "aspect_ratio",
                                                                                   "eccentricity", "major_axis_um", "minor_axis_um", "boundary_irregularity")},
        "touches_edge": touches, "candidate_class": "review", "decision_basis": "v2_component_tree",
    }


def propose(field, settings: dict, shared: dict) -> list[Region]:
    _, objects, labels, _, _ = segment_amyloid_candidates(field.rgb, field.threshold, field.pixel_width_um, field.pixel_height_um,
                                                          field.exclusion, split_touching=True, statistics_mask=field.standard)
    regions = label_regions(labels, "legacy", style="amyloid")
    occupancy = Occupancy(field.shape)
    for row in objects:
        region = regions.get(int(row["label"]))
        if region is None:
            continue
        region.legacy = row
        region.flags["artifact_shape"] = row["candidate_class"] == "artifact"
        occupancy.add(region)
    pixel_area = field.pixel_size_um ** 2
    allowed = field.tissue & ~field.exclusion

    def keep(region: Region) -> bool:
        how, index = same_object(region, occupancy, shared["same_iou"], shared["same_inside"], shared["covered"],
                                 round(settings["significant_um2"] / pixel_area))
        if how == "same":
            occupancy.regions[index].add_source(region.source)
        elif how == "new":
            region.style = "amyloid"
            region.legacy = deposit_row(field, region)
            region.flags["artifact_shape"] = False
            occupancy.add(region)
        # a faint node joining kept deposits (a merge) or mostly covered by them: its children may be deposits of their
        # own (on a dark-background core the legacy seed level is far above the slide threshold, and the faint field at
        # 0.5 × joins its plaques to the seeded ones)
        return how in ("same", "new") or not settings.get("descend", True)

    coarse = settings.get("coarse")
    if coarse:  # pale diffuse plaques as one region each (the legacy seed misses them, the tree sees specks), before the tree
        for region in coarse_regions(field, coarse["smooth_um"], coarse["level"], coarse["contrast"], allowed, coarse["min_um2"],
                                     settings["max_um2"]):
            keep(region)
    if settings["mode"] == "roots":
        walk_roots(field, settings["levels"], allowed, settings["min_um2"], settings["max_um2"], settings["significant_um2"], keep)
    else:
        for region in tree_regions(field, settings["levels"], allowed, settings["min_um2"], settings["max_um2"], settings["significant_um2"],
                                   settings["split_max_um2"], settings["mode"]):
            keep(region)
    return occupancy.regions


def extras(field, regions: list[Region], ring_um: float = 15.0) -> list[dict]:
    """Per deposit: lumen / tissue-edge contact, smoothed-DAB quantiles (dense core at any threshold later), and the nuclei
    inside it and in rings of 15 and 45 µm around it (the plaque-niche measures)."""
    non_tissue = ~field.tissue
    valid = field.tissue & ~field.exclusion
    points = nucleus_centroids(field.nuclei, valid, field.pixel_size_um)
    point_map = np.zeros(field.shape, np.int32)
    if len(points):
        np.add.at(point_map, (points[:, 1], points[:, 0]), 1)
    smooth = cv2.GaussianBlur(field.dab, (0, 0), 1.0)
    ps = field.pixel_size_um
    out = []
    for region in regions:
        box12, local12 = region.padded(12, field.shape)
        ring_px, far_px = max(1, round(ring_um / ps)), max(1, round(3 * ring_um / ps))
        box, local = region.padded(far_px + 1, field.shape)
        distance = cv2.distanceTransform((~local).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        ring = (distance <= ring_px) & ~local & valid[box]
        far = (distance <= far_px) & ~local & valid[box]
        quantiles = np.quantile(smooth[region.box][region.mask], CORE_QUANTILES) / field.threshold
        out.append({
            "lumen_contact_fraction": outer_boundary_contact(local12, non_tissue[box12]),
            "core_dab_q": " ".join(f"{v:.4f}" for v in quantiles),
            "normalized_inner_dab": float(region.legacy["inner_mean_dab_od"]) / field.threshold,
            "nuclei_inside": int(point_map[region.box][region.mask].sum()),
            "nuclei_ring15": int(point_map[box][ring].sum()), "ring15_area_um2": float(ring.sum() * ps * ps),
            "nuclei_ring45": int(point_map[box][far].sum()), "ring45_area_um2": float(far.sum() * ps * ps),
        })
    return out


# Slide-scaled inputs (`s_<name>`, for models trained with `slide_scaled`): the DAB intensities in slide thresholds, and the
# dense core and seed measured at levels relative to the slide threshold. The legacy inputs use absolute OD (a core above
# max(0.10 OD, …), seeds above max(0.035 OD, …)), so on a pale slide such as LIP-6 (threshold 0.026 OD against 0.07–0.11
# elsewhere) plaques have no dense core and low intensities, and the current model rejects them.
SCALED_INTENSITY = scaled.INTENSITY["6E10"]
SCALED_CORE = scaled.RECOMPUTED["6E10"]
CORE_FLOOR = 1.25  # × the slide threshold: the legacy 0.10 OD at the cohort's typical threshold (0.08 OD)
SEED_FLOOR = 1.0  # × the slide threshold (the legacy floor 0.035 OD binds only below a 0.035 OD threshold)


def scaled_names(names: list[str]) -> list[str]:
    """A model's input names with each one that has a slide-scaled version replaced by it (`s_<name>`)."""
    return scaled.names("6E10", names)


def scaled_core(field, region: Region) -> dict:
    """Seed and dense-core measures of a deposit at levels relative to the slide threshold: seed above max(SEED_FLOOR ×
    threshold, background median + 6 σ), core above max(CORE_FLOOR × threshold, 1.2 × that seed level, the deposit's 85th
    percentile), as `deposit_row` measures them at the absolute levels."""
    levels = _levels(field)
    if "scaled_seed" not in field.memo:
        seed_level = max(SEED_FLOOR * field.threshold, levels["background_median"] + 6.0 * levels["background_sigma"])
        pixel_area = field.pixel_width_um * field.pixel_height_um
        field.memo["scaled_seed"] = (seed_level, _remove_small_components(levels["valid"] & (levels["local_dab"] >= seed_level),
                                                                          max(1, round(3.0 / pixel_area))))
    seed_level, seed = field.memo["scaled_seed"]
    pixel_area = field.pixel_width_um * field.pixel_height_um
    box, mask = region.box, region.mask
    values = field.dab[box][mask]
    area = int(mask.sum()) * pixel_area
    core_level = max(CORE_FLOOR * field.threshold, 1.2 * seed_level, float(np.quantile(values, 0.85)))
    core = _remove_small_components(mask & (levels["local_dab"][box] >= core_level), max(1, round(4.0 / pixel_area)))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(core.astype(np.uint8), 8)
    largest = labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) if count > 1 else np.zeros(core.shape, bool)
    radius = np.sqrt(area / np.pi)
    offset = float("nan")
    if largest.any():
        ys, xs = np.nonzero(mask)
        cy, cx = np.nonzero(largest)
        offset = float(np.hypot(cx.mean() - xs.mean(), cy.mean() - ys.mean()) * np.sqrt(pixel_area) / radius) if radius else float("nan")
    core_area = int(largest.sum()) * pixel_area
    return {"s_seed_area_um2": int((mask & seed[box]).sum()) * pixel_area, "s_core_area_um2": core_area,
            "s_core_fraction": core_area / area if area else float("nan"), "s_normalized_core_offset": offset}


def model_inputs(field, regions: list[Region]) -> list[dict]:
    """The 6E10 identity model's inputs (morphology and the 140 µm review-patch context), as m_ columns, and their
    slide-scaled versions (`s_` columns, `scaled_names`)."""
    if not regions:
        return []
    rows = [r.legacy for r in regions]
    matrix, names = identity_features(field.rgb, rows, field.pixel_size_um)
    out = []
    for region, values in zip(regions, matrix):
        row = {f"m_{n}": float(v) for n, v in zip(names, values)}
        row |= scaled.intensities("6E10", row, field.threshold)
        out.append(row | scaled_core(field, region))
    return out


__all__ = ["MORPHOLOGY_FEATURES", "SCALED_CORE", "SCALED_INTENSITY", "deposit_row", "extras", "model_inputs", "propose", "scaled_core",
           "scaled_names"]
