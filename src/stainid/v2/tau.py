"""AT8 proposals: the legacy tau+ neuron candidates (stained rings around Cellpose nuclei, dense bodies above the 60 µm
background), fainter rings (down to `ring_contrast`, flagged), and DAB bodies of the component tree and the local
background pass (pretangle-like bodies without a ring and small bodies the legacy thickness / size filters missed).
Also the AT8 maps the measures need: burden, the legacy network, and the neuropil thread skeleton."""
from __future__ import annotations

import cv2
import numpy as np

from stainid.stains.tau.network import segment_at8_network
from stainid.stains.tau.neurons import TAU_FEATURES, _local_features, nucleus_rings, tau_neuron_candidates
from stainid.stains.tau.threads import thread_network
from stainid.v2 import scaled
from stainid.v2.geometry import Occupancy, Region, label_regions, same_object
from stainid.v2.tree import local_background_regions, tree_regions

LEGACY_CONTRAST = 0.3


TOO_LITTLE_TISSUE = "AT8 network analysis requires at least 100"  # segment_at8_network refuses tiles with < 100 (valid) tissue pixels
NETWORK_MAPS = ("valid", "excluded", "manual_excluded", "edge_excluded", "edge_excluded_display", "positive", "hole_rim_removed", "thread",
                "thread_skeleton", "branchpoints", "endpoints")


def _no_network(field) -> dict:
    """The maps of a tile with (almost) no valid tissue, e.g. the corner of an image that is mostly glass: all empty."""
    empty = np.zeros(field.shape, bool)
    network_maps = {name: empty for name in NETWORK_MAPS} | {"compact_labels": np.zeros(field.shape, np.int32),
                                                           "cluster_labels": np.zeros(field.shape, np.int32), "local_density": np.zeros(field.shape, np.float32)}
    threads = {"line": empty, "skeleton": empty, "branchpoints": empty, "endpoints": empty, "width_um": np.zeros(field.shape, np.float32)}
    return {"summary": {"edge_guard_triggered": False}, "objects": [], "maps": network_maps, "threads": threads}


def maps(field) -> dict:
    """segment_at8_network's maps (valid tissue, burden, legacy compact profiles and threads) and the thread network;
    all empty for a tile with fewer than 100 valid tissue pixels (the network analysis refuses those)."""
    if "at8" not in field.memo:
        try:
            summary, objects, network_maps = segment_at8_network(field.rgb, field.threshold, field.pixel_width_um, field.pixel_height_um, field.exclusion)
        except ValueError as error:
            if TOO_LITTLE_TISSUE not in str(error):
                raise
            field.memo["at8"] = _no_network(field)
            return field.memo["at8"]
        threads = thread_network(field.rgb, network_maps["valid"], field.threshold, field.pixel_size_um, nuclei=field.nuclei)
        field.memo["at8"] = {"summary": summary, "objects": objects, "maps": network_maps, "threads": threads}
    return field.memo["at8"]


def _soma_smooth(field) -> np.ndarray:
    if "tau_smooth" not in field.memo:
        field.memo["tau_smooth"] = cv2.GaussianBlur(field.dab, (0, 0), 1.0 / field.pixel_size_um)
    return field.memo["tau_smooth"]


def rings(field, minimum: float, ring_um: float = 5.0) -> list[dict]:
    """The nucleus rings of the field (`stainid.stains.tau.neurons.nucleus_rings`, somata down to contrast `minimum`),
    made once: the legacy candidates (contrast >= LEGACY_CONTRAST) and the fainter ones are the same loop."""
    key = ("tau_rings", minimum, ring_um)
    if key not in field.memo:
        field.memo[key] = nucleus_rings(field.nuclei, field.tissue & ~field.exclusion, _soma_smooth(field), field.threshold,
                                        field.pixel_size_um, ring_um, min(minimum, LEGACY_CONTRAST))
    return field.memo[key]


def faint_rings(field, minimum: float, below: float = LEGACY_CONTRAST, ring_um: float = 5.0) -> list[tuple[str, tuple, np.ndarray, tuple]]:
    """Nucleus-ring candidates whose ring contrast is in [minimum, below): the legacy loop with a lower bar."""
    return [("ring_faint", ring["center"], ring["soma"], ring["origin"]) for ring in rings(field, minimum, ring_um)
            if minimum <= ring["contrast"] < below]


def tau_row(field, kind: str, center: tuple[float, float], soma: np.ndarray, origin: tuple[int, int]) -> dict:
    """The legacy tau candidate measurements of a soma mask (local, at `origin`) with a margin around it."""
    tissue = field.tissue & ~field.exclusion
    features = _local_features(field.dab, field.hematoxylin, tissue, field.nuclei, center, soma, origin, field.threshold, field.pixel_size_um)
    x0, y0 = origin
    features["edge_fraction"] = float(field.edge_band[y0: y0 + soma.shape[0], x0: x0 + soma.shape[1]][soma].mean())
    return {"candidate_source": kind, "centroid_x_px": float(center[0]), "centroid_y_px": float(center[1]), **features}


def propose(field, settings: dict, shared: dict) -> list[Region]:
    ps = field.pixel_size_um
    objects, labels = tau_neuron_candidates(field.rgb, field.nuclei, field.threshold, ps, field.exclusion,
                                            rings=rings(field, settings["ring_contrast"]))
    regions = label_regions(labels, "legacy", style="tau")
    occupancy = Occupancy(field.shape)
    for row in objects:
        region = regions.get(int(row["label"]))
        if region is None:
            continue
        region.source = "ring" if row["candidate_source"] == "nucleus_ring" else "dense"
        region.sources = [region.source]
        region.legacy = row
        occupancy.add(region)
    same = dict(same_iou=shared["same_iou"], same_inside=shared["same_inside"], covered=shared["covered"],
                significant_px=round(settings["significant_um2"] / ps ** 2))
    for kind, center, soma, (x0, y0) in faint_rings(field, settings["ring_contrast"]):
        region = Region(y0, x0, soma, "ring_faint", style="tau")
        how, index = same_object(region, occupancy, **same)
        if how == "same":
            occupancy.regions[index].add_source("ring_faint")
        elif how == "new":
            region.legacy = tau_row(field, "nucleus_ring", center, soma, (x0, y0))
            occupancy.add(region)
    allowed = field.tissue & ~field.exclusion
    found = tree_regions(field, settings["levels"], allowed, settings["min_um2"], settings["max_um2"], settings["significant_um2"],
                         settings["split_max_um2"], settings["mode"], settings.get("split"))
    if settings.get("local_contrast"):
        found += local_background_regions(field, settings["local_contrast"], allowed, settings["min_um2"], settings["max_um2"])
    margin = max(1, round(14 / ps))
    for region in found:
        how, index = same_object(region, occupancy, **same)
        if how == "same":
            occupancy.regions[index].add_source(region.source)
        elif how == "new":
            region.style = "tau"
            box, local = region.padded(margin, field.shape)
            region.legacy = tau_row(field, "dab_body", region.centroid(), local, (box[1].start, box[0].start))
            occupancy.add(region)
    return occupancy.regions


def extras(field, regions: list[Region]) -> list[dict]:
    """Per candidate: whether its centre lies in valid tissue (the legacy filter) and its share on traced threads."""
    found = maps(field)
    valid, line = found["maps"]["valid"], found["threads"]["line"]
    out = []
    for region in regions:
        x, y = float(region.legacy["centroid_x_px"]), float(region.legacy["centroid_y_px"])
        out.append({"centroid_valid": bool(valid[min(int(y), field.shape[0] - 1), min(int(x), field.shape[1] - 1)]),
                    "thread_fraction": float(line[region.box][region.mask].mean())})
    return out


def model_inputs(regions: list[Region], contexts: list[dict], threshold: float | None = None) -> list[dict]:
    """The tau model's inputs (`stainid.stains.tau.neurons.tau_feature_matrix`), as m_ columns, and with the slide's
    `threshold` their slide-scaled versions (`s_`, `stainid.v2.scaled`)."""
    out = []
    for region, context in zip(regions, contexts):
        row = region.legacy
        values = {f"m_{name}": float(row.get(name, np.nan)) for name in TAU_FEATURES}
        values |= {f"m_ctx_{k}": v for k, v in context.items()}
        values["m_from_ring"] = float(row["candidate_source"] == "nucleus_ring")
        if threshold:
            values |= scaled.intensities("AT8", values, threshold)
        out.append(values)
    return out


def context_diameters(regions: list[Region]) -> list[float]:
    return [2 * np.sqrt(float(r.legacy["soma_area_um2"]) / np.pi) for r in regions]
