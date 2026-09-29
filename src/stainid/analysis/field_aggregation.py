"""Summarise 6E10/AT8 field-pipeline tiles into core- or donor-region features."""
from __future__ import annotations

import numpy as np


def total(rows, key):
    return sum(float(row[key]) for row in rows)


def ratio(numerator, denominator):
    return numerator / denominator if denominator else float("nan")


def summarize(stain: str, tiles: list[dict[str, str]], objects: list[dict[str, str]]) -> dict[str, float]:
    area = total(tiles, "v2_tissue_area_mm2")
    with_nuclei = [row for row in tiles if row.get("nuclei_available", "true") == "true"]
    out = {"v2_nucleus_density_mm2": ratio(total(with_nuclei, "nucleus_count"), total(with_nuclei, "v2_tissue_area_mm2"))}
    if stain == "AT8":
        count = total(tiles, "tau_neuron_count")
        length = total(tiles, "thread_ridge_length_mm")
        areas = [float(row["soma_area_um2"]) for row in objects]
        out |= {
            "at8_nucleus_density_mm2": out.pop("v2_nucleus_density_mm2"),
            "tau_neuron_count": count,
            "tau_neuron_density_mm2": ratio(count, area),
            "tau_neuron_mature_fraction": ratio(total(tiles, "tau_neuron_mature_count"), count) if count >= 5 else float("nan"),
            "tau_neuron_median_soma_um2": float(np.median(areas)) if len(areas) >= 5 else float("nan"),
            "thread_length_density_mm_per_mm2": ratio(length, area),
            "thread_mean_width_um": ratio(total(tiles, "thread_ridge_width_sum_um"), total(tiles, "thread_ridge_skeleton_px")) if length >= 0.5 else float("nan"),
            "thread_branchpoints_per_mm": ratio(total(tiles, "thread_branchpoint_count"), length) if length >= 0.5 else float("nan"),
            "thread_to_neuron_ratio_mm": ratio(length, count) if count >= 5 else float("nan"),
        }
    else:
        plaques = total(tiles, "v2_plaque_count")
        plaque_area = total(tiles, "v2_plaque_area_sum_um2")
        niche_plaques = total(with_nuclei, "v2_plaque_count")
        background = ratio(total(with_nuclei, "v2_far_nuclei"), total(with_nuclei, "v2_far_area_mm2"))
        ring = ratio(total(with_nuclei, "v2_ring_nuclei"), total(with_nuclei, "v2_ring_area_um2") / 1e6)
        eligible = [row for row in objects if float(row["equivalent_diameter_um"]) >= 15.0]
        out |= {
            "amyloid_nucleus_density_mm2": out.pop("v2_nucleus_density_mm2"),
            "v2_plaque_density_mm2": ratio(plaques, area),
            "plaque_density_ge10um_mm2": ratio(sum(float(r["equivalent_diameter_um"]) >= 10.0 for r in objects), area),
            "vascular_or_edge_amyloid_density_mm2": ratio(total(tiles, "v2_vascular_or_edge_count"), area),
            "compact_fraction_v2": ratio(sum(r["plaque_class"] == "compact" for r in eligible), sum(r["plaque_class"] in {"compact", "diffuse"} for r in eligible)) if len(eligible) >= 10 else float("nan"),
            "v2_plaque_mean_area_um2": ratio(plaque_area, plaques),
            "plaque_dense_core_area_fraction": ratio(total(tiles, "v2_dense_core_area_sum_um2"), plaque_area) if plaques >= 10 else float("nan"),
            "plaque_median_dense_core_fraction": float(np.median([float(r["dense_core_fraction"]) for r in eligible])) if len(eligible) >= 10 else float("nan"),
            "periplaque_nucleus_enrichment": ratio(ring, background) if niche_plaques >= 10 and total(with_nuclei, "v2_far_area_mm2") >= 0.05 else float("nan"),
            "nuclei_per_plaque": ratio(total(with_nuclei, "v2_plaque_nuclei") + total(with_nuclei, "v2_ring_nuclei"), niche_plaques) if niche_plaques >= 10 else float("nan"),
            "plaque_circularity_median": float(np.median([float(r["circularity"]) for r in eligible])) if len(eligible) >= 10 else float("nan"),
        }
    return out
