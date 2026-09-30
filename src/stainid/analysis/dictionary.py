"""Plain-language meaning of every column in the results tables."""
from __future__ import annotations

import re

COLUMNS = {
    "core_id": "Core: TMA and grid position (e.g. LIP-3_B-2).",
    "sample_region_id": "Donor and brain region (all replicate cores of that region pooled).",
    "donor_id": "Donor identifier from the TMA map.",
    "tma": "TMA number.",
    "region": "Brain region.",
    "disease_group": "Diagnostic group from the TMA map.",
    "neun_analyzed_tile_count": "NeuN fields measured.",
    "neun_analyzed_core_count": "NeuN cores measured.",
    "neun_tissue_area_mm2": "Usable tissue measured on NeuN slides (mm²), after removing folds, holes and artifacts.",
    "neun_mean_artifact_exclusion_area_fraction": "Average share of each NeuN field removed as artifact.",
    "neun_profile_density_mm2": "NeuN+ neurons per mm² of tissue (main neuron measure).",
    "neun_positive_profile_count": "NeuN+ neurons counted.",
    "neun_candidate_profile_count": "All candidate cell profiles considered.",
    "neun_candidate_profile_density_mm2": "Candidate cell profiles per mm² (neurons + other cells).",
    "neun_positive_fraction_of_candidates": "Share of candidate cells that are NeuN+ neurons (less sensitive to tissue density).",
    "neun_positive_profile_area_fraction": "Share of tissue area covered by NeuN+ neurons.",
    "neun_union_profile_density_mm2": "Sensitivity version of neuron density using a more inclusive rule.",
    "neun_review_profile_count": "Candidates the model was unsure about (probability 0.2–0.8).",
    "neun_local_density_cv_100um": "Patchiness of neurons: variation of density between 100 µm windows.",
    "neun_local_density_window_count": "Windows used for the patchiness measure.",
    "neun_median_profile_area_um2": "Median size of NeuN+ neuron profiles (µm²).",
    "neun_median_profile_dab_od": "Median NeuN stain intensity of neurons (optical density).",
    "amyloid_nucleus_density_mm2": "Cell nuclei per mm² on 6E10 slides (cellularity).",
    "v2_plaque_density_mm2": "Amyloid plaques per mm² of tissue.",
    "plaque_area_fraction": "Share of tissue area covered by plaques (amyloid burden).",
    "plaque_density_ge10um_mm2": "Plaques at least 10 µm across, per mm².",
    "vascular_or_edge_amyloid_density_mm2": "Amyloid deposits on vessels or tissue edges per mm² (kept out of plaque measures).",
    "amyloid_speck_density_mm2": "Accepted deposits of 10 µm or less per mm² (specks, not counted as plaques).",
    "compact_fraction_v2": "Share of plaques (≥ 15 µm) that are compact / cored rather than diffuse.",
    "v2_plaque_mean_area_um2": "Average plaque size (µm²).",
    "plaque_dense_core_area_fraction": "Share of total plaque area that is dense core.",
    "plaque_median_dense_core_fraction": "Median share of each plaque that is dense core.",
    "periplaque_nucleus_enrichment": "Nuclei around plaques relative to nuclei far from plaques (> 1 = cells gather at plaques).",
    "nuclei_per_plaque": "Nuclei inside or within 15 µm of each plaque.",
    "plaque_circularity_median": "Median plaque roundness (1 = circle).",
    "at8_nucleus_density_mm2": "Cell nuclei per mm² on AT8 slides.",
    "at8_positive_area_fraction": "Share of tissue stained for phospho-tau (tau burden).",
    "tau_neuron_count": "Tau+ neurons counted (tangles and pretangles).",
    "tau_neuron_density_mm2": "Tau+ neurons per mm².",
    "tau_neuron_mature_fraction": "Share of tau+ neurons that look like mature tangles.",
    "tau_neuron_median_soma_um2": "Median size of tau+ neuron cell bodies (µm²).",
    "thread_length_density_mm_per_mm2": "Length of tau neuropil threads per mm² of tissue.",
    "thread_mean_width_um": "Average thread width (µm).",
    "thread_branchpoints_per_mm": "Thread branch points per mm of thread (network complexity).",
    "thread_to_neuron_ratio_mm": "Thread length per tau+ neuron (mm).",
}
MASK_PREFIX = {"neun_soma": "NeuN+ neuron", "tau_neuron": "tau+ neuron", "plaque": "plaque"}
MASK_SUFFIX = {
    "mask_count": "{} outlines drawn by Segment Anything.",
    "mask_density_mm2": "{} outlines per mm².",
    "area_median_um2": "Median {} area from its outline (µm²).",
    "area_p90_um2": "Size of the largest 10% of {}s (90th percentile area, µm²).",
    "area_cv": "Size variability of {}s (coefficient of variation).",
    "solidity_median": "Median {} solidity (1 = convex, lower = irregular).",
    "circularity_median": "Median {} roundness from its outline (1 = circle).",
    "elongation_median": "Median {} elongation (long / short axis).",
    "normalized_dab_median": "Median {} stain intensity relative to the slide threshold.",
    "mask_area_fraction": "Share of tissue covered by {} outlines.",
    "dense_core_fraction_median": "Median share of each {} that is dense core (from outlines).",
    "cored_fraction": "Share of {}s with a dense core covering ≥ 25% of the outline.",
    "multicore_fraction": "Share of {}s with two or more dense cores.",
    "mask_mature_fraction": "Share of {}s that look like mature tangles (outline-based).",
}


def describe(column: str) -> str:
    if column in COLUMNS:
        return COLUMNS[column]
    for prefix, noun in MASK_PREFIX.items():
        match = re.fullmatch(rf"{prefix}_(.+)", column)
        if match and match.group(1) in MASK_SUFFIX:
            text = MASK_SUFFIX[match.group(1)].format(noun)
            return text[0].upper() + text[1:]
    return ""
