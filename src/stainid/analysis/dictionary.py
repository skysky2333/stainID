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
    "tau_neuron_nucleus_visible_fraction": "Share of tau+ neurons whose own nucleus shows through the stain (a blue nucleus inside the soma; pretangles more often than mature tangles).",
    "tau_neuron_nucleus_visible_fraction_mature": "Share of mature tangles whose nucleus shows through the stain.",
    "tau_neuron_nucleus_visible_fraction_pretangle": "Share of pretangles whose nucleus shows through the stain.",
    "tau_neuron_nucleus_offset_median": "How far the nucleus sits from the middle of the tau+ neuron's cell body, in cell-body radii (median; a tangle pushes it aside).",
    "plaque_profile_core_contrast_median": "How much darker a plaque's centre (first 4 µm) is than the 4-10 µm around it, in slide thresholds (median over plaques ≥ 15 µm).",
    "plaque_profile_core_to_halo_median": "Plaque centre stain over its halo's (4-20 µm), median: high for bare cores.",
    "plaque_profile_halo_width_um_median": "Width of the stained halo outside a plaque's dense core (µm, median).",
    "plaque_bare_core_fraction": "Share of plaques with a dense core but almost no halo (burnt-out cores).",
    "plaque_haloed_core_fraction": "Share of plaques with a dense core and a stained halo at least 4 µm wide.",
    "neun_vessel_density_mm2": "Vessel lumens per mm² on the NeuN slide.",
    "neun_vessel_area_fraction": "Share of the NeuN slide's tissue that is vessel lumen.",
    "amyloid_vessel_density_mm2": "Vessel lumens per mm² on the 6E10 slide.",
    "amyloid_vessel_area_fraction": "Share of the 6E10 slide's tissue that is vessel lumen.",
    "at8_vessel_density_mm2": "Vessel lumens per mm² on the AT8 slide.",
    "at8_vessel_area_fraction": "Share of the AT8 slide's tissue that is vessel lumen.",
    "caa_vessel_density_mm2": "Vessels with amyloid around most of their wall (CAA-like rings) per mm².",
    "caa_vessel_fraction": "Share of vessels with amyloid around most of their wall (CAA-like).",
    "plaque_vessel_touch_fraction": "Share of plaques whose edge touches a vessel lumen (within 2 µm).",
    "plaque_vessel_near_fraction": "Share of plaques whose edge lies within 10 µm of a vessel lumen.",
    "tau_neuron_vessel_near_density_mm2": "Tau+ neurons per mm² within 20 µm of a vessel.",
    "tau_neuron_vessel_far_density_mm2": "Tau+ neurons per mm² further than 20 µm from any vessel.",
    "tau_neuron_vessel_enrichment": "Tau+ neuron density near vessels over away from them (1 = no difference).",
    "thread_vessel_near_density_mm_per_mm2": "Thread length per mm² within 20 µm of a vessel.",
    "thread_vessel_far_density_mm_per_mm2": "Thread length per mm² further than 20 µm from any vessel.",
    "thread_vessel_enrichment": "Thread density near vessels over away from them (1 = no difference).",
    "neun_neuron_nucleus_area_median_um2": "Median size of the nucleus inside NeuN+ neurons (µm²).",
    "neun_large_nucleus_density_mm2": "Neuron-sized nuclei (≥ 50 µm²) per mm² of the NeuN slide, stained or not: fewer when neurons are lost.",
    "neun_large_nucleus_in_neuron_fraction": "Share of neuron-sized nuclei inside a counted NeuN+ neuron.",
    "neun_large_nucleus_weak_fraction": "Share of neuron-sized nuclei in no counted neuron and only weakly NeuN-stained (half to one slide threshold): possible antigen loss.",
    "neun_large_nucleus_unstained_fraction": "Share of neuron-sized nuclei in no counted neuron and unstained for NeuN (under half a slide threshold): antigen loss, or large other cells.",
    "tangle_nn_median_um": "Typical distance from a mature tangle to the nearest other mature tangle in the same core (median, µm; at least 10 tangles).",
    "tangle_l200_excess_um": "How much mature tangles gather within 200 µm compared with random placement in the same tissue (µm; above 0 they cluster, below 0 they keep apart; at least 10 tangles in a core).",
    "tau_processes_per_neuron": "Thread stretches reaching each tau+ neuron (within 2 µm of its outline), on average.",
    "tau_processes_per_neuron_mature": "Thread stretches reaching each mature tangle, on average.",
    "tau_processes_per_neuron_pretangle": "Thread stretches reaching each pretangle, on average.",
    "tau_neurons_with_process_fraction": "Share of tau+ neurons with at least one thread stretch reaching them.",
    "tau_attached_thread_um_per_neuron": "Thread length attached to each tau+ neuron, on average (µm; a stretch reaching several neurons is split equally).",
    "tau_attached_thread_um_per_neuron_mature": "Thread length attached to each mature tangle, on average (µm).",
    "tau_attached_thread_um_per_neuron_pretangle": "Thread length attached to each pretangle, on average (µm).",
    "thread_attached_density_mm_per_mm2": "Length of thread attached to a tau+ neuron, per mm² of tissue.",
    "thread_unattached_density_mm_per_mm2": "Length of thread attached to no tau+ neuron, per mm² of tissue.",
    "thread_attached_share": "Share of thread length that is attached to a tau+ neuron.",
    "plaque_white_area_mm2": "White matter measured on the 6E10 slide (mm²).",
    "plaque_density_white_mm2": "Plaques per mm² inside the white matter (the plaque density column is the grey matter's).",
    "plaque_area_fraction_white": "Share of the white matter covered by plaques.",
    "amyloid_positive_area_fraction_white": "Share of the white matter stained for amyloid.",
    "tau_white_area_mm2": "White matter measured on the AT8 slide (mm²).",
    "tau_neuron_density_white_mm2": "Tau+ neurons per mm² inside the white matter (the tau+ neuron density column is the grey matter's).",
    "thread_density_white_mm_per_mm2": "Thread length per mm² inside the white matter.",
    "at8_positive_area_fraction_white": "Share of the white matter stained for phospho-tau.",
    "neun_edge_touch_fraction": "Quality check: share of counted neurons touching a tissue edge, a hole or an excluded area.",
    "neun_edge_mostly_fraction": "Quality check: share of counted neurons lying mostly within 5 µm of a tissue edge or hole.",
    "plaque_edge_touch_fraction": "Quality check: share of counted plaques touching a tissue edge, a hole or an excluded area.",
    "plaque_edge_mostly_fraction": "Quality check: share of counted plaques lying mostly within 5 µm of a tissue edge or hole.",
    "tau_neuron_edge_touch_fraction": "Quality check: share of counted tau+ neurons touching a tissue edge, a hole or an excluded area (mostly small holes next to real neurons; differs between slides).",
    "tau_neuron_edge_mostly_fraction": "Quality check: share of counted tau+ neurons lying mostly within 5 µm of a tissue edge or hole.",
    "tau_neuron_boundary_fraction": "Quality check: share of counted tau+ neurons at a tissue boundary (a large area without tissue, such as the core's edge or a tear, within 15 µm). Most objects that are not real cell bodies sit there.",
    "tau_neuron_density_no_boundary_mm2": "Tau+ neurons per mm², leaving out those at a tissue boundary. The main density still counts them.",
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


def _sentence_noun(noun: str) -> str:
    return noun[:1].upper() + noun[1:]


def describe(column: str) -> str:
    if column in COLUMNS:
        return COLUMNS[column]
    for prefix, noun in MASK_PREFIX.items():
        match = re.fullmatch(rf"{prefix}_(.+)", column)
        if match and match.group(1) in MASK_SUFFIX:
            text = MASK_SUFFIX[match.group(1)].format(noun)
            return text[0].upper() + text[1:]
    return ""


# Readable names, units and topics for showing measures to people (the web app's Results page). The tables keep their
# column names.
TOPICS = {
    "neurons": "Neurons (NeuN)",
    "plaques": "Amyloid plaques (6E10)",
    "tau": "Tau (AT8)",
    "tissue": "Tissue measured and quality",
    "spatial": "Spatial interactions (what lies next to what)",
    "other": "Other measures",
}
KEY_MEASURES = (
    "neun_profile_density_mm2", "v2_plaque_density_mm2", "plaque_area_fraction", "compact_fraction_v2",
    "tau_neuron_density_mm2", "at8_positive_area_fraction", "thread_length_density_mm_per_mm2",
)
FRACTION = "fraction, 0–1"
LABELS = {  # column: (label, unit, topic)
    "neun_analyzed_tile_count": ("NeuN fields measured", "fields", "tissue"),
    "neun_analyzed_core_count": ("NeuN cores measured", "cores", "tissue"),
    "neun_tissue_area_mm2": ("NeuN tissue area measured", "mm²", "tissue"),
    "neun_mean_artifact_exclusion_area_fraction": ("NeuN area removed as artifact", FRACTION, "tissue"),
    "neun_candidate_profile_count": ("Candidate cells considered", "cells", "tissue"),
    "neun_review_profile_count": ("Candidates the model was unsure about", "cells", "tissue"),
    "neun_local_density_window_count": ("Windows used for patchiness", "windows", "tissue"),
    "neun_profile_density_mm2": ("Neuron density", "neurons per mm²", "neurons"),
    "neun_positive_profile_count": ("Neurons counted", "neurons", "neurons"),
    "neun_candidate_profile_density_mm2": ("Candidate cell density", "cells per mm²", "neurons"),
    "neun_positive_fraction_of_candidates": ("Neurons among candidate cells", FRACTION, "neurons"),
    "neun_positive_profile_area_fraction": ("Tissue area covered by neurons", FRACTION, "neurons"),
    "neun_union_profile_density_mm2": ("Neuron density, inclusive rule", "neurons per mm²", "neurons"),
    "neun_local_density_cv_100um": ("Neuron patchiness", "CV across 100 µm windows", "neurons"),
    "neun_median_profile_area_um2": ("Neuron size, median", "µm²", "neurons"),
    "neun_median_profile_dab_od": ("Neuron stain intensity, median", "optical density", "neurons"),
    "amyloid_nucleus_density_mm2": ("Cell nuclei density on 6E10 slides", "nuclei per mm²", "plaques"),
    "v2_plaque_density_mm2": ("Plaque density", "plaques per mm²", "plaques"),
    "plaque_area_fraction": ("Amyloid burden (plaque area)", FRACTION, "plaques"),
    "plaque_density_ge10um_mm2": ("Density of plaques ≥ 10 µm", "plaques per mm²", "plaques"),
    "vascular_or_edge_amyloid_density_mm2": ("Amyloid on vessels or tissue edges", "deposits per mm²", "plaques"),
    "amyloid_speck_density_mm2": ("Amyloid specks (≤ 10 µm)", "specks per mm²", "plaques"),
    "compact_fraction_v2": ("Compact (cored) plaques", "fraction of plaques, 0–1", "plaques"),
    "v2_plaque_mean_area_um2": ("Plaque size, mean", "µm²", "plaques"),
    "plaque_dense_core_area_fraction": ("Dense-core share of plaque area", FRACTION, "plaques"),
    "plaque_median_dense_core_fraction": ("Dense-core share per plaque, median", FRACTION, "plaques"),
    "periplaque_nucleus_enrichment": ("Cells gathered around plaques", "ratio, 1 = none", "plaques"),
    "nuclei_per_plaque": ("Nuclei per plaque", "nuclei", "plaques"),
    "plaque_circularity_median": ("Plaque roundness, median", "1 = circle", "plaques"),
    "at8_nucleus_density_mm2": ("Cell nuclei density on AT8 slides", "nuclei per mm²", "tau"),
    "at8_positive_area_fraction": ("Tau burden (stained area)", FRACTION, "tau"),
    "tau_neuron_count": ("Tau+ neurons counted", "neurons", "tau"),
    "tau_neuron_density_mm2": ("Tau+ neuron density", "neurons per mm²", "tau"),
    "tau_neuron_mature_fraction": ("Mature tangles among tau+ neurons", FRACTION, "tau"),
    "tau_neuron_median_soma_um2": ("Tau+ neuron size, median", "µm²", "tau"),
    "thread_length_density_mm_per_mm2": ("Neuropil thread length", "mm per mm²", "tau"),
    "thread_mean_width_um": ("Thread width, mean", "µm", "tau"),
    "thread_branchpoints_per_mm": ("Thread branching", "branch points per mm", "tau"),
    "thread_to_neuron_ratio_mm": ("Thread length per tau+ neuron", "mm per neuron", "tau"),
    "tau_neuron_nucleus_visible_fraction": ("Tau+ neurons with a visible nucleus", FRACTION, "tau"),
    "tau_neuron_nucleus_visible_fraction_mature": ("Mature tangles with a visible nucleus", FRACTION, "tau"),
    "tau_neuron_nucleus_visible_fraction_pretangle": ("Pretangles with a visible nucleus", FRACTION, "tau"),
    "tau_neuron_nucleus_offset_median": ("Nucleus pushed aside in tau+ neurons", "soma radii", "tau"),
    "plaque_profile_core_contrast_median": ("Plaque core contrast, median", "× slide threshold", "plaques"),
    "plaque_profile_core_to_halo_median": ("Plaque core-to-halo ratio, median", "ratio", "plaques"),
    "plaque_profile_halo_width_um_median": ("Plaque halo width, median", "µm", "plaques"),
    "plaque_bare_core_fraction": ("Bare (burnt-out) cores", FRACTION, "plaques"),
    "plaque_haloed_core_fraction": ("Cored plaques with a halo", FRACTION, "plaques"),
    "neun_vessel_density_mm2": ("Vessel density, NeuN slide", "vessels per mm²", "tissue"),
    "neun_vessel_area_fraction": ("Vessel lumen area, NeuN slide", FRACTION, "tissue"),
    "amyloid_vessel_density_mm2": ("Vessel density, 6E10 slide", "vessels per mm²", "tissue"),
    "amyloid_vessel_area_fraction": ("Vessel lumen area, 6E10 slide", FRACTION, "tissue"),
    "at8_vessel_density_mm2": ("Vessel density, AT8 slide", "vessels per mm²", "tissue"),
    "at8_vessel_area_fraction": ("Vessel lumen area, AT8 slide", FRACTION, "tissue"),
    "caa_vessel_density_mm2": ("Amyloid-ringed vessels (CAA-like)", "vessels per mm²", "plaques"),
    "caa_vessel_fraction": ("Vessels with amyloid in the wall", FRACTION, "plaques"),
    "plaque_vessel_touch_fraction": ("Plaques touching a vessel", FRACTION, "plaques"),
    "plaque_vessel_near_fraction": ("Plaques near a vessel", FRACTION, "plaques"),
    "tau_neuron_vessel_near_density_mm2": ("Tau+ neurons near vessels", "neurons per mm²", "tau"),
    "tau_neuron_vessel_far_density_mm2": ("Tau+ neurons away from vessels", "neurons per mm²", "tau"),
    "tau_neuron_vessel_enrichment": ("Tau+ neurons gathered at vessels", "ratio, 1 = none", "tau"),
    "thread_vessel_near_density_mm_per_mm2": ("Threads near vessels", "mm per mm²", "tau"),
    "thread_vessel_far_density_mm_per_mm2": ("Threads away from vessels", "mm per mm²", "tau"),
    "thread_vessel_enrichment": ("Threads gathered at vessels", "ratio, 1 = none", "tau"),
    "neun_neuron_nucleus_area_median_um2": ("Neuron nucleus size, median", "µm²", "neurons"),
    "neun_large_nucleus_density_mm2": ("Neuron-sized nuclei", "nuclei per mm²", "neurons"),
    "neun_large_nucleus_in_neuron_fraction": ("Neuron-sized nuclei in counted neurons", FRACTION, "neurons"),
    "neun_large_nucleus_weak_fraction": ("Neuron-sized nuclei, weak NeuN", FRACTION, "neurons"),
    "neun_large_nucleus_unstained_fraction": ("Neuron-sized nuclei, no NeuN", FRACTION, "neurons"),
    "tangle_nn_median_um": ("Distance between neighbouring mature tangles", "µm", "tau"),
    "tangle_l200_excess_um": ("Tangle clustering within 200 µm", "µm above random", "tau"),
    "tau_processes_per_neuron": ("Thread stretches per tau+ neuron", "per neuron", "tau"),
    "tau_processes_per_neuron_mature": ("Thread stretches per mature tangle", "per neuron", "tau"),
    "tau_processes_per_neuron_pretangle": ("Thread stretches per pretangle", "per neuron", "tau"),
    "tau_neurons_with_process_fraction": ("Tau+ neurons with a thread attached", FRACTION, "tau"),
    "tau_attached_thread_um_per_neuron": ("Attached thread per tau+ neuron", "µm per neuron", "tau"),
    "tau_attached_thread_um_per_neuron_mature": ("Attached thread per mature tangle", "µm per neuron", "tau"),
    "tau_attached_thread_um_per_neuron_pretangle": ("Attached thread per pretangle", "µm per neuron", "tau"),
    "thread_attached_density_mm_per_mm2": ("Thread attached to neurons", "mm per mm²", "tau"),
    "thread_unattached_density_mm_per_mm2": ("Thread not attached to neurons", "mm per mm²", "tau"),
    "thread_attached_share": ("Share of thread attached to neurons", FRACTION, "tau"),
    "plaque_white_area_mm2": ("White matter measured, 6E10", "mm²", "tissue"),
    "plaque_density_white_mm2": ("Plaque density in white matter", "plaques per mm²", "plaques"),
    "plaque_area_fraction_white": ("Amyloid burden (plaque area) in white matter", FRACTION, "plaques"),
    "amyloid_positive_area_fraction_white": ("Amyloid-stained area in white matter", FRACTION, "plaques"),
    "tau_white_area_mm2": ("White matter measured, AT8", "mm²", "tissue"),
    "tau_neuron_density_white_mm2": ("Tau+ neuron density in white matter", "neurons per mm²", "tau"),
    "thread_density_white_mm_per_mm2": ("Thread length in white matter", "mm per mm²", "tau"),
    "at8_positive_area_fraction_white": ("Tau burden (stained area) in white matter", FRACTION, "tau"),
    "neun_edge_touch_fraction": ("Neurons touching an edge or excluded area", FRACTION, "tissue"),
    "neun_edge_mostly_fraction": ("Neurons mostly at an edge or hole", FRACTION, "tissue"),
    "plaque_edge_touch_fraction": ("Plaques touching an edge or excluded area", FRACTION, "tissue"),
    "plaque_edge_mostly_fraction": ("Plaques mostly at an edge or hole", FRACTION, "tissue"),
    "tau_neuron_edge_touch_fraction": ("Tau+ neurons touching an edge or excluded area", FRACTION, "tissue"),
    "tau_neuron_edge_mostly_fraction": ("Tau+ neurons mostly at an edge or hole", FRACTION, "tissue"),
    "tau_neuron_boundary_fraction": ("Tau+ neurons at a tissue boundary", FRACTION, "tissue"),
    "tau_neuron_density_no_boundary_mm2": ("Tau+ neuron density, without those at a tissue boundary", "tau+ neurons per mm²", "tau"),
}
# the continuous type scores of stage 5 (`stainid.v2.scores`): per score its object, what 0 and 1 mean, its topic, bins
TYPE_SCORES = {
    "tau_maturity": ("tau+ neuron", "maturity", "pretangle (0) to mature tangle (1)", "tau", ("pretangle", "intermediate", "mature")),
    "plaque_compactness": ("plaque", "compactness", "diffuse (0) to compact, dense-cored (1); plaques of at least 15 µm", "plaques",
                           ("diffuse", "intermediate", "compact")),
}
for _prefix, (_noun, _name, _span, _topic, _bins) in TYPE_SCORES.items():
    COLUMNS[f"{_prefix}_scored_count"] = f"{_sentence_noun(_noun)}s with a {_name} score and bin."
    COLUMNS[f"{_prefix}_score_median"] = f"Median {_noun} {_name} score, {_span}."
    COLUMNS[f"{_prefix}_score_iqr"] = f"Spread of the {_noun} {_name} scores (interquartile range)."
    COLUMNS[f"{_prefix}_score_mean"] = f"Mean {_noun} {_name} score, {_span}."
    LABELS[f"{_prefix}_scored_count"] = (f"{_sentence_noun(_noun)}s with a {_name} score", f"{_noun}s", _topic)
    LABELS[f"{_prefix}_score_median"] = (f"{_sentence_noun(_noun)} {_name}, median", "score, 0–1", _topic)
    LABELS[f"{_prefix}_score_iqr"] = (f"{_sentence_noun(_noun)} {_name}, spread", "interquartile range", _topic)
    LABELS[f"{_prefix}_score_mean"] = (f"{_sentence_noun(_noun)} {_name}, mean", "score, 0–1", _topic)
    for _bin in _bins:
        COLUMNS[f"{_prefix}_{_bin}_fraction"] = f"Share of {_noun}s in the {_bin} bin of the {_name} score."
        COLUMNS[f"{_prefix}_{_bin}_density_mm2"] = f"{_sentence_noun(_noun)}s in the {_bin} bin of the {_name} score per mm²."
        LABELS[f"{_prefix}_{_bin}_fraction"] = (f"{_sentence_noun(_noun)}s, {_bin} ({_name} score)", FRACTION, _topic)
        LABELS[f"{_prefix}_{_bin}_density_mm2"] = (f"{_sentence_noun(_noun)}s, {_bin} ({_name} score), density", f"{_noun}s per mm²", _topic)
# the spatial interactions of stage 5 (`stainid.v2.interactions`): column -> (label, unit, what it means)
RATIO_1 = "ratio, 1 = as random"
SPATIAL = {
    "plaque_contact_neuron_share_ratio": ("Neuron share touching plaques", RATIO_1,
        "Share of neuron-like nuclei within 10 µm of a plaque (≥ 20 µm) relative to far grey matter, against plaques placed at random. Descriptive only: nuclei under dense stain are missed."),
    "plaque_ring10_30_neuron_share_ratio": ("Neuron share 10–30 µm from plaques", RATIO_1,
        "Share of nuclei that look neuronal 10–30 µm from a plaque edge, relative to grey matter at least 100 µm from any plaque, against plaques placed at random (below 1 = neurons missing next to plaques)."),
    "plaque_ring30_60_neuron_share_ratio": ("Neuron share 30–60 µm from plaques", RATIO_1, "The same 30–60 µm from a plaque edge."),
    "plaque_ring_neuron_share_slope": ("Recovery of neurons away from plaques", "difference of ratios",
        "Neuron share 30–60 µm minus 10–30 µm from plaques (above 0 = neurons recover with distance)."),
    "plaque_ring10_30_neuronlike_density_ratio": ("Neuron-like nuclei 10–30 µm from plaques", RATIO_1,
        "Density of neuron-like nuclei 10–30 µm from plaques relative to far grey matter, against random placement."),
    "plaque_ring10_30_other_density_ratio": ("Other nuclei 10–30 µm from plaques", RATIO_1,
        "Density of the other (non-neuronal-looking) nuclei 10–30 µm from plaques relative to far grey matter, against random placement (above 1 = cells gather)."),
    "plaque_compact_ring10_30_neuron_share_ratio": ("Neuron share 10–30 µm from compact plaques", RATIO_1,
        "As the 10–30 µm neuron share, around compact (cored) plaques only."),
    "plaque_diffuse_ring10_30_neuron_share_ratio": ("Neuron share 10–30 µm from diffuse plaques", RATIO_1,
        "As the 10–30 µm neuron share, around diffuse plaques only."),
    "tangle_ring15_60_neuron_share_ratio": ("Neuron share around mature tangles", RATIO_1,
        "Share of neuron-like nuclei 15–60 µm from a mature tangle relative to grey matter at least 100 µm from tau+ neurons, against random placement (AT8 section)."),
    "tau_pretangle_ring15_60_neuron_share_ratio": ("Neuron share around pretangles", RATIO_1, "The same around pretangles."),
    "tangle_ring15_60_other_density_ratio": ("Other nuclei around mature tangles", RATIO_1,
        "Density of non-neuronal-looking nuclei 15–60 µm from mature tangles relative to far grey matter, against random placement."),
    "tau_mature_no_nucleus_fraction": ("Mature tangles without a nucleus", FRACTION,
        "Share of mature tangles whose outline holds no nucleus (a dead neuron's ghost tangle, or a nucleus outside the section)."),
    "tau_pretangle_no_nucleus_fraction": ("Pretangles without a nucleus", FRACTION, "Share of pretangles whose outline holds no nucleus."),
    "neun_no_nucleus_fraction": ("Neurons without a nucleus in the section", FRACTION,
        "Share of NeuN+ neurons whose outline holds no counterstained nucleus: how often the section misses an intact neuron's nucleus."),
    "tangle_ghost_excess": ("Ghost-tangle excess", "difference of fractions",
        "Mature tangles without a nucleus minus NeuN+ neurons without one: the share of tangle-bearing neurons that have died, beyond what the section plane explains."),
    "tau_satellites_per_neuron": ("Nuclei hugging each tau+ neuron", "per neuron", "Nuclei just outside a tau+ neuron's outline (within their radius + 2 µm), per neuron."),
    "tau_satellite_enrichment": ("Nuclei hugging tau+ neurons, enrichment", RATIO_1,
        "Those nuclei relative to what the section's nucleus density would put in the same band."),
    "neun_microcolumn_strength": ("Neuronal microcolumns", RATIO_1,
        "Neuron pairs lined up along the radial axis (within 10 µm of it, 30–150 µm apart) over the same across it, relative to random neurons in the same tissue (above 1 = neurons in columns; needs white matter for the axis)."),
    "amyloid_section_microcolumn_strength": ("Neuronal microcolumns, 6E10 section", RATIO_1,
        "The microcolumn measure on the 6E10 section's neuron-like nuclei: a replicate within the core."),
    "tau_section_microcolumn_strength": ("Neuronal microcolumns, AT8 section", RATIO_1,
        "The microcolumn measure on the AT8 section's neuron-like nuclei: a replicate within the core."),
    "amyloid_section_neuronlike_fraction": ("Neuron-like nuclei, 6E10 section", FRACTION,
        "Share of the 6E10 section's nuclei that look neuronal (a nuclear-shape model trained on the NeuN section)."),
    "tau_section_neuronlike_fraction": ("Neuron-like nuclei, AT8 section", FRACTION, "The same on the AT8 section."),
    "tangle_pretangle_mature_mix_50um": ("Pretangles next to mature tangles, 50 µm", RATIO_1,
        "Pretangle–mature tangle pairs within 50 µm relative to random labelling of maturity among the same tau+ neurons (above 1 = they sit together)."),
    "tangle_pretangle_mature_mix_100um": ("Pretangles next to mature tangles, 100 µm", RATIO_1, "The same within 100 µm."),
    "tau_neuron_column_alignment": ("Tau+ neurons lined up in columns", RATIO_1,
        "Pairs of tau+ neurons within 300 µm that lie within 30° of the radial axis, relative to random placement in the same tissue."),
    "tau_neuron_depth_excess_um": ("Depth of tau+ neurons", "µm above the tissue's mean",
        "Mean distance of tau+ neurons from the white matter minus that of the grey matter (within 2 mm); above 0 = nearer the surface."),
    "tangle_depth_excess_um": ("Depth of mature tangles", "µm above the tissue's mean", "The same for mature tangles."),
    "tau_pretangle_depth_excess_um": ("Depth of pretangles", "µm above the tissue's mean", "The same for pretangles."),
    "thread_depth_excess_um": ("Depth of neuropil threads", "µm above the tissue's mean", "The same for thread length."),
    "plaque_depth_excess_um": ("Depth of plaques", "µm above the tissue's mean", "The same for plaques."),
    "plaque_compact_depth_excess_um": ("Depth of compact plaques", "µm above the tissue's mean", "The same for compact plaques."),
    "plaque_diffuse_depth_excess_um": ("Depth of diffuse plaques", "µm above the tissue's mean", "The same for diffuse plaques."),
    "neun_depth_excess_um": ("Depth of neurons", "µm above the tissue's mean", "The same for NeuN+ neurons."),
    "neun_large_neuron_depth_excess_um": ("Depth of large neurons", "µm above the tissue's mean", "The same for NeuN+ neurons of at least 150 µm²."),
    "tau_neuron_deep_upper_ratio": ("Tau+ neurons deep vs upper cortex", "ratio of densities",
        "Tau+ neuron density within 1 mm of the white matter over that 1–2 mm from it."),
    "neun_deep_upper_ratio": ("Neurons deep vs upper cortex", "ratio of densities", "NeuN+ neuron density within 1 mm of the white matter over that 1–2 mm from it."),
    "neun_large_neuron_peak_depth_um": ("Depth of the large-neuron layer", "µm from white matter",
        "Distance from the white matter of the 250 µm band densest in large neurons (≥ 150 µm²): a layer landmark."),
    "tangle_plaque_depth_difference_um": ("Tangles vs compact plaques, depth", "µm",
        "Depth of mature tangles minus depth of compact plaques, each relative to its own section's tissue (the laminar mismatch)."),
    "plaque_thread_coupling_rho": ("Plaques and threads in the same places", "Spearman rho",
        "Rank correlation of plaque area and thread length over 100 µm squares of grey matter on the lined-up sections."),
    "plaque_thread_coupling_excess": ("Plaque–thread coupling beyond tissue layout", "rho above shifted",
        "That correlation minus its mean when the thread map is shifted by 300–1000 µm: the part not explained by where tissue and cortex are."),
    "plaque_thread_coupling_depth_adjusted": ("Plaque–thread coupling at equal depth", "partial rho",
        "The plaque–thread correlation within 250 µm depth bands from the white matter."),
    "plaque_thread_correlation_length_um": ("Reach of plaque–thread coupling", "µm",
        "Distance at which the plaque–thread correlation falls to half (from rings of squares around each square)."),
    "plaque_compact_thread_coupling": ("Compact plaques and threads together", "Spearman rho", "As the coupling, with compact plaque area."),
    "plaque_diffuse_thread_coupling": ("Diffuse plaques and threads together", "Spearman rho", "As the coupling, with diffuse plaque area."),
    "plaque_tangle_coupling_excess": ("Plaque–tau+ neuron coupling beyond tissue layout", "rho above shifted",
        "Rank correlation of plaque area and tau+ neuron density over 100 µm squares, minus its shifted-map mean."),
    "neun_other_to_neuron_clq": ("Other cells next to neurons", RATIO_1,
        "Co-location quotient: how often a non-neuronal nucleus's nearest nucleus is a neuron's, relative to the neurons' share (NeuN section)."),
    "neun_neuron_to_other_clq": ("Neurons next to other cells", RATIO_1, "How often a neuron's nucleus has a non-neuronal nucleus nearest, relative to their share."),
    "neun_satellite_fraction": ("Neurons with a satellite cell", FRACTION, "Share of NeuN+ neurons with a non-neuronal nucleus just outside their outline (satellitosis)."),
    "neun_large_satellite_fraction": ("Large neurons with a satellite cell", FRACTION, "The same for neurons of at least 150 µm²."),
    "neun_satellite_enrichment": ("Satellite cells, enrichment", RATIO_1,
        "Non-neuronal nuclei just outside neurons relative to what their density would put in the same band."),
    "neun_gap_area_fraction": ("Gaps without neurons", FRACTION,
        "Share of grey matter farther from any neuron than 95 % of the control cores' grey matter is (foci of neuron loss)."),
    "neun_neuron_share_cv_100um": ("Patchiness of the neuron share", "CV across 100 µm windows", "Variation of the neurons' share of nuclei between 100 µm windows."),
    "plaque_size_mark_correlation_50um": ("Plaque sizes alike within 50 µm", RATIO_1,
        "Mark correlation: mean product of the sizes of plaques within 50 µm of each other over the squared mean size (above 1 = large plaques near large ones)."),
    "plaque_size_mark_correlation_100um": ("Plaque sizes alike within 100 µm", RATIO_1, "The same within 100 µm."),
    "plaque_compact_in_diffuse_fraction": ("Compact plaques in diffuse amyloid", FRACTION, "Share of compact plaques centred inside or within 20 µm of a diffuse plaque."),
    "plaque_compact_in_diffuse_enrichment": ("Compact plaques in diffuse amyloid, enrichment", RATIO_1,
        "That share relative to the share of grey matter within 20 µm of a diffuse plaque."),
    "plaque_compactness_diffuse_neighbourhood_rho": ("Compactness vs diffuse amyloid near by", "Spearman rho",
        "Rank correlation of a plaque's compactness score with the diffuse plaque area within 100 µm of it."),
}
for _size in ("100", "200"):
    _where = f"{_size} µm squares of grey matter on the lined-up sections"
    SPATIAL |= {
        f"niche{_size}_amyloid_tau_fraction": (f"Amyloid + tau squares ({_size} µm)", FRACTION,
            f"Share of {_where} in the top third of their TMA's squares for both plaque area and tau."),
        f"niche{_size}_amyloid_only_fraction": (f"Amyloid-only squares ({_size} µm)", FRACTION, f"Share of {_where} high for plaques only."),
        f"niche{_size}_tau_only_fraction": (f"Tau-only squares ({_size} µm)", FRACTION, f"Share of {_where} high for tau only."),
        f"niche{_size}_quiet_fraction": (f"Quiet squares ({_size} µm)", FRACTION, f"Share of {_where} high for neither."),
        f"niche{_size}_amyloid_tau_neuron_poor_fraction": (f"Amyloid + tau, neuron-poor squares ({_size} µm)", FRACTION,
            f"Share of {_where} high for plaques and tau and in the bottom third for neurons."),
        f"niche{_size}_entropy": (f"Mix of square types ({_size} µm)", "0 = one type, 1 = all eight evenly", "How evenly the core's squares spread over the eight types."),
        f"niche{_size}_damage_quiet_contact_ratio": (f"Damaged squares bordering quiet ones ({_size} µm)", RATIO_1,
            "Amyloid + tau squares sharing a side with quiet squares, relative to random arrangement (below 1 = damage kept together)."),
        f"niche{_size}_amyloid_tau_cohesion": (f"Damaged squares bordering each other ({_size} µm)", RATIO_1,
            "Amyloid + tau squares sharing a side with each other, relative to random arrangement."),
    }
for _column, (_label, _unit, _text) in SPATIAL.items():
    COLUMNS[_column] = _text
    LABELS[_column] = (_label, _unit, "spatial")
MASK_OBJECT = {"neun_soma": ("neuron", "neurons"), "tau_neuron": ("tau+ neuron", "tau"), "plaque": ("plaque", "plaques")}
MASK_LABEL = {  # suffix: (label with {} for the object, unit)
    "mask_count": ("{} outlines drawn", "outlines"),
    "mask_density_mm2": ("{} outline density", "outlines per mm²"),
    "area_median_um2": ("{} size from outline, median", "µm²"),
    "area_p90_um2": ("{} size, largest 10%", "µm²"),
    "area_cv": ("{} size variability", "CV"),
    "solidity_median": ("{} solidity, median", "1 = convex"),
    "circularity_median": ("{} roundness from outline, median", "1 = circle"),
    "elongation_median": ("{} elongation, median", "long ÷ short axis"),
    "normalized_dab_median": ("{} stain intensity, median", "× slide threshold"),
    "mask_area_fraction": ("Tissue area covered by {} outlines", FRACTION),
    "dense_core_fraction_median": ("{} dense-core share from outline, median", FRACTION),
    "cored_fraction": ("Cored {}s", FRACTION),
    "multicore_fraction": ("{}s with several dense cores", FRACTION),
    "mask_mature_fraction": ("Mature tangles from outlines", FRACTION),
}
UNIT_SUFFIX = (  # for other columns: (suffix, words it becomes in the label, unit)
    ("_density_mm2", " density", "per mm²"), ("_mm_per_mm2", "", "mm per mm²"), ("_um2", "", "µm²"), ("_mm2", "", "mm²"),
    ("_um", "", "µm"), ("_mm", "", "mm"), ("_fraction", " fraction", FRACTION), ("_count", " count", "count"),
)


def _sentence(text: str) -> str:
    return text[:1].upper() + text[1:]


def measure(column: str) -> dict:
    """How to show one table column: a readable label, its unit, its topic and what it means."""
    info = {"column": column, "description": describe(column), "key": column in KEY_MEASURES, "outline": False}
    if column in LABELS:
        label, unit, topic = LABELS[column]
        return {**info, "label": label, "unit": unit, "topic": topic}
    for prefix, (noun, topic) in MASK_OBJECT.items():
        rest = column[len(prefix) + 1:] if column.startswith(f"{prefix}_") else ""
        if rest in MASK_LABEL:
            label, unit = MASK_LABEL[rest]
            return {**info, "label": _sentence(label.format(noun)), "unit": unit, "topic": topic, "outline": True}
    stem, unit = column, ""
    for suffix, words, suffix_unit in UNIT_SUFFIX:
        if column.endswith(suffix) and len(column) > len(suffix):
            stem, unit = column[: -len(suffix)] + words, suffix_unit
            break
    return {**info, "label": _sentence(stem.replace("_", " ").strip()), "unit": unit, "topic": "other"}
