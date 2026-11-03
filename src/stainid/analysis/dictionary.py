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
