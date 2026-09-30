from __future__ import annotations

import numpy as np

from stainid.nuclei import clean_nuclei, nucleus_centroids
from stainid.pipelines.cohort_v1 import centered_objects
from stainid.stains.tau.network import segment_at8_network
from stainid.stains.tau.neurons import add_context_features, classify_tau_neurons, tau_neuron_candidates
from stainid.stains.tau.threads import thread_network


def analyze_at8_v2(
    rgb: np.ndarray,
    inner: tuple[slice, slice],
    nuclei: np.ndarray,
    dab_threshold: float,
    pixel_size_um: float,
    exclusion: np.ndarray,
    tau_bundle: dict[str, object],
) -> tuple[dict[str, float | int], list[dict[str, object]]]:
    nuclei = clean_nuclei(nuclei, rgb)
    _, _, maps = segment_at8_network(rgb, dab_threshold, pixel_size_um, pixel_size_um, exclusion)
    valid = maps["valid"]
    inner_valid = valid[inner]
    area_mm2 = inner_valid.sum() * pixel_size_um**2 / 1e6
    objects, _ = tau_neuron_candidates(rgb, nuclei, dab_threshold, pixel_size_um, exclusion)
    rows = classify_tau_neurons(add_context_features(rgb, objects, pixel_size_um), tau_bundle, pixel_size_um)
    rows = [row for row in centered_objects(rows, inner) if valid[int(row["centroid_y_px"]), int(row["centroid_x_px"])]]
    neurons = [row for row in rows if row["tau_neuron"]]
    mature = sum(bool(row["tau_neuron_mature"]) for row in neurons)
    network = thread_network(rgb, valid, dab_threshold, pixel_size_um, nuclei=nuclei)
    skeleton = network["skeleton"][inner] & inner_valid
    length_mm = skeleton.sum() * pixel_size_um / 1000
    widths = network["width_um"][inner][skeleton]
    points = nucleus_centroids(nuclei, valid, pixel_size_um)
    inside = points[(points[:, 0] >= inner[1].start) & (points[:, 0] < inner[1].stop) & (points[:, 1] >= inner[0].start) & (points[:, 1] < inner[0].stop)] if len(points) else points
    summary = {
        "v2_tissue_area_mm2": area_mm2,
        "at8_positive_area_mm2": float((maps["positive"][inner] & inner_valid).sum() * pixel_size_um**2 / 1e6),
        "tau_neuron_count": len(neurons),
        "tau_neuron_mature_count": mature,
        "tau_neuron_density_mm2": len(neurons) / area_mm2 if area_mm2 else float("nan"),
        "tau_neuron_soma_area_sum_um2": float(sum(float(row["soma_area_um2"]) for row in neurons)),
        "thread_ridge_length_mm": length_mm,
        "thread_ridge_length_density_mm_per_mm2": length_mm / area_mm2 if area_mm2 else float("nan"),
        "thread_ridge_area_fraction": float((network["line"][inner] & inner_valid).sum() / max(inner_valid.sum(), 1)),
        "thread_ridge_width_sum_um": float(widths.sum()),
        "thread_ridge_skeleton_px": int(skeleton.sum()),
        "thread_branchpoint_count": int((network["branchpoints"][inner] & inner_valid).sum()),
        "thread_endpoint_count": int((network["endpoints"][inner] & inner_valid).sum()),
        "nucleus_count": int(len(inside)),
        "nucleus_density_mm2": len(inside) / area_mm2 if area_mm2 else float("nan"),
    }
    keep = [
        "candidate_source", "centroid_x_px", "centroid_y_px", "soma_area_um2", "soma_solidity", "soma_circularity",
        "soma_elongation", "soma_dab_mean", "soma_dab_p90", "normalized_contrast", "tau_neuron_probability", "tau_neuron", "tau_neuron_mature",
    ]
    return summary, [{key: row[key] for key in keep} for row in neurons]
