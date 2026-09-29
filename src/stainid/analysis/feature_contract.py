from __future__ import annotations

import csv
from pathlib import Path

FEATURES = (
    {
        "feature_id": "amyloid_positive_area_fraction",
        "domain": "burden",
        "stain": "6E10",
        "analysis_level": "donor_region",
        "endpoint_role": "adjustment",
        "definition": "DAB-positive amyloid area divided by usable tissue area",
        "replicate_aggregation": "sum positive and usable areas, then divide",
        "validation_gate": "positive-area calibration and replicate ICC",
        "registration_required": "false",
        "current_status": "screening_only",
    },
    {
        "feature_id": "tau_positive_area_fraction",
        "domain": "burden",
        "stain": "AT8",
        "analysis_level": "donor_region",
        "endpoint_role": "adjustment",
        "definition": "AT8-positive area divided by usable tissue area",
        "replicate_aggregation": "sum positive and usable areas, then divide",
        "validation_gate": "positive-area calibration and replicate ICC",
        "registration_required": "false",
        "current_status": "screening_only",
    },
    {
        "feature_id": "compact_plaque_fraction",
        "domain": "plaque_morphology",
        "stain": "6E10",
        "analysis_level": "donor_region",
        "endpoint_role": "primary",
        "definition": "compact or cored plaques divided by accepted plaques of at least 15 micrometers equivalent diameter; compact when inner DAB exceeds 1.589 times the slide threshold",
        "replicate_aggregation": "sum class counts, then divide",
        "validation_gate": "reference class labels and held-out-TMA F1",
        "registration_required": "false",
        "current_status": "final_test_passed",
    },
    {
        "feature_id": "tau_noncompact_area_fraction_of_at8",
        "domain": "tau_composition",
        "stain": "AT8",
        "analysis_level": "donor_region",
        "endpoint_role": "primary",
        "definition": "AT8-positive area outside thick (at least 3.6 micrometers) compact profiles divided by total AT8-positive area, after removing vacuole and hole rims",
        "replicate_aggregation": "sum noncompact and total AT8 areas, then divide",
        "validation_gate": "reference compact-profile labels, neuritic-region masks, and held-out-TMA overlap",
        "registration_required": "false",
        "current_status": "revised_pending_cohort_audit",
    },
    {
        "feature_id": "neun_profile_density_mm2",
        "domain": "neuronal_context",
        "stain": "NeuN",
        "analysis_level": "donor_region",
        "endpoint_role": "primary",
        "definition": "accepted NeuN-positive profiles per square millimeter of usable parenchyma",
        "replicate_aggregation": "sum profiles and usable parenchymal area, then divide",
        "validation_gate": "reference field counts, object labels, and held-out-TMA count calibration",
        "registration_required": "false",
        "current_status": "final_test_passed",
    },
    {
        "feature_id": "plaque_density_mm2",
        "domain": "plaque_morphology",
        "stain": "6E10",
        "analysis_level": "donor_region",
        "endpoint_role": "secondary",
        "definition": "accepted parenchymal plaques per square millimeter of usable tissue",
        "replicate_aggregation": "sum plaques and usable areas, then divide",
        "validation_gate": "reference class labels and held-out-TMA F1",
        "registration_required": "false",
        "current_status": "development",
    },
    {
        "feature_id": "plaque_median_area_um2",
        "domain": "plaque_morphology",
        "stain": "6E10",
        "analysis_level": "donor_region",
        "endpoint_role": "secondary",
        "definition": "median accepted plaque deposit area",
        "replicate_aggregation": "pool accepted objects across technical cores",
        "validation_gate": "held-out boundary IoU and size calibration",
        "registration_required": "false",
        "current_status": "development",
    },
    {
        "feature_id": "plaque_boundary_irregularity_median",
        "domain": "plaque_morphology",
        "stain": "6E10",
        "analysis_level": "donor_region",
        "endpoint_role": "secondary",
        "definition": "median perimeter-squared to area ratio among accepted plaques",
        "replicate_aggregation": "pool accepted objects across technical cores",
        "validation_gate": "held-out boundary IoU and perturbation stability",
        "registration_required": "false",
        "current_status": "development",
    },
    {
        "feature_id": "at8_compact_profile_density_mm2",
        "domain": "tau_composition",
        "stain": "AT8",
        "analysis_level": "donor_region",
        "endpoint_role": "secondary",
        "definition": "accepted compact AT8 neuronal profiles per square millimeter",
        "replicate_aggregation": "sum profiles and usable areas, then divide",
        "validation_gate": "reference labels and held-out-TMA F1",
        "registration_required": "false",
        "current_status": "development",
    },
    {
        "feature_id": "at8_neuritic_plaque_density_mm2",
        "domain": "tau_composition",
        "stain": "AT8",
        "analysis_level": "donor_region",
        "endpoint_role": "secondary",
        "definition": "reference-confirmed AT8 neuritic plaques per square millimeter of usable tissue",
        "replicate_aggregation": "sum plaques and usable areas, then divide",
        "validation_gate": "reference instance labels and held-out-TMA F1",
        "registration_required": "false",
        "current_status": "annotation_required",
    },
    {
        "feature_id": "at8_thread_length_density_mm_per_mm2",
        "domain": "tau_composition",
        "stain": "AT8",
        "analysis_level": "donor_region",
        "endpoint_role": "secondary",
        "definition": "skeletonized AT8 thread length per square millimeter of usable tissue",
        "replicate_aggregation": "sum thread lengths and usable areas, then divide",
        "validation_gate": "reference thread masks and skeleton perturbation stability",
        "registration_required": "false",
        "current_status": "development",
    },
    {
        "feature_id": "neun_median_profile_area_um2",
        "domain": "neuronal_context",
        "stain": "NeuN",
        "analysis_level": "donor_region",
        "endpoint_role": "secondary",
        "definition": "median two-dimensional area of accepted NeuN-positive profiles",
        "replicate_aggregation": "pool accepted profiles across technical cores",
        "validation_gate": "common-boundary reference IoU and batch stability",
        "registration_required": "false",
        "current_status": "annotation_required",
    },
    {
        "feature_id": "neun_local_density_cv_100um",
        "domain": "neuronal_context",
        "stain": "NeuN",
        "analysis_level": "donor_region",
        "endpoint_role": "secondary",
        "definition": "coefficient of variation of NeuN-positive profile density in 100-micrometer windows",
        "replicate_aggregation": "usable-area-weighted mean across cores",
        "validation_gate": "reference counts and window-sampling stability",
        "registration_required": "false",
        "current_status": "development",
    },
    {
        "feature_id": "neun_positive_fraction_of_candidates",
        "domain": "neuronal_context_qc",
        "stain": "NeuN",
        "analysis_level": "donor_region",
        "endpoint_role": "secondary",
        "definition": "NeuN-positive profiles divided by all hematoxylin-supported profile candidates",
        "replicate_aggregation": "sum positive and candidate profile counts, then divide",
        "validation_gate": "reference positive labels and stain-batch stability",
        "registration_required": "false",
        "current_status": "development",
    },
    {
        "feature_id": "plaque_neun_neighborhood_density",
        "domain": "cross_stain_spatial",
        "stain": "6E10+NeuN",
        "analysis_level": "donor_region",
        "endpoint_role": "exploratory",
        "definition": "NeuN-positive profile density in prespecified mesoscopic plaque neighborhoods",
        "replicate_aggregation": "sum profiles and common analyzed neighborhood area, then divide",
        "validation_gate": "independent landmark target-registration error",
        "registration_required": "true",
        "current_status": "blocked_by_registration",
    },
    {
        "feature_id": "tau_neun_neighborhood_density",
        "domain": "cross_stain_spatial",
        "stain": "AT8+NeuN",
        "analysis_level": "donor_region",
        "endpoint_role": "exploratory",
        "definition": "NeuN-positive profile density in prespecified mesoscopic AT8-rich neighborhoods",
        "replicate_aggregation": "sum profiles and common analyzed neighborhood area, then divide",
        "validation_gate": "independent landmark target-registration error",
        "registration_required": "true",
        "current_status": "blocked_by_registration",
    },
)


def validate_feature_contract(rows: tuple[dict[str, str], ...] = FEATURES) -> None:
    feature_ids = [row["feature_id"] for row in rows]
    if len(feature_ids) != len(set(feature_ids)):
        raise ValueError("Feature IDs must be unique")
    if sum(row["endpoint_role"] == "primary" for row in rows) != 3:
        raise ValueError("The confirmatory contract requires exactly three primary endpoints")
    for row in rows:
        if row["registration_required"] == "true" and row["current_status"] != "blocked_by_registration":
            raise ValueError(f"Spatial feature is not registration-gated: {row['feature_id']}")


def write_feature_contract(path: Path) -> Path:
    validate_feature_contract()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(FEATURES[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(FEATURES)
    return path


__all__ = [
    "FEATURES",
    "validate_feature_contract",
    "write_feature_contract",
]
