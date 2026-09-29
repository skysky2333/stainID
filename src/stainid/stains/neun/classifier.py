from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np

from stainid.stains.amyloid.classifier import context_features, raw_review_patch
from stainid.stains.neun.audit import crop_with_padding

MEASUREMENT_FEATURES = (
    "selection_area_um2",
    "stained_area_um2",
    "profile_area_um2",
    "area_um2",
    "aspect_ratio",
    "solidity",
    "circularity",
    "mean_dab_od",
    "mean_hematoxylin_od",
    "neighborhood_dab_p80",
    "chromogen_fraction",
    "mean_chromogen_fraction",
    "tissue_fraction",
)

ALLOWED_LABELS = {
    "neun_positive_profile",
    "neun_negative_nucleus",
    "artifact_or_nonspecific",
    "truncated_or_uncertain",
}
REFERENCE_LABELS = {
    "neun_positive_profile",
    "neun_negative_nucleus",
    "artifact_or_nonspecific",
    "merged_truncated_or_uncertain",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_review_records(review_dirs: list[Path]) -> list[dict[str, object]]:
    records = []
    candidate_ids: set[str] = set()
    for review_dir in review_dirs:
        keys = {row["review_id"]: row for row in read_csv(review_dir / "selection_key.csv")}
        labels = read_csv(review_dir / "review_labels.csv")
        if set(keys) != {row["review_id"] for row in labels}:
            raise ValueError(f"Review IDs do not match in {review_dir}")
        for label in labels:
            visual_label = label["visual_label"]
            if visual_label not in ALLOWED_LABELS:
                raise ValueError(f"Unresolved visual label in {review_dir}: {visual_label}")
            if visual_label == "truncated_or_uncertain":
                continue
            row = keys[label["review_id"]]
            candidate_id = row["candidate_id"]
            if candidate_id in candidate_ids:
                raise ValueError(f"Candidate reviewed more than once: {candidate_id}")
            candidate_ids.add(candidate_id)
            records.append(
                {
                    **row,
                    "visual_label": visual_label,
                    "target": int(visual_label == "neun_positive_profile"),
                    "review_confidence": label["confidence"],
                    "review_rationale": label["rationale"],
                    "reviewer_type": label["reviewer_type"],
                    "review_round": review_dir.parent.name,
                    "patch_path": str(
                        review_dir / "patches" / f"{label['review_id']}.jpg"
                    ),
                }
            )
    return records


def load_reference_records(label_paths: list[Path]) -> list[dict[str, object]]:
    records = []
    candidate_ids: set[str] = set()
    for label_path in label_paths:
        review_dir = label_path.parent
        keys = {
            row["review_id"]: row
            for row in read_csv(review_dir / "selection_key.csv")
        }
        labels = read_csv(label_path)
        if set(keys) != {row["review_id"] for row in labels}:
            raise ValueError(f"Reference IDs do not match in {label_path}")
        for label in labels:
            visual_label = label["expert_label"]
            if visual_label not in REFERENCE_LABELS:
                raise ValueError(f"Unresolved reference label in {label_path}: {visual_label}")
            if visual_label == "merged_truncated_or_uncertain":
                continue
            row = keys[label["review_id"]]
            candidate_id = row["candidate_id"]
            if candidate_id in candidate_ids:
                raise ValueError(f"Candidate reviewed more than once: {candidate_id}")
            candidate_ids.add(candidate_id)
            records.append(
                {
                    **row,
                    "visual_label": visual_label,
                    "target": int(visual_label == "neun_positive_profile"),
                    "review_confidence": label["confidence"],
                    "review_rationale": label["notes"],
                    "reviewer_type": label["reviewer"],
                    "review_round": review_dir.parent.name,
                    "patch_path": str(
                        review_dir / "patches" / f"{label['review_id']}.jpg"
                    ),
                }
            )
    return records


def _number(value: object) -> float:
    text = str(value).strip().lower()
    if text == "true":
        return 1.0
    if text == "false":
        return 0.0
    return float(text) if text else float("nan")


def build_tabular_matrix(
    records: list[dict[str, object]],
    include_context: bool,
) -> tuple[np.ndarray, list[str]]:
    contexts = None
    if include_context:
        contexts = []
        for row in records:
            diameter_um = 2.0 * np.sqrt(float(row["selection_area_um2"]) / np.pi)
            contexts.append(
                context_features(
                    raw_review_patch(Path(str(row["patch_path"]))),
                    diameter_um,
                    context_um=55.0,
                )
            )
    return build_matrix_from_contexts(records, contexts)


def build_matrix_from_contexts(
    records: list[dict[str, object]],
    contexts: list[dict[str, float]] | None,
) -> tuple[np.ndarray, list[str]]:
    names = list(MEASUREMENT_FEATURES) + [
        "source_is_dab_profile",
        "proposal_is_positive_profile",
        "source_candidate_is_positive",
        "source_candidate_is_review",
    ]
    if contexts is not None:
        if len(contexts) != len(records):
            raise ValueError("NeuN contexts must match the record count")
        names += list(contexts[0])
    matrix = []
    for index, row in enumerate(records):
        values = [_number(row.get(name, "")) for name in MEASUREMENT_FEATURES]
        values += [
            float(row["source_kind"] == "dab_profile"),
            float(row["candidate_class"] == "positive_profile"),
            float(row["source_candidate_class"] == "positive"),
            float(row["source_candidate_class"] == "review"),
        ]
        if contexts is not None:
            values += [
                contexts[index][name]
                for name in names[len(MEASUREMENT_FEATURES) + 4 :]
            ]
        matrix.append(values)
    return np.asarray(matrix, dtype=float), names


def object_contexts(
    bgr: np.ndarray,
    rows: list[dict[str, object]],
    pixel_size_um: float,
    context_um: float,
) -> list[dict[str, float]]:
    crop_size = max(64, round(context_um / pixel_size_um))
    contexts = []
    for row in rows:
        crop = crop_with_padding(
            bgr,
            float(row["centroid_x_px"]),
            float(row["centroid_y_px"]),
            crop_size,
        )
        diameter_um = 2.0 * np.sqrt(float(row["selection_area_um2"]) / np.pi)
        contexts.append(
            context_features(
                cv2.cvtColor(crop, cv2.COLOR_BGR2RGB),
                diameter_um,
                context_um=context_um,
            )
        )
    return contexts


def rule_probabilities(
    records: list[dict[str, object]],
    expanded: bool,
) -> np.ndarray:
    return np.asarray(
        [
            float(
                row["candidate_class"] == "positive_profile"
                or (
                    expanded
                    and row["source_kind"] == "cellpose"
                    and row["source_candidate_class"] == "positive"
                )
            )
            for row in records
        ]
    )


__all__ = [
    "MEASUREMENT_FEATURES",
    "build_matrix_from_contexts",
    "build_tabular_matrix",
    "load_reference_records",
    "load_review_records",
    "object_contexts",
    "rule_probabilities",
]
