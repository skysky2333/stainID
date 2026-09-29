from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

POSITIVE_LABEL = "neun_positive_profile"
NONPOSITIVE_LABELS = {"neun_negative_nucleus", "artifact_or_nonspecific"}
EXCLUDED_LABEL = "merged_truncated_or_uncertain"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def candidate_validation_rows(
    selection: list[dict[str, str]],
    expert: list[dict[str, str]],
    predictions: list[dict[str, str]],
    threshold: float,
    probability_column: str = "positive_probability",
) -> list[dict[str, object]]:
    selection_by_review = {row["review_id"]: row for row in selection}
    if set(selection_by_review) != {row["review_id"] for row in expert}:
        raise ValueError("NeuN expert candidate labels do not match the selection key")
    prediction_by_id = {row["candidate_id"]: row for row in predictions}
    rows = []
    for label in expert:
        expert_label = label["expert_label"]
        if expert_label not in {POSITIVE_LABEL, *NONPOSITIVE_LABELS, EXCLUDED_LABEL}:
            raise ValueError(f"Unresolved NeuN expert label: {expert_label}")
        selected = selection_by_review[label["review_id"]]
        candidate_id = selected["candidate_id"]
        if candidate_id not in prediction_by_id:
            raise ValueError(f"Missing locked NeuN prediction: {candidate_id}")
        prediction = prediction_by_id[candidate_id]
        if expert_label == EXCLUDED_LABEL:
            continue
        target = int(expert_label == POSITIVE_LABEL)
        probability = float(prediction[probability_column])
        rows.append(
            {
                "review_id": label["review_id"],
                "candidate_id": candidate_id,
                "tma": selected["tma"],
                "sampling_group": selected["sampling_group"],
                "sampling_weight": float(selected["sampling_weight"]),
                "expert_label": expert_label,
                "target": target,
                "probability_source": probability_column,
                "positive_probability": probability,
                "prediction": int(probability >= threshold),
            }
        )
    return rows


def weighted_metrics(rows: list[dict[str, object]]) -> dict[str, float | int]:
    if not rows:
        raise ValueError("NeuN validation has no resolved candidate labels")
    target = np.asarray([int(row["target"]) for row in rows])
    prediction = np.asarray([int(row["prediction"]) for row in rows])
    weight = np.asarray([float(row["sampling_weight"]) for row in rows])
    tp = float(weight[(target == 1) & (prediction == 1)].sum())
    fp = float(weight[(target == 0) & (prediction == 1)].sum())
    tn = float(weight[(target == 0) & (prediction == 0)].sum())
    fn = float(weight[(target == 1) & (prediction == 0)].sum())
    raw_tp = int(np.sum((target == 1) & (prediction == 1)))
    raw_fp = int(np.sum((target == 0) & (prediction == 1)))
    raw_tn = int(np.sum((target == 0) & (prediction == 0)))
    raw_fn = int(np.sum((target == 1) & (prediction == 0)))
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    specificity = tn / (tn + fp) if tn + fp else float("nan")
    raw_precision = raw_tp / (raw_tp + raw_fp) if raw_tp + raw_fp else float("nan")
    raw_recall = raw_tp / (raw_tp + raw_fn) if raw_tp + raw_fn else float("nan")
    raw_specificity = raw_tn / (raw_tn + raw_fp) if raw_tn + raw_fp else float("nan")
    return {
        "reviewed_n": len(rows),
        "unweighted_true_positive": raw_tp,
        "unweighted_false_positive": raw_fp,
        "unweighted_true_negative": raw_tn,
        "unweighted_false_negative": raw_fn,
        "unweighted_precision": raw_precision,
        "unweighted_recall": raw_recall,
        "unweighted_specificity": raw_specificity,
        "unweighted_balanced_accuracy": (raw_recall + raw_specificity) / 2,
        "unweighted_f1": 2 * raw_precision * raw_recall / (raw_precision + raw_recall)
        if raw_precision + raw_recall
        else float("nan"),
        "weighted_population_n": float(weight.sum()),
        "weighted_true_positive": tp,
        "weighted_false_positive": fp,
        "weighted_true_negative": tn,
        "weighted_false_negative": fn,
        "weighted_precision": precision,
        "weighted_recall": recall,
        "weighted_specificity": specificity,
        "weighted_balanced_accuracy": (recall + specificity) / 2,
        "weighted_f1": 2 * precision * recall / (precision + recall)
        if precision + recall
        else float("nan"),
    }


def field_validation_rows(
    selection: list[dict[str, str]],
    expert: list[dict[str, str]],
    predictions: list[dict[str, str]],
) -> list[dict[str, object]]:
    selection_by_review = {row["review_id"]: row for row in selection}
    prediction_by_review = {row["review_id"]: row for row in predictions}
    if set(selection_by_review) != {row["review_id"] for row in expert}:
        raise ValueError("NeuN expert field counts do not match the selection key")
    if set(selection_by_review) != set(prediction_by_review):
        raise ValueError("NeuN field predictions do not match the selection key")
    rows = []
    for label in expert:
        if label["unscorable"].lower() not in {"true", "false"}:
            raise ValueError(f"NeuN field scorable status is unresolved: {label['review_id']}")
        if label["unscorable"].lower() == "true":
            continue
        if not label["neun_positive_profile_count"]:
            raise ValueError(f"NeuN field count is unresolved: {label['review_id']}")
        selected = selection_by_review[label["review_id"]]
        prediction = prediction_by_review[label["review_id"]]
        rows.append(
            {
                "review_id": label["review_id"],
                "image_id": selected["image_id"],
                "tma": selected["tma"],
                "expert_count": int(label["neun_positive_profile_count"]),
                "uncertain_count": int(label["uncertain_profile_count"] or 0),
                "predicted_count": int(prediction["classifier_positive_count"]),
            }
        )
    return rows


def field_count_metrics(rows: list[dict[str, object]]) -> dict[str, float | int]:
    if len(rows) < 3:
        raise ValueError("NeuN count calibration requires at least three fields")
    expert = np.asarray([int(row["expert_count"]) for row in rows], dtype=float)
    predicted = np.asarray([int(row["predicted_count"]) for row in rows], dtype=float)
    difference = predicted - expert
    return {
        "field_n": len(rows),
        "expert_total": int(expert.sum()),
        "predicted_total": int(predicted.sum()),
        "mean_bias_profiles": float(difference.mean()),
        "mean_absolute_error_profiles": float(np.abs(difference).mean()),
        "root_mean_squared_error_profiles": float(np.sqrt(np.mean(difference**2))),
        "pearson_r": float(np.corrcoef(expert, predicted)[0, 1]),
    }


__all__ = [
    "EXCLUDED_LABEL",
    "NONPOSITIVE_LABELS",
    "POSITIVE_LABEL",
    "candidate_validation_rows",
    "field_count_metrics",
    "field_validation_rows",
    "read_csv",
    "weighted_metrics",
]
