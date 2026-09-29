from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from stainid.stains.amyloid.classifier import context_features, efficientnet_embeddings, raw_review_patch

MORPHOLOGY_FEATURES = (
    "area_um2",
    "perimeter_um",
    "circularity",
    "solidity",
    "aspect_ratio",
    "major_axis_um",
    "minor_axis_um",
    "mean_dab_od",
    "max_dab_od",
    "mean_local_thread_fraction",
    "tangle_tracer_overlap",
    "tangle_tracer_max_confidence",
)

ALLOWED_LABELS = {
    "tau_soma_or_nft",
    "neuritic_plaque",
    "neuropil_thread_region",
    "other_at8_positive",
    "artifact_or_nonspecific",
    "uncertain",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_review_records(review_dirs: list[Path]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
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
            if visual_label == "uncertain":
                continue
            row = keys[label["review_id"]]
            candidate_id = row["candidate_id"]
            if candidate_id in candidate_ids:
                raise ValueError(f"Candidate reviewed more than once: {candidate_id}")
            candidate_ids.add(candidate_id)
            source_kind = row["source_kind"]
            if source_kind == "compact_profile":
                task = "compact_soma"
                target = int(visual_label == "tau_soma_or_nft")
            elif source_kind == "thread_cluster":
                task = "thread_region"
                target = int(visual_label == "neuropil_thread_region")
            else:
                raise ValueError(f"Unsupported AT8 candidate source: {source_kind}")
            records.append(
                {
                    **row,
                    "visual_label": visual_label,
                    "review_confidence": label.get("confidence", ""),
                    "review_rationale": label.get("rationale", ""),
                    "reviewer_type": label.get("reviewer_type", ""),
                    "task": task,
                    "target": target,
                    "review_round": review_dir.name,
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
    names = list(MORPHOLOGY_FEATURES)
    context_by_path: dict[str, dict[str, float]] = {}
    if include_context:
        for row in records:
            area_um2 = float(row["area_um2"])
            context_by_path[str(row["patch_path"])] = context_features(
                raw_review_patch(Path(str(row["patch_path"]))),
                2.0 * np.sqrt(area_um2 / np.pi),
                context_um=110.0,
            )
        names += list(next(iter(context_by_path.values())))
    matrix = []
    for row in records:
        values = [_number(row[name]) for name in MORPHOLOGY_FEATURES]
        if include_context:
            values += [
                context_by_path[str(row["patch_path"])][name]
                for name in names[len(MORPHOLOGY_FEATURES) :]
            ]
        matrix.append(values)
    return np.asarray(matrix, dtype=float), names


__all__ = [
    "MORPHOLOGY_FEATURES",
    "build_tabular_matrix",
    "efficientnet_embeddings",
    "load_review_records",
]
