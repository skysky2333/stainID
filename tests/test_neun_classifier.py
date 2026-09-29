import csv
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from stainid.stains.neun.classifier import (
    build_matrix_from_contexts,
    build_tabular_matrix,
    load_reference_records,
    load_review_records,
    rule_probabilities,
)


class NeuNClassifierTest(unittest.TestCase):
    def test_loads_reference_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            review = Path(directory)
            (review / "patches").mkdir()
            with (review / "selection_key.csv").open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["review_id", "candidate_id", "tma"])
                writer.writerows([["NR001", "C1", "1"], ["NR002", "C2", "1"]])
            with (review / "investigator_labels.csv").open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    ["review_id", "expert_label", "confidence", "notes", "reviewer"]
                )
                writer.writerows(
                    [
                        ["NR001", "neun_positive_profile", "high", "positive", "investigator"],
                        ["NR002", "merged_truncated_or_uncertain", "high", "exclude", "investigator"],
                    ]
                )
            records = load_reference_records([review / "investigator_labels.csv"])
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["candidate_id"], "C1")
            self.assertEqual(records[0]["target"], 1)

    def test_loads_resolved_records_and_builds_features(self):
        with tempfile.TemporaryDirectory() as directory:
            review = Path(directory)
            (review / "patches").mkdir()
            fields = [
                "review_id",
                "candidate_id",
                "tma",
                "candidate_class",
                "source_kind",
                "source_candidate_class",
                "selection_area_um2",
                "mean_dab_od",
            ]
            rows = [
                ["NR001", "C1", "1", "positive_profile", "dab_profile", "", "40", "0.1"],
                ["NR002", "C2", "2", "review", "cellpose", "positive", "20", "0.05"],
                ["NR003", "C3", "3", "negative", "cellpose", "negative", "15", "0.01"],
            ]
            with (review / "selection_key.csv").open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(fields)
                writer.writerows(rows)
            with (review / "review_labels.csv").open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    ["review_id", "visual_label", "confidence", "rationale", "reviewer_type"]
                )
                writer.writerows(
                    [
                        ["NR001", "neun_positive_profile", "high", "", "AI_visual_audit"],
                        ["NR002", "truncated_or_uncertain", "low", "", "AI_visual_audit"],
                        ["NR003", "neun_negative_nucleus", "high", "", "AI_visual_audit"],
                    ]
                )
            for review_id in ("NR001", "NR003"):
                image = np.full((256, 440, 3), 220, dtype=np.uint8)
                cv2.imwrite(str(review / "patches" / f"{review_id}.jpg"), image)
            records = load_review_records([review])
            self.assertEqual(len(records), 2)
            matrix, names = build_tabular_matrix(records, True)
            self.assertEqual(matrix.shape, (2, len(names)))
            self.assertEqual(rule_probabilities(records, False).tolist(), [1.0, 0.0])
            matrix_from_contexts, context_names = build_matrix_from_contexts(
                records,
                [
                    {name: float(index) for index, name in enumerate(names[-13:])}
                    for _ in records
                ],
            )
            self.assertEqual(matrix_from_contexts.shape, (2, len(context_names)))


if __name__ == "__main__":
    unittest.main()
