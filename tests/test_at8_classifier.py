import csv
import tempfile
import unittest
from pathlib import Path

from stainid.stains.tau.classifier import MORPHOLOGY_FEATURES, build_tabular_matrix, load_review_records


class At8ClassifierTest(unittest.TestCase):
    def test_loads_tasks_and_excludes_uncertain(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fields = ["review_id", "candidate_id", "source_kind"]
            with (root / "selection_key.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(
                    [
                        {
                            "review_id": "TR001",
                            "candidate_id": "C1",
                            "source_kind": "compact_profile",
                        },
                        {
                            "review_id": "TR002",
                            "candidate_id": "C2",
                            "source_kind": "thread_cluster",
                        },
                        {
                            "review_id": "TR003",
                            "candidate_id": "C3",
                            "source_kind": "thread_cluster",
                        },
                    ]
                )
            with (root / "review_labels.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["review_id", "visual_label"])
                writer.writeheader()
                writer.writerows(
                    [
                        {"review_id": "TR001", "visual_label": "tau_soma_or_nft"},
                        {"review_id": "TR002", "visual_label": "neuropil_thread_region"},
                        {"review_id": "TR003", "visual_label": "uncertain"},
                    ]
                )
            records = load_review_records([root])
            self.assertEqual([row["task"] for row in records], ["compact_soma", "thread_region"])
            self.assertEqual([row["target"] for row in records], [1, 1])

    def test_builds_morphology_matrix_with_boolean_feature(self):
        record = {name: "1.5" for name in MORPHOLOGY_FEATURES}
        record["tangle_tracer_overlap"] = "true"
        matrix, names = build_tabular_matrix([record], False)
        self.assertEqual(matrix.shape, (1, len(MORPHOLOGY_FEATURES)))
        self.assertEqual(matrix[0, names.index("tangle_tracer_overlap")], 1.0)


if __name__ == "__main__":
    unittest.main()
