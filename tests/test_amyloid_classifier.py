import csv
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from stainid.stains.amyloid.classifier import context_features, load_review_records, raw_review_patch


class AmyloidClassifierTests(unittest.TestCase):
    def test_loads_resolved_labels_and_excludes_uncertain(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with (root / "selection_key.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["review_id", "candidate_id"])
                writer.writeheader()
                writer.writerows(
                    [
                        {"review_id": "AR001", "candidate_id": "C1"},
                        {"review_id": "AR002", "candidate_id": "C2"},
                    ]
                )
            with (root / "review_labels.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["review_id", "visual_label"])
                writer.writeheader()
                writer.writerows(
                    [
                        {"review_id": "AR001", "visual_label": "plaque"},
                        {"review_id": "AR002", "visual_label": "uncertain"},
                    ]
                )
            records = load_review_records([root])
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["target"], 1)

    def test_extracts_raw_half_and_context_features(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "patch.jpg"
            image = np.full((34 + 80, 160, 3), 255, dtype=np.uint8)
            image[34:, :80] = (80, 120, 180)
            cv2.imwrite(str(path), image)
            raw = raw_review_patch(path)
            features = context_features(raw, 20.0)
            self.assertEqual(raw.shape, (80, 80, 3))
            self.assertIn("center_high_hematoxylin_fraction", features)
            self.assertTrue(all(np.isfinite(list(features.values()))))


if __name__ == "__main__":
    unittest.main()
