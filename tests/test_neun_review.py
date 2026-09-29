import csv
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from stainid.stains.neun.profiles import combine_profile_and_cellpose_reviews, segment_dab_profiles, write_review_geojson, write_review_tables


class NeuNReviewTest(unittest.TestCase):
    def test_combines_visible_perikarya_with_cellpose_candidates(self):
        image = np.full((256, 256, 3), (225, 220, 225), dtype=np.uint8)
        cv2.circle(image, (70, 70), 18, (125, 85, 50), 7)
        cv2.circle(image, (180, 70), 13, (105, 65, 35), -1)
        cv2.circle(image, (70, 180), 9, (65, 75, 170), -1)
        profiles, profile_labels = segment_dab_profiles(
            image, 0.04, 0.274, 0.274
        )
        self.assertGreaterEqual(len(profiles), 2)
        self.assertTrue(any(row["candidate_class"] == "positive_profile" for row in profiles))

        cellpose_labels = np.zeros(image.shape[:2], dtype=np.int32)
        cv2.circle(cellpose_labels, (70, 180), 9, 1, -1)
        cv2.circle(cellpose_labels, (220, 180), 10, 2, -1)
        cv2.circle(cellpose_labels, (180, 180), 10, 3, -1)
        cellpose_objects = [
            {
                "label": 1,
                "centroid_x_px": 70.0,
                "centroid_y_px": 180.0,
                "area_um2": 20.0,
                "mean_hematoxylin_od": 0.10,
                "mean_dab_od": 0.005,
                "candidate_class": "negative",
                "touches_edge": "false",
            },
            {
                "label": 2,
                "centroid_x_px": 220.0,
                "centroid_y_px": 180.0,
                "area_um2": 25.0,
                "mean_hematoxylin_od": 0.03,
                "mean_dab_od": 0.04,
                "candidate_class": "positive",
                "touches_edge": "false",
            },
            {
                "label": 3,
                "centroid_x_px": 180.0,
                "centroid_y_px": 180.0,
                "area_um2": 25.0,
                "mean_hematoxylin_od": 0.015,
                "mean_dab_od": 0.005,
                "candidate_class": "negative",
                "touches_edge": "false",
            },
        ]
        objects = combine_profile_and_cellpose_reviews(
            0.04,
            profiles,
            profile_labels,
            cellpose_labels,
            cellpose_objects,
        )
        classes = [row["candidate_class"] for row in objects]
        self.assertIn("negative", classes)
        self.assertIn("review", classes)
        self.assertFalse(
            any(
                row["source_kind"] == "cellpose" and row["label"] == 3
                for row in objects
            )
        )

        with tempfile.TemporaryDirectory() as directory:
            path = write_review_geojson(
                image,
                profile_labels,
                cellpose_labels,
                objects,
                Path(directory) / "review.geojson",
                "M001_NeuN",
                "test_review",
            )
            self.assertTrue(path.is_file())
            object_path, summary_path = write_review_tables(
                objects,
                Path(directory) / "objects.csv",
                Path(directory) / "summary.csv",
                "M001_NeuN",
                0.125,
            )
            with object_path.open(newline="", encoding="utf-8") as handle:
                object_rows = list(csv.DictReader(handle))
            with summary_path.open(newline="", encoding="utf-8") as handle:
                summary = next(csv.DictReader(handle))
            self.assertEqual(len(object_rows), len(objects))
            self.assertIn("decision_basis", object_rows[0])
            self.assertIn("mean_dab_od", object_rows[0])
            self.assertEqual(summary["manual_exclusion_fraction"], "0.12500000")


if __name__ == "__main__":
    unittest.main()
