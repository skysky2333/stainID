import csv
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from stainid.stains.amyloid.segmentation import apply_amyloid_class_overrides, segment_amyloid_candidates, write_amyloid_geojson, write_amyloid_tables


class AmyloidReviewTest(unittest.TestCase):
    def test_segments_nested_plaque_extent_and_compact_core(self):
        image = np.full((512, 512, 3), (225, 220, 225), dtype=np.uint8)
        cv2.circle(image, (140, 150), 42, (180, 145, 110), -1)
        cv2.circle(image, (140, 150), 12, (80, 45, 25), -1)
        cv2.circle(image, (365, 350), 38, (175, 135, 100), -1)
        cv2.circle(image, (345, 340), 8, (145, 100, 70), -1)
        cv2.circle(image, (380, 360), 7, (145, 100, 70), -1)

        summary, objects, labels, cores, excluded = segment_amyloid_candidates(
            image, 0.05, 0.274, 0.274
        )

        self.assertGreaterEqual(len(objects), 2)
        self.assertTrue(any(row["candidate_class"] == "compact" for row in objects))
        self.assertGreater(int((cores > 0).sum()), 0)
        self.assertGreater(summary["accepted_plaque_candidate_count"], 0)

        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            geojson = write_amyloid_geojson(
                labels,
                cores,
                objects,
                excluded,
                directory_path / "candidates.geojson",
                "M001_6E10",
                "test_amyloid",
            )
            object_path, summary_path = write_amyloid_tables(
                objects,
                summary,
                directory_path / "objects.csv",
                directory_path / "summary.csv",
                "M001_6E10",
            )
            self.assertTrue(geojson.is_file())
            with object_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            with summary_path.open(newline="", encoding="utf-8") as handle:
                stored_summary = next(csv.DictReader(handle))
            self.assertEqual(len(rows), len(objects))
            self.assertIn("radial_dab_contrast", rows[0])
            self.assertEqual(
                stored_summary["review_status"], "ai_proposed_development"
            )

    def test_marks_long_dab_structure_as_artifact(self):
        image = np.full((512, 512, 3), (225, 220, 225), dtype=np.uint8)
        cv2.line(image, (40, 250), (470, 270), (75, 50, 45), 20)
        _, objects, _, _, _ = segment_amyloid_candidates(
            image, 0.05, 0.274, 0.274
        )
        self.assertTrue(any(row["candidate_class"] == "artifact" for row in objects))

    def test_applies_auditable_candidate_class_override(self):
        objects = [
            {
                "candidate_class": "review",
                "decision_basis": "automatic",
                "deposit_area_um2": 25.0,
            }
        ]
        summary = {"tissue_area_mm2": 0.25}
        apply_amyloid_class_overrides(
            objects,
            summary,
            "M001_6E10",
            {"M001_6E10_AI_A00001": "artifact"},
        )
        self.assertEqual(objects[0]["candidate_class"], "artifact")
        self.assertEqual(summary["accepted_plaque_candidate_count"], 0)
        self.assertEqual(summary["artifact_candidate_count"], 1)

    def test_keeps_irregular_solid_deposit_as_biological_candidate(self):
        image = np.full((512, 512, 3), (225, 220, 225), dtype=np.uint8)
        for center, radius in [((180, 250), 70), ((265, 205), 65), ((320, 285), 75)]:
            cv2.circle(image, center, radius, (155, 115, 80), -1)
        cv2.circle(image, (265, 245), 25, (75, 45, 25), -1)
        _, objects, _, _, _ = segment_amyloid_candidates(
            image, 0.05, 0.274, 0.274
        )
        large = max(objects, key=lambda row: float(row["deposit_area_um2"]))
        self.assertNotEqual(large["candidate_class"], "artifact")


if __name__ == "__main__":
    unittest.main()
