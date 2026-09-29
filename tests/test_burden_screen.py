import unittest

import cv2
import numpy as np

from stainid.analysis.burden_screen import aggregate_donor_regions, common_range, icc_oneway, summarize_preview_burden


class BurdenScreenTest(unittest.TestCase):
    def test_summarizes_and_aggregates_positive_area(self):
        image = np.full((100, 100, 3), (210, 205, 210), dtype=np.uint8)
        cv2.circle(image, (50, 50), 15, (100, 65, 35), -1)

        summary = summarize_preview_burden(image, 0.08, 200, 200, 0.5, 0.5)

        self.assertGreater(summary["positive_area_fraction"], 0.0)
        rows = [
            {
                "sample_region_id": "BRC_1_FR",
                "donor_id": "1",
                "region": "frontal",
                "disease_group": "ASYMP",
                "cerad": "C",
                "braak": "5",
                "stain": "6E10",
                "tissue_area_mm2": 2.0,
                "positive_area_mm2": 0.2,
            },
            {
                "sample_region_id": "BRC_1_FR",
                "donor_id": "1",
                "region": "frontal",
                "disease_group": "ASYMP",
                "cerad": "C",
                "braak": "5",
                "stain": "6E10",
                "tissue_area_mm2": 1.0,
                "positive_area_mm2": 0.4,
            },
        ]
        observed = aggregate_donor_regions(rows)[0]
        self.assertEqual(observed["technical_core_count"], 2)
        self.assertAlmostEqual(observed["positive_area_fraction"], 0.2)

    def test_common_range_and_icc(self):
        self.assertEqual(
            common_range(np.array([0.0, 2.0]), np.array([1.0, 3.0])),
            (1.0, 2.0),
        )
        self.assertAlmostEqual(
            icc_oneway(np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])),
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
