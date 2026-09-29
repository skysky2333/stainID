import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np

from stainid.imaging.calibration import calibrate_threshold
from stainid.imaging.color import rgb_to_hed
from stainid.imaging.tissue import field_tissue_mask
from stainid.pipelines.threshold_baseline import summarize_field
from stainid.tables import write_csv


class StainBaselineTest(unittest.TestCase):
    def test_tissue_and_dab_object_are_measured(self):
        rgb = np.full((256, 256, 3), 255, dtype=np.uint8)
        cv2.rectangle(rgb, (16, 16), (239, 239), (225, 215, 225), -1)
        cv2.circle(rgb, (128, 128), 25, (120, 75, 45), -1)
        tissue = field_tissue_mask(rgb)
        dab = rgb_to_hed(rgb)[..., 2]
        threshold, _ = calibrate_threshold(dab[tissue])

        summary, objects, _, positive = summarize_field(
            rgb, "6E10", threshold, 0.25, 0.25
        )

        self.assertGreater(summary["positive_area_fraction"], 0.0)
        self.assertGreaterEqual(summary["object_count"], 1)
        self.assertGreater(len(objects), 0)
        self.assertTrue(positive[128, 128])
        self.assertFalse(tissue[0, 0])

    def test_calibration_rejects_too_few_pixels(self):
        with self.assertRaisesRegex(ValueError, "At least 100"):
            calibrate_threshold(np.zeros(99, dtype=np.float32))

    def test_empty_csv_uses_explicit_columns(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "objects.csv"
            write_csv(path, [], ["object_id", "area_um2"])
            self.assertEqual(path.read_text(), "object_id,area_um2\n")


if __name__ == "__main__":
    unittest.main()
