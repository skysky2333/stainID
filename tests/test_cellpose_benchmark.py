import unittest

import numpy as np

from stainid.stains.neun.cellpose import cellpose_input, classify_cellpose_masks, instance_agreement


class CellposeBenchmarkTests(unittest.TestCase):
    def test_cellpose_input_modes(self) -> None:
        rgb = np.full((16, 18, 3), 220, dtype=np.uint8)
        rgb[4:12, 6:14] = [95, 70, 45]
        self.assertTrue(np.array_equal(cellpose_input(rgb, "rgb"), rgb))
        hdab = cellpose_input(rgb, "hdab")
        self.assertEqual(hdab.shape, rgb.shape)
        self.assertEqual(hdab.dtype, np.float32)
        self.assertGreater(float(hdab[8, 8].max()), float(hdab[0, 0].max()))
        with self.assertRaises(ValueError):
            cellpose_input(rgb, "other")

    def test_instance_agreement_uses_one_to_one_iou_matching(self) -> None:
        reference = np.zeros((20, 20), dtype=np.int32)
        reference[2:8, 2:8] = 1
        reference[12:18, 12:18] = 2
        comparison = np.zeros_like(reference)
        comparison[3:9, 3:9] = 1
        comparison[0:3, 15:18] = 2
        result = instance_agreement(reference, comparison, minimum_iou=0.3)
        self.assertEqual(result["matched_count"], 1)
        self.assertEqual(result["reference_match_fraction"], 0.5)
        self.assertEqual(result["comparison_match_fraction"], 0.5)
        self.assertAlmostEqual(result["median_matched_iou"], 25 / 47)

    def test_cellpose_mask_classification_filters_background(self) -> None:
        rgb = np.full((128, 128, 3), 245, dtype=np.uint8)
        rgb[8:120, 8:120] = [205, 190, 180]
        rgb[38:58, 38:58] = [90, 55, 35]
        masks = np.zeros((128, 128), dtype=np.int32)
        masks[38:58, 38:58] = 1
        masks[0:6, 0:6] = 2
        summary, objects, labels = classify_cellpose_masks(
            masks,
            rgb,
            dab_threshold=0.02,
            pixel_width_um=0.3,
            pixel_height_um=0.3,
        )
        self.assertEqual(summary["raw_mask_count"], 2)
        self.assertEqual(len(objects), 1)
        self.assertEqual(int(labels.max()), 1)
        self.assertTrue(objects[0]["neun_positive"])


if __name__ == "__main__":
    unittest.main()
