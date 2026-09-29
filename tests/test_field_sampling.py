import unittest
from collections import Counter

import cv2
import numpy as np

from stainid.sampling.fields import select_field, target_quantile


class FieldSamplingTest(unittest.TestCase):
    def test_selects_local_dab_quantiles_inside_tissue(self):
        image = np.full((512, 512, 3), 250, dtype=np.uint8)
        cv2.rectangle(image, (40, 40), (471, 471), (220, 205, 220), -1)
        for x in range(40, 472):
            strength = (x - 40) / 431
            image[40:472, x, 0] = np.rint(220 - 120 * strength)
            image[40:472, x, 1] = np.rint(205 - 100 * strength)
        for y in range(48, 472, 16):
            for x in range(48, 472, 16):
                cv2.circle(image, (x, y), 2, (100, 100, 155), -1)

        low = select_field(image, 512, 512, 0.15, 128)
        high = select_field(image, 512, 512, 0.85, 128)

        self.assertEqual(low["selection_status"], "tissue_rich")
        self.assertGreaterEqual(low["preview_tissue_fraction"], 0.75)
        self.assertGreater(high["preview_mean_dab_od"], low["preview_mean_dab_od"])

    def test_target_quantiles_are_balanced_per_stain(self):
        for stain in ("6E10", "AT8", "NeuN"):
            counts = Counter(
                target_quantile(f"M{index:03d}_{stain}") for index in range(1, 43)
            )
            self.assertEqual(set(counts.values()), {14})


if __name__ == "__main__":
    unittest.main()
