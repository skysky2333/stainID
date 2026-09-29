import unittest

import cv2
import numpy as np

from stainid.sampling.cohort import select_systematic_fields, spatially_spread_indices


class CohortSamplingTest(unittest.TestCase):
    def test_farthest_point_order_is_deterministic_and_spread(self):
        candidates = [
            {"center_x_fraction": x, "center_y_fraction": y}
            for y in (0.1, 0.5, 0.9)
            for x in (0.1, 0.5, 0.9)
        ]
        selected = spatially_spread_indices(candidates, 5)
        self.assertEqual(selected[0], 4)
        points = {
            (
                candidates[index]["center_x_fraction"],
                candidates[index]["center_y_fraction"],
            )
            for index in selected
        }
        self.assertIn((0.5, 0.5), points)
        self.assertGreaterEqual(len(points & {(0.1, 0.1), (0.1, 0.9), (0.9, 0.1), (0.9, 0.9)}), 3)

    def test_selects_nested_native_resolution_fields(self):
        image = np.full((512, 512, 3), 255, dtype=np.uint8)
        cv2.circle(image, (256, 256), 245, (220, 215, 220), -1)
        image[::20, ::20] = (80, 65, 100)
        fields, summary = select_systematic_fields(
            image, 4096, 4096, field_size_px=1024, fields_per_core=8
        )
        self.assertEqual(len(fields), 8)
        self.assertEqual(sum(row["nested_sample"] == "primary_four" for row in fields), 4)
        self.assertTrue(all(row["width_px"] == 1024 for row in fields))
        self.assertGreaterEqual(summary["eligible_candidate_count"], 8)
        for first_index, first in enumerate(fields):
            for second in fields[first_index + 1 :]:
                separated = (
                    first["x_px"] + first["width_px"] <= second["x_px"]
                    or second["x_px"] + second["width_px"] <= first["x_px"]
                    or first["y_px"] + first["height_px"] <= second["y_px"]
                    or second["y_px"] + second["height_px"] <= first["y_px"]
                )
                self.assertTrue(separated)


if __name__ == "__main__":
    unittest.main()
