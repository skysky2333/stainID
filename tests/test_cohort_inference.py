import unittest

import numpy as np
from PIL import Image

from stainid.pipelines.cohort_v1 import centered_objects, context_crop, local_profile_density


class CohortInferenceTest(unittest.TestCase):
    def test_context_crop_tracks_inner_sampling_window(self):
        image = Image.fromarray(np.zeros((200, 300, 3), dtype=np.uint8))
        rgb, inner = context_crop(image, 50, 60, 100, 80, 20)
        self.assertEqual(rgb.shape, (120, 140, 3))
        self.assertEqual(inner, (slice(20, 100), slice(20, 120)))

    def test_centroid_rule_assigns_each_object_to_central_window(self):
        objects = [
            {"centroid_x_px": 19.9, "centroid_y_px": 30.0},
            {"centroid_x_px": 20.0, "centroid_y_px": 30.0},
            {"centroid_x_px": 119.9, "centroid_y_px": 99.9},
            {"centroid_x_px": 120.0, "centroid_y_px": 30.0},
        ]
        selected = centered_objects(objects, (slice(20, 100), slice(20, 120)))
        self.assertEqual(selected, objects[1:3])

    def test_local_profile_density_uses_tissue_supported_windows(self):
        tissue = np.ones((200, 200), dtype=bool)
        objects = [
            {"centroid_x_px": 25.0, "centroid_y_px": 25.0},
            {"centroid_x_px": 30.0, "centroid_y_px": 30.0},
            {"centroid_x_px": 125.0, "centroid_y_px": 25.0},
            {"centroid_x_px": 25.0, "centroid_y_px": 125.0},
        ]
        result = local_profile_density(
            objects,
            tissue,
            (slice(0, 200), slice(0, 200)),
            1.0,
            1.0,
        )
        self.assertEqual(result["neun_local_density_window_count"], 4)
        self.assertGreater(result["neun_local_density_cv_100um"], 0)


if __name__ == "__main__":
    unittest.main()
