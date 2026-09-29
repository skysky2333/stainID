import unittest

import cv2
import numpy as np

from stainid.stains.tau.network import segment_at8_network


class At8NetworkTest(unittest.TestCase):
    def test_separates_compact_profiles_and_thin_network(self):
        image = np.full((512, 512, 3), (225, 220, 225), dtype=np.uint8)
        cv2.circle(image, (150, 180), 24, (95, 60, 35), -1)
        cv2.circle(image, (350, 320), 20, (110, 70, 40), -1)
        cv2.line(image, (40, 80), (470, 420), (135, 90, 55), 4)
        cv2.line(image, (70, 450), (440, 110), (135, 90, 55), 4)
        summary, objects, maps = segment_at8_network(
            image, 0.04, 0.274, 0.274
        )
        self.assertGreater(summary["at8_positive_area_fraction"], 0)
        self.assertGreater(summary["compact_profile_count"], 0)
        self.assertGreater(summary["thread_skeleton_length_mm_per_mm2"], 0)
        self.assertTrue(any(row["source_kind"] == "compact_profile" for row in objects))
        self.assertGreater(int(maps["thread_skeleton"].sum()), 0)

    def test_excludes_manual_artifact_region(self):
        image = np.full((256, 256, 3), (225, 220, 225), dtype=np.uint8)
        cv2.line(image, (10, 128), (245, 128), (70, 45, 35), 20)
        exclusion = np.zeros((256, 256), dtype=bool)
        exclusion[100:155] = True
        summary, _, maps = segment_at8_network(
            image, 0.04, 0.274, 0.274, exclusion
        )
        self.assertGreater(summary["manual_exclusion_fraction"], 0)
        self.assertFalse(np.any(maps["positive"] & exclusion))

    def test_excludes_stained_external_cut_edge(self):
        image = np.full((256, 256, 3), 255, dtype=np.uint8)
        image[:, :210] = (225, 220, 225)
        image[:, 195:210] = (95, 60, 35)
        cv2.line(image, (40, 40), (170, 170), (135, 90, 55), 3)
        summary, _, maps = segment_at8_network(
            image, 0.04, 0.5, 0.5, edge_guard_um=15.0
        )
        self.assertTrue(summary["edge_guard_triggered"])
        self.assertGreater(summary["edge_exclusion_fraction"], 0)
        self.assertTrue(np.any(maps["edge_excluded_display"]))
        self.assertFalse(np.any(maps["positive"][:, 195:210]))
        self.assertTrue(np.any(maps["positive"][30:180, 30:180]))


if __name__ == "__main__":
    unittest.main()


class HoleRimTest(unittest.TestCase):
    def test_removes_vacuole_rim_but_keeps_parenchymal_thread(self):
        import cv2

        rgb = np.full((600, 600, 3), (205, 195, 215), dtype=np.uint8)
        cv2.circle(rgb, (150, 150), 60, (255, 255, 255), -1)
        cv2.circle(rgb, (150, 150), 62, (130, 80, 40), 3)
        cv2.line(rgb, (350, 350), (550, 520), (130, 80, 40), 3)
        summary, _, maps = segment_at8_network(rgb, 0.05, 0.27, 0.27)
        self.assertGreater(summary["hole_rim_removed_area_fraction"], 0)
        self.assertFalse(maps["positive"][85:215, 85:215].any())
        self.assertTrue(maps["positive"][350:520, 350:550].any())
