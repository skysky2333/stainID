import unittest

import cv2
import numpy as np

from stainid.stains.neun.audit import (
    REVIEW_OUTLINE_COLOR,
    positive_overlay,
    render_raw_field_panel,
    select_candidate_review,
    select_field_windows,
    target_overlay,
)


class NeuNAuditTest(unittest.TestCase):
    def test_selects_candidate_groups_deterministically(self):
        rows = [
            {
                "candidate_id": f"C{source}{group}{index}",
                "tma": "1",
                "source_kind": source,
                "candidate_class": group,
                "selection_area_um2": index + 1,
                "touches_edge": "false",
            }
            for source, group in (
                ("dab_profile", "positive_profile"),
                ("dab_profile", "review"),
                ("cellpose", "review"),
                ("cellpose", "negative"),
            )
            for index in range(4)
        ]
        selected = select_candidate_review(rows, 2)
        self.assertEqual(len(selected), 8)
        self.assertEqual(selected, select_candidate_review(rows, 2))
        self.assertTrue(all(row["sampling_weight"] == 2 for row in selected))

    def test_target_overlay_marks_only_selected_object(self):
        image = np.full((64, 64, 3), 220, dtype=np.uint8)
        profile = np.zeros((64, 64), dtype=np.int32)
        profile[10:20, 10:20] = 1
        profile[40:50, 40:50] = 2
        overlay = target_overlay(
            image,
            profile,
            np.zeros_like(profile),
            {
                "source_kind": "dab_profile",
                "source_label": 1,
                "candidate_class": "positive_profile",
            },
        )
        self.assertFalse(np.array_equal(overlay[10:20, 10:20], image[10:20, 10:20]))
        self.assertTrue(np.array_equal(overlay[40:50, 40:50], image[40:50, 40:50]))
        self.assertTrue(np.any(np.all(overlay == REVIEW_OUTLINE_COLOR, axis=2)))

    def test_selects_low_and_high_density_tissue_windows(self):
        image = np.full((512, 512, 3), (190, 175, 180), dtype=np.uint8)
        objects = [
            {
                "source_kind": "dab_profile",
                "candidate_class": "positive_profile",
                "touches_edge": "false",
                "centroid_x_px": 320 + index * 10,
                "centroid_y_px": 320,
                "source_label": index + 1,
            }
            for index in range(5)
        ]
        windows = select_field_windows(
            image,
            objects,
            np.zeros((512, 512), dtype=bool),
            1.0,
            256.0,
            2,
        )
        self.assertEqual([row["proposal_count"] for row in windows], [0, 5])

        labels = np.zeros((512, 512), dtype=np.int32)
        cv2.circle(labels, (320, 320), 12, 1, -1)
        overlay = positive_overlay(image, labels, objects)
        self.assertFalse(np.array_equal(overlay, image))
        panel = render_raw_field_panel(
            image,
            {"x_px": 0, "y_px": 0, "size_px": 256, "review_id": "NF001"},
            128,
        )
        self.assertEqual(panel.shape, (164, 128, 3))


if __name__ == "__main__":
    unittest.main()
