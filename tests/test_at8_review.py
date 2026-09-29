import unittest

import numpy as np

from stainid.stains.tau.review import select_at8_review, target_overlay


class At8ReviewTest(unittest.TestCase):
    def test_selects_each_candidate_group_deterministically(self):
        rows = [
            {
                "candidate_id": f"C{index}",
                "source_image_id": "M001_AT8",
                "source_kind": source_kind,
                "candidate_class": candidate_class,
                "area_um2": index + 1,
            }
            for source_kind, candidate_class in (
                ("compact_profile", "soma_review"),
                ("thread_cluster", "thread_cluster"),
            )
            for index in range(5)
        ]
        selected = select_at8_review(rows, 2)
        self.assertEqual(len(selected), 4)
        self.assertEqual(selected, select_at8_review(rows, 2))
        self.assertEqual(
            {row["review_id"] for row in selected},
            {"TR001", "TR002", "TR003", "TR004"},
        )

    def test_target_overlay_uses_only_selected_label(self):
        image = np.full((64, 64, 3), 220, dtype=np.uint8)
        compact = np.zeros((64, 64), dtype=np.int32)
        compact[10:20, 10:20] = 1
        compact[40:50, 40:50] = 2
        masks = {
            "compact_labels": compact,
            "cluster_labels": np.zeros_like(compact),
            "thread": np.zeros_like(compact, dtype=bool),
        }
        overlay = target_overlay(
            image,
            masks,
            {"source_kind": "compact_profile", "label": 1},
        )
        self.assertFalse(np.array_equal(overlay[10:20, 10:20], image[10:20, 10:20]))
        self.assertTrue(np.array_equal(overlay[40:50, 40:50], image[40:50, 40:50]))


if __name__ == "__main__":
    unittest.main()
