import unittest

import numpy as np

from stainid.stains.tau.tangle_tracer import non_max_suppression, tile_origins


class TangleTracerTest(unittest.TestCase):
    def test_tile_origins_cover_edges_without_duplicate_last_tile(self):
        self.assertEqual(tile_origins(800, 1024, 128), [0])
        self.assertEqual(tile_origins(2048, 1024, 128), [0, 896, 1024])
        self.assertEqual(tile_origins(2816, 1024, 128), [0, 896, 1792])

    def test_non_max_suppression_retains_distinct_highest_boxes(self):
        boxes = np.asarray(
            [[0, 0, 100, 100], [5, 5, 105, 105], [200, 200, 250, 250]],
            dtype=np.float64,
        )
        scores = np.asarray([0.8, 0.9, 0.7], dtype=np.float64)
        self.assertEqual(non_max_suppression(boxes, scores, 0.3), [1, 2])


if __name__ == "__main__":
    unittest.main()
