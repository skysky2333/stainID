import unittest

from stainid.qc.exclusions import rasterize_core_exclusions


class CohortExclusionsTest(unittest.TestCase):
    def test_rasterizes_global_polygon_in_tile_coordinates(self):
        exclusions = [
            {
                "polygon_core_px": [[110, 210], [130, 210], [130, 230], [110, 230]]
            }
        ]
        mask = rasterize_core_exclusions(exclusions, 100, 200, (50, 50))
        self.assertTrue(mask[20, 20])
        self.assertFalse(mask[5, 5])


if __name__ == "__main__":
    unittest.main()
