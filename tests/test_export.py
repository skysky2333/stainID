import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from stainid.slides.export import core_bounds, export_matches, png_matches, read_core


class CoreMeta:
    shape = (1, 1, 1, 4, 5, 3)
    dtype = np.dtype("uint8")


class FakeBioFile:
    core_meta = CoreMeta()

    def __init__(self):
        self.image = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)

    def _get_plane(self, y, x):
        return self.image[y, x]


class ExportTest(unittest.TestCase):
    def test_bounds_preserve_native_pixel_dimensions(self):
        row = {
            "slide": "slide",
            "core_label": "A-1",
            "center_x_px": "10.5",
            "center_y_px": "20.5",
            "diameter_x_px": "7.9",
            "diameter_y_px": "9.1",
        }
        self.assertEqual(core_bounds(row), (6, 16, 8, 9))

    def test_edge_core_is_white_padded_without_resampling(self):
        source = FakeBioFile()
        core, bounds, padding = read_core(source, -2, 1, 5, 3)
        self.assertEqual(core.shape, (3, 5, 3))
        self.assertEqual(bounds, (0, 1, 3, 3))
        self.assertEqual(padding, (2, 0, 0, 0))
        np.testing.assert_array_equal(core[:, :2], 255)
        np.testing.assert_array_equal(core[:, 2:], source.image[1:4, 0:3])

    def test_dimension_change_marks_existing_png_for_reexport(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "core.png"
            Image.fromarray(np.zeros((20, 30, 3), dtype=np.uint8)).save(path)

            self.assertTrue(png_matches(path, (30, 20)))
            self.assertFalse(png_matches(path, (32, 20)))

    def test_source_offset_change_marks_existing_png_for_reexport(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "core.png"
            Image.fromarray(np.zeros((20, 30, 3), dtype=np.uint8)).save(path)
            row = {
                "source_x_px": "10",
                "source_y_px": "20",
                "source_width_px": "30",
                "source_height_px": "20",
                "padding_left_px": "0",
                "padding_top_px": "0",
                "padding_right_px": "0",
                "padding_bottom_px": "0",
            }

            self.assertTrue(
                export_matches(row, path, (30, 20), (10, 20, 30, 20), (0, 0, 0, 0))
            )
            self.assertFalse(
                export_matches(row, path, (30, 20), (11, 20, 30, 20), (0, 0, 0, 0))
            )


if __name__ == "__main__":
    unittest.main()
