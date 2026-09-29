import unittest
from pathlib import Path

import numpy as np

from stainid.slides.core_sheets import crop_preview, render_core_sheet
from stainid.slides.vsi import SceneInfo, SlidePreview


def preview() -> SlidePreview:
    rgb = np.full((500, 600, 3), 255, dtype=np.uint8)
    rgb[20:480, 20:580] = [180, 150, 130]
    scene = SceneInfo(0, "scene", 600, 500)
    return SlidePreview(Path("slide.vsi"), rgb, scene, scene, 2.0, 2.0, 0.27, 0.27)


class CoreReviewTest(unittest.TestCase):
    def test_crop_preview_pads_outside_image(self):
        row = {
            "center_x_px": "20",
            "center_y_px": "20",
            "diameter_x_px": "80",
            "diameter_y_px": "80",
        }

        crop = crop_preview(preview(), row)

        self.assertEqual(crop.shape, (40, 40, 3))
        np.testing.assert_array_equal(crop[0, 0], [255, 255, 255])
        np.testing.assert_array_equal(crop[-1, -1], [180, 150, 130])

    def test_sheet_places_all_positions(self):
        rows = []
        for row in range(1, 6):
            for column in range(1, 7):
                rows.append(
                    {
                        "row": str(row),
                        "column": str(column),
                        "core_label": f"{chr(64 + column)}-{row}",
                        "center_x_px": str(column * 150),
                        "center_y_px": str(row * 150),
                        "diameter_x_px": "120",
                        "diameter_y_px": "120",
                        "provisional_tissue_status": "substantial",
                        "center_source": "component",
                        "center_offset_preview_px": "4",
                    }
                )

        sheet = render_core_sheet(preview(), rows, tile_size=40, label_height=16)

        self.assertEqual(sheet.shape, (326, 240, 3))


if __name__ == "__main__":
    unittest.main()
