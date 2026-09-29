import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from stainid.slides.field_export import build_export_rows, export_field


class FieldExportTest(unittest.TestCase):
    def test_native_crop_matches_source_pixels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.png"
            output_dir = root / "output"
            rgb = np.arange(20 * 30 * 3, dtype=np.uint8).reshape(20, 30, 3)
            Image.fromarray(rgb).save(source)
            row = {
                "image_id": "M001_6E10",
                "image_path": str(source),
                "x": 7,
                "y": 4,
                "width": 11,
                "height": 9,
            }

            output_dir.mkdir()
            output = export_field(row, output_dir, overwrite=False)

            with Image.open(output) as image:
                observed = np.asarray(image)
            np.testing.assert_array_equal(observed, rgb[4:13, 7:18])

    def test_manifest_sets_must_match(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            annotations = root / "data" / "annotations" / "annotation.csv"
            fields = root / "data" / "annotations" / "fields.csv"
            annotations.parent.mkdir(parents=True)
            with annotations.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "image_id",
                        "image_path",
                        "native_width_px",
                        "native_height_px",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "image_id": "M001_6E10",
                        "image_path": "missing.png",
                        "native_width_px": 100,
                        "native_height_px": 100,
                    }
                )
            with fields.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["image_id", "x_px", "y_px", "width_px", "height_px"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "image_id": "M002_6E10",
                        "x_px": 0,
                        "y_px": 0,
                        "width_px": 10,
                        "height_px": 10,
                    }
                )

            with self.assertRaisesRegex(ValueError, "image sets disagree"):
                build_export_rows(annotations, fields)


if __name__ == "__main__":
    unittest.main()
