import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from stainid.slides.export_audit import audit_exports


class ExportAuditTest(unittest.TestCase):
    def test_validates_native_png_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "core.png"
            Image.fromarray(np.zeros((20, 30, 3), dtype=np.uint8)).save(image)
            manifest = root / "manifest.csv"
            row = {
                "slide": "slide",
                "core_label": "A-1",
                "center_x_px": "15",
                "center_y_px": "10",
                "diameter_x_px": "30",
                "diameter_y_px": "20",
                "output_path": "core.png",
                "export_status": "complete",
                "export_downsample": "1",
                "output_width_px": "30",
                "output_height_px": "20",
            }
            with manifest.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
                writer.writerow(row)

            self.assertEqual(audit_exports(manifest), 1)


if __name__ == "__main__":
    unittest.main()
