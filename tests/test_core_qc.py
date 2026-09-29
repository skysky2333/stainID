import csv
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from stainid.qc.core_qc import analyze_manifest, classify_coverage, measure_tissue
from stainid.qc.focus import low_focus_mask, measure_focus


def synthetic_core(fraction: float, size: int = 1024) -> np.ndarray:
    image = np.full((size, size, 3), 248, dtype=np.uint8)
    center = (size // 2, size // 2)
    radius = round(size * 0.47)
    cv2.circle(image, center, radius, (185, 175, 205), -1)
    if fraction < 1.0:
        cutoff = round(size * fraction)
        image[:, cutoff:] = 248
    return image


class TissueQcTest(unittest.TestCase):
    def test_coverage_categories(self):
        self.assertEqual(classify_coverage(0.0), "empty")
        self.assertEqual(classify_coverage(0.1), "sparse")
        self.assertEqual(classify_coverage(0.4), "partial")
        self.assertEqual(classify_coverage(0.7), "partial")
        self.assertEqual(classify_coverage(0.8), "substantial")

    def test_full_core_has_substantial_coverage(self):
        qc, _, _ = measure_tissue(synthetic_core(1.0))
        self.assertEqual(qc.status, "substantial")
        self.assertGreater(qc.tissue_fraction, 0.8)

    def test_fragment_is_retained(self):
        qc, mask, _ = measure_tissue(synthetic_core(0.35))
        self.assertEqual(qc.status, "partial")
        self.assertGreater(mask.sum(), 0)
        self.assertGreater(qc.tissue_fraction, 0.2)

    def test_stain_optical_density_metrics(self):
        brown = np.full((1024, 1024, 3), 248, dtype=np.uint8)
        cv2.circle(brown, (512, 512), 480, (95, 55, 25), -1)
        qc, _, _ = measure_tissue(brown)
        self.assertGreater(qc.dab_quantiles[1], qc.hematoxylin_quantiles[1])
        self.assertGreater(qc.background_brightness, 0.9)

    def test_local_blur_is_flagged(self):
        rng = np.random.default_rng(7)
        image = np.clip(rng.normal(170, 35, (512, 512, 3)), 0, 255).astype(np.uint8)
        image[256:, :256] = cv2.GaussianBlur(image[256:, :256], (31, 31), 0)
        tissue = np.ones(image.shape[:2], dtype=bool)

        qc, tiles = measure_focus(image, tissue, tile_size=128)

        flagged = {(tile.x, tile.y) for tile in tiles if tile.low_focus}
        mask = low_focus_mask(tissue.shape, tiles)
        self.assertGreaterEqual(qc.low_focus_tissue_fraction, 0.20)
        self.assertIn((0, 256), flagged)
        self.assertNotIn((256, 0), flagged)
        self.assertTrue(mask[300, 50])
        self.assertFalse(mask[50, 300])

    def test_manifest_checkpoints_after_each_slide(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "core.png"
            cv2.imwrite(str(image_path), cv2.cvtColor(synthetic_core(1.0), cv2.COLOR_RGB2BGR))
            manifest_path = root / "manifest.csv"
            with manifest_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["slide", "output_path"])
                writer.writeheader()
                writer.writerows(
                    [
                        {"slide": "slide-1", "output_path": image_path.name},
                        {"slide": "slide-2", "output_path": "missing.png"},
                    ]
                )

            with self.assertRaises(FileNotFoundError):
                analyze_manifest(manifest_path, write_overlays=False, workers=2)

            with manifest_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertTrue(rows[0]["tissue_status"])
            self.assertFalse(rows[1]["tissue_status"])


if __name__ == "__main__":
    unittest.main()
