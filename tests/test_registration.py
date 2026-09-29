import csv
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from stainid.registration.rigid import native_matrix, register_previews, register_triplet_manifest


def synthetic_tissue(size=768):
    image = np.full((size, size, 3), 248, dtype=np.uint8)
    cv2.circle(image, (size // 2, size // 2), 310, (215, 205, 225), -1)
    generator = np.random.default_rng(19)
    for _ in range(180):
        center = tuple(generator.integers(100, size - 100, size=2))
        radius = int(generator.integers(3, 15))
        color = (95, 65, 45) if generator.random() > 0.5 else (65, 70, 140)
        cv2.circle(image, center, radius, color, -1)
    cv2.line(image, (180, 120), (580, 620), (120, 100, 150), 8)
    return image


class RegistrationTest(unittest.TestCase):
    def test_affine_registration_improves_tissue_overlap(self):
        reference = synthetic_tissue()
        transform = cv2.getRotationMatrix2D((384, 384), 3.0, 1.0)
        transform[:, 2] += (28, -21)
        moving = cv2.warpAffine(
            reference,
            transform,
            (768, 768),
            borderValue=(255, 255, 255),
        )
        result = register_previews(reference, moving)
        self.assertEqual(result.method, "mask_structure_rigid")
        self.assertGreater(result.structural_correlation, 0.8)
        np.testing.assert_allclose(
            result.matrix,
            cv2.invertAffineTransform(transform),
            atol=0.25,
        )
        self.assertGreaterEqual(result.final_dice, result.initial_dice - 0.01)

    def test_native_matrix_accounts_for_image_scaling(self):
        preview = np.array([[1.0, 0.0, 10.0], [0.0, 1.0, -5.0]])
        native = native_matrix(preview, (100, 200), (800, 1600), (400, 800))
        np.testing.assert_allclose(
            native,
            np.array([[2.0, 0.0, 80.0], [0.0, 2.0, -40.0]]),
        )

    def test_registration_checkpoint_resumes_complete_triplet(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = synthetic_tissue()
            rows = []
            for stain in ("6E10", "AT8", "NeuN"):
                path = root / f"{stain}.png"
                self.assertTrue(cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)))
                rows.append(
                    {
                        "core_id": "core-1",
                        "stain": stain,
                        "image_path": str(path),
                        "native_width_px": image.shape[1],
                        "native_height_px": image.shape[0],
                    }
                )
            manifest = root / "manifest.csv"
            with manifest.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

            output = root / "registration.csv"
            register_triplet_manifest(manifest, output, "core_id")
            checkpoint = output.with_suffix(".tmp.csv")
            output.replace(checkpoint)
            register_triplet_manifest(manifest, output, "core_id")

            self.assertTrue(output.exists())
            self.assertFalse(checkpoint.exists())
            with output.open(newline="", encoding="utf-8") as handle:
                records = list(csv.DictReader(handle))
            self.assertEqual(len(records), 2)
            self.assertEqual({record["moving_stain"] for record in records}, {"6E10", "AT8"})

    def test_targeted_registration_preserves_other_triplets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = synthetic_tissue()
            rows = []
            for core_id in ("core-1", "core-2"):
                for stain in ("6E10", "AT8", "NeuN"):
                    path = root / f"{core_id}_{stain}.png"
                    self.assertTrue(
                        cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                    )
                    rows.append(
                        {
                            "core_id": core_id,
                            "stain": stain,
                            "image_path": str(path),
                            "native_width_px": image.shape[1],
                            "native_height_px": image.shape[0],
                        }
                    )
            manifest = root / "manifest.csv"
            with manifest.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            output = root / "registration.csv"
            register_triplet_manifest(manifest, output, "core_id")
            with output.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                records = list(reader)
                fields = list(reader.fieldnames or [])
            for record in records:
                if record["core_id"] == "core-2":
                    record["method"] = "preserved"
            with output.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(records)

            register_triplet_manifest(
                manifest,
                output,
                "core_id",
                selected_identifiers={"core-1"},
            )

            with output.open(newline="", encoding="utf-8") as handle:
                updated = list(csv.DictReader(handle))
            self.assertEqual(len(updated), 4)
            self.assertEqual(
                {record["method"] for record in updated if record["core_id"] == "core-2"},
                {"preserved"},
            )


if __name__ == "__main__":
    unittest.main()
