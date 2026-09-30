import csv
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from stainid.slides.dearray import Component, GridFit, core_records, fit_grid, refine_core_centers, write_table
from stainid.slides.vsi import SceneInfo, SlidePreview


class DearrayTest(unittest.TestCase):
    def test_local_component_recenters_displaced_core(self):
        origin = np.array([100.0, 100.0])
        column_vector = np.array([120.0, 0.0])
        row_vector = np.array([0.0, 120.0])
        components = []
        for row in range(5):
            for column in range(6):
                center = origin + column * column_vector + row * row_vector
                if (row, column) == (1, 2):
                    center = center + np.array([18.0, -12.0])
                components.append(Component(center[0], center[1], 90, 90, 6_000))
        fit = GridFit(
            origin=origin,
            column_vector=column_vector,
            row_vector=row_vector,
            radius=49.2,
            components=tuple(components),
            inlier_components=tuple(components),
            rmse=0.0,
            strict_threshold=1.0,
        )

        refinement = refine_core_centers(fit, 5, 6)[1, 2]

        np.testing.assert_allclose(refinement.center, [358.0, 208.0])
        self.assertEqual(refinement.source, "component")
        self.assertAlmostEqual(refinement.offset, np.hypot(18.0, 12.0))

    def test_local_component_accepts_audited_moderate_displacement(self):
        origin = np.array([100.0, 100.0])
        components = []
        for row in range(5):
            for column in range(6):
                center = origin + np.array([120.0 * column, 120.0 * row])
                if (row, column) == (1, 2):
                    center = center + np.array([36.0, 0.0])
                components.append(Component(center[0], center[1], 90, 90, 6_000))
        fit = GridFit(
            origin=origin,
            column_vector=np.array([120.0, 0.0]),
            row_vector=np.array([0.0, 120.0]),
            radius=49.2,
            components=tuple(components),
            inlier_components=tuple(components),
            rmse=0.0,
            strict_threshold=1.0,
        )

        refinement = refine_core_centers(fit, 5, 6)[1, 2]

        np.testing.assert_allclose(refinement.center, [376.0, 220.0])
        self.assertEqual(refinement.source, "component")

    def test_write_table_preserves_existing_downstream_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["slide", "core_label", "center_x_px", "donor_id"],
                )
                writer.writeheader()
                writer.writerows(
                    [
                        {"slide": "slide-a", "core_label": "A-1", "center_x_px": "1", "donor_id": "10"},
                        {"slide": "slide-b", "core_label": "A-1", "center_x_px": "2", "donor_id": "20"},
                    ]
                )

            write_table(
                [{"slide": "slide-a", "core_label": "A-1", "center_x_px": 3}],
                path,
            )

            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0], {"slide": "slide-a", "core_label": "A-1", "center_x_px": "3", "donor_id": "10"})
            self.assertEqual(rows[1], {"slide": "slide-b", "core_label": "A-1", "center_x_px": "2", "donor_id": "20"})

    def test_lattice_recovers_missing_and_malformed_cores(self):
        mask = np.zeros((780, 900), dtype=np.uint8)
        origin = np.array([125.0, 125.0])
        column_vector = np.array([126.0, 3.0])
        row_vector = np.array([-2.0, 122.0])
        expected = {}

        for row in range(5):
            for column in range(6):
                center = origin + column * column_vector + row * row_vector
                expected[row, column] = center
                if (row, column) in {(0, 0), (3, 4)}:
                    continue
                cv2.circle(mask, tuple(np.rint(center).astype(int)), 48, 1, -1)

        malformed_center = tuple(np.rint(expected[1, 2]).astype(int))
        cv2.rectangle(
            mask,
            (malformed_center[0] - 55, malformed_center[1] - 55),
            (malformed_center[0] + 10, malformed_center[1] + 55),
            0,
            -1,
        )

        fit = fit_grid(mask.astype(bool), rows=5, columns=6, strict_threshold=1.0)

        errors = [
            np.linalg.norm(fit.center(row, column) - expected[row, column])
            for row in range(5)
            for column in range(6)
        ]
        self.assertLess(max(errors), 3.0)
        self.assertGreaterEqual(len(fit.inlier_components), 24)
        self.assertAlmostEqual(fit.radius, 0.41 * 122.0, delta=2.0)

        preview = SlidePreview(
            path=Path("TMA LIP-1 NeuN.vsi"),
            rgb=np.repeat((255 - mask * 80)[..., None], 3, axis=2).astype(np.uint8),
            full_scene=SceneInfo(0, "full", mask.shape[1], mask.shape[0]),
            preview_scene=SceneInfo(0, "preview", mask.shape[1], mask.shape[0]),
            scale_x=1.0,
            scale_y=1.0,
            pixel_width_um=1.0,
            pixel_height_um=1.0,
        )
        records = core_records(preview, fit, 5, 6, "1", "NeuN")
        labels = [record["core_label"] for record in records]
        self.assertEqual(labels[:6], ["A-1", "B-1", "C-1", "D-1", "E-1", "F-1"])
        self.assertEqual(labels[-1], "F-5")
        self.assertAlmostEqual(
            records[0]["crop_diameter_um"],
            1.12 * records[0]["estimated_core_diameter_um"],
        )


if __name__ == "__main__":
    unittest.main()
