import csv
import tempfile
import unittest
from pathlib import Path

from stainid.qc.summary import write_review_queue


class QcSummaryTest(unittest.TestCase):
    def test_review_queue_contains_only_actionable_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.csv"
            output = root / "review.csv"
            rows = [
                {
                    "slide": "slide",
                    "tma": "1",
                    "stain": "6E10",
                    "core_label": "A-1",
                    "donor_id": "1",
                    "region": "frontal",
                    "tissue_status": "partial",
                    "tissue_fraction": "0.5",
                    "center_source": "lattice",
                    "center_offset_preview_px": "0",
                    "low_focus_tissue_fraction": "0",
                    "focus_review_required": "false",
                    "output_width_px": "100",
                    "output_height_px": "100",
                    "padding_left_px": "0",
                    "padding_top_px": "0",
                    "padding_right_px": "0",
                    "padding_bottom_px": "0",
                },
                {
                    "slide": "slide",
                    "tma": "1",
                    "stain": "AT8",
                    "core_label": "A-1",
                    "donor_id": "1",
                    "region": "frontal",
                    "tissue_status": "substantial",
                    "tissue_fraction": "0.9",
                    "center_source": "component",
                    "center_offset_preview_px": "4",
                    "low_focus_tissue_fraction": "0",
                    "focus_review_required": "false",
                    "output_width_px": "100",
                    "output_height_px": "100",
                    "padding_left_px": "0",
                    "padding_top_px": "0",
                    "padding_right_px": "0",
                    "padding_bottom_px": "0",
                },
                {
                    "slide": "slide",
                    "tma": "1",
                    "stain": "NeuN",
                    "core_label": "A-1",
                    "donor_id": "1",
                    "region": "frontal",
                    "tissue_status": "sparse",
                    "tissue_fraction": "0.1",
                    "center_source": "component",
                    "center_offset_preview_px": "5",
                    "low_focus_tissue_fraction": "0.2",
                    "focus_review_required": "true",
                    "output_width_px": "100",
                    "output_height_px": "100",
                    "padding_left_px": "11",
                    "padding_top_px": "0",
                    "padding_right_px": "0",
                    "padding_bottom_px": "0",
                },
            ]
            with manifest.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

            write_review_queue(manifest, output)

            with output.open(newline="", encoding="utf-8") as handle:
                observed = list(csv.DictReader(handle))
            self.assertEqual(len(observed), 2)
            self.assertEqual(observed[0]["review_reasons"], "unresolved_local_center")
            self.assertEqual(
                observed[1]["review_reasons"],
                "sparse_tissue;low_focus;slide_edge_padding",
            )
            self.assertEqual(observed[1]["slide_edge_padding_fraction"], "0.110000")

    def test_review_queue_ignores_minor_slide_edge_padding(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.csv"
            output = root / "review.csv"
            row = {
                "slide": "slide",
                "tma": "1",
                "stain": "6E10",
                "core_label": "A-1",
                "donor_id": "1",
                "region": "frontal",
                "tissue_status": "partial",
                "tissue_fraction": "0.5",
                "center_source": "component",
                "center_offset_preview_px": "2",
                "low_focus_tissue_fraction": "0",
                "focus_review_required": "false",
                "output_width_px": "100",
                "output_height_px": "100",
                "padding_left_px": "9",
                "padding_top_px": "0",
                "padding_right_px": "0",
                "padding_bottom_px": "0",
            }
            with manifest.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
                writer.writerow(row)

            write_review_queue(manifest, output)

            with output.open(newline="", encoding="utf-8") as handle:
                self.assertEqual(list(csv.DictReader(handle)), [])


if __name__ == "__main__":
    unittest.main()
