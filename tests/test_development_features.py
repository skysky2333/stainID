import csv
import tempfile
import unittest
from pathlib import Path

from stainid.analysis.development_features import ReviewSource, assemble_development_features


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class DevelopmentFeaturesTest(unittest.TestCase):
    def test_joins_review_measurements_to_blinded_field_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_rows(
                root / "annotations.csv",
                [{
                    "annotation_id": "M001",
                    "image_id": "M001_AT8",
                    "stain": "AT8",
                    "pixel_width_um": "0.27",
                    "pixel_height_um": "0.27",
                    "tissue_status": "substantial",
                    "tissue_fraction": "0.8",
                }],
            )
            write_rows(
                root / "fields.csv",
                [{
                    "image_id": "M001_AT8",
                    "field_id": "M001_AT8_F01",
                    "stain": "AT8",
                    "target_dab_quantile": "0.85",
                    "x_px": "10",
                    "y_px": "20",
                    "width_px": "2048",
                    "height_px": "2048",
                }],
            )
            write_rows(
                root / "review.csv",
                [{"image_id": "M001_AT8", "tma": "1", "selection_role": "high"}],
            )
            proposals = root / "proposals"
            write_rows(
                proposals / "M001_AT8_summary.csv",
                [{"image_id": "M001_AT8", "at8_positive_area_fraction": "0.1"}],
            )
            write_rows(
                proposals / "M001_AT8_objects.csv",
                [{
                    "image_id": "M001_AT8",
                    "candidate_id": "M001_AT8_AI_S00001",
                    "candidate_class": "soma_review",
                }],
            )
            source = ReviewSource(root / "review.csv", proposals, "at8_v1", "at8")
            features, objects = assemble_development_features(
                root / "annotations.csv", root / "fields.csv", [source]
            )
            self.assertEqual(features[0]["annotation_id"], "M001")
            self.assertEqual(features[0]["selection_role"], "high")
            self.assertEqual(features[0]["proposal_source"], "at8_v1")
            self.assertEqual(objects[0]["candidate_id"], "M001_AT8_AI_S00001")


if __name__ == "__main__":
    unittest.main()
