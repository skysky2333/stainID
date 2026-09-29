import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from stainid.imaging.tissue import linear_artifact_mask
from stainid.stains.neun.candidates import segment_neun_candidates, write_candidate_results


class NeuNCandidatesTest(unittest.TestCase):
    def test_separates_and_classifies_neuronal_profile_candidates(self):
        image = np.full((256, 256, 3), (225, 220, 225), dtype=np.uint8)
        for center in ((55, 60), (100, 100), (170, 70)):
            cv2.circle(image, center, 9, (105, 65, 35), -1)
        for center in ((70, 180), (140, 160), (195, 195)):
            cv2.circle(image, center, 8, (70, 80, 170), -1)

        summary, objects, labels = segment_neun_candidates(
            image, 0.08, 0.274, 0.274
        )

        self.assertEqual(labels.shape, image.shape[:2])
        self.assertGreaterEqual(summary["candidate_profile_count"], 5)
        self.assertGreaterEqual(summary["neun_positive_profile_count"], 2)
        self.assertTrue(any(not row["neun_positive"] for row in objects))

        with tempfile.TemporaryDirectory() as directory:
            outputs = write_candidate_results(
                image,
                labels,
                objects,
                summary,
                Path(directory) / "field_candidates",
                "image_id",
                "M001_NeuN",
                "1",
                0.08,
            )
            self.assertTrue(all(path.is_file() for path in outputs))
            with outputs[-1].open(encoding="utf-8") as handle:
                geojson = json.load(handle)
            self.assertEqual(geojson["type"], "FeatureCollection")
            self.assertEqual(len(geojson["features"]), len(objects))
            self.assertTrue(
                all(
                    feature["properties"]["metadata"]["proposal_source"]
                    == "neun_candidates"
                    for feature in geojson["features"]
                )
            )

    def test_masks_large_linear_artifact(self):
        image = np.full((512, 512, 3), (220, 215, 220), dtype=np.uint8)
        cv2.line(image, (120, 0), (511, 391), (70, 70, 85), 24)
        cv2.circle(image, (90, 300), 10, (105, 65, 35), -1)

        artifact = linear_artifact_mask(image)

        self.assertTrue(artifact[200, 320])
        self.assertFalse(artifact[300, 90])

    def test_masks_long_thin_off_axis_artifact(self):
        image = np.full((512, 512, 3), (220, 215, 220), dtype=np.uint8)
        cv2.line(image, (15, 480), (430, 40), (95, 90, 105), 8)
        cv2.circle(image, (450, 450), 10, (105, 65, 35), -1)

        artifact = linear_artifact_mask(image)

        self.assertTrue(artifact[250, 232])
        self.assertFalse(artifact[450, 450])

    def test_does_not_mask_long_brown_dab_process(self):
        image = np.full((512, 512, 3), (220, 215, 220), dtype=np.uint8)
        cv2.line(image, (15, 480), (430, 40), (105, 65, 35), 12)

        artifact = linear_artifact_mask(image)

        self.assertFalse(artifact.any())


if __name__ == "__main__":
    unittest.main()
