import unittest

import numpy as np

from stainid.stains.amyloid.classifier import MORPHOLOGY_FEATURES, context_features
from stainid.stains.amyloid.model import classify_amyloid_objects, morphotype_threshold


class FixedClassifier:
    def __init__(self, probabilities):
        self.probabilities = np.asarray(probabilities)

    def predict_proba(self, matrix):
        return np.column_stack((1 - self.probabilities, self.probabilities))[: len(matrix)]


class AmyloidFrozenTest(unittest.TestCase):
    def test_assigns_identity_size_and_morphotype(self):
        rgb = np.full((400, 400, 3), 200, dtype=np.uint8)
        context_names = list(context_features(np.full((180, 180, 3), 200, dtype=np.uint8), 20.0))
        base = {name: 1.0 for name in MORPHOLOGY_FEATURES}
        objects = [
            {**base, "candidate_class": "compact", "centroid_x_px": 100, "centroid_y_px": 100, "equivalent_diameter_um": 30.0, "inner_mean_dab_od": 0.2},
            {**base, "candidate_class": "diffuse", "centroid_x_px": 200, "centroid_y_px": 200, "equivalent_diameter_um": 30.0, "inner_mean_dab_od": 0.05},
            {**base, "candidate_class": "review", "centroid_x_px": 300, "centroid_y_px": 300, "equivalent_diameter_um": 12.0, "inner_mean_dab_od": 0.2},
            {**base, "candidate_class": "review", "centroid_x_px": 150, "centroid_y_px": 250, "equivalent_diameter_um": 8.0, "inner_mean_dab_od": 0.2},
            {**base, "candidate_class": "diffuse", "centroid_x_px": 50, "centroid_y_px": 300, "equivalent_diameter_um": 30.0, "inner_mean_dab_od": 0.2},
            {**base, "candidate_class": "artifact", "centroid_x_px": 300, "centroid_y_px": 50, "equivalent_diameter_um": 30.0, "inner_mean_dab_od": 0.2},
        ]
        bundle = {
            "identity_classifier": FixedClassifier([0.9, 0.9, 0.9, 0.9, 0.1]),
            "identity_feature_names": list(MORPHOLOGY_FEATURES) + context_names,
            "identity_threshold": 0.5,
            "morphotype_threshold": 1.5,
            "morphotype_minimum_diameter_um": 15.0,
        }
        classes = [row["candidate_class"] for row in classify_amyloid_objects(rgb, objects, 0.1, 0.27, bundle)]
        self.assertEqual(classes, ["compact", "diffuse", "small_plaque", "speck", "rejected", "rejected"])

    def test_morphotype_threshold_separates_classes(self):
        values = np.asarray([0.5, 1.0, 1.2, 2.0, 2.5])
        compact = np.asarray([False, False, False, True, True])
        self.assertEqual(morphotype_threshold(values, compact), 2.0)


if __name__ == "__main__":
    unittest.main()
