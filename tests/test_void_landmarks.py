import unittest

import cv2
import numpy as np

from stainid.registration.void_landmarks import detect_void_landmarks, match_void_landmarks


class VoidLandmarkTest(unittest.TestCase):
    def test_detects_enclosed_voids_and_matches_translation(self):
        reference = np.full((256, 256, 3), 250, dtype=np.uint8)
        cv2.circle(reference, (128, 128), 105, (190, 180, 205), -1)
        for center in ((85, 100), (145, 80), (155, 165), (90, 175)):
            cv2.circle(reference, center, 7, (250, 250, 250), -1)
        moving = cv2.warpAffine(
            reference,
            np.array([[1.0, 0.0, 5.0], [0.0, 1.0, -3.0]]),
            (256, 256),
            borderValue=(250, 250, 250),
        )

        reference_landmarks = detect_void_landmarks(reference, 1.0, 40.0)
        moving_landmarks = detect_void_landmarks(moving, 1.0, 40.0)
        matches = match_void_landmarks(reference_landmarks, moving_landmarks, 1.0, 20.0)

        self.assertEqual(len(reference_landmarks), 4)
        self.assertEqual(len(matches), 4)
        np.testing.assert_allclose([match.error_um for match in matches], np.hypot(5, 3), atol=0.2)


if __name__ == "__main__":
    unittest.main()
