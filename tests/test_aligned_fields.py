import unittest

import numpy as np

from stainid.registration.aligned_fields import AlignedField, local_matrix, preview_matrix, source_bounds


class AlignedFieldsTest(unittest.TestCase):
    def test_preview_matrix_preserves_native_mapping(self):
        native = np.array([[1.0, 0.0, 200.0], [0.0, 1.0, -100.0]])

        observed = preview_matrix(native, (1000, 1200), (12000, 10000), (800, 1000), (10000, 8000))

        point = np.array([400.0, 300.0, 1.0])
        np.testing.assert_allclose(observed @ point, [420.0, 290.0])

    def test_local_matrix_maps_source_crop_into_field(self):
        matrix = np.array([[1.0, 0.0, 150.0], [0.0, 1.0, -80.0]])
        field = AlignedField(1000, 1200, 2048, 2048, 1.0, 0.1, 10, "common_tissue")
        bounds = source_bounds(matrix, field, (10000, 10000), margin=0)
        local = local_matrix(matrix, bounds[0], bounds[1], field)

        self.assertEqual(bounds, (850, 1280, 2048, 2048))
        np.testing.assert_allclose(local @ np.array([0.0, 0.0, 1.0]), [0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
