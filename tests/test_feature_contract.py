import tempfile
import unittest
from pathlib import Path

from stainid.analysis.feature_contract import FEATURES, validate_feature_contract, write_feature_contract


class FeatureContractTest(unittest.TestCase):
    def test_primary_endpoints_are_limited_and_spatial_features_are_blocked(self):
        validate_feature_contract()
        primary = [row["feature_id"] for row in FEATURES if row["endpoint_role"] == "primary"]
        self.assertEqual(
            primary,
            [
                "compact_plaque_fraction",
                "tau_noncompact_area_fraction_of_at8",
                "neun_profile_density_mm2",
            ],
        )
        spatial = [row for row in FEATURES if row["registration_required"] == "true"]
        self.assertTrue(spatial)
        self.assertTrue(all(row["current_status"] == "blocked_by_registration" for row in spatial))

    def test_writes_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            output = write_feature_contract(Path(directory) / "features.csv")
            self.assertEqual(len(output.read_text().splitlines()), len(FEATURES) + 1)


if __name__ == "__main__":
    unittest.main()
