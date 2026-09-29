import unittest

from stainid.sampling.stability import compare_sampling_extents


class SamplingStabilityTest(unittest.TestCase):
    def test_compares_matched_core_estimates(self):
        rows = []
        for index, (primary, extended) in enumerate(((1.0, 1.1), (2.0, 2.2), (3.0, 3.3))):
            for extent, value in (("primary_four", primary), ("extended_eight", extended)):
                rows.append(
                    {
                        "core_id": f"C{index}",
                        "stain": "NeuN",
                        "sampling_extent": extent,
                        "neun_profile_density_mm2": str(value),
                    }
                )
        result = compare_sampling_extents(
            rows, {"NeuN": "neun_profile_density_mm2"}
        )[0]
        self.assertEqual(result["paired_core_count"], 3)
        self.assertAlmostEqual(result["pearson_r"], 1.0)
        self.assertAlmostEqual(result["spearman_r"], 1.0)
        self.assertAlmostEqual(result["median_relative_absolute_difference"], 2 / 21)


if __name__ == "__main__":
    unittest.main()
