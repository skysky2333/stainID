import unittest

from stainid.analysis.aggregation import aggregate_level, pivot_donor_region


class CohortAggregationTest(unittest.TestCase):
    def test_preserves_count_and_area_numerators(self):
        rows = []
        for order, area, count, eligible, compact in ((1, 1.0, 2, 2, 1), (2, 2.0, 6, 4, 2)):
            rows.append(
                {
                    "tile_id": f"T{order}",
                    "core_id": "C1",
                    "tma": "1",
                    "donor_id": "D1",
                    "sample_region_id": "D1_FR",
                    "region": "frontal",
                    "disease_group": "ASYMP",
                    "technical_replicate": "1",
                    "stain": "6E10",
                    "selection_order": str(order),
                    "tissue_area_mm2": str(area),
                    "total_artifact_exclusion_area_fraction": "0.0",
                    "accepted_plaque_candidate_count": str(count),
                    "morphotype_eligible_plaque_count": str(eligible),
                    "compact_candidate_count": str(compact),
                    "accepted_plaque_deposit_area_fraction": "0.1",
                    "amyloid_positive_area_fraction": "0.2",
                }
            )
        aggregated = aggregate_level(rows, [], "core")
        primary = next(row for row in aggregated if row["sampling_extent"] == "primary_four")
        self.assertEqual(primary["accepted_plaque_count"], 8)
        self.assertAlmostEqual(primary["plaque_density_mm2"], 8 / 3)
        self.assertAlmostEqual(primary["compact_plaque_fraction"], 0.5)
        self.assertEqual(primary["small_plaque_count"], 2)
        self.assertAlmostEqual(primary["accepted_plaque_area_fraction"], 0.1)
        self.assertAlmostEqual(primary["amyloid_positive_area_fraction"], 0.2)
        self.assertFalse(primary["plaque_composition_eligible"])

    def test_neun_aggregation_preserves_candidate_denominator(self):
        rows = [
            {
                "tile_id": "T1",
                "core_id": "C1",
                "tma": "1",
                "donor_id": "D1",
                "sample_region_id": "D1_FR",
                "region": "frontal",
                "disease_group": "AD",
                "technical_replicate": "1",
                "stain": "NeuN",
                "selection_order": "1",
                "tissue_area_mm2": "2.0",
                "total_artifact_exclusion_area_fraction": "0.0",
                "candidate_profile_count": "10",
                "neun_positive_profile_count": "4",
                "neun_union_positive_profile_count": "5",
                "neun_review_profile_count": "1",
                "neun_positive_profile_area_fraction": "0.02",
                "neun_local_density_cv_100um": "0.5",
                "neun_local_density_window_count": "8",
            }
        ]
        aggregated = aggregate_level(rows, [], "core")[0]
        self.assertEqual(aggregated["neun_candidate_profile_count"], 10)
        self.assertAlmostEqual(aggregated["neun_positive_fraction_of_candidates"], 0.4)
        self.assertAlmostEqual(aggregated["neun_union_profile_density_mm2"], 2.5)
        self.assertAlmostEqual(aggregated["neun_positive_profile_area_fraction"], 0.02)
        self.assertAlmostEqual(aggregated["neun_local_density_cv_100um"], 0.5)

    def test_at8_aggregation_uses_noncompact_area_name_with_legacy_input(self):
        rows = [
            {
                "tile_id": "T1",
                "core_id": "C1",
                "tma": "1",
                "donor_id": "D1",
                "sample_region_id": "D1_FR",
                "region": "frontal",
                "disease_group": "AD",
                "technical_replicate": "1",
                "stain": "AT8",
                "selection_order": "1",
                "tissue_area_mm2": "2.0",
                "total_artifact_exclusion_area_fraction": "0.0",
                "at8_positive_area_fraction": "0.2",
                "thread_area_fraction": "0.15",
                "compact_profile_count": "4",
                "thread_skeleton_length_mm_per_mm2": "3.0",
                "edge_guard_triggered": "false",
            }
        ]
        aggregated = aggregate_level(rows, [], "core")[0]
        self.assertAlmostEqual(
            aggregated["tau_noncompact_area_fraction_of_at8"], 0.75
        )
        self.assertAlmostEqual(aggregated["at8_noncompact_area_mm2"], 0.3)

    def test_pivots_stains_without_creating_new_samples(self):
        rows = [
            {
                "tma": "1",
                "donor_id": "D1",
                "sample_region_id": "D1_FR",
                "region": "frontal",
                "disease_group": "AD",
                "analysis_level": "donor_region",
                "sampling_extent": "primary_four",
                "technical_replicate": "pooled",
                "stain": stain,
                "tissue_area_mm2": index,
            }
            for index, stain in enumerate(("6E10", "AT8", "NeuN"), start=1)
        ]
        wide = pivot_donor_region(rows)
        self.assertEqual(len(wide), 1)
        self.assertEqual(wide[0]["available_stain_count"], 3)
        self.assertEqual(wide[0]["neun_tissue_area_mm2"], 3)


if __name__ == "__main__":
    unittest.main()
