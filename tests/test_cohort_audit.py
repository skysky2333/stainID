import unittest

from stainid.analysis.cohort_audit import build_stage_overlap


class CohortAuditTest(unittest.TestCase):
    def test_joint_stage_overlap_counts_exact_matches(self):
        donors = [
            {"disease_group": "ASYMP", "cerad": "B", "braak": "4"},
            {"disease_group": "ASYMP", "cerad": "C", "braak": "5"},
            {"disease_group": "AD", "cerad": "C", "braak": "5"},
            {"disease_group": "AD", "cerad": "C", "braak": "6"},
            {"disease_group": "CT", "cerad": "0", "braak": "0"},
        ]

        observed = build_stage_overlap(donors)

        self.assertEqual(
            observed,
            [
                {
                    "cerad": "B",
                    "braak": "4",
                    "asymp_donors": 1,
                    "ad_donors": 0,
                    "exact_stage_matched_pairs": 0,
                },
                {
                    "cerad": "C",
                    "braak": "5",
                    "asymp_donors": 1,
                    "ad_donors": 1,
                    "exact_stage_matched_pairs": 1,
                },
                {
                    "cerad": "C",
                    "braak": "6",
                    "asymp_donors": 0,
                    "ad_donors": 1,
                    "exact_stage_matched_pairs": 0,
                },
            ],
        )


if __name__ == "__main__":
    unittest.main()
