import unittest

from stainid.registration.validation import select_validation_cores


class RegistrationValidationTest(unittest.TestCase):
    def test_includes_flags_and_balanced_pass_examples(self):
        registrations = []
        keys = []
        for tma in (1, 2):
            for index, (group, region) in enumerate(
                (("ASYMP", "frontal"), ("ASYMP", "occipital"), ("AD", "frontal"), ("AD", "occipital")),
                start=1,
            ):
                core_label = f"A-{index}"
                core_id = f"LIP-{tma}_{core_label}"
                keys.append(
                    {
                        "annotation_id": f"M{tma}{index}",
                        "tma": str(tma),
                        "core_label": core_label,
                        "disease_group": group,
                        "region": region,
                    }
                )
                for stain in ("6E10", "AT8"):
                    registrations.append(
                        {
                            "core_id": core_id,
                            "moving_stain": stain,
                            "status": "pass",
                        }
                    )
        registrations[0]["status"] = "review"

        selected = select_validation_cores(registrations, keys)

        self.assertEqual(selected["LIP-1_A-1"], "coarse_qc_flag")
        self.assertEqual(selected["LIP-1_A-4"], "balanced_landmark_sample")
        self.assertEqual(selected["LIP-2_A-2"], "balanced_landmark_sample")
        self.assertEqual(selected["LIP-2_A-3"], "balanced_landmark_sample")

    def test_includes_development_pilot_when_present(self):
        registrations = [
            {"core_id": "LIP-5_C-1", "moving_stain": stain, "status": "pass"}
            for stain in ("6E10", "AT8")
        ]

        selected = select_validation_cores(registrations, [])

        self.assertEqual(selected["LIP-5_C-1"], "development_pilot")


if __name__ == "__main__":
    unittest.main()
