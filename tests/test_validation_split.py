import unittest
from collections import Counter

from stainid.sampling.validation_split import select_validation_triplets, subset_by_image_id, validation_rows


class ValidationSplitTest(unittest.TestCase):
    def test_selects_balanced_unused_triplets(self):
        rows = []
        for tma in range(1, 4):
            for index, (group, region) in enumerate(
                (("CT", "frontal"), ("ASYMP", "occipital"), ("AD", "frontal"))
            ):
                rows.append(
                    {
                        "annotation_id": f"M{tma}{index}",
                        "tma": str(tma),
                        "disease_group": group,
                        "region": region,
                        "selection_distance": str(index / 10),
                        "minimum_tissue_fraction": "0.8",
                    }
                )
        selected = select_validation_triplets(rows, {"M10"})
        self.assertEqual({row["tma"] for row in selected}, {"1", "2", "3"})
        self.assertNotIn("M10", {row["annotation_id"] for row in selected})
        self.assertLessEqual(max(Counter(row["disease_group"] for row in selected).values()), 1)

    def test_builds_three_stain_blinded_manifest(self):
        selected = [
            {
                "annotation_id": "M01",
                "tma": "1",
                "core_label": "A-1",
                "donor_id": "0001",
                "sample_region_id": "BRC_0001_FR",
                "disease_group": "CT",
                "region": "frontal",
                "cerad": "0",
                "braak": "0",
                "technical_replicate": "1",
            }
        ]
        annotations = []
        fields = []
        for stain in ("6E10", "AT8", "NeuN"):
            image_id = f"M01_{stain}"
            annotations.append(
                {"annotation_id": "M01", "image_id": image_id, "stain": stain}
            )
            fields.append(
                {
                    "annotation_id": "M01",
                    "image_id": image_id,
                    "field_id": f"{image_id}_F01",
                    "target_dab_quantile": "0.50",
                }
            )
        blinded, key = validation_rows(selected, annotations, fields)
        self.assertEqual(len(blinded), 3)
        self.assertNotIn("disease_group", blinded[0])
        self.assertEqual(key[0]["disease_group"], "CT")

    def test_subsets_complete_adjudication_images(self):
        rows = [{"image_id": "M02"}, {"image_id": "M01"}, {"image_id": "M03"}]
        selected = subset_by_image_id(rows, {"M01", "M03"})
        self.assertEqual([row["image_id"] for row in selected], ["M01", "M03"])
        with self.assertRaises(ValueError):
            subset_by_image_id(rows, {"M04"})


if __name__ == "__main__":
    unittest.main()
