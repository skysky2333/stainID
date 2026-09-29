import unittest

from stainid.sampling.analysis_manifest import analysis_rows


def row(stain, donor="1001"):
    return {
        "tma": "1",
        "core_label": "B-2",
        "stain": stain,
        "core_role": "biological",
        "output_path": f"cores/LIP-1/B-2/{stain}.png",
        "output_width_px": "14000",
        "output_height_px": "13900",
        "slide_path": f"slides/{stain}.vsi",
        "center_x_px": "21000",
        "center_y_px": "18000",
        "diameter_x_px": "14000",
        "diameter_y_px": "13900",
        "source_pixel_width_um": "0.274",
        "source_pixel_height_um": "0.274",
        "donor_id": donor,
        "region": "frontal",
        "disease_group": "AD",
        "cerad": "C",
        "braak": "6",
        "technical_replicate": "1",
        "sample_region_id": "BRC_1001_FR",
        "tissue_status": "substantial",
        "tissue_fraction": "0.9",
        "review_required": "false",
    }


class AnalysisManifestTest(unittest.TestCase):
    def test_builds_ordered_triplet(self):
        records = analysis_rows([row("NeuN"), row("6E10"), row("AT8")])
        self.assertEqual([record["stain"] for record in records], ["6E10", "AT8", "NeuN"])
        self.assertTrue(all(record["core_id"] == "LIP-1_B-2" for record in records))
        self.assertEqual(records[0]["image_path"], "data/cores/LIP-1/B-2/6E10.png")
        self.assertEqual(records[0]["slide_path"], "slides/6E10.vsi")

    def test_rejects_cross_stain_identity_disagreement(self):
        rows = [row("6E10"), row("AT8"), row("NeuN", donor="1002")]
        with self.assertRaisesRegex(ValueError, "donor_id"):
            analysis_rows(rows)


if __name__ == "__main__":
    unittest.main()
