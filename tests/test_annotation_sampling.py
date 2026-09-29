import csv
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from stainid.sampling.annotation import create_annotation_manifests


def manifest_rows():
    rows = []
    groups = ("AD", "ASYMP", "CT")
    regions = ("frontal", "occipital")
    core_number = 0
    for tma in ("1", "2"):
        for group in groups:
            for region in regions:
                for candidate in range(2):
                    core_number += 1
                    core_label = f"A-{core_number}"
                    for stain_index, stain in enumerate(("6E10", "AT8", "NeuN")):
                        status = (
                            "sparse"
                            if tma == "1"
                            and group == "AD"
                            and region == "frontal"
                            and candidate == 0
                            and stain == "AT8"
                            else "substantial"
                        )
                        rows.append(
                            {
                                "slide": f"TMA LIP-{tma} {stain}",
                                "tma": tma,
                                "core_label": core_label,
                                "stain": stain,
                                "core_role": "biological",
                                "donor_id": f"D{core_number:03d}",
                                "sample_region_id": f"D{core_number:03d}_{region}",
                                "disease_group": group,
                                "region": region,
                                "cerad": "C",
                                "braak": "6",
                                "technical_replicate": "1",
                                "tissue_status": status,
                                "tissue_fraction": "0.10" if status == "sparse" else "0.90",
                                "dab_od_p90": f"{candidate + stain_index / 10 + 0.1}",
                                "output_path": f"cores/LIP-{tma}/{core_label}/{stain}.png",
                                "output_width_px": "14000",
                                "output_height_px": "14000",
                                "source_pixel_width_um": "0.274",
                                "source_pixel_height_um": "0.274",
                            }
                        )
    return rows


class AnnotationSamplingTest(unittest.TestCase):
    def test_balanced_triplets_and_blinded_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "core_manifest.csv"
            rows = manifest_rows()
            with manifest.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

            paths = create_annotation_manifests(manifest, root / "annotations", seed=7)
            with paths["annotation_manifest"].open(newline="", encoding="utf-8") as handle:
                public = list(csv.DictReader(handle))
            with paths["qupath_manifest"].open(encoding="utf-8") as handle:
                qupath_manifest = json.load(handle)
            with paths["selection_key"].open(newline="", encoding="utf-8") as handle:
                key = list(csv.DictReader(handle))
            with paths["qc_challenge_key"].open(newline="", encoding="utf-8") as handle:
                challenges = list(csv.DictReader(handle))

            self.assertEqual(len(key), 12)
            self.assertEqual(len(public), 36)
            self.assertEqual(qupath_manifest, public)
            self.assertEqual(Counter(row["stain"] for row in public), {"6E10": 12, "AT8": 12, "NeuN": 12})
            self.assertNotIn("disease_group", public[0])
            self.assertEqual(
                Counter((row["tma"], row["disease_group"], row["region"]) for row in key),
                Counter(
                    (tma, group, region)
                    for tma in ("1", "2")
                    for group in ("AD", "ASYMP", "CT")
                    for region in ("frontal", "occipital")
                ),
            )
            self.assertEqual(len(challenges), 1)
            self.assertEqual(challenges[0]["AT8_tissue_status"], "sparse")


if __name__ == "__main__":
    unittest.main()
