from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import yaml


@pytest.fixture()
def synthetic_project(tmp_path: Path, monkeypatch) -> Path:
    """A one-core, three-stain project with tiny images and pre-made pipeline outputs."""
    rng = np.random.default_rng(0)
    core_dir = tmp_path / "data" / "cores" / "LIP-1" / "B-2"
    core_dir.mkdir(parents=True)
    tiles, images = [], []
    for stain in ("NeuN", "6E10", "AT8"):
        rgb = np.full((1200, 1200, 3), 225, np.uint8)
        for x, y in rng.integers(200, 1000, size=(25, 2)):
            cv2.circle(rgb, (int(x), int(y)), 12, (120, 80, 40), -1)
        cv2.imwrite(str(core_dir / f"{stain}.png"), rgb)
        images.append({"core_id": "LIP-1_B-2", "tma": 1, "stain": stain, "image_path": f"data/cores/LIP-1/B-2/{stain}.png",
                       "native_width_px": 1200, "native_height_px": 1200, "tissue_fraction": 0.9, "tissue_status": "complete"})
        tiles.append({"tile_id": f"LIP-1_B-2_{stain}_S01", "core_id": "LIP-1_B-2", "tma": 1, "core_label": "B-2", "donor_id": "0001",
                      "sample_region_id": "BRC_0001_FR", "region": "frontal", "disease_group": "AD", "technical_replicate": 1, "stain": stain,
                      "image_path": f"data/cores/LIP-1/B-2/{stain}.png", "pixel_width_um": 0.2738, "pixel_height_um": 0.2738,
                      "selection_order": 1, "nested_sample": "primary_four", "x_px": 300, "y_px": 300, "width_px": 512, "height_px": 512})
    analysis = tmp_path / "data" / "analysis"
    analysis.mkdir(parents=True)
    pd.DataFrame(tiles).to_csv(analysis / "cohort_primary_tiles.csv", index=False)
    pd.DataFrame(images).to_csv(analysis / "core_images.csv", index=False)
    pd.DataFrame([{"tma": 1, "stain": s, "threshold_dab_od": 0.05} for s in ("NeuN", "6E10", "AT8")]).to_csv(analysis / "slide_dab_calibration.csv", index=False)
    pd.DataFrame([{"tma": 1, "core_label": "B-2", "donor_id": "0001", "region": "frontal", "disease_group": "AD"}]).to_csv(tmp_path / "data" / "tma_layout.csv", index=False)
    pd.DataFrame([{"donor_id": "0001", "tma": 1, "disease_group": "AD"}]).to_csv(tmp_path / "data" / "donor_metadata.csv", index=False)
    parts = analysis / "cohort_v3" / "parts"
    parts.mkdir(parents=True)
    common = {"tile_id": "LIP-1_B-2_6E10_S01", "core_id": "LIP-1_B-2", "tma": 1, "donor_id": "0001", "sample_region_id": "BRC_0001_FR",
              "region": "frontal", "disease_group": "AD", "technical_replicate": 1, "stain": "6E10", "selection_order": 1}
    plaques = [{**common, "centroid_x_px": 100 + 30 * i, "centroid_y_px": 200, "plaque_class": ["compact", "diffuse"][i % 2], "plaque_probability": 0.9,
                "plaque_area_um2": 300.0, "dense_core_fraction": 0.2, "equivalent_diameter_um": 20.0} for i in range(8)]
    pd.DataFrame(plaques).to_csv(parts / "LIP-1_B-2_6E10_objects.csv", index=False)
    (tmp_path / "data" / "annotations").mkdir()
    (tmp_path / "data" / "annotations" / "cohort_manual_exclusions.json").write_text(json.dumps({}))
    (tmp_path / "stainid.yaml").write_text(yaml.safe_dump({"name": "synthetic"}))
    monkeypatch.setenv("STAINID_PROJECT", str(tmp_path))
    return tmp_path
