"""Summarise SAM object masks into core- or donor-region shape features."""
from __future__ import annotations

import numpy as np

MINIMUM_OBJECTS = {"NeuN": 20, "AT8": 5, "6E10": 10}




def values(rows, key):
    return np.asarray([float(r[key]) for r in rows if r.get(key) not in ("", "nan", None)], dtype=float)


def summarize(stain: str, objects: list[dict[str, str]], tissue_mm2: float) -> dict[str, float]:
    prefix = {"NeuN": "neun_soma", "AT8": "tau_neuron", "6E10": "plaque"}[stain]
    enough = len(objects) >= MINIMUM_OBJECTS[stain]
    med = lambda key: float(np.median(values(objects, key))) if enough else float("nan")
    area = values(objects, "area_um2")
    out = {
        f"{prefix}_mask_count": len(objects),
        f"{prefix}_mask_density_mm2": len(objects) / tissue_mm2 if tissue_mm2 else float("nan"),
        f"{prefix}_area_median_um2": med("area_um2"),
        f"{prefix}_area_p90_um2": float(np.quantile(area, 0.9)) if enough else float("nan"),
        f"{prefix}_area_cv": float(area.std() / area.mean()) if enough else float("nan"),
        f"{prefix}_solidity_median": med("solidity"),
        f"{prefix}_circularity_median": med("circularity"),
        f"{prefix}_elongation_median": med("elongation"),
        f"{prefix}_normalized_dab_median": med("normalized_mean_dab"),
    }
    if stain == "6E10":
        core = values(objects, "dense_core_fraction")
        out |= {
            "plaque_mask_area_fraction": float(area.sum() / 1e6 / tissue_mm2) if tissue_mm2 else float("nan"),
            "plaque_dense_core_fraction_median": med("dense_core_fraction"),
            "plaque_cored_fraction": float((core >= 0.25).mean()) if enough else float("nan"),
            "plaque_multicore_fraction": float((values(objects, "dense_core_count") >= 2).mean()) if enough else float("nan"),
        }
    if stain == "AT8":
        mature = [r.get("seed_mature", "").lower() == "true" for r in objects]
        out["tau_neuron_mask_mature_fraction"] = float(np.mean(mature)) if enough else float("nan")
    return out
