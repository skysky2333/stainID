"""Slide-scaled model inputs (`s_<name>` columns of the store), for models trained with `v2.training.<stain>.slide_scaled`.

The current models take DAB intensities in absolute optical density, and some measures are cut at absolute OD floors.
Slides differ 3–4 fold in staining strength (6E10 thresholds 0.026–0.108 OD, AT8 0.014–0.055, NeuN 0.026–0.099), so an
object on a pale slide looks faint to such a model however clear it is on its own slide. Each `s_` input is its `m_`
input in slide thresholds, or the measure recomputed at a level relative to the threshold where the legacy one used an
absolute floor:
- 6E10: the identity model's 14 DAB intensities, and the seed and dense core at relative levels (`amyloid.scaled_core`);
- NeuN: mean DAB, the 80th percentile around the nucleus, the context DAB, and the stained area above 0.35 × threshold
  without the 0.012 OD floor;
- AT8: the soma's DAB 90th percentile, the background median and the context DAB (contrast and soma DAB are already
  in thresholds); the mature-tangle cut-off then applies to the scaled 90th percentile.
"""
from __future__ import annotations

import numpy as np

CONTEXT_DAB = ("context_dab_mean", "context_dab_p90", "context_dab_std", "center_dab_mean", "center_dab_p90", "annulus_dab_mean",
               "center_annulus_dab_contrast")
INTENSITY = {
    "6E10": ("mean_dab_od", "median_dab_od", "p90_dab_od", "max_dab_od", "inner_mean_dab_od", "outer_mean_dab_od", "radial_dab_contrast",
             *CONTEXT_DAB),
    "NeuN": ("mean_dab_od", "neighborhood_dab_p80", *CONTEXT_DAB),
    "AT8": ("soma_dab_p90", "background_dab_median", *(f"ctx_{n}" for n in CONTEXT_DAB)),
}
RECOMPUTED = {"6E10": ("seed_area_um2", "core_area_um2", "core_fraction", "normalized_core_offset"), "NeuN": ("stained_area_um2",), "AT8": ()}
MATURE = "s_soma_dab_p90"  # the AT8 maturity feature of a slide-scaled model


def names(stain: str, inputs: list[str]) -> list[str]:
    """A model's input names with each one that has a slide-scaled version replaced by it (`s_<name>`)."""
    scaled = set(INTENSITY.get(stain, ())) | set(RECOMPUTED.get(stain, ()))
    return [f"s_{n}" if n in scaled else n for n in inputs]


def intensities(stain: str, values: dict, threshold: float) -> dict:
    """The `s_` intensities of one candidate from its `m_` inputs."""
    return {f"s_{n}": values[f"m_{n}"] / threshold for n in INTENSITY.get(stain, ()) if f"m_{n}" in values}


def neun_stained_area(field, region, legacy_um2: float) -> float:
    """NeuN's stained area (tissue, not excluded, DAB ≥ 0.35 × threshold, at least half chromogen) without the legacy
    0.012 OD floor: the legacy area times the share the relative level adds within the candidate (the same value where
    the floor does not bind, i.e. thresholds above 0.034 OD)."""
    floor = max(0.012, 0.35 * field.threshold)
    if 0.35 * field.threshold >= floor or not np.isfinite(legacy_um2):
        return float(legacy_um2)
    box, mask = region.box, region.mask
    keep = field.tissue[box] & ~field.neun_exclusion[box] & (field.chromogen_fraction[box] >= 0.50) & mask
    dab = field.dab[box]
    absolute, relative = int((keep & (dab >= floor)).sum()), int((keep & (dab >= 0.35 * field.threshold)).sum())
    if absolute:
        return float(legacy_um2) * relative / absolute
    return relative * field.pixel_width_um * field.pixel_height_um


__all__ = ["CONTEXT_DAB", "INTENSITY", "MATURE", "RECOMPUTED", "intensities", "names", "neun_stained_area"]
