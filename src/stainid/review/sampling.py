"""Draw objects for a new review set from pipeline outputs (random, uncertain-probability, or per-class)."""
from __future__ import annotations

import pandas as pd

from stainid.outputs import objects
from stainid.project import Project

DEFAULT_LABELS = {
    "NeuN": ["neuron", "neun_negative_cell", "artifact", "uncertain"],
    "6E10": ["compact_plaque", "diffuse_plaque", "vascular", "not_plaque", "uncertain"],
    "AT8": ["tau_neuron", "neurite_fragment", "artifact", "negative", "uncertain"],
}
DEFAULT_FOV_UM = {"NeuN": 45.0, "6E10": 85.0, "AT8": 55.0}


def sample_items(project: Project, tiles: pd.DataFrame, stain: str, n: int, strategy: str = "random", model_class: str | None = None,
                 low: float = 0.3, high: float = 0.7, per_group: bool = True, seed: int = 0) -> pd.DataFrame:
    pool = objects(project, tiles, stain)
    if pool.empty:
        return pool
    if model_class:
        pool = pool[pool.model_class == model_class]
    if strategy == "uncertain":
        pool = pool[pool.probability.between(low, high)]
    if per_group and "disease_group" in pool:
        groups = pool.disease_group.dropna().unique()
        take = max(1, n // max(len(groups), 1))
        pool = pd.concat([g.sample(min(len(g), take), random_state=seed) for _, g in pool.groupby("disease_group")])
    else:
        pool = pool.sample(min(len(pool), n), random_state=seed)
    return pool[["tile_id", "stain", "x", "y", "model_class", "probability", "disease_group", "donor_id", "sample_region_id", "region"]]
