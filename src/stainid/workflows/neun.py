"""NeuN neuron detection: stain-contour + Cellpose-SAM candidates, context random forest."""
from __future__ import annotations

import os

from stainid.project import Project
from stainid.stains.neun.pipeline import merge_neun_cohort, run_neun_cohort


def run_neun(project: Project, device: str = "cpu", threads: int = 8, batch_size: int = 1, shard_index: int = 0, shard_count: int = 1,
             tile_ids: set[str] | None = None, nested_samples: set[str] | None = None) -> None:
    project.apply_environment()
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/stainid_mpl_cache")
    run_neun_cohort(project.input("tile_manifest"), project.input("calibration"), project.model("neun"), project.model("cellpose"),
                    project.output("neun"), project.input("manual_exclusions"), tile_ids, nested_samples, shard_index, shard_count, threads,
                    device=device, batch_size=batch_size)


def merge_neun(project: Project):
    out = project.output("neun")
    return merge_neun_cohort(out, out / "tile_features.csv", out / "objects.csv")
