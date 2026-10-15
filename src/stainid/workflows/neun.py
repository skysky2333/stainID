"""NeuN neuron detection: stain-contour + Cellpose-SAM candidates, context random forest."""
from __future__ import annotations

import os

from stainid.project import Project
from stainid.stains.neun.pipeline import merge_neun_cohort, run_neun_cohort
from stainid.tables import read_csv
from stainid.workflows import follow_nuclei


def run_neun(project: Project, device: str = "cpu", threads: int = 8, batch_size: int = 1, shard_index: int = 0, shard_count: int = 1,
             tile_ids: set[str] | None = None) -> None:
    project.apply_environment()
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/stainid_mpl_cache")
    accept = project.config["acceptance"].get("neun")
    run_neun_cohort(project.analysis_tiles(), project.input("calibration"), project.model("neun"), project.model("cellpose"),
                    project.output("neun"), project.output("nuclei"), project.input("manual_exclusions"), tile_ids, None, shard_index, shard_count, threads,
                    device=device, batch_size=batch_size, precision=project.precision("cellpose"),
                    accept_threshold=None if accept is None else float(accept))


def follow_neun(project: Project, device: str = "cpu", threads: int = 8, shard_index: int = 0, shard_count: int = 1) -> None:
    """Detect NeuN on each core image as soon as Find nuclei has finished it (see `follow_nuclei`)."""
    tiles = read_csv(project.analysis_tiles())

    def run(_, core_ids):
        run_neun(project, device, threads, tile_ids={t["tile_id"] for t in tiles if t["stain"] == "NeuN" and t["core_id"] in core_ids})

    follow_nuclei(project, {"NeuN"}, run, shard_index, shard_count)


def merge_neun(project: Project):
    out = project.output("neun")
    return merge_neun_cohort(out, out / "tile_features.csv", out / "objects.csv")
