"""Cached, read-mostly view of a project's manifests and outputs for the web API."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import pandas as pd

from stainid.project import Project, load_project

GROUP_LABELS = {"CT": "Control", "ASYMP": "ASYMAD", "AD": "AD"}


def _read(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, **kwargs) if path.exists() else pd.DataFrame()


@dataclass
class ProjectState:
    project: Project
    _cache: dict = field(default_factory=dict)

    @property
    def root(self) -> Path:
        return self.project.root

    def cache_dir(self, name: str) -> Path:
        path = self.root / "data" / "cache" / name
        path.mkdir(parents=True, exist_ok=True)
        return path

    @cached_property
    def tiles(self) -> pd.DataFrame:
        return _read(self.project.input("tile_manifest"), dtype={"tma": str, "donor_id": str})

    @cached_property
    def core_images(self) -> pd.DataFrame:
        return _read(self.project.input("core_images"), dtype={"tma": str, "donor_id": str})

    @cached_property
    def layout(self) -> pd.DataFrame:
        return _read(self.project.input("tma_layout"), dtype={"tma": str, "donor_id": str})

    @cached_property
    def donors(self) -> pd.DataFrame:
        return _read(self.project.input("donor_metadata"), dtype={"tma": str, "donor_id": str})

    def calibration(self) -> pd.DataFrame:
        return _read(self.project.input("calibration"), dtype={"tma": str})

    def tile(self, tile_id: str) -> dict:
        rows = self.tiles[self.tiles.tile_id == tile_id]
        if rows.empty:
            raise KeyError(tile_id)
        return rows.iloc[0].to_dict()

    def core_image(self, core_id: str, stain: str) -> dict:
        rows = self.core_images[(self.core_images.core_id == core_id) & (self.core_images.stain == stain)]
        if rows.empty:
            raise KeyError(f"{core_id} {stain}")
        return rows.iloc[0].to_dict()

    def output_status(self) -> dict[str, dict[str, int]]:
        """How many cores / fields each pipeline stage has finished."""
        tiles = self.tiles
        fields_parts = self.project.output("fields") / "parts"
        masks = self.project.output("masks")
        status = {}
        for stain in sorted(tiles.stain.unique()) if not tiles.empty else []:
            stain_tiles = tiles[tiles.stain == stain]
            cores = set(stain_tiles.core_id)
            if stain == "NeuN":
                done = sum((self.project.output("neun") / "tiles" / f"{t}_features.csv").exists() for t in stain_tiles.tile_id)
                detection = {"done": done, "total": len(stain_tiles), "unit": "fields"}
            else:
                done = sum((fields_parts / f"{c}_{stain}_features.csv").exists() for c in cores)
                detection = {"done": done, "total": len(cores), "unit": "cores"}
            nuclei = sum((self.project.output("nuclei") / f"{t}.npz").exists() for t in stain_tiles.tile_id)
            mask_done = sum((masks / stain / "parts" / f"{c}_{stain}_objects.csv").exists() for c in cores)
            status[stain] = {"detection": detection, "nuclei": {"done": nuclei, "total": len(stain_tiles), "unit": "fields"},
                             "masks": {"done": mask_done, "total": len(cores), "unit": "cores"}}
        return status


_STATE: ProjectState | None = None


def get_state() -> ProjectState:
    global _STATE
    if _STATE is None:
        project = load_project(os.environ.get("STAINID_PROJECT"))
        os.chdir(project.root)
        project.apply_environment()
        _STATE = ProjectState(project)
    return _STATE
