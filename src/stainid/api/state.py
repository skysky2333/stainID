"""Cached, read-mostly view of a project's manifests and outputs for the web API."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import pandas as pd

from stainid.project import Project, load_project

SETTINGS = Path.home() / ".stainid" / "app.json"


def _read(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, **kwargs) if path.exists() else pd.DataFrame()


@dataclass
class ProjectState:
    project: Project
    _cache: dict = field(default_factory=dict)

    @property
    def root(self) -> Path:
        return self.project.root

    @cached_property
    def jobs(self):
        from stainid.api.jobs import JobManager

        return JobManager(self.root, self.project.output("jobs"))

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

    def groups(self) -> list[dict[str, str]]:
        """Diagnostic groups in plot order: from the project settings, else as found in the data."""
        configured = self.project.config.get("groups") or []
        found = sorted(set(self.tiles.disease_group.dropna()) if "disease_group" in self.tiles else set())
        codes = [g["code"] for g in configured]
        return list(configured) + [{"code": c, "label": c} for c in found if c not in codes]

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


def _settings() -> dict:
    return json.loads(SETTINGS.read_text()) if SETTINGS.exists() else {"recent": []}


def recent_projects() -> list[str]:
    return [p for p in _settings()["recent"] if Path(p).exists()]


def remembered_project() -> Path:
    """The project opened last in the app, else the current folder."""
    recent = recent_projects()
    return Path(recent[0]) if recent else Path.cwd()


def _remember(root: Path) -> None:
    settings = _settings()
    settings["recent"] = [str(root)] + [p for p in settings["recent"] if p != str(root)][:9]
    SETTINGS.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS.write_text(json.dumps(settings, indent=1))


def open_project(path: Path | str) -> ProjectState:
    global _STATE
    if _STATE is not None and "jobs" in _STATE.__dict__ and _STATE.jobs.active():
        raise RuntimeError("Jobs are still running in this project; wait for them or cancel them before switching projects")
    project = load_project(path)
    os.chdir(project.root)
    for key in ("HF_HOME", "CELLPOSE_LOCAL_MODELS_PATH", "STAINID_PLAQUE_CNN_DIR"):
        os.environ.pop(key, None)
    project.apply_environment()
    os.environ["STAINID_PROJECT"] = str(project.root)
    if project.is_configured:
        _remember(project.root)
    _STATE = ProjectState(project)
    return _STATE


def get_state() -> ProjectState:
    return _STATE or open_project(os.environ.get("STAINID_PROJECT", "."))
