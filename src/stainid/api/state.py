"""The projects open in the server: a cached, read-mostly view of each one's manifests and outputs, and its job queue.

Several projects can be open at once. Each request names its project (`/api/projects/<id>/...`, see `stainid.api.app`),
which sets `CURRENT` for that request only, so `get_state()` returns that project's state. A request without one (the
routes of older web apps) uses the project opened last. Nothing here changes the server's working folder or
environment, so projects never see each other's settings.
"""
from __future__ import annotations

import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import pandas as pd

from stainid.api import registry
from stainid.jobs import Job, JobStore, alive
from stainid.project import Project, load_project

CURRENT: ContextVar[str | None] = ContextVar("stainid_project", default=None)


class UnknownProject(KeyError):
    """A request named a project this server does not know."""


def _read(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, **kwargs) if path.exists() else pd.DataFrame()


@dataclass
class ProjectState:
    project: Project
    _cache: dict = field(default_factory=dict)

    @property
    def root(self) -> Path:
        return self.project.root

    @property
    def id(self) -> str:
        return registry.project_id(self.root)

    @cached_property
    def jobs(self):
        from stainid.api.jobs import JobManager

        return JobManager(self.root, self.project.output("jobs"), name=self.project.name)

    def cache_dir(self, name: str) -> Path:
        path = self.root / "data" / "cache" / name
        path.mkdir(parents=True, exist_ok=True)
        return path

    @cached_property
    def tiles(self) -> pd.DataFrame:
        return _read(self.project.input("tile_manifest"), dtype={"tma": str, "donor_id": str})

    @cached_property
    def analysis_tiles(self) -> pd.DataFrame:
        """The tiles detection runs on (whole-core tiles or the sampled fields, per Settings → Analysis)."""
        return _read(self.project.analysis_tiles(), dtype={"tma": str, "donor_id": str})

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


_STATES: dict[str, ProjectState] = {}  # the open projects, by id
_LAST: str | None = None  # the project opened last: requests that name no project use it
_LOCK = threading.RLock()


def _load(path: Path | str) -> ProjectState:
    """Read a project's settings into a fresh state; a project already open keeps its job runner."""
    project = load_project(path)
    state = ProjectState(project)
    old = _STATES.get(state.id)
    runner = old.__dict__.get("jobs") if old is not None else None
    if runner is not None:
        runner.root, runner.name = project.root, project.name
        state.__dict__["jobs"] = runner
    state.jobs  # start the job runner now, inside the lock, so parallel first requests share one runner
    _STATES[state.id] = state
    return state


def open_project(path: Path | str) -> ProjectState:
    """Open a project (or bring an open one forward): it becomes the project opened last, and is remembered."""
    global _LAST
    with _LOCK:
        state = _load(path)
        _LAST = state.id
        if state.project.is_configured:
            registry.remember(state.root)
        return state


def reload_project(path: Path | str) -> ProjectState:
    """Read an open project's settings and tables again after a change; its job runner carries on."""
    with _LOCK:
        return _load(path)


def get_state() -> ProjectState:
    """The state of the request's project (see `CURRENT`), else of the project opened last."""
    wanted = CURRENT.get()
    with _LOCK:
        if wanted is None:
            if _LAST in _STATES:
                return _STATES[_LAST]
            return open_project(os.environ.get("STAINID_PROJECT", "."))
        if wanted in _STATES:
            return _STATES[wanted]
        start = Path(os.environ.get("STAINID_PROJECT", ".")).resolve()
        root = start if registry.project_id(start) == wanted else registry.find(wanted)
        if root is None or not (root / "stainid.yaml").exists():
            raise UnknownProject(wanted)
        return open_project(root)


def default_id() -> str | None:
    """The project requests without one use: the one opened last, else the one the server was started with."""
    if _LAST in _STATES:
        return _LAST
    start = Path(os.environ.get("STAINID_PROJECT", "."))
    return registry.project_id(start) if (start / "stainid.yaml").exists() else None


@contextmanager
def use_project(project_id: str | None) -> Iterator[None]:
    """Handle what runs inside as a request to this project (None: the project opened last)."""
    token = CURRENT.set(project_id)
    try:
        yield
    finally:
        CURRENT.reset(token)


def open_states() -> list[ProjectState]:
    with _LOCK:
        return list(_STATES.values())


def close_project(project_id: str) -> bool:
    """Let go of an open project whose steps have all ended (True), or keep it open while steps run (False)."""
    global _LAST
    with _LOCK:
        state = _STATES.get(project_id)
        runner = state.__dict__.get("jobs") if state is not None else None
        if runner is not None and runner.active():
            return False
        if runner is not None:
            runner.close()
        _STATES.pop(project_id, None)
        if _LAST == project_id:
            _LAST = None
        return True


def reset() -> None:
    """Close every open project (for tests)."""
    global _LAST
    with _LOCK:
        for state in _STATES.values():
            if "jobs" in state.__dict__:
                state.jobs.close()
        _STATES.clear()
        _LAST = None


# ---------------------------------------------------------------- the projects list

def _open_runner(root: Path):
    state = _STATES.get(registry.project_id(root))
    return state.__dict__.get("jobs") if state is not None else None


def _title(job) -> str:
    from stainid.workflows.steps import BY_ID

    return BY_ID[job.workflow].title if job.workflow in BY_ID else job.workflow


def project_status(root: Path) -> dict:
    """How far a project has got, cheaply (a few file checks and the newest job records): not set up, set up, steps
    running (and which), or results ready."""
    root = Path(root)
    if not (root / "stainid.yaml").exists():
        return {"state": "missing", "text": "Folder not found" if not root.exists() else "No stainid.yaml in this folder", "running": []}
    project = load_project(root)
    runner = _open_runner(root)
    if runner is not None:
        running = [_title(j) for j in list(runner.jobs.values()) if j.status == "running"]
    else:
        running = [_title(j) for j in JobStore(project.output("jobs"), create=False).newest() if j.status == "running" and alive(j.pid, j.started)]
    if running:
        return {"state": "running", "text": "Running: " + ", ".join(dict.fromkeys(running)), "running": running}
    if (project.output("tables") / "results_donor_region.csv").exists():
        return {"state": "results", "text": "Results ready", "running": []}
    if all(project.input(key).exists() for key in ("slides_table", "core_manifest", "tma_layout")):
        return {"state": "set_up", "text": "Set up", "running": []}
    return {"state": "not_set_up", "text": "Not set up yet", "running": []}


def running_elsewhere() -> list[tuple[str, Job]]:
    """Heavy steps running in the projects this server knows but has not opened (command-line runs, or app jobs started
    before a restart), as (project name, job). The open projects report their own."""
    from stainid.workflows.steps import BY_ID

    with _LOCK:
        open_ids = set(_STATES)
    out = []
    for entry in registry.entries():
        if entry.id in open_ids or not entry.exists:
            continue
        store = JobStore(load_project(entry.path).output("jobs"), create=False)
        out += [(entry.name, job) for job in store.newest() if job.status == "running" and job.workflow in BY_ID and BY_ID[job.workflow].heavy
                and alive(job.pid, job.started)]
    return out

