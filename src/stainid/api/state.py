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

from stainid import cache, disk_cache
from stainid.api import registry
from stainid.jobs import Job, JobStore, alive
from stainid.project import Project, load_project
from stainid.sampling.tiles import known_tiles, labelling_tiles, table

CURRENT: ContextVar[str | None] = ContextVar("stainid_project", default=None)


class UnknownProject(KeyError):
    """A request named a project this server does not know."""


def _read(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, **kwargs) if path.exists() else pd.DataFrame()


def _table(path: Path) -> pd.DataFrame:
    """A project table, read again only when the file changes (shared between requests: never change it in place)."""
    return table(path)


def _count(folder: Path, names) -> int:
    """How many of these file names are in the folder (one listing, see `stainid.cache.names`)."""
    existing = cache.names(folder)
    return sum(name in existing for name in names)


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
        """The folder of one kind of cache (see `stainid.disk_cache`: record what is written and read there)."""
        return disk_cache.folder(self.root, name)

    @property
    def tiles(self) -> pd.DataFrame:
        """Every tile a label, session or viewer may name (see `stainid.sampling.tiles.known_tiles`)."""
        return known_tiles(self.project)

    @property
    def analysis_tiles(self) -> pd.DataFrame:
        """The tiles detection runs on: every tissue tile of each core."""
        return _table(self.project.analysis_tiles())

    @property
    def labelling_tiles(self) -> pd.DataFrame:
        """The tiles new labelling fields are drawn from."""
        return labelling_tiles(self.project)

    @property
    def core_images(self) -> pd.DataFrame:
        return _table(self.project.input("core_images"))

    @property
    def layout(self) -> pd.DataFrame:
        return _table(self.project.input("tma_layout"))

    @property
    def donors(self) -> pd.DataFrame:
        return _table(self.project.input("donor_metadata"))

    def groups(self) -> list[dict[str, str]]:
        """Diagnostic groups in plot order: from the project settings, else as found in the data."""
        configured = self.project.config.get("groups") or []
        tiles = self.analysis_tiles
        found = sorted(set(tiles.disease_group.dropna()) if "disease_group" in tiles else set())
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
        """How many cores / tiles each pipeline stage has finished."""
        tiles = self.analysis_tiles
        fields_parts = self.project.output("fields") / "parts"
        masks = self.project.output("masks")
        status = {}
        for stain in sorted(tiles.stain.unique()) if not tiles.empty else []:
            stain_tiles = tiles[tiles.stain == stain]
            cores = set(stain_tiles.core_id)
            if stain == "NeuN":
                done = _count(self.project.output("neun") / "tiles", (f"{t}_features.csv" for t in stain_tiles.tile_id))
                detection = {"done": done, "total": len(stain_tiles), "unit": "tiles"}
            else:
                done = _count(fields_parts, (f"{c}_{stain}_features.csv" for c in cores))
                detection = {"done": done, "total": len(cores), "unit": "cores"}
            nuclei = _count(self.project.output("nuclei"), (f"{t}.npz" for t in stain_tiles.tile_id))
            mask_done = _count(masks / stain / "parts", (f"{c}_{stain}_objects.csv" for c in cores))
            status[stain] = {"detection": detection, "nuclei": {"done": nuclei, "total": len(stain_tiles), "unit": "tiles"},
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
        new = registry.project_id(path) not in _STATES
        state = _load(path)
        _LAST = state.id
        if state.project.is_configured:
            registry.remember(state.root)
            if new:
                disk_cache.watch(state.root)
                disk_cache.know(*registry.recent_projects())  # a drive low on space takes from every project on it
                threading.Thread(target=_warm_up, args=(state,), daemon=True).start()
        return state


def _warm_up(state: ProjectState) -> None:
    """Read what the step overview and the labels page need once, in the background, so the first page to ask for them
    does not wait for the big tables to be read (see `stainid.cache`)."""
    from stainid.labels import overview
    from stainid.workflows.steps import describe

    try:
        describe(state.project)
        for stain in sorted(set(state.tiles.stain)) if "stain" in state.tiles else []:
            overview.cached_stain_overview(state.project, stain, state.project.output("reviews"), state.tiles)
    except Exception:  # a half-set-up project: the page that asks will say what is wrong
        pass


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
    needed = ("slides_table", "core_manifest", "tma_layout") if project.is_tma else ("image_list", "whole_core_manifest")
    if all(project.input(key).exists() for key in needed):
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

