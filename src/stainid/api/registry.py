"""The projects this server knows: the ones opened in the app (remembered in ~/.stainid/app.json) and every folder with a
stainid.yaml in the projects folder (`stainid serve --projects-dir DIR`, or set on the Projects page).

Each project has a stable id made from its folder (`alzheimer-tma-3f9c2b`: the folder name and a short hash of its
path), which the web app puts in its addresses (/p/<id>/...) and API calls (/api/projects/<id>/...). Removing a
project from the list only forgets it; its folder is never touched. The settings file keeps the plain `recent` list
older stainIDs read, and adds when each project was opened last, the projects folder and the projects removed from it.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

from stainid.project import CONFIG_NAME

SETTINGS = Path.home() / ".stainid" / "app.json"
PROJECTS_DIR_ENV = "STAINID_PROJECTS_DIR"
KEEP_RECENT = 50
_LOCK = threading.RLock()


@dataclass
class Entry:
    id: str
    name: str
    path: str
    last_opened: float | None
    in_projects_dir: bool
    exists: bool


def project_id(root: Path | str) -> str:
    """A short, stable id for the project in this folder: its folder name made safe for addresses, and a hash of its path."""
    path = str(Path(root).expanduser().resolve())
    slug = re.sub(r"[^a-z0-9]+", "-", Path(path).name.lower()).strip("-")[:32] or "project"
    return f"{slug}-{hashlib.sha1(path.encode()).hexdigest()[:6]}"


def read_settings() -> dict:
    try:
        settings = json.loads(SETTINGS.read_text()) if SETTINGS.exists() else {}
    except (OSError, ValueError):
        settings = {}
    settings.setdefault("recent", [])
    settings.setdefault("opened", {})
    settings.setdefault("hidden", [])
    return settings


def _write(settings: dict) -> None:
    SETTINGS.parent.mkdir(parents=True, exist_ok=True)
    temporary = SETTINGS.with_name(f".{SETTINGS.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(settings, indent=1))
    temporary.replace(SETTINGS)


def remember(root: Path | str) -> None:
    """Put the project first in the list of recent projects, opened now (and show it again if it had been removed)."""
    path = str(Path(root).resolve())
    with _LOCK:
        settings = read_settings()
        settings["recent"] = [path] + [p for p in settings["recent"] if p != path][: KEEP_RECENT - 1]
        settings["opened"][path] = time.time()
        settings["hidden"] = [p for p in settings["hidden"] if p != path]
        settings["opened"] = {p: t for p, t in settings["opened"].items() if p in settings["recent"]}
        _write(settings)


def add(root: Path | str) -> bool:
    """Put a project on the list without opening it (`stainid projects add`): it goes after the projects listed, with no
    time of opening, so the list's order and the project the app opens last stay as they are. False if it was listed."""
    path = str(Path(root).expanduser().resolve())
    if not (Path(path) / CONFIG_NAME).exists():
        raise FileNotFoundError(f"{path} is not a stainID project (it has no {CONFIG_NAME})")
    with _LOCK:
        settings = read_settings()
        listed = path in settings["recent"] and path not in settings["hidden"]
        if path not in settings["recent"]:
            settings["recent"] = (settings["recent"] + [path])[-KEEP_RECENT:]
        settings["hidden"] = [p for p in settings["hidden"] if p != path]
        _write(settings)
    return not listed


def forget(root: Path | str) -> None:
    """Take the project off the list (a project in the projects folder stays hidden until it is opened again)."""
    path = str(Path(root).resolve())
    with _LOCK:
        settings = read_settings()
        settings["recent"] = [p for p in settings["recent"] if p != path]
        settings["opened"].pop(path, None)
        if path not in settings["hidden"]:
            settings["hidden"].append(path)
        _write(settings)


def projects_dir() -> Path | None:
    """The folder whose subfolders with a stainid.yaml are listed: `--projects-dir` for this run, else the saved setting."""
    value = os.environ.get(PROJECTS_DIR_ENV) or read_settings().get("projects_dir") or ""
    return Path(value).expanduser() if value else None


def set_projects_dir(path: str) -> None:
    with _LOCK:
        settings = read_settings()
        settings["projects_dir"] = str(Path(path).expanduser().resolve()) if path.strip() else ""
        _write(settings)


def recent_projects() -> list[str]:
    """Folders of the recent projects that still exist, most recent first."""
    return [p for p in read_settings()["recent"] if Path(p).exists()]


def remembered_project() -> Path:
    """The project opened last in the app, else the current folder."""
    recent = recent_projects()
    return Path(recent[0]) if recent else Path.cwd()


def _name(root: Path) -> str:
    try:
        config = yaml.safe_load((root / CONFIG_NAME).read_text()) or {}
        return str(config.get("name") or root.name)
    except (OSError, yaml.YAMLError, AttributeError):
        return root.name


def _folder_projects(folder: Path | None) -> list[Path]:
    if folder is None or not folder.is_dir():
        return []
    try:
        children = sorted(os.scandir(folder), key=lambda e: e.name.lower())
    except OSError:
        return []
    return [Path(c.path).resolve() for c in children if c.is_dir() and not c.name.startswith(".") and (Path(c.path) / CONFIG_NAME).exists()]


def entries(extra: list[Path] = ()) -> list[Entry]:
    """Every project the server knows, most recently opened first, then the others by name. `extra`: projects open in
    the server that are not remembered (e.g. the one given with --project before it was set up)."""
    settings = read_settings()
    hidden = set(settings["hidden"])
    folder = projects_dir()
    in_folder = [p for p in _folder_projects(folder) if str(p) not in hidden]
    opened_here = [str(Path(p).resolve()) for p in extra]
    paths = list(dict.fromkeys([*settings["recent"], *(str(p) for p in in_folder), *(p for p in opened_here if p not in hidden)]))
    out = []
    for path in paths:
        root = Path(path)
        exists = (root / CONFIG_NAME).exists()
        out.append(Entry(id=project_id(root), name=_name(root) if exists else root.name, path=path, last_opened=settings["opened"].get(path),
                         in_projects_dir=folder is not None and root.parent == folder.resolve(), exists=exists))
    out.sort(key=lambda e: (e.last_opened is None, -(e.last_opened or 0), e.name.lower()))
    return out


def find(project: str) -> Path | None:
    """The folder of the project with this id, if the server knows it."""
    return next((Path(e.path) for e in entries() if e.id == project), None)


def as_dict(entry: Entry) -> dict:
    return asdict(entry)
