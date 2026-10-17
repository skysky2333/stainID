"""Applying area edits by themselves: at most every `v2.apply_area_edits_minutes` (default 10; 0 never) the app starts
the "Apply area edits" step (`stainid.v2.area_edits`) for each open project that has edits to apply, but only when the
calls exist, no job of the project is queued or running and no detection run is working on the project from outside
the app (a v2 claim touched in the last `BUSY_MINUTES`), so it never interrupts one.
"""
from __future__ import annotations

import os
import threading
import time

BUSY_MINUTES = 15
_started: dict[str, float] = {}
_thread: threading.Thread | None = None


def every_minutes(project) -> float:
    return float((project.config.get("v2") or {}).get("apply_area_edits_minutes", 10))


def detection_busy(project, now: float | None = None) -> bool:
    """A v2 stage is working on this project from outside the app (its claims were touched lately)."""
    folder = project.output("v2") / "claims"
    now = time.time() if now is None else now
    try:
        with os.scandir(folder) as found:
            return any(now - entry.stat().st_mtime < BUSY_MINUTES * 60 for entry in found)
    except OSError:
        return False


def due(state, now: float | None = None) -> bool:
    """Whether to start "Apply area edits" for this open project now."""
    from stainid.v2 import area_edits  # noqa: PLC0415

    project = state.project
    minutes = every_minutes(project)
    now = time.time() if now is None else now
    if minutes <= 0 or now - _started.get(str(project.root), 0.0) < minutes * 60:
        return False
    if not (project.output("v2") / "manifests" / "calls").exists() or state.jobs.active() or detection_busy(project, now):
        return False
    return area_edits.status(project)["cores"] > 0


def tick(now: float | None = None) -> list[str]:
    """Start the step where it is due; the projects it was started for."""
    from stainid.api import state as states  # noqa: PLC0415

    started = []
    for state in states.open_states():
        try:
            if due(state, now):
                state.jobs.submit("v2_area_edits", {})
                _started[str(state.project.root)] = time.time() if now is None else now
                started.append(state.project.name)
        except Exception as error:  # noqa: BLE001  (a project that cannot be looked at now is tried again next time)
            print(f"apply area edits: {state.project.name}: {error}", flush=True)
    return started


def start(period_s: float = 60.0) -> None:
    global _thread
    if _thread is not None:
        return

    def loop() -> None:
        while True:
            time.sleep(period_s)
            tick()

    _thread = threading.Thread(target=loop, name="apply area edits", daemon=True)
    _thread.start()
