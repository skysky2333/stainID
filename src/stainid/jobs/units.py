"""Units of work: where a job may stop when it is paused, pre-empted or cancelled, and pick up again later.

Every job works in units (a tile, a core image and stain, a slide, a file) and writes each unit's outputs atomically
(and, for the v2 stages, its manifest), so a run that starts again skips the units that are done. Between two units a
job calls `boundary()`: when the queue has asked it to stop (`stainid.jobs.queue`: the job's control file says "pause"
or "cancel"), it ends there with exit code `PAUSED` and loses nothing. Work that runs in other processes or threads
calls `stopping()` instead and simply does not start another unit.

Outside the queue (a test, a command typed without the queue) there is no control file and both are free.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

CONTROL_ENV = "STAINID_JOB_CONTROL"  # the file the queue writes "pause" or "cancel" into for this job
PAUSED = 75  # exit code of a job that stopped at a unit boundary because it was asked to (EX_TEMPFAIL: run it again later)
CHECK_SECONDS = 1.0

_seen: tuple[float, str] = (0.0, "")


class Paused(SystemExit):
    """Raised by `boundary()`: the job stops here, between two units, with exit code `PAUSED`."""

    def __init__(self, request: str):
        super().__init__(PAUSED)
        self.request = request


def control_file() -> Path | None:
    value = os.environ.get(CONTROL_ENV)
    return Path(value) if value else None


def requested(fresh: bool = True) -> str:
    """What the queue asks of this job now: "pause", "cancel" or "" (with `fresh=False` looked at no more than once a
    second, for checks between small steps)."""
    global _seen
    path = control_file()
    if path is None:
        return ""
    now = time.monotonic()
    if not fresh and now - _seen[0] < CHECK_SECONDS:
        return _seen[1]
    try:
        value = path.read_text().strip()
    except OSError:
        value = ""
    _seen = (now, value if value in ("pause", "cancel") else "")
    return _seen[1]


def stopping() -> bool:
    """Whether to start no more units (for worker threads and processes; the main loop then calls `boundary()`)."""
    return bool(requested(fresh=False))


def boundary(done: str = "") -> None:
    """Between two units: stop here (exit code `PAUSED`) when asked to; `done` says what was finished, for the log."""
    request = requested()
    if request:
        print(f"Stopping between two units ({request} requested){': ' + done if done else ''}; done units are kept and "
              "a new run continues from here", flush=True)
        raise Paused(request)


SINCE_ENV = "STAINID_JOB_SINCE"  # set on a job's later runs: when its first run started


def since() -> float | None:
    """When this job's first run started, on a later run (after a pause or a restart): a step asked to redo everything
    leaves what it redid since then as it is."""
    try:
        return float(os.environ[SINCE_ENV])
    except (KeyError, ValueError):
        return None


def drain(futures) -> None:
    """When asked to stop: cancel the units not started yet (those running finish; the caller then calls `boundary()`)."""
    if stopping():
        for future in futures:
            future.cancel()


def forget() -> None:
    """Look at the control file again on the next call (tests)."""
    global _seen
    _seen = (0.0, "")
