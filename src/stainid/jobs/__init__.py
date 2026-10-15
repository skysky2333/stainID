"""The job store: one JSON record (and usually one log) per run of a workflow step, in the project's jobs folder.

The web server and the command line share it. The server writes the jobs it starts (source "app"); a workflow command
typed in a terminal registers itself (source "command line"), so both kinds show in one list in the app and wait for
each other. Records are replaced atomically, so a reader never sees a half-written one.
"""
from __future__ import annotations

import functools
import json
import os
import re
import signal
import stat
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from stainid.tables import write_text_atomic

JOB_ENV = "STAINID_JOB_ID"  # set by the server for the commands it starts, so they do not register a second time
APP, COMMAND_LINE = "app", "command line"
ACTIVE = ("running", "queued")
PROGRESS = re.compile(r"\[(\d+)/(\d+)\]")
ERROR = re.compile(r"^[A-Za-z_.]*(Error|Exception): ")
NOISE = re.compile(r"Warning|Matplotlib is building|^\s*warnings\.warn|SLF4J")


@dataclass
class Job:
    id: str
    workflow: str
    options: dict
    argv: list[str]
    status: str = "queued"
    created: float = field(default_factory=time.time)
    started: float | None = None
    ended: float | None = None
    pid: int | None = None
    returncode: int | None = None
    waiting: str = ""
    outcome: dict = field(default_factory=dict)
    source: str = APP
    log_file: str = ""


def new_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]


class JobStore:
    """Job records in one folder: `<id>.json`, plus `<id>.log` (output of app jobs) and `<id>.exit` (their exit code)."""

    def __init__(self, directory: Path, create: bool = True):
        """`create=False` only reads: the folder is not made when it is missing (a project nothing has run in)."""
        self.dir = Path(directory)
        if create:
            self.dir.mkdir(parents=True, exist_ok=True)

    def path(self, job_id: str) -> Path:
        return self.dir / f"{job_id}.json"

    def exit_path(self, job_id: str) -> Path:
        return self.dir / f"{job_id}.exit"

    def log_path(self, job: Job) -> Path | None:
        """Where the job's output goes: its own log for app jobs, the file a command-line run printed to (if any)."""
        if job.log_file:
            return Path(job.log_file)
        return self.dir / f"{job.id}.log" if job.source == APP else None

    def save(self, job: Job) -> None:
        write_text_atomic(self.path(job.id), json.dumps(asdict(job)))

    def load(self, path: Path) -> Job | None:
        """A record, or None for a file that is not one (unreadable, or from a newer stainID with other required fields)."""
        try:
            data = json.loads(path.read_text())
            known = {f.name for f in fields(Job)}
            return Job(**{k: v for k, v in data.items() if k in known})
        except (OSError, ValueError, TypeError):
            return None

    def all(self) -> list[Job]:
        paths = (self.dir / name for name in os.listdir(self.dir) if name.endswith(".json") and not name.startswith("."))
        return [job for job in map(self.load, paths) if job is not None]

    def newest(self, count: int = 40) -> list[Job]:
        """The newest records (ids start with the time they were made), for a quick look at what is running."""
        try:
            names = sorted((n for n in os.listdir(self.dir) if n.endswith(".json") and not n.startswith(".")), reverse=True)[:count]
        except OSError:
            return []
        return [job for job in (self.load(self.dir / name) for name in names) if job is not None]

    def exit_code(self, job_id: str) -> int | None:
        try:
            return int(self.exit_path(job_id).read_text())
        except (OSError, ValueError):
            return None

    def progress(self, job: Job) -> dict:
        path = self.log_path(job)
        return progress(path) if path is not None else {"done": 0, "total": 0, "last_line": "", "error": ""}


def progress(path: Path) -> dict:
    """How far a step has got, from the `[i/n]` lines it prints, its last line and the error it ended with."""
    try:
        with path.open("rb") as handle:
            handle.seek(max(0, handle.seek(0, os.SEEK_END) - 20000))
            tail = handle.read().decode(errors="replace").splitlines()
    except OSError:
        return {"done": 0, "total": 0, "last_line": "", "error": ""}
    lines = [line for line in tail if line.strip() and not NOISE.search(line)]
    last = next((m for m in map(PROGRESS.search, reversed(lines)) if m), None)
    error = ERROR.sub("", next((line for line in reversed(lines) if ERROR.match(line)), ""))
    return {"done": int(last.group(1)) if last else 0, "total": int(last.group(2)) if last else 0, "last_line": lines[-1] if lines else "",
            "error": error}


def alive(pid: int | None, since: float | None = None) -> bool:
    """Whether the process still runs. With `since` (when the job started), a process that started clearly later is a
    different one that got the same number (after a restart of the computer); this is checked where Linux tells."""
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    started = _process_start(pid) if since else None
    return started is None or started <= since + 60


@functools.cache
def _boot_time() -> int:
    """When the computer started (Linux), read once: /proc/stat is slow to read on machines with many processors."""
    return next(int(line.split()[1]) for line in Path("/proc/stat").read_text().splitlines() if line.startswith("btime"))


def _process_start(pid: int) -> float | None:
    try:
        ticks = int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19])
        return _boot_time() + ticks / os.sysconf("SC_CLK_TCK")
    except (OSError, ValueError, IndexError, StopIteration):
        return None


def stop(pid: int) -> None:
    """Stop a job's process and everything it started: its whole process group when it leads one (the server's jobs, a
    command typed in a terminal), else only the process (one of several commands started by a script), and never the
    group of the caller itself."""
    try:
        group = os.getpgid(pid)
        if group == pid and group != os.getpgrp():
            os.killpg(group, signal.SIGTERM)
        else:
            os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass


def output_file() -> str:
    """The file this process prints to, when its output is redirected to a regular file (`stainid ... > run.log`)."""
    try:
        if not stat.S_ISREG(os.fstat(sys.stdout.fileno()).st_mode):
            return ""
        if sys.platform == "darwin":
            import fcntl

            return os.fsdecode(fcntl.fcntl(sys.stdout.fileno(), fcntl.F_GETPATH, bytes(1024)).rstrip(b"\0"))
        return os.readlink(f"/proc/self/fd/{sys.stdout.fileno()}")
    except (OSError, ValueError, AttributeError):
        return ""


@contextmanager
def command_line_job(store: JobStore, workflow: str, options: dict, argv: list[str]):
    """Record a step run from the command line for as long as it runs, and how it ended: finished, failed (an error),
    or stopped (Ctrl+C, or stopped from the app)."""
    log_file = output_file()
    job = Job(id=new_id(), workflow=workflow, options=options, argv=argv, status="running", started=time.time(), pid=os.getpid(),
              source=COMMAND_LINE, log_file=log_file)
    store.save(job)
    os.environ[JOB_ENV] = job.id  # commands this one starts belong to this job

    def end(status: str, code: int, error: str = "") -> None:
        if os.getpid() != job.pid or job.ended:  # a forked worker, or already recorded
            return
        sys.stdout.flush()
        job.status, job.returncode, job.ended = status, code, time.time()
        job.outcome = progress(Path(log_file)) if log_file else {"done": 0, "total": 0, "last_line": error, "error": error}
        if error and not job.outcome["error"]:
            job.outcome["error"] = error
        store.save(job)

    def terminated(signum, frame):
        end("cancelled", 128 + signum)
        signal.signal(signum, signal.SIG_DFL)  # then stop exactly as before: at once, without unwinding
        os.kill(os.getpid(), signum)

    previous = signal.getsignal(signal.SIGTERM) if threading.current_thread() is threading.main_thread() else None
    if previous is signal.SIG_DFL:
        signal.signal(signal.SIGTERM, terminated)
    try:
        yield job
    except KeyboardInterrupt:
        end("cancelled", 130, "Stopped with Ctrl+C")
        raise
    except SystemExit as exit_:
        code = exit_.code if isinstance(exit_.code, int) else (0 if exit_.code is None else 1)
        end("finished" if code == 0 else "failed", code, "" if code == 0 else str(exit_.code))
        raise
    except BaseException as error:
        end("failed", 1, ERROR.sub("", f"{type(error).__name__}: {error}"))
        raise
    else:
        end("finished", 0)
    finally:
        if previous is signal.SIG_DFL:
            signal.signal(signal.SIGTERM, previous)
