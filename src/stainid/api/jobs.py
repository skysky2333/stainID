"""Background pipeline jobs: `stainid` CLI subprocesses with a concurrency cap, persistent logs and progress parsing."""
from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

from stainid.project import load_project
from stainid.workflows.steps import BY_ID, RESOURCES, STEPS, resource_info, to_argv, wait_reason

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


class JobManager:
    def __init__(self, project_root: Path, directory: Path, max_heavy: int = 2):
        self.root, self.dir, self.max_heavy = project_root, directory, max_heavy
        self.dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.processes: dict[str, subprocess.Popen] = {}
        self.jobs = {j.id: j for j in (self._load(p) for p in self.dir.glob("*.json"))}
        for job in self.jobs.values():
            if job.status == "running" and not _alive(job.pid):
                job.status, job.outcome = "interrupted", self.progress(job)
            if job.status == "queued" and job.workflow not in BY_ID:
                job.status = "cancelled"
        self.open = True
        threading.Thread(target=self._loop, daemon=True).start()

    def _load(self, path: Path) -> Job:
        return Job(**json.loads(path.read_text()))

    def _save(self, job: Job) -> None:
        (self.dir / f"{job.id}.json").write_text(json.dumps(asdict(job)))

    def active(self) -> bool:
        return any(j.status in ("running", "queued") for j in self.jobs.values())

    def log_path(self, job_id: str) -> Path:
        return self.dir / f"{job_id}.log"

    def submit(self, workflow: str, options: dict) -> Job:
        if workflow not in BY_ID or BY_ID[workflow].command is None:
            raise KeyError(workflow)
        options = {**{o.key: o.default for o in BY_ID[workflow].options}, **options}
        job = Job(id=time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6], workflow=workflow, options=options, argv=to_argv(workflow, options))
        with self.lock:
            self.jobs[job.id] = job
            self._save(job)
        return job

    def cancel(self, job_id: str) -> Job:
        with self.lock:
            job = self.jobs[job_id]
            if job.status == "queued":
                job.status = "cancelled"
            elif job.status == "running" and job.pid:
                os.killpg(os.getpgid(job.pid), signal.SIGTERM)
                job.status = "cancelled"
            self._save(job)
            self._skip_dependents(job, "was stopped")
            return job

    def progress(self, job: Job) -> dict:
        path = self.log_path(job.id)
        if not path.exists():
            return {"done": 0, "total": 0, "last_line": "", "error": ""}
        tail = path.read_bytes()[-20000:].decode(errors="replace").splitlines()
        lines = [line for line in tail if line.strip() and not NOISE.search(line)]
        matches = [PROGRESS.search(line) for line in lines]
        last = next((m for m in reversed(matches) if m), None)
        error = ERROR.sub("", next((line for line in reversed(lines) if ERROR.match(line)), ""))
        return {"done": int(last.group(1)) if last else 0, "total": int(last.group(2)) if last else 0, "last_line": lines[-1] if lines else "",
                "error": error}

    def describe(self, job: Job) -> dict:
        progress = job.outcome or self.progress(job)
        return {**asdict(job), "title": BY_ID[job.workflow].title if job.workflow in BY_ID else job.workflow, "progress": progress}

    def close(self) -> None:
        self.open = False

    def _loop(self) -> None:
        while self.open:
            with self.lock:
                self._reap()
                self._schedule()
            time.sleep(1.0)

    def _reap(self) -> None:
        for job_id, process in list(self.processes.items()):
            code = process.poll()
            if code is not None:
                self._finish(self.jobs[job_id], code)
                del self.processes[job_id]
        for job in self.jobs.values():
            if job.status == "running" and job.id not in self.processes and not _alive(job.pid):
                self._finish(job, 1 if self.progress(job)["error"] else 0)

    def _finish(self, job: Job, code: int) -> None:
        job.returncode, job.ended, job.outcome = code, time.time(), self.progress(job)
        if job.status != "cancelled":
            job.status = "finished" if code == 0 else "failed"
        self._save(job)
        if job.status == "failed":
            self._skip_dependents(job, "failed")

    def _skip_dependents(self, job: Job, what: str) -> None:
        """Queued jobs that were waiting for this one cannot do anything useful: skip them (and whatever waits for them)."""
        for other in sorted(self.jobs.values(), key=lambda j: j.created):
            if other.status == "queued" and other.created > job.created and wait_reason(BY_ID[job.workflow], BY_ID[other.workflow]):
                self._skip(other, f"Not started because “{BY_ID[job.workflow].title}” {what}")

    def _skip(self, job: Job, reason: str) -> None:
        job.status, job.waiting, job.ended = "skipped", "", time.time()
        job.outcome = {"done": 0, "total": 0, "last_line": reason, "error": reason}
        self._save(job)
        self._skip_dependents(job, "was skipped")

    def _missing(self, step) -> str:
        project = load_project(self.root)
        missing = [r for r in step.needs if not resource_info(project, r)["exists"]]
        if not missing:
            return ""
        makers = {r: next((s.title for s in STEPS if r in s.produces), "") for r in missing}
        return "Not started: missing " + "; ".join(f"{RESOURCES[r].label}" + (f" (made by “{m}”)" if m else "") for r, m in makers.items())

    def _schedule(self) -> None:
        """Start queued jobs oldest first. A job waits while an earlier running or waiting job makes something it reads or
        writes the same file (see `wait_reason`); parts of one step with different options run side by side. Heavy jobs also
        wait for a free slot."""
        active = [j for j in self.jobs.values() if j.status in ("running", "queued")]
        running_heavy = sum(BY_ID[j.workflow].heavy for j in active if j.status == "running")
        for job in sorted((j for j in active if j.status == "queued"), key=lambda j: j.created):
            step = BY_ID[job.workflow]
            reasons = [wait_reason(BY_ID[j.workflow], step) for j in active if j.created < job.created and j.status in ("running", "queued")
                       and not (j.workflow == job.workflow and j.options != job.options)]
            reasons = [r for r in reasons if r]
            if reasons:
                job.waiting = reasons[0]
            elif step.heavy and running_heavy >= self.max_heavy:
                job.waiting = f"Waits for a free slot: {self.max_heavy} heavy steps are already running (at most {self.max_heavy} at a time)"
            elif missing := self._missing(step):
                self._skip(job, missing)
            else:
                job.waiting = ""
                self._start(job)
                running_heavy += step.heavy

    def _start(self, job: Job) -> None:
        log = self.log_path(job.id).open("ab")
        process = subprocess.Popen([sys.executable, "-u", "-m", "stainid.cli", "--project", str(self.root), *job.argv],
                                   cwd=self.root, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        self.processes[job.id] = process
        job.status, job.pid, job.started = "running", process.pid, time.time()
        self._save(job)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True
