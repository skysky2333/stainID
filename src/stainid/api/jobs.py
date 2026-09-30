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

from stainid.workflows.steps import BY_ID, to_argv

PROGRESS = re.compile(r"\[(\d+)/(\d+)\]")
ERROR = re.compile(r"^[A-Za-z_.]*(Error|Exception): ")

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


class JobManager:
    def __init__(self, project_root: Path, directory: Path, max_heavy: int = 2):
        self.root, self.dir, self.max_heavy = project_root, directory, max_heavy
        self.dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.processes: dict[str, subprocess.Popen] = {}
        self.jobs = {j.id: j for j in (self._load(p) for p in self.dir.glob("*.json"))}
        for job in self.jobs.values():
            if job.status in ("running", "queued") and not (job.pid and _alive(job.pid)):
                job.status = "interrupted" if job.status == "running" else job.status
            if job.status == "queued" and job.workflow not in BY_ID:
                job.status = "cancelled"
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
            return job

    def progress(self, job: Job) -> dict:
        path = self.log_path(job.id)
        if not path.exists():
            return {"done": 0, "total": 0, "last_line": ""}
        tail = path.read_bytes()[-20000:].decode(errors="replace").splitlines()
        lines = [line for line in tail if line.strip() and "Warning" not in line]
        matches = [PROGRESS.search(line) for line in lines]
        last = next((m for m in reversed(matches) if m), None)
        error = ERROR.sub("", next((line for line in reversed(lines) if ERROR.match(line)), ""))
        return {"done": int(last.group(1)) if last else 0, "total": int(last.group(2)) if last else 0, "last_line": lines[-1] if lines else "",
                "error": error}

    def describe(self, job: Job) -> dict:
        return {**asdict(job), "title": BY_ID[job.workflow].title if job.workflow in BY_ID else job.workflow, "progress": self.progress(job)}

    def _loop(self) -> None:
        while True:
            with self.lock:
                for job_id, process in list(self.processes.items()):
                    code = process.poll()
                    if code is not None:
                        job = self.jobs[job_id]
                        job.returncode, job.ended = code, time.time()
                        if job.status != "cancelled":
                            job.status = "finished" if code == 0 else "failed"
                        self._save(job)
                        del self.processes[job_id]
                running_heavy = sum(BY_ID[self.jobs[j].workflow].heavy for j in self.processes)
                for job in sorted((j for j in self.jobs.values() if j.status == "queued"), key=lambda j: j.created):
                    heavy = BY_ID[job.workflow].heavy
                    if heavy and running_heavy >= self.max_heavy:
                        continue
                    self._start(job)
                    running_heavy += heavy
            time.sleep(1.0)

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
