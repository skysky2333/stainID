"""Background pipeline jobs: the server's queue over the shared job store (`stainid.jobs`).

The server starts `stainid` commands as subprocesses (through `stainid.jobs.run`, which records each exit code) with a
cap on heavy steps at once. Steps someone runs from the command line register in the same store; the server picks
them up every few seconds, so they show in the app, count toward the cap and make conflicting queued jobs wait.
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from stainid.jobs import ACTIVE, APP, COMMAND_LINE, JOB_ENV, Job, JobStore, alive, new_id, stop
from stainid.project import load_project
from stainid.workflows.steps import BY_ID, RESOURCES, STEPS, resource_info, to_argv, wait_reason

RESCAN_SECONDS = 3.0


class JobManager:
    def __init__(self, project_root: Path, directory: Path, max_heavy: int = 2):
        self.root, self.max_heavy = project_root, max_heavy
        self.store = JobStore(directory)
        self.dir = self.store.dir
        self.lock = threading.Lock()
        self.processes: dict[str, subprocess.Popen] = {}
        self.jobs = {j.id: j for j in self.store.all()}
        self.seen = {job_id: self._mtime(job_id) for job_id in self.jobs}
        self.scanned = time.time()
        for job in self.jobs.values():
            if job.status == "queued" and job.workflow not in BY_ID:
                job.status = "cancelled"
                self._save(job)
            if job.status == "running" and job.source == APP and self.store.exit_code(job.id) is None and not alive(job.pid, job.started):
                job.status, job.outcome = "interrupted", self.progress(job)  # stopped with the server, before it could finish
                self._save(job)
        self._reap()  # jobs that ended while the server was away get their real exit code; the others are followed again
        self.open = True
        threading.Thread(target=self._loop, daemon=True).start()

    def _save(self, job: Job) -> None:
        self.store.save(job)
        self.seen[job.id] = self._mtime(job.id)

    def _mtime(self, job_id: str) -> float:
        try:
            return self.store.path(job_id).stat().st_mtime
        except OSError:
            return 0.0

    def active(self) -> bool:
        return any(j.status in ACTIVE for j in self.jobs.values())

    def log_path(self, job_id: str) -> Path | None:
        job = self.jobs.get(job_id)
        return self.store.log_path(job) if job else None

    def submit(self, workflow: str, options: dict) -> Job:
        if workflow not in BY_ID or BY_ID[workflow].command is None:
            raise KeyError(workflow)
        options = {**{o.key: o.default for o in BY_ID[workflow].options}, **options}
        job = Job(id=new_id(), workflow=workflow, options=options, argv=to_argv(workflow, options))
        job.log_file = str(self.store.dir / f"{job.id}.log")
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
                stop(job.pid)
                job.status = "cancelled"
            if job.source != COMMAND_LINE:  # a command-line run records how it ended itself (it is still writing)
                self._save(job)
            self._skip_dependents(job, "was stopped")
            return job

    def progress(self, job: Job) -> dict:
        return self.store.progress(job)

    def describe(self, job: Job) -> dict:
        progress = job.outcome or self.progress(job)
        return {**job.__dict__, "title": BY_ID[job.workflow].title if job.workflow in BY_ID else job.workflow, "progress": progress}

    def close(self) -> None:
        self.open = False

    def _loop(self) -> None:
        while self.open:
            with self.lock:
                if time.time() - self.scanned >= RESCAN_SECONDS:
                    self._rescan()
                self._reap()
                self._schedule()
            time.sleep(1.0)

    def _rescan(self) -> None:
        """Pick up steps run from the command line: new records, and changes to the ones already known (they ended)."""
        self.scanned = time.time()
        try:
            names = [n for n in os.listdir(self.dir) if n.endswith(".json") and not n.startswith(".")]
        except OSError:
            return
        for name in names:
            job_id = name[:-5]
            known = self.jobs.get(job_id)
            if known is not None and known.source != COMMAND_LINE:
                continue
            mtime = self._mtime(job_id)
            if mtime == self.seen.get(job_id):
                continue
            self._reload(job_id, known)

    def _reload(self, job_id: str, known: Job | None) -> Job | None:
        """Take a command-line run's record as it is on disk now; when it has just ended, skip what waited for it."""
        job = self.store.load(self.store.path(job_id))
        if job is None or job.source != COMMAND_LINE:
            return None
        self.jobs[job_id], self.seen[job_id] = job, self._mtime(job_id)
        if known is not None and known.status == "running" and job.status in ("failed", "cancelled"):
            self._skip_dependents(job, "failed" if job.status == "failed" else "was stopped")
        return job

    def _reap(self) -> None:
        for job_id, process in list(self.processes.items()):
            code = process.poll()
            if code is not None:
                recorded = self.store.exit_code(job_id)
                self._finish(self.jobs[job_id], code if recorded is None else recorded)
                del self.processes[job_id]
        for job in list(self.jobs.values()):
            if job.status != "running" or job.id in self.processes or job.pid is None:
                continue
            if job.source == APP and (code := self.store.exit_code(job.id)) is not None:
                self._finish(job, code)  # an adopted job (started before the server restarted) that has ended
            elif alive(job.pid, job.started):
                continue
            elif job.source == COMMAND_LINE:
                job = self._reload(job.id, job) or job  # it may have just recorded how it ended
                if job.status == "running":  # ended without saying how (killed, or the computer restarted)
                    job.status, job.ended, job.outcome = "interrupted", time.time(), self.progress(job)
                    self._save(job)
                    self._skip_dependents(job, "was stopped")
            else:  # started before exit codes were recorded, or its runner was killed
                self._finish(job, 1 if self.progress(job)["error"] else 0)

    def _finish(self, job: Job, code: int) -> None:
        job.returncode, job.ended, job.outcome = code, time.time(), self.progress(job)
        if job.status != "cancelled":
            job.status = "finished" if code == 0 else "failed"
        self._save(job)
        if job.status == "failed":
            self._skip_dependents(job, "failed")

    def _skip_dependents(self, job: Job, what: str) -> None:
        """Queued jobs that were waiting for this one cannot do anything useful: skip them (and whatever waits for them).
        A command-line run is waited for by every queued job it conflicts with, so it is checked against all of them."""
        if job.workflow not in BY_ID:
            return
        for other in sorted(self.jobs.values(), key=lambda j: j.created):
            if (other.status == "queued" and (other.created > job.created or job.source == COMMAND_LINE)
                    and wait_reason(BY_ID[job.workflow], BY_ID[other.workflow])):
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
        writes the same file (see `wait_reason`); parts of one step with different options run side by side. A step
        running from the command line counts as earlier than every queued job. Heavy jobs also wait for a free slot."""
        active = [j for j in self.jobs.values() if j.status in ACTIVE and j.workflow in BY_ID]
        running = [j for j in active if j.status == "running"]
        running_heavy = sum(BY_ID[j.workflow].heavy for j in running)
        outside = sum(BY_ID[j.workflow].heavy for j in running if j.source == COMMAND_LINE)
        for job in sorted((j for j in active if j.status == "queued"), key=lambda j: j.created):
            step = BY_ID[job.workflow]
            reasons = [wait_reason(BY_ID[j.workflow], step, "command line" if j.source == COMMAND_LINE else "") for j in active
                       if j is not job and (j.created < job.created or j.source == COMMAND_LINE)
                       and not (j.workflow == job.workflow and j.options != job.options)]
            reasons = [r for r in reasons if r]
            if reasons:
                job.waiting = reasons[0]
            elif step.heavy and running_heavy >= self.max_heavy:
                job.waiting = (f"Waits for a free slot: {running_heavy} heavy steps are already running"
                               + (f", {outside} of them from the command line" if outside else "") + f" (at most {self.max_heavy} at a time)")
            elif missing := self._missing(step):
                self._skip(job, missing)
            else:
                job.waiting = ""
                self._start(job)
                running_heavy += step.heavy

    def _start(self, job: Job) -> None:
        command = [sys.executable, "-u", "-m", "stainid.cli", "--project", str(self.root), *job.argv]
        log = Path(job.log_file or self.dir / f"{job.id}.log").open("ab")
        self.store.exit_path(job.id).unlink(missing_ok=True)
        process = subprocess.Popen([sys.executable, "-m", "stainid.jobs.run", str(self.store.exit_path(job.id)), "--", *command],
                                   cwd=self.root, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                   env={**os.environ, JOB_ENV: job.id})
        log.close()
        self.processes[job.id] = process
        job.status, job.pid, job.started = "running", process.pid, time.time()
        self._save(job)
