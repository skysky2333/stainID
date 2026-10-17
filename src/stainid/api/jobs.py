"""One project's view of the shared job queue (`stainid.jobs.queue`).

The queue is one for the whole computer: every project's steps, the command line's (`stainid jobs submit`) and the
runs typed in a terminal (which register themselves). The app runs it (`Scheduler.start_thread`) unless another
process (`stainid jobs serve`) already does; a project's pages see and act on that project's jobs through a
`JobManager`. Records written by earlier versions in the project's own jobs folder stay readable as history; jobs
that were still waiting there move into the shared queue.
"""
from __future__ import annotations

from pathlib import Path

from stainid.jobs import ACTIVE, Job, JobStore
from stainid.jobs import queue as job_queue


class JobManager:
    def __init__(self, project_root: Path, directory: Path | None = None, name: str = "", scheduler: job_queue.Scheduler | None = None):
        """`directory`: the project's own (older) jobs folder, read for history."""
        from stainid.project import load_project

        self.root, self.name = Path(project_root).resolve(), name or Path(project_root).name
        self.scheduler = scheduler or job_queue.shared()
        self.scheduler.start_thread()
        self.open = True
        self.legacy: dict[str, Job] = {}
        if directory is not None and Path(directory).is_dir():
            old = JobStore(Path(directory), create=False)
            try:
                project = load_project(self.root)
            except Exception:  # noqa: BLE001 - history only
                project = None
            for job in old.all():
                if job.status == "queued" and project is not None:  # waiting under the old queue: the shared queue runs it now
                    try:
                        self.scheduler.submit_step(project, job.workflow, job.options)
                    except KeyError:
                        pass
                    old.path(job.id).unlink(missing_ok=True)
                    continue
                if job.status == "running":
                    job.status = "interrupted"  # an older app's job: its record no longer changes
                self.legacy[job.id] = job

    @property
    def store(self) -> JobStore:
        return self.scheduler.store

    @property
    def jobs(self) -> dict[str, Job]:
        mine = str(self.root)
        with self.scheduler.lock:
            found = {j.id: j for j in self.scheduler.jobs.values() if j.project == mine}
        return {**self.legacy, **found}

    def active(self) -> bool:
        return any(j.status in ACTIVE for j in self.jobs.values())

    def log_path(self, job_id: str) -> Path | None:
        job = self.jobs.get(job_id)
        if job is None:
            return None
        if job.log_file:
            return Path(job.log_file)
        return self.store.log_path(job)

    def submit(self, workflow: str, options: dict, klass: str | None = None) -> Job:
        from stainid.project import load_project

        return self.scheduler.submit_step(load_project(self.root), workflow, options, klass)  # started on the queue's next round

    def act(self, op: str, job_id: str, **args) -> Job:
        if job_id not in self.jobs:
            raise KeyError(job_id)
        self.scheduler.request(op, job_id, **args)
        return self.jobs[job_id]

    def cancel(self, job_id: str) -> Job:
        return self.act("cancel", job_id)

    def progress(self, job: Job) -> dict:
        return self.store.progress(job)

    def describe(self, job: Job) -> dict:
        if job.id in self.legacy:
            progress = job.outcome or {"done": 0, "total": 0, "last_line": "", "error": ""}
            return {**job.__dict__, "env": {}, "title": job_queue.title_of(job), "progress": progress, "queue": None, "class": "pipeline"}
        return self.scheduler.describe(job)

    def close(self) -> None:
        self.open = False  # the shared queue carries on
