"""The one job queue of this computer: every long piece of work of every project, run by priority within a budget.

Records live in one folder (`queue_dir()`, default ~/.stainid/queue): `<id>.json` per job (a `Job`), its log, exit
code, pid and control file. The web app and the command line (`stainid jobs ...`) both submit to it; one process at a
time schedules (the "leader", holding `.scheduler.lock`: the app, or `stainid jobs serve`), the others only read the
records and leave requests (`requests/*.json`) that the leader carries out on its next round, about once a second.

Priority classes, highest first (interactive requests are not queued: they run in the app's request pool):

1. annotation   annotation work and applying area edits
2. pipeline     main-study pipeline stages
3. models       model comparison, training, checks
4. public       public-dataset projects (any project with a `dataset:` section, or `jobs: {class: public}`)
5. downloads    network only: runs alongside the others (it takes no processors from the budget)

Within a class jobs start first in, first out (`order`, which Move up / down changes). Each job declares what it
needs (processors, GPUs, memory); the budget (`budget.json`, default: this computer's processors and GPUs and its
memory less what the app keeps) decides what runs at once. A job that does not fit waits and keeps its resources
reserved from lower classes, so they only use what is spare (a smaller job of the same class may use what is left); when it could run if lower-class jobs stopped, those are
pre-empted: asked to stop at their next unit boundary (`stainid.jobs.units`), stopped hard after their class's grace
time, and queued again at their place. They resume by running again: every step skips the units it has done.

Jobs run in transient systemd user units (`stainid-job-<id>`, in `stainid-jobs.slice`, with their class's Nice,
CPUWeight and IOWeight), so they keep running when the app restarts; a scheduler that starts adopts the running ones,
and queues again the ones a reboot ended. Without systemd (macOS, tests) they run as processes in their own session.
"""
from __future__ import annotations

import fcntl
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

from stainid.jobs import ACTIVE, APP, COMMAND_LINE, JOB_ENV, Job, JobStore, alive, new_id, stop
from stainid.jobs.units import CONTROL_ENV, PAUSED, SINCE_ENV
from stainid.tables import write_text_atomic

QUEUE_ENV = "STAINID_QUEUE_DIR"
LAUNCHER_ENV = "STAINID_JOB_LAUNCHER"  # "systemd" or "process" (default: systemd where it can be used)
CLASSES = ("annotation", "pipeline", "models", "public", "downloads")
RANK = {name: number for number, name in enumerate(CLASSES, start=1)}  # 0: interactive requests (not queued)
CLASS_TITLES = {"annotation": "Annotation and area edits", "pipeline": "Main-study pipeline", "models": "Model comparison and training",
                "public": "Public datasets", "downloads": "Downloads"}
WEIGHTS = {"annotation": (0, 400, 400), "pipeline": (5, 100, 100), "models": (10, 50, 50), "public": (15, 20, 20), "downloads": (19, 10, 10)}
GRACE = {"annotation": 300.0, "pipeline": 300.0, "models": 120.0, "public": 30.0, "downloads": 10.0}  # seconds to reach a unit boundary
COMMAND_GRACE = 30.0
KILL_AFTER = 30.0  # seconds from the polite stop (SIGTERM) to SIGKILL
MAX_ATTEMPTS = 5  # runs that vanished without an exit code (reboot, killed) before the job is given up
FINAL = ("finished", "failed", "cancelled", "skipped", "interrupted")
TICK_SECONDS = 1.0

# the class of a workflow step when it is not a pipeline stage
STEP_CLASS = {"prepare_labels": "annotation", "train": "models", "score_segmentation": "models", "download_models": "downloads"}
# what a step needs: processors, GPUs, memory (GB); `network` steps take no processors from the budget
STEP_NEEDS = {
    "slides": {"cpus": 1, "ram_gb": 1}, "layout": {"cpus": 1, "ram_gb": 1}, "summarize": {"cpus": 1, "ram_gb": 4},
    "dearray": {"cpus": 2, "ram_gb": 6}, "export": {"cpus": 4, "ram_gb": 8}, "qc": {"cpus": 4, "ram_gb": 6},
    "tiles": {"cpus": 2, "ram_gb": 4}, "images": {"cpus": 4, "ram_gb": 6}, "calibrate": {"cpus": 2, "ram_gb": 6},
    "nuclei": {"cpus": 4, "gpus": 1, "ram_gb": 8}, "neun": {"cpus": 8, "ram_gb": 8}, "fields": {"cpus": 4, "gpus": 1, "ram_gb": 8},
    "masks": {"cpus": 2, "gpus": 1, "ram_gb": 8}, "v2_proposals": {"cpus": 12, "ram_gb": 16}, "v2_calls": {"cpus": 4, "ram_gb": 6},
    "v2_outlines": {"cpus": 2, "gpus": 1, "ram_gb": 8}, "v2_measures": {"cpus": 2, "ram_gb": 8},
    "download_models": {"cpus": 0, "ram_gb": 1, "network": True}, "prepare_labels": {"cpus": 2, "ram_gb": 4},
    "train": {"cpus": 4, "ram_gb": 6}, "score_segmentation": {"cpus": 2, "ram_gb": 4},
}
EXEC_NEEDS = {"cpus": 2, "ram_gb": 4}


def queue_dir() -> Path:
    return Path(os.environ.get(QUEUE_ENV) or Path.home() / ".stainid" / "queue")


# ---------------------------------------------------------------- budget and needs

@dataclass
class Budget:
    cpus: float
    gpus: int
    ram_gb: float

    def fits(self, need: dict, used: dict) -> bool:
        return all(used.get(k, 0) + need.get(k, 0) <= getattr(self, k) + 1e-9 for k in ("cpus", "gpus", "ram_gb"))

    def clamp(self, need: dict) -> dict:
        """A job that asks for more than there is gets all there is (else it would never start)."""
        return {**need, **{k: min(need.get(k, 0), getattr(self, k)) for k in ("cpus", "gpus", "ram_gb")}}


def _gpu_count() -> int:
    if sys.platform == "darwin":
        return 1  # the Apple GPU
    if not shutil.which("nvidia-smi"):
        return 0
    try:
        found = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return 0
    return sum(line.startswith("GPU ") for line in found.splitlines())


def _memory_gb() -> float:
    try:
        with open("/proc/meminfo") as handle:
            return int(next(line for line in handle if line.startswith("MemTotal")).split()[1]) / 2**20
    except (OSError, StopIteration, ValueError):
        return 16.0


APP_RESERVE_GB = 18.0  # what the web app may take (its own cap), kept out of the jobs' budget


def default_budget() -> Budget:
    return Budget(float(os.cpu_count() or 2), _gpu_count(), max(4.0, round(_memory_gb() - APP_RESERVE_GB - 3, 1)))


def load_budget(directory: Path) -> Budget:
    """`budget.json` in the queue folder ({"cpus": 24, "gpus": 2, "ram_gb": 30}, any of them), else this computer's."""
    budget = default_budget()
    try:
        saved = json.loads((directory / "budget.json").read_text())
    except (OSError, ValueError):
        saved = {}
    for key in ("cpus", "gpus", "ram_gb"):
        if isinstance(saved.get(key), (int, float)):
            setattr(budget, key, type(getattr(budget, key))(saved[key]))
    return budget


def needs_of(workflow: str, options: dict | None = None) -> dict:
    """What a workflow step needs. A step told to run on the processor needs no GPU; v2 proposals with a number of cores
    at once need about 2 processors and 2.5 GB per core image."""
    from stainid.workflows.steps import BY_ID

    options = options or {}
    step = BY_ID.get(workflow)
    need = dict(STEP_NEEDS.get(workflow) or ({"cpus": 4, "ram_gb": 8} if step is not None and step.heavy else {"cpus": 1, "ram_gb": 2}))
    if str(options.get("device", "")).lower() == "cpu" and need.get("gpus"):
        need["gpus"], need["cpus"] = 0, max(need["cpus"], 8)
    if workflow == "v2_proposals" and int(options.get("workers") or 0) > 0:
        workers = int(options["workers"])
        need["cpus"], need["ram_gb"] = 2 * workers, round(2.5 * workers + 1, 1)
    return {"cpus": 0, "gpus": 0, "ram_gb": 0, "network": False, **need}


def class_of(project, workflow: str) -> str:
    """The priority class of a step of this project (see the module docstring)."""
    klass = STEP_CLASS.get(workflow, "pipeline")
    if klass != "pipeline":
        return klass
    chosen = (project.config.get("jobs") or {}).get("class") if project is not None else None
    if chosen in CLASSES:
        return chosen
    return "public" if project is not None and project.config.get("dataset") else "pipeline"


def job_class(job: Job) -> str:
    return job.klass if job.klass in CLASSES else "pipeline"


def job_needs(job: Job) -> dict:
    if job.needs:
        return {"cpus": 0, "gpus": 0, "ram_gb": 0, "network": False, **job.needs}
    return needs_of(job.workflow, job.options) if not job.command else {"cpus": 0, "gpus": 0, "ram_gb": 0, "network": False, **EXEC_NEEDS}


def _key(job: Job) -> tuple:
    return (RANK.get(job_class(job), len(CLASSES)), job.order or job.created, job.created)


def title_of(job: Job) -> str:
    from stainid.workflows.steps import BY_ID

    if job.title:
        return job.title
    return BY_ID[job.workflow].title if job.workflow in BY_ID else job.workflow


# ---------------------------------------------------------------- launchers

class ProcessLauncher:
    """Each job a process in its own session, under `stainid.jobs.run` (stops with the computer, not with the app)."""

    name = "process"

    def __init__(self):
        self.processes: dict[str, subprocess.Popen] = {}

    def start(self, job: Job, command: list[str], env: dict, cwd: str, files: dict) -> None:
        nice = WEIGHTS[job_class(job)][0]
        prefix = ["nice", "-n", str(nice)] if nice and shutil.which("nice") else []
        with Path(files["log"]).open("ab") as log:
            process = subprocess.Popen([sys.executable, "-m", "stainid.jobs.run", files["exit"], "--pid-file", files["pid"], "--", *prefix, *command],
                                       cwd=cwd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=env)
        self.processes[job.id] = process
        job.pid = process.pid

    def ended(self, job: Job, files: dict) -> bool | None:
        """True when this launcher's process for the job has ended, None when it did not start it."""
        process = self.processes.get(job.id)
        if process is None:
            return None
        if process.poll() is None:
            return False
        del self.processes[job.id]
        return True

    def kill(self, job: Job, pids: tuple[int, ...]) -> None:
        for pid in pids:
            try:
                os.killpg(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass


class SystemdLauncher(ProcessLauncher):
    """Each job a transient systemd user service in `stainid-jobs.slice`: it outlives the app (a restart of the app's
    service no longer ends it) and gets its class's Nice, CPUWeight and IOWeight."""

    name = "systemd"
    SLICE = "stainid-jobs.slice"

    def __init__(self, ram_gb: float | None = None):
        super().__init__()
        if ram_gb:  # every job together stays within the budget's memory: the app and the computer keep theirs
            subprocess.run(["systemctl", "--user", "set-property", "--runtime", self.SLICE, f"MemoryMax={int(ram_gb * 1.15)}G"],
                           capture_output=True, timeout=20)

    @staticmethod
    def usable() -> bool:
        if sys.platform != "linux" or not shutil.which("systemd-run") or not os.environ.get("XDG_RUNTIME_DIR"):
            return False
        try:
            return subprocess.run(["systemctl", "--user", "is-system-running"], capture_output=True, timeout=10).returncode in (0, 1)
        except (OSError, subprocess.SubprocessError):
            return False

    def start(self, job: Job, command: list[str], env: dict, cwd: str, files: dict) -> None:
        nice, cpu, io = WEIGHTS[job_class(job)]
        write_text_atomic(Path(files["env"]), json.dumps(env))
        os.chmod(files["env"], 0o600)
        job.unit = f"stainid-job-{job.id}"
        argv = ["systemd-run", "--user", "--quiet", "--collect", f"--unit={job.unit}", f"--slice={self.SLICE}", f"--working-directory={cwd}",
                "-p", f"Nice={nice}", "-p", f"CPUWeight={cpu}", "-p", f"IOWeight={io}", "-p", "KillMode=control-group",
                "-p", f"StandardOutput=append:{files['log']}", "-p", f"StandardError=append:{files['log']}",
                "-p", f"Description=stainID job {title_of(job)} ({job.project_name or job.id})",
                *[f"--setenv={key}={env[key]}" for key in ("PYTHONPATH", "PATH", "VIRTUAL_ENV") if env.get(key)],  # to start the runner
                sys.executable, "-m", "stainid.jobs.run", files["exit"], "--pid-file", files["pid"], "--env-file", files["env"], "--", *command]
        done = subprocess.run(argv, capture_output=True, text=True, timeout=60)
        if done.returncode != 0:
            raise RuntimeError(f"systemd-run failed: {done.stderr.strip() or done.stdout.strip()}")
        job.pid = None  # from the pid file, once the runner has started

    def ended(self, job: Job, files: dict) -> bool | None:
        """Before the runner has written its pid file: whether its unit has stopped already (it could not start)."""
        if not job.unit or Path(files["pid"]).exists():
            return None
        try:
            found = subprocess.run(["systemctl", "--user", "show", "-p", "ActiveState", "--value", job.unit], capture_output=True, text=True,
                                   timeout=20).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None
        return True if found in ("inactive", "failed") else None

    def kill(self, job: Job, pids: tuple[int, ...]) -> None:
        if job.unit:
            subprocess.run(["systemctl", "--user", "kill", "--signal=SIGKILL", job.unit], capture_output=True, timeout=20)
        super().kill(job, pids)


def default_launcher(budget: Budget | None = None) -> ProcessLauncher:
    choice = os.environ.get(LAUNCHER_ENV, "")
    if choice == "process" or (choice != "systemd" and not SystemdLauncher.usable()):
        return ProcessLauncher()
    return SystemdLauncher(budget.ram_gb if budget else None)


# ---------------------------------------------------------------- the scheduler

class Scheduler:
    """The queue over the shared folder. `lead=True` tries to become the one that starts and stops jobs; otherwise (or
    when another process leads) it only reads the records and leaves requests for the leader."""

    def __init__(self, directory: Path | str | None = None, lead: bool = True, launcher: ProcessLauncher | None = None,
                 budget: Budget | None = None):
        self.store = JobStore(Path(directory) if directory else queue_dir())
        self.dir = self.store.dir
        (self.dir / "requests").mkdir(exist_ok=True)
        self.lock = threading.RLock()
        self.jobs: dict[str, Job] = {}
        self.seen: dict[str, float] = {}
        self._budget, self._launcher = budget, launcher
        self._handle = None
        self._stops: dict[str, tuple[str, float]] = {}  # job id -> (how far its hard stop has got, when)
        self._thread: threading.Thread | None = None
        self._closed = False
        self.wants_lead = lead
        self.refresh()
        if lead:
            self.try_lead()

    # -- leadership

    @property
    def leader(self) -> bool:
        return self._handle is not None

    def try_lead(self) -> bool:
        if self._handle is not None:
            return True
        handle = (self.dir / ".scheduler.lock").open("a+")
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            handle.close()
            return False
        handle.seek(0)
        handle.truncate()
        handle.write(f"{os.getpid()}\n")
        handle.flush()
        self._handle = handle
        with self.lock:
            self.refresh()
            self.recover()
        return True

    def led_elsewhere(self) -> bool:
        """Whether another process leads the queue now."""
        if self.leader:
            return False
        try:
            with (self.dir / ".scheduler.lock").open("a+") as handle:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(handle, fcntl.LOCK_UN)
            return False
        except OSError:
            return True

    def release(self) -> None:
        if self._handle is not None:
            try:
                fcntl.flock(self._handle, fcntl.LOCK_UN)
            finally:
                self._handle.close()
                self._handle = None

    @property
    def budget(self) -> Budget:
        return self._budget or load_budget(self.dir)

    @property
    def launcher(self) -> ProcessLauncher:
        if self._launcher is None:
            self._launcher = default_launcher(self.budget)
        return self._launcher

    def start_thread(self) -> None:
        """Run the queue in the background of this process (the app): rounds about once a second; a process that does not
        lead keeps trying to, so the queue goes on when the leader stops."""
        if self._thread is not None and self._thread.is_alive():
            return

        def loop() -> None:
            last_try = 0.0
            while not self._closed:
                try:
                    if not self.leader and self.wants_lead and time.time() - last_try > 10:
                        last_try = time.time()
                        self.try_lead()
                    self.tick()
                except Exception as error:  # noqa: BLE001 - the queue must go on
                    print(f"job queue: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
                time.sleep(TICK_SECONDS)

        self._thread = threading.Thread(target=loop, name="stainid-queue", daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._closed = True
        self.release()

    # -- records

    def _mtime(self, job_id: str) -> float:
        try:
            return self.store.path(job_id).stat().st_mtime_ns
        except OSError:
            return 0

    def _save(self, job: Job) -> None:
        self.store.save(job)
        self.seen[job.id] = self._mtime(job.id)

    def refresh(self) -> None:
        """Read the records that are new or changed since the last look (jobs submitted from elsewhere, command-line runs,
        and, when another process leads, everything it changed)."""
        try:
            names = [n for n in os.listdir(self.dir) if n.endswith(".json") and not n.startswith(".")]
        except OSError:
            return
        present = set()
        for name in names:
            job_id = name[:-5]
            present.add(job_id)
            mtime = self._mtime(job_id)
            if self.seen.get(job_id) == mtime and job_id in self.jobs:
                continue
            job = self.store.load(self.store.path(job_id))
            if job is None:
                continue
            known = self.jobs.get(job_id)
            self.jobs[job_id], self.seen[job_id] = job, mtime
            if self.leader and known is not None and known.source == COMMAND_LINE and known.status == "running" and job.status in ("failed", "cancelled"):
                self._skip_dependents(job, "failed" if job.status == "failed" else "was stopped")
        for job_id in set(self.jobs) - present:
            if self.jobs[job_id].status in FINAL:  # removed by hand: forget it
                self.jobs.pop(job_id, None)

    def files(self, job: Job) -> dict:
        return {"log": job.log_file or str(self.dir / f"{job.id}.log"), "exit": str(self.store.exit_path(job.id)),
                "pid": str(self.dir / f"{job.id}.pid"), "env": str(self.dir / f"{job.id}.env"), "control": str(self.dir / f"{job.id}.control")}

    def _pids(self, job: Job) -> tuple[int, ...]:
        """The runner's and the command's process (from the runner's pid file)."""
        try:
            return tuple(int(p) for p in Path(self.files(job)["pid"]).read_text().split()[:2])
        except (OSError, ValueError):
            return (job.pid,) if job.pid else ()

    # -- submitting

    def submit(self, job: Job) -> Job:
        """Add a job (any process may: a new record is never written by two processes)."""
        job.status = "queued"
        job.order = job.order or time.time()
        job.klass = job_class(job) if job.klass else "pipeline"
        if not job.log_file:
            job.log_file = str(self.dir / f"{job.id}.log")
        with self.lock:
            self.jobs[job.id] = job
            self._save(job)
        return job

    def submit_step(self, project, workflow: str, options: dict, klass: str | None = None, needs: dict | None = None,
                    argv: list[str] | None = None, after: list[str] | None = None) -> Job:
        """Queue a workflow step of a project (`argv`: its exact arguments, when they hold more than its options)."""
        from stainid.workflows.steps import BY_ID, to_argv

        if workflow not in BY_ID or BY_ID[workflow].command is None:
            raise KeyError(workflow)
        options = {**{o.key: o.default for o in BY_ID[workflow].options}, **options}
        job = Job(id=new_id(), workflow=workflow, options=options, argv=list(argv) if argv else to_argv(workflow, options),
                  project=str(Path(project.root).resolve()), project_name=project.name, klass=klass or class_of(project, workflow),
                  needs=needs or needs_of(workflow, options), after=list(after or []))
        return self.submit(job)

    def submit_command(self, command: list[str], cwd: str, title: str, klass: str, needs: dict | None = None, project=None,
                       env: dict | None = None, grace: float = 0.0, after: list[str] | None = None) -> Job:
        """Queue any command (a stainID command that is not a step, a script): it should skip what it has done when run again."""
        job = Job(id=new_id(), workflow="command", options={}, argv=list(command), command=list(command), cwd=str(cwd), title=title,
                  klass=klass, needs={**EXEC_NEEDS, **(needs or {})}, env=dict(env or {}), grace=grace, after=list(after or []),
                  project=str(Path(project.root).resolve()) if project is not None else "", project_name=project.name if project is not None else "")
        return self.submit(job)

    # -- requests (pause, resume, cancel, move, class, pause all)

    def request(self, op: str, job_id: str = "", **args) -> Job | None:
        """Carry out `op` now when this process leads, else leave it for the leader. Returns the job as it is now."""
        if self.leader:
            with self.lock:
                self._apply({"op": op, "id": job_id, **args})
                return self.jobs.get(job_id)
        write_text_atomic(self.dir / "requests" / f"{time.time():.6f}-{uuid.uuid4().hex[:6]}.json", json.dumps({"op": op, "id": job_id, **args}))
        return self.jobs.get(job_id)

    def _take_requests(self) -> None:
        folder = self.dir / "requests"
        try:
            names = sorted(n for n in os.listdir(folder) if n.endswith(".json"))
        except OSError:
            return
        for name in names:
            try:
                request = json.loads((folder / name).read_text())
            except (OSError, ValueError):
                request = None
            (folder / name).unlink(missing_ok=True)
            if request:
                self._apply(request)

    def _control(self, job: Job, value: str) -> None:
        path = Path(self.files(job)["control"])
        if value:
            write_text_atomic(path, value)
        else:
            path.unlink(missing_ok=True)

    def _ask_to_stop(self, job: Job, why: str) -> None:
        """Ask a running job to stop at its next unit boundary ("pause", "preempt", "all": it is queued or paused again
        afterwards; "cancel")."""
        if job.request == "cancel" or job.request == why:
            return
        job.request, job.requested_at = why, time.time()
        self._control(job, "cancel" if why == "cancel" else "pause")
        if job.source == COMMAND_LINE and why == "cancel":
            self._hard_stop(job, now=True)
        self._save_if_ours(job)

    def _save_if_ours(self, job: Job) -> None:
        if job.source != COMMAND_LINE:
            self._save(job)

    def paused_all(self) -> bool:
        return (self.dir / "paused_all").exists()

    def _apply(self, request: dict) -> None:
        op, job = request.get("op"), self.jobs.get(request.get("id", ""))
        if op == "pause_all":
            write_text_atomic(self.dir / "paused_all", time.strftime("%F %T"))
            for other in list(self.jobs.values()):
                if other.status == "running":
                    self._ask_to_stop(other, "all")
            return
        if op == "resume_all":
            (self.dir / "paused_all").unlink(missing_ok=True)
            for other in list(self.jobs.values()):
                if other.status == "paused" and other.paused_by == "all":
                    other.status, other.paused_by, other.waiting = "queued", "", ""
                    self._save(other)
            return
        if job is None:
            return
        if op == "pause":
            if job.status == "queued":
                job.status, job.paused_by, job.waiting = "paused", "user", ""
                self._save(job)
            elif job.status == "running":
                self._ask_to_stop(job, "pause")
        elif op == "resume":
            if job.status == "running" and job.request in ("pause", "all"):
                job.request, job.requested_at = "", None
                self._control(job, "")
                self._save_if_ours(job)
            elif job.status in ("paused", "interrupted", "failed", "cancelled") or (job.status == "skipped" and job.source != COMMAND_LINE):
                if job.source == COMMAND_LINE:  # from now on the queue runs it (its argv resumes where it stopped)
                    job.source, job.log_file = APP, str(self.dir / f"{job.id}.log")
                job.status, job.paused_by, job.waiting, job.ended, job.returncode = "queued", "", "", None, None
                job.attempts = 0
                self._save(job)
        elif op == "cancel":
            if job.status in ("queued", "paused"):
                job.status, job.ended, job.waiting = "cancelled", time.time(), ""
                self._save(job)
                self._skip_dependents(job, "was stopped")
            elif job.status == "running":
                self._ask_to_stop(job, "cancel")
                self._hard_stop(job, now=True)
        elif op == "move":
            self._move(job, request.get("where", "up"))
        elif op == "class" and request.get("klass") in CLASSES:
            job.klass = request["klass"]
            self._save_if_ours(job)

    def _move(self, job: Job, where: str) -> None:
        """Move a waiting job up or down (or to the top) among the waiting jobs of its class."""
        if job.status not in ("queued", "paused"):
            return
        mates = sorted((j for j in self.jobs.values() if j.status in ("queued", "paused") and job_class(j) == job_class(job)), key=_key)
        place = mates.index(job)
        if where == "top" and place > 0:
            job.order = (mates[0].order or mates[0].created) - 1.0
            self._save(job)
        elif where in ("up", "down"):
            other = place - 1 if where == "up" else place + 1
            if 0 <= other < len(mates):
                neighbour = mates[other]
                a, b = job.order or job.created, neighbour.order or neighbour.created
                job.order, neighbour.order = (b, a) if a != b else (b - 0.001 if where == "up" else b + 0.001, a)
                self._save(job)
                self._save(neighbour)

    # -- one round

    def tick(self) -> None:
        with self.lock:
            self.refresh()
            if not self.leader:
                return
            self._take_requests()
            self._reap()
            self._enforce_stops()
            self._schedule()

    def recover(self) -> None:
        """After a start: adopt the jobs still running, finish those that ended while no scheduler ran, and queue again
        those a restart of the computer (or a killed runner) ended."""
        for job in list(self.jobs.values()):
            if job.status == "running" and job.source != COMMAND_LINE:
                self._check_running(job, adopting=True)

    def _check_running(self, job: Job, adopting: bool = False) -> None:
        code = self.store.exit_code(job.id)
        if code is not None:
            self._ended(job, code)
            return
        ended = self.launcher.ended(job, self.files(job))
        if ended is False:
            return
        pids = self._pids(job)
        if pids and job.pid != pids[0]:
            job.pid = pids[0]
            self._save(job)
        if ended is None and pids and alive(pids[0], job.started):
            return
        if ended is None and not pids and time.time() - (job.started or 0) < 60 and not adopting:
            return  # the runner is still starting
        time.sleep(0.05)
        code = self.store.exit_code(job.id)
        self._ended(job, code)

    def _reap(self) -> None:
        for job in list(self.jobs.values()):
            if job.status != "running":
                continue
            if job.source == COMMAND_LINE:
                if not alive(job.pid, job.started):
                    fresh = self.store.load(self.store.path(job.id)) or job
                    self.jobs[job.id] = fresh
                    if fresh.status == "running":  # ended without saying how (killed, or the computer restarted)
                        fresh.status, fresh.ended, fresh.outcome = "interrupted", time.time(), self.store.progress(fresh)
                        self._save(fresh)
                        self._skip_dependents(fresh, "was stopped")
                continue
            self._check_running(job)

    def _ended(self, job: Job, code: int | None) -> None:
        now = time.time()
        job.runs = [*job.runs, [job.started, now]]
        job.returncode, job.outcome = code, self.store.progress(job)
        request, job.request, job.requested_at = job.request, "", None
        self._control(job, "")
        self._stops.pop(job.id, None)
        for name in ("pid", "env"):
            Path(self.files(job)[name]).unlink(missing_ok=True)
        job.pid, job.unit = None, ""
        if request == "cancel":
            job.status, job.ended = "cancelled", now
            self._skip_dependents(job, "was stopped")
        elif code == 0:
            job.status, job.ended = "finished", now
        elif request in ("pause", "preempt", "all") or code == PAUSED:
            job.status = "queued" if request == "preempt" else "paused"
            job.paused_by = request or "user"
            job.waiting = "Paused for higher-priority work: continues where it stopped once there is room" if request == "preempt" else ""
            job.ended = None
        elif code is None:  # vanished without an exit code: the computer restarted, or its runner was killed
            job.attempts += 1
            if job.attempts <= MAX_ATTEMPTS:
                job.status, job.ended = "queued", None
                job.waiting = "Interrupted (the computer restarted or the run was killed): continues where it stopped"
            else:
                job.status, job.ended = "interrupted", now
                self._skip_dependents(job, "was stopped")
        else:
            job.status, job.ended = "failed", now
            self._skip_dependents(job, "failed")
        self.store.exit_path(job.id).unlink(missing_ok=True)
        self._save(job)

    def _hard_stop(self, job: Job, now: bool = False) -> None:
        stage, _ = self._stops.get(job.id, ("", 0.0))
        if stage:
            return
        pids = self._pids(job) if job.source != COMMAND_LINE else ((job.pid,) if job.pid else ())
        if pids:
            stop(pids[0])
        self._stops[job.id] = ("term", time.time())

    def _enforce_stops(self) -> None:
        """A job asked to stop that has not reached a unit boundary within its grace time is stopped (SIGTERM, then
        SIGKILL): it loses at most the unit it was on."""
        now = time.time()
        for job in list(self.jobs.values()):
            if job.status != "running" or not job.request:
                continue
            stage, when = self._stops.get(job.id, ("", 0.0))
            # a command that is not a stainID step may not look at its control file: it gets at most COMMAND_GRACE
            grace = job.grace or (min(GRACE[job_class(job)], COMMAND_GRACE) if job.command else GRACE[job_class(job)])
            if not stage and now - (job.requested_at or now) > grace:
                self._hard_stop(job)
            elif stage == "term" and now - when > KILL_AFTER:
                self.launcher.kill(job, self._pids(job))
                self._stops[job.id] = ("kill", now)

    # -- what to start

    def _used(self, running: list[Job]) -> tuple[dict, set[int]]:
        used = {"cpus": 0.0, "gpus": 0, "ram_gb": 0.0}
        gpus: set[int] = set()
        for job in running:
            need = self.budget.clamp(job_needs(job))
            for key in used:
                used[key] += 0 if key == "cpus" and need.get("network") else need.get(key, 0)
            gpus |= set(job.gpus or [])
        return used, gpus

    def _schedule(self) -> None:
        """Start what fits, highest class first and first in, first out within one; pre-empt lower classes for a job that
        could run if they stopped; skip what cannot run (a step it waits for failed, an input is missing)."""
        from stainid.workflows.steps import BY_ID, wait_reason

        budget = self.budget
        active = [j for j in self.jobs.values() if j.status in ACTIVE]
        blocking = [j for j in self.jobs.values() if j.status in (*ACTIVE, "paused")]  # a paused step still holds back what needs it
        running = [j for j in active if j.status == "running"]
        used, busy_gpus = self._used(running)
        waiting: list[tuple[int, dict]] = []  # (class rank, what it needs) of jobs that wait for room
        stopping = {j.id for j in running if j.request}
        held = self.paused_all()
        for job in sorted((j for j in active if j.status == "queued"), key=_key):
            if held:
                job.waiting = "All background work is paused"
                continue
            step = BY_ID.get(job.workflow) if not job.command else None
            reasons = []
            for before in job.after or []:  # the jobs it was queued after
                other = self.jobs.get(before)
                if other is None or other.status == "finished":
                    continue
                if other.status in FINAL:
                    reasons = []
                    self._skip(job, f"Not started because “{title_of(other)}” {'failed' if other.status == 'failed' else 'did not finish'}")
                    break
                reasons.append(f"Waits for “{title_of(other)}” to finish")
            if job.status != "queued":
                continue
            if step is not None:
                for other in blocking:
                    if other is job or other.project != job.project or other.workflow not in BY_ID or other.command:
                        continue
                    if (other.created < job.created or other.source == COMMAND_LINE) and not (other.workflow == job.workflow and other.options != job.options):
                        reason = wait_reason(BY_ID[other.workflow], step, "command line" if other.source == COMMAND_LINE else "")
                        if reason:
                            reasons.append(reason)
            if reasons:
                self._wait(job, reasons[0])
                continue
            need = budget.clamp(job_needs(job))
            counted = {**need, "cpus": 0 if need.get("network") else need.get("cpus", 0)}
            rank = RANK[job_class(job)]
            # what higher classes wait for stays free of this one; within a class a smaller job may use what is left
            reserved = {k: sum(n.get(k, 0) for r, n in waiting if r < rank) for k in used}
            free = {k: getattr(budget, k) - used[k] - reserved[k] for k in used}
            if all(counted.get(k, 0) <= free[k] + 1e-9 for k in used if counted.get(k, 0) > 0):
                if step is not None and (missing := self._missing(job, step)):
                    self._skip(job, missing)
                    continue
                gpus = [g for g in range(budget.gpus) if g not in busy_gpus][: int(counted.get("gpus", 0))]
                try:
                    self._start(job, gpus)
                except Exception as error:  # noqa: BLE001 - one job that cannot start does not stop the queue
                    job.status, job.ended = "failed", time.time()
                    job.outcome = {"done": 0, "total": 0, "last_line": str(error), "error": f"Could not start: {error}"}
                    self._save(job)
                    continue
                for key in used:
                    used[key] += counted.get(key, 0)
                busy_gpus |= set(gpus)
                running.append(job)
                continue
            # it does not fit: lower classes give way (they stop at their next unit boundary and wait again)
            lower = sorted((j for j in running if RANK[job_class(j)] > rank and j.source != COMMAND_LINE),
                           key=lambda j: (-RANK[job_class(j)], -(j.started or 0)))
            freed = {k: 0.0 for k in used}
            for j in lower:
                if j.id in stopping:
                    n = budget.clamp(job_needs(j))
                    for key in freed:
                        freed[key] += 0 if key == "cpus" and n.get("network") else n.get(key, 0)
            victims = []
            for j in lower:
                if all(counted.get(k, 0) <= free[k] + freed[k] + 1e-9 for k in used if counted.get(k, 0) > 0):
                    break
                if j.id in stopping:
                    continue
                n = budget.clamp(job_needs(j))
                if not any(counted.get(k, 0) > free[k] + freed[k] + 1e-9 and n.get(k, 0) > 0 and not (k == "cpus" and n.get("network")) for k in used):
                    continue  # it holds nothing this job lacks
                victims.append(j)
                for key in freed:
                    freed[key] += 0 if key == "cpus" and n.get("network") else n.get(key, 0)
            if all(counted.get(k, 0) <= free[k] + freed[k] + 1e-9 for k in used if counted.get(k, 0) > 0):
                for j in victims:
                    self._ask_to_stop(j, "preempt")
                    stopping.add(j.id)
                self._wait(job, "Starts as soon as lower-priority work has paused")
            else:
                short = [k for k in used if counted.get(k, 0) > free[k] + freed[k] + 1e-9]
                names = {"cpus": "processors", "gpus": "GPUs", "ram_gb": "GB of memory"}
                self._wait(job, "Waits for free capacity: needs " + ", ".join(f"{counted[k]:g} {names[k]} ({max(0, free[k]):g} free)" for k in short))
            waiting.append((rank, counted))

    def _wait(self, job: Job, reason: str) -> None:
        if job.waiting != reason:
            job.waiting = reason
            self._save(job)

    def _missing(self, job: Job, step) -> str:
        from stainid.project import load_project
        from stainid.workflows.steps import RESOURCES, maker_of, resource_info

        if not job.project:
            return ""
        try:
            project = load_project(job.project)
        except Exception as error:  # noqa: BLE001 - a project folder that is gone
            return f"Not started: the project cannot be read ({error})"
        missing = [r for r in step.needs if not resource_info(project, r)["exists"]]
        if not missing:
            return ""
        makers = {r: getattr(maker_of(project, r), "title", "") for r in missing}
        return "Not started: missing " + "; ".join(f"{RESOURCES[r].label}" + (f" (made by “{m}”)" if m else "") for r, m in makers.items())

    def _skip(self, job: Job, reason: str) -> None:
        job.status, job.waiting, job.ended = "skipped", "", time.time()
        job.outcome = {"done": 0, "total": 0, "last_line": reason, "error": reason}
        self._save(job)
        self._skip_dependents(job, "was skipped")

    def _skip_dependents(self, job: Job, what: str) -> None:
        """Queued jobs of the same project waiting for this one cannot do anything useful: skip them (and what waits for them)."""
        from stainid.workflows.steps import BY_ID, wait_reason

        for other in list(self.jobs.values()):
            if other.status in ("queued", "paused") and job.id in (other.after or []):
                self._skip(other, f"Not started because “{title_of(job)}” {what}")
        if job.workflow not in BY_ID or job.command:
            return
        for other in sorted(self.jobs.values(), key=lambda j: j.created):
            if (other.status in ("queued", "paused") and other.project == job.project and other.workflow in BY_ID and not other.command
                    and (other.created > job.created or job.source == COMMAND_LINE) and wait_reason(BY_ID[job.workflow], BY_ID[other.workflow])):
                self._skip(other, f"Not started because “{BY_ID[job.workflow].title}” {what}")

    def environment(self, job: Job) -> dict[str, str]:
        """The command's environment: this process's, with the project's model folders (never another project's), the
        job's GPUs and budget, and where it reads whether to stop."""
        from stainid.project import ENVIRONMENT_KEYS, load_project

        env = {k: v for k, v in os.environ.items() if k not in ENVIRONMENT_KEYS and k not in (JOB_ENV, CONTROL_ENV, SINCE_ENV)}
        if job.project:
            try:
                env |= load_project(job.project).environment()
            except Exception:  # noqa: BLE001 - a command without a readable project still runs
                pass
            env["STAINID_PROJECT"] = job.project
        need = self.budget.clamp(job_needs(job))
        env |= {JOB_ENV: job.id, CONTROL_ENV: self.files(job)["control"], "STAINID_JOB_CPUS": str(max(1, int(need.get("cpus") or 1))),
                "STAINID_JOB_RAM_GB": str(need.get("ram_gb") or 0), QUEUE_ENV: str(self.dir)}
        if job.runs and job.runs[0][0]:
            env[SINCE_ENV] = str(job.runs[0][0])  # a later run: a step redoing everything keeps what it redid already
        if need.get("gpus"):
            env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, job.gpus))
        return {**env, **{k: str(v) for k, v in (job.env or {}).items()}}

    def _start(self, job: Job, gpus: list[int]) -> None:
        files = self.files(job)
        for name in ("exit", "pid", "control"):
            Path(files[name]).unlink(missing_ok=True)
        job.gpus, job.request, job.requested_at, job.waiting, job.paused_by = gpus, "", None, "", ""
        job.log_file = files["log"]
        if job.runs and not job.command and job.options.get("fresh"):  # it moved the old results aside on its first run already
            job.options = {**job.options, "fresh": False}
            job.argv = [word for word in job.argv if word != "--fresh"]
        command = job.command or [sys.executable, "-u", "-m", "stainid.cli", "--project", job.project, *job.argv]
        cwd = job.cwd or job.project or str(Path.home())
        self.launcher.start(job, command, self.environment(job), cwd, files)
        job.status, job.started, job.ended, job.returncode = "running", time.time(), None, None
        self._save(job)

    # -- what pages show

    def describe(self, job: Job) -> dict:
        """A job as the Run page shows it: its class, place in the queue, progress and an estimate of the time left."""
        from stainid.jobs import progress as log_progress

        progress = job.outcome if job.status in FINAL and job.outcome else (log_progress(Path(job.log_file)) if job.log_file else self.store.progress(job))
        out = {**asdict(job), "env": {}, "title": title_of(job), "class": job_class(job), "class_title": CLASS_TITLES[job_class(job)],
               "rank": RANK[job_class(job)], "needs": job_needs(job), "progress": progress, "queue": None, "eta_s": None}
        if job.status in ("queued", "paused"):
            waiting = sorted((j for j in self.jobs.values() if j.status in ("queued", "paused")), key=_key)
            mates = [j for j in waiting if job_class(j) == job_class(job)]
            out["queue"] = {"ahead": waiting.index(job) if job in waiting else 0, "position": mates.index(job) + 1 if job in mates else 0,
                            "in_class": len(mates), "running": sum(1 for j in self.jobs.values() if j.status == "running")}
            out["typical_s"] = self.typical(job)
        elif job.status == "running":
            done, total = progress.get("done", 0), progress.get("total", 0)
            elapsed = time.time() - (job.started or time.time())
            if done and total and done < total:
                out["eta_s"] = round(elapsed / done * (total - done))
            out["stopping"] = job.request
        return out

    def typical(self, job: Job) -> float | None:
        """How long the same step of the same project took when it last ran to the end (for a waiting job's estimate)."""
        times = sorted(sum((r[1] or 0) - (r[0] or 0) for r in j.runs if r and r[0]) for j in self.jobs.values()
                       if j.status == "finished" and j.workflow == job.workflow and j.project == job.project and j.runs)
        return round(times[len(times) // 2]) if times else None

    def overview(self) -> dict:
        """Every job that is not over, by class (highest first), plus the ones that ended in the last day."""
        with self.lock:
            jobs = list(self.jobs.values())
        live = sorted((j for j in jobs if j.status not in FINAL), key=lambda j: (j.status != "running", _key(j)))
        recent = sorted((j for j in jobs if j.status in FINAL and (j.ended or 0) > time.time() - 86400), key=lambda j: -(j.ended or 0))[:50]
        budget = self.budget
        used, _ = self._used([j for j in jobs if j.status == "running"])
        return {"classes": [{"id": c, "title": CLASS_TITLES[c], "rank": RANK[c], "jobs": [self.describe(j) for j in live if job_class(j) == c]}
                            for c in CLASSES],
                "recent": [self.describe(j) for j in recent], "paused_all": self.paused_all(), "led": self.leader or self.led_elsewhere(),
                "budget": asdict(budget), "used": used}


_SHARED: Scheduler | None = None
_SHARED_LOCK = threading.Lock()


def shared(lead: bool = True) -> Scheduler:
    """This process's scheduler over `queue_dir()` (made on first use; a new one when the folder changes, as in tests)."""
    global _SHARED
    with _SHARED_LOCK:
        if _SHARED is None or _SHARED.dir != queue_dir():
            if _SHARED is not None:
                _SHARED.close()
            _SHARED = Scheduler(queue_dir(), lead=lead)
        return _SHARED


def reset() -> None:
    """Let go of this process's scheduler (tests)."""
    global _SHARED
    with _SHARED_LOCK:
        if _SHARED is not None:
            _SHARED.close()
        _SHARED = None
