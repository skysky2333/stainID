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
memory less what the app keeps) decides what runs at once, and work outside the queue is looked at too (free memory,
GPUs other programs use, the load average for the lowest classes). A job that does not fit waits and keeps what it needs
reserved from lower classes, so they only use what is spare (a smaller job of its own class may use what is left); when
it could run if lower-class jobs stopped, those are pre-empted: asked to stop at their next unit boundary (`stainid.jobs.units`), stopped hard after their class's grace
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
from dataclasses import asdict, dataclass, field
from pathlib import Path

from stainid.jobs import ACTIVE, APP, COMMAND_LINE, JOB_ENV, Job, JobStore, alive, new_id, policy, stop
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
OUTSIDE_RAM_MARGIN_GB = 2.0  # memory left free on the computer when a job starts
OUTSIDE_GPU_MB = 1500  # a GPU on which a program the queue did not start uses this much memory counts as busy
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


def _in_queue(pid: int) -> bool:
    """Whether a process belongs to a job the queue started (its systemd unit), or is the app itself."""
    try:
        groups = Path(f"/proc/{pid}/cgroup").read_text()
    except OSError:
        return False
    return "stainid-job-" in groups or "stainid-app" in groups


def queue_dir() -> Path:
    return Path(os.environ.get(QUEUE_ENV) or Path.home() / ".stainid" / "queue")


# ---------------------------------------------------------------- budget and needs

@dataclass
class Budget:
    cpus: float
    gpus: int
    ram_gb: float
    io_per_disk: int = policy.IO_PER_DISK  # io-heavy jobs at once on one disk
    caps: dict = field(default_factory=lambda: {k: dict(v) for k, v in policy.DEFAULT_CAPS.items()})  # class -> {"big": n}

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
    """`budget.json` in the queue folder ({"cpus": 24, "gpus": 2, "ram_gb": 30, "io_per_disk": 2, "caps": {"public": {"big": 1}}},
    any of them), else this computer's."""
    budget = default_budget()
    try:
        saved = json.loads((directory / "budget.json").read_text())
    except (OSError, ValueError):
        saved = {}
    for key in ("cpus", "gpus", "ram_gb", "io_per_disk"):
        if isinstance(saved.get(key), (int, float)):
            setattr(budget, key, type(getattr(budget, key))(saved[key]))
    if isinstance(saved.get("caps"), dict):
        budget.caps = {k: dict(v) for k, v in saved["caps"].items() if isinstance(v, dict)}
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
                 budget: Budget | None = None, outside: bool | None = None):
        """`outside`: also look at what this computer is doing outside the queue (free memory, busy GPUs, load) before
        starting a job; on by default unless a budget is given (tests)."""
        self.watch_outside = budget is None if outside is None else outside
        self._outside: tuple[float, dict] = (0.0, {})
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
        job.order = job.order or job.created or time.time()
        job.klass = job_class(job) if job.klass else "pipeline"
        if not job.log_file:
            job.log_file = str(self.dir / f"{job.id}.log")
        with self.lock:
            self.jobs[job.id] = job
            self._save(job)
        return job

    def submit_step(self, project, workflow: str, options: dict, klass: str | None = None, needs: dict | None = None,
                    argv: list[str] | None = None, after: list[str] | None = None, array: bool | str | int | None = None,
                    time_s: float = 0.0, owner: str = "", conflicts: list[str] | None = None, title: str = "") -> Job:
        """Queue a workflow step of a project (`argv`: its exact arguments, when they hold more than its options). A big
        step over many core images becomes a job array (`stainid.jobs.arrays`): `array=None` decides by itself, False
        never, True / "tma" / a number of core images per chunk always when there are two chunks or more."""
        from stainid.jobs import arrays
        from stainid.workflows.steps import BY_ID, to_argv

        if workflow not in BY_ID or BY_ID[workflow].command is None:
            raise KeyError(workflow)
        options = {**{o.key: o.default for o in BY_ID[workflow].options}, **options}
        argv = list(argv) if argv else to_argv(workflow, options)
        job = Job(id=new_id(), workflow=workflow, options=options, argv=argv, project=str(Path(project.root).resolve()), project_name=project.name,
                  klass=klass or class_of(project, workflow), needs=needs or needs_of(workflow, options), after=list(after or []),
                  time_s=float(time_s or 0), owner=owner, conflicts=list(conflicts or []), title=title)
        chunks = None
        if array is not False:
            try:
                chunks = arrays.plan(project, workflow, options, argv, None if array in (None, True) else array)
            except (OSError, ValueError, KeyError):
                chunks = None
        if not chunks:
            return self.submit(job)
        return self._submit_array(job, chunks)

    def _submit_array(self, parent: Job, chunks: list[dict]) -> Job:
        """A parent and one chunk job per part (`--core` with the part's core images), queued in order."""
        from stainid.jobs import arrays

        folder = self.dir / "arrays" / parent.id
        folder.mkdir(parents=True, exist_ok=True)
        parent.order = parent.order or time.time()
        parent.units = sum(c["units"] for c in chunks)
        base = arrays.without_cores(parent.argv)
        children = []
        for number, chunk in enumerate(chunks):
            listing = folder / f"{number + 1:04d}.txt"
            write_text_atomic(listing, "\n".join(chunk["cores"]) + "\n")
            child = Job(id=f"{parent.id}.{number + 1:04d}", workflow=parent.workflow, options=parent.options, argv=[*base, "--core", str(listing)],
                        project=parent.project, project_name=parent.project_name, klass=parent.klass, needs=dict(parent.needs),
                        after=list(parent.after), owner=parent.owner, conflicts=list(parent.conflicts), array_of=parent.id,
                        units=chunk["units"], title=f"{title_of(parent)} · {chunk['label']}",
                        created=parent.created + (number + 1) * 1e-6, order=parent.order + (number + 1) * 1e-6)
            children.append(child)
        parent.children = [c.id for c in children]
        parent.status = "queued"
        with self.lock:
            for child in children:
                self.submit(child)
            self.submit(parent)
        return parent

    def submit_command(self, command: list[str], cwd: str, title: str, klass: str, needs: dict | None = None, project=None,
                       env: dict | None = None, grace: float = 0.0, after: list[str] | None = None, time_s: float = 0.0,
                       conflicts: list[str] | None = None, owner: str = "") -> Job:
        """Queue any command (a stainID command that is not a step, a script): it should skip what it has done when run again."""
        job = Job(id=new_id(), workflow="command", options={}, argv=list(command), command=list(command), cwd=str(cwd), title=title,
                  klass=klass, needs={**EXEC_NEEDS, **(needs or {})}, env=dict(env or {}), grace=grace, after=list(after or []),
                  time_s=float(time_s or 0), conflicts=list(conflicts or []), owner=owner,
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
        if job.children:
            self._apply_to_array(job, request)
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

    def _apply_to_array(self, parent: Job, request: dict) -> None:
        """Pause, Resume, Cancel and the class act on every chunk; Move moves them all together."""
        op = request.get("op")
        chunks = [self.jobs[c] for c in parent.children if c in self.jobs]
        if op == "move":
            self._move(parent, request.get("where", "up"))
            return
        if op == "class" and request.get("klass") in CLASSES:
            parent.klass = request["klass"]
        for chunk in chunks:
            if op in ("pause", "resume", "cancel", "class"):
                self._apply({**request, "id": chunk.id})
        self._update_parent(parent, force=True)

    def _update_parents(self) -> None:
        for job in list(self.jobs.values()):
            if job.children:
                self._update_parent(job)

    def _update_parent(self, parent: Job, force: bool = False) -> None:
        """A parent's state from its chunks: running while one runs, waiting while one waits, paused when the rest are
        paused, and at the end finished (all finished), failed or stopped."""
        chunks = [self.jobs[c] for c in parent.children if c in self.jobs]
        states = {c.status for c in chunks}
        if "running" in states:
            status = "running"
        elif "queued" in states:
            status = "queued"
        elif "paused" in states:
            status = "paused"
        elif not chunks or states == {"finished"}:
            status = "finished"
        elif "failed" in states:
            status = "failed"
        elif "cancelled" in states:
            status = "cancelled"
        elif states <= {"finished", "skipped"}:
            status = "skipped" if "finished" not in states else "finished"
        else:
            status = "interrupted"
        started = [c.started for c in chunks if c.started]
        changed = status != parent.status or force
        parent.status = status
        parent.started = min(started) if started else None
        if status in FINAL and parent.ended is None:
            parent.ended, changed = max((c.ended or 0) for c in chunks) if chunks else time.time(), True
        elif status not in FINAL:
            parent.ended = None
        if status == "paused":
            parent.paused_by = next((c.paused_by for c in chunks if c.status == "paused"), "user")
        if changed:
            if status in FINAL:
                parent.outcome = self._array_progress(parent, chunks)
            self._save(parent)
            if status in ("failed", "cancelled", "skipped", "interrupted"):
                self._skip_dependents(parent, "failed" if status == "failed" else "did not finish")

    def _array_progress(self, parent: Job, chunks: list[Job]) -> dict:
        from stainid.jobs import progress as log_progress

        done = total = 0
        error = ""
        for chunk in chunks:
            total += chunk.units
            if chunk.status == "finished":
                done += chunk.units
            elif chunk.status == "running" and chunk.log_file:
                p = log_progress(Path(chunk.log_file))
                if p["total"]:
                    done += min(chunk.units, round(chunk.units * p["done"] / p["total"]))
            if chunk.status == "failed" and not error:
                error = f"{chunk.title}: {(chunk.outcome or {}).get('error') or 'failed'}"
        counts = {state: sum(1 for c in chunks if c.status == state) for state in ("finished", "running", "queued", "paused", "failed")}
        line = f"{counts['finished']} of {len(chunks)} chunks done" + (f", {counts['running']} running" if counts["running"] else "")
        return {"done": done, "total": total, "last_line": line, "error": error, "chunks": {**counts, "total": len(chunks)}}

    def _move(self, job: Job, where: str) -> None:
        """Move a waiting job up or down (or to the top) among the waiting jobs of its class."""
        if job.status not in ("queued", "paused"):
            return
        mates = sorted((j for j in self.jobs.values() if j.status in ("queued", "paused") and job_class(j) == job_class(job)
                        and not j.array_of), key=_key)  # whole jobs and arrays (an array's chunks move with it)
        if job not in mates:
            return
        place = mates.index(job)
        before = job.order or job.created
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
                self._shift_chunks(neighbour, a - b)
        self._shift_chunks(job, (job.order or job.created) - before)

    def _shift_chunks(self, job: Job, delta: float) -> None:
        if not delta:
            return
        for chunk_id in job.children or []:
            chunk = self.jobs.get(chunk_id)
            if chunk is not None:
                chunk.order = (chunk.order or chunk.created) + delta
                self._save(chunk)

    # -- one round

    def tick(self) -> None:
        with self.lock:
            self.refresh()
            if not self.leader:
                return
            self._take_requests()
            self._reap()
            self._enforce_stops()
            self._update_parents()
            self._schedule()
            self._update_parents()

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

    # -- estimates

    def _history(self) -> dict:
        """Per (step, project) and per step: seconds and units of the runs that ended, and the durations of the jobs that
        finished (for estimates), remade at most every 10 s."""
        when, found = getattr(self, "_history_cache", (0.0, None))
        if found is not None and time.time() - when < 10:
            return found
        found = {"unit": {}, "whole": {}}
        for job in list(self.jobs.values()):
            if job.children or not job.runs:
                continue
            last = job.runs[-1]
            last_seconds = (last[1] or 0) - (last[0] or 0) if last and last[0] else 0
            done = (job.outcome or {}).get("done", 0)  # the units its last run did (a run prints [done/total] of what it had to do)
            name = job.title if job.command else job.workflow
            for key in ((name, job.project), (name, "")):
                if done and last_seconds > 0 and not job.command:
                    unit = found["unit"].setdefault(key, [0.0, 0])
                    unit[0] += last_seconds
                    unit[1] += done
                if job.status == "finished" and not job.array_of:
                    seconds = sum((r[1] or 0) - (r[0] or 0) for r in job.runs if r and r[0] and r[1])
                    if seconds > 0:
                        found["whole"].setdefault(key, []).append(seconds)
        self._history_cache = (time.time(), found)
        return found

    def estimate(self, job: Job) -> float | None:
        """How long the job is expected to run in all, in seconds (None: not known)."""
        if job.time_s:
            return job.time_s
        if job.children:
            parts = [self.estimate(self.jobs[c]) for c in job.children if c in self.jobs]
            known = [p for p in parts if p is not None]
            return sum(known) if known else None
        history = self._history()
        if job.units and not job.command:
            for key in ((job.workflow, job.project), (job.workflow, "")):
                seconds, done = history["unit"].get(key, (0.0, 0))
                if done >= 3:
                    return job.units * seconds / done
            if job.workflow in policy.UNIT_SECONDS:
                return job.units * policy.UNIT_SECONDS[job.workflow]
        for key in (((job.title, job.project), (job.title, "")) if job.command else ((job.workflow, job.project), (job.workflow, ""))):
            times = sorted(history["whole"].get(key, []))
            if times:
                return times[len(times) // 2]
        return None if job.command else policy.STEP_SECONDS.get(job.workflow)

    def remaining(self, job: Job, now: float | None = None) -> float | None:
        """Seconds a running job is expected to take still (from its progress, else its estimate)."""
        now = now or time.time()
        elapsed = now - (job.started or now)
        if job.request:  # asked to stop: by the end of its grace time at the latest
            return max(0.0, (job.requested_at or now) + (job.grace or GRACE[job_class(job)]) + KILL_AFTER - now)
        from stainid.jobs import progress as log_progress

        p = log_progress(Path(job.log_file)) if job.log_file else {"done": 0, "total": 0}
        if p["done"] and p["total"] and p["done"] < p["total"] and elapsed > 0:
            return elapsed / p["done"] * (p["total"] - p["done"])
        total = self.estimate(job)
        return None if total is None else max(30.0, total - elapsed)

    def size(self, job: Job) -> str:
        need = job_needs(job)
        return policy.size_of(need, self.estimate(job))

    def _top(self, job: Job) -> Job:
        return self.jobs.get(job.array_of, job) if job.array_of else job

    # -- what to start

    def _schedule(self) -> None:
        """Start what fits (`stainid.jobs.policy`): by class, taking turns between projects within one; a job that does
        not fit gets a reservation (EASY backfill: later jobs start only if they do not delay it) and pre-empts lower
        classes when that makes room; writers of the same outputs never run at once; io-heavy jobs per disk and big
        jobs per class are capped; skip what cannot run (a step it waits for failed, an input is missing)."""
        from stainid.workflows.steps import BY_ID, wait_reason

        budget, now = self.budget, time.time()
        jobs = [j for j in self.jobs.values() if not j.children]
        running = [j for j in jobs if j.status == "running"]
        blocking = [j for j in jobs if j.status in (*ACTIVE, "paused")]  # a paused step still holds back what needs it
        used, busy_gpus = self._used(running)
        free = policy.minus({k: getattr(budget, k) for k in policy.KEYS}, used)
        io_busy: dict = {}
        big: dict[str, set[str]] = {}
        sizes: dict[str, str] = {}
        size = lambda top: sizes.setdefault(top.id, self.size(top))  # noqa: E731 - once per job and round
        held_until = {}  # running job -> (expected end, what it holds): when what it holds is free again
        for job in running:
            if policy.io_of(job, job_needs(job)) == "heavy":
                io_busy[policy.disk_of(job)] = io_busy.get(policy.disk_of(job), 0) + 1
            top = self._top(job)
            if size(top) == "big":
                big.setdefault(job_class(job), set()).add(top.id)
            left = self.remaining(job, now)
            held_until[job.id] = (now + left if left is not None else policy.INF, policy.counted(budget.clamp(job_needs(job))))
        by_project: dict[str, list[Job]] = {}
        for other in blocking:
            by_project.setdefault(other.project, []).append(other)
        self._dependency_cache = {}
        stopping = {j.id for j in running if j.request}
        reservations: list[dict] = []  # per class, its first job that waits for room: when it can start, what is left then
        held = self.paused_all()
        queued = [j for j in jobs if j.status == "queued"]
        for job in policy.fair_order(queued, running, _key, lambda j: RANK[job_class(j)]):
            if held:
                job.waiting = "All background work is paused"
                continue
            reason = self._blocked(job, by_project.get(job.project, []) + ([] if not job.project else by_project.get("", [])), BY_ID, wait_reason)
            if job.status != "queued":
                continue
            if reason:
                self._wait(job, reason)
                continue
            klass, rank = job_class(job), RANK[job_class(job)]
            top = self._top(job)
            cap = (budget.caps.get(klass) or {}).get("big")
            if cap is not None and size(top) == "big" and len(big.get(klass, set()) - {top.id}) >= cap:
                self._wait(job, f"At most {cap} big {CLASS_TITLES[klass].lower()} job{'s' if cap != 1 else ''} at a time")
                continue
            need = policy.counted(budget.clamp(job_needs(job)))
            heavy = policy.io_of(job, job_needs(job)) == "heavy"
            disk = policy.disk_of(job)
            if heavy and io_busy.get(disk, 0) >= budget.io_per_disk:
                self._wait(job, f"Waits for its disk: {io_busy[disk]} io-heavy jobs use it already")
                continue
            outside = self.outside()["gpus"] if self.watch_outside and need["gpus"] else set()
            gpus = [g for g in range(budget.gpus) if g not in busy_gpus and g not in outside][: int(need["gpus"])]
            estimate = self.estimate(job)
            if policy.fits(need, free) and len(gpus) >= int(need["gpus"]):
                end = now + (estimate if estimate is not None else policy.INF)
                delays = next((r for r in reservations if end > r["shadow"] and not policy.fits(need, r["extra"])), None)
                if delays is not None:
                    self._wait(job, f"Waits so as not to delay “{title_of(delays['job'])}”" + _starts_in(delays["shadow"], now))
                    continue
                if why := self._no_room_outside(job, need, gpus, used):
                    self._wait(job, why)
                    continue
                step = BY_ID.get(job.workflow) if not job.command else None
                if step is not None and (missing := self._missing(job, step)):
                    self._skip(job, missing)
                    continue
                try:
                    self._start(job, gpus)
                except Exception as error:  # noqa: BLE001 - one job that cannot start does not stop the queue
                    job.status, job.ended = "failed", time.time()
                    job.outcome = {"done": 0, "total": 0, "last_line": str(error), "error": f"Could not start: {error}"}
                    self._save(job)
                    continue
                free, used = policy.minus(free, need), policy.plus(used, need)
                for r in reservations:
                    if end > r["shadow"]:
                        r["extra"] = policy.minus(r["extra"], need)  # it uses part of what is left over at that time
                busy_gpus |= set(gpus)
                running.append(job)
                held_until[job.id] = (end, need)
                if heavy:
                    io_busy[disk] = io_busy.get(disk, 0) + 1
                if size(top) == "big":
                    big.setdefault(klass, set()).add(top.id)
                continue
            # it does not fit: lower classes give way (they stop at their next unit boundary and wait again)
            lower = sorted((j for j in running if RANK[job_class(j)] > rank and j.source != COMMAND_LINE),
                           key=lambda j: (-RANK[job_class(j)], -(j.started or 0)))
            freed = {k: 0.0 for k in policy.KEYS}
            for j in lower:
                if j.id in stopping:
                    freed = policy.plus(freed, policy.counted(budget.clamp(job_needs(j))))
            victims = []
            for j in lower:
                if policy.fits(need, policy.plus(free, freed)):
                    break
                if j.id in stopping:
                    continue
                n = policy.counted(budget.clamp(job_needs(j)))
                if not any(need[k] > free[k] + freed[k] + 1e-9 and n[k] > 0 for k in policy.KEYS):
                    continue  # it holds nothing this job lacks
                victims.append(j)
                freed = policy.plus(freed, n)
            ends = [(now + (j.grace or GRACE[job_class(j)]) + KILL_AFTER, held_until[j.id][1]) if j in victims else held_until[j.id]
                    for j in running]
            shadow, extra = policy.shadow(need, free, ends, now)
            if policy.fits(need, policy.plus(free, freed)) and victims:
                for j in victims:
                    self._ask_to_stop(j, "preempt")
                    stopping.add(j.id)
                self._wait(job, "Starts as soon as lower-priority work has paused")
            elif policy.fits(need, policy.plus(free, freed)) and stopping:
                self._wait(job, "Starts as soon as lower-priority work has paused")
            else:
                short = [k for k in policy.KEYS if need[k] > free[k] + 1e-9] or ["gpus"]
                names = {"cpus": "processors", "gpus": "GPUs", "ram_gb": "GB of memory"}
                self._wait(job, "Waits for room: needs " + ", ".join(f"{need[k]:g} {names[k]} ({max(0, free[k]):g} free)" for k in short)
                           + _starts_in(shadow, now))
            if not any(r["rank"] == rank for r in reservations):
                reservations.append({"rank": rank, "job": job, "shadow": shadow, "extra": extra})

    def _blocked(self, job: Job, blocking: list[Job], steps: dict, wait_reason) -> str:
        """Why a waiting job cannot start whatever the room: a job it was queued after, a step it needs the outputs of,
        a step writing the same file, or a job writing the same outputs."""
        for before in job.after or []:  # the jobs it was queued after
            other = self.jobs.get(before)
            if other is None or other.status == "finished":
                continue
            if other.status in FINAL:
                self._skip(job, f"Not started because “{title_of(other)}” {'failed' if other.status == 'failed' else 'did not finish'}")
                return ""
            return f"Waits for “{title_of(other)}” to finish"
        step = steps.get(job.workflow) if not job.command else None
        cache = getattr(self, "_dependency_cache", {})
        for other in blocking:
            if other is job or (job.array_of and other.array_of == job.array_of):
                continue  # the chunks of one array work on different core images
            earlier = other.status == "running" or other.created < job.created or other.source == COMMAND_LINE
            if not earlier:
                continue
            if step is not None and other.project == job.project and other.workflow in steps and not other.command and other.workflow != job.workflow:
                key = (other.workflow, job.workflow, other.source == COMMAND_LINE)  # the same for every pair of these steps
                if key not in cache:
                    cache[key] = wait_reason(steps[other.workflow], step, "command line" if other.source == COMMAND_LINE else "")
                if cache[key]:
                    return cache[key]
            if (other.status == "running" or other.workflow == job.workflow or other.command or job.command) and policy.conflict(job, other):
                return f"Waits for “{title_of(other)}”: both write the same outputs"
        return ""

    def outside(self) -> dict:
        """What work outside the queue (other programs, runs started by hand) leaves for it, looked at every 10 s: free
        memory (GB), GPUs whose memory is mostly taken by processes the queue did not start, and the load average."""
        when, found = self._outside
        if time.time() - when < 10:
            return found
        found = {"ram_gb": None, "gpus": set(), "load": None}
        try:
            with open("/proc/meminfo") as handle:
                info = {line.split(":")[0]: int(line.split()[1]) for line in handle}
            found["ram_gb"] = info.get("MemAvailable", 0) / 2**20
        except (OSError, ValueError, IndexError):
            pass
        try:
            found["load"] = os.getloadavg()[0]
        except OSError:
            pass
        if shutil.which("nvidia-smi"):
            try:
                query = lambda what: [r.split(", ") for r in subprocess.run(  # noqa: E731
                    ["nvidia-smi", f"--query-{what}", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout.splitlines()]
                index = {uuid: int(i) for i, uuid in query("gpu=index,uuid")}
                for uuid, pid, used in query("compute-apps=gpu_uuid,pid,used_memory"):
                    if float(used) >= OUTSIDE_GPU_MB and int(pid) != os.getpid() and not _in_queue(int(pid)):
                        found["gpus"].add(index[uuid])
            except (OSError, ValueError, KeyError, subprocess.SubprocessError):
                pass
        self._outside = (time.time(), found)
        return found

    def _no_room_outside(self, job: Job, need: dict, gpus: list[int], used: dict) -> str:
        """Why a job that fits the budget cannot start because of work outside the queue ("" when it can)."""
        if not self.watch_outside:
            return ""
        found = self.outside()
        if found["ram_gb"] is not None and need.get("ram_gb", 0) > found["ram_gb"] - OUTSIDE_RAM_MARGIN_GB:
            return f"Waits for memory: needs {need['ram_gb']:g} GB, {max(0.0, found['ram_gb'] - OUTSIDE_RAM_MARGIN_GB):.0f} GB free on this computer"
        if need.get("gpus") and len(gpus) < int(need["gpus"]):
            return "Waits for a GPU: the free ones are in use by work outside the queue"
        if RANK[job_class(job)] >= RANK["public"] and found["load"] is not None and not need.get("network"):
            if found["load"] + need.get("cpus", 0) > self.budget.cpus:  # the lowest classes run only on spare processors
                return f"Waits for spare processors (this computer's load is {found['load']:.0f})"
        return ""

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
                    and other.workflow != job.workflow and (other.created > job.created or job.source == COMMAND_LINE) and wait_reason(BY_ID[job.workflow], BY_ID[other.workflow])):
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

        chunks = [self.jobs[c] for c in job.children if c in self.jobs] if job.children else []
        if job.children:
            progress = job.outcome if job.status in FINAL and job.outcome else self._array_progress(job, chunks)
        else:
            progress = job.outcome if job.status in FINAL and job.outcome else (log_progress(Path(job.log_file)) if job.log_file else self.store.progress(job))
        estimate = self.estimate(job)
        out = {**asdict(job), "env": {}, "title": title_of(job), "class": job_class(job), "class_title": CLASS_TITLES[job_class(job)],
               "rank": RANK[job_class(job)], "needs": job_needs(job), "progress": progress, "queue": None, "eta_s": None,
               "estimate_s": round(estimate) if estimate is not None else None, "size": policy.size_of(job_needs(job), estimate),
               "io": policy.io_of(job, job_needs(job)), "children": list(job.children)}
        if job.children:
            out["chunks"] = progress.get("chunks")
            out["running_chunks"] = [c.title for c in chunks if c.status == "running"][:4]
            left = [self.remaining(c) if c.status == "running" else self.estimate(c) for c in chunks if c.status not in FINAL]
            parallel = max(1, sum(1 for c in chunks if c.status == "running"))
            if left and all(x is not None for x in left):
                out["eta_s"] = round(sum(left) / parallel)
        if job.status in ("queued", "paused"):
            waiting = sorted((j for j in self.jobs.values() if j.status in ("queued", "paused") and not j.array_of), key=_key)
            mates = [j for j in waiting if job_class(j) == job_class(job)]
            out["queue"] = {"ahead": waiting.index(job) if job in waiting else 0, "position": mates.index(job) + 1 if job in mates else 0,
                            "in_class": len(mates), "running": sum(1 for j in self.jobs.values() if j.status == "running")}
            out["typical_s"] = self.typical(job)
        elif job.status == "running" and not job.children:
            left = self.remaining(job)
            out["eta_s"] = round(left) if left is not None else None
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
        tops = [j for j in jobs if not j.array_of]  # an array shows as its parent
        live = sorted((j for j in tops if j.status not in FINAL), key=lambda j: (j.status != "running", _key(j)))
        recent = sorted((j for j in tops if j.status in FINAL and (j.ended or 0) > time.time() - 86400), key=lambda j: -(j.ended or 0))[:50]
        budget = self.budget
        used, _ = self._used([j for j in jobs if j.status == "running" and not j.children])
        return {"classes": [{"id": c, "title": CLASS_TITLES[c], "rank": RANK[c], "jobs": [self.describe(j) for j in live if job_class(j) == c]}
                            for c in CLASSES],
                "recent": [self.describe(j) for j in recent], "paused_all": self.paused_all(), "led": self.leader or self.led_elsewhere(),
                "budget": asdict(budget), "used": used,
                "outside": {**(found := self.outside()), "gpus": sorted(found["gpus"])} if self.watch_outside else None}


def _starts_in(shadow: float, now: float) -> str:
    text = policy.duration_text(shadow - now) if shadow != policy.INF else ""
    return f"; expected to start in about {text}" if text and shadow > now else ""


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
