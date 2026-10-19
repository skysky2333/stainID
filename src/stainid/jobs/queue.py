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
import functools
import json
import logging
import math
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

from stainid.jobs import ACTIVE, APP, COMMAND_LINE, JOB_ENV, PROGRESS_FILE_ENV, Job, JobStore, alive, new_id, policy, progress_file, stop
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
OUTSIDE_RAM_MARGIN_GB = 2.0  # memory left free on the computer when a job starts, on top of the app's headroom
APP_HEADROOM_GB = 4.0  # kept free for the web app to grow into (many annotators); below it the queue pauses jobs at once
SYSTEM_RESERVE_GB = 3.0  # for the system and the desktop
RAM_WARMUP_S = 120.0  # a job that started less than this ago counts what it declared (its memory is still growing)
RAM_FLOOR_GB = 0.5  # the least a running job counts
RAM_MARGIN = 1.1  # a running job counts its highest measured memory and this much more
PRESSURE_GRACE = 20.0  # seconds a job paused for memory pressure has to reach a unit boundary
AGE_S = 1800.0  # a job waiting this long goes ahead of its class's newer jobs and may pre-empt newer ones of its class
WATCHDOG_S = 900.0  # a job waiting this long although what it needs is (or can be made) free is reported
MEMORY_EVERY = 3.0  # seconds between two looks at the memory (and at the processors the running jobs use)
CPU_WINDOW_S = 60.0  # a running job's processor use: its average over this long
CPU_WARMUP_S = 60.0  # a job that started less than this ago counts the processors it declared
CPU_MARGIN = 1.25  # a running job counts its measured processor use and this much more (never more than it declared)
CPU_FLOOR = 0.5  # the fewest processors a running job counts
CPU_OVER_S = 60.0  # running jobs using more processors than there are for this long: the lowest class pauses
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


LOG = logging.getLogger("stainid.jobs")


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
    ram_cap: float | None = None  # a limit set by hand (`stainid jobs budget --ram`); else the memory follows the computer

    def fits(self, need: dict, used: dict) -> bool:
        return all(used.get(k, 0) + need.get(k, 0) <= getattr(self, k) + 1e-9 for k in ("cpus", "gpus", "ram_gb"))

    def clamp(self, need: dict) -> dict:
        """A job that asks for more than there is gets all there is (else it would never start)."""
        return {**need, **{k: min(need.get(k, 0), getattr(self, k)) for k in ("cpus", "gpus", "ram_gb")}}


@functools.lru_cache(maxsize=1)
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


@functools.lru_cache(maxsize=1)
def _memory_gb() -> float:
    try:
        with open("/proc/meminfo") as handle:
            return int(next(line for line in handle if line.startswith("MemTotal")).split()[1]) / 2**20
    except (OSError, StopIteration, ValueError):
        return 16.0


def default_budget() -> Budget:
    """This computer's processors and GPUs; its memory less the system's reserve and the app's headroom (while the queue
    runs, the memory follows what the app and other programs use: `Scheduler.current_budget`)."""
    return Budget(float(os.cpu_count() or 2), _gpu_count(), max(4.0, round(_memory_gb() - SYSTEM_RESERVE_GB - APP_HEADROOM_GB - 3, 1)))


def load_budget(directory: Path) -> Budget:
    """`budget.json` in the queue folder ({"cpus": 24, "gpus": 2, "ram_gb": 30, "io_per_disk": 2}, any of them), else this
    computer's. A memory limit given there caps the memory that otherwise follows the computer."""
    budget = default_budget()
    try:
        saved = json.loads((directory / "budget.json").read_text())
    except (OSError, ValueError):
        saved = {}
    for key in ("cpus", "gpus", "ram_gb", "io_per_disk"):
        if isinstance(saved.get(key), (int, float)):
            setattr(budget, key, type(getattr(budget, key))(saved[key]))
    if isinstance(saved.get("ram_gb"), (int, float)):
        budget.ram_cap = float(saved["ram_gb"])
    return budget


def needs_of(workflow: str, options: dict | None = None) -> dict:
    """What a workflow step needs. A step told to run on the processor needs no GPU. v2 proposals (`stainid.v2.run`):
    with `processes` tile processes, `threads` processors and 1.3 GB each plus 1.4 GB per core image in flight
    (`workers`, default 3); with a number of core images at once and no `processes`, 2 processors per core image
    (the tile processes the job then starts: `default_processes`); with `processes` 0 (the older layout, one process
    per core image), 2 processors and 2.5 GB per core image."""
    from stainid.workflows.steps import BY_ID

    options = options or {}
    step = BY_ID.get(workflow)
    need = dict(STEP_NEEDS.get(workflow) or ({"cpus": 4, "ram_gb": 8} if step is not None and step.heavy else {"cpus": 1, "ram_gb": 2}))
    if str(options.get("device", "")).lower() == "cpu" and need.get("gpus"):
        need["gpus"], need["cpus"] = 0, max(need["cpus"], 8)
    if workflow == "v2_proposals":
        workers, threads = int(options.get("workers") or 0), max(1, int(options.get("threads") or 1))
        processes = int(options["processes"]) if str(options.get("processes", "")).lstrip("-").isdigit() else -1
        if processes > 0:
            need["cpus"], need["ram_gb"] = processes * threads, round(1.3 * processes + 1.4 * (workers or 3) + 1, 1)
        elif processes == 0 and workers > 0:
            need["cpus"], need["ram_gb"] = 2 * workers, round(2.5 * workers + 1, 1)
        elif workers > 0:
            need["cpus"], need["ram_gb"] = 2 * workers, round(1.3 * 2 * workers + 1.4 * workers + 1, 1)
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
        self._ram_seen: dict[str, tuple] = {}  # job id -> (its start, highest measured GB, latest measured GB)
        self._cpu_seen: dict[str, tuple] = {}  # job id -> (its start, [(when, processor seconds so far), ...] over CPU_WINDOW_S)
        self._cpu_over_since: float | None = None  # since when the running jobs use more processors than there are
        self._memory: tuple[float, dict] = (0.0, {})
        self._urgent: set[str] = set()  # jobs asked to stop for memory pressure (short grace)
        self._making_room: dict[str, float] = {}  # job id -> since when lower-priority jobs pause to make room for it
        self.alerts: dict[str, str] = {}  # job id -> why the watchdog reports it
        self._ready_since: dict[str, float] = {}  # queued job id -> since when nothing but room holds it back (ageing)
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

    def cap(self, job: Job) -> str:
        """A job that asks for more than this computer can ever give jobs (`machine`) is given the most there is, with a
        warning saying so plainly (it would otherwise wait for ever); returns the warning ("" when it fits)."""
        need, most = policy.counted(job_needs(job)), self.machine()
        over = [k for k in policy.KEYS if need[k] > most[k] + 1e-9]
        if not over:
            return ""
        names = {"cpus": ("processors", ""), "gpus": ("GPUs", ""), "ram_gb": ("GB of memory", " GB")}
        given = {k: (math.floor(most[k] * 10) / 10 if k == "ram_gb" else math.floor(most[k])) for k in over}
        text = "; ".join(f"asks for {need[k]:g} {names[k][0]}; at most {given[k]:g}{names[k][1] or ' ' + names[k][0]} can be given to jobs on this computer"
                         for k in over) + ": it runs with " + ", ".join(f"{given[k]:g} {names[k][0]}" for k in over)
        job.needs = {**job_needs(job), **given}
        job.warning = text
        LOG.warning("queue: “%s” %s", title_of(job), text)
        return text

    def check(self, job: Job) -> None:
        """Give a job that asks for more than this computer can ever give jobs the most there is, with a warning (`cap`),
        and refuse one that waits for a job that does not exist or (through others) for itself."""
        self.cap(job)
        seen, todo = set(), list(job.after or [])
        while todo:
            other_id = todo.pop()
            if other_id == job.id:
                raise ValueError("--after: the jobs would wait for each other (a cycle)")
            if other_id in seen:
                continue
            seen.add(other_id)
            other = self.jobs.get(other_id) or self.store.load(self.store.path(other_id))
            if other is None:
                raise ValueError(f"--after: there is no job {other_id}")
            todo += list(other.after or [])

    def machine(self) -> dict:
        """The most jobs can ever be given: the processors and GPUs of the budget, and the memory the jobs could have with
        none of them running (the total less the system's reserve, the app and its headroom, and no more than is available
        once the jobs' own memory is given back, less the app's headroom and the margin a job start keeps free)."""
        budget = self.budget
        ram = budget.ram_gb
        if self.watch_outside:
            memory = self.memory()
            if memory.get("total_gb"):
                ram = memory["total_gb"] - SYSTEM_RESERVE_GB - APP_HEADROOM_GB - memory.get("app_gb", 0.0)
            if memory.get("available_gb") is not None:
                jobs_gb = 0.0  # given back: what each running job uses now (what it declared until it is measured, so never less)
                for job in [j for j in self.jobs.values() if j.status == "running" and not j.children]:
                    seen = self._ram_seen.get(job.id)
                    jobs_gb += seen[2] if seen is not None and seen[0] == job.started else float(job_needs(job).get("ram_gb", 0) or 0)
                ram = min(ram, memory["available_gb"] + jobs_gb - APP_HEADROOM_GB - OUTSIDE_RAM_MARGIN_GB)
            if budget.ram_cap:
                ram = min(ram, budget.ram_cap)
        return {"cpus": budget.cpus, "gpus": budget.gpus, "ram_gb": max(0.0, ram)}

    def submit(self, job: Job, checked: bool = False) -> Job:
        """Add a job (any process may: a new record is never written by two processes). Raises ValueError for a job
        that waits for one that does not exist or for itself (`check`); one that asks for more than jobs can ever be
        given is given the most there is, with a warning (`cap`)."""
        if not checked:
            with self.lock:
                self.check(job)
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
        self.cap(parent)  # (the chunks are given what the parent is)
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
            self.check(parent)
            for child in children:
                self.submit(child, checked=True)
            self.submit(parent, checked=True)
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
        done = total = 0
        error = ""
        for chunk in chunks:
            total += chunk.units
            if chunk.status == "finished":
                done += chunk.units
            elif chunk.status in ("running", "queued", "paused") and (chunk.log_file or chunk.outcome):
                p = self.progress_of(chunk)
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
        outcome = self.progress_of(job)  # over all runs so far (before this run joins `runs`)
        outcome.pop("run_done", None)
        job.runs = [*job.runs, [job.started, now]]
        job.returncode, job.outcome = code, outcome
        request, job.request, job.requested_at = job.request, "", None
        self._control(job, "")
        self._stops.pop(job.id, None)
        urgent = job.id in self._urgent
        self._urgent.discard(job.id)
        self._ram_seen.pop(job.id, None)
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
            job.waiting = ("Paused to keep memory free: continues where it stopped once there is room" if urgent and request == "preempt"
                           else "Paused for higher-priority work: continues where it stopped once there is room" if request == "preempt" else "")
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
            if job.id in self._urgent:  # memory is running out: no time to wait for a long unit
                grace = min(grace, PRESSURE_GRACE)
            if not stage and now - (job.requested_at or now) > grace:
                self._hard_stop(job)
            elif stage == "term" and now - when > KILL_AFTER:
                self.launcher.kill(job, self._pids(job))
                self._stops[job.id] = ("kill", now)

    # -- what to start

    def _used(self, running: list[Job]) -> tuple[dict, set[int]]:
        """What the running jobs hold (their memory as measured, `_holds`) and the GPUs they have."""
        used = {"cpus": 0.0, "gpus": 0, "ram_gb": 0.0}
        gpus: set[int] = set()
        for job in running:
            held = self._holds(job)
            for key in used:
                used[key] += held.get(key, 0)
            gpus |= set(job.gpus or [])
        return used, gpus

    # -- memory

    def memory(self) -> dict:
        """This computer's memory (GB), looked at every few seconds: total, available, the app's, and each running job's
        (its cgroup's anonymous memory, else the resident memory of its processes)."""
        when, found = self._memory
        if found and time.time() - when < MEMORY_EVERY:
            return found
        found = {}
        try:
            with open("/proc/meminfo") as handle:
                info = {line.split(":")[0]: int(line.split()[1]) for line in handle}
            found = {"total_gb": info["MemTotal"] / 2**20, "available_gb": info.get("MemAvailable", 0) / 2**20}
        except (OSError, ValueError, IndexError, KeyError):
            pass
        if found:
            found["app_gb"] = _app_memory_gb()
            running = [j for j in list(self.jobs.values()) if j.status == "running" and not j.children]
            for job in running:
                gb = self.measure(job)
                if gb is not None:
                    before = self._ram_seen.get(job.id)
                    peak = max(gb, before[1]) if before and before[0] == job.started else gb
                    self._ram_seen[job.id] = (job.started, peak, gb)
            self._sample_cpu(running)
        self._memory = (time.time(), found)
        return found

    # -- processors

    def _sample_cpu(self, running: list[Job]) -> None:
        """Note the processor seconds each running job has used so far (its cgroup's, else its processes')."""
        now, tree = time.time(), None
        for job in running:
            pids = self._pids(job) if job.source != COMMAND_LINE else ((job.pid,) if job.pid else ())
            if not pids:
                continue
            group = _cgroup_of(pids[0])
            seconds = _cgroup_cpu_seconds(group) if group and "stainid-job-" in group else None
            if seconds is None:
                tree = tree if tree is not None else _process_tree()
                seconds = _tree_cpu_seconds(pids, tree)
            if seconds is None:
                continue
            before = self._cpu_seen.get(job.id)
            samples = before[1] if before and before[0] == job.started else []
            if samples and seconds < samples[-1][1]:  # a process of it ended and took its time along: start again
                samples = []
            samples = [s for s in samples if now - s[0] <= CPU_WINDOW_S] + [(now, seconds)]
            self._cpu_seen[job.id] = (job.started, samples)
        alive_ids = {j.id for j in running}
        self._cpu_seen = {k: v for k, v in self._cpu_seen.items() if k in alive_ids}

    def cpu_use(self, job: Job) -> float | None:
        """The processors a running job used on average over the last CPU_WINDOW_S (None: not measured for 10 s yet)."""
        seen = self._cpu_seen.get(job.id)
        if not seen or seen[0] != job.started or len(seen[1]) < 2:
            return None
        (t0, c0), (t1, c1) = seen[1][0], seen[1][-1]
        return max(0.0, (c1 - c0) / (t1 - t0)) if t1 - t0 >= 10.0 else None

    def _relieve_cpus(self, running: list[Job], budget: Budget) -> None:
        """Jobs start on the processors the running ones use (measured), not on what they declared; when the running
        jobs then use more processors than there are for CPU_OVER_S (one grows into what it declared), the lowest-class
        jobs (newest first) are asked to pause until the rest fit; they queue again at their place. The highest-class
        job running is never paused for this."""
        if not self.watch_outside:
            return
        use = {j.id: self.cpu_use(j) or 0.0 for j in running}
        total, now = sum(use.values()), time.time()
        if total <= budget.cpus + 1.0:
            self._cpu_over_since = None
            return
        self._cpu_over_since = self._cpu_over_since or now
        if now - self._cpu_over_since < CPU_OVER_S:
            return
        total -= sum(use[j.id] for j in running if j.request)  # stopping already
        keep = min(running, key=lambda j: (RANK[job_class(j)], j.started or 0))  # the oldest of the highest class
        pool = sorted((j for j in running if j is not keep and j.source != COMMAND_LINE and not j.request and use[j.id] > 0),
                      key=lambda j: (-RANK[job_class(j)], -(j.started or 0)))
        for job in pool:
            if total <= budget.cpus:
                break
            LOG.warning("processors oversubscribed (%.1f of %g used): pausing %s", total, budget.cpus, title_of(job))
            self._ask_to_stop(job, "preempt")
            total -= use[job.id]
        self._cpu_over_since = None

    def measure(self, job: Job) -> float | None:
        """A running job's memory in GB (None: cannot tell)."""
        pids = self._pids(job) if job.source != COMMAND_LINE else ((job.pid,) if job.pid else ())
        if not pids:
            return None
        group = _cgroup_of(pids[0])
        if group and "stainid-job-" in group:
            found = _cgroup_memory_gb(group)
            if found is not None:
                return found
        return _tree_rss_gb(pids)

    def _holds(self, job: Job) -> dict:
        """What a running job takes from the budget. Memory: what it declared until it has been measured for a while, then
        its highest measured memory (and a margin); `ram_free` is what it takes from the memory available right now.
        Processors alike: what it declared for its first CPU_WARMUP_S, then its use over the last CPU_WINDOW_S and a
        margin, never more than it declared (`cpus_declared`, `cpus_now`: what it declared and uses). GPUs: as declared."""
        need = policy.counted(self.budget.clamp(job_needs(job)))
        seen = self._ram_seen.get(job.id)
        held = dict(need)
        if self.watch_outside and seen is not None and seen[0] == job.started:
            warm = time.time() - (job.started or 0) < RAM_WARMUP_S
            held["ram_gb"] = max(need["ram_gb"], seen[1] * RAM_MARGIN) if warm else max(RAM_FLOOR_GB, seen[1] * RAM_MARGIN)
            held["ram_free"] = max(need["ram_gb"] - seen[2], 0.0) if warm else 0.0  # still to come out of what is available
            held["ram_now"] = seen[2]
        else:
            held["ram_free"] = need["ram_gb"]  # not measured yet: all of it is still to come
            held["ram_now"] = 0.0
        use = self.cpu_use(job) if self.watch_outside else None
        held["cpus_declared"], held["cpus_now"] = need["cpus"], use if use is not None else 0.0
        if use is not None and time.time() - (job.started or 0) >= CPU_WARMUP_S:
            held["cpus"] = min(need["cpus"], max(CPU_FLOOR, use * CPU_MARGIN))  # work-conserving: what it uses, at most what it declared
        disk = policy.disk_of(job) if policy.io_of(job, job_needs(job)) == "heavy" and not need["gpus"] else None
        if disk is not None:  # (a GPU job is held back by its GPU, not its disk)
            held[f"io:{disk}"] = 1
        return held

    def current_budget(self) -> Budget:
        """The budget with its memory as this computer has it now: total less the app's use, the app's headroom and the
        system's reserve (capped by a limit set by hand)."""
        from dataclasses import replace

        budget = self.budget
        if not self.watch_outside:
            return budget
        memory = self.memory()
        if not memory.get("total_gb"):
            return budget
        ram = memory["total_gb"] - memory.get("app_gb", 0.0) - APP_HEADROOM_GB - SYSTEM_RESERVE_GB
        return replace(budget, ram_gb=round(max(0.0, min(ram, budget.ram_cap) if budget.ram_cap else ram), 1))

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

    def progress_of(self, job: Job) -> dict:
        """How far a job is over all its runs: what its log (or progress file) says, on top of what earlier runs did. A
        step resumed after a pause often counts only what is left ("[1/30]" where 42 of 72 were done): it then still
        shows 43 of 72. `run_done`: units done in this run (for the pace)."""
        from stainid.jobs import progress as log_progress

        p = log_progress(Path(job.log_file)) if job.log_file else self.store.progress(job)
        before = job.outcome if job.runs and (job.outcome or {}).get("total") else None
        if not before:
            return {**p, "run_done": p["done"]}
        done_before, total_before = int(before.get("done") or 0), int(before["total"])
        if not p["total"]:  # the new run has said nothing yet
            return {**p, "done": done_before, "total": total_before, "run_done": 0}
        if p["total"] < total_before:  # it counts what is left
            return {**p, "done": max(done_before, total_before - (p["total"] - p["done"])), "total": total_before, "run_done": p["done"]}
        if p["total"] == total_before:  # it counts everything again (finished units first)
            return {**p, "done": max(done_before, p["done"]), "run_done": max(0, p["done"] - done_before)}
        return {**p, "run_done": p["done"]}

    def remaining(self, job: Job, now: float | None = None) -> float | None:
        """Seconds a running job is expected to take still: from its progress, else from its estimate while that has not
        passed (None: no basis for one, or it runs longer than expected, see `overdue`, or it is being paused or stopped;
        never a made-up "30 s")."""
        now = now or time.time()
        elapsed = now - (job.started or now)
        if job.request:  # pausing or stopping: no time left to show (the pages say "Pausing…")
            return None
        p = self.progress_of(job)
        if p["run_done"] and p["total"] and p["done"] < p["total"] and elapsed > 0:
            return elapsed / p["run_done"] * (p["total"] - p["done"])
        total = self.estimate(job)
        return None if total is None or total <= elapsed else total - elapsed

    def overdue(self, job: Job, now: float | None = None) -> bool:
        """A running job past the time it was expected to take, with no progress to estimate from."""
        if job.status != "running" or job.request or not job.started:
            return False
        if job.children:
            return any(self.overdue(self.jobs[c], now) for c in job.children if c in self.jobs)
        p = self.progress_of(job)
        if p["run_done"] and p["total"] and p["done"] < p["total"]:
            return False
        total = self.estimate(job)
        return total is not None and (now or time.time()) - job.started > total

    def size(self, job: Job) -> str:
        need = job_needs(job)
        return policy.size_of(need, self.estimate(job))

    def _top(self, job: Job) -> Job:
        return self.jobs.get(job.array_of, job) if job.array_of else job

    # -- what to start

    def _free(self, budget: Budget, running: list[Job]) -> dict:
        """What is free now: processors, GPUs, memory (`ram_gb` against the budget; `ram_free` against what this computer
        has available, less the app's headroom), and io slots per disk."""
        used, busy = self._used(running)
        free = {"cpus": budget.cpus - used["cpus"], "gpus": budget.gpus - max(len(busy), used["gpus"]), "ram_gb": budget.ram_gb - used["ram_gb"]}
        outside = self.outside()["gpus"] if self.watch_outside else set()
        free["gpus"] -= len(set(outside) - busy)
        if self.watch_outside and (memory := self.memory()).get("available_gb") is not None:
            free["ram_free"] = memory["available_gb"] - APP_HEADROOM_GB - OUTSIDE_RAM_MARGIN_GB - sum(self._holds(j)["ram_free"] for j in running)
        for job in running:
            for key, value in self._holds(job).items():
                if key.startswith("io:"):
                    free[key] = free.get(key, budget.io_per_disk) - value
        return free

    def _need(self, job: Job, budget: Budget) -> dict:
        need = policy.counted(budget.clamp(job_needs(job)))
        if self.watch_outside:
            need["ram_free"] = need["ram_gb"]
        disk = policy.disk_of(job) if policy.io_of(job, job_needs(job)) == "heavy" and not need["gpus"] else None
        if disk is not None:
            need[f"io:{disk}"] = 1
        return need

    @staticmethod
    def _fits(need: dict, free: dict, budget: Budget) -> bool:
        for key, value in need.items():
            if value <= 0 or key in ("ram_now", "cpus_now", "cpus_declared"):
                continue
            have = free.get(key, budget.io_per_disk if key.startswith("io:") else None)
            if have is not None and value > have + 1e-9:
                return False
        return True

    def _aged(self, job: Job, now: float) -> bool:
        """Whether the job has waited longer than `AGE_S` with nothing but room holding it back."""
        return now - self._ready_since.get(job.id, now) > AGE_S

    def _order(self, queued: list[Job], running: list[Job], now: float, free_gpus: int = 0) -> list[Job]:
        """By class; within a class the jobs that have waited longer than `AGE_S` first (oldest first), then the others:
        those that fit a free GPU first (so no GPU stays idle while a GPU job waits), then taking turns between projects
        (fair share)."""
        aged = {j.id for j in queued if self._aged(j, now)}
        ordered = policy.fair_order([j for j in queued if j.id not in aged], running, _key, lambda j: RANK[job_class(j)])
        old = sorted((j for j in queued if j.id in aged), key=lambda j: (RANK[job_class(j)], j.created))
        gpu = {j.id for j in ordered if 0 < job_needs(j).get("gpus", 0) <= free_gpus
               and job_needs(j).get("cpus", 0) <= policy.SMALL["cpus"] and job_needs(j).get("ram_gb", 0) <= policy.SMALL["ram_gb"]}
        out = []
        for rank in sorted({RANK[job_class(j)] for j in queued}):
            mine = [j for j in ordered if RANK[job_class(j)] == rank]
            out += [j for j in old if RANK[job_class(j)] == rank] + [j for j in mine if j.id in gpu] + [j for j in mine if j.id not in gpu]
        return out

    def _may_preempt(self, job: Job, other: Job, now: float) -> bool:
        """Whether `job` may pause `other` to get room: a lower class always; the same class only when `job` has waited
        longer than `AGE_S` and `other` is a newer job that started after it was queued (older before newer, never back)."""
        if other.source == COMMAND_LINE or other.request == "cancel":
            return False
        if RANK[job_class(other)] > RANK[job_class(job)]:
            return True
        return (RANK[job_class(other)] == RANK[job_class(job)] and self._aged(job, now)
                and other.created > job.created and (other.started or 0) > job.created)

    def _victims(self, job: Job, need: dict, free: dict, running: list[Job], budget: Budget, now: float) -> list[Job] | None:
        """The fewest running jobs to pause so that `job` fits: lowest class first, then the largest holders of what it
        lacks; jobs pausing already count as freed. None when even pausing every job it may pause would not do."""
        room = dict(free)
        for other in running:
            if other.request:  # stopping already: what it holds comes free soon
                room = self._released(room, other)
        if self._fits(need, room, budget):
            return []
        pool = [o for o in running if not o.request and self._may_preempt(job, o, now)]
        chosen: list[Job] = []
        while not self._fits(need, room, budget):
            short = {k: need[k] - room.get(k, 0) for k in need if need[k] > 0 and k in room and need[k] > room[k] + 1e-9}
            best, score = None, None
            for other in pool:
                held = self._released({k: 0.0 for k in short}, other)
                gain = sum(min(held.get(k, 0), lack) / max(lack, 1e-9) for k, lack in short.items())
                if gain <= 0:
                    continue
                key = (RANK[job_class(other)], gain, other.started or 0)
                if score is None or key > score:
                    best, score = other, key
            if best is None:
                return None
            pool.remove(best)
            chosen.append(best)
            room = self._released(room, best)
        for other in list(reversed(chosen)):  # drop any that turned out not to be needed
            rest = self._released(room, other, -1)
            if self._fits(need, rest, budget):
                chosen.remove(other)
                room = rest
        return chosen

    def _released(self, room: dict, job: Job, sign: int = 1) -> dict:
        """`room` once `job` has stopped: what it holds comes free (from what is available: its measured memory too)."""
        held = self._holds(job)
        freed = {k: v for k, v in held.items() if k not in ("ram_now", "cpus_now", "cpus_declared")}
        freed["ram_free"] = held["ram_free"] + held["ram_now"]
        return {**room, **{k: room[k] + sign * v for k, v in freed.items() if k in room}}

    def _relieve_memory(self, running: list[Job]) -> None:
        """When the memory available falls under the app's headroom, pause the lowest-class jobs (largest first) at once
        (short grace), before the kernel's out-of-memory killer acts."""
        if not self.watch_outside:
            return
        memory = self.memory()
        if memory.get("available_gb") is None or memory["available_gb"] >= APP_HEADROOM_GB:
            return
        lack = APP_HEADROOM_GB + OUTSIDE_RAM_MARGIN_GB - memory["available_gb"]
        lack -= sum(self._holds(j)["ram_now"] for j in running if j.id in self._urgent and j.request)
        pool = sorted((j for j in running if j.source != COMMAND_LINE and not j.request),
                      key=lambda j: (-RANK[job_class(j)], -self._holds(j)["ram_now"]))
        for job in pool:
            if lack <= 0:
                break
            self._urgent.add(job.id)
            self._ask_to_stop(job, "preempt")
            lack -= max(self._holds(job)["ram_now"], 0.5)
            LOG.warning("memory pressure (%.1f GB available): pausing %s", memory["available_gb"], title_of(job))

    def _schedule(self) -> None:
        """Start every waiting job that fits, in priority order (work-conserving: nothing stays idle while a job that fits
        waits). A job starts only when all it needs (processors, GPU, memory, an io slot on its disk, and no running
        writer of the same outputs) is free at once, and takes it all in one step. A job that does not fit pauses the
        fewest lower-priority jobs that make it fit (they resume later, skipping finished units); what it will take is
        not handed to lower jobs while they stop (at most their grace time). See `docs/job_system.md`."""
        from stainid.workflows.steps import BY_ID, wait_reason

        budget, now = self.current_budget(), time.time()
        jobs = [j for j in self.jobs.values() if not j.children]
        running = [j for j in jobs if j.status == "running"]
        blocking = [j for j in jobs if j.status in (*ACTIVE, "paused")]  # a paused step still holds back what needs it
        self._relieve_memory(running)
        self._relieve_cpus(running, budget)
        free = self._free(budget, running)
        busy_gpus = {g for j in running for g in (j.gpus or [])} | (set(self.outside()["gpus"]) if self.watch_outside else set())
        by_project: dict[str, list[Job]] = {}
        for other in blocking:
            by_project.setdefault(other.project, []).append(other)
        self._dependency_cache = {}
        held = self.paused_all()
        queued = [j for j in jobs if j.status == "queued"]
        making_room: dict[str, float] = {}
        alerts: dict[str, str] = {}
        self._ready_since = {k: v for k, v in self._ready_since.items() if k in {j.id for j in queued}}
        for job in self._order(queued, running, now, int(free["gpus"])):
            if held:
                job.waiting = "All background work is paused"
                self._ready_since.pop(job.id, None)
                continue
            reason = self._blocked(job, by_project.get(job.project, []) + ([] if not job.project else by_project.get("", [])), BY_ID, wait_reason)
            if job.status != "queued":
                continue
            if reason:
                self._wait(job, reason)
                self._ready_since.pop(job.id, None)
                continue
            self._ready_since.setdefault(job.id, now)
            if not running and self.cap(job):  # with nothing running: it asks for more than jobs can ever be given
                self._save(job)
            need = self._need(job, budget)
            if self._fits(need, free, budget):
                gpus = [g for g in range(budget.gpus) if g not in busy_gpus][: int(need["gpus"])]
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
                free = {k: v - need.get(k, 0) for k, v in free.items()} | {k: budget.io_per_disk - v for k, v in need.items() if k not in free}
                busy_gpus |= set(gpus)
                running.append(job)
                self._ready_since.pop(job.id, None)
                continue
            victims = self._victims(job, need, free, running, budget, now)
            if victims is None:
                self._wait(job, self._room_text(job, need, free, running))
                continue
            for other in victims:
                self._ask_to_stop(other, "preempt")
            making_room[job.id] = self._making_room.get(job.id, now)
            # what it will take is not handed to lower jobs meanwhile (bounded: the pausing jobs stop within their grace)
            free = {k: v - need.get(k, 0) for k, v in free.items()}
            self._wait(job, "Starts as soon as lower-priority work has paused")
            if now - making_room[job.id] > WATCHDOG_S:  # the jobs asked to pause should have stopped long ago
                alerts[job.id] = f"“{title_of(job)}” has waited {policy.duration_text(now - making_room[job.id])} for lower-priority work to pause"
        self._making_room = making_room
        for job in queued:  # the watchdog: a job that fits (or can be made to) should not wait long
            if job.status == "queued" and job.id not in alerts and not held and now - self._queued_since(job) > WATCHDOG_S \
                    and not job.waiting.startswith(("Waits for “", "Waits for room", "Waits for its disk")):
                alerts[job.id] = f"“{title_of(job)}” has waited {policy.duration_text(now - self._queued_since(job))}: {job.waiting or 'not started'}"
        for job_id, text in alerts.items():
            if self.alerts.get(job_id) != text and job_id not in self.alerts:
                LOG.warning("queue watchdog: %s", text)
        self.alerts = alerts

    def _queued_since(self, job: Job) -> float:
        ended = [r[1] for r in job.runs if r and r[1]]
        return max([job.created or 0.0, *ended])

    def _blocked(self, job: Job, blocking: list[Job], steps: dict, wait_reason) -> str:
        """Why a waiting job cannot start whatever the room: a job it was queued after, a step it needs the outputs of,
        or a writer of the same outputs that runs or was queued before it (waits only ever point at running or earlier
        jobs, so they cannot go round in a circle)."""
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
            writing = other.status == "running" or (other.status == "queued" and other.created < job.created)  # a paused job lets go
            if writing and (other.status == "running" or other.workflow == job.workflow or other.command or job.command) and policy.conflict(job, other):
                return f"Waits for “{title_of(other)}”: both write the same outputs"
        return ""

    def outside(self) -> dict:
        """What work outside the queue (other programs, runs started by hand) leaves for it, looked at every 10 s: free
        memory (GB) and the GPUs whose memory is mostly taken by processes the queue did not start."""
        when, found = self._outside
        if time.time() - when < 10:
            return found
        found = {"ram_gb": self.memory().get("available_gb"), "gpus": set()}
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

    def _room_text(self, job: Job, need: dict, free: dict, running: list[Job]) -> str:
        """Why a job that does not fit waits: what it needs and what is free, and who holds the rest — work of the same
        or a higher priority, lower-priority work that cannot be paused, or (when no job holds it) programs outside the
        queue."""
        short = [k for k in policy.KEYS if need[k] > free.get(k, 0) + 1e-9] or (["ram_gb"] if need.get("ram_free", 0) > free.get("ram_free", 0) else [])
        if not short:
            return "Waits for its disk: other io-heavy jobs use it"
        names = {"cpus": "processors", "gpus": "GPUs", "ram_gb": "GB of memory"}
        text = ", ".join(f"{need[k]:g} {names[k]} ({max(0, min(free.get(k, 0), free.get('ram_free', free.get(k, 0))) if k == 'ram_gb' else free.get(k, 0)):.0f} free)"
                         for k in short)
        rank = RANK.get(job_class(job), len(CLASSES))
        holders = [j for j in running if any(self._holds(j).get(k, 0) > 0 for k in short)]
        if any(RANK.get(job_class(j), len(CLASSES)) <= rank for j in holders):
            who = "held by work of the same or a higher priority"
        elif holders:
            who = "held by lower-priority work that cannot be paused now"
        elif short == ["gpus"]:
            who = "the GPUs are used by programs outside the queue"
        elif "ram_gb" in short:
            who = "the rest of the memory is used by the app and other programs on this computer, not by queued work"
        else:
            who = "this computer has no more for jobs"
        return f"Waits for room: needs {text} ({who})"

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
        env |= {JOB_ENV: job.id, CONTROL_ENV: self.files(job)["control"], PROGRESS_FILE_ENV: str(progress_file(self.files(job)["log"])), "STAINID_JOB_CPUS": str(max(1, int(need.get("cpus") or 1))),
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
        chunks = [self.jobs[c] for c in job.children if c in self.jobs] if job.children else []
        if job.children:
            progress = job.outcome if job.status in FINAL and job.outcome else self._array_progress(job, chunks)
        else:
            progress = job.outcome if job.status in FINAL and job.outcome else self.progress_of(job)
        estimate = self.estimate(job)
        out = {**asdict(job), "env": {}, "title": title_of(job), "class": job_class(job), "class_title": CLASS_TITLES[job_class(job)],
               "rank": RANK[job_class(job)], "needs": job_needs(job), "progress": progress, "queue": None, "eta_s": None,
               "estimate_s": round(estimate) if estimate is not None else None, "size": policy.size_of(job_needs(job), estimate),
               "io": policy.io_of(job, job_needs(job)), "children": list(job.children),
               "elapsed_s": round(time.time() - job.started) if job.status == "running" and job.started else None, "overdue": self.overdue(job)}
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
            use = self.cpu_use(job)
            out["cpus_now"] = round(use, 1) if use is not None else None
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
        budget = self.current_budget()
        running = [j for j in jobs if j.status == "running" and not j.children]
        used, _ = self._used(running)
        holds = [self._holds(j) for j in running]
        used["cpus"] = round(used["cpus"], 1)
        used["cpus_declared"] = round(sum(h["cpus_declared"] for h in holds), 1)
        used["cpus_now"] = round(sum(h["cpus_now"] for h in holds), 1)
        memory = None
        if self.watch_outside and (found := self.memory()).get("total_gb"):
            free = self._free(budget, running)
            memory = {"total_gb": round(found["total_gb"], 1), "used_gb": round(found["total_gb"] - found["available_gb"], 1),
                      "available_gb": round(found["available_gb"], 1), "app_gb": round(found.get("app_gb", 0.0), 1),
                      "jobs_gb": round(sum(self._holds(j)["ram_now"] for j in running), 1), "jobs_counted_gb": round(used["ram_gb"], 1),
                      "for_jobs_gb": budget.ram_gb, "free_for_jobs_gb": round(max(0.0, min(free["ram_gb"], free.get("ram_free", free["ram_gb"]))), 1),
                      "headroom_gb": APP_HEADROOM_GB}
        return {"classes": [{"id": c, "title": CLASS_TITLES[c], "rank": RANK[c], "jobs": [self.describe(j) for j in live if job_class(j) == c]}
                            for c in CLASSES],
                "recent": [self.describe(j) for j in recent], "paused_all": self.paused_all(), "led": self.leader or self.led_elsewhere(),
                "budget": asdict(budget), "used": used, "memory": memory,
                "alerts": [{"id": k, "text": v} for k, v in self.alerts.items()],
                "outside": {**(found := self.outside()), "gpus": sorted(found["gpus"])} if self.watch_outside else None}


def _cgroup_of(pid: int) -> str:
    try:
        for line in Path(f"/proc/{pid}/cgroup").read_text().splitlines():
            if line.startswith("0::"):
                return line[3:].strip()
    except OSError:
        pass
    return ""


def _cgroup_memory_gb(group: str) -> float | None:
    """A cgroup's anonymous and shared memory (what it would lose; page cache is left out, it can be given back)."""
    try:
        stat = dict(line.split() for line in Path(f"/sys/fs/cgroup{group}/memory.stat").read_text().splitlines())
        return (int(stat.get("anon", 0)) + int(stat.get("shmem", 0))) / 2**30
    except (OSError, ValueError):
        return None


def _cgroup_cpu_seconds(group: str) -> float | None:
    """The processor seconds a cgroup's processes have used, ended ones included."""
    try:
        for line in Path(f"/sys/fs/cgroup{group}/cpu.stat").read_text().splitlines():
            if line.startswith("usage_usec"):
                return int(line.split()[1]) / 1e6
    except (OSError, ValueError, IndexError):
        pass
    return None


def _process_tree() -> dict[int, list[int]]:
    """Every process's children (pid -> [pid, ...])."""
    children: dict[int, list[int]] = {}
    for entry in Path("/proc").iterdir():
        if entry.name.isdigit():
            try:
                parent = int((entry / "stat").read_text().rsplit(")", 1)[1].split()[1])
            except (OSError, ValueError, IndexError):
                continue
            children.setdefault(parent, []).append(int(entry.name))
    return children


_TICKS = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100


def _tree_cpu_seconds(pids: tuple[int, ...], children: dict[int, list[int]]) -> float | None:
    """The processor seconds of these processes and all their children (and of the children they waited for)."""
    todo, seen, ticks = list(pids), set(), 0
    while todo:
        pid = todo.pop()
        if pid in seen:
            continue
        try:
            fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
            ticks += sum(int(fields[i]) for i in (11, 12, 13, 14))  # utime stime cutime cstime
        except (OSError, ValueError, IndexError):
            continue
        seen.add(pid)
        todo += children.get(pid, [])
    return ticks / _TICKS if seen else None


def _tree_rss_gb(pids: tuple[int, ...]) -> float | None:
    """The resident memory of these processes and all their children."""
    children: dict[int, list[int]] = {}
    for entry in Path("/proc").iterdir():
        if entry.name.isdigit():
            try:
                parent = int((entry / "stat").read_text().rsplit(")", 1)[1].split()[1])
            except (OSError, ValueError, IndexError):
                continue
            children.setdefault(parent, []).append(int(entry.name))
    todo, seen, total = list(pids), set(), 0
    while todo:
        pid = todo.pop()
        if pid in seen:
            continue
        seen.add(pid)
        try:
            total += int(next(line for line in Path(f"/proc/{pid}/status").read_text().splitlines() if line.startswith("VmRSS")).split()[1])
        except (OSError, StopIteration, ValueError, IndexError):
            continue
        todo += children.get(pid, [])
    return total / 2**20 if seen else None


_APP_GROUP: list = [0.0, ""]


def _app_memory_gb() -> float:
    """What the web app uses now: its systemd unit's memory, else this process's (when the queue runs inside it)."""
    group = _cgroup_of(os.getpid())
    if "stainid-app" not in group:
        if time.time() - _APP_GROUP[0] > 60:
            _APP_GROUP[0] = time.time()
            try:
                shown = subprocess.run(["systemctl", "--user", "show", "-p", "ControlGroup", "--value", "stainid-app.service"],
                                       capture_output=True, text=True, timeout=5).stdout.strip()
            except (OSError, subprocess.SubprocessError):
                shown = ""
            _APP_GROUP[1] = shown if shown and "stainid-app" in shown else ""
        group = _APP_GROUP[1]
    if group:
        found = _cgroup_memory_gb(group)
        if found is not None:
            return found
    found = _tree_rss_gb((os.getpid(),)) if not group else None
    return found or 0.0


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
