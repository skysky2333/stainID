"""`stainid jobs ...`: the shared job queue from the command line (see `stainid.jobs.queue`).

    stainid --project P jobs submit [--class C] [--cpus N --gpus N --ram GB --network] -- nuclei --stain AT8
    stainid --project P jobs submit -- score                       # any stainid command of the project
    stainid jobs submit --exec --class downloads --network --cwd DIR --title "Big downloads" -- bash download_big.sh
    stainid jobs list [--all]          stainid jobs show ID
    stainid jobs pause|resume|cancel ID...
    stainid jobs priority ID (--up | --down | --top | --class C)
    stainid jobs pause-all | resume-all
    stainid jobs budget [--cpus N --gpus N --ram GB]
    stainid jobs serve                 # run the queue without the app (only one process runs it at a time)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def add_parser(sub) -> None:
    p = sub.add_parser("jobs", help="the shared job queue: submit, list, pause, resume, cancel, reorder; `jobs serve` runs it")
    actions = p.add_subparsers(dest="action", required=True)
    s = actions.add_parser("submit", help="queue a workflow step or any command (after --)")
    s.add_argument("--class", dest="klass", default=None, choices=("annotation", "pipeline", "models", "public", "downloads"),
                   help="priority class (default: from the step and the project; a command: pipeline)")
    s.add_argument("--cpus", type=float, default=None)
    s.add_argument("--gpus", type=int, default=None)
    s.add_argument("--ram", type=float, default=None, help="memory it needs, in GB")
    s.add_argument("--network", action="store_true", help="network work only: takes no processors from the budget")
    s.add_argument("--exec", action="store_true", help="the words after -- are a command of their own, not a stainid command")
    s.add_argument("--cwd", default=None, help="where an --exec command runs (default: the project folder, else here)")
    s.add_argument("--title", default="", help="what the Run page calls it")
    s.add_argument("--grace", type=float, default=0.0, help="seconds it gets to reach a unit boundary when paused (default: its class's)")
    s.add_argument("--env", action="append", default=[], metavar="KEY=VALUE", help="extra environment for the command")
    s.add_argument("--after", action="append", default=[], metavar="ID", help="start only once this job has finished (repeat for several)")
    s.add_argument("--id-only", action="store_true", help="print only the new job's id (for scripts: --after $(...))")
    s.add_argument("words", nargs=argparse.REMAINDER)
    p_list = actions.add_parser("list", help="the queue by priority class (with --all also what ended in the last day)")
    p_list.add_argument("--all", action="store_true")
    p_list.add_argument("--json", action="store_true")
    actions.add_parser("show", help="one job's record and the end of its log").add_argument("id")
    for name in ("pause", "resume", "cancel"):
        actions.add_parser(name, help=f"{name} jobs").add_argument("ids", nargs="+")
    pr = actions.add_parser("priority", help="move a waiting job within its class, or give it another class")
    pr.add_argument("id")
    where = pr.add_mutually_exclusive_group(required=True)
    where.add_argument("--up", action="store_true")
    where.add_argument("--down", action="store_true")
    where.add_argument("--top", action="store_true")
    where.add_argument("--class", dest="klass", choices=("annotation", "pipeline", "models", "public", "downloads"))
    actions.add_parser("pause-all", help="pause every running job between two units and start nothing new")
    actions.add_parser("resume-all", help="undo pause-all")
    b = actions.add_parser("budget", help="show or set what all jobs together may use")
    b.add_argument("--cpus", type=float, default=None)
    b.add_argument("--gpus", type=int, default=None)
    b.add_argument("--ram", type=float, default=None)
    actions.add_parser("serve", help="run the queue in this process (when the app is not running it)")


def _duration(seconds) -> str:
    if seconds is None:
        return ""
    seconds = int(seconds)
    return f"{seconds // 3600}h{seconds % 3600 // 60:02d}m" if seconds >= 3600 else f"{seconds // 60}m{seconds % 60:02d}s"


def _line(info: dict) -> str:
    progress = info.get("progress") or {}
    done = f"{progress.get('done', 0)}/{progress.get('total', 0)}" if progress.get("total") else ""
    state = info["status"] + (f" ({info['stopping']}ing)" if info.get("stopping") else "")
    where = f"#{info['queue']['position']}" if info.get("queue") else ""
    eta = f"~{_duration(info['eta_s'])} left" if info.get("eta_s") else ""
    note = info.get("waiting") or ""
    return f"  {info['id']}  {state:<12} {where:<4} {info['title'][:34]:<34} {info.get('project_name', '')[:28]:<28} {done:<10} {eta:<12} {note}"


def run(args, project_path: Path | None) -> None:
    from stainid.jobs.queue import Budget, Scheduler, queue_dir

    if args.action == "serve":
        scheduler = Scheduler(queue_dir(), lead=True)
        if not scheduler.leader:
            print("Another process runs the queue now (the app, or another `stainid jobs serve`); this one takes over when it stops",
                  flush=True)
        print(f"Running the job queue in {scheduler.dir} (Ctrl+C to stop; running jobs carry on)", flush=True)
        scheduler.start_thread()
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            scheduler.close()
        return
    scheduler = Scheduler(queue_dir(), lead=False)
    if args.action == "submit":
        _submit(scheduler, args, project_path)
    elif args.action == "list":
        overview = scheduler.overview()
        if args.json:
            print(json.dumps(overview, indent=1, default=str))
            return
        print(f"Queue {scheduler.dir}: " + ("run by " + ("this process" if scheduler.leader else "the app or `stainid jobs serve`")
                                              if overview["led"] else "NOT RUNNING: start the app or `stainid jobs serve`")
              + (" · ALL BACKGROUND WORK PAUSED" if overview["paused_all"] else ""))
        b, used = overview["budget"], overview["used"]
        print(f"Budget: {used['cpus']:g}/{b['cpus']:g} processors, {used['gpus']}/{b['gpus']} GPUs, {used['ram_gb']:g}/{b['ram_gb']:g} GB in use")
        for group in overview["classes"]:
            if group["jobs"]:
                print(f"{group['rank']}. {group['title']}")
                for info in group["jobs"]:
                    print(_line(info))
        if args.all and overview["recent"]:
            print("Ended in the last day")
            for info in overview["recent"]:
                print(_line(info))
    elif args.action == "show":
        job = scheduler.jobs.get(args.id)
        if job is None:
            raise SystemExit(f"No job {args.id} in {scheduler.dir}")
        info = scheduler.describe(job)
        print(json.dumps(info, indent=1, default=str))
        log = Path(job.log_file) if job.log_file else None
        if log is not None and log.exists():
            print("--- end of", log)
            print("\n".join(log.read_text(errors="replace").splitlines()[-20:]))
    elif args.action in ("pause", "resume", "cancel"):
        for job_id in args.ids:
            if job_id not in scheduler.jobs:
                print(f"No job {job_id}", file=sys.stderr)
                continue
            scheduler.request(args.action, job_id)
            print(f"{args.action}: {job_id}")
        _hint(scheduler)
    elif args.action == "priority":
        if args.klass:
            scheduler.request("class", args.id, klass=args.klass)
        else:
            scheduler.request("move", args.id, where="up" if args.up else "down" if args.down else "top")
        _hint(scheduler)
    elif args.action in ("pause-all", "resume-all"):
        scheduler.request(args.action.replace("-", "_"))
        _hint(scheduler)
    elif args.action == "budget":
        path = scheduler.dir / "budget.json"
        try:
            saved = json.loads(path.read_text())
        except (OSError, ValueError):
            saved = {}
        for key, value in (("cpus", args.cpus), ("gpus", args.gpus), ("ram_gb", args.ram)):
            if value is not None:
                saved[key] = value
        if any(v is not None for v in (args.cpus, args.gpus, args.ram)):
            path.write_text(json.dumps(saved, indent=1))
        b: Budget = scheduler.budget
        print(f"{b.cpus:g} processors, {b.gpus} GPUs, {b.ram_gb:g} GB of memory for all jobs together ({path})")


def _hint(scheduler) -> None:
    if not scheduler.led_elsewhere():
        print("Note: no process runs the queue now; it happens once the app or `stainid jobs serve` runs", file=sys.stderr)


def _submit(scheduler, args, project_path: Path | None) -> None:
    import os

    from stainid.project import load_project

    words = list(args.words)
    if words and words[0] == "--":
        words = words[1:]
    if not words:
        raise SystemExit("Nothing to submit: put the command after --")
    project = None
    if not args.exec or project_path is not None or os.environ.get("STAINID_PROJECT"):
        try:
            project = load_project(project_path)
        except Exception as error:  # noqa: BLE001
            if not args.exec:
                raise SystemExit(f"No project: {error}")
    needs = {k: v for k, v in (("cpus", args.cpus), ("gpus", args.gpus), ("ram_gb", args.ram)) if v is not None}
    if args.network:
        needs["network"] = True
    env = dict(item.split("=", 1) for item in args.env if "=" in item)
    if args.exec:
        cwd = args.cwd or (str(project.root) if project is not None else os.getcwd())
        job = scheduler.submit_command(words, cwd, args.title or " ".join(words)[:60], args.klass or "pipeline", needs, project, env, args.grace,
                                       args.after)
    else:
        from stainid.cli import _workflow_step, build_parser

        step, _ = _workflow_step(words)
        if step is not None and not env and not args.grace:
            parsed = build_parser().parse_args(["--project", str(project.root), *words])
            options = {o.key: o.default if getattr(parsed, o.key, None) is None else getattr(parsed, o.key) for o in step.options}
            job = scheduler.submit_step(project, step.id, options, args.klass, {**_step_needs(step.id, options), **needs} if needs else None,
                                        argv=words, after=args.after)
            if args.title:
                job.title = args.title
                scheduler.store.save(job)
        else:
            from stainid.jobs.queue import class_of

            command = [sys.executable, "-u", "-m", "stainid.cli", "--project", str(project.root), *words]
            klass = args.klass or (class_of(project, step.id) if step is not None else class_of(project, "command"))
            job = scheduler.submit_command(command, str(project.root), args.title or f"stainid {' '.join(words)}"[:60], klass, needs, project,
                                           env, args.grace, args.after)
            if step is not None:
                job.workflow = step.id
                scheduler.store.save(job)
    if args.id_only:
        print(job.id)
        return
    info = scheduler.describe(job)
    print(f"Queued {job.id} ({info['class_title']}, place {info['queue']['position']} of {info['queue']['in_class']}): {info['title']}")
    _hint(scheduler)


def _step_needs(workflow: str, options: dict) -> dict:
    from stainid.jobs.queue import needs_of

    return needs_of(workflow, options)
