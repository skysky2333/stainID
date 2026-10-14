"""`stainid` command line interface. Every web-app step runs one of these commands."""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from stainid.jobs import JOB_ENV, JobStore, command_line_job
from stainid.project import load_project, write_default_config


def _shard(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)


def _follow(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--follow", action="store_true",
                        help="run alongside Find nuclei: detect each core image as soon as its nuclei exist, until all are done")


def _fresh(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--fresh", action="store_true", help="move existing results aside and start over")


def archive(path: Path) -> None:
    """Move a results folder aside (never deleted) so a step can start over."""
    if path.exists():
        target = path.with_name(f"{path.name}_archived_{time.strftime('%Y%m%d-%H%M%S')}")
        path.rename(target)
        print(f"Previous results moved to {target}", flush=True)


def archive_neun(folder: Path) -> None:
    """Start NeuN over but keep its Cellpose cache (the slow part, which does not depend on the model)."""
    target = folder.with_name(f"{folder.name}_archived_{time.strftime('%Y%m%d-%H%M%S')}")
    for name in ("tiles", "provenance.json", "tile_features.csv", "objects.csv"):
        if (folder / name).exists():
            target.mkdir(exist_ok=True)
            (folder / name).rename(target / name)
    print(f"Previous NeuN results moved to {target}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="stainid", description="Stain-level morphology for brain tissue microarrays.")
    parser.add_argument("--project", type=Path, default=None, help="project folder or stainid.yaml (default: $STAINID_PROJECT or cwd)")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init", help="write a default stainid.yaml in the project folder")
    sub.add_parser("info", help="show resolved project paths and whether they exist")
    sub.add_parser("status", help="show every workflow step and how far it has got")

    p = sub.add_parser("dearray", help="find the cores on every slide in the slides table")
    p.add_argument("--redo", action="store_true")
    p.add_argument("--slide", action="append", default=None, help="only this slide (file name; repeat for several), found again")
    sub.add_parser("layout", help="attach the TMA map (donor, region, group) to the core table")
    for name, text in (("export", "export native-resolution core images"), ("qc", "core tissue and focus QC")):
        p = sub.add_parser(name, help=text)
        p.add_argument("--workers", type=int, default=2)
        p.add_argument("--overwrite", action="store_true")

    p = sub.add_parser("select", help="core image table and analysis-field selection")
    p.add_argument("--fields-per-core", type=int, default=8)
    p.add_argument("--primary", type=int, default=4, help="first N fields per stain-core form the analysis manifest")
    p.add_argument("--field-size", type=int, default=2048)

    p = sub.add_parser("tiles", help="cut every core into tissue tiles for whole-core analysis")
    p.add_argument("--field-size", type=int, default=2048)

    p = sub.add_parser("calibrate", help="per-slide DAB thresholds (TMA x stain)")
    p.add_argument("--stain", action="append")
    p.add_argument("--fields-per-core", type=int, default=1)
    p.add_argument("--output", type=Path)

    p = sub.add_parser("nuclei", help="Cellpose-SAM nuclei for analysis fields")
    p.add_argument("--stain", action="append", required=True)
    p.add_argument("--device", default="mps")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--workers", type=int, default=2, help="tiles in flight at once (one model); keeps the GPU busy")
    _shard(p)

    p = sub.add_parser("neun", help="NeuN neuron detection")
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--merge", action="store_true", help="merge finished tiles into cohort tables")
    _follow(p)
    _fresh(p)
    _shard(p)

    p = sub.add_parser("fields", help="6E10 plaque and AT8 tau field pipelines")
    p.add_argument("--stain", action="append", choices=("6E10", "AT8"))
    _follow(p)
    _fresh(p)
    _shard(p)

    p = sub.add_parser("masks", help="SAM object outlines for accepted objects")
    p.add_argument("--stain", action="append", choices=("NeuN", "6E10", "AT8"))
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=4)
    _fresh(p)
    _shard(p)

    p = sub.add_parser("aggregate", help="core / donor-region feature tables for one kind of output")
    p.add_argument("what", choices=("fields", "masks", "neun"))
    p.add_argument("--level", choices=("sample_region_id", "core_id"), default="sample_region_id")
    p.add_argument("--output", type=Path)

    p = sub.add_parser("summarize", help="one results table with every stain (results_<level>.csv)")
    p.add_argument("--level", choices=("sample_region_id", "core_id"), default="sample_region_id")

    p = sub.add_parser("labels", help="prepare candidate objects for labelling, or import earlier label sets")
    p.add_argument("action", choices=("prepare", "import"))
    p.add_argument("--stain", choices=("NeuN", "6E10", "AT8"))
    p.add_argument("--fields", type=int, default=0, help="prepare this many more fields that are not prepared yet")
    p.add_argument("--session", default="", help="prepare every field of this labelling session")
    p.add_argument("--seed", type=int, default=0)

    p = sub.add_parser("train", help="train a stain model from the labels in the label store")
    p.add_argument("--stain", required=True, choices=("NeuN", "6E10", "AT8"))

    p = sub.add_parser("download-models", help="download the public model weights (Cellpose-SAM, SAM, Phikon)")
    p.add_argument("--which", action="append", choices=("cellpose", "sam", "huggingface_home"))

    p = sub.add_parser("serve", help="start the web app")
    p.add_argument("--host", default="0.0.0.0", help="0.0.0.0 (default) lets other computers reach the annotation pages; 127.0.0.1 keeps it on this computer")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--no-browser", action="store_true", help="do not open the web browser")
    p.add_argument("--reload", action="store_true")
    p.add_argument("--allow-origin", action="append", default=[], metavar="URL",
                   help="let the web app served from this address (e.g. http://lab-pc:8080) use this server; repeat for several "
                        "(also $STAINID_ALLOW_ORIGINS, comma-separated)")
    return parser


def _workflow_step(argv: list[str]):
    """The workflow step a command runs (the one whose `command` starts its arguments, else None), and the arguments
    without --project, as the app shows them."""
    from stainid.workflows.steps import STEPS

    words = [w for i, w in enumerate(argv) if not w.startswith("--project") and not (i and argv[i - 1] == "--project")]
    return next((s for s in STEPS if s.command and tuple(words[:len(s.command)]) == s.command), None), words


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else list(argv)
    args = build_parser().parse_args(argv)
    if args.command == "init":
        print(write_default_config(args.project or Path.cwd()))
        return
    if args.command == "serve":
        serve(args)
        return
    project = load_project(args.project)
    os.chdir(project.root)
    step, words = _workflow_step(argv)
    if step is None or os.environ.get(JOB_ENV):  # not a workflow step, or started by the app (which keeps its own record)
        run(args, project)
        return
    options = {o.key: o.default if getattr(args, o.key, None) is None else getattr(args, o.key) for o in step.options}
    with command_line_job(JobStore(project.output("jobs")), step.id, options, words):
        run(args, project)


def run(args: argparse.Namespace, project) -> None:
    if args.command == "info":
        print(f"project: {project.name}  root: {project.root}")
        for section in ("inputs", "models", "outputs"):
            for key in project.config[section]:
                path = project.resolve(section, key)
                print(f"  {section}.{key:<18} {'ok ' if path.exists() else '-- '} {path}")
    elif args.command == "status":
        from stainid.workflows.steps import describe

        for step in describe(project):
            progress = step["progress"]
            print(f"{step['stage']:<12} {step['title']:<28} {progress['state']:<8} {progress['done']}/{progress['total']} {progress['unit']}")
    elif args.command in ("dearray", "layout", "export", "qc"):
        from stainid.workflows import slides

        if args.command == "dearray":
            slides.dearray_slides(project, args.redo, only=args.slide)
        elif args.command == "layout":
            slides.attach_map(project)
        elif args.command == "export":
            slides.export(project, args.overwrite, args.workers)
        else:
            slides.core_qc(project, args.overwrite, args.workers)
    elif args.command == "select":
        from stainid.workflows.select import select_fields

        for path in select_fields(project, args.fields_per_core, args.primary, args.field_size):
            print(path)
    elif args.command == "tiles":
        from stainid.sampling.whole_core import write_whole_core_manifest

        write_whole_core_manifest(project, args.field_size)
    elif args.command == "calibrate":
        from stainid.workflows.calibrate import calibrate_slides

        print(calibrate_slides(project, args.stain, args.fields_per_core, output=args.output))
    elif args.command == "nuclei":
        from stainid.workflows.nuclei import run_nuclei

        run_nuclei(project, args.stain, args.device, args.batch_size, args.threads, args.shard_index, args.shard_count, args.workers)
    elif args.command == "neun":
        from stainid.workflows.neun import follow_neun, merge_neun, run_neun

        if args.merge:
            for path in merge_neun(project):
                print(path)
            return
        if args.fresh:
            archive_neun(project.output("neun"))
        if args.follow:
            follow_neun(project, args.device, args.threads, args.shard_index, args.shard_count)
        else:
            run_neun(project, args.device, args.threads, args.batch_size, args.shard_index, args.shard_count)
    elif args.command == "fields":
        from stainid.workflows.fields import follow_fields, run_fields

        if args.fresh:
            archive(project.output("fields"))
        (follow_fields if args.follow else run_fields)(project, args.stain or ["6E10", "AT8"], args.shard_index, args.shard_count)
    elif args.command == "masks":
        from stainid.workflows.masks import run_masks

        for stain in args.stain or ["NeuN", "6E10", "AT8"]:
            if args.fresh:
                archive(project.output("masks") / stain)
            run_masks(project, stain, args.device, args.threads, args.shard_index, args.shard_count)
    elif args.command == "aggregate":
        from stainid.workflows.aggregate import aggregate_fields, aggregate_masks, aggregate_neun

        {"fields": aggregate_fields, "masks": aggregate_masks, "neun": aggregate_neun}[args.what](project, args.level, args.output)
    elif args.command == "summarize":
        from stainid.workflows.aggregate import summarize_results

        summarize_results(project, args.level)
    elif args.command == "labels":
        from stainid.labels.prepare import run

        run(project, args.action, args.stain, args.fields, args.session, args.seed)
    elif args.command == "download-models":
        from stainid.workflows.downloads import download_models

        download_models(project, args.which)
    elif args.command == "train":
        from stainid.training.train import train_model

        train_model(project, args.stain)


def serve(args: argparse.Namespace) -> None:
    import socket
    import threading
    import webbrowser

    import uvicorn

    from stainid.api.session import allowed_origins
    from stainid.api.state import remembered_project

    local = "127.0.0.1" if args.host == "0.0.0.0" else args.host
    url = f"http://{local}:{args.port}"
    with socket.socket() as probe:
        if probe.connect_ex((local, args.port)) == 0:
            print(f"stainID is already running at {url}; opening it", flush=True)
            if not args.no_browser:
                webbrowser.open(url)
            return
    os.environ["STAINID_PROJECT"] = str(Path(args.project).resolve() if args.project else remembered_project())
    os.environ["STAINID_HOST"] = args.host
    origins = allowed_origins(args.allow_origin)
    os.environ["STAINID_ALLOW_ORIGINS"] = ",".join(origins)
    if not args.no_browser:
        threading.Timer(1.5, webbrowser.open, (url,)).start()
    print(f"stainID is running at {url}  (press Ctrl+C to stop)", flush=True)
    if args.host != "127.0.0.1":
        print(f"Annotators on your network: http://{socket.gethostname()}:{args.port}/annotate (once you start a session in the app)", flush=True)
    if origins:
        print(f"The web app served from {', '.join(origins)} may use this server", flush=True)
    uvicorn.run("stainid.api.app:create_app", factory=True, host=args.host, port=args.port, reload=args.reload, log_level="warning")


if __name__ == "__main__":
    main()
