"""`stainid` command line interface. Every web-app step runs one of these commands."""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from stainid.project import load_project, write_default_config


def _shard(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)


def _fresh(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--fresh", action="store_true", help="move existing results aside and start over")


def archive(path: Path) -> None:
    """Move a results folder aside (never deleted) so a step can start over."""
    if path.exists():
        target = path.with_name(f"{path.name}_archived_{time.strftime('%Y%m%d-%H%M%S')}")
        path.rename(target)
        print(f"Previous results moved to {target}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="stainid", description="Stain-level morphology for brain tissue microarrays.")
    parser.add_argument("--project", type=Path, default=None, help="project folder or stainid.yaml (default: $STAINID_PROJECT or cwd)")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init", help="write a default stainid.yaml in the project folder")
    sub.add_parser("info", help="show resolved project paths and whether they exist")
    sub.add_parser("status", help="show every workflow step and how far it has got")

    p = sub.add_parser("dearray", help="find the cores on every slide in the slides table")
    p.add_argument("--redo", action="store_true")
    sub.add_parser("layout", help="attach the TMA map (donor, region, group) to the core table")
    for name, text in (("export", "export native-resolution core images"), ("qc", "core tissue and focus QC")):
        p = sub.add_parser(name, help=text)
        p.add_argument("--workers", type=int, default=2)
        p.add_argument("--overwrite", action="store_true")

    p = sub.add_parser("select", help="core image table and analysis-field selection")
    p.add_argument("--fields-per-core", type=int, default=8)
    p.add_argument("--primary", type=int, default=4, help="first N fields per stain-core form the analysis manifest")
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
    _shard(p)

    p = sub.add_parser("neun", help="NeuN neuron detection")
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--merge", action="store_true", help="merge finished tiles into cohort tables")
    _fresh(p)
    _shard(p)

    p = sub.add_parser("fields", help="6E10 plaque and AT8 tau field pipelines")
    p.add_argument("--stain", action="append", choices=("6E10", "AT8"))
    _fresh(p)
    _shard(p)

    p = sub.add_parser("masks", help="SAM object outlines for accepted objects")
    p.add_argument("--stain", required=True, choices=("NeuN", "6E10", "AT8"))
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

    p = sub.add_parser("training-set", help="sample candidate objects to label for model training")
    p.add_argument("--stain", required=True, choices=("NeuN", "6E10", "AT8"))
    p.add_argument("--name", required=True)
    p.add_argument("--fields", type=int, default=12)
    p.add_argument("--per-field", type=int, default=10)
    p.add_argument("--strategy", choices=("uncertain", "random"), default="uncertain")
    p.add_argument("--enrich", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--seed", type=int, default=0)

    p = sub.add_parser("train", help="train a stain model from labelled training sets")
    p.add_argument("--stain", required=True, choices=("NeuN", "6E10", "AT8"))

    p = sub.add_parser("download-models", help="download the public model weights (Cellpose-SAM, SAM, Phikon)")
    p.add_argument("--which", action="append", choices=("cellpose", "sam", "huggingface_home"))

    p = sub.add_parser("serve", help="start the web app")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--no-browser", action="store_true", help="do not open the web browser")
    p.add_argument("--reload", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "init":
        print(write_default_config(args.project or Path.cwd()))
        return
    if args.command == "serve":
        serve(args)
        return
    project = load_project(args.project)
    os.chdir(project.root)
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
            slides.dearray_slides(project, args.redo)
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
    elif args.command == "calibrate":
        from stainid.workflows.calibrate import calibrate_slides

        print(calibrate_slides(project, args.stain, args.fields_per_core, output=args.output))
    elif args.command == "nuclei":
        from stainid.workflows.nuclei import run_nuclei

        run_nuclei(project, args.stain, args.device, args.batch_size, args.threads, args.shard_index, args.shard_count)
    elif args.command == "neun":
        from stainid.workflows.neun import merge_neun, run_neun

        if args.merge:
            for path in merge_neun(project):
                print(path)
            return
        if args.fresh:
            archive(project.output("neun"))
        run_neun(project, args.device, args.threads, args.batch_size, args.shard_index, args.shard_count)
    elif args.command == "fields":
        from stainid.workflows.fields import run_fields

        if args.fresh:
            archive(project.output("fields"))
        run_fields(project, args.stain or ["6E10", "AT8"], args.shard_index, args.shard_count)
    elif args.command == "masks":
        from stainid.workflows.masks import run_masks

        if args.fresh:
            archive(project.output("masks") / args.stain)
        run_masks(project, args.stain, args.device, args.threads, args.shard_index, args.shard_count)
    elif args.command == "aggregate":
        from stainid.workflows.aggregate import aggregate_fields, aggregate_masks, aggregate_neun

        {"fields": aggregate_fields, "masks": aggregate_masks, "neun": aggregate_neun}[args.what](project, args.level, args.output)
    elif args.command == "summarize":
        from stainid.workflows.aggregate import summarize_results

        summarize_results(project, args.level)
    elif args.command == "training-set":
        from stainid.training.sets import create_training_set

        create_training_set(project, args.stain, args.name, args.fields, args.per_field, args.strategy, args.enrich, args.seed)
    elif args.command == "download-models":
        from stainid.workflows.downloads import download_models

        download_models(project, args.which)
    elif args.command == "train":
        from stainid.training.train import train_model

        train_model(project, args.stain)


def serve(args: argparse.Namespace) -> None:
    import threading
    import webbrowser

    import uvicorn

    from stainid.api.state import remembered_project

    os.environ["STAINID_PROJECT"] = str(Path(args.project).resolve() if args.project else remembered_project())
    url = f"http://{args.host}:{args.port}"
    if not args.no_browser:
        threading.Timer(1.5, webbrowser.open, (url,)).start()
    print(f"stainID is running at {url}  (press Ctrl+C to stop)", flush=True)
    uvicorn.run("stainid.api.app:create_app", factory=True, host=args.host, port=args.port, reload=args.reload, log_level="warning")


if __name__ == "__main__":
    main()
