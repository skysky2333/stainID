"""`stainid` command line interface."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from stainid.project import load_project, write_default_config


def _shard(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="stainid", description="Stain-level morphology for brain tissue microarrays.")
    parser.add_argument("--project", type=Path, default=None, help="project folder or stainid.yaml (default: $STAINID_PROJECT or cwd)")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init", help="write a default stainid.yaml in the project folder")
    sub.add_parser("info", help="show resolved project paths and whether they exist")

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
    _shard(p)

    p = sub.add_parser("fields", help="6E10 plaque and AT8 tau field pipelines")
    p.add_argument("--stain", action="append", choices=("6E10", "AT8"))
    _shard(p)

    p = sub.add_parser("masks", help="SAM object outlines for accepted objects")
    p.add_argument("--stain", required=True, choices=("NeuN", "6E10", "AT8"))
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=4)
    _shard(p)

    p = sub.add_parser("aggregate", help="core / donor-region feature tables")
    p.add_argument("what", choices=("fields", "masks"))
    p.add_argument("--level", choices=("sample_region_id", "core_id"), default="sample_region_id")
    p.add_argument("--output", type=Path)

    p = sub.add_parser("serve", help="start the web app")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--reload", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "init":
        print(write_default_config(args.project or Path.cwd()))
        return
    project = load_project(args.project)
    os.chdir(project.root)
    if args.command == "info":
        print(f"project: {project.name}  root: {project.root}")
        for section in ("inputs", "models", "outputs"):
            for key in project.config[section]:
                path = project._resolve(section, key)
                print(f"  {section}.{key:<18} {'ok ' if path.exists() else '-- '} {path}")
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
        else:
            run_neun(project, args.device, args.threads, args.batch_size, args.shard_index, args.shard_count)
    elif args.command == "fields":
        from stainid.workflows.fields import run_fields

        run_fields(project, args.stain or ["6E10", "AT8"], args.shard_index, args.shard_count)
    elif args.command == "masks":
        from stainid.workflows.masks import run_masks

        run_masks(project, args.stain, args.device, args.threads, args.shard_index, args.shard_count)
    elif args.command == "aggregate":
        from stainid.workflows.aggregate import aggregate_fields, aggregate_masks

        (aggregate_fields if args.what == "fields" else aggregate_masks)(project, args.level, args.output)
    elif args.command == "serve":
        import uvicorn

        os.environ["STAINID_PROJECT"] = str(project.root)
        uvicorn.run("stainid.api.app:create_app", factory=True, host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
