"""`stainid` command line interface. Every web-app step runs one of these commands."""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from stainid.jobs import JOB_ENV, JobStore, command_line_job
from stainid.project import CONFIG_NAME, load_project, write_default_config


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

    p = sub.add_parser("tiles", help="core image table, and every core cut into tissue tiles (detection runs on these)")
    p.add_argument("--tile-size", "--field-size", dest="tile_size", type=int, default=2048, help="side of each square tile in pixels")

    p = sub.add_parser("images", help="image sets and whole slides: core images at the analysis scale, tiles, and a dataset package's labels")
    p.add_argument("--tile-size", dest="tile_size", type=int, default=2048, help="largest side of a tile in pixels")

    p = sub.add_parser("package", help="make a project from a dataset package (dataset.yaml, images.csv, annotations.csv, donors.csv)")
    p.add_argument("package", type=Path, help="the package folder (or the dataset folder holding package/)")
    p.add_argument("folder", type=Path, help="the new project's folder")
    p.add_argument("--name", default="")

    sub.add_parser("score", help="score the active models on the labels imported from a dataset package")

    p = sub.add_parser("calibrate", help="per-slide DAB thresholds (TMA x stain) from every tile of every core")
    p.add_argument("--stain", action="append")
    p.add_argument("--redo", action="store_true", help="calibrate slides again that already have a threshold")
    p.add_argument("--output", type=Path)
    p.add_argument("--core", action="append", help="only the slides of this core image, from its tiles (repeat; a file of core ids works too)")
    p.add_argument("--workers", type=int, default=1, help="core images read at once")

    p = sub.add_parser("nuclei", help="Cellpose-SAM nuclei for every tile")
    p.add_argument("--stain", action="append", required=True)
    p.add_argument("--device", default="mps")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--workers", type=int, default=2, help="tiles in flight at once (one model); keeps the GPU busy")
    p.add_argument("--core", action="append", help="only this core image (repeat for several; a file with one core id per line works too)")
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

    p = sub.add_parser("v2", help="v2 detection: proposals, calls, outlines and measures (outputs.v2); resumable, redoes what is outdated")
    p.add_argument("stage", choices=("proposals", "calls", "outlines", "measures", "status"))
    p.add_argument("--stain", action="append", choices=("NeuN", "6E10", "AT8"))
    p.add_argument("--core", action="append", help="only this core (repeat for several; a file with one core id per line works too)")
    p.add_argument("--workers", type=int, default=0,
                   help="core images at once (0: proposals 3, sharing the tile processes; calls as many as the processors allow)")
    p.add_argument("--processes", type=int, default=-1,
                   help="proposals: tile processes, one thread each, shared by the core images in flight (-1: a quarter of the "
                        "processors, where memory bandwidth tops out, bounded by memory; 0: the older layout, one process per core "
                        "image with --threads and --tiles-at-once)")
    p.add_argument("--threads", type=int, default=0,
                   help="proposals: threads per tile process (0: 1); with --processes 0, filter threads per core image (0: the "
                        "processors the workers leave idle, at most 4)")
    p.add_argument("--tiles-at-once", type=int, default=1,
                   help="proposals with --processes 0: tiles of a core image worked on at once (threads sharing the decoded image; ~0.7 GB each)")
    p.add_argument("--device", default="", help="outlines: cuda, cuda:1 or cpu (default: a GPU when there is one)")
    p.add_argument("--redo", action="store_true", help="redo every core image, not only missing and outdated ones")
    _shard(p)

    p = sub.add_parser("aggregate", help="core / donor-region feature tables for one kind of output")
    p.add_argument("what", choices=("fields", "masks", "neun"))
    p.add_argument("--level", choices=("sample_region_id", "core_id"), default="sample_region_id")
    p.add_argument("--output", type=Path)

    p = sub.add_parser("summarize", help="one results table with every stain (results_<level>.csv)")
    p.add_argument("--level", choices=("sample_region_id", "core_id"), default="sample_region_id")

    p = sub.add_parser("labels", help="prepare candidate objects for labelling, import earlier label sets, or split an old label log")
    p.add_argument("action", choices=("prepare", "import", "store-from-pool", "split"),
                   help="store-from-pool: write a v2 candidate store from the prepared pool (to try the annotation rounds before the v2 "
                        "detection); split: once, with the app stopped, move an earlier release's labels.jsonl into one log per session "
                        "and label set (the old file is kept as labels.jsonl.before_split)")
    p.add_argument("--all", action="store_true", help="split: every project the app knows (its Projects page), not only --project")
    p.add_argument("--stain", choices=("NeuN", "6E10", "AT8"))
    p.add_argument("--fields", type=int, default=0, help="prepare this many more fields that are not prepared yet (-1: every one)")
    p.add_argument("--session", default="", help="prepare every field of this labelling session")
    p.add_argument("--seed", type=int, default=0)

    p = sub.add_parser("review", help="AI agent review batches: picture sheets for AI reviewers, and their labels into the label log")
    kinds = p.add_subparsers(dest="review_kind", required=True)
    a = kinds.add_parser("ai-batch", help="build: choose fields and write the sheets, overviews, manifest, instructions and assignments; "
                         "ingest: import the reviewers' CSVs as the session \"AI review round N · <stain>\"")
    a.add_argument("action", choices=("build", "ingest"))
    a.add_argument("--stain", action="append", choices=("NeuN", "6E10", "AT8"), help="repeat for several (default: every one)")
    a.add_argument("--round", type=int, default=4, help="the review round (its folder: <reviews>/ai_review_round<N>)")
    a.add_argument("--fields", type=int, default=8, help="build: fields per stain")
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--csv", action="append", type=Path, default=[], help="ingest: these CSVs (default: the batch's review_labels_*.csv)")
    a.add_argument("--dry-run", action="store_true", help="ingest: report what would be written, write nothing")

    p = sub.add_parser("train", help="train a stain model from the labels in the label store")
    p.add_argument("--stain", required=True, choices=("NeuN", "6E10", "AT8"))
    p.add_argument("--session", action="append", help="train only on the labels of this session (repeat for several; \"reference\" = the "
                   "imported reference labels); default: every session")

    p = sub.add_parser("score-segmentation", help="segmentation check: how well detection finds and outlines the objects people outlined")
    p.add_argument("--stain", default="all", choices=("all", "NeuN", "6E10", "AT8"))
    p.add_argument("--session", default="", help="only these outline sessions (session IDs, comma-separated); default: every one")
    p.add_argument("--proposals-only", action="store_true", help="only find the candidates of the check's fields (no scores)")
    p.add_argument("--redo", action="store_true", help="find the candidates of every field again")

    p = sub.add_parser("download-models", help="download the public model weights (Cellpose-SAM, SAM, Phikon)")
    p.add_argument("--which", action="append", choices=("cellpose", "sam", "huggingface_home"))

    from stainid.jobs.cli import add_parser as add_jobs

    add_jobs(sub)
    from stainid.api.accounts_cli import add_parser as add_accounts

    add_accounts(sub)
    from stainid.tutorial import add_parser as add_tutorial

    add_tutorial(sub)
    p = sub.add_parser("serve", help="start the web app")
    p.add_argument("--host", default="0.0.0.0", help="0.0.0.0 (default) lets other computers reach the annotation pages; 127.0.0.1 keeps it on this computer")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--no-browser", action="store_true", help="do not open the web browser")
    p.add_argument("--reload", action="store_true")
    p.add_argument("--allow-origin", action="append", default=[], metavar="URL",
                   help="let the web app served from this address (e.g. http://lab-pc:8080) use this server; repeat for several "
                        "(also $STAINID_ALLOW_ORIGINS, comma-separated)")
    p.add_argument("--cache-limit", default=None, metavar="SIZE",
                   help="space each project's pictures for viewing (data/cache) may take unless its settings say otherwise, e.g. 100G "
                        "(the default) or 500M; the least recently used are deleted beyond it and made again when needed")
    p.add_argument("--cache-min-free", default=None, metavar="SIZE",
                   help="free space every drive keeps: when a drive has less, the pictures for viewing of all projects on it are "
                        "deleted, least recently used first, until it has; e.g. 50G, 10%% or '50G or 10%%' (the default: the smaller)")
    p.add_argument("--projects-dir", type=Path, default=None, metavar="DIR",
                   help="list every folder in DIR that holds a stainid.yaml on the Projects page (also settable there)")
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
    if args.command == "jobs":
        from stainid.jobs.cli import run as jobs

        jobs(args, args.project)
        return
    if args.command == "accounts":
        from stainid.api.accounts_cli import run as account_command

        account_command(args)
        return
    if args.command == "tutorial":
        from stainid.tutorial import run as tutorial_command

        tutorial_command(args)
        return
    if args.command == "package":
        from stainid.imageset.package import create_project

        project = create_project(args.package, args.folder, args.name)
        print(f"{project.name}: a new {project.kind} project in {project.root}; next: stainid --project {project.root} images", flush=True)
        return
    if args.command == "labels" and args.action == "split":
        split_labels(args)
        return
    project = load_project(args.project)
    os.chdir(project.root)
    step, words = _workflow_step(argv)
    if step is None or os.environ.get(JOB_ENV):  # not a workflow step, or started by the app (which keeps its own record)
        run(args, project)
        return
    options = {o.key: o.default if getattr(args, o.key, None) is None else getattr(args, o.key) for o in step.options}
    from stainid.jobs.queue import queue_dir

    with command_line_job(JobStore(queue_dir()), step.id, options, words, project):
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
    elif args.command == "tiles":
        from stainid.sampling.whole_core import write_whole_core_manifest

        write_whole_core_manifest(project, args.tile_size)
    elif args.command == "images":
        from stainid.imageset.prepare import prepare_images

        prepare_images(project, args.tile_size)
    elif args.command == "score":
        from stainid.imageset.scoring import score

        for line in score(project)["summary"]:
            print(line, flush=True)
    elif args.command == "calibrate":
        from stainid.workflows.calibrate import calibrate_slides

        print(calibrate_slides(project, args.stain, args.redo, output=args.output, cores=_cores(args.core), workers=args.workers))
    elif args.command == "nuclei":
        from stainid.workflows.nuclei import run_nuclei

        run_nuclei(project, args.stain, args.device, args.batch_size, args.threads, args.shard_index, args.shard_count, args.workers,
                   _cores(args.core))
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
    elif args.command == "v2":
        run_v2(args, project)
    elif args.command == "aggregate":
        from stainid.workflows.aggregate import aggregate_fields, aggregate_masks, aggregate_neun

        {"fields": aggregate_fields, "masks": aggregate_masks, "neun": aggregate_neun}[args.what](project, args.level, args.output)
    elif args.command == "summarize":
        from stainid.workflows.aggregate import summarize_results

        summarize_results(project, args.level)
    elif args.command == "labels":
        from stainid.labels.prepare import run

        run(project, args.action, args.stain, args.fields, args.session, args.seed)
    elif args.command == "review":
        from stainid.review import ai_batch, ai_ingest

        stains = args.stain or ["6E10", "AT8", "NeuN"]
        if args.action == "build":
            ai_batch.build(project, stains, args.round, args.fields, args.seed)
        else:
            for stain in stains:
                if (ai_batch.stain_folder(project, stain, args.round) / "manifest.csv").exists():
                    ai_ingest.ingest(project, stain, args.round, list(args.csv) or None, args.dry_run)
    elif args.command == "score-segmentation":
        from stainid.segmentation.report import run as score_segmentation

        score_segmentation(project, args.stain, args.session, args.proposals_only, args.redo)
    elif args.command == "download-models":
        from stainid.workflows.downloads import download_models

        download_models(project, args.which)
    elif args.command == "train":
        from stainid.training.train import train_model

        train_model(project, args.stain, [s.strip() for v in args.session or [] for s in v.split(",") if s.strip()] or None)


def _cores(values: list[str] | None) -> set[str] | None:
    """Core ids from --core values (an existing file holds one per line)."""
    if not values:
        return None
    out = set()
    for value in values:
        path = Path(value)
        out |= {line.strip() for line in path.read_text().splitlines() if line.strip()} if path.is_file() else {value}
    return out


def run_v2(args: argparse.Namespace, project) -> None:
    from stainid.v2 import run
    from stainid.v2.params import STAINS

    stains = args.stain or list(STAINS)
    cores = _cores(args.core)
    if args.stage == "status":
        for stage, per in run.status(project).items():
            print(f"{stage:<11} " + "  ".join(f"{s}: {v['done']}/{v['total']} done, {v['outdated']} outdated" for s, v in per.items()))
    elif args.stage == "proposals":
        run.run_proposals(project, stains, cores, args.workers or None, args.shard_index, args.shard_count, args.redo, args.threads or None,
                          max(1, args.tiles_at_once), None if args.processes < 0 else args.processes)
    elif args.stage == "calls":
        run.run_calls(project, stains, cores, args.workers or None, args.redo)
    elif args.stage == "outlines":
        run.run_outlines(project, stains, cores, args.device or None, args.shard_index, args.shard_count, args.redo)
    else:
        run.run_measures(project)


def split_labels(args: argparse.Namespace) -> None:
    """`stainid labels split`: the one-off move of an earlier release's labels.jsonl into one log per session and set
    (`labels.store.split_legacy`), for --project or, with --all, every project the app knows. Run with the app stopped."""
    from stainid.labels import store

    if args.all:
        from stainid.api.registry import entries

        roots = [Path(e.path) for e in entries() if e.exists]
    else:
        roots = [load_project(args.project).root]
    failed = False
    for root in roots:
        try:
            reviews = load_project(root).output("reviews")
        except Exception as error:  # noqa: BLE001 - one project that does not load stops no other
            print(f"{root}: not split, the project does not load ({error})", flush=True)
            failed = True
            continue
        try:
            found = store.layout(reviews)
        except store.LayoutError as error:
            print(f"{root}: not split: {error}", flush=True)
            failed = True
            continue
        if found != "legacy":
            print(f"{root}: nothing to split ({'split already' if found == 'split' else 'no labels yet'})", flush=True)
            continue
        done = store.split_legacy(reviews)
        print(f"{root}: {done['labels']} labels split into {len(done['files'])} logs under {store.folder(reviews)}; "
              f"the old log is kept as {done['kept']}", flush=True)
        for name, count in done["files"].items():
            print(f"  {name}: {count}", flush=True)
    if failed:
        raise SystemExit(1)


def refuse_old_label_logs(extra: list[Path]) -> None:
    """The app does not start while a project it knows keeps its labels in an earlier release's single labels.jsonl, or
    in both layouts (`labels.store.check_layout`): split them first, with `stainid labels split`."""
    from stainid.api.registry import entries
    from stainid.labels import store

    roots = list(dict.fromkeys([*(Path(r).resolve() for r in extra), *(Path(e.path).resolve() for e in entries() if e.exists)]))
    problems = []
    for root in roots:
        try:
            reviews = load_project(root).output("reviews")
        except Exception:  # noqa: BLE001 - not a project (the Projects page says why)
            continue
        try:
            store.check_layout(reviews)
        except store.LayoutError as error:
            problems.append(f"{root}: {error}")
    if problems:
        raise SystemExit("stainID does not start: " + "\n".join(problems) + "\nWith the app stopped: stainid labels split --all")


def serve(args: argparse.Namespace) -> None:
    import socket
    import threading
    import webbrowser

    import uvicorn

    from stainid.api.registry import PROJECTS_DIR_ENV, project_id, remembered_project
    from stainid.api.session import allowed_origins

    local = "127.0.0.1" if args.host == "0.0.0.0" else args.host
    url = f"http://{local}:{args.port}"
    # with --project the browser opens that project; otherwise the Projects page
    start = (args.project if args.project.is_dir() else args.project.parent).resolve() if args.project else remembered_project()
    page = f"{url}/p/{project_id(start)}/" if args.project and (start / CONFIG_NAME).exists() else url
    with socket.socket() as probe:
        if probe.connect_ex((local, args.port)) == 0:
            print(f"stainID is already running at {url}; opening it", flush=True)
            if not args.no_browser:
                webbrowser.open(page)
            return
    refuse_old_label_logs([start])
    os.environ["STAINID_PROJECT"] = str(start)
    os.environ["STAINID_HOST"] = args.host
    if args.cache_limit:
        from stainid.disk_cache import LIMIT_ENV, parse_size

        try:
            parse_size(args.cache_limit)
        except ValueError as error:
            raise SystemExit(f"--cache-limit: {error}")
        os.environ[LIMIT_ENV] = args.cache_limit
    if args.cache_min_free:
        from stainid.disk_cache import MIN_FREE_ENV, parse_min_free

        try:
            parse_min_free(args.cache_min_free)
        except ValueError as error:
            raise SystemExit(f"--cache-min-free: {error}")
        os.environ[MIN_FREE_ENV] = args.cache_min_free
    if args.projects_dir:
        os.environ[PROJECTS_DIR_ENV] = str(args.projects_dir.expanduser().resolve())
    origins = allowed_origins(args.allow_origin)
    os.environ["STAINID_ALLOW_ORIGINS"] = ",".join(origins)
    if not args.no_browser:
        threading.Timer(1.5, webbrowser.open, (page,)).start()
    print(f"stainID is running at {url}  (press Ctrl+C to stop)", flush=True)
    if args.host != "127.0.0.1":
        print(f"Annotators on your network: http://{socket.gethostname()}:{args.port}/p/<project>/annotate (each project's Labels page "
              "shows its link once you start a session)", flush=True)
    if origins:
        print(f"The web app served from {', '.join(origins)} may use this server", flush=True)
    uvicorn.run("stainid.api.app:create_app", factory=True, host=args.host, port=args.port, reload=args.reload, log_level="warning")


if __name__ == "__main__":
    main()
