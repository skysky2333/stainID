"""Slide-level steps: find cores on each slide, attach the TMA map, export native cores, core QC."""
from __future__ import annotations

import json
import time
from pathlib import Path

from stainid.imaging import thumbnails
from stainid.project import Project
from stainid.qc.core_qc import analyze_manifest
from stainid.slides.export import export_cores
from stainid.slides.layout import attach_layout
from stainid.slides.table import read_slides, set_slide_grid
from stainid.tables import read_csv, write_records, write_text_atomic

READER_NOTE = "The first time, the Bio-Formats slide reader is downloaded (about 60 MB); this can take a minute."


def dearray_slides(project: Project, redo: bool = False, target_long_side: int = 4500, only: list[str] | None = None) -> Path:
    """Fit the TMA grid on every slide in the slides table; slides already in the core manifest are skipped unless `redo`
    or named in `only` (file names or names without the extension). A slide without its own rows / columns gets the
    best-fitting grid size, which is written to the slides table. A slide that fails does not stop the others."""
    from stainid.slides.dearray import process_slide

    manifest = project.input("core_manifest")
    table = project.input("slides_table")
    slides = read_slides(table)
    if not slides:
        raise ValueError("The slides table is empty: add your slide scans on the Slides step first")
    names = {name.strip() for item in only or [] for name in item.split(",") if name.strip()}
    unknown = sorted(names - {Path(s["slide_path"]).name for s in slides} - {Path(s["slide_path"]).stem for s in slides})
    if unknown:
        raise ValueError(f"Not in the slides table: {', '.join(unknown)}")
    existing = read_csv(manifest) if manifest.exists() else []
    done = {row["slide_path"] for row in existing}
    chosen = [s for s in slides if {Path(s["slide_path"]).name, Path(s["slide_path"]).stem} & names]
    pending = chosen if names else [s for s in slides if redo or str(Path(s["slide_path"]).resolve()) not in done]
    print(f"[0/{len(pending)}] {len(pending)} of {len(slides)} slides to process. {READER_NOTE}", flush=True)
    default = int(project.config["tma"]["rows"]), int(project.config["tma"]["columns"])
    qc_dir = project.output("qc") / "grids"
    failed = []
    for number, slide in enumerate(pending, start=1):
        path = Path(slide["slide_path"])
        rows, columns = (int(slide["rows"]), int(slide["columns"])) if slide.get("rows") and slide.get("columns") else (None, None)
        try:
            records, choice = process_slide(path, slide["tma"], slide["stain"], rows, columns, target_long_side, qc_dir,
                                            slide.get("orientation", ""), default)
        except ValueError as error:
            failed.append(f"{path.name}: {error}")
            grid_report(project, path, {"tma": slide["tma"], "stain": slide["stain"], "error": str(error)})
            print(f"[{number}/{len(pending)}] {path.name}: no cores found. {error}", flush=True)
            continue
        existing = [row for row in existing if row["slide_path"] != str(path.resolve())] + records
        write_records(manifest, existing)
        if choice.source == "auto":
            set_slide_grid(table, slide["slide_path"], choice.rows, choice.columns, slide.get("orientation", ""), "auto")
        found = sum(r["provisional_tissue_status"] != "empty" for r in records)
        grid_report(project, path, {"tma": slide["tma"], "stain": slide["stain"], "rows": choice.rows, "columns": choice.columns,
                                    "grid_source": choice.source, "orientation": slide.get("orientation", ""), "score": round(choice.score, 3),
                                    "rmse_preview_px": round(choice.fit.rmse, 2), "cores_on_grid": len(choice.fit.inlier_components),
                                    "positions": len(records), "positions_with_tissue": found,
                                    "tried": [{"rows": r, "columns": c, "score": score} for r, c, score in choice.tried]})
        how = ", found automatically" if choice.source == "auto" else ""
        print(f"[{number}/{len(pending)}] {path.name}: {found} of {len(records)} positions contain tissue "
              f"({choice.rows} x {choice.columns} grid{how})", flush=True)
    if failed:
        raise ValueError(f"No cores found on {len(failed)} of {len(pending)} slides (the others are done). " + " · ".join(failed)
                         + ". Check these slides on the Cores screen of the guided setup: set their rows and columns, or turn them.")
    return manifest


def grid_report(project: Project, slide: Path, report: dict) -> Path:
    """What Find cores did on one slide (`qc/grids/<slide>.json`): the grid size, how it was chosen and how well it fits."""
    from stainid.slides.dearray import slide_slug

    path = project.output("qc") / "grids" / f"{slide_slug(slide)}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_text_atomic(path, json.dumps({"slide": slide.stem, "slide_path": str(slide), "time": time.time(), **report}, indent=1))
    return path


def read_grid_report(project: Project, slide: Path) -> dict:
    from stainid.slides.dearray import slide_slug

    path = project.output("qc") / "grids" / f"{slide_slug(slide)}.json"
    return json.loads(path.read_text()) if path.exists() else {}


def attach_map(project: Project) -> Path:
    attach_layout(project.input("core_manifest"), project.input("tma_layout"))
    print(f"TMA map attached to {project.relative(project.input('core_manifest'))}", flush=True)
    return project.input("core_manifest")


def export(project: Project, overwrite: bool = False, workers: int = 2) -> None:
    slides = {r["slide"] for r in read_csv(project.input("core_manifest"))}
    print(f"[0/{len(slides)}] Opening {len(slides)} slides. {READER_NOTE}", flush=True)
    export_cores(project.input("core_manifest"), None, None, overwrite, workers, project.tma_prefix, thumbnails.cache_folder(project.root))


def core_qc(project: Project, overwrite: bool = False, workers: int = 2) -> Path:
    return analyze_manifest(project.input("core_manifest"), True, overwrite, workers)
