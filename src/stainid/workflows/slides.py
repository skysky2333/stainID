"""Slide-level steps: find cores on each slide, attach the TMA map, export native cores, core QC."""
from __future__ import annotations

from pathlib import Path

from stainid.project import Project
from stainid.qc.core_qc import analyze_manifest
from stainid.slides.export import export_cores
from stainid.slides.layout import attach_layout
from stainid.slides.table import read_slides
from stainid.tables import read_csv, write_records

READER_NOTE = "The first time, the Bio-Formats slide reader is downloaded (about 60 MB); this can take a minute."


def dearray_slides(project: Project, redo: bool = False, target_long_side: int = 4500) -> Path:
    """Fit the TMA grid on every slide in the slides table; slides already in the core manifest are skipped unless `redo`."""
    from stainid.slides.dearray import process_slide

    manifest = project.input("core_manifest")
    slides = read_slides(project.input("slides_table"))
    if not slides:
        raise ValueError("The slides table is empty: add your slide scans on the Slides step first")
    existing = read_csv(manifest) if manifest.exists() else []
    done = {row["slide_path"] for row in existing}
    pending = [s for s in slides if redo or str(Path(s["slide_path"]).resolve()) not in done]
    print(f"[0/{len(pending)}] {len(pending)} of {len(slides)} slides to process. {READER_NOTE}", flush=True)
    rows, columns = int(project.config["tma"]["rows"]), int(project.config["tma"]["columns"])
    for number, slide in enumerate(pending, start=1):
        path = Path(slide["slide_path"])
        records = process_slide(path, slide["tma"], slide["stain"], rows, columns, target_long_side, project.output("qc") / "grids")
        existing = [row for row in existing if row["slide_path"] != str(path.resolve())] + records
        write_records(manifest, existing)
        found = sum(r["provisional_tissue_status"] != "empty" for r in records)
        print(f"[{number}/{len(pending)}] {path.name}: {found} of {len(records)} positions contain tissue", flush=True)
    return manifest


def attach_map(project: Project) -> Path:
    attach_layout(project.input("core_manifest"), project.input("tma_layout"))
    print(f"TMA map attached to {project.relative(project.input('core_manifest'))}", flush=True)
    return project.input("core_manifest")


def export(project: Project, overwrite: bool = False, workers: int = 2) -> None:
    slides = {r["slide"] for r in read_csv(project.input("core_manifest"))}
    print(f"[0/{len(slides)}] Opening {len(slides)} slides. {READER_NOTE}", flush=True)
    export_cores(project.input("core_manifest"), None, None, overwrite, workers, project.tma_prefix)


def core_qc(project: Project, overwrite: bool = False, workers: int = 2) -> Path:
    return analyze_manifest(project.input("core_manifest"), True, overwrite, workers)
