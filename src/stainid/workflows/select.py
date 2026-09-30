"""Analysis-field selection: core image table, spatially balanced native fields per stain-core, primary subset."""
from __future__ import annotations

from stainid.project import Project
from stainid.sampling.analysis_manifest import write_analysis_manifest
from stainid.sampling.cohort import create_systematic_tile_manifest
from stainid.tables import read_csv, write_records


def select_fields(project: Project, fields_per_core: int = 8, primary: int = 4, field_size_px: int = 2048) -> list:
    core_images = write_analysis_manifest(project.input("core_manifest"), project.input("core_images"), project.tma_prefix)
    all_fields = project.input("field_manifest")
    create_systematic_tile_manifest(core_images, all_fields, all_fields.with_name(f"{all_fields.stem}_qc.csv"), fields_per_core, field_size_px)
    rows = [r for r in read_csv(all_fields) if int(r["selection_order"]) <= primary]
    write_records(project.input("tile_manifest"), rows)
    return [core_images, all_fields, project.input("tile_manifest")]
