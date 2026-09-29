from __future__ import annotations

from fastapi import APIRouter

from stainid.api.state import GROUP_LABELS, get_state
from stainid.api.util import records

router = APIRouter(tags=["project"])


@router.get("/project")
def project_info() -> dict:
    state = get_state()
    project = state.project
    paths = {section: {key: {"path": str(project._resolve(section, key)), "exists": project._resolve(section, key).exists()}
                       for key in project.config[section]} for section in ("inputs", "models", "outputs")}
    return {"name": project.name, "root": str(project.root), "pixel_size_um": project.pixel_size_um, "stains": project.stains,
            "groups": GROUP_LABELS, "paths": paths}


@router.get("/summary")
def summary() -> dict:
    state = get_state()
    tiles = state.tiles
    donors = state.donors
    groups = donors.disease_group.value_counts().to_dict() if not donors.empty else {}
    return {
        "donors": int(donors.donor_id.nunique()) if not donors.empty else 0,
        "groups": {k: int(v) for k, v in groups.items()},
        "tmas": sorted(tiles.tma.astype(str).unique().tolist()) if not tiles.empty else [],
        "cores": int(tiles.core_id.nunique()) if not tiles.empty else 0,
        "donor_regions": int(tiles.sample_region_id.nunique()) if not tiles.empty else 0,
        "fields": {k: int(v) for k, v in tiles.stain.value_counts().items()} if not tiles.empty else {},
        "status": state.output_status(),
    }


@router.get("/calibration")
def calibration() -> list[dict]:
    return records(get_state().calibration())
