from __future__ import annotations

from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException

from stainid.api.state import get_state
from stainid.api.util import records

router = APIRouter(tags=["analysis"])
GROUP_COLUMNS = ("disease_group", "region", "tma")


def tables_root() -> Path:
    return get_state().project.output("tables")


def resolve(name: str) -> Path:
    path = (tables_root() / name).resolve()
    if tables_root().resolve() not in path.parents or path.suffix != ".csv" or not path.exists():
        raise HTTPException(404, name)
    return path


@router.get("/tables")
def tables() -> list[dict]:
    base = tables_root()
    out = []
    for path in sorted(base.glob("*.csv")) + sorted(base.glob("biology/**/*.csv")):
        with path.open() as handle:
            header = handle.readline().strip().split(",")
        out.append({"name": str(path.relative_to(base)), "columns": len(header), "size_kb": round(path.stat().st_size / 1024, 1),
                    "donor_region": "sample_region_id" in header, "contrasts": "std_effect" in header or "effect_sd" in header})
    return out


@router.get("/tables/{name:path}/columns")
def columns(name: str) -> dict:
    frame = pd.read_csv(resolve(name), nrows=500, dtype={"donor_id": str, "tma": str})
    skip = {"tma", "technical_replicate", "selection_order", "label"}
    numeric = [c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c]) and c not in skip and not c.endswith("_id")]
    return {"columns": list(frame.columns), "numeric": numeric, "groupable": [c for c in GROUP_COLUMNS if c in frame]}


@router.get("/tables/{name:path}/rows")
def rows(name: str, limit: int = 2000) -> dict:
    frame = pd.read_csv(resolve(name), dtype={"donor_id": str, "tma": str})
    return {"total": len(frame), "rows": records(frame.head(limit))}


@router.get("/tables/{name:path}/feature")
def feature(name: str, column: str, group: str = "disease_group", facet: str | None = None) -> dict:
    frame = pd.read_csv(resolve(name), dtype={"donor_id": str, "tma": str})
    if "sampling_extent" in frame:
        frame = frame[frame.sampling_extent == frame.sampling_extent.iloc[0]]
    if column not in frame or group not in frame:
        raise HTTPException(400, "unknown column")
    keep = [c for c in dict.fromkeys(["sample_region_id", "core_id", "donor_id", group, facet, column]) if c and c in frame]
    return {"column": column, "group": group, "facet": facet, "points": records(frame[keep].dropna(subset=[column]))}
