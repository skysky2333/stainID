"""The space a project's pictures for viewing take (`stainid.disk_cache`): how much, its limit, and Clear all."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from stainid import disk_cache
from stainid.api.state import get_state, reload_project
from stainid.project import save_project

router = APIRouter(tags=["pictures for viewing"])
SMALLEST_GB = 0.5


class LimitChange(BaseModel):
    limit_gb: float | None  # None: the server's default


@router.get("/project/cache")
def usage() -> dict:
    return disk_cache.usage(get_state().root)


@router.put("/project/cache")
def set_limit(request: LimitChange) -> dict:
    if request.limit_gb is not None and not request.limit_gb >= SMALLEST_GB:
        raise HTTPException(400, f"The space must be at least {SMALLEST_GB} GB.")
    state = get_state()
    reload_project(save_project(state.project, {disk_cache.SETTING: request.limit_gb}).root)
    disk_cache.check(state.root)
    return disk_cache.usage(state.root)


@router.post("/project/cache/clear")
def clear() -> dict:
    """Delete every picture the app made for viewing, except those in use right now (in the background)."""
    root = get_state().root
    disk_cache.clear_in_background(root)
    return disk_cache.usage(root)
