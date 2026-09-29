from __future__ import annotations

from functools import lru_cache

from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from stainid.api.jobs import WORKFLOWS, JobManager
from stainid.api.state import get_state

router = APIRouter(tags=["jobs"])


@lru_cache(maxsize=1)
def manager() -> JobManager:
    state = get_state()
    return JobManager(state.root, state.project.output("jobs"))


class JobRequest(BaseModel):
    workflow: str
    options: dict = {}


@router.get("/workflows")
def workflows() -> dict:
    return {k: {"title": v["title"], "heavy": v["heavy"]} for k, v in WORKFLOWS.items()}


@router.get("/jobs")
def list_jobs() -> list[dict]:
    jobs = manager()
    return [jobs.describe(j) for j in sorted(jobs.jobs.values(), key=lambda j: j.created, reverse=True)]


@router.post("/jobs")
def submit(request: JobRequest) -> dict:
    try:
        return manager().describe(manager().submit(request.workflow, request.options))
    except KeyError:
        raise HTTPException(400, f"unknown workflow {request.workflow}")


@router.delete("/jobs/{job_id}")
def cancel(job_id: str) -> dict:
    return manager().describe(manager().cancel(job_id))


@router.get("/jobs/{job_id}/log", response_class=PlainTextResponse)
def log(job_id: str, tail: int = 400) -> str:
    path = manager().log_path(job_id)
    if not path.exists():
        return ""
    lines = path.read_text(errors="replace").splitlines()
    return "\n".join(line for line in lines[-tail:] if "Warning" not in line)
