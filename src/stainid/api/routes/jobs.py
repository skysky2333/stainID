from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from stainid.api.jobs import NOISE
from stainid.api.state import get_state
from stainid.workflows.steps import describe

router = APIRouter(tags=["jobs"])


class JobRequest(BaseModel):
    workflow: str
    options: dict = {}


@router.get("/steps")
def steps() -> list[dict]:
    """Every workflow step with what it needs / produces (and whether those exist), its options, progress and latest job."""
    state = get_state()
    latest: dict[str, dict] = {}
    for job in sorted(state.jobs.jobs.values(), key=lambda j: j.created):
        latest[job.workflow] = state.jobs.describe(job)
    return [{**step, "job": latest.get(step["id"])} for step in describe(state.project)]


@router.get("/jobs")
def list_jobs() -> list[dict]:
    jobs = get_state().jobs
    return [jobs.describe(j) for j in sorted(jobs.jobs.values(), key=lambda j: j.created, reverse=True)]


@router.post("/jobs")
def submit(request: JobRequest) -> dict:
    jobs = get_state().jobs
    try:
        return jobs.describe(jobs.submit(request.workflow, request.options))
    except KeyError:
        raise HTTPException(400, f"unknown step {request.workflow}")


@router.delete("/jobs/{job_id}")
def cancel(job_id: str) -> dict:
    jobs = get_state().jobs
    return jobs.describe(jobs.cancel(job_id))


@router.get("/jobs/{job_id}/log", response_class=PlainTextResponse)
def log(job_id: str, tail: int = 400) -> str:
    path = get_state().jobs.log_path(job_id)
    if not path.exists():
        return ""
    lines = path.read_text(errors="replace").splitlines()
    return "\n".join(line for line in lines[-tail:] if not NOISE.search(line))
