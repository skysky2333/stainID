from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse, Response
from pydantic import BaseModel

from stainid.api import work
from stainid.api.state import get_state
from stainid.jobs import ACTIVE, NOISE
from stainid.workflows.steps import describe

router = APIRouter(tags=["jobs"])
LOG_BYTES = 400_000


class JobRequest(BaseModel):
    workflow: str
    options: dict = {}


def described():
    """`describe` of the project, counted in the background (`work`) so that a cold count of a large project (thousands
    of core images) answers "preparing" instead of keeping the page waiting; parallel requests share one count."""
    state = get_state()
    return work.answer("steps", ("steps", str(state.root)), lambda task: describe(state.project), work.NOW, "Looking at where the study stands",
                       fresh=work.within(3), alone=True)


@router.get("/steps")
def steps():
    """Every workflow step with what it needs / produces (and whether those exist), its options, progress, latest job and
    the jobs of it running or waiting now (a step can run in several parts)."""
    state = get_state()
    found = described()
    if isinstance(found, Response):
        return found
    latest: dict[str, dict] = {}
    active: dict[str, list[dict]] = {}
    for job in sorted(state.jobs.jobs.values(), key=lambda j: j.created):
        latest[job.workflow] = info = state.jobs.describe(job)
        if job.status in ACTIVE:
            active.setdefault(job.workflow, []).append(info)
    return [{**step, "job": latest.get(step["id"]), "active": active.get(step["id"], [])} for step in found]


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
    if path is None:
        return "(This run prints to its terminal. Start it with `> run.log` at the end to follow its output here.)"
    if not path.is_file():
        return ""
    with path.open("rb") as handle:  # only the end: a long run's log can be many MB
        start = max(0, handle.seek(0, 2) - LOG_BYTES)
        handle.seek(start)
        lines = handle.read().decode(errors="replace").splitlines()[1 if start else 0:]  # not a cut-off first line
    return "\n".join(line for line in lines[-tail:] if not NOISE.search(line))
