from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel

from stainid.api import session, work
from stainid.api.state import get_state

router = APIRouter(tags=["session"])
YEAR = 365 * 24 * 3600


@router.get("/session/work/{task_id}")
def work_status(task_id: str, request: Request):
    """Where a slow request's background work stands (`stainid.api.work`): pending (202, with the queue position and an
    estimate), its result once done, or its error."""
    task = work.find(task_id)
    if task is None:
        raise HTTPException(404, "this work is not known (any more): ask again")
    if task.admin and not session.is_admin(request):
        raise HTTPException(401, "admin sign-in required")
    if task.state in ("waiting", "running"):
        return work.pending(task, "result")
    return {"result": work.outcome(task)}


@router.get("/session/work", dependencies=[Depends(session.require_admin)])
def work_queue() -> dict:
    """Everything the background pool is doing and has queued."""
    return work.overview()


class Register(BaseModel):
    name: str


class Password(BaseModel):
    password: str


class NewPassword(BaseModel):
    current: str
    new: str


@router.get("/session")
def current(request: Request) -> dict:
    admin = session.is_admin(request)
    return {"admin": admin, "annotator": session.annotator_name(request), "default_password": session.default_password(),
            "local": session.is_local(request), "project": get_state().project.name if admin else None,
            "allowed_origins": session.allowed_origins() if admin else []}


@router.post("/session/annotator")
def register(request: Register, http: Request, response: Response) -> dict:
    name = " ".join(request.name.split())
    if not 1 <= len(name) <= 40:
        raise HTTPException(400, "enter a name of 1–40 characters")
    response.set_cookie(session.ANNOTATOR_COOKIE, session.sign_annotator(name), max_age=YEAR, httponly=True, samesite="lax")
    return {"annotator": name, "token": session.session_token(session.is_admin(http), name)}


@router.delete("/session/annotator")
def unregister(request: Request, response: Response) -> dict:
    response.delete_cookie(session.ANNOTATOR_COOKIE)
    return {"ok": True, "token": session.session_token(session.is_admin(request), None)}


@router.post("/session/admin")
def admin_sign_in(body: Password, request: Request, response: Response) -> dict:
    if not session.check_password(body.password):
        raise HTTPException(403, "wrong password")
    response.set_cookie(session.ADMIN_COOKIE, session.admin_token(), max_age=30 * 24 * 3600, httponly=True, samesite="lax")
    return {"admin": True, "token": session.session_token(True, session.annotator_name(request))}


@router.delete("/session/admin")
def admin_sign_out(request: Request, response: Response) -> dict:
    response.delete_cookie(session.ADMIN_COOKIE)
    return {"ok": True, "token": session.session_token(False, session.annotator_name(request))}


@router.put("/session/admin/password", dependencies=[Depends(session.require_admin)])
def change_password(body: NewPassword, request: Request, response: Response) -> dict:
    if not session.check_password(body.current):
        raise HTTPException(403, "the current password is wrong")
    if len(body.new) < 6 or body.new == session.DEFAULT_PASSWORD:
        raise HTTPException(400, "choose a password of at least 6 characters (not 'admin')")
    session.set_password(body.new)
    response.set_cookie(session.ADMIN_COOKIE, session.admin_token(), max_age=30 * 24 * 3600, httponly=True, samesite="lax")
    return {"ok": True, "token": session.session_token(True, session.annotator_name(request))}
