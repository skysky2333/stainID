from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel

from stainid.api import session
from stainid.api.state import get_state

router = APIRouter(tags=["session"])
YEAR = 365 * 24 * 3600


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
            "local": session.is_local(request), "project": get_state().project.name if admin else None}


@router.post("/session/annotator")
def register(request: Register, response: Response) -> dict:
    name = " ".join(request.name.split())
    if not 1 <= len(name) <= 40:
        raise HTTPException(400, "enter a name of 1–40 characters")
    response.set_cookie(session.ANNOTATOR_COOKIE, session.sign_annotator(name), max_age=YEAR, httponly=True, samesite="lax")
    return {"annotator": name}


@router.delete("/session/annotator")
def unregister(response: Response) -> dict:
    response.delete_cookie(session.ANNOTATOR_COOKIE)
    return {"ok": True}


@router.post("/session/admin")
def admin_sign_in(body: Password, request: Request, response: Response) -> dict:
    if not session.check_password(body.password):
        raise HTTPException(403, "wrong password")
    if session.default_password() and not session.is_local(request):
        raise HTTPException(403, "the admin password is still the default; change it on the computer running stainID (Settings) before signing in from elsewhere")
    response.set_cookie(session.ADMIN_COOKIE, session.admin_token(), max_age=30 * 24 * 3600, httponly=True, samesite="lax")
    return {"admin": True}


@router.delete("/session/admin")
def admin_sign_out(response: Response) -> dict:
    response.delete_cookie(session.ADMIN_COOKIE)
    return {"ok": True}


@router.put("/session/admin/password", dependencies=[Depends(session.require_admin)])
def change_password(body: NewPassword, response: Response) -> dict:
    if not session.check_password(body.current):
        raise HTTPException(403, "the current password is wrong")
    if len(body.new) < 6 or body.new == session.DEFAULT_PASSWORD:
        raise HTTPException(400, "choose a password of at least 6 characters (not 'admin')")
    session.set_password(body.new)
    response.set_cookie(session.ADMIN_COOKIE, session.admin_token(), max_age=30 * 24 * 3600, httponly=True, samesite="lax")
    return {"ok": True}
