from __future__ import annotations

import hmac

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel

from stainid.api import session
from stainid.api.state import get_state

router = APIRouter(tags=["session"])


class SignIn(BaseModel):
    name: str
    code: str = ""


@router.get("/session")
def current(request: Request) -> dict:
    return {"mode": "annotate" if session.shared() else "full", "annotator": session.verify(request.cookies.get(session.cookie())),
            "code_required": bool(session.access_code()), "project": get_state().project.name}


@router.post("/session")
def sign_in(request: SignIn, response: Response) -> dict:
    name = " ".join(request.name.split())
    if not 1 <= len(name) <= 40:
        raise HTTPException(400, "enter a name of 1–40 characters")
    if not hmac.compare_digest(request.code.strip(), session.access_code()):
        raise HTTPException(403, "wrong access code")
    response.set_cookie(session.cookie(), session.sign(name), max_age=365 * 24 * 3600, httponly=True, samesite="lax")
    return {"annotator": name}


@router.delete("/session")
def sign_out(response: Response) -> dict:
    response.delete_cookie(session.cookie())
    return {"ok": True}
