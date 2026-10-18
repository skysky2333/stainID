from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel

from stainid.api import accounts, session, work
from stainid.api.state import get_state

router = APIRouter(tags=["session"])
SESSION_SECONDS = 30 * 24 * 3600


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


class Credentials(BaseModel):
    name: str
    password: str


class Password(BaseModel):
    password: str


class NewPassword(BaseModel):
    current: str = ""
    new: str


def _error(error: accounts.AccountError):
    raise HTTPException(error.status, str(error))


def _address(request: Request) -> str:
    forwarded = request.headers.get("cf-connecting-ip") or request.headers.get("x-forwarded-for", "").split(",")[0].strip()
    return forwarded or (request.client.host if request.client else "")


def _secure(request: Request) -> bool:
    """Behind the Cloudflare tunnel the page is https: the cookie then only travels over https."""
    return request.url.scheme == "https" or request.headers.get("x-forwarded-proto", "") == "https"


def _signed_in(account: accounts.Account, request: Request, response: Response) -> dict:
    response.set_cookie(session.ACCOUNT_COOKIE, accounts.session_value(account), max_age=SESSION_SECONDS, httponly=True, samesite="lax",
                        secure=_secure(request))
    return {"account": account.to_api(), "admin": account.admin and not account.must_change, "annotator": account.handle,
            "token": session.session_token(account)}


@router.get("/session")
def current(request: Request) -> dict:
    account = session.current_account(request)
    admin = session.is_admin(request)
    return {"admin": admin, "annotator": session.annotator_name(request), "account": account.to_api() if account else None,
            "default_password": any(accounts.uses_default_password(a) for a in accounts.everyone() if a.admin), "local": session.is_local(request),
            "project": get_state().project.name if admin else None, "allowed_origins": session.allowed_origins() if admin else []}


@router.post("/session/login")
def sign_in(body: Credentials, request: Request, response: Response) -> dict:
    """Everyone signs in here; what they see follows from their role."""
    try:
        account = accounts.sign_in(body.name, body.password, _address(request))
    except accounts.AccountError as error:
        _error(error)
    return _signed_in(account, request, response)


@router.post("/session/register")
def sign_up(body: Credentials, request: Request, response: Response) -> dict:
    """A new annotator account (admins are made by an admin)."""
    try:
        accounts._limited("register", [f"ip:{_address(request)}"], 10)
        account = accounts.register(body.name, body.password)
        accounts._failed("register", [f"ip:{_address(request)}"])  # every sign-up counts: at most 30 an address per quarter hour
    except accounts.AccountError as error:
        _error(error)
    return _signed_in(account, request, response)


@router.post("/session/logout")
def sign_out(response: Response) -> dict:
    response.delete_cookie(session.ACCOUNT_COOKIE)
    return {"ok": True, "token": ""}


@router.post("/session/admin")
def admin_sign_in(body: Password, request: Request, response: Response) -> dict:
    """The sign-in of earlier versions (password only): the account `admin`."""
    return sign_in(Credentials(name="admin", password=body.password), request, response)


@router.delete("/session/admin")
def admin_sign_out(response: Response) -> dict:
    return sign_out(response)


@router.put("/session/password")
def change_password(body: NewPassword, request: Request, response: Response) -> dict:
    """The person's own new password (after an admin's reset, the one-time password signed them in already)."""
    account = session.current_account(request)
    if account is None:
        raise HTTPException(401, "sign in first")
    try:
        account = accounts.set_password(account.id, body.new, body.current)
    except accounts.AccountError as error:
        _error(error)
    return _signed_in(account, request, response)


@router.put("/session/admin/password")
def change_admin_password(body: NewPassword, request: Request, response: Response) -> dict:
    return change_password(body, request, response)


# ---------------------------------------------------------------- people (admins)

class Person(BaseModel):
    name: str
    password: str
    role: str = accounts.ANNOTATOR


class Change(BaseModel):
    role: str | None = None
    past_name: str | None = None


admin = [Depends(session.require_admin)]


@router.get("/session/people", dependencies=admin)
def people() -> dict:
    return {"people": [a.to_api() for a in accounts.everyone()], "past_names": accounts.past_names()}


@router.post("/session/people", dependencies=admin)
def add_person(body: Person) -> dict:
    try:
        return accounts.register(body.name, body.password, body.role, by_admin=True).to_api()
    except accounts.AccountError as error:
        _error(error)


@router.put("/session/people/{account_id}", dependencies=admin)
def change_person(account_id: str, body: Change) -> dict:
    try:
        account = accounts.get(account_id)
        if account is None:
            raise accounts.AccountError("no such account", 404)
        if body.role is not None:
            account = accounts.set_role(account_id, body.role)
        if body.past_name is not None:
            account = accounts.link_past_name(account_id, body.past_name)
        return account.to_api()
    except accounts.AccountError as error:
        _error(error)


@router.post("/session/people/{account_id}/reset", dependencies=admin)
def reset_person(account_id: str) -> dict:
    """A one-time password to give the person; they choose a new one when they sign in with it."""
    try:
        return {"password": accounts.reset_password(account_id)}
    except accounts.AccountError as error:
        _error(error)


@router.delete("/session/people/{account_id}", dependencies=admin)
def remove_person(account_id: str) -> dict:
    try:
        accounts.remove(account_id)
    except accounts.AccountError as error:
        _error(error)
    return {"ok": True}


# ---------------------------------------------------------------- API tokens (one's own)

class TokenRequest(BaseModel):
    label: str = ""


@router.get("/session/tokens")
def my_tokens(request: Request) -> list[dict]:
    return accounts.tokens(session.require_account(request).id)


@router.post("/session/tokens")
def make_token(body: TokenRequest, request: Request) -> dict:
    """A new API token of the signed-in account, for scripts (`Authorization: Bearer <token>`); shown once."""
    return {"token": accounts.new_token(session.require_account(request).id, body.label)}


@router.delete("/session/tokens/{token_id}")
def drop_token(token_id: str, request: Request) -> dict:
    accounts.revoke_token(session.require_account(request).id, token_id)
    return {"ok": True}


# ---------------------------------------------------------------- the person's labelling sets

class Key(BaseModel):
    key: str


class TutorialDone(BaseModel):
    kind: str = "objects"  # "objects" (Fix the field) or "areas" (Mark bad areas)
    done: bool = True


@router.get("/session/tutorials")
def tutorials(request: Request) -> dict:
    """Which tutorials the signed-in person went through or skipped (kept with their account, on every project);
    nobody signed in: none (the page keeps a guest's in the browser)."""
    account = session.current_account(request)
    if account is None or account.must_change:
        return {kind: False for kind in accounts.TUTORIALS} | {"kept": False}
    return accounts.tutorials_done(account.id) | {"kept": True}


@router.post("/session/tutorials")
def tutorial_done(body: TutorialDone, request: Request) -> dict:
    """Remember that the signed-in person finished (or skipped) a tutorial, or forget it; a guest: not kept."""
    account = session.current_account(request)
    if account is None or account.must_change:
        return {kind: kind == body.kind and body.done for kind in accounts.TUTORIALS} | {"kept": False}
    try:
        return accounts.set_tutorial(account.id, body.kind, body.done) | {"kept": True}
    except accounts.AccountError as error:
        _error(error)


@router.get("/session/sets")
def my_sets(request: Request, here: str = "") -> list[dict]:
    """The labelling sets in the signed-in person's collection (every project), with their progress; an admin also sees
    every session of the project `here` (the page's project)."""
    from stainid.api.routes import accounts as access_routes  # noqa: PLC0415

    return access_routes.sets_of(session.require_account(request), request, here or None)


@router.post("/session/sets")
def add_set(body: Key, request: Request) -> dict:
    """Add the set a join key opens to the person's collection."""
    from stainid.api.routes import accounts as access_routes  # noqa: PLC0415

    account = session.require_account(request)
    try:
        project, name = accounts.join(account.id, body.key, _address(request))
    except accounts.AccountError as error:
        _error(error)
    found = [s for s in access_routes.sets_of(account, request) if s["project"] == project and s["name"] == name]
    return found[0] if found else {"project": project, "name": name}
