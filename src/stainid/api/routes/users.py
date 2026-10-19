"""The admin's **Users** page (`/api/admin/users…`, admins only): every account on the server with its role, when it was
made and last signed in, the labelling sets it joined (of every project), the labels it made and whether it is
disabled; adding people (with a one-time password, shown once), password resets (the same), roles, disabling,
removing (typed confirmation; labels are never touched), memberships in any set of any project, and API tokens.

Guards: an admin cannot remove, disable or demote their own account, and the last admin who can sign in stays
(`accounts`). Every action is kept in the audit log (`accounts.audit`), shown at the bottom of the page. One-time
passwords and tokens are returned only by the request that makes them: the store keeps their hashes only.
"""
from __future__ import annotations

import threading
from collections import defaultdict
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from stainid.api import accounts, session

router = APIRouter(tags=["users"])
admin = [Depends(session.require_admin)]
_LABELS: dict[str, tuple] = {}
_LABELS_LOCK = threading.Lock()


def _error(error: accounts.AccountError):
    raise HTTPException(error.status, str(error))


def _actor(request: Request) -> accounts.Account:
    return session.current_account(request)


def _target(account_id: str) -> accounts.Account:
    found = accounts.get(account_id)
    if found is None:
        raise HTTPException(404, "no such account")
    return found


def _not_self(request: Request, account_id: str, what: str) -> None:
    if _actor(request).id == account_id:
        raise HTTPException(409, f"you cannot {what} your own account: ask another admin")


def _projects() -> list[tuple[str, str, Path]]:
    """(id, name, folder) of every project the server knows, the open ones included."""
    from stainid.api import registry, state  # noqa: PLC0415

    found: dict[str, tuple[str, str, Path]] = {}
    for entry in registry.entries([s.root for s in state.open_states()]):
        if entry.exists:
            found.setdefault(entry.id, (entry.id, entry.name, Path(entry.path)))
    return list(found.values())


def _sets() -> list[dict]:
    """Every labelling set of every project: {project, project_name, session, title, stain, kind}."""
    from stainid.labels import sessions  # noqa: PLC0415
    from stainid.project import load_project  # noqa: PLC0415

    out = []
    for project_id, name, root in _projects():
        try:
            reviews = load_project(root).output("reviews")
            names = sessions.names(reviews)
        except Exception:  # noqa: BLE001 - a project that cannot be read has no sets to offer
            continue
        for session_name in names:
            try:
                meta = sessions.meta(reviews, session_name)
            except (OSError, ValueError):
                meta = {}
            out.append({"project": project_id, "project_name": name, "session": session_name, "title": meta.get("title") or session_name,
                        "stain": meta.get("stain", ""), "kind": meta.get("kind", "")})
    return out


def _label_counts() -> tuple[dict[str, int], dict[str, int]]:
    """Labels made, of every project: by account id (labels since accounts carry it) and by name (earlier ones)."""
    from stainid.labels import store  # noqa: PLC0415
    from stainid.project import load_project  # noqa: PLC0415

    by_account: dict[str, int] = defaultdict(int)
    by_name: dict[str, int] = defaultdict(int)
    for _, _, root in _projects():
        try:
            reviews = load_project(root).output("reviews")
            stamp = store.signature(reviews)
        except Exception:  # noqa: BLE001
            continue
        key = str(reviews)
        with _LABELS_LOCK:
            kept = _LABELS.get(key)
        if kept is None or kept[0] != stamp:
            try:
                log = store.current(reviews)
            except Exception:  # noqa: BLE001
                continue
            accounts_of = log["account"].fillna("").astype(str) if "account" in log else None
            names = log["source"].fillna("").astype(str).str.casefold() if "source" in log else None
            ids = accounts_of.value_counts().to_dict() if accounts_of is not None else {}
            named = names[accounts_of == ""].value_counts().to_dict() if names is not None and accounts_of is not None else (
                names.value_counts().to_dict() if names is not None else {})
            kept = (stamp, ids, named)
            with _LABELS_LOCK:
                _LABELS[key] = kept
        for k, n in kept[1].items():
            if k:
                by_account[k] += int(n)
        for k, n in kept[2].items():
            if k:
                by_name[k] += int(n)
    return by_account, by_name


def _rows() -> list[dict]:
    sets = {(s["project"], s["session"]): s for s in _sets()}
    joined: dict[str, list[dict]] = defaultdict(list)
    for m in accounts.memberships():
        s = sets.get((m["project"], m["session"]))
        joined[m["account"]].append({"project": m["project"], "session": m["session"], "joined": m["joined"],
                                     "project_name": s["project_name"] if s else m["project"], "title": s["title"] if s else m["session"],
                                     "exists": s is not None})
    by_account, by_name = _label_counts()
    tokens = accounts.token_counts()
    return [{**a.to_api(), "sets": joined.get(a.id, []), "labels": by_account.get(a.id, 0) + by_name.get(a.handle.casefold(), 0),
             "tokens": tokens.get(a.id, 0)} for a in accounts.everyone()]


@router.get("/admin/users", dependencies=admin)
def users(request: Request) -> dict:
    """Every account (with its sets and labels), every labelling set to add people to, and the audit log."""
    return {"users": _rows(), "sets": _sets(), "audit": accounts.audit_log(), "me": _actor(request).id, "past_names": accounts.past_names()}


class NewUser(BaseModel):
    name: str
    role: str = accounts.ANNOTATOR
    project: str = ""
    session: str = ""


@router.post("/admin/users", dependencies=admin)
def add_user(body: NewUser, request: Request) -> dict:
    """A new account with a one-time password (returned only here), optionally in a labelling set."""
    try:
        made, password = accounts.add_with_temporary(body.name, body.role)
    except accounts.AccountError as error:
        _error(error)
    accounts.audit(_actor(request), "add", made, f"role {made.role}")
    if body.project and body.session:
        accounts.add_member(made.id, body.project, body.session)
        accounts.audit(_actor(request), "add to set", made, f"{body.project} / {body.session}")
    return {"user": made.to_api(), "password": password}


@router.post("/admin/users/{account_id}/reset", dependencies=admin)
def reset(account_id: str, request: Request) -> dict:
    """A new one-time password (returned only here); the person chooses their own when signing in with it."""
    _not_self(request, account_id, "reset the password of (change it on your Account page)")
    target = _target(account_id)
    try:
        password = accounts.reset_password(account_id)
    except accounts.AccountError as error:
        _error(error)
    accounts.audit(_actor(request), "reset password", target)
    return {"password": password}


class Role(BaseModel):
    role: str


@router.put("/admin/users/{account_id}/role", dependencies=admin)
def role(account_id: str, body: Role, request: Request) -> dict:
    target = _target(account_id)
    if body.role != accounts.ADMIN:
        _not_self(request, account_id, "demote")
    try:
        changed = accounts.set_role(account_id, body.role)
    except accounts.AccountError as error:
        _error(error)
    accounts.audit(_actor(request), "role", target, f"{target.role} → {changed.role}")
    return changed.to_api()


class Disabled(BaseModel):
    disabled: bool


@router.put("/admin/users/{account_id}/disabled", dependencies=admin)
def disable(account_id: str, body: Disabled, request: Request) -> dict:
    target = _target(account_id)
    if body.disabled:
        _not_self(request, account_id, "disable")
    try:
        changed = accounts.set_disabled(account_id, body.disabled)
    except accounts.AccountError as error:
        _error(error)
    accounts.audit(_actor(request), "disable" if body.disabled else "enable", target)
    return changed.to_api()


@router.delete("/admin/users/{account_id}", dependencies=admin)
def remove(account_id: str, request: Request, confirm: str = "") -> dict:
    """Delete the account (typed confirmation: its name). Its labels keep their attribution and are never deleted."""
    _not_self(request, account_id, "remove")
    target = _target(account_id)
    if " ".join(confirm.split()).casefold() != target.name.casefold():
        raise HTTPException(400, f"type the name “{target.name}” to remove this account")
    try:
        accounts.remove(account_id)
    except accounts.AccountError as error:
        _error(error)
    accounts.audit(_actor(request), "remove", None, "labels kept", target_name=target.name)
    return {"ok": True}


class Membership(BaseModel):
    project: str
    session: str


@router.post("/admin/users/{account_id}/sets", dependencies=admin)
def add_to_set(account_id: str, body: Membership, request: Request) -> dict:
    target = _target(account_id)
    if not any(s["project"] == body.project and s["session"] == body.session for s in _sets()):
        raise HTTPException(404, "no such labelling set")
    accounts.add_member(account_id, body.project, body.session)
    accounts.audit(_actor(request), "add to set", target, f"{body.project} / {body.session}")
    return {"ok": True}


@router.delete("/admin/users/{account_id}/sets/{project}/{name}", dependencies=admin)
def remove_from_set(account_id: str, project: str, name: str, request: Request) -> dict:
    target = _target(account_id)
    accounts.remove_member(account_id, project, name)
    accounts.audit(_actor(request), "remove from set", target, f"{project} / {name}")
    return {"ok": True}


@router.get("/admin/users/{account_id}/tokens", dependencies=admin)
def tokens(account_id: str) -> list[dict]:
    """The account's API tokens: label, made, last used (never the token: only its hash is kept)."""
    _target(account_id)
    return accounts.tokens(account_id)


class TokenLabel(BaseModel):
    label: str = ""


@router.post("/admin/users/{account_id}/tokens", dependencies=admin)
def new_token(account_id: str, body: TokenLabel, request: Request) -> dict:
    """A new API token of the account (returned only here)."""
    target = _target(account_id)
    token = accounts.new_token(account_id, body.label)
    accounts.audit(_actor(request), "new token", target, body.label)
    return {"token": token}


@router.delete("/admin/users/{account_id}/tokens/{token_id}", dependencies=admin)
def revoke(account_id: str, token_id: str, request: Request) -> dict:
    target = _target(account_id)
    accounts.revoke_token(account_id, token_id)
    accounts.audit(_actor(request), "revoke token", target, token_id)
    return {"ok": True}


class PastName(BaseModel):
    past_name: str = ""


@router.put("/admin/users/{account_id}/past-name", dependencies=admin)
def past_name(account_id: str, body: PastName, request: Request) -> dict:
    """"This account is Ann": it labels under a name past labels carry and sees that history ("" unlinks)."""
    target = _target(account_id)
    try:
        changed = accounts.link_past_name(account_id, body.past_name)
    except accounts.AccountError as error:
        _error(error)
    accounts.audit(_actor(request), "past name", target, body.past_name or "unlinked")
    return changed.to_api()


@router.get("/admin/audit", dependencies=admin)
def audit(limit: int = 200) -> list[dict]:
    return accounts.audit_log(min(max(1, limit), 2000))
