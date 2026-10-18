"""Who may label a session (admins): its join key and its members (`stainid.api.accounts`), and the sets of a person
across the server's projects (their "my sets" page)."""
from __future__ import annotations

from collections import defaultdict

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from stainid.api import accounts, state
from stainid.api.session import require_admin
from stainid.labels import sessions

router = APIRouter(tags=["labels"])
admin = [Depends(require_admin)]


def _root():
    return state.get_state().project.output("reviews")


def _session(name: str) -> str:
    if name not in sessions.names(_root()):
        raise HTTPException(404, f"no labelling session {name}")
    return name


def _made(root, name: str) -> None:
    """Every new session gets its join key at once (`sessions.ON_CREATE`)."""
    accounts.access(state.get_state().id, name)


if _made not in sessions.ON_CREATE:
    sessions.ON_CREATE.append(_made)


class Joining(BaseModel):
    enabled: bool


class Member(BaseModel):
    account: str


@router.get("/labels/sessions/{name}/access", dependencies=admin)
def access(name: str) -> dict:
    """The session's join key (people who have it add the session to their sets), whether joining is on, its members."""
    return accounts.access(state.get_state().id, _session(name))


@router.post("/labels/sessions/{name}/access/key", dependencies=admin)
def new_key(name: str) -> dict:
    """A new key: the old one stops working; people who joined keep their access."""
    return accounts.regenerate_key(state.get_state().id, _session(name))


@router.put("/labels/sessions/{name}/access", dependencies=admin)
def joining(name: str, body: Joining) -> dict:
    return accounts.set_joining(state.get_state().id, _session(name), body.enabled)


@router.post("/labels/sessions/{name}/access/members", dependencies=admin)
def add_member(name: str, body: Member) -> dict:
    if accounts.get(body.account) is None:
        raise HTTPException(404, "no such account")
    accounts.add_member(body.account, state.get_state().id, _session(name))
    return accounts.access(state.get_state().id, name)


@router.delete("/labels/sessions/{name}/access/members/{account_id}", dependencies=admin)
def remove_member(name: str, account_id: str) -> dict:
    accounts.remove_member(account_id, state.get_state().id, _session(name))
    return accounts.access(state.get_state().id, name)


def may_label(request: Request, name: str) -> None:
    """Annotators only reach the sessions they joined (admins every one); checked on every annotation request."""
    from stainid.api import session  # noqa: PLC0415

    account = session.current_account(request)
    if account is None:
        raise HTTPException(401, "sign in first")
    if account.must_change:
        raise HTTPException(403, "choose a new password first")
    if account.admin:
        return
    if not accounts.is_member(account.id, state.get_state().id, name):
        raise HTTPException(403, "this labelling set is not in your sets: add it with the key you were given")


def sets_of(account: accounts.Account, request: Request, here: str | None = None) -> list[dict]:
    """The labelling sets of the person, of every project, with their progress; for an admin also every session of the
    project `here` (admins open any session)."""
    from stainid.api.routes import annotate  # noqa: PLC0415

    joined: dict[str, dict[str, float]] = defaultdict(dict)
    for project, name, when in accounts.sets_of(account.id):
        joined[project][name] = when
    if account.admin and here:
        joined.setdefault(here, {})
    out = []
    for project, names in joined.items():
        try:
            with state.use_project(project):
                current = state.get_state()
                every = account.admin and project == here
                rows = annotate.open_sessions(request, account.handle, None if every else set(names))
                title = current.project.name
        except (state.UnknownProject, OSError, ValueError):
            continue
        for row in rows:
            out.append({**row, "project": project, "project_name": title, "joined": names.get(row["name"]),
                        "link": f"/p/{project}/annotate/{row['name']}"})
    return sorted(out, key=lambda r: -(r["joined"] or 0))
