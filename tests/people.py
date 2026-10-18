"""Tests: people with accounts (`stainid.api.accounts`) who joined labelling sets."""
from __future__ import annotations

from fastapi.testclient import TestClient

from stainid.api import accounts, state
from stainid.labels import sessions as session_store

PASSWORD = "correct horse"


def annotator(admin: TestClient, name: str, *sessions: str, every: bool = False) -> TestClient:
    """A new browser of a person who signed up as `name` (an annotator) and is a member of `sessions` (`every`: of
    every session of the project, also the ones made later in the test)."""
    person = TestClient(admin.app)
    made = person.post("/api/session/register", json={"name": name, "password": PASSWORD})
    if made.status_code == 409:  # the name exists already (another browser of the same person), or past labels carry it
        made = person.post("/api/session/login", json={"name": name, "password": PASSWORD})
        if made.status_code != 200:
            assert admin.post("/api/session/people", json={"name": name, "password": PASSWORD}).status_code == 200
            made = person.post("/api/session/login", json={"name": name, "password": PASSWORD})
    assert made.status_code == 200, made.text
    person.account = made.json()["account"]
    for session in sessions:
        join(admin, person, session)
    if every:
        account = person.account["id"]
        for session in session_store.names(state.get_state().project.output("reviews")):
            accounts.add_member(account, state.get_state().id, session)

        def member_of_new(root, session: str) -> None:
            accounts.add_member(account, state.get_state().id, session)

        member_of_new.test_only = True  # removed after the test (conftest)
        session_store.ON_CREATE.append(member_of_new)
    return person


def join(admin: TestClient, person: TestClient, session: str, base: str = "") -> None:
    """The admin adds the person to a session (as joining with its key does)."""
    added = admin.post(f"{base}/api/labels/sessions/{session}/access/members", json={"account": person.account["id"]})
    assert added.status_code == 200, added.text
