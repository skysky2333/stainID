"""The admin's Users page (`api.routes.users`): every route admins only, the self and last-admin guards, one-time
passwords and tokens returned once only, disabled accounts locked out (sign-in, sessions, tokens), memberships of any
set, removal keeping labels, the audit log, and everyone's own "sign out elsewhere"."""
# ruff: noqa: F811 - the fixtures of the API tests are imported and used as arguments
from __future__ import annotations

import json
import sqlite3

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402
from people import PASSWORD, annotator  # noqa: E402
from test_api import client, new_session, prepared  # noqa: E402, F401

from stainid.api import accounts, state  # noqa: E402

ROUTES = [("get", "/api/admin/users", None), ("post", "/api/admin/users", {"name": "Zed"}), ("post", "/api/admin/users/X/reset", {}),
          ("put", "/api/admin/users/X/role", {"role": "admin"}), ("put", "/api/admin/users/X/disabled", {"disabled": True}),
          ("delete", "/api/admin/users/X?confirm=X", None), ("post", "/api/admin/users/X/sets", {"project": "p", "session": "s"}),
          ("delete", "/api/admin/users/X/sets/p/s", None), ("get", "/api/admin/users/X/tokens", None), ("post", "/api/admin/users/X/tokens", {}),
          ("delete", "/api/admin/users/X/tokens/t", None), ("put", "/api/admin/users/X/past-name", {"past_name": ""}), ("get", "/api/admin/audit", None)]


def _call(who: TestClient, method: str, path: str, body):
    return getattr(who, method)(path, json=body) if body is not None else getattr(who, method)(path)


def _me(client) -> str:
    return client.get("/api/session").json()["account"]["id"]


def test_every_users_route_is_for_admins_only(client):
    ann = annotator(client, "Ann")
    guest = TestClient(client.app)
    for method, path, body in ROUTES:
        path = path.replace("X", ann.account["id"])
        assert _call(ann, method, path, body).status_code == 403, path
        assert _call(guest, method, path, body).status_code == 401, path
    assert accounts.get(ann.account["id"]).role == "annotator" and not accounts.audit_log()  # nothing happened


def test_adding_resetting_and_one_time_passwords_shown_once(prepared):
    client = prepared
    session_name = new_session(client, open=True)["name"]
    project = state.get_state().id
    made = client.post("/api/admin/users", json={"name": "Bea", "role": "annotator", "project": project, "session": session_name})
    assert made.status_code == 200, made.text
    bea, password = made.json()["user"], made.json()["password"]
    listed = client.get("/api/admin/users").json()
    row = next(u for u in listed["users"] if u["id"] == bea["id"])
    assert [(s["project"], s["session"]) for s in row["sets"]] == [(project, session_name)] and row["labels"] == 0 and not row["disabled"]
    assert any(s["session"] == session_name for s in listed["sets"])
    assert password not in json.dumps(listed)  # never retrievable later: only its hash is kept
    with sqlite3.connect(accounts.db_path()) as db:
        assert password not in json.dumps(db.execute("select * from accounts").fetchall())
    person = TestClient(client.app)
    signed = person.post("/api/session/login", json={"name": "Bea", "password": password}).json()
    assert signed["account"]["must_change"] is True  # one-time: they choose their own
    assert person.put("/api/session/password", json={"new": PASSWORD}).status_code == 200
    again = client.post(f"/api/admin/users/{bea['id']}/reset").json()["password"]
    assert again != password and again not in json.dumps(client.get("/api/admin/users").json())
    assert TestClient(client.app).post("/api/session/login", json={"name": "Bea", "password": PASSWORD}).status_code == 403
    assert TestClient(client.app).post("/api/session/login", json={"name": "Bea", "password": again}).status_code == 200
    # tokens: shown once, listed without the token
    token = client.post(f"/api/admin/users/{bea['id']}/tokens", json={"label": "script"}).json()["token"]
    found = client.get(f"/api/admin/users/{bea['id']}/tokens").json()
    assert [t["label"] for t in found] == ["script"] and token not in json.dumps(found)
    assert TestClient(client.app).get("/api/session", headers={"authorization": f"Bearer {token}"}).json()["account"]["name"] == "Bea"
    assert client.delete(f"/api/admin/users/{bea['id']}/tokens/{found[0]['id']}").status_code == 200
    assert TestClient(client.app).get("/api/session", headers={"authorization": f"Bearer {token}"}).json()["account"] is None
    # memberships of any set
    assert client.delete(f"/api/admin/users/{bea['id']}/sets/{project}/{session_name}").status_code == 200
    assert not accounts.is_member(bea["id"], project, session_name)
    assert client.post(f"/api/admin/users/{bea['id']}/sets", json={"project": project, "session": "nope"}).status_code == 404
    assert client.post(f"/api/admin/users/{bea['id']}/sets", json={"project": project, "session": session_name}).status_code == 200
    assert accounts.is_member(bea["id"], project, session_name)


def test_disabled_accounts_cannot_sign_in_or_use_sessions_or_tokens(client):
    ann = annotator(client, "Ann")
    token = accounts.new_token(ann.account["id"], "x")
    assert ann.get("/api/session").json()["account"]["name"] == "Ann"
    assert client.put(f"/api/admin/users/{ann.account['id']}/disabled", json={"disabled": True}).json()["disabled"] is True
    assert ann.get("/api/session").json()["account"] is None  # its session ends
    assert TestClient(client.app).post("/api/session/login", json={"name": "Ann", "password": PASSWORD}).status_code == 403
    assert TestClient(client.app).get("/api/session", headers={"authorization": f"Bearer {token}"}).json()["account"] is None
    assert client.put(f"/api/admin/users/{ann.account['id']}/disabled", json={"disabled": False}).json()["disabled"] is False
    assert TestClient(client.app).post("/api/session/login", json={"name": "Ann", "password": PASSWORD}).status_code == 200
    assert TestClient(client.app).get("/api/session", headers={"authorization": f"Bearer {token}"}).json()["account"]["name"] == "Ann"


def test_self_and_last_admin_guards(client):
    me = _me(client)
    for method, path, body in (("put", f"/api/admin/users/{me}/role", {"role": "annotator"}), ("put", f"/api/admin/users/{me}/disabled", {"disabled": True}),
                               ("delete", f"/api/admin/users/{me}?confirm=admin", None), ("delete", f"/api/session/people/{me}", None),
                               ("put", f"/api/session/people/{me}", {"role": "annotator"})):
        assert _call(client, method, path, body).status_code == 409, path
    assert accounts.get(me).admin and not accounts.get(me).disabled
    # the last admin who can sign in stays, whoever asks
    other = client.post("/api/admin/users", json={"name": "Second", "role": "admin"}).json()
    second = TestClient(client.app)
    second.post("/api/session/login", json={"name": "Second", "password": other["password"]})
    second.put("/api/session/password", json={"new": PASSWORD})
    assert second.put(f"/api/admin/users/{me}/disabled", json={"disabled": True}).status_code == 200  # two admins: fine
    with pytest.raises(accounts.AccountError):
        accounts.set_disabled(other["user"]["id"], True)  # the only enabled admin now
    with pytest.raises(accounts.AccountError):
        accounts.set_role(other["user"]["id"], "annotator")
    with pytest.raises(accounts.AccountError):
        accounts.remove(other["user"]["id"])
    assert second.put(f"/api/admin/users/{me}/disabled", json={"disabled": False}).status_code == 200


def test_removing_needs_the_typed_name_keeps_labels_and_everything_is_audited(client):
    ann = annotator(client, "Ann")
    ann_id = ann.account["id"]
    assert client.put(f"/api/admin/users/{ann_id}/role", json={"role": "admin"}).json()["role"] == "admin"
    assert client.put(f"/api/admin/users/{ann_id}/role", json={"role": "annotator"}).json()["role"] == "annotator"
    assert client.delete(f"/api/admin/users/{ann_id}?confirm=Bob").status_code == 400
    assert accounts.get(ann_id) is not None
    root = state.get_state().project.output("reviews")
    from stainid.labels import store  # noqa: PLC0415

    store.append(root, [{"id": "note:x", "task": "note", "value": "hi", "source": "Ann", "account": ann_id, "session": "", "tile_id": "t"}])
    assert next(u for u in client.get("/api/admin/users").json()["users"] if u["id"] == ann_id)["labels"] == 1
    assert client.delete(f"/api/admin/users/{ann_id}?confirm=ann").status_code == 200
    assert accounts.get(ann_id) is None
    kept = store.current(root)
    assert kept[kept.id == "note:x"].iloc[0]["source"] == "Ann"  # labels keep their attribution
    log = client.get("/api/admin/audit").json()
    me = client.get("/api/session").json()["account"]["name"]
    assert [e["action"] for e in log][:3] == ["remove", "role", "role"] and all(e["actor_name"] == me and e["time"] for e in log)
    assert log[0]["target_name"] == "Ann" and "labels kept" in log[0]["detail"]
    assert client.get("/api/admin/users").json()["audit"][0]["action"] == "remove"


def test_everyone_signs_out_their_other_sessions_and_changes_their_own_password(client):
    ann = annotator(client, "Ann")
    other = TestClient(client.app)
    other.post("/api/session/login", json={"name": "Ann", "password": PASSWORD})
    assert other.get("/api/session").json()["account"]["name"] == "Ann"
    assert ann.post("/api/session/sign-out-elsewhere").status_code == 200
    assert ann.get("/api/session").json()["account"]["name"] == "Ann"  # this one stays
    assert other.get("/api/session").json()["account"] is None
    assert ann.put("/api/session/password", json={"current": "wrong one!", "new": "another password"}).status_code == 403
    assert ann.put("/api/session/password", json={"current": PASSWORD, "new": "another password"}).status_code == 200
    assert TestClient(client.app).post("/api/session/login", json={"name": "Ann", "password": "another password"}).status_code == 200
