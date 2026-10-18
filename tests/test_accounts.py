"""Accounts: one sign-in for admins and annotators, roles, join keys per labelling session, membership checked on every
annotation request, labels attributed to the account, past names reserved (`stainid.api.accounts`)."""
# ruff: noqa: F811 - the fixtures of the API tests (`client`, `prepared`) are imported and used as arguments
from __future__ import annotations

import json
import re
import sqlite3

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402
from people import PASSWORD, annotator  # noqa: E402
from test_api import TILE, client, new_session, prepared  # noqa: E402, F401 - the fixtures of the API tests

from stainid.api import accounts, session  # noqa: E402
from stainid.labels import store  # noqa: E402


def _login(app_client, name, password=PASSWORD):
    return TestClient(app_client.app).post("/api/session/login", json={"name": name, "password": password})


def test_sign_up_sign_in_wrong_password_and_rate_limit(client):
    person = TestClient(client.app)
    made = person.post("/api/session/register", json={"name": "Jane Doe", "password": PASSWORD})
    assert made.status_code == 200 and made.json()["account"]["role"] == "annotator" and made.json()["admin"] is False
    assert person.get("/api/session").json()["annotator"] == "Jane Doe"
    cookie = made.headers["set-cookie"].lower()
    assert "httponly" in cookie and "samesite=lax" in cookie
    assert person.post("/api/session/register", json={"name": "JANE  doe", "password": PASSWORD}).status_code == 409  # case-insensitive
    assert person.post("/api/session/register", json={"name": "Short", "password": "short"}).status_code == 400
    assert _login(client, "jane doe").status_code == 200
    assert _login(client, "Jane Doe", "wrong password").status_code == 403
    # passwords are kept as scrypt hashes with their own salt, never as text
    with sqlite3.connect(accounts.db_path()) as db:
        rows = db.execute("select name, kdf, salt, hash from accounts").fetchall()
    assert all(PASSWORD not in json.dumps(r) for r in rows) and {r[1] for r in rows} == {"scrypt"} and len({r[2] for r in rows}) == len(rows)
    # sign-ins are rate-limited per name: after `LOGIN_TRIES` wrong ones even the right password waits
    for _ in range(accounts.LOGIN_TRIES):
        _login(client, "Jane Doe", "wrong password")
    assert _login(client, "Jane Doe", "wrong password").status_code == 429
    assert _login(client, "Jane Doe").status_code == 429
    accounts.reset_limits()
    assert _login(client, "Jane Doe").status_code == 200
    # behind https (the Cloudflare tunnel) the cookie is Secure
    secure = TestClient(client.app).post("/api/session/login", json={"name": "Jane Doe", "password": PASSWORD}, headers={"x-forwarded-proto": "https"})
    assert "secure" in secure.headers["set-cookie"].lower()


def test_one_sign_in_page_serves_both_roles_and_annotators_get_403_on_admin_apis(client):
    ann = annotator(client, "Ann")
    admin = TestClient(client.app)
    assert admin.post("/api/session/login", json={"name": "admin", "password": "admin"}).json()["admin"] is True  # the same page
    assert admin.get("/api/session").json()["account"]["role"] == "admin"
    for path in ("/api/cores", "/api/summary", "/api/labels/overview", "/api/session/people", "/api/session/work", "/api/projects"):
        assert ann.get(path).status_code == 403, path
        assert admin.get(path).status_code == 200, path
    assert ann.post("/api/labels/sessions", json={"name": "x", "stain": "6E10", "tile_ids": [TILE]}).status_code == 403
    assert ann.put("/api/session/people/whoever", json={"role": "admin"}).status_code == 403
    assert TestClient(client.app).get("/api/cores").status_code == 401  # nobody signed in


def test_admins_manage_people_and_there_is_always_an_admin(client):
    people = client.get("/api/session/people").json()["people"]
    first = next(p for p in people if p["name"] == "admin")
    assert client.put(f"/api/session/people/{first['id']}", json={"role": "annotator"}).status_code == 409  # the last admin
    assert client.delete(f"/api/session/people/{first['id']}").status_code == 409
    ann = annotator(client, "Ann")
    ann_id = ann.account["id"]
    assert client.put(f"/api/session/people/{ann_id}", json={"role": "admin"}).json()["role"] == "admin"
    assert ann.get("/api/cores").status_code == 401  # a new role signs the account in again
    ann = TestClient(client.app)
    ann.post("/api/session/login", json={"name": "Ann", "password": PASSWORD})
    assert ann.get("/api/cores").status_code == 200
    assert client.put(f"/api/session/people/{first['id']}", json={"role": "annotator"}).json()["role"] == "annotator"
    assert ann.put(f"/api/session/people/{ann_id}", json={"role": "annotator"}).status_code == 409  # now Ann is the last


def test_an_admin_resets_a_password_and_the_person_sets_a_new_one(client):
    ann = annotator(client, "Ann")
    temporary = client.post(f"/api/session/people/{ann.account['id']}/reset").json()["password"]
    assert _login(client, "Ann").status_code == 403  # the old password stops working
    person = TestClient(client.app)
    signed = person.post("/api/session/login", json={"name": "Ann", "password": temporary}).json()
    assert signed["account"]["must_change"] is True
    assert person.get("/api/annotate").status_code == 401  # nothing until a new password is chosen
    assert person.put("/api/session/password", json={"new": "a brand new one"}).status_code == 200
    assert person.get("/api/annotate").status_code == 200
    assert _login(client, "Ann", "a brand new one").status_code == 200


def test_the_old_admin_password_becomes_the_first_admin_account(client):
    accounts.db_path().unlink()
    accounts.forget()
    session.set_legacy_password("old-lab-password")
    assert _login(client, "admin", "old-lab-password").json()["admin"] is True
    assert _login(client, "admin", "admin").status_code == 403
    assert TestClient(client.app).post("/api/session/admin", json={"password": "old-lab-password"}).json()["admin"] is True  # old page


def test_api_tokens_of_an_account_for_scripts(client):
    token = client.post("/api/session/tokens", json={"label": "pipeline script"}).json()["token"]
    script = TestClient(client.app)
    assert script.get("/api/cores", headers={"Authorization": f"Bearer {token}"}).status_code == 200
    assert script.get("/api/cores", params={"access_token": token}).status_code == 200
    listed = client.get("/api/session/tokens").json()
    assert [t["label"] for t in listed] == ["pipeline script"] and token not in json.dumps(listed)
    client.delete(f"/api/session/tokens/{listed[0]['id']}")
    assert script.get("/api/cores", headers={"Authorization": f"Bearer {token}"}).status_code == 401
    ann = annotator(client, "Ann")
    ann_token = ann.post("/api/session/tokens", json={}).json()["token"]
    assert script.get("/api/cores", headers={"Authorization": f"Bearer {ann_token}"}).status_code == 403  # an annotator's token


def test_a_join_key_adds_the_set_and_a_new_key_stops_the_old_one(prepared):
    client = prepared
    new_session(client, open=True)
    access = client.get("/api/labels/sessions/plaques_together/access").json()
    key = access["key"]
    assert re.fullmatch(r"[23456789a-hjkmnp-z]{5}-[23456789a-hjkmnp-z]{5}", key) and access["enabled"] and access["members"] == []
    ann = annotator(client, "Ann")
    assert ann.get("/api/annotate/plaques_together").status_code == 403
    assert ann.get("/api/session/sets").json() == []
    added = ann.post("/api/session/sets", json={"key": key.upper().replace("-", " ")})  # typed loosely
    assert added.status_code == 200 and added.json()["name"] == "plaques_together" and added.json()["link"].endswith("/annotate/plaques_together")
    assert [s["name"] for s in ann.get("/api/session/sets").json()] == ["plaques_together"]
    assert ann.get("/api/annotate/plaques_together").status_code == 200
    assert [m["name"] for m in client.get("/api/labels/sessions/plaques_together/access").json()["members"]] == ["Ann"]

    fresh = client.post("/api/labels/sessions/plaques_together/access/key").json()["key"]
    assert fresh != key
    ben = annotator(client, "Ben")
    assert ben.post("/api/session/sets", json={"key": key}).status_code == 404  # the old key stops working
    assert ann.get("/api/annotate/plaques_together").status_code == 200  # Ann keeps her access
    client.put("/api/labels/sessions/plaques_together/access", json={"enabled": False})
    off = ben.post("/api/session/sets", json={"key": fresh})
    assert off.status_code == 403 and "switched off" in off.json()["detail"]  # joining is off: said so, not "replaced"
    client.put("/api/labels/sessions/plaques_together/access", json={"enabled": True})
    assert ben.post("/api/session/sets", json={"key": fresh}).status_code == 200
    client.delete(f"/api/labels/sessions/plaques_together/access/members/{ben.account['id']}")
    assert ben.get("/api/annotate/plaques_together").status_code == 403  # removed by the admin
    # wrong keys are rate-limited
    for _ in range(accounts.JOIN_TRIES):
        ben.post("/api/session/sets", json={"key": "aaaaa-aaaaa"})
    assert ben.post("/api/session/sets", json={"key": fresh}).status_code == 429


def _filled(route, session_name: str) -> tuple[str, dict, dict | None]:
    """A request to an annotation route with every parameter given: the address, the query and a body."""
    values = {"name": session_name, "tile_id": TILE, "core_id": "LIP-1_B-2", "stain": "6E10", "layer": "dab", "level": "0", "col": "0", "row": "0"}
    path = "/api" + re.sub(r"\{(\w+)(:\w+)?\}", lambda m: values.get(m.group(1), "x"), route.path)
    query = {p.name: (TILE if p.name == "tile_id" else 1) for p in route.dependant.query_params if p.field_info.is_required()}
    body = None
    if route.body_field is not None:
        schema = route.body_field.field_info.annotation.model_json_schema()
        dummy = {"string": "x", "number": 1, "integer": 1, "boolean": False, "array": [], "object": {}}
        body = {k: dummy.get(schema["properties"][k].get("type", "string"), "x") for k in schema.get("required", [])}
    return path, query, body


def test_a_non_member_gets_403_on_every_annotation_request_and_labels_carry_the_account(prepared):
    client = prepared
    new_session(client, open=True)
    outsider = annotator(client, "Olga")
    checked = 0
    from stainid.api.routes import annotate, areas, classes, outline, rounds

    for route in [r for module in (annotate, rounds, areas, outline, classes) for r in module.router.routes]:
        if not route.path.startswith("/annotate/{name}"):
            continue
        for method in route.methods - {"HEAD"}:
            path, query, body = _filled(route, "plaques_together")
            answer = outsider.request(method, path, params=query, json=body)
            assert answer.status_code == 403, (method, route.path, answer.status_code, answer.text)
            checked += 1
    assert checked >= 30
    assert outsider.get("/api/annotate").json() == []  # not listed either

    ann = annotator(client, "Ann", "plaques_together")
    saved = ann.post("/api/annotate/plaques_together/labels", json={"records": [{"task": "class", "tile_id": TILE, "x": 100.0, "y": 200.0,
                                                                                  "value": "not a plaque"}]}).json()
    assert saved[0]["source"] == "Ann" and saved[0]["account"] == ann.account["id"]
    logged = [json.loads(line) for line in store.log_path(_reviews()).read_text().splitlines()]
    assert logged[-1]["account"] == ann.account["id"]


def _reviews():
    from stainid.api.state import get_state

    return get_state().project.output("reviews")


def test_the_command_line_makes_api_tokens_for_scripts(client, capsys):
    from stainid.cli import main

    main(["accounts", "token", "admin", "--label", "nightly export"])
    token = capsys.readouterr().out.strip()
    assert token.startswith(accounts.TOKEN_PREFIX)
    assert TestClient(client.app).get("/api/cores", headers={"Authorization": f"Bearer {token}"}).status_code == 200
    main(["accounts", "list"])
    assert "admin" in capsys.readouterr().out


def test_the_tutorial_works_without_an_account(prepared):
    guest = TestClient(prepared.app)
    assert guest.get("/api/tutorial").status_code == 200
    assert guest.get("/api/annotate").status_code == 401  # but no real session


def test_past_names_are_reserved_and_an_admin_links_an_account_to_one(prepared):
    client = prepared
    new_session(client, open=True)
    store.append(_reviews(), [{"source": "Ann", "session": "plaques_together", "stain": "6E10", "task": "class", "tile_id": TILE, "x": 100.0,
                               "y": 200.0, "value": "not a plaque"}])
    taken = TestClient(client.app).post("/api/session/register", json={"name": "ann", "password": PASSWORD})
    assert taken.status_code == 409 and "labels by" in taken.json()["detail"]
    names = client.get("/api/session/people").json()["past_names"]
    assert {"name": "ann", "account": None, "account_name": None} in names
    anna = annotator(client, "Anna Smith", "plaques_together")
    assert anna.get("/api/annotate").json()[0]["my_labels"] == 0
    assert client.put(f"/api/session/people/{anna.account['id']}", json={"past_name": "Nobody"}).status_code == 404
    linked = client.put(f"/api/session/people/{anna.account['id']}", json={"past_name": "Ann"}).json()
    assert linked["handle"] == "Ann"
    assert anna.get("/api/annotate").json()[0]["my_labels"] == 1  # the account sees Ann's history as its own
    other = annotator(client, "Other")
    assert client.put(f"/api/session/people/{other.account['id']}", json={"past_name": "Ann"}).status_code == 409  # one account per name
