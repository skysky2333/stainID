# ruff: noqa: F811  (the rounds fixture is shared with test_annotation_rounds)
"""The tutorial's practice keeps nothing, whatever is clicked: every write endpoint of the annotation pages, called for a
practice session (as a guest and signed in), leaves every store as it was — the project's files (label log, sessions,
candidate store, added objects, exclusions), the accounts database, and the holds and "who is on which field" claims.
Under the routes, the stores themselves refuse to write while practising (`stainid.practice`)."""
from __future__ import annotations

import copy
import os
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402
from test_annotation_rounds import TILE, rounds_project  # noqa: E402, F401

from stainid import practice, tutorial  # noqa: E402
from stainid.api import registry, state  # noqa: E402
from stainid.api.routes import annotate, areas, rounds  # noqa: E402
from stainid.labels import sessions, store  # noqa: E402

FIELD = "practice-2-6E10"
CORES = "practice-areas"


def _files(*roots: Path) -> dict[str, tuple[int, int]]:
    """Every file under these folders, with its size and change time (the derived caches aside: a practice field's
    picture may be cut once, like anyone's look at it)."""
    out = {}
    for top in roots:
        for folder, _, names in os.walk(top):
            if "cache" in Path(folder).relative_to(top).parts[:3]:
                continue
            for name in names:
                path = Path(folder) / name
                stat = path.stat()
                out[str(path)] = (stat.st_size, stat.st_mtime_ns)
    return out


def _claims() -> list:
    return copy.deepcopy([dict(annotate._BUSY), rounds._TILES_HELD._held, areas._HELD._held, annotate._HELD._held])


def _writes(client: TestClient) -> dict[str, tuple[str, dict]]:
    """A call of every write endpoint of the annotation pages for the practice sessions, with a body that would change
    something in a real session."""
    field = client.get(f"/api/annotate/{FIELD}/fields/{TILE}").json()
    own = next(o for o in field["objects"] if o["own"])
    detail = client.get(f"/api/annotate/{CORES}").json()
    core = detail["meta"]["cores"][0]
    stains = detail["meta"]["stains_of"][core]
    square = [[10.0, 10.0], [60.0, 10.0], [60.0, 60.0], [10.0, 60.0]]
    return {
        "/api/annotate/{name}/labels": (f"/api/annotate/{FIELD}/labels", {"records": [{"task": "class", "tile_id": TILE, "x": 20.0, "y": 20.0, "value": "CAA"}]}),
        "/api/annotate/{name}/fields/{tile_id}/finish": (f"/api/annotate/{FIELD}/fields/{TILE}/finish", {}),
        "/api/annotate/{name}/live": (f"/api/annotate/{FIELD}/live", {"tile_id": TILE}),
        "/api/annotate/{name}/fields/{tile_id}/leave": (f"/api/annotate/{FIELD}/fields/{TILE}/leave", {}),
        "/api/annotate/{name}/tiles/{tile_id}/save": (f"/api/annotate/{FIELD}/tiles/{TILE}/save",
                                                       {"records": [{"task": "class", "candidate": own["candidate_id"], "value": "CAA"},
                                                                    {"task": "class", "tile_id": TILE, "x": 30.0, "y": 30.0, "value": "CAA"}], "submit": True}),
        "/api/annotate/{name}/fields/{tile_id}/split": (f"/api/annotate/{FIELD}/fields/{TILE}/split", {"points": square, "seed": [20.0, 20.0]}),
        "/api/annotate/{name}/fields/{tile_id}/merge": (f"/api/annotate/{FIELD}/fields/{TILE}/merge",
                                                        {"outlines": [square, [[p[0] + 45, p[1]] for p in square]]}),
        "/api/annotate/{name}/fields/{tile_id}/segment": (f"/api/annotate/{FIELD}/fields/{TILE}/segment", {"points": [[20.0, 20.0, 1]]}),
        "/api/annotate/{name}/cores/{core_id}/hold": (f"/api/annotate/{CORES}/cores/{core}/hold", {}),
        "/api/annotate/{name}/cores/{core_id}/save": (f"/api/annotate/{CORES}/cores/{core}/save", {"checked": stains, "edits": {}}),
    }


def test_every_write_of_a_practice_session_leaves_every_store_unchanged(rounds_project):
    admin, (ann, _), _ = rounds_project
    project = state.get_state().project
    guest = TestClient(admin.app)  # nobody signed in: the tutorial is open to anyone
    for who in (guest, ann):
        calls = _writes(who)
        writes = {path for path, methods in admin.app.openapi()["paths"].items() if path.startswith("/api/annotate/{name}")
                  and set(methods) & {"post", "put", "patch", "delete"}}
        assert writes == set(calls), "a new write endpoint: add it here (and make it keep nothing in practice)"
        roots = (project.root, registry.SETTINGS.parent, tutorial.folder())  # the tutorial's own project too
        before, claims, log = _files(*roots), _claims(), store.history(project.output("reviews"))
        for template, (path, body) in calls.items():
            answer = who.post(path, json=body)
            assert answer.status_code < 500, (template, answer.status_code, answer.text)
            assert _files(*roots) == before, template
            assert _claims() == claims, template
        assert store.history(project.output("reviews")).equals(log)
        saved = who.post(f"/api/annotate/{FIELD}/tiles/{TILE}/save", json={"records": [], "submit": True})
        assert saved.status_code == 200 and saved.json()["submitted"]  # Save and continue answers as if it worked


def test_the_stores_refuse_to_write_while_practising(tmp_path):
    with practice.practice():
        with pytest.raises(practice.NotKept):
            store.append(tmp_path, [{"id": "x", "source": "Ann", "task": "class"}])
        with pytest.raises(practice.NotKept):
            sessions.Reservations().hold(("p", "s"), "Ann", ["tile"])
        with pytest.raises(practice.NotKept):
            sessions.update(tmp_path, "some_session", open=False)
    assert not store.log_path(tmp_path).exists() and not store.files(tmp_path)
    assert store.append(tmp_path, [{"id": "x", "source": "Ann", "task": "class"}])  # outside the practice: kept as always


def test_a_guest_opens_the_practice_session_its_field_and_its_picture(rounds_project):
    admin, _, _ = rounds_project
    guest = TestClient(admin.app)
    info = guest.get("/api/tutorial").json()
    assert info["practice"]["round2"] == FIELD and info["practice"]["credit"]  # the server's one tutorial, with its credit line
    detail = guest.get(f"/api/annotate/{FIELD}")
    assert detail.status_code == 200 and detail.json()["meta"]["fields"] == [info["practice"]["tile_id"]]  # the field is the session's own
    assert guest.get(f"/api/annotate/{FIELD}/fields/{TILE}").status_code == 200
    picture = guest.get(f"/api/annotate/{FIELD}/fields/{TILE}/image.jpg")
    assert picture.status_code == 200 and picture.headers["content-type"].startswith("image/")
    assert guest.get(f"/api/annotate/{CORES}").status_code == 200
    scoped = guest.get(f"/api/projects/{state.get_state().id}/tutorial").json()  # the page asks it of its own project
    assert scoped["practice"] == info["practice"]


def test_tutorials_are_kept_with_the_account_and_admins_sets_are_the_ones_they_opened(rounds_project):
    admin, (ann, ben), _ = rounds_project
    assert ann.get("/api/session/tutorials").json() == {"objects": False, "areas": False, "threads": False, "kept": True}
    assert ann.post("/api/session/tutorials", json={"kind": "areas", "done": True}).json()["areas"] is True
    assert ann.post("/api/tutorial", json={"done": True}).json() == {"done": True}
    assert ann.get("/api/session/tutorials").json() == {"objects": True, "areas": True, "threads": False, "kept": True}
    assert ann.get("/api/tutorial").json()["done"] is True and ben.get("/api/tutorial").json()["done"] is False
    guest = TestClient(admin.app)
    assert guest.post("/api/session/tutorials", json={"kind": "objects"}).json()["kept"] is False  # a guest's stays in the browser
    made = admin.post("/api/labels/sessions", json={"stain": "6E10", "round": 2, "open": True, "tile_ids": [TILE]}).json()["name"]
    assert made not in {s["name"] for s in admin.get("/api/session/sets").json()}
    assert admin.get(f"/api/annotate/{made}").status_code == 200  # an admin opens it: it joins their sets
    assert made in {s["name"] for s in admin.get("/api/session/sets").json()}
