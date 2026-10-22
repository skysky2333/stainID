"""A finished field opens to look at (read only, "Done by <name> on <date>", holding nothing); "Redo this field" lets
the person who finished it change their own labels, "Add an extra annotation" lets someone else annotate it again from
detection's output and their own labels only, added alongside the first person's; and "Skip for now" asks the server
for the next field."""
from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from test_annotation_rounds import OTHER, TILE, _png, _save, rounds_project  # noqa: E402,F401  (the fixture)
from test_separate_sessions import _mask  # noqa: E402

from stainid.api import state  # noqa: E402
from stainid.labels import decisions  # noqa: E402


def _session(admin) -> str:
    made = admin.post("/api/labels/sessions", json={"stain": "6E10", "round": 2, "open": True, "tile_ids": [TILE, OTHER]})
    assert made.status_code == 200, made.text
    return made.json()["name"]


def test_skip_for_now_asks_the_server_and_never_lands_on_a_finished_field(rounds_project):  # noqa: F811
    admin, (ann, ben), _ = rounds_project
    name = _session(admin)
    assert _save(ben, name, OTHER, [], submit=True).status_code == 200  # Ben finished the other field
    ben.post(f"/api/annotate/{name}/fields/{OTHER}/leave")
    assert ann.get(f"/api/annotate/{name}/fields/{TILE}").status_code == 200
    skipped = ann.get(f"/api/annotate/{name}/next", params={"skip": TILE}).json()
    assert skipped["tile_id"] is None  # TILE put aside, OTHER finished by Ben: nothing else waits for her
    assert ben.get(f"/api/annotate/{name}/fields/{TILE}").status_code == 200  # skipping freed TILE at once


def test_a_finished_field_opens_read_only_and_can_be_redone_or_annotated_again(rounds_project):  # noqa: F811
    admin, (ann, ben), rows = rounds_project
    _mask(state.get_state().project)
    name = _session(admin)
    field = ann.get(f"/api/annotate/{name}/fields/{TILE}").json()
    own = {(round(o["x"]), round(o["y"])): o["candidate_id"] for o in field["objects"] if o["own"]}
    painted, changed = own[(228, 228)], own[(288, 228)]
    saved = _save(ann, name, TILE, [{"task": "class", "candidate": changed, "value": "CAA"},
                                    {"task": "exclude", "pngs": {"4": _png((52, 62, 52, 62))}, "scale": 4}], submit=True)
    assert saved.status_code == 200, saved.text

    # looked at: read only, who finished it and when, nothing held (the others may look too)
    mine = ann.get(f"/api/annotate/{name}/fields/{TILE}").json()
    assert mine["read_only"] and mine["complete"] and [f["person"] for f in mine["finished"]] == ["Ann"] and mine["finished"][0]["time"] > 0
    theirs = ben.get(f"/api/annotate/{name}/fields/{TILE}").json()
    assert theirs["read_only"] and theirs["open_by"] == []
    ben.post(f"/api/annotate/{name}/live", json={"tile_id": TILE, "view": True})
    assert ann.get(f"/api/annotate/{name}/fields/{TILE}", params={"edit": True}).status_code == 200  # looking held nothing

    # Redo this field: Ann works on her own labels again, holding it
    redo = ann.get(f"/api/annotate/{name}/fields/{TILE}", params={"edit": True}).json()
    assert not redo["read_only"] and any(m.get("candidate") == changed and m["value"] == "CAA" for m in redo["mine"])
    taken = ben.get(f"/api/annotate/{name}/fields/{TILE}", params={"own": True})
    assert taken.status_code == 409 and "Ann" in taken.json()["detail"]  # in progress with someone: locked, as before
    ann.post(f"/api/annotate/{name}/fields/{TILE}/leave")

    # Add an extra annotation: Ben starts from detection's output and his own labels only
    extra = ben.get(f"/api/annotate/{name}/fields/{TILE}", params={"own": True}).json()
    assert not extra["read_only"] and extra["own"] and extra["checked"] == {} and extra["mine"] == []
    assert not next(o for o in extra["objects"] if o["candidate_id"] == painted)["excluded"]  # Ann's bad area is not his start
    assert next(o for o in redo["objects"] if o["candidate_id"] == painted)["excluded"]
    assert "4" not in ben.get(f"/api/annotate/{name}/fields/{TILE}/areas", params={"own": True}).json()["kinds"]
    assert ben.get(f"/api/annotate/{name}/fields/{TILE}/annotated", params={"own": True}).json()["regions"] == []
    assert ben.post(f"/api/annotate/{name}/live", json={"tile_id": TILE, "own": True}).json().get("checked") is None
    done = ben.post(f"/api/annotate/{name}/tiles/{TILE}/save", json={"records": [], "submit": True, "own": True})
    assert done.status_code == 200, done.text
    confirmed = {r["candidate"] for r in done.json()["saved"] if r["task"] == "class"}
    assert painted in confirmed  # what he was shown (not faded by Ann's area) he checked

    # both are done; Ann's labels stay as they were, his are added alongside
    states = {f["tile_id"]: f for f in admin.get(f"/api/labels/sessions/{name}/fields").json()}
    assert states[TILE]["done_by"] == ["Ann", "Ben"]
    again = ann.get(f"/api/annotate/{name}/fields/{TILE}").json()
    assert [f["person"] for f in again["finished"]] == ["Ann", "Ben"]
    decisions.forget()
    decided = decisions.session_people(state.get_state().project, "6E10", state.get_state().tiles, name)
    calls = {(r.source, r.candidate): r.identity for r in decided.rows.itertuples()}
    assert calls[("Ann", changed)] == "CAA" and calls[("Ben", changed)] == "plaque"
