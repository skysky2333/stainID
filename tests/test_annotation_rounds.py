"""The two annotation rounds over the v2 candidate store: the store as the contract writes it, the most informative
candidates first, one shared decision store keyed by candidate id (old labels matched by position), and two people at
once: what one decides leaves the other's page, fields are handed out once, and a field finished in round 2 counts as
decided for round 1."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from stainid.labels import candidate_store, decisions, rounds

TILE = "LIP-1_B-2_6E10_S01"  # x_px = y_px = 300 in a 1200 px core image, field margin 128 px: field origin (172, 172)
OTHER = "LIP-1_B-2_6E10_S02"  # a second field of the same core, right of the first (x_px 812)


def _store_row(x: float, y: float, p: float, accepted: bool, **more) -> dict:
    return {"candidate_id": candidate_store.candidate_id("6E10", "LIP-1_B-2", x, y), "x_core": x, "y_core": y, "tile_id": TILE, "area_um2": 300.0,
            "radius_px": 10.0, "sources": "dab@1", "proposal_threshold": 1.0, "outline": f"{x - 8:.0f} {y:.0f} {x:.0f} {y - 8:.0f} {x + 8:.0f} {y:.0f}",
            "sam_outline": "", "sam_status": "", "probability": p, "model_id": "m@1", "decision": "accepted" if accepted else "rejected",
            "reason": "accepted" if accepted else "rejected_by_model", "class": "plaque" if accepted else "not a plaque",
            "subtype": "compact" if accepted else "", "mean_dab_od": 0.3, "m_context_dab_mean": 0.1 + p, **more}


def test_the_store_is_read_as_the_contract_writes_it(tmp_path):
    rings = candidate_store._polygon("0 0 10 0 10 10 0 10;50 50 51 50 51 51")
    assert rings == [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]  # the largest ring of several
    assert candidate_store._polygon("[[1, 2], [3, 4], [5, 7]]") == [[1.0, 2.0], [3.0, 4.0], [5.0, 7.0]]
    assert candidate_store._polygon("") is None and candidate_store._polygon(float("nan")) is None
    masks = candidate_store._polygons("0 0 4 0 4 4|0 0 8 0 8 8|0 0 16 0 16 16")
    assert [m[1][0] for m in masks] == [4.0, 8.0, 16.0]
    row = {"sam_masks": masks, "sam_scores": [0.7, 0.95, 0.6], "sam_default": float("nan"), "sam_outline": None, "outline": None}
    assert candidate_store.masks(row)[1] == 1  # no default stored: the best score
    assert candidate_store.ai_outline({**row, "sam_default": 2}) == masks[2]
    assert candidate_store.masks({"sam_masks": None, "sam_outline": None, "outline": [[0, 0], [1, 0], [1, 1]]}) == ([[[0, 0], [1, 0], [1, 1]]], 0)


def test_most_informative_first():
    frame = pd.DataFrame([_store_row(100 + 10 * i, 100, p, p >= 0.5) for i, p in enumerate([0.95, 0.52, 0.1, 0.3, 0.75, 0.45, 0.62])])
    assert rounds.decision_threshold(frame) == pytest.approx(0.52)
    ordered = rounds.rank(frame)
    assert list(ordered.probability) == [0.52, 0.45, 0.62, 0.3, 0.1, 0.75, 0.95]  # near the threshold, rejected strong, accepted weak
    assert list(ordered.tier) == [0, 0, 0, 1, 1, 2, 2]
    assert len(candidate_store.at_level(frame.assign(proposal_threshold=[0.5, 1, 1.5, np.nan, 0.75, 1, 1]), 1.0)) == 5


def test_holds_and_labels_needed_decide_who_is_handed_what():
    frame = pd.DataFrame({"candidate_id": list("abcde"), "order": range(5), "tile_id": "T", "x": 1.0, "y": 1.0})
    state = decisions._state_of(pd.DataFrame([
        {"candidate": "a", "source": "Ben", "identity": "plaque", "subtype": "", "role": "positive", "time": 1.0, "session": "s", "confirmed": True},
        {"candidate": "b", "source": "Cy", "identity": "unsure", "subtype": "", "role": "skip", "time": 2.0, "session": "s", "confirmed": False}]))
    one = {"labels_needed": 1}
    chosen, left = rounds.serve(one, frame, state, "Ann", {"c": {"Ben"}}, 5)
    assert list(chosen.candidate_id) == ["b", "d", "e"] and left == 3  # a decided, c held by Ben; b only "unsure"
    chosen, _ = rounds.serve(one, frame, state, "Cy", {}, 5)
    assert "b" not in set(chosen.candidate_id)  # Cy answered b
    chosen, _ = rounds.serve({"labels_needed": 2}, frame, state, "Ann", {"c": {"Ben"}}, 5)
    assert list(chosen.candidate_id) == ["a", "b", "c", "d", "e"]  # two people needed: a and c stay open for a second one
    assert rounds.gone(["a", "b", "c", "d"], state, "Ann", 1, {"c": {"Ben"}}) == ["a", "c"]


pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from stainid.api import state  # noqa: E402
from stainid.api.app import create_app  # noqa: E402

CANDIDATES = [(400.0, 400.0, 0.9, True), (460.0, 400.0, 0.55, True), (520.0, 400.0, 0.45, False), (400.0, 480.0, 0.2, False),
              (460.0, 480.0, 0.97, True), (520.0, 480.0, 0.05, False), (900.0, 400.0, 0.6, True)]


@pytest.fixture()
def rounds_project(synthetic_project, monkeypatch):
    """The synthetic project with a second 6E10 field and a candidate store for its core (one candidate on the second
    field); an admin and two people."""
    from stainid.api.routes import annotate

    monkeypatch.chdir(synthetic_project)
    tiles = pd.read_csv(synthetic_project / "data/analysis/cohort_whole_core_tiles.csv")
    extra = tiles[tiles.tile_id == TILE].assign(tile_id=OTHER, x_px=812, selection_order=2)
    pd.concat([tiles, extra]).to_csv(synthetic_project / "data/analysis/cohort_whole_core_tiles.csv", index=False)
    rows = [_store_row(x, y, p, a) for x, y, p, a in CANDIDATES]
    rows[-1]["tile_id"] = OTHER
    monkeypatch.setattr(annotate, "prepare_job", lambda *args, **kwargs: None)
    monkeypatch.setattr(annotate, "measure_added_soon", lambda *args, **kwargs: None)
    state.reset()
    decisions.forget()
    admin = TestClient(create_app(), client=("127.0.0.1", 50000))
    assert admin.post("/api/session/admin", json={"password": "admin"}).status_code == 200
    candidate_store.write_core(state.get_state().project, "6E10", "LIP-1_B-2", pd.DataFrame(rows))
    people = []
    for name in ("Ann", "Ben"):
        person = TestClient(admin.app)
        person.post("/api/session/annotator", json={"name": name})
        people.append(person)
    yield admin, people, rows
    state.reset()
    decisions.forget()


def _ids(rows) -> list[str]:
    return [r["candidate_id"] for r in rows]


def test_round_one_two_people_at_once(rounds_project):
    admin, (ann, ben), rows = rounds_project
    scope = admin.get("/api/labels/rounds/scope?stain=6E10").json()
    assert scope["cores_ready"] == 1 and scope["fields_ready"] == 2
    made = admin.post("/api/labels/sessions", json={"stain": "6E10", "round": 1, "n_objects": 0, "title": "Check", "open": True})
    assert made.status_code == 200, made.text
    meta = made.json()
    assert meta["kind"] == "queue" and meta["objects"] == 7 and meta["source"] == "candidates"
    name = meta["name"]
    detail = ann.get(f"/api/annotate/{name}").json()["meta"]
    assert detail["round"] == 1 and detail["progress"] == {"total": 7, "decided": 0, "people": []} and detail["remove_class"] == "not a plaque"

    first = ann.get(f"/api/annotate/{name}/next?count=3").json()
    assert [o["ai_probability"] for o in first["objects"]] == [0.55, 0.6, 0.45]  # nearest the threshold (0.55) first
    item = first["objects"][0]
    assert item["ai_value"] == "plaque" and item["ai_subtype"] == "compact" and len(item["masks"]) == 1 and len(item["masks"][0]) == 3
    others = ben.get(f"/api/annotate/{name}/next?count=3").json()["objects"]
    assert not set(_ids(first["objects"])) & set(_ids(others))  # held for Ann

    saved = ann.post(f"/api/annotate/{name}/labels", json={"records": [{"task": "class", "candidate": item["candidate_id"], "keep": True, "mask": 0}]})
    assert saved.status_code == 200, saved.text
    decision, outline = saved.json()
    assert decision["value"] == "plaque" and decision["subtype"] == "compact" and decision["confirmed"] and decision["candidate"] == item["candidate_id"]
    assert decision["tile_id"] == TILE and decision["x"] == pytest.approx(460 - 172) and outline["task"] == "outline" and outline["confirmed"]
    second = first["objects"][2]
    ann.post(f"/api/annotate/{name}/labels", json={"records": [{"task": "class", "candidate": second["candidate_id"], "value": "CAA"}]})

    # Ben decides one of the objects on Ann's page from round 2: it leaves her page at her next look
    fix = admin.post("/api/labels/sessions", json={"stain": "6E10", "round": 2, "tile_ids": [TILE, OTHER], "open": True}).json()
    third = first["objects"][1]  # on the other field
    flipped = ben.post(f"/api/annotate/{fix['name']}/labels", json={"records": [{"task": "class", "candidate": third["candidate_id"], "value": "not a plaque"}]})
    assert flipped.status_code == 200, flipped.text
    live = ann.post(f"/api/annotate/{name}/live", json={"keys": _ids(first["objects"])}).json()
    assert set(live["gone"]) == set(_ids(first["objects"]))  # two answered by her, one decided by Ben
    assert live["progress"]["decided"] == 3 and {p["person"] for p in live["progress"]["people"]} == {"Ann", "Ben"}
    assert not set(_ids(ben.get(f"/api/annotate/{name}/next?count=10").json()["objects"])) & set(_ids(first["objects"]))

    edited = ann.post(f"/api/annotate/{name}/labels", json={"records": [  # a precise edit of an outline, then the answer without a mask
        {"task": "outline", "candidate": second["candidate_id"], "value": "edited", "points": [[1, 2], [30, 2], [30, 40]]}]})
    assert edited.status_code == 200 and edited.json()[0]["x"] == pytest.approx(520 - 172)
    removed = ann.post(f"/api/annotate/{name}/labels", json={"records": [{"task": "class", "candidate": first["objects"][0]["candidate_id"],
                                                                           "value": "not a plaque", "mask": 0}]}).json()
    assert len(removed) == 1  # "not an object" keeps no mask
    assert ann.post(f"/api/annotate/{name}/labels", json={"records": [{"task": "tag", "tile_id": TILE, "value": "fold or tear"}]}).status_code == 400
    undo = ann.post(f"/api/annotate/{name}/labels", json={"records": [{"task": "class", "candidate": second["candidate_id"], "deleted": True}]})
    assert undo.status_code == 200
    assert ann.post(f"/api/annotate/{name}/live", json={"keys": []}).json()["progress"]["decided"] == 2
    assert admin.get("/api/labels/overview").json()["sessions"][0]["round_progress"]["total"] in (7, 2)


def test_round_two_fields_are_handed_out_once_and_finishing_decides_their_candidates(rounds_project):
    admin, (ann, ben), rows = rounds_project
    meta = admin.post("/api/labels/sessions", json={"stain": "6E10", "round": 2, "n_fields": 2, "open": True}).json()
    name = meta["name"]
    assert sorted(meta["fields"]) == sorted([TILE, OTHER])
    a = ann.get(f"/api/annotate/{name}/next").json()["tile_id"]
    b = ben.get(f"/api/annotate/{name}/next").json()
    assert a and b["tile_id"] and a != b["tile_id"]  # field reservations: one field each
    assert ann.get(f"/api/annotate/{name}/next").json()["tile_id"] == a  # still hers
    third = TestClient(admin.app)
    third.post("/api/session/annotator", json={"name": "Cy"})
    assert third.get(f"/api/annotate/{name}/next").json() == {"tile_id": None, "waiting": 0, "taken": 2}

    field = ann.get(f"/api/annotate/{name}/fields/{TILE}").json()
    own = [o for o in field["objects"] if o["own"]]
    assert len(own) == 6 and all(o["masks"] for o in own) and field["checked"] == {}
    target = own[0]
    ben.post(f"/api/annotate/{name}/labels", json={"records": [{"task": "class", "candidate": target["candidate_id"], "value": "CAA"}]})
    live = ann.post(f"/api/annotate/{name}/live", json={"tile_id": TILE}).json()
    assert live["checked"][target["candidate_id"]]["by"] == ["Ben"] and live["checked"][target["candidate_id"]]["identity"] == "CAA"
    assert live["open_by"] == ["Ben"]  # he just changed a call here

    added = ann.post(f"/api/annotate/{name}/labels", json={"records": [
        {"task": "class", "tile_id": TILE, "x": 50, "y": 60, "value": "plaque"},
        {"task": "outline", "tile_id": TILE, "x": 50, "y": 60, "points": [[45, 55], [55, 55], [55, 65]]}]})
    assert added.status_code == 200, added.text
    done = ann.post(f"/api/annotate/{name}/fields/{TILE}/finish", json={}).json()
    confirmed = [r for r in done if r["task"] == "class"]
    assert len(confirmed) == 5 and all(r["confirmed"] and r["candidate"] for r in confirmed)  # Ben's one is his
    outlines = [r for r in done if r["task"] == "outline"]
    assert len(outlines) == 2  # the AI’s outlines of the objects that stay plaques (Ben made one CAA)
    check = admin.post("/api/labels/sessions", json={"stain": "6E10", "round": 1, "n_objects": 0, "tile_ids": [TILE], "open": True})
    assert check.status_code == 400  # every candidate of the field is decided: nothing left to check
    fields = admin.get(f"/api/labels/sessions/{name}/fields").json()
    assert {f["tile_id"]: f["state"] for f in fields}[TILE] == "done"


def test_old_labels_count_by_position_and_training_reads_the_store(rounds_project):
    from stainid.labels import store
    from stainid.sampling.tiles import known_tiles

    admin, _, rows = rounds_project
    project = state.get_state().project
    store.append(project.output("reviews"), [{"id": "class:x", "source": "import: old", "stain": "6E10", "task": "class", "tile_id": TILE,
                                               "x": 400 - 172 + 1.5, "y": 400 - 172 - 1.0, "value": "compact plaque"}])
    found = decisions.state(project, "6E10", known_tiles(project))
    assert found.by(rows[0]["candidate_id"]) == ["import: old"] and found.calls[rows[0]["candidate_id"]] == ("plaque", "compact")
    made = admin.post("/api/labels/sessions", json={"stain": "6E10", "round": 1, "n_objects": 0, "open": True}).json()
    assert made["objects"] == 6  # the object labelled before candidate ids existed is not asked again
    six = next(s for s in admin.get("/api/labels/overview").json()["stains"] if s["stain"] == "6E10")
    assert six["detection"]["found"] == 1 and six["detection"]["never_proposed"] == 0  # a store candidate counts as proposed

    objects = pd.DataFrame([{"tile_id": TILE, "x": 460 - 172, "y": 400 - 172}])
    features, names = candidate_store.training_rows(project, "6E10", objects, known_tiles(project), None)
    assert names == ["context_dab_mean"] and features[0]["context_dab_mean"] == pytest.approx(0.65) and features[0]["tma"] == "1"


def test_a_store_can_be_made_from_the_pool_for_trying_the_rounds(rounds_project):
    from stainid.labels import pool
    from stainid.sampling.tiles import known_tiles

    project = state.get_state().project
    folder = pool.folder(project, "6E10")
    folder.mkdir(parents=True)
    pd.DataFrame({"x": [200.0, 250.0], "y": [200.0, 220.0], "radius_px": 9.0, "ai_value": ["compact plaque", "not a plaque"],
                  "ai_probability": [0.8, 0.2], "tma": "1", "complete": True}).to_csv(folder / f"{TILE}.csv", index=False)
    written = candidate_store.from_pool(project, "6E10", known_tiles(project))
    assert written == {"LIP-1_B-2": 2}
    frame = candidate_store.read_core(project, "6E10", "LIP-1_B-2")
    assert list(frame.decision) == ["accepted", "rejected"] and frame["class"].iloc[0] == "plaque" and len(frame.sam_masks.iloc[0]) == 3
    assert json.loads((candidate_store.folder(project, "6E10") / ".from_pool.json").read_text())["cores"] == written


def test_the_tutorial_practises_on_a_real_field_and_keeps_nothing(rounds_project):
    from stainid.labels import store

    _, (ann, _), _ = rounds_project
    first = ann.get("/api/session/tutorial").json()
    assert first["done"] is False and first["practice"]["tile_id"] == TILE and first["practice"]["round2"] == "practice-2-6E10"
    detail = ann.get("/api/annotate/practice-2-6E10").json()
    assert detail["meta"]["round"] == 2 and detail["fields"][0]["tile_id"] == TILE
    field = ann.get(f"/api/annotate/practice-2-6E10/fields/{TILE}").json()
    target = next(o for o in field["objects"] if o["own"])
    saved = ann.post("/api/annotate/practice-2-6E10/labels", json={"records": [{"task": "class", "candidate": target["candidate_id"], "value": "CAA"}]})
    assert saved.status_code == 200 and saved.json()[0]["value"] == "CAA"
    assert ann.post(f"/api/annotate/practice-2-6E10/fields/{TILE}/finish", json={}).status_code == 200
    objects = ann.get("/api/annotate/practice-1-6E10/next?count=3").json()["objects"]
    assert len(objects) == 3
    ann.post("/api/annotate/practice-1-6E10/labels", json={"records": [{"task": "class", "candidate": objects[0]["candidate_id"], "keep": True}]})
    assert store.current(state.get_state().project.output("reviews")).empty  # nothing of the practice is kept
    assert ann.get("/api/annotate/practice-1-6E10/next?count=3").json()["objects"][0]["key"] == objects[0]["key"]  # nor held or decided
    assert ann.post("/api/session/tutorial", json={"done": True}).json() == {"done": True}
    assert ann.get("/api/session/tutorial").json()["done"] is True
    assert ann.get("/api/annotate").json() == []  # practice sessions are not listed
