"""The two annotation rounds over the v2 candidate store: the store as the contract writes it, the most informative
candidates first, one shared decision store keyed by candidate id (old labels matched by position), and the tile as the
unit people claim: a tile someone is on, or that enough people finished, is handed to nobody else; nothing reaches the
label log until a person saves a tile, and then all of it at once ("Save" keeps the tile, "Save and continue" finishes
it)."""
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


def test_fields_are_drawn_at_random_from_the_remaining_ones():
    fields = [f"T{i}" for i in range(50)]
    left = rounds.remaining_fields(fields, finished={"T0", "T1"}, assigned={"T2"})
    assert left == fields[3:]  # not annotated yet and not in another open session
    assert sorted(rounds.draw_fields(left, rounds.ALL_FIELDS, seed=7)) == sorted(left)  # -1: every remaining field
    some = rounds.draw_fields(left, 10, seed=7)
    assert len(some) == 10 == len(set(some)) and set(some) <= set(left)  # n distinct remaining fields
    assert rounds.draw_fields(left, 10, seed=7) == some and rounds.draw_fields(left, 10, seed=8) != some  # the seed reproduces the draw
    assert sorted(rounds.draw_fields(left, 500, seed=1)) == sorted(left)  # more than remain: all of them
    first = [rounds.draw_fields(left, 1, seed=s)[0] for s in range(2000)]  # uniform: every field comes up
    assert set(first) == set(left)


def test_a_tile_is_handed_to_one_person_at_a_time_until_enough_finished_it():
    tiles = ["t1", "t2", "t3"]
    one = {"t1": {"Ben"}}  # Ben finished t1
    assert rounds.next_tile(tiles, one, "Ann", {"t2": {"Cy"}}, 1) == ("t3", 1)  # t1 done, t2 on Cy's screen
    assert rounds.next_tile(tiles, one, "Ann", {"t2": {"Cy"}, "t3": {"Ann"}}, 1) == ("t3", 0)  # her tile in progress first
    assert rounds.next_tile(tiles, one, "Ann", {"t2": {"Cy"}, "t3": {"Ann"}}, 1, skip={"t3"}) == (None, 1)  # put aside: nothing else free
    assert rounds.next_tile(tiles, one, "Ann", {}, 2) == ("t1", 0)  # two people needed: t1 still needs one more
    assert rounds.next_tile(tiles, one, "Ben", {}, 2) == ("t2", 0)  # but not Ben again
    assert rounds.next_tile(tiles, {}, "Ann", {}, 1, prefer={"t3"}) == ("t3", 0)  # pictures made first
    assert rounds.next_tile(tiles, {}, "Ann", {}, 1, ready={"t2"}) == ("t2", 0)
    assert rounds.progress(tiles, one, {"t2": {"Cy"}}, 1) == {"total": 3, "decided": 1, "open_now": 1, "left": 1,
                                                             "people": [{"person": "Ben", "decided": 1}]}
    assert rounds.progress(tiles, one, {}, 2)["open_now"] == 1  # finished by one of two: in progress


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


def _person(admin, name):
    person = TestClient(admin.app)
    person.post("/api/session/annotator", json={"name": name})
    return person


def _log_lines() -> int:
    from stainid.labels import store

    path = store.log_path(state.get_state().project.output("reviews"))
    return len(path.read_text().splitlines()) if path.exists() else 0


def _save(client, name, tile, records, submit=False):
    return client.post(f"/api/annotate/{name}/tiles/{tile}/save", json={"records": records, "submit": submit})


def test_round_one_hands_out_tiles_and_saves_a_tile_at_once(rounds_project):
    admin, (ann, ben), rows = rounds_project
    scope = admin.get("/api/labels/rounds/scope?stain=6E10").json()
    assert scope["cores_ready"] == 1 and scope["fields_ready"] == 2
    made = admin.post("/api/labels/sessions", json={"stain": "6E10", "round": 1, "n_objects": 0, "title": "Check", "open": True})
    assert made.status_code == 200, made.text
    meta = made.json()
    assert meta["kind"] == "queue" and meta["objects"] == 7 and meta["source"] == "candidates"
    name = meta["name"]
    detail = ann.get(f"/api/annotate/{name}").json()["meta"]
    assert detail["round"] == 1 and detail["remove_class"] == "not a plaque"
    assert detail["progress"] == {"total": 2, "decided": 0, "open_now": 0, "left": 2, "people": []}  # counted in tiles

    first = ann.get(f"/api/annotate/{name}/next").json()
    assert first["tile_id"] == TILE and first["tiles_left"] == 2  # the tile of the most informative candidate first
    assert [o["ai_probability"] for o in first["objects"]] == [0.55, 0.45, 0.2, 0.05, 0.9, 0.97]  # all of it, most informative first
    item = first["objects"][0]
    assert item["ai_value"] == "plaque" and item["ai_subtype"] == "compact" and len(item["masks"]) == 1 and item["mine"] is None
    other = ben.get(f"/api/annotate/{name}/next").json()
    assert other["tile_id"] == OTHER and len(other["objects"]) == 1  # the tile Ann is on is skipped
    cy = _person(admin, "Cy")
    assert cy.get(f"/api/annotate/{name}/next").json()["tile_id"] is None  # both tiles are being edited
    live = ann.post(f"/api/annotate/{name}/live", json={"tile_id": TILE}).json()
    assert live["progress"]["open_now"] == 2 and live["open_by"] == [] and live["done_by"] == []
    assert _log_lines() == 0  # opening, being handed and the heartbeat save nothing

    ids = _ids(first["objects"])
    saved = _save(ann, name, TILE, [{"task": "class", "candidate": ids[0], "keep": True, "mask": 0},
                                     {"task": "class", "candidate": ids[1], "value": "CAA"},
                                     {"task": "outline", "candidate": ids[1], "value": "edited", "points": [[1, 2], [30, 2], [30, 40]]}])
    assert saved.status_code == 200, saved.text
    decision, outline, caa, edited = saved.json()["saved"]
    assert decision["value"] == "plaque" and decision["subtype"] == "compact" and decision["confirmed"] and decision["candidate"] == ids[0]
    assert decision["tile_id"] == TILE and decision["x"] == pytest.approx(460 - 172) and outline["task"] == "outline" and outline["confirmed"]
    assert edited["x"] == pytest.approx(520 - 172)
    assert saved.json()["progress"]["decided"] == 0  # "Save" keeps the tile open: it is not finished

    again = ann.get(f"/api/annotate/{name}/next").json()  # a reload: her tile, with what she saved
    assert again["tile_id"] == TILE
    mine = {o["candidate_id"]: o["mine"] for o in again["objects"]}
    assert mine[ids[0]]["keep"] and mine[ids[0]]["mask"] == 0 and mine[ids[1]]["identity"] == "CAA" and len(mine[ids[1]]["outline"]) == 3
    assert mine[ids[2]] is None

    before = _log_lines()
    wrong = _save(ann, name, TILE, [{"task": "class", "candidate": ids[2], "value": "not a plaque"}, {"task": "class", "candidate": ids[3], "value": "no such class"}])
    assert wrong.status_code == 400 and _log_lines() == before
    elsewhere = _save(ann, name, TILE, [{"task": "class", "candidate": other["objects"][0]["candidate_id"], "value": "CAA"}])
    assert elsewhere.status_code == 400 and _log_lines() == before
    assert ann.post(f"/api/annotate/{name}/labels", json={"records": [{"task": "class", "candidate": ids[2], "value": "CAA"}]}).status_code == 400

    done = _save(ann, name, TILE, [{"task": "class", "candidate": ids[2], "value": "not a plaque"}], submit=True)  # the rest left as they are
    assert done.status_code == 200, done.text
    kept = [r for r in done.json()["saved"] if r["task"] == "class" and r.get("confirmed")]
    assert {r["candidate"] for r in kept} == set(ids[3:])  # what she did not answer keeps detection's call, as her answer
    assert done.json()["saved"][-1]["task"] == "done" and done.json()["saved"][-1]["value"] == "calls"
    assert done.json()["progress"] == {"total": 2, "decided": 1, "open_now": 1, "left": 0, "people": [{"person": "Ann", "decided": 1}]}
    assert ann.get(f"/api/annotate/{name}/next").json()["tile_id"] is None  # her tile is done, Ben has the other
    assert cy.get(f"/api/annotate/{name}/next").json()["tile_id"] is None  # a finished tile is skipped too

    put_aside = ben.get(f"/api/annotate/{name}/next?skip={OTHER}").json()
    assert put_aside["tile_id"] is None  # nothing else for Ben; the tile he put aside is free again
    assert cy.get(f"/api/annotate/{name}/next").json()["tile_id"] == OTHER

    fix = admin.post("/api/labels/sessions", json={"stain": "6E10", "round": 2, "tile_ids": [TILE, OTHER], "open": True}).json()
    assert ann.get(f"/api/annotate/{fix['name']}/next").json()["tile_id"] == TILE  # finished in round 1, still to do in round 2
    overview = admin.get("/api/labels/overview").json()
    assert next(s for s in overview["sessions"] if s["name"] == name)["round_progress"]["decided"] == 1


def test_round_two_fields_are_claimed_and_saved_as_a_whole(rounds_project):
    admin, (ann, ben), rows = rounds_project
    meta = admin.post("/api/labels/sessions", json={"stain": "6E10", "round": 2, "n_fields": 2, "open": True}).json()
    name = meta["name"]
    assert sorted(meta["fields"]) == sorted([TILE, OTHER])
    a = ann.get(f"/api/annotate/{name}/next").json()["tile_id"]
    b = ben.get(f"/api/annotate/{name}/next").json()
    assert a and b["tile_id"] and a != b["tile_id"]  # one field each
    assert ann.get(f"/api/annotate/{name}/next").json()["tile_id"] == a  # still hers
    cy = _person(admin, "Cy")
    assert cy.get(f"/api/annotate/{name}/next").json() == {"tile_id": None, "waiting": 0, "taken": 2}

    field = ann.get(f"/api/annotate/{name}/fields/{TILE}").json()
    own = [o for o in field["objects"] if o["own"]]
    assert len(own) == 6 and all(o["masks"] for o in own) and field["checked"] == {} and field["mine"] == []
    target = own[0]
    flipped = _save(ben, name, TILE, [{"task": "class", "candidate": target["candidate_id"], "value": "CAA"}])  # Ben opened it by its link
    assert flipped.status_code == 200, flipped.text
    live = ann.post(f"/api/annotate/{name}/live", json={"tile_id": TILE}).json()
    assert live["checked"][target["candidate_id"]]["by"] == ["Ben"] and live["checked"][target["candidate_id"]]["identity"] == "CAA"
    assert live["open_by"] == ["Ben"]  # he saved here: the field is his too now

    before = _log_lines()
    saved = _save(ann, name, TILE, [{"task": "class", "x": 50, "y": 60, "value": "plaque"},
                                     {"task": "outline", "x": 50, "y": 60, "points": [[45, 55], [55, 55], [55, 65]]},
                                     {"task": "tag", "value": "fold or tear"}])
    assert saved.status_code == 200, saved.text
    assert _log_lines() == before + 3 and all(r["tile_id"] == TILE for r in saved.json()["saved"])
    states = {f["tile_id"]: f["state"] for f in admin.get(f"/api/labels/sessions/{name}/fields").json()}
    assert states[TILE] == "in_progress"  # "Save" is not finishing

    done = _save(ann, name, TILE, [{"task": "tag", "value": "fold or tear", "deleted": True}], submit=True).json()["saved"]
    confirmed = [r for r in done if r["task"] == "class"]
    assert len(confirmed) == 6 and all(r["confirmed"] and r["candidate"] for r in confirmed)  # the whole field is hers: also the one Ben decided
    assert len([r for r in done if r["task"] == "outline"]) == 3  # detection's outlines of the objects that stay plaques in her answers
    assert done[-1]["id"] == f"done:{TILE}" and done[0]["deleted"]
    states = {f["tile_id"]: f["state"] for f in admin.get(f"/api/labels/sessions/{name}/fields").json()}
    assert states[TILE] == "done"
    assert cy.get(f"/api/annotate/{name}/next").json()["tile_id"] is None  # the finished field is skipped, the other is Ben's
    check = admin.post("/api/labels/sessions", json={"stain": "6E10", "round": 1, "n_objects": 0, "tile_ids": [TILE], "open": True})
    assert check.status_code == 400  # every candidate of the field is decided: nothing left to check
    assert ann.post(f"/api/annotate/{name}/fields/{OTHER}/finish", json={}).status_code == 200  # the older finish: a save with nothing changed
    assert ann.post(f"/api/annotate/{name}/live", json={}).json()["progress"]["decided"] == 2


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
    saved = _save(ann, "practice-2-6E10", TILE, [{"task": "class", "candidate": target["candidate_id"], "value": "CAA"}])
    assert saved.status_code == 200 and saved.json()["saved"][0]["value"] == "CAA"
    finished = _save(ann, "practice-2-6E10", TILE, [], submit=True)
    assert finished.status_code == 200 and finished.json()["saved"][-1]["task"] == "done"
    tile = ann.get("/api/annotate/practice-1-6E10/next").json()
    assert tile["tile_id"] == TILE and len(tile["objects"]) == 6
    first_id = tile["objects"][0]["candidate_id"]
    assert _save(ann, "practice-1-6E10", TILE, [{"task": "class", "candidate": first_id, "keep": True}], submit=True).status_code == 200
    assert store.current(state.get_state().project.output("reviews")).empty  # nothing of the practice is kept
    again = ann.get("/api/annotate/practice-1-6E10/next").json()
    assert again["tile_id"] == TILE and again["objects"][0]["mine"] is None  # nor held or finished
    assert ann.post("/api/session/tutorial", json={"done": True}).json() == {"done": True}
    assert ann.get("/api/session/tutorial").json()["done"] is True
    assert ann.get("/api/annotate").json() == []  # practice sessions are not listed


def test_finishing_a_field_records_every_candidate_as_checked_by_the_person(rounds_project):
    admin, (ann, ben), _ = rounds_project
    rows = [_store_row(320.0 + 14 * (i % 10), 320.0 + 14 * (i // 10), 0.3 + 0.01 * i, i % 3 != 0) for i in range(50)]
    candidate_store.write_core(state.get_state().project, "6E10", "LIP-1_B-2", pd.DataFrame(rows))
    decisions.forget()
    name = admin.post("/api/labels/sessions", json={"stain": "6E10", "round": 2, "open": True, "tile_ids": [TILE]}).json()["name"]
    ann.get(f"/api/annotate/{name}/fields/{TILE}")
    ben_first = rows[0]["candidate_id"]  # Ben decided one already: Ann's finished field still gives her own answer on it
    assert _save(ben, name, TILE, [{"task": "class", "candidate": ben_first, "value": "CAA"}]).status_code == 200
    edits = [{"task": "class", "candidate": rows[1]["candidate_id"], "value": "CAA"}, {"task": "class", "candidate": rows[2]["candidate_id"], "value": "not a plaque"}]
    assert _save(ann, name, TILE, edits, submit=True).status_code == 200  # 2 edits of 50, then Save and continue
    decided = decisions.state(state.get_state().project, "6E10", state.get_state().tiles)
    hers = decided.rows[decided.rows.source == "Ann"]
    assert hers.candidate.nunique() == 50  # the whole field is checked by her
    calls = {r["candidate_id"]: ("plaque" if r["decision"] == "accepted" else "not a plaque") for r in rows}
    same = [c for c, identity in zip(hers.candidate, hers.identity) if identity == calls[c]]
    assert len(same) == 48 and hers.confirmed.sum() == 48  # her 48 untouched answers are detection's calls


def _png(*boxes):
    """A painted mask at a quarter of the field picture (200 × 200), painted in these (row, row, column, column) boxes."""
    import base64

    import cv2

    painted = np.zeros((200, 200, 4), np.uint8)
    for r0, r1, c0, c1 in boxes:
        painted[r0:r1, c0:c1] = 255
    return "data:image/png;base64," + base64.b64encode(cv2.imencode(".png", painted)[1].tobytes()).decode()


def _plain_mask(project):
    """A stage-2 mask of the 6E10 core: analysed tissue everywhere, a fold over core rows 360-439 (cells 45-54)."""
    import cv2

    from stainid import exclusions
    from stainid.v2.proposals import BITS

    bits = np.full((150, 150), BITS["tile"] | BITS["tissue"], np.uint8)
    bits[45:55, :] |= BITS["fold"]
    target = exclusions.mask_path(project, "LIP-1_B-2", "6E10")
    target.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(target), bits)


def test_brush_edits_are_people_s_areas_that_scoring_follows_and_the_views_draw_apart(rounds_project):
    """Each Save of the brush is one edit of people's areas: painting keeps the objects there out of everyone's scoring
    and training, a later clear (also over detection's fold) counts them again; the analysis' layer (detection's areas,
    drawn as "Excluded from analysis" in both views, the same pixels) never changes, and people's areas are drawn apart."""
    import cv2

    from stainid import exclusions

    admin, (ann, ben), _ = rounds_project
    project = state.get_state().project
    _plain_mask(project)
    name = admin.post("/api/labels/sessions", json={"stain": "6E10", "round": 2, "open": True, "tile_ids": [TILE]}).json()["name"]
    field = ann.get(f"/api/annotate/{name}/fields/{TILE}").json()
    own = {(round(o["x"]), round(o["y"])): o["candidate_id"] for o in field["objects"] if o["own"]}
    painted_on, folded = own[(228, 228)], own[(288, 228)]  # (400, 400) and (460, 400) on the core, both in the fold rows
    assert _save(ben, name, TILE, [{"task": "class", "candidate": c, "value": "CAA"} for c in (painted_on, folded)]).status_code == 200

    def scored():
        decisions.forget()
        rows = decisions.state(project, "6E10", state.get_state().tiles).rows
        return set(rows.candidate)

    assert painted_on not in scored() and folded not in scored()  # detection's fold: out of scoring from the start
    # Ann clears the fold over both, then paints "out of focus" over the first: two edits, the latest wins
    clear = {"1": "", "2": "", "3": "", "4": "", "include": _png((50, 65, 50, 80))}
    assert _save(ann, name, TILE, [{"task": "exclude", "pngs": clear, "scale": 4}]).status_code == 200
    assert {painted_on, folded} <= scored()
    paint = {"4": _png((52, 62, 52, 62))}
    assert _save(ann, name, TILE, [{"task": "exclude", "pngs": paint, "scale": 4}]).status_code == 200
    now = scored()
    assert painted_on not in now and folded in now  # Ben's answer on it is left out too: the area is excluded for everyone
    edits = exclusions.edits(project, "LIP-1_B-2", "6E10")
    assert [(e.source, sorted(set(e.kinds))) for e in edits] == [("Ann", ["include"]), ("Ann", ["4"])]  # the history: one edit per Save
    # nothing painted: no edit at all
    assert _save(ann, name, TILE, [{"task": "exclude", "pngs": {"1": ""}, "scale": 4}]).status_code == 200
    assert len(exclusions.edits(project, "LIP-1_B-2", "6E10")) == 2
    assert _save(ann, name, TILE, [{"task": "exclude", "pngs": {"9": _png((1, 2, 1, 2))}, "scale": 4}]).status_code == 400  # an unknown type

    # the annotation view and the Cores view draw the same automatic pixels; people's areas come apart
    info = ann.get(f"/api/annotate/{name}/fields/{TILE}/excluded").json()
    core = admin.get("/api/cores/LIP-1_B-2/6E10/excluded").json()
    assert info["status"] == core["status"] == "done" and info["version"] == core["version"]
    whole = cv2.imdecode(np.frombuffer(admin.get("/api/cores/LIP-1_B-2/6E10/excluded.png").content, np.uint8), cv2.IMREAD_UNCHANGED)
    part = cv2.imdecode(np.frombuffer(ann.get(f"/api/annotate/{name}/fields/{TILE}/excluded.png").content, np.uint8), cv2.IMREAD_UNCHANGED)
    x0, y0 = 300 - project.context_px, 300 - project.context_px
    r0, c0 = (y0 + info["y"]) // 8, (x0 + info["x"]) // 8
    assert (part == whole[r0:r0 + part.shape[0], c0:c0 + part.shape[1]]).all()
    auto = exclusions.automatic(project, "LIP-1_B-2", "6E10")
    final = exclusions.final_say(project, "LIP-1_B-2", "6E10")
    cell = ((y0 + 228) // 8, (x0 + 228) // 8)
    assert auto.codes[cell] == 1 and final.codes[cell] == 4 and part[cell[0] - r0, cell[1] - c0, 3] > 0  # the fold stays excluded from analysis
    cleared_cell = ((y0 + 228) // 8, (x0 + 300) // 8)
    assert auto.codes[cleared_cell] == 1 and final.codes[cleared_cell] == 0 and part[cleared_cell[0] - r0, cleared_cell[1] - c0, 3] > 0
    assert {k["id"] for k in core["kinds"] if k["fraction"]} == {"fold"}  # people's "focus" area is not in the analysis layer
    people = ann.get(f"/api/annotate/{name}/fields/{TILE}/annotated").json()
    assert {r["kind"] for r in people["regions"]} == {"4", "include"} and people["agreement"]["cleared_auto_cells"] > 0
    ring = people["regions"][0]["rings"][0]
    assert all(-200 < x < 1000 and -200 < y < 1000 for x, y in ring)  # field-picture pixels
    layer = admin.get("/api/cores/LIP-1_B-2/6E10/annotations").json()["areas"]
    assert {r["kind"] for r in layer["regions"]} == {"4", "include"}  # the Cores view's Annotations layer: core pixels


def test_a_fix_the_field_session_takes_every_remaining_field_by_default(rounds_project):
    admin, _, _ = rounds_project
    scope = admin.get("/api/labels/rounds/scope", params={"stain": "6E10"}).json()
    assert scope["fields_remaining"] == 2
    first = admin.post("/api/labels/sessions", json={"stain": "6E10", "round": 2, "open": True, "seed": 3}).json()  # no n_fields: -1
    assert sorted(first["fields"]) == sorted([TILE, OTHER]) and first["fields_asked"] == -1 and first["field_seed"] == 3
    assert admin.get("/api/labels/rounds/scope", params={"stain": "6E10"}).json()["fields_remaining"] == 0  # both are in an open session
    taken = admin.post("/api/labels/sessions", json={"stain": "6E10", "round": 2, "n_fields": -1})
    assert taken.status_code == 400 and "open session" in taken.text
    admin.post(f"/api/labels/sessions/{first['name']}/open", json={"open": False})  # a closed session's fields are free again
    again = admin.post("/api/labels/sessions", json={"stain": "6E10", "round": 2, "n_fields": 5, "seed": 3}).json()
    assert sorted(again["fields"]) == sorted([TILE, OTHER]) and again["fields_remaining"] == 2  # asked for more than remain: all
    one = admin.post("/api/labels/sessions", json={"stain": "6E10", "round": 2, "n_fields": 1, "seed": 3}).json()
    assert one["fields"] == rounds.draw_fields([TILE, OTHER], 1, 3)  # the recorded seed reproduces the draw


def test_the_admin_can_preview_the_tutorial_without_a_name(rounds_project):
    admin, (ann, _), _ = rounds_project
    preview = admin.get("/api/session/tutorial").json()  # the admin, signed in, with no annotator name
    assert preview["done"] is False and preview["practice"]["round2"] == "practice-2-6E10"
    field = admin.get(f"/api/annotate/practice-2-6E10/fields/{TILE}")
    assert field.status_code == 200 and field.json()["objects"]
    assert _save(admin, "practice-2-6E10", TILE, [], submit=True).status_code == 200  # echoed back, nothing kept
    # the preview opens only the practice: real sessions still need a name
    made = admin.post("/api/labels/sessions", json={"stain": "6E10", "round": 2, "open": True, "tile_ids": [TILE]}).json()
    assert admin.get(f"/api/annotate/{made['name']}/fields/{TILE}").status_code == 401
    assert ann.get(f"/api/annotate/{made['name']}/fields/{TILE}").status_code == 200
