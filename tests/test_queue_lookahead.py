"""The quick-check queue without waiting: the next few objects handed out at once and held for that person (so people
labelling together get different ones), objects whose picture is made first, and every picture cut ahead with one
read of each core image."""
from __future__ import annotations

import time

import cv2
import numpy as np
import pandas as pd
import pytest

from stainid.labels import sessions

QUEUE = {"name": "q", "stain": "6E10", "rule": "random"}


def _objects(n: int) -> pd.DataFrame:
    return pd.DataFrame({"tile_id": "T", "x": [10.0 * i for i in range(n)], "y": 5.0, "radius_px": 4.0, "ai_value": "not a plaque",
                         "ai_probability": 0.5})


def _labels(*rows: tuple[str, float]) -> pd.DataFrame:
    return pd.DataFrame([{"source": who, "stain": "6E10", "task": "class", "tile_id": "T", "x": x, "y": 5.0, "value": "unsure"} for who, x in rows],
                        columns=["source", "stain", "task", "tile_id", "x", "y", "value"])


def test_people_labelling_at_once_get_different_objects_while_there_are_any():
    clock = [0.0]
    held = sessions.Reservations(seconds=60, clock=lambda: clock[0])
    objects, labels = _objects(10), _labels()
    ann, _ = sessions.next_objects(QUEUE, objects, labels, "Ann", 3, held=held.held("q"))
    held.hold("q", "Ann", [i["key"] for i in ann])
    again, left = sessions.next_objects(QUEUE, objects, labels, "Ann", 3, held=held.held("q"))
    assert [i["key"] for i in again] == [i["key"] for i in ann] and left == 10  # the same objects stay on Ann's screen
    ben, _ = sessions.next_objects(QUEUE, objects, labels, "Ben", 5, held=held.held("q"))
    held.hold("q", "Ben", [i["key"] for i in ben])
    assert not {i["key"] for i in ben} & {i["key"] for i in ann}
    cy, _ = sessions.next_objects(QUEUE, objects, labels, "Cy", 2, held=held.held("q"))
    held.hold("q", "Cy", [i["key"] for i in cy])
    assert not {i["key"] for i in cy} & {i["key"] for i in ann + ben}  # 3 + 5 + 2 of 10: nobody shares
    dee, _ = sessions.next_objects(QUEUE, objects, labels, "Dee", 2, held=held.held("q"))
    assert len(dee) == 2  # everything is held: Dee still gets objects (labelled twice then, which the queue accepts)

    held.release("q", "Ann", [ann[0]["key"]])  # Ann answered her first object
    after, _ = sessions.next_objects(QUEUE, objects, _labels(("Ann", ann[0]["x"])), "Ann", 3, held=held.held("q"))
    assert [i["key"] for i in after[:2]] == [i["key"] for i in ann[1:]] and ann[0]["key"] not in {i["key"] for i in after}
    clock[0] = 61.0  # holds run out (a page closed without answering)
    assert held.held("q") == {}


def test_objects_with_their_picture_ready_come_first_then_fewest_labels():
    objects = _objects(6)
    ready = pd.Series([False, False, True, False, True, False])
    first, _ = sessions.next_objects(QUEUE, objects, _labels(), "Ann", 6, ready=ready)
    assert {i["x"] for i in first[:2]} == {20.0, 40.0}
    labelled, _ = sessions.next_objects(QUEUE, objects, _labels(("Ben", 0.0), ("Cy", 10.0), ("Cy", 30.0)), "Ann", 6)
    assert [i["labels_so_far"] for i in labelled] == [0, 0, 0, 1, 1, 1]


def test_pictures_made_ahead_follow_the_rule():
    objects = _objects(6).assign(ai_probability=[0.1, 0.4, 0.5, 0.9, 0.6, 0.2])
    labels = _labels(("Ben", 0.0), ("Ben", 50.0))
    assert sorted(sessions.may_serve({**QUEUE, "rule": "uncertain"}, objects, labels).x) == [10.0, 20.0, 40.0]
    assert sorted(sessions.may_serve({**QUEUE, "rule": "second"}, objects, labels).x) == [0.0, 50.0]
    assert len(sessions.may_serve(QUEUE, objects, labels)) == 6


pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from stainid.api import pyramid, state  # noqa: E402
from stainid.api.app import create_app  # noqa: E402

TILE = "LIP-1_B-2_6E10_S01"  # x_px = y_px = 300 in a 1200 px core image, field margin 128 px
DOT = (328.0, 348.0)  # a green dot, in field-picture pixels (500, 520 on the core image)


@pytest.fixture()
def queue(synthetic_project, monkeypatch):
    """A 6E10 field with 12 candidates (one on a green dot) and an open quick-check queue over it; counts core decodes."""
    from stainid.api.routes import annotate
    from stainid.labels import pool

    monkeypatch.chdir(synthetic_project)
    core = synthetic_project / "data/cores/LIP-1/B-2/6E10.png"
    image = cv2.imread(str(core))
    cv2.circle(image, (500, 520), 6, (0, 255, 0), -1)
    cv2.imwrite(str(core), image)
    reads = []
    read = pyramid._read
    monkeypatch.setattr(pyramid, "_read", lambda source, *more: reads.append(source) or read(source, *more))
    monkeypatch.setattr(annotate, "prepare_job", lambda *args, **kwargs: None)
    monkeypatch.setattr(annotate, "measure_added_soon", lambda *args, **kwargs: None)
    state.reset()
    admin = TestClient(create_app(), client=("127.0.0.1", 50000))
    assert admin.post("/api/session/admin", json={"password": "admin"}).status_code == 200
    rows = [{"x": DOT[0], "y": DOT[1]}] + [{"x": 150.0 + 30 * i, "y": 200.0} for i in range(11)]
    folder = pool.folder(state.get_state().project, "6E10")
    folder.mkdir(parents=True)
    pd.DataFrame(rows).assign(radius_px=10.0, ai_value="not a plaque", ai_probability=0.5, tma="1", complete=True).to_csv(folder / f"{TILE}.csv", index=False)
    made = admin.post("/api/labels/sessions", json={"name": "quick", "stain": "6E10", "kind": "queue", "rule": "random", "tools": ["class"], "open": True})
    assert made.status_code == 200, made.text
    people = []
    for name in ("Ann", "Ben"):
        person = TestClient(admin.app)
        person.post("/api/session/annotator", json={"name": name})
        people.append(person)
    yield admin, people, reads
    state.reset()


def _pictures(admin) -> dict:
    return next(s for s in admin.get("/api/labels/overview").json()["sessions"] if s["name"] == "quick")["pictures"]


def test_every_queue_picture_is_cut_ahead_with_one_read_of_the_core(queue):
    admin, (ann, _), reads = queue
    deadline = time.monotonic() + 30
    while (pictures := _pictures(admin))["ready"] < pictures["total"] and time.monotonic() < deadline:
        time.sleep(0.1)
    assert pictures == {"ready": 12, "total": 12}  # made when the session was, before anyone asked
    assert len(reads) == 1  # one decode of the core image for all its objects

    response = ann.get(f"/api/annotate/quick/object.jpg?tile_id={TILE}&x={DOT[0]}&y={DOT[1]}")
    assert response.status_code == 200 and response.headers["content-type"] == "image/jpeg"
    crop = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
    b, g, r = crop[crop.shape[0] // 2, crop.shape[1] // 2].astype(int)
    assert g > 200 and r < 80 and b < 80  # centred on its object
    assert len(reads) == 1
    assert ann.get("/api/annotate/quick/next?count=3").json()["pictures"] == {"ready": 12, "total": 12}


def test_a_picture_asked_before_it_was_made_is_cut_at_once(queue, monkeypatch):
    from stainid.api import queue_pictures

    admin, (ann, _), reads = queue
    monkeypatch.setattr(queue_pictures, "_queue", lambda *args: None)  # nothing made ahead
    for crop in state.get_state().cache_dir("label_crops").glob("*.jpg"):
        crop.unlink()
    response = ann.get(f"/api/annotate/quick/object.jpg?tile_id={TILE}&x={DOT[0]}&y={DOT[1]}")
    assert response.status_code == 200
    assert len(list(state.get_state().cache_dir("label_crops").glob("*.jpg"))) == 12  # the rest of the core on the way
    assert ann.get("/api/annotate/quick/object.jpg?tile_id=nowhere&x=1&y=1").status_code == 404


def test_lookahead_is_held_for_its_person_and_answers_release_it(queue):
    _, (ann, ben), _ = queue
    first = ann.get("/api/annotate/quick/next?count=5").json()
    keys = [o["key"] for o in first["objects"]]
    assert len(set(keys)) == 5 and first["object"] == first["objects"][0] and first["remaining"] == 12
    others = ben.get("/api/annotate/quick/next?count=5").json()["objects"]
    assert not {o["key"] for o in others} & set(keys)

    answer = first["objects"][0]
    saved = ann.post("/api/annotate/quick/labels", json={"records": [{"task": "class", "tile_id": answer["tile_id"], "x": answer["x"], "y": answer["y"], "value": "unsure"}]})
    assert saved.status_code == 200
    after = ann.get("/api/annotate/quick/next?count=5").json()
    assert [o["key"] for o in after["objects"]][:4] == keys[1:] and after["remaining"] == 11  # the rest stay in order
    assert not {o["key"] for o in after["objects"]} & {o["key"] for o in others}
    undo = ann.post("/api/annotate/quick/labels", json={"records": [{"task": "class", "tile_id": answer["tile_id"], "x": answer["x"], "y": answer["y"], "deleted": True}]})
    assert undo.status_code == 200 and ann.get("/api/annotate/quick/next?count=12").json()["remaining"] == 12
