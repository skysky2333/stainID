"""A cold field opens fast and the idle computer prepares the open sessions' fields ahead.

Opening a field answers its objects without waiting for its picture (that is the page's own request, cut from only the
rows of the core image down to the field, a blurred preview from the core's thumbnail meanwhile, and the field's size
known before either). The idle-time preparation (`stainid.api.ahead`) plans what people open next first (the cores
they have open, then each session's next fields, then the rest), makes it only while nothing else wants the computer
(someone waiting, jobs, its pause switch) and keeps within the cache's space."""
from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from test_annotation_rounds import OTHER, TILE, rounds_project  # noqa: E402,F401  (the fixture)

from stainid import disk_cache  # noqa: E402
from stainid.api import ahead, render, state, work  # noqa: E402


def test_a_field_is_cut_from_only_the_rows_down_to_it(tmp_path):
    rgb = np.random.default_rng(1).integers(0, 255, (600, 400, 3), np.uint8)
    path = tmp_path / "core.png"
    Image.fromarray(rgb).save(path)
    with render.open_rows(path, 250) as part:
        assert part.size == (400, 250)
        assert np.array_equal(np.asarray(part.convert("RGB")), rgb[:250])
    with render.open_rows(path, 900) as whole:  # past the bottom: the whole image
        assert whole.size == (400, 600)


def _wait_idle(seconds: float = 30.0) -> None:
    end = time.monotonic() + seconds
    while not work.idle() and time.monotonic() < end:
        time.sleep(0.05)


def _session(admin) -> str:
    made = admin.post("/api/labels/sessions", json={"stain": "6E10", "round": 2, "open": True, "tile_ids": [TILE, OTHER]})
    assert made.status_code == 200, made.text
    _wait_idle()
    return made.json()["name"]


def _forget_pictures(*tiles) -> None:
    for tile in tiles:
        render.field_path(state.get_state(), tile).unlink(missing_ok=True)


def test_the_objects_answer_before_the_picture_and_the_size_is_known_first(rounds_project, monkeypatch):  # noqa: F811
    admin, (ann, _ben), _ = rounds_project
    name = _session(admin)
    from stainid.api import sam_embeddings
    from stainid.api.routes import rounds

    monkeypatch.setattr(rounds, "prefetch", lambda *a, **k: None)  # nothing made ahead here
    monkeypatch.setattr(rounds, "warm_up", lambda *a, **k: None)  # nor by the project's warm-up (its first fields)
    monkeypatch.setattr(sam_embeddings, "prepare_field", lambda *a, **k: {"state": "unavailable", "ready": False})
    time.sleep(state.WARM_UP_AFTER + 0.5)  # a warm-up started before the patch: let it finish and its work be done
    with state._WARMING:
        pass
    _wait_idle(120)  # the work earlier tests left in the app's pool, too
    assert work.idle()
    submit = work.submit
    monkeypatch.setattr(work, "submit", lambda kind, key, run, priority=work.NOW, *a, **k:
                        submit(kind, key, run, priority, *a, **k) if priority < work.AHEAD else None)  # nothing made ahead meanwhile
    _forget_pictures(TILE, OTHER)
    field = ann.get(f"/api/annotate/{name}/fields/{TILE}")
    assert field.status_code == 200, field.text
    assert field.json()["objects"] and not render.field_path(state.get_state(), TILE).exists()  # no picture waited for
    preview = ann.get(f"/api/annotate/{name}/fields/{TILE}/preview.jpg")
    assert preview.status_code in (200, 404)  # 404: this core's thumbnail needs the whole image decoded
    picture = ann.get(f"/api/annotate/{name}/fields/{TILE}/image.jpg")
    assert picture.status_code == 200
    with Image.open(render.field_path(state.get_state(), TILE)) as made:
        assert list(made.size) == field.json()["picture"]
    again = ann.get(f"/api/annotate/{name}/fields/{TILE}/preview.jpg")  # once made: the sharp one itself
    assert again.status_code == 200 and again.content == picture.content


def test_the_plan_puts_what_people_open_next_first(rounds_project):  # noqa: F811
    admin, (ann, _ben), _ = rounds_project
    name = _session(admin)
    assert ann.get(f"/api/annotate/{name}/fields/{OTHER}").status_code == 200
    _wait_idle()
    project = state.get_state()
    units, counts = ahead.plan(project)
    assert counts["sessions"] == 1 and counts["fields"] == 2 and counts["cores"] == 1
    assert counts["estimate_bytes"] == 2 * ahead.PICTURE_BYTES + ahead.THUMB_BYTES + counts["near_fields"] * ahead.SAM_BYTES
    first = units[0]
    assert first.tier == 1 and first.kind == "pictures" and set(first.tiles) == {TILE, OTHER}  # the core someone has open: all its fields
    assert [u.tier for u in units] == sorted(u.tier for u in units)
    assert any(u.kind == "caches" and u.tier == 3 for u in units)


def test_it_makes_the_pictures_only_while_nothing_else_wants_the_computer(rounds_project, monkeypatch):  # noqa: F811
    admin, _people, _ = rounds_project
    _session(admin)
    _forget_pictures(TILE, OTHER)
    ahead._PAUSED[0] = -1e9
    assert ahead.busy_now() == ""

    from stainid.jobs import queue

    job = SimpleNamespace(status="running", waiting="")
    monkeypatch.setattr(queue, "_SHARED", SimpleNamespace(paused_all=lambda: False, jobs={"j": job}, close=lambda: None))
    assert ahead.busy_now() == "jobs are running"
    job.status, job.waiting = "queued", "Waits for “Detect plaques” to finish"
    assert ahead.busy_now() == ""  # a job that could not start anyway
    job.waiting = ""
    assert ahead.busy_now() == "jobs are waiting to run"
    job.status = "done"

    ahead.set_paused(True)
    assert ahead.busy_now() == "paused" and ahead.status()["paused"]
    ahead.set_paused(False)

    with render.someone_waits():
        assert "waiting" in ahead.busy_now()
    token = ahead.request_started()
    ahead._REQUESTS[token] -= 1.0  # a page request running for a second
    assert "waiting" in ahead.busy_now()
    ahead.request_ended(token)

    made = ahead.run_once(quiet_look=False)
    assert made >= 1
    assert render.field_path(state.get_state(), TILE).exists() and render.field_path(state.get_state(), OTHER).exists()
    status = ahead.status()
    assert status["fields"] == 2 and status["fields_ready"] == 2


def test_the_pause_switch_is_kept_and_served(rounds_project):  # noqa: F811
    admin, _people, _ = rounds_project
    ahead._PAUSED[0] = -1e9
    assert admin.post("/api/ahead", json={"paused": True}).json()["paused"]
    ahead._PAUSED[0] = -1e9  # read again from its file, as after a restart
    assert admin.get("/api/ahead").json()["state"] == "paused"
    assert not admin.post("/api/ahead", json={"paused": False}).json()["paused"]


def test_features_of_fields_far_off_make_room_for_pictures(tmp_path):
    root = tmp_path / "project"
    for i in range(4):
        disk_cache.write(disk_cache.path(root, "sam_embeddings", f"T{i}.abc.0-0.npz"), b"x" * 1000)
        time.sleep(0.01)
    disk_cache.write(disk_cache.path(root, "fields", "T0.jpg"), b"y" * 500)
    disk_cache.flush()
    assert disk_cache.kind_bytes(root, "sam_embeddings") == 4000
    done = disk_cache.evict_kind(root, "sam_embeddings", 2000, keep_seconds=0)
    assert done["files"] == 2 and disk_cache.kind_bytes(root, "sam_embeddings") == 2000
    left = sorted(p.name for p in disk_cache.folder(root, "sam_embeddings").iterdir())
    assert left == ["T2.abc.0-0.npz", "T3.abc.0-0.npz"]  # the least recently used went first
    assert disk_cache.path(root, "fields", "T0.jpg").exists()  # other kinds untouched
