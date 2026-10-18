from __future__ import annotations

import time

import pandas as pd
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402
from people import annotator, join  # noqa: E402

from stainid.api import state  # noqa: E402
from stainid.api.app import create_app  # noqa: E402


@pytest.fixture()
def client(synthetic_project, monkeypatch, tmp_path_factory):
    monkeypatch.chdir(synthetic_project)
    state.reset()
    admin = TestClient(create_app(), client=("127.0.0.1", 50000))
    assert admin.post("/api/session/admin", json={"password": "admin"}).status_code == 200
    yield admin
    state.reset()


def test_summary_and_cores(client):
    summary = client.get("/api/summary").json()
    assert summary["cores"] == 1 and summary["donors"] == 1
    assert summary["status"]["6E10"]["detection"]["total"] == 1
    cores = client.get("/api/cores?tma=1").json()
    assert cores[0]["core_id"] == "LIP-1_B-2" and cores[0]["stains"] == ["6E10", "AT8", "NeuN"]


def test_images_and_objects(client):
    thumb = client.get("/api/images/cores/LIP-1_B-2/NeuN.jpg?size=128")
    assert thumb.status_code == 200 and thumb.headers["content-type"] == "image/jpeg"
    field = client.get("/api/images/fields/LIP-1_B-2_6E10_S01.jpg")
    assert field.status_code == 200
    layer = client.get("/api/images/fields/LIP-1_B-2_6E10_S01/dab.png")
    assert layer.status_code == 200
    objects = client.get("/api/fields/LIP-1_B-2_6E10_S01/objects").json()["objects"]
    assert len(objects) == 8 and {o["model_class"] for o in objects} == {"compact", "diffuse"}


def test_tables(client):
    names = [t["name"] for t in client.get("/api/tables").json()]
    assert "slide_dab_calibration.csv" in names
    rows = client.get("/api/tables/slide_dab_calibration.csv/rows").json()
    assert rows["total"] == 3


def test_results_table_measures_paging_and_changes(client, synthetic_project):
    path = synthetic_project / "data/analysis/results_donor_region.csv"
    frame = pd.DataFrame({"sample_region_id": [f"D{i:02d}_frontal" for i in range(30)], "donor_id": [f"D{i:02d}" for i in range(30)],
                          "region": "frontal", "disease_group": ["AD", "Control"] * 15, "thread_mean_width_um": [i / 10 for i in range(30)],
                          "plaque_cored_fraction": 0.5, "made_up_size_um2": 1.0})
    frame.to_csv(path, index=False)
    info = client.get("/api/tables/results_donor_region.csv/columns").json()
    measures = {m["column"]: m for m in info["measures"]}
    assert measures["thread_mean_width_um"]["label"] == "Thread width, mean" and measures["thread_mean_width_um"]["unit"] == "µm"
    assert measures["thread_mean_width_um"]["topic"] == "tau" and measures["plaque_cored_fraction"]["label"] == "Cored plaques"
    assert measures["made_up_size_um2"] == {**measures["made_up_size_um2"], "label": "Made up size", "unit": "µm²", "topic": "other"}
    page = client.get("/api/tables/results_donor_region.csv/rows", params={"offset": 10, "limit": 5, "sort": "thread_mean_width_um", "desc": True}).json()
    assert page["total"] == 30 and [r["thread_mean_width_um"] for r in page["rows"]] == [1.9, 1.8, 1.7, 1.6, 1.5]
    found = client.get("/api/tables/results_donor_region.csv/rows", params={"q": "d07", "columns": "donor_id,thread_mean_width_um"}).json()
    assert found["total"] == 1 and found["rows"] == [{"donor_id": "D07", "thread_mean_width_um": 0.7}]
    frame.head(4).to_csv(path, index=False)  # a step writes the table again: the next request sees the new file
    assert client.get("/api/tables/results_donor_region.csv/rows").json()["total"] == 4
    listed = next(t for t in client.get("/api/tables").json() if t["name"] == "results_donor_region.csv")
    assert listed["main"] and listed["columns"] == 7 and listed["title"] == "All measures, per donor-region"
    (synthetic_project / "data/analysis/biology/extra").mkdir(parents=True)
    frame.to_csv(synthetic_project / "data/analysis/biology/extra/new_table.csv", index=False)  # a table made later is listed too
    assert "biology/extra/new_table.csv" in [t["name"] for t in client.get("/api/tables").json()]


def test_steps_describe_needs_and_progress(client):
    steps = {s["id"]: s for s in client.get("/api/steps").json()}
    assert "select" not in steps  # whole cores only: no field-selection step
    assert steps["tiles"]["progress"] == {"state": "done", "done": 3, "total": 3, "unit": "tiles", "note": "3 core images", "outdated": False}
    assert steps["tiles"]["after"] == ["layout", "qc"] and "tiles" in steps["calibrate"]["after"] and "tiles" in steps["nuclei"]["after"]
    assert steps["fields"]["progress"]["total"] == 2
    assert [n["id"] for n in steps["fields"]["needs"]][:2] == ["tiles", "calibration"]
    assert steps["slides"]["command"] is None and steps["dearray"]["options"][0]["key"] == "redo"


def test_settings_save_and_reopen(client, synthetic_project):
    info = client.put("/api/project/config", json={"changes": {"tma": {"prefix": "LIP-"}, "groups": [{"code": "AD", "label": "Alzheimer"}]}}).json()
    assert info["tma_prefix"] == "LIP-" and info["groups"][0] == {"code": "AD", "label": "Alzheimer"}
    assert "prefix: LIP-" in (synthetic_project / "stainid.yaml").read_text()


def test_create_open_project_and_templates(client, tmp_path):
    new = tmp_path / "study2"
    created = client.post("/api/project/create", json={"path": str(new), "name": "Second study"}).json()
    assert created["name"] == "Second study" and created["configured"]
    assert client.post("/api/project/open", json={"path": str(tmp_path / "missing")}).status_code == 404
    listing = client.get("/api/fs", params={"path": str(new)}).json()
    assert listing["path"] == str(new) and "stainid.yaml" in listing["files"]
    assert client.get("/api/templates/tma_layout.csv").text.startswith("tma,core_label,donor_id")


def test_upload_tma_map_validates(client, synthetic_project):
    bad = client.post("/api/upload/tma_layout", files={"file": ("map.csv", b"tma,core_label\n1,B-2\n")})
    assert bad.status_code == 400 and "missing columns" in bad.json()["detail"]
    good = b"tma,core_label,donor_id,region,disease_group\n1,B-2,0001,frontal,AD\n"
    saved = client.post("/api/upload/tma_layout", files={"file": ("map.csv", good)})
    assert saved.status_code == 200 and saved.json()["rows"] == 1


def test_download_is_limited_to_project(client):
    assert client.get("/api/download", params={"path": "data/analysis/slide_dab_calibration.csv"}).status_code == 200
    assert client.get("/api/download", params={"path": "../../etc/hosts"}).status_code == 403


def test_settings_can_change_while_steps_run(client, synthetic_project, tmp_path, monkeypatch):
    from stainid.jobs import queue

    runner = state.get_state().jobs
    monkeypatch.setattr(queue.Scheduler, "_start", lambda self, job, gpus: None)
    monkeypatch.setattr(queue.Scheduler, "_reap", lambda self: None)
    job = runner.submit("summarize", {})
    job.status = "running"
    assert client.put("/api/project/config", json={"changes": {"name": "Renamed"}}).status_code == 200
    assert state.get_state().jobs is runner
    # another project can be opened while this one's steps run: it gets its own queue, and this one carries on
    first = state.get_state().id
    assert client.post("/api/project/create", json={"path": str(tmp_path / "other")}).status_code == 200
    assert state.get_state().jobs is not runner and runner.open and job.status == "running"
    assert state.get_state().jobs.scheduler is runner.scheduler  # one queue for every project
    assert client.get(f"/api/projects/{first}/jobs").json()[0]["id"] == job.id
    job.status = "finished"


def test_tma_map_upload_fills_groups(client):
    good = b"tma,core_label,donor_id,region,disease_group\n1,B-2,0001,frontal,AD\n1,C-2,0002,frontal,CT\n"
    assert client.post("/api/upload/tma_layout", files={"file": ("map.csv", good)}).status_code == 200
    assert [g["code"] for g in client.get("/api/project").json()["groups"]] == ["AD", "CT"]


def test_missing_inputs_say_how_to_get_them(client, tmp_path):
    client.post("/api/project/create", json={"path": str(tmp_path / "fresh")})
    needs = {n["id"]: n for s in client.get("/api/steps").json() for n in s["needs"]}
    assert needs["core_manifest"]["made_by"]["title"] == "Find cores"
    assert needs["model_cellpose"]["download"] == "cellpose" and needs["model_cellpose"]["made_by"] is None
    assert needs["tma_layout"]["template"] == "tma_layout" and "core_label" in needs["tma_layout"]["format"]
    assert needs["model_neun"]["setting"] == "models.neun" and needs["slides_table"]["made_by"]["id"] == "slides"


TILE = "LIP-1_B-2_6E10_S01"


@pytest.fixture()
def prepared(client, synthetic_project, monkeypatch):
    """A prepared 6E10 field: the 8 detected plaques plus 4 rejected candidates; no background job in tests."""
    from stainid.api.routes import annotate
    from stainid.api.state import get_state
    from stainid.labels import pool

    monkeypatch.setattr(annotate, "prepare_job", lambda *args, **kwargs: None)
    monkeypatch.setattr(annotate, "measure_added_soon", lambda *args, **kwargs: None)
    project = get_state().project
    rows = [{"x": 100.0 + 30 * i, "y": 200.0, "radius_px": 20.0, "ai_value": ["compact plaque", "diffuse plaque"][i % 2], "ai_probability": 0.9} for i in range(8)]
    rows += [{"x": 100.0 + 30 * i, "y": 400.0, "radius_px": 10.0, "ai_value": "not a plaque", "ai_probability": 0.2} for i in range(4)]
    folder = pool.folder(project, "6E10")
    folder.mkdir(parents=True)
    pd.DataFrame(rows).assign(tma="1", complete=True, f1=1.0).to_csv(folder / f"{TILE}.csv", index=False)
    return client


def new_session(client, **options):
    body = {"name": "plaques_together", "stain": "6E10", "tile_ids": [TILE], **options}
    created = client.post("/api/labels/sessions", json=body)
    assert created.status_code == 200, created.text
    return created.json()


def test_admin_password_guards_the_app(client):
    visitor = TestClient(client.app)
    assert visitor.get("/api/cores").status_code == 401
    assert visitor.get("/api/session").json()["admin"] is False
    assert visitor.post("/api/session/admin", json={"password": "wrong"}).status_code == 403
    assert visitor.post("/api/session/admin", json={"password": "admin"}).status_code == 200  # the default works from anywhere
    visitor.cookies.clear()
    assert client.put("/api/session/admin/password", json={"current": "admin", "new": "admin"}).status_code == 400
    assert client.put("/api/session/admin/password", json={"current": "admin", "new": "s3cret-pw"}).status_code == 200
    assert client.get("/api/cores").status_code == 200
    assert visitor.post("/api/session/admin", json={"password": "admin"}).status_code == 403
    assert visitor.post("/api/session/admin", json={"password": "s3cret-pw"}).status_code == 200
    assert visitor.get("/api/cores").status_code == 200


def test_field_review_prefills_ai_and_confirms_on_finish(prepared):
    client = prepared
    assert new_session(client)["open"] is False
    assert TestClient(client.app).get("/api/annotate").status_code == 401
    ann = annotator(client, "Ann")
    assert ann.get("/api/annotate").json() == [] and ann.get("/api/annotate/plaques_together/next").status_code == 403
    assert ann.post("/api/labels/sessions/plaques_together/open", json={"open": True}).status_code == 403  # admins only
    client.post("/api/labels/sessions/plaques_together/open", json={"open": True})
    assert ann.get("/api/annotate/plaques_together/next").status_code == 403  # open, but not one of Ann's sets
    join(client, ann, "plaques_together")
    assert ann.get("/api/annotate/plaques_together/next").json()["tile_id"] == TILE

    work = ann.get(f"/api/annotate/plaques_together/fields/{TILE}").json()
    assert len(work["objects"]) == 12 and work["objects"][0]["ai_value"] == "compact plaque" and work["mine"] == []
    url = "/api/annotate/plaques_together/labels"
    assert ann.post(url, json={"records": [{"task": "class", "tile_id": TILE, "x": 100, "y": 200, "value": "cat"}]}).status_code == 400
    saved = ann.post(url, json={"records": [
        {"task": "class", "tile_id": TILE, "x": 100.0, "y": 200.0, "value": "not a plaque"},
        {"task": "class", "tile_id": TILE, "x": 500.0, "y": 500.0, "value": "diffuse plaque"},
        {"task": "outline", "tile_id": TILE, "x": 500.0, "y": 500.0, "points": [[490, 490], [510, 490], [510, 510]]},
        {"task": "tag", "tile_id": TILE, "value": "unusable"}]}).json()
    assert all(r["source"] == "Ann" and r["session"] == "plaques_together" for r in saved)
    finished = ann.post(f"/api/annotate/plaques_together/fields/{TILE}/finish").json()
    assert sum(r.get("confirmed") for r in finished) == 11 and finished[-1]["task"] == "done"
    assert ann.get("/api/annotate/plaques_together/next").json()["tile_id"] is None

    ben = annotator(client, "Ben", "plaques_together")
    assert ben.get(f"/api/annotate/plaques_together/fields/{TILE}").json()["mine"] == []
    ben.post(f"/api/annotate/plaques_together/fields/{TILE}/finish")
    overview = client.get("/api/labels/overview").json()
    plaques = next(s for s in overview["stains"] if s["stain"] == "6E10")
    assert plaques["fields_reviewed"] == 1 and plaques["objects_labelled"] == 13 and plaques["objects_added"] == 1
    assert plaques["objects_with_several_labels"] == 12 and plaques["agreement"] == 11 / 12
    # Ann's added plaque at (500, 500) is one the AI never proposed: a miss; (100, 200) is split between the two of them
    assert plaques["detection"] | {"recall": None} == {"real": 8, "found": 7, "never_proposed": 1, "rejected": 0, "recall": None,
                                                       "never_proposed_share": 1 / 8, "rejected_share": 0.0, "not_checked": 0}
    assert plaques["ai"]["recall"] == 1.0 and plaques["added"] == {"objects": 1, "measured": 0, "could_not_be_measured": 0, "waiting": 1}
    assert overview["sessions"][0]["fields_covered"] == 1 and overview["sessions"][0]["people"] == 2
    people = {p["person"]: p for p in client.get("/api/labels/sessions/plaques_together/people").json()}
    assert people["Ann"]["changed"] == 2 and people["Ben"]["confirmed"] == 12
    export = client.get("/api/labels/export.csv?session=plaques_together").text.splitlines()
    assert export[0].startswith("id,source,stain,task") and len(export) == 1 + 4 + 11 + 1 + 12 + 1

    client.post("/api/labels/sessions/plaques_together/open", json={"open": False})
    assert ann.get(f"/api/annotate/plaques_together/fields/{TILE}").status_code == 403
    assert ".deleted" in client.delete("/api/labels/sessions/plaques_together").json()["moved_to"]


def test_session_named_from_title_opened_at_once_and_shows_who_labels(prepared):
    client = prepared
    first = client.post("/api/labels/sessions", json={"title": "Plaques: round 1", "stain": "6E10", "tile_ids": [TILE], "open": True}).json()
    assert first["name"] == "plaques_round_1" and first["open"] is True
    second = client.post("/api/labels/sessions", json={"stain": "6E10", "kind": "queue", "tools": ["class"], "tile_ids": None}).json()
    assert second["name"] == "6e10_queue" and second["open"] is False
    client.delete("/api/labels/sessions/plaques_round_1")
    assert client.post("/api/labels/sessions", json={"title": "Plaques: round 1", "stain": "6E10", "tile_ids": [TILE]}).json()["name"] == "plaques_round_1_2"

    client.post("/api/labels/sessions/plaques_round_1_2/open", json={"open": True})
    ann = annotator(client, "Ann", "plaques_round_1_2")
    assert ann.get("/api/annotate").json()[0]["instructions"] == ""
    ann.post(f"/api/annotate/plaques_round_1_2/fields/{TILE}/finish")
    listed = {s["name"]: s for s in client.get("/api/labels/overview").json()["sessions"]}
    ann_row = listed["plaques_round_1_2"]["labellers"][0]
    assert ann_row["person"] == "Ann" and ann_row["fields_done"] == 1 and ann_row["confirmed"] == 12
    assert listed["plaques_round_1_2"]["labelling_now"] == 1 and listed["6e10_queue"]["labellers"] == []


def test_blind_queue_hides_ai_and_serves_crops(prepared):
    client = prepared
    new_session(client, name="q", kind="queue", rule="random", ai_view="positions", tools=["class"], tile_ids=None)
    client.post("/api/labels/sessions/q/open", json={"open": True})
    ann = annotator(client, "Ann", "q")
    item = ann.get("/api/annotate/q/next").json()["object"]
    assert item["tile_id"] == TILE and item["remaining"] == 12
    crop = ann.get(f"/api/annotate/q/object.jpg?tile_id={TILE}&x={item['x']}&y={item['y']}")
    assert crop.status_code == 200 and crop.headers["content-type"] == "image/jpeg"
    ann.post("/api/annotate/q/labels", json={"records": [{"task": "class", "tile_id": TILE, "x": item["x"], "y": item["y"], "value": "unsure"}]})
    assert ann.get("/api/annotate/q/next").json()["object"]["remaining"] == 11
    assert client.get("/api/labels/share").json()["links"][0].endswith("/annotate")


def _ready_picture(client, url, seconds=30):
    """Ask for a core picture until its pyramid is prepared (it is built on a background thread)."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        picture = client.get(url).json()
        if picture["ready"]:
            return picture
        assert picture["state"] in ("waiting", "preparing"), picture
        time.sleep(0.1)
    raise AssertionError("the core picture was never prepared")


def test_core_picture_tiles_and_detections(client, synthetic_project):
    picture = _ready_picture(client, "/api/cores/LIP-1_B-2/6E10/picture")
    assert (picture["width"], picture["height"]) == (1200, 1200)
    assert [(lv["cols"], lv["rows"]) for lv in picture["levels"]] == [(3, 3), (2, 2), (1, 1)]  # 1200, 600, 300 px
    tile = client.get(f"/api/images/cores/LIP-1_B-2/6E10/tiles/0/2_2.jpg?v={picture['version']}")
    assert tile.status_code == 200 and tile.headers["content-type"] == "image/jpeg" and "immutable" in tile.headers["cache-control"]
    assert client.get("/api/images/cores/LIP-1_B-2/6E10/tiles/0/3_0.jpg").status_code == 404
    assert client.get("/api/images/cores/LIP-1_B-2/6E10/tiles/1/0_0_dab.png").headers["content-type"] == "image/png"
    assert (synthetic_project / "data" / "cache" / "thumbs" / "LIP-1_B-2_6E10_base.jpg").exists()  # made on the way
    found = client.get("/api/cores/LIP-1_B-2/6E10/objects").json()
    assert found["status"] == "done" and len(found["rows"]) == 8 and found["columns"][:2] == ["x", "y"]
    assert found["tiles"] == [[300, 300, 512, 512]]
    assert client.get("/api/cores/LIP-1_B-2/NeuN/objects").json() == {"status": "not_run", "tiles_done": 0, "tiles_total": 1, "columns": found["columns"], "rows": [], "tiles": [[300, 300, 512, 512]], "outlines": False, "threads": None, "source": "current", "unproposed": None}


def test_cohort_lists_tmas_and_grid_without_pipeline_outputs(client):
    assert client.get("/api/cohort").json() == {"tmas": [{"tma": "1", "cores": 1, "stains": ["6E10", "AT8", "NeuN"]}], "stains": ["6E10", "AT8", "NeuN"]}
    grid = client.get("/api/tmas/1/grid?stain=6E10").json()
    assert grid["columns"] == ["B"] and grid["rows"] == [2] and grid["detection"] == {"done": 1, "total": 1}
    cell = grid["cells"][0]
    assert cell["core_id"] == "LIP-1_B-2" and cell["detection"] == "done" and cell["disease_group"] == "AD"
    deadline = time.monotonic() + 30
    while not client.get("/api/tmas/1/grid?stain=6E10").json()["cells"][0]["thumb"]:  # queued, then made in the background
        assert time.monotonic() < deadline
        time.sleep(0.1)
    assert client.get("/api/images/cores/LIP-1_B-2/6E10.jpg?size=256").status_code == 200
    detail = client.get("/api/cores/LIP-1_B-2").json()
    assert detail["neighbours"] == {"previous": None, "next": None} and detail["images"][0]["stain"] == "6E10"


def test_tokens_for_a_web_app_served_elsewhere(client):
    """Bearer tokens (and `access_token` on plain GETs) carry the same identities as the cookies."""
    remote = TestClient(client.app)  # another computer, without cookies
    local_token = client.post("/api/session/admin", json={"password": "admin"}).json()["token"]
    assert remote.get("/api/session", headers={"Authorization": f"Bearer {local_token}"}).json()["admin"] is True
    assert client.put("/api/session/admin/password", json={"current": "admin", "new": "s3cret-pw"}).status_code == 200
    token = remote.post("/api/session/admin", json={"password": "s3cret-pw"}).json()["token"]
    remote.cookies.clear()
    bearer = {"Authorization": f"Bearer {token}"}
    assert remote.get("/api/cores").status_code == 401
    assert remote.get("/api/session", headers=bearer).json()["admin"] is True
    assert remote.get("/api/cores", headers=bearer).status_code == 200
    assert remote.get("/api/cores", params={"access_token": token}).status_code == 200
    assert remote.post("/api/project/open", params={"access_token": token}, json={"path": "."}).status_code == 401  # GET only
    assert remote.get("/api/cores", headers={"Authorization": "Bearer admin.forged"}).status_code == 401

    other = TestClient(client.app)
    ann = other.post("/api/session/register", json={"name": "Ann, Lee", "password": "correct horse"}).json()["token"]
    other.cookies.clear()
    session = other.get("/api/session", headers={"Authorization": f"Bearer {ann}"}).json()
    assert session["admin"] is False and session["annotator"] == "Ann, Lee"
    assert other.get("/api/cores", headers={"Authorization": f"Bearer {ann}"}).status_code == 403
    assert other.delete("/api/session/admin", headers=bearer).json()["token"] == ""


def test_cross_origin_requests_only_for_allowed_origins(client, monkeypatch):
    preflight = {"Origin": "http://lab-pc:8080", "Access-Control-Request-Method": "GET", "Access-Control-Request-Headers": "authorization"}
    assert "access-control-allow-origin" not in client.options("/api/session", headers=preflight).headers
    assert "access-control-allow-origin" not in client.get("/api/session", headers={"Origin": "http://lab-pc:8080"}).headers

    monkeypatch.setenv("STAINID_ALLOW_ORIGINS", "http://lab-pc:8080/, https://stainid.example.org")
    app = TestClient(create_app(), client=("127.0.0.1", 50000))
    allowed = app.options("/api/cores", headers=preflight)
    assert allowed.status_code == 200 and allowed.headers["access-control-allow-origin"] == "http://lab-pc:8080"
    assert allowed.headers["access-control-allow-credentials"] == "true" and "authorization" in allowed.headers["access-control-allow-headers"].lower()
    refused = app.get("/api/cores", headers={"Origin": "http://lab-pc:8080"})
    assert refused.status_code == 401 and refused.headers["access-control-allow-origin"] == "http://lab-pc:8080"  # the app can read the 401
    other = app.options("/api/cores", headers={**preflight, "Origin": "http://elsewhere:8080"})
    assert "access-control-allow-origin" not in other.headers
    app.post("/api/session/admin", json={"password": "admin"})
    assert app.get("/api/session").json()["allowed_origins"] == ["http://lab-pc:8080", "https://stainid.example.org"]


def test_steps_list_every_running_part(client, monkeypatch):
    from stainid.jobs import COMMAND_LINE, Job, queue

    monkeypatch.setattr(queue.Scheduler, "_reap", lambda self: None)
    runner = state.get_state().jobs
    for index in range(2):  # two parts of one step, run from a terminal (no process to follow here)
        job = Job(id=f"part-{index}", workflow="nuclei", options={"shard_index": index, "shard_count": 2}, argv=["nuclei"], status="running",
                  source=COMMAND_LINE, project=str(runner.root))
        runner.scheduler.jobs[job.id] = job
    steps = {s["id"]: s for s in client.get("/api/steps").json()}
    assert sorted(j["id"] for j in steps["nuclei"]["active"]) == ["part-0", "part-1"]
    assert steps["neun"]["active"] == [] and steps["nuclei"]["job"]["id"] in ("part-0", "part-1")


def test_share_link_uses_the_public_address_behind_a_tunnel(client):
    """Opened through a tunnel or proxy (a host name, https), the labellers' link is that address, not this computer's."""
    tunnelled = client.get("/api/labels/share", headers={"host": "tma.example.org", "x-forwarded-proto": "https"}).json()
    assert tunnelled["links"][0].startswith("https://tma.example.org/p/") and tunnelled["links"][0].endswith("/annotate")
    assert tunnelled["network"] is True
    local = client.get("/api/labels/share", headers={"host": "127.0.0.1:7070"}).json()
    assert not any("example.org" in link for link in local["links"])
