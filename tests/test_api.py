from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

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


def test_steps_describe_needs_and_progress(client):
    steps = {s["id"]: s for s in client.get("/api/steps").json()}
    assert steps["select"]["progress"] == {"state": "done", "done": 3, "total": 3, "unit": "fields", "note": "1 cores", "outdated": False}
    assert steps["fields"]["progress"]["total"] == 2
    assert [n["id"] for n in steps["fields"]["needs"]][:2] == ["analysis_tiles", "calibration"]
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
    runner = state.get_state().jobs
    monkeypatch.setattr(runner, "_start", lambda job: None)
    job = runner.submit("summarize", {})
    job.status = "running"
    assert client.put("/api/project/config", json={"changes": {"name": "Renamed"}}).status_code == 200
    assert state.get_state().jobs is runner
    # another project can be opened while this one's steps run: it gets its own queue, and this one carries on
    first = state.get_state().id
    assert client.post("/api/project/create", json={"path": str(tmp_path / "other")}).status_code == 200
    assert state.get_state().jobs is not runner and runner.open and job.status == "running"
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
    ann, ben = TestClient(client.app), TestClient(client.app)
    assert ann.get("/api/annotate").status_code == 401
    ann.post("/api/session/annotator", json={"name": "Ann"})
    assert ann.get("/api/annotate").json() == [] and ann.get("/api/annotate/plaques_together/next").status_code == 403
    assert ann.post("/api/labels/sessions/plaques_together/open", json={"open": True}).status_code == 401
    client.post("/api/labels/sessions/plaques_together/open", json={"open": True})
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

    ben.post("/api/session/annotator", json={"name": "Ben"})
    assert ben.get(f"/api/annotate/plaques_together/fields/{TILE}").json()["mine"] == []
    ben.post(f"/api/annotate/plaques_together/fields/{TILE}/finish")
    overview = client.get("/api/labels/overview").json()
    plaques = next(s for s in overview["stains"] if s["stain"] == "6E10")
    assert plaques["fields_reviewed"] == 1 and plaques["objects_labelled"] == 13 and plaques["objects_added"] == 1
    assert plaques["objects_with_several_labels"] == 12 and plaques["agreement"] == 11 / 12
    assert overview["sessions"][0]["fields_covered"] == 1 and overview["sessions"][0]["people"] == 2
    people = {p["person"]: p for p in client.get("/api/labels/sessions/plaques_together/people").json()}
    assert people["Ann"]["changed"] == 2 and people["Ben"]["confirmed"] == 12
    export = client.get("/api/labels/export.csv?session=plaques_together").text.splitlines()
    assert export[0].startswith("id,source,stain,task") and len(export) == 1 + 4 + 11 + 1 + 12 + 1

    client.post("/api/labels/sessions/plaques_together/open", json={"open": False})
    assert ann.get(f"/api/annotate/plaques_together/fields/{TILE}").status_code == 403
    assert ".deleted" in client.delete("/api/labels/sessions/plaques_together").json()["moved_to"]


def test_blind_queue_hides_ai_and_serves_crops(prepared):
    client = prepared
    new_session(client, name="q", kind="queue", rule="random", ai_view="positions", tools=["class"], tile_ids=None)
    client.post("/api/labels/sessions/q/open", json={"open": True})
    ann = TestClient(client.app)
    ann.post("/api/session/annotator", json={"name": "Ann"})
    item = ann.get("/api/annotate/q/next").json()["object"]
    assert item["tile_id"] == TILE and item["remaining"] == 12
    crop = ann.get(f"/api/annotate/q/object.jpg?tile_id={TILE}&x={item['x']}&y={item['y']}")
    assert crop.status_code == 200 and crop.headers["content-type"] == "image/jpeg"
    ann.post("/api/annotate/q/labels", json={"records": [{"task": "class", "tile_id": TILE, "x": item["x"], "y": item["y"], "value": "unsure"}]})
    assert ann.get("/api/annotate/q/next").json()["object"]["remaining"] == 11
    assert client.get("/api/labels/share").json()["links"][0].endswith("/annotate")


def test_core_region_preview_and_detections(client):
    region = client.get("/api/cores/LIP-1_B-2/6E10/region?x=0&y=0&size=4000").json()
    assert region["x0"] == 0 and region["y0"] == 0 and len(region["objects"]) == 8
    image = client.get("/api/images/cores/LIP-1_B-2/6E10/region.jpg?x=0&y=0&size=256")
    assert image.status_code == 200 and image.headers["content-type"] == "image/jpeg"
    assert client.get("/api/images/cores/LIP-1_B-2/6E10/region_dab.png?x=0&y=0&size=256").status_code == 200


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

    both = remote.post("/api/session/annotator", json={"name": "Ann, Lee"}, headers=bearer).json()["token"]
    remote.cookies.clear()
    session = remote.get("/api/session", headers={"Authorization": f"Bearer {both}"}).json()
    assert session["admin"] is True and session["annotator"] == "Ann, Lee"
    signed_out = remote.delete("/api/session/admin", headers={"Authorization": f"Bearer {both}"}).json()["token"]
    session = remote.get("/api/session", headers={"Authorization": f"Bearer {signed_out}"}).json()
    assert session["admin"] is False and session["annotator"] == "Ann, Lee"


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
