"""Several projects on one server: the projects list, one state per project, one heavy-step limit for all of them,
annotation links that name their project, and the addresses of the previous release."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest
import yaml

from stainid.api import jobs as server_jobs
from stainid.api import registry
from stainid.jobs import COMMAND_LINE, Job, JobStore


def _project(folder: Path, name: str, **settings) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "stainid.yaml").write_text(yaml.safe_dump({"name": name, **settings}))
    return folder.resolve()


# ---------------------------------------------------------------- the projects list

def test_registry_lists_recent_projects_and_the_projects_folder(tmp_path, monkeypatch):
    elsewhere = _project(tmp_path / "elsewhere" / "Old Study", "Old study")
    folder = tmp_path / "studies"
    first, second = _project(folder / "alpha", "Alpha"), _project(folder / "beta", "Beta")
    (folder / "not-a-project").mkdir()
    registry.remember(elsewhere)
    registry.set_projects_dir(str(folder))
    listed = registry.entries()
    assert [e.path for e in listed] == [str(elsewhere), str(first), str(second)]  # opened first, then by name
    assert [e.name for e in listed] == ["Old study", "Alpha", "Beta"] and listed[1].in_projects_dir and not listed[0].in_projects_dir
    assert listed[0].id == registry.project_id(elsewhere) and listed[0].id.startswith("old-study-") and len({e.id for e in listed}) == 3
    assert registry.project_id(str(elsewhere) + "/") == listed[0].id  # stable, from the folder
    assert registry.find(listed[2].id) == second

    registry.forget(first)  # off the list, files untouched
    assert [e.name for e in registry.entries()] == ["Old study", "Beta"] and (first / "stainid.yaml").exists()
    registry.remember(first)  # opening it again brings it back, first
    assert registry.entries()[0].path == str(first)
    saved = json.loads(registry.SETTINGS.read_text())
    assert saved["recent"] == [str(first), str(elsewhere)] and saved["projects_dir"] == str(folder.resolve())  # older stainIDs read "recent"

    monkeypatch.setenv(registry.PROJECTS_DIR_ENV, str(tmp_path / "elsewhere"))  # `stainid serve --projects-dir` wins over the setting
    assert registry.projects_dir() == tmp_path / "elsewhere"


# ---------------------------------------------------------------- the server

@pytest.fixture()
def server(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from stainid.api.app import create_app

    monkeypatch.setattr(server_jobs.JobManager, "_start", lambda self, job: setattr(job, "status", "running"))
    monkeypatch.setattr(server_jobs.JobManager, "_missing", lambda self, step: "")
    a = _project(tmp_path / "study-a", "Study A", groups=[{"code": "AD", "label": "AD"}])
    b = _project(tmp_path / "study-b", "Study B", tma={"prefix": "LIP-"})
    for root, tma in ((a, "1"), (b, "7")):
        (root / "data").mkdir()
        (root / "data" / "tma_layout.csv").write_text(f"tma,core_label,donor_id,region,disease_group\n{tma},A-1,0001,frontal,AD\n")
    monkeypatch.setenv("STAINID_PROJECT", str(a))
    client = TestClient(create_app(), client=("127.0.0.1", 50000))
    assert client.post("/api/session/admin", json={"password": "admin"}).status_code == 200
    opened = client.post("/api/projects/open", json={"path": str(b)}).json()
    assert opened["name"] == "Study B" and opened["status"]["state"] == "not_set_up"
    client.ids = {"a": registry.project_id(a), "b": opened["id"]}
    client.roots = {"a": a, "b": b}
    return client


def test_each_request_sees_its_own_project(server):
    a, b = (f"/api/projects/{server.ids[k]}" for k in "ab")
    assert server.get(f"{a}/project").json()["name"] == "Study A" and server.get(f"{b}/project").json()["name"] == "Study B"
    assert server.get(f"{b}/project").json()["id"] == server.ids["b"]
    assert server.get(f"{a}/setup/map").json()["tmas"][0]["tma"] == "1" and server.get(f"{b}/setup/map").json()["tmas"][0]["tma"] == "7"
    assert server.put(f"{a}/project/config", json={"changes": {"tma": {"prefix": "BRC-"}}}).status_code == 200
    assert server.get(f"{a}/project").json()["tma_prefix"] == "BRC-" and server.get(f"{b}/project").json()["tma_prefix"] == "LIP-"
    job = server.post(f"{b}/jobs", json={"workflow": "summarize", "options": {}}).json()
    assert [j["id"] for j in server.get(f"{b}/jobs").json()] == [job["id"]] and server.get(f"{a}/jobs").json() == []
    assert (server.roots["b"] / "data" / "jobs" / f"{job['id']}.json").exists()  # each project keeps its own job store
    assert not (server.roots["a"] / "data" / "jobs" / f"{job['id']}.json").exists()
    assert server.get("/api/projects/nothing-123456/project").status_code == 404


def test_projects_page_lists_status_and_removes_without_deleting(server):
    listed = server.get("/api/projects").json()
    assert [p["name"] for p in listed["projects"]] == ["Study B", "Study A"] and listed["default"] == server.ids["b"]
    assert {p["status"]["state"] for p in listed["projects"]} == {"not_set_up"}
    results = server.roots["a"] / "data" / "analysis" / "results_donor_region.csv"
    results.parent.mkdir(parents=True)
    results.write_text("donor_id\n0001\n")
    states = {p["id"]: p["status"]["state"] for p in server.get("/api/projects").json()["projects"]}
    assert states[server.ids["a"]] == "results"
    removed = server.delete(f"/api/projects/{server.ids['b']}").json()
    assert removed == {"removed": server.ids["b"], "still_running": False} and (server.roots["b"] / "stainid.yaml").exists()
    assert [p["name"] for p in server.get("/api/projects").json()["projects"]] == ["Study A"]


def test_new_study_can_start_with_another_projects_settings(server, tmp_path):
    a = f"/api/projects/{server.ids['a']}"
    server.put(f"{a}/project/config", json={"changes": {"models": {"neun": "data/models/neun.joblib"}, "donor_ids": {"digits": 5}}})
    created = server.post("/api/projects/create", json={"path": str(tmp_path / "study-c"), "name": "Study C", "copy_from": server.ids["a"],
                                                        "settings": ["groups", "models", "naming"]}).json()
    config = server.get(f"/api/projects/{created['id']}/project").json()["config"]
    assert config["name"] == "Study C" and config["groups"] == [{"code": "AD", "label": "AD"}] and config["donor_ids"]["digits"] == 5
    assert config["models"]["neun"] == str(server.roots["a"] / "data" / "models" / "neun.joblib")  # the same model file
    assert server.post("/api/projects/create", json={"path": str(tmp_path / "study-c")}).status_code == 409
    assert server.post("/api/projects/open", json={"path": str(tmp_path / "nowhere")}).status_code == 404


def test_old_addresses_use_the_project_opened_last(server):
    assert server.get("/api/project").json()["name"] == "Study B"  # opened last
    info = server.post("/api/project/open", json={"path": str(server.roots["a"])}).json()  # the previous release's switch
    assert info["name"] == "Study A" and server.get("/api/project").json()["name"] == "Study A"
    assert server.get("/api/summary").status_code == 200 and server.get("/api/jobs").json() == []
    assert server.get(f"/api/projects/{server.ids['b']}/project").json()["name"] == "Study B"  # and the other is still open
    assert server.get("/api/session").json()["admin"] is True  # server-wide routes have no project


def test_annotation_links_name_their_project(server):
    b = server.ids["b"]
    links = server.get(f"/api/projects/{b}/labels/share").json()["links"]
    assert links and all(link.endswith(f"/p/{b}/annotate") for link in links)
    reviews = server.roots["b"] / "data" / "annotations" / "sessions" / "s1"
    reviews.mkdir(parents=True)
    from stainid.api import state
    from stainid.api.routes import annotate

    with state.use_project(b):
        assert annotate.annotate_path("s1") == f"/p/{b}/annotate/s1"
    assert server.get(f"/api/projects/{b}/annotate").status_code == 200  # the annotation pages stay public, per project


# ---------------------------------------------------------------- one heavy-step limit for the whole server

def _queue(root: Path, name: str, slots: server_jobs.HeavySlots, monkeypatch, started: list) -> server_jobs.JobManager:
    monkeypatch.setattr(server_jobs.JobManager, "_loop", lambda self: None)
    monkeypatch.setattr(server_jobs.JobManager, "_missing", lambda self, step: "")
    monkeypatch.setattr(server_jobs.JobManager, "_start", lambda self, job: (started.append((self.name, job.workflow)), setattr(job, "status", "running")))
    return server_jobs.JobManager(root, root / "data" / "jobs", name=name, slots=slots)


def test_heavy_steps_share_one_limit_across_projects(tmp_path, monkeypatch):
    slots, started = server_jobs.HeavySlots(limit=1), []
    a = _queue(_project(tmp_path / "a", "Study A"), "Study A", slots, monkeypatch, started)
    b = _queue(_project(tmp_path / "b", "Study B"), "Study B", slots, monkeypatch, started)
    first = a.submit("nuclei", {})
    a._schedule()
    waiting = b.submit("export", {})
    light = b.submit("summarize", {})
    b._schedule()
    assert started == [("Study A", "nuclei"), ("Study B", "summarize")] and light.status == "running"
    assert waiting.status == "queued" and "1 in the project “Study A” (Find nuclei)" in waiting.waiting and "at most 1 at a time on this server" in waiting.waiting
    first.status = "finished"
    b._schedule()
    assert started[-1] == ("Study B", "export") and waiting.waiting == ""
    a.close()
    assert a not in slots.managers


def test_command_line_runs_in_projects_not_open_count_too(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from stainid.api import state

    other = _project(tmp_path / "other", "Other study")
    registry.remember(other)
    store = JobStore(other / "data" / "jobs")
    store.save(Job(id=time.strftime("%Y%m%d-%H%M%S-") + "abcdef", workflow="nuclei", options={}, argv=["nuclei"], status="running",
                   started=time.time(), pid=os.getpid(), source=COMMAND_LINE))
    store.save(Job(id="20200101-000000-old000", workflow="nuclei", options={}, argv=["nuclei"], status="finished", pid=os.getpid()))
    running = state.running_elsewhere()
    assert [(name, job.workflow) for name, job in running] == [("Other study", "nuclei")]
    assert state.project_status(other)["text"] == "Running: Find nuclei"

    slots, started = server_jobs.HeavySlots(limit=1), []
    slots.elsewhere = state.running_elsewhere
    here = _queue(_project(tmp_path / "here", "Here"), "Here", slots, monkeypatch, started)
    job = here.submit("nuclei", {})
    here._schedule()
    assert started == [] and "“Other study” (Find nuclei)" in job.waiting
