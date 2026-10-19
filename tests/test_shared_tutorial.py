# ruff: noqa: F811  (the rounds fixture is shared with test_annotation_rounds)
"""The one tutorial of the server (`stainid.tutorial`): built once from a public dataset's field and core, then served
the same by every project — the study's, another one, a new empty one — from its own small project, so the study's
practice routes never show the study's own tissue; nothing of a practice is kept."""
from __future__ import annotations

import io
import shutil

import cv2
import numpy as np
import pytest
from PIL import Image

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402
from test_annotation_rounds import TILE, rounds_project  # noqa: E402, F401

from stainid import tutorial  # noqa: E402
from stainid.api import registry, state  # noqa: E402
from stainid.labels import store  # noqa: E402
from stainid.project import write_default_config  # noqa: E402

FIELD = "practice-2-6E10"
CORES = "practice-areas"
CREDIT = "Images: a public test dataset, MIT licence"


@pytest.fixture()
def public(rounds_project, tmp_path):
    """The tutorial built from another project (a copy of the study's whose 6E10 tissue is blue), as `stainid
    tutorial build` makes it."""
    study = state.get_state().project.root
    other = tmp_path / "public_dataset"
    shutil.copytree(study, other, ignore=shutil.ignore_patterns("cache"))
    picture = other / "data/cores/LIP-1/B-2/6E10.png"
    blue = np.full((1200, 1200, 3), (200, 60, 30), np.uint8)  # BGR: blue tissue, nothing like the study's
    cv2.imwrite(str(picture), blue)
    meta = tutorial.build(other, TILE, "LIP-1_B-2", CREDIT)
    return rounds_project, meta


def _picture(client, base="/api"):
    """The practice field's picture as asked of a project (`base`: /api/projects/<id>), its mean colour."""
    answer = client.get(f"{base}/annotate/{FIELD}/fields/{TILE}/image.jpg")
    assert answer.status_code == 200, answer.text
    return np.asarray(Image.open(io.BytesIO(answer.content)).convert("RGB")).reshape(-1, 3).mean(0)


def test_the_tutorial_is_built_once_and_every_project_serves_it(public, tmp_path):
    (admin, _, _), meta = public
    assert meta["field"] == {"stain": "6E10", "tile_id": TILE, "core_id": "LIP-1_B-2"} and meta["areas"]["stains"] == ["6E10", "AT8", "NeuN"]
    assert (tutorial.folder() / "stainid.yaml").exists() and tutorial.info()["credit"] == CREDIT
    study = state.get_state().id
    empty = tmp_path / "new_project"  # a project made today, with nothing analysed yet
    empty.mkdir()
    write_default_config(empty)
    registry.remember(empty)
    new = registry.project_id(empty)
    guest = TestClient(admin.app)
    seen = []
    for scope in (f"/api/projects/{study}", f"/api/projects/{new}"):
        info = guest.get(f"{scope}/tutorial").json()
        assert info["practice"] == tutorial.practice() and info["practice"]["credit"] == CREDIT
        detail = guest.get(f"{scope}/annotate/{FIELD}").json()
        assert detail["meta"]["fields"] == [TILE] and detail["meta"]["credit"] == CREDIT
        field = guest.get(f"{scope}/annotate/{FIELD}/fields/{TILE}").json()
        cores = guest.get(f"{scope}/annotate/{CORES}").json()["meta"]
        assert cores["stains_of"] == {"LIP-1_B-2": ["6E10", "AT8", "NeuN"]} and cores["credit"] == CREDIT
        seen.append(({o["candidate_id"] for o in field["objects"]}, tuple(np.round(_picture(guest, scope)))))
    assert seen[0] == seen[1]  # the same field, objects and picture, whatever the project


def test_the_study_s_tutorial_routes_no_longer_show_the_study_s_tissue(public):
    (admin, _, _), _ = public
    guest = TestClient(admin.app)
    study = state.get_state().id
    mean = _picture(guest, f"/api/projects/{study}")  # asked of the study
    assert mean[2] > mean[0] + 60, mean  # the public dataset's blue tissue, not the study's grey and brown
    core = guest.get(f"/api/projects/{study}/annotate/{CORES}/cores/LIP-1_B-2/6E10/thumb.jpg?size=128")
    assert core.status_code == 200
    rgb = np.asarray(Image.open(io.BytesIO(core.content)).convert("RGB")).reshape(-1, 3).mean(0)
    assert rgb[2] > rgb[0] + 60, rgb


def test_practice_writes_nothing_in_the_study_or_the_tutorial(public):
    (admin, (ann, _), _), _ = public
    project = state.get_state().project
    log = store.history(project.output("reviews"))
    own = next(o for o in ann.get(f"/api/annotate/{FIELD}/fields/{TILE}").json()["objects"] if o["own"])
    saved = ann.post(f"/api/annotate/{FIELD}/tiles/{TILE}/save", json={"records": [{"task": "class", "candidate": own["candidate_id"], "value": "CAA"}],
                                                                       "submit": True})
    assert saved.status_code == 200
    assert store.history(project.output("reviews")).equals(log)
    assert not store.log_path(tutorial.folder() / "data/annotations").exists()


def test_without_a_built_tutorial_the_practice_says_so(rounds_project, monkeypatch, tmp_path):
    admin, _, _ = rounds_project
    monkeypatch.setenv("STAINID_TUTORIAL", str(tmp_path / "none"))
    guest = TestClient(admin.app)
    assert guest.get("/api/tutorial").json()["practice"] is None
    answer = guest.get(f"/api/annotate/{FIELD}")
    assert answer.status_code == 404 and "stainid tutorial build" in answer.json()["detail"]
    assert guest.get(f"/api/annotate/{CORES}").status_code == 404
