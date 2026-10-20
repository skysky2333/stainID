"""AI review rounds (`review.ai_ingest`, kind "ai_review") are read-only label sets: the Labels page's card shows their
labels, reviewers and agreement (no field progress), and nothing prepares their fields, hands them out, gives them a
join key or puts them in anyone's sets."""
# ruff: noqa: F811 - the fixtures of the API tests are imported and used as arguments
from __future__ import annotations

import json

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from test_api import TILE, client, prepared  # noqa: E402, F401

from stainid.api import accounts, state  # noqa: E402
from stainid.labels import sessions, store  # noqa: E402

NAME = "ai_review_round4_6e10"


def _ai_review(project) -> None:
    from stainid.review import ai_batch  # noqa: PLC0415

    root = project.output("reviews")
    sessions.create(root, NAME, kind="ai_review", tools=["class", "outline"], ai_view="full", per_field=1, rule="random", title="AI review round 4 · 6E10",
                    stain="6E10", fields=[TILE], read_only=True, ai_review={"round": 4, "reviewers": ["ai:a", "ai:b"]})
    base = {"task": "class", "stain": "6E10", "tile_id": TILE, "session": NAME, "value": "plaque", "confirmed": True}
    store.append(root, [{**base, "id": "class:1", "source": "ai:a", "candidate": "6E10:c:1", "x": 10.0, "y": 10.0},
                        {**base, "id": "class:2", "source": "ai:a", "candidate": "6E10:c:2", "x": 50.0, "y": 50.0},
                        {**base, "id": "class:1", "source": "ai:b", "candidate": "6E10:c:1", "x": 10.0, "y": 10.0}])
    folder = ai_batch.stain_folder(project, "6E10", 4)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "ingest_report.json").write_text(json.dumps({"overlap_objects": 1, "overlap_agreed": 1}))


def test_the_card_of_an_ai_review_round(prepared):
    client = prepared
    _ai_review(state.get_state().project)
    listed = {s["name"]: s for s in client.get("/api/labels/overview").json()["sessions"]}
    card = listed[NAME]
    assert card["kind_title"] == "AI review · read only" and card["label_set"] is True and card["read_only"] is True
    assert card["ai_summary"] == {"objects": 2, "fields": 1, "stains": {"6E10": 2}, "reviewers": [{"person": "ai:a", "objects": 2}, {"person": "ai:b", "objects": 1}],
                                  "agreement": {"objects": 1, "agreed": 1}, "round": 4}
    assert not {"fields_covered", "fields_prepared", "round_progress"} & set(card)  # no field progress, nothing "being prepared"
    assert all(s["kind_title"] for s in listed.values())  # every kind has words
    assert sessions.kind_title({"kind": "new_kind"}) == "new kind" and sessions.kind_title({}) == "labelling session"
    assert client.get(f"/api/labels/sessions/{NAME}/objects").status_code == 200  # "Look at the labels"


def test_nothing_prepares_hands_out_or_shares_an_ai_review_round(prepared, monkeypatch, capsys):
    from stainid.api import work  # noqa: PLC0415
    from stainid.api.routes import annotate  # noqa: PLC0415
    from stainid.labels import pool  # noqa: PLC0415
    from stainid.labels import prepare as preparing  # noqa: PLC0415

    client = prepared
    project = state.get_state().project
    asked = []
    monkeypatch.setattr(annotate, "prepare_job", lambda *a, **k: asked.append(("job", a)))
    real_submit = work.submit
    monkeypatch.setattr(work, "submit", lambda kind, key, *a, **k: (asked.append((kind, key)) if NAME in str(key) else None) or real_submit(kind, key, *a, **k))
    monkeypatch.setattr(pool, "prepare", lambda *a, **k: asked.append(("pool", a)))
    _ai_review(project)
    assert accounts.access(state.get_state().id, NAME, make=False)["key"] is None  # no join key made for it
    me = client.get("/api/session").json()["account"]["id"]
    assert client.get(f"/api/annotate/{NAME}").status_code == 409
    assert client.get(f"/api/annotate/{NAME}/next").status_code == 409
    assert client.post(f"/api/labels/sessions/{NAME}/fields", json={"n": 3}).status_code == 409
    assert client.get(f"/api/labels/sessions/{NAME}/access").status_code == 409
    assert client.post(f"/api/labels/sessions/{NAME}/access/members", json={"account": me}).status_code == 409
    client.get("/api/labels/overview")
    client.get(f"/api/labels/sessions/{NAME}/objects")
    assert NAME not in {s["name"] for s in client.get("/api/session/sets").json()}  # not in My labelling sets (an admin's neither)
    assert NAME not in {s["session"] for s in client.get("/api/admin/users").json()["sets"]}  # nobody is added to it
    assert not accounts.is_member(me, state.get_state().id, NAME)  # looking at its labels joins nothing
    preparing.run(project, "prepare", None, 0, NAME, 0)
    assert "read-only label set" in capsys.readouterr().out
    assert asked == []  # nothing prepared its fields
