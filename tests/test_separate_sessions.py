"""Labelling sessions are separate: each session's labels are in its own log (`labels.store`), inside a session only its
own labels count (who finished or is on a field, the progress, the lock, what a field starts from, its bad areas),
training and the Labels page read the union of every log, closing a session keeps its labels, deleting it removes them
all (into an archive nothing reads), and an earlier release's single labels.jsonl is split once by `stainid labels
split`."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from test_annotation_rounds import OTHER, TILE, _png, _save, rounds_project  # noqa: E402,F401  (the fixture)

from stainid.api import state  # noqa: E402
from stainid.labels import decisions, sessions, store  # noqa: E402


def _session(admin, title: str) -> str:
    made = admin.post("/api/labels/sessions", json={"stain": "6E10", "round": 2, "open": True, "tile_ids": [TILE, OTHER], "title": title})
    assert made.status_code == 200, made.text
    return made.json()["name"]


def _reviews():
    return state.get_state().project.output("reviews")


def _mask(project) -> None:
    """A stage-2 mask of the 6E10 core: analysed tissue everywhere, nothing excluded by detection."""
    import cv2

    from stainid import exclusions
    from stainid.v2.proposals import BITS

    target = exclusions.mask_path(project, "LIP-1_B-2", "6E10")
    target.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(target), np.full((150, 150), BITS["tile"] | BITS["tissue"], np.uint8))


def test_a_session_sees_and_counts_only_its_own_labels(rounds_project):  # noqa: F811
    admin, (ann, ben), rows = rounds_project
    _mask(state.get_state().project)
    a, b = _session(admin, "first"), _session(admin, "second")
    field = ann.get(f"/api/annotate/{a}/fields/{TILE}").json()
    target = next(o for o in field["objects"] if o["own"])["candidate_id"]
    assert _save(ann, a, TILE, [{"task": "class", "candidate": target, "value": "CAA"},
                                {"task": "exclude", "pngs": {"4": _png((52, 62, 52, 62))}, "scale": 4}]).status_code == 200  # over (228, 228)
    assert _save(ann, a, TILE, [], submit=True).status_code == 200  # Ann finished TILE in the first session
    assert ann.get(f"/api/annotate/{a}/fields/{TILE}").status_code == 200  # and has it open again there

    # the lock is per session: Ben may open TILE in the second one while Ann holds it in the first
    opened = ben.get(f"/api/annotate/{b}/fields/{TILE}")
    assert opened.status_code == 200, opened.text
    seen = opened.json()
    assert seen["checked"] == {} and seen["done_by"] == [] and seen["open_by"] == []  # nothing of the first session shows
    ben.post(f"/api/annotate/{b}/fields/{TILE}/leave")
    ann_b = ann.get(f"/api/annotate/{b}/fields/{TILE}").json()
    assert ann_b["mine"] == [] and ann_b["checked"] == {}  # her own labels of the first session do not prefill the second

    # bad areas: what a field of the second session starts from is detection's areas and its own edits only
    areas_a = ann.get(f"/api/annotate/{a}/fields/{TILE}/areas").json()["kinds"]
    areas_b = ann.get(f"/api/annotate/{b}/fields/{TILE}/areas").json()["kinds"]
    assert "4" in areas_a and "4" not in areas_b
    assert {r["kind"] for r in ann.get(f"/api/annotate/{a}/fields/{TILE}/annotated").json()["regions"]} == {"4"}
    assert ann.get(f"/api/annotate/{b}/fields/{TILE}/annotated").json()["regions"] == []
    excluded_a = {o["candidate_id"] for o in ann.get(f"/api/annotate/{a}/fields/{TILE}").json()["objects"] if o["excluded"]}
    excluded_b = {o["candidate_id"] for o in ann_b["objects"] if o["excluded"]}
    assert excluded_a and not excluded_b  # the first session's painted area fades the objects there in it only

    # states, progress, who labelled, the next field: the first session's only
    states_a = {f["tile_id"]: f for f in admin.get(f"/api/labels/sessions/{a}/fields").json()}
    states_b = {f["tile_id"]: f for f in admin.get(f"/api/labels/sessions/{b}/fields").json()}
    assert states_a[TILE]["state"] == "done" and states_a[TILE]["done_by"] == ["Ann"]
    assert states_b[TILE]["done_by"] == [] and states_b[TILE]["state"] != "done"
    listed = {s["name"]: s for s in admin.get("/api/labels/overview").json()["sessions"]}
    assert [p["person"] for p in listed[a]["labellers"]] == ["Ann"] and listed[a]["round_progress"]["decided"] == 1
    assert listed[b]["labellers"] == [] and listed[b]["labels"] == 0 and listed[b]["round_progress"]["decided"] == 0  # "Nobody has labelled yet" and 0 done agree
    assert admin.get(f"/api/labels/sessions/{b}/people").json() == []
    page_b = {f["tile_id"]: f for f in ann.get(f"/api/annotate/{b}").json()["fields"]}
    assert not page_b[TILE]["done_by_me"] and page_b[TILE]["done_by"] == [] and page_b[TILE].get("done_at") == {}
    mine = {s["name"]: s for s in ann.get("/api/annotate").json()}
    assert mine[a]["done_by_me"] == 1 and mine[b]["done_by_me"] == 0 and mine[b]["my_labels"] == 0

    # each session's own log; training and the Labels page read the union
    layout = {p.relative_to(store.folder(_reviews())).as_posix() for p in store.files(_reviews())}
    assert layout == {f"sessions/{a}.jsonl"}
    union = store.current(_reviews())
    assert set(union.session) == {a} and store.current(_reviews(), session=b).empty
    decisions.forget()
    assert set(decisions.state(state.get_state().project, "6E10", state.get_state().tiles).rows.source) == {"Ann"}  # her field, outside her bad area


def test_closing_keeps_a_session_s_labels_and_deleting_removes_them(rounds_project):  # noqa: F811
    admin, (ann, ben), rows = rounds_project
    a, b = _session(admin, "kept"), _session(admin, "dropped")
    first, second = rows[0]["candidate_id"], rows[1]["candidate_id"]
    assert _save(ann, a, TILE, [{"task": "class", "candidate": first, "value": "CAA"}], submit=True).status_code == 200
    assert _save(ben, b, TILE, [{"task": "class", "candidate": second, "value": "CAA"}], submit=True).status_code == 200
    tiles = state.get_state().tiles

    def decided() -> set[str]:
        decisions.forget()
        return set(decisions.state(state.get_state().project, "6E10", tiles).rows.source)

    assert decided() == {"Ann", "Ben"}  # training's truth: the union of the sessions

    # closing keeps the labels: still in the union, training and the label sets models learn from
    assert admin.post(f"/api/labels/sessions/{a}/open", json={"open": False}).json()["open"] is False
    assert decided() == {"Ann", "Ben"}
    sets = {s["name"] for s in admin.get("/api/labels/sets", params={"stain": "6E10"}).json()}
    assert {a, b} <= sets

    # deleting removes every label of the session, and says how many first
    count = admin.get(f"/api/labels/sessions/{b}/labels-count").json()["labels"]
    assert count == len(store.current(_reviews(), session=b)) > 0
    gone = admin.delete(f"/api/labels/sessions/{b}").json()
    assert gone["labels_removed"] == count
    assert b not in sessions.names(_reviews()) and decided() == {"Ann"}
    assert set(store.current(_reviews()).session) == {a}
    assert b not in {s["name"] for s in admin.get("/api/labels/sets", params={"stain": "6E10"}).json()}
    archive = Path(gone["moved_to"])
    assert archive.parent.name == ".deleted" and (archive / "meta.json").exists()
    archived = [json.loads(line) for line in (archive / "labels-sessions.jsonl").read_text().splitlines()]
    assert {r["session"] for r in archived} == {b} and len(archived) >= count
    assert not (store.folder(_reviews()) / "sessions" / f"{b}.jsonl").exists()
    assert {r["session"] for r in store.raw_records(_reviews())} == {a}


def test_deleting_a_session_leaves_a_label_set_of_the_same_name(tmp_path):
    root = tmp_path / "reviews"
    store.append(root, [{"id": "c1", "source": "import: pilot", "stain": "6E10", "task": "class", "tile_id": "t", "x": 1.0, "y": 1.0, "value": "plaque"}])
    store.append(root, [{"id": "c2", "source": "import: someone", "stain": "6E10", "task": "class", "tile_id": "t", "x": 2.0, "y": 2.0, "value": "plaque",
                         "session": "pilot"}])  # an imported session named like the set: one log, two owners
    assert len(store.files(root)) == 1 and len(store.current(root, session="pilot")) == 1
    archive = tmp_path / "archive"
    assert store.remove_session(root, "pilot", archive) == 1
    left = store.current(root)
    assert list(left.id) == ["c1"]  # the set's label stays
    assert [json.loads(line)["id"] for line in (archive / "labels-imports.jsonl").read_text().splitlines()] == ["c2"]


def _legacy_rows() -> list[dict]:
    base = {"stain": "6E10", "task": "class", "tile_id": "t", "x": 1.0, "y": 2.0, "value": "plaque", "deleted": False, "confirmed": False}
    return [
        {**base, "id": "a", "source": "Ann", "session": "fix_one", "time": 1.0},
        {**base, "id": "b", "source": "ai:alpha", "session": "ai_review_round4_6e10", "time": 2.0},
        {**base, "id": "c", "source": "import: pkg rater1", "session": "import-pkg-rater1", "time": 3.0},
        {**base, "id": "d", "source": "import: round3_at8", "session": "", "time": 4.0},
        {**base, "id": "e", "source": "AI reviewer", "session": "", "time": 5.0, "task": "exclude"},
        {**base, "id": "f", "source": "Ben", "session": "", "time": 6.0},
        {**base, "id": "a", "source": "Ann", "session": "fix_one", "time": 7.0, "value": "CAA"},  # an edit: the history keeps both
    ]


def test_the_old_label_log_is_split_once_by_session_and_set(tmp_path):
    root = tmp_path / "reviews"
    legacy = store.legacy_path(root)
    legacy.parent.mkdir(parents=True)
    legacy.write_text("".join(json.dumps(r) + "\n" for r in _legacy_rows()))
    assert store.layout(root) == "legacy"
    with pytest.raises(store.LayoutError, match="labels split"):
        store.current(root)  # the app reads no old log: split it first
    with pytest.raises(store.LayoutError):
        store.append(root, [{"id": "x", "source": "Ann", "session": "fix_one"}])
    done = store.split_legacy(root)
    assert done["labels"] == 7 and done["files"] == {"ai/AI_reviewer.jsonl": 1, "ai/ai_review_round4_6e10.jsonl": 1, "imports/import-pkg-rater1.jsonl": 1,
                                                     "imports/round3_at8.jsonl": 1, "other/Ben.jsonl": 1, "sessions/fix_one.jsonl": 2}
    assert not legacy.exists() and (store.folder(root) / store.BEFORE_SPLIT).read_text().count("\n") == 7  # the old file is kept
    assert store.layout(root) == "split"
    history = store.history(root)
    assert list(history.time) == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]  # one log again, in the order the labels were saved
    current = store.current(root)
    assert len(current) == 6 and current.set_index("id").value["a"] == "CAA"
    with pytest.raises(store.LayoutError):
        store.split_legacy(root)  # once
    legacy.write_text(json.dumps(_legacy_rows()[0]) + "\n")  # an old app wrote to the old log after the split: both layouts
    with pytest.raises(store.LayoutError, match="both"):
        store.layout(root)
    with pytest.raises(store.LayoutError):
        store.current(root)


def test_the_split_command_and_the_app_refusing_an_old_log(synthetic_project, capsys):
    from stainid import cli
    from stainid.api.state import open_project
    from stainid.project import load_project

    root = load_project(synthetic_project).output("reviews")
    legacy = store.legacy_path(root)
    legacy.parent.mkdir(parents=True, exist_ok=True)
    legacy.write_text("".join(json.dumps(r) + "\n" for r in _legacy_rows()))
    with pytest.raises(SystemExit, match="labels split"):
        cli.refuse_old_label_logs([synthetic_project])  # `stainid serve` does not start
    with pytest.raises(store.LayoutError):
        open_project(synthetic_project)
    cli.main(["--project", str(synthetic_project), "labels", "split"])
    said = capsys.readouterr().out
    assert "7 labels split into 6 logs" in said and "sessions/fix_one.jsonl: 2" in said
    cli.refuse_old_label_logs([synthetic_project])  # starts now
    cli.main(["--project", str(synthetic_project), "labels", "split"])
    assert "nothing to split (split already)" in capsys.readouterr().out
    assert pd.DataFrame(store.raw_records(root)).id.tolist() == [r["id"] for r in _legacy_rows()]


def test_mark_bad_areas_sessions_are_separate_too(rounds_project):  # noqa: F811
    from test_mark_bad_areas import CORE, _square

    admin, (ann, ben), _ = rounds_project
    _mask(state.get_state().project)

    def made(title: str) -> str:
        found = admin.post("/api/labels/sessions", json={"stain": "all", "kind": "areas", "open": True, "order": "random", "seed": 1, "title": title})
        assert found.status_code == 200, found.text
        return found.json()["name"]

    one = made("areas one")
    admin.post(f"/api/labels/sessions/{one}/open", json={"open": False})  # a new session takes cores no open one has
    two = made("areas two")
    admin.post(f"/api/labels/sessions/{one}/open", json={"open": True})
    assert ann.get(f"/api/annotate/{one}/areas/next").json()["core_id"] == CORE
    assert ben.get(f"/api/annotate/{two}/areas/next").json()["core_id"] == CORE  # Ann's hold is in the first session only
    edits = {"6E10": {"pngs": {"2": _square(150, (100, 110, 0, 40))}, "scale": 8}}
    saved = ann.post(f"/api/annotate/{one}/cores/{CORE}/save", json={"checked": ["6E10", "AT8", "NeuN"], "edits": edits})
    assert saved.status_code == 200, saved.text
    assert saved.json()["progress"]["decided"] == 1
    there = ben.get(f"/api/annotate/{two}/cores/{CORE}").json()
    assert all(i["checked_by"] == [] and i["edits"] == 0 for i in there["images"]) and there["progress"]["decided"] == 0
    assert ben.get(f"/api/annotate/{two}/cores/{CORE}/6E10/annotated").json()["regions"] == []
    assert "2" not in ben.get(f"/api/annotate/{two}/cores/{CORE}/6E10/areas").json()["kinds"]
    assert "2" in ann.get(f"/api/annotate/{one}/cores/{CORE}/6E10/areas").json()["kinds"]
    assert admin.get("/api/labels/areas").json()["cores_checked"] == 1  # the overview reads every session


def test_two_sessions_of_a_stain_hold_the_same_fields_and_finishing_in_one_leaves_the_other(rounds_project):  # noqa: F811
    """Sessions are independent, choosing fields too: a second session of the same stain draws the fields the first
    one holds, and a field finished in one stays to do in the other (only marked as labelled in another session)."""
    admin, (ann, ben), _ = rounds_project
    made = []
    for title in ("first", "second"):
        r = admin.post("/api/labels/sessions", json={"stain": "6E10", "round": 2, "n_fields": -1, "open": True, "title": title})
        assert r.status_code == 200, r.text
        made.append(r.json())
    a, b = made
    assert TILE in a["fields"] and sorted(a["fields"]) == sorted(b["fields"])  # the first session holding them leaves them in

    assert _save(ann, a["name"], TILE, [], submit=True).status_code == 200
    states_a = {f["tile_id"]: f for f in admin.get(f"/api/labels/sessions/{a['name']}/fields").json()}
    states_b = {f["tile_id"]: f for f in admin.get(f"/api/labels/sessions/{b['name']}/fields").json()}
    assert states_a[TILE]["done_by"] == ["Ann"]
    assert states_b[TILE]["done_by"] == [] and states_b[TILE]["state"] != "done"
    page_b = {f["tile_id"]: f for f in ben.get(f"/api/annotate/{b['name']}").json()["fields"]}
    assert page_b[TILE]["done_by"] == [] and not page_b[TILE]["done_by_me"]
    assert page_b[TILE]["elsewhere"] == [{"session": a["name"], "title": "first", "finished": 1}]  # a mark, nothing more
    page_a = {f["tile_id"]: f for f in ann.get(f"/api/annotate/{a['name']}").json()["fields"]}
    assert page_a[TILE]["elsewhere"] == []

    # a field finished in another session and held by two open sessions is still drawn
    third = admin.post("/api/labels/sessions", json={"stain": "6E10", "round": 2, "n_fields": -1, "open": True, "title": "third"})
    assert third.status_code == 200, third.text
    assert TILE in third.json()["fields"]


def test_the_labels_page_sees_a_change_at_once(rounds_project):  # noqa: F811
    """Closing, opening, adding fields to or deleting a session shows on the Labels page's very next look (its counts
    are kept for a few seconds otherwise)."""
    admin, _, _ = rounds_project
    name = _session(admin, "quick")
    listed = lambda: {s["name"]: s for s in admin.get("/api/labels/overview").json()["sessions"]}  # noqa: E731
    assert listed()[name]["open"] is True
    assert admin.post(f"/api/labels/sessions/{name}/open", json={"open": False}).status_code == 200
    assert listed()[name]["open"] is False
    assert admin.post(f"/api/labels/sessions/{name}/fields/remove", json={"tile_ids": [OTHER]}).status_code == 200
    assert listed()[name]["fields"] == 1 or listed()[name].get("round_progress", {}).get("total") == 1
    assert admin.delete(f"/api/labels/sessions/{name}").status_code == 200
    assert name not in listed()


def test_a_closed_session_is_closed_for_everyone(rounds_project):  # noqa: F811
    """Closed means closed for admins too: its fields open to look at (read-only) and nothing in it is saved; reopened,
    it is as before."""
    admin, (ann, _), _ = rounds_project
    name = _session(admin, "closing")
    assert admin.post(f"/api/labels/sessions/{name}/open", json={"open": False}).status_code == 200
    field = admin.get(f"/api/annotate/{name}/fields/{TILE}").json()
    assert field["read_only"] is True and field["closed"] is True
    refused = _save(admin, name, TILE, [], submit=True)
    assert refused.status_code == 409 and "closed" in refused.text
    assert ann.get(f"/api/annotate/{name}").status_code == 200  # people who joined: it opens to look at, read-only too
    assert ann.get(f"/api/annotate/{name}/fields/{TILE}").json()["read_only"] is True
    assert _save(ann, name, TILE, [], submit=True).status_code == 409
    assert admin.post(f"/api/labels/sessions/{name}/open", json={"open": True}).status_code == 200
    field = admin.get(f"/api/annotate/{name}/fields/{TILE}").json()
    assert field["read_only"] is False and field["closed"] is False
    assert _save(admin, name, TILE, [], submit=True).status_code == 200
