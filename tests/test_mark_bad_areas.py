# ruff: noqa: F811  (the rounds fixture is shared from test_annotation_rounds)
"""The "Mark bad areas" round (labels/areas.py, api/routes/areas.py): a core with all its stain images is the unit people
claim; saving checks every stain image (also with no edits) and puts the edits into the resolved layer; the object
rounds draw fields from cores whose areas were checked first; and the Cores view's review layers (the minimap's tile
status, the "Annotations" layer) read the same label log, through the readers scoring and the Labels overview use."""
from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from test_annotation_rounds import TILE, _plain_mask, _png, _save, rounds_project  # noqa: E402,F401

from stainid import exclusions  # noqa: E402
from stainid.api import state  # noqa: E402
from stainid.labels import areas, compare, decisions, rounds, status, store  # noqa: E402

CORE = "LIP-1_B-2"


def _session(admin, **more) -> dict:
    body = {"stain": "all", "kind": "areas", "open": True, "order": "random", "seed": 1, **more}
    made = admin.post("/api/labels/sessions", json=body)
    assert made.status_code == 200, made.text
    return made.json()


def test_a_session_claims_whole_cores_and_a_save_checks_every_stain_image(rounds_project):
    admin, (ann, ben), _ = rounds_project
    project = state.get_state().project
    _plain_mask(project)
    meta = _session(admin)
    assert meta["kind"] == "areas" and meta["cores"] == [CORE] and meta["stains_of"][CORE] == ["6E10", "AT8", "NeuN"]
    name = meta["name"]
    first = ann.get(f"/api/annotate/{name}/areas/next").json()
    assert first["core_id"] == CORE
    assert ben.get(f"/api/annotate/{name}/areas/next").json()["core_id"] is None  # Ann holds it: nobody else is handed it
    core = ann.get(f"/api/annotate/{name}/cores/{CORE}").json()
    assert [i["stain"] for i in core["images"]] == ["6E10", "AT8", "NeuN"] and core["images"][0]["excluded"]["status"] == "done"
    assert "group" not in str(core) and "disease" not in str(core)  # blinded

    early = ann.post(f"/api/annotate/{name}/cores/{CORE}/save", json={"checked": ["6E10"]})
    assert early.status_code == 400  # every stain image must have been looked at
    before = exclusions.digest(project, CORE, "6E10")
    saved = ann.post(f"/api/annotate/{name}/cores/{CORE}/save", json={"checked": ["6E10", "AT8", "NeuN"]})
    assert saved.status_code == 200, saved.text
    labels = store.current(project.output("reviews"))
    assert {k: sorted(v) for k, v in status.areas_checked(labels).items()} == {(CORE, s): ["Ann"] for s in ("6E10", "AT8", "NeuN")}
    assert exclusions.digest(project, CORE, "6E10") == before  # nothing changed: detection's areas confirmed, the layer as it was
    assert saved.json()["progress"]["decided"] == 1
    assert ann.get(f"/api/annotate/{name}/areas/next").json()["core_id"] is None  # done


def test_area_edits_are_people_s_areas_and_leave_the_analysis_layer_alone(rounds_project):
    admin, (ann, _), _ = rounds_project
    project = state.get_state().project
    _plain_mask(project)
    name = _session(admin)["name"]
    before = exclusions.digest(project, CORE, "6E10")
    # 150 × 150 cells for the 1200-px core: clear rows 45–54 (the fold) in columns 0–39, paint "artifact" at rows 100–109
    edits = {"6E10": {"pngs": {"include": _square(150, (45, 55, 0, 40)), "2": _square(150, (100, 110, 0, 40))}, "scale": 8}}
    saved = ann.post(f"/api/annotate/{name}/cores/{CORE}/save", json={"checked": ["6E10", "AT8", "NeuN"], "edits": edits})
    assert saved.status_code == 200, saved.text
    assert exclusions.digest(project, CORE, "6E10") == before  # the analysis' input (and every stage digest) is untouched
    people = exclusions.annotated(project, CORE, "6E10")
    assert people.codes[50, 10] == exclusions.CLEARED and people.codes[50, 100] == 0 and people.codes[105, 10] == 2
    final = exclusions.final_say(project, CORE, "6E10")
    assert final.codes[50, 10] == 0 and final.codes[50, 100] == 1 and final.codes[105, 10] == 2
    assert exclusions.automatic(project, CORE, "6E10").codes[50, 10] == 1
    page = ann.get(f"/api/annotate/{name}/cores/{CORE}/6E10/annotated").json()
    assert {r["kind"] for r in page["regions"]} == {"include", "2"} and page["edits"][0]["person"] == "Ann"
    (edit,) = exclusions.edits(project, CORE, "6E10")
    assert edit.source == "Ann" and set(edit.kinds) == {"include", "2"}
    assert exclusions.digest(project, CORE, "AT8") is None  # no AT8 mask in this project


def _square(size: int, box) -> str:
    import base64

    import cv2
    import numpy as np

    painted = np.zeros((size, size, 4), np.uint8)
    r0, r1, c0, c1 = box
    painted[r0:r1, c0:c1] = 255
    return "data:image/png;base64," + base64.b64encode(cv2.imencode(".png", painted)[1].tobytes()).decode()


def test_object_rounds_keep_the_brush_and_fade_objects_in_excluded_areas(rounds_project):
    admin, (ann, _), _ = rounds_project
    project = state.get_state().project
    _plain_mask(project)  # a fold over core rows 360–439
    name = admin.post("/api/labels/sessions", json={"stain": "6E10", "round": 2, "open": True, "tile_ids": [TILE]}).json()["name"]
    field = ann.get(f"/api/annotate/{name}/fields/{TILE}").json()
    faded = {round(o["y"]) + 172: o["excluded"] for o in field["objects"] if o["own"]}
    assert faded[400] is True and faded[480] is False  # (·, 400) lies in the fold, (·, 480) does not
    paint = _save(ann, name, TILE, [{"task": "exclude", "pngs": {"3": _png((100, 110, 100, 110))}, "scale": 4}])
    assert paint.status_code == 200  # the brush stays in the object rounds: the same layer
    assert exclusions.edits(project, CORE, "6E10")[-1].kinds == ("3",)


def test_object_rounds_draw_fields_of_cores_with_checked_areas_first():
    fields = [f"T{i}" for i in range(10)]
    first = {"T3", "T7", "T8"}
    drawn = rounds.draw_fields(fields, 4, seed=5, first=first)
    assert set(drawn[:3]) == first and drawn[3] not in first and len(drawn) == 4
    assert set(rounds.draw_fields(fields, 2, seed=5, first=first)) <= first
    every = rounds.draw_fields(fields, -1, seed=5, first=first)
    assert set(every[:3]) == first and sorted(every) == sorted(fields)
    assert rounds.draw_fields(fields, 4, seed=5, first=first) == drawn  # the same seed, the same draw


def test_a_new_round_counts_its_fields_from_cores_with_checked_areas(rounds_project):
    admin, (ann, _), _ = rounds_project
    _plain_mask(state.get_state().project)
    name = _session(admin)["name"]
    assert ann.post(f"/api/annotate/{name}/cores/{CORE}/save", json={"checked": ["6E10", "AT8", "NeuN"]}).status_code == 200
    made = admin.post("/api/labels/sessions", json={"stain": "6E10", "round": 2, "open": True, "n_fields": -1, "seed": 2}).json()
    assert made["fields_from_checked_areas"] == len(made["fields"]) == 2


def test_the_minimap_status_follows_the_label_log(rounds_project):
    admin, (ann, ben), _ = rounds_project
    project = state.get_state().project
    _plain_mask(project)
    name = admin.post("/api/labels/sessions", json={"stain": "6E10", "round": 2, "open": True, "tile_ids": [TILE]}).json()["name"]
    found = admin.get(f"/api/cores/{CORE}/6E10/status").json()
    states = {t["tile_id"]: t["state"] for t in found["tiles"]}
    assert set(states.values()) == {"not_annotated"} and found["width"] == 1200 and found["areas_checked_by"] == []
    ann.get(f"/api/annotate/{name}/fields/{TILE}")  # opening a field holds it: in progress
    assert {t["tile_id"]: t["state"] for t in admin.get(f"/api/cores/{CORE}/6E10/status").json()["tiles"]}[TILE] == "in_progress"
    assert _save(ann, name, TILE, [], submit=True).status_code == 200
    after = admin.get(f"/api/cores/{CORE}/6E10/status").json()
    mine = {t["tile_id"]: t for t in after["tiles"]}[TILE]
    assert mine["state"] == "reviewed" and mine["by"] == ["Ann"] and after["counts"]["reviewed"] == 1
    labels = store.current(project.output("reviews"))
    assert status.tile_states(labels, "6E10", [TILE])[TILE]["state"] == "reviewed"  # the same reader
    assert after["sessions"] and after["sessions"][0]["name"] == name  # a tile can be opened in annotation from the minimap


def test_the_annotations_layer_is_scorings_comparison_and_never_replaces_detections_outline(rounds_project):
    admin, (ann, ben), rows = rounds_project
    project = state.get_state().project
    name = admin.post("/api/labels/sessions", json={"stain": "6E10", "round": 2, "open": True, "tile_ids": [TILE]}).json()["name"]
    outlines_before = admin.get(f"/api/cores/{CORE}/6E10/outlines").json()
    ids = [r["candidate_id"] for r in rows]
    edits = [{"task": "class", "candidate": ids[0], "value": "plaque", "subtype": "diffuse"}, {"task": "class", "candidate": ids[1], "value": "not a plaque"},
             {"task": "outline", "candidate": ids[0], "points": [[220, 220], [236, 220], [236, 236], [220, 236]]}]
    assert _save(ben, name, TILE, edits).status_code == 200
    layer = admin.get(f"/api/cores/{CORE}/6E10/annotations").json()
    kinds = {o["candidate_id"]: o["kind"] for o in layer["objects"]}
    assert kinds[ids[0]] == "changed" and kinds[ids[1]] == "rejected"
    decisions.forget()
    assert {k: layer["counts"][k] for k in compare.KINDS} == {k: v for k, v in compare.summary(project, "6E10", state.get_state().tiles, CORE).items()
                                                             if k in compare.KINDS}
    scored = decisions.state(project, "6E10", state.get_state().tiles)
    assert set(kinds) == set(scored.rows.candidate)  # every candidate scoring counts, and only those
    fixed = next(o for o in layer["objects"] if o["candidate_id"] == ids[0])["outline_fixed"]
    assert fixed and fixed[0]["person"] == "Ben" and fixed[0]["points"][0] == [392.0, 392.0]  # core pixels
    assert layer["counts"]["outline_fixed"] == 1
    assert admin.get(f"/api/cores/{CORE}/6E10/outlines").json() == outlines_before  # the Cores view keeps detection's outlines


def test_agreement_and_detections_areas_against_peoples(rounds_project):
    admin, (ann, ben), _ = rounds_project
    project = state.get_state().project
    _plain_mask(project)
    meta = _session(admin)
    from stainid.labels import sessions

    sessions.update(project.output("reviews"), meta["name"], second=[CORE])  # this core gets a second, independent check
    name = meta["name"]
    a = {"6E10": {"pngs": {"include": _square(150, (45, 55, 0, 75))}, "scale": 8}}
    b = {"6E10": {"pngs": {"1": _square(150, (100, 120, 0, 75))}, "scale": 8}}
    assert ann.post(f"/api/annotate/{name}/cores/{CORE}/save", json={"checked": ["6E10", "AT8", "NeuN"], "edits": a}).status_code == 200
    assert ben.get(f"/api/annotate/{name}/areas/next").json()["core_id"] == CORE  # a second check is someone else
    assert ben.post(f"/api/annotate/{name}/cores/{CORE}/save", json={"checked": ["6E10", "AT8", "NeuN"], "edits": b}).status_code == 200
    overview = admin.get("/api/labels/areas").json()
    detector = admin.get("/api/models/bad-areas").json()
    assert overview["cores_checked"] == 1 and {s["stain"]: s["double_checked"] for s in overview["stains"]}["6E10"] == 1
    agreement = overview["agreement"]
    assert agreement["pairs"] == 3 and [d["stain"] for d in agreement["disagreements"]] == ["6E10"]  # AT8, NeuN: nobody changed anything
    folds = [r for r in detector["by_tma"] if r["stain"] == "6E10" and r["tma"] == "all" and r["type"] == 1]
    assert folds and folds[0]["cleared_mm2"] > 0 and folds[0]["added_mm2"] > 0 and 0 < folds[0]["f1"] < 1
    assert len({r["donor_id"] for r in detector["by_donor"]}) == 2 and detector["model"] is None  # one donor and "all"; the hook for a trained model
    assert areas.needed({"second": [CORE]}, CORE) == 2


def test_mark_bad_areas_uses_the_field_tool_s_brush_model(rounds_project):
    """The same brush as the field tool: it starts from each type's current area over the whole core image, a drag adds
    to the chosen type (extending detection's fold, one region), Shift-drag takes out of it (detection's part too:
    "marked fine" is no more than a part taken out of a type), and the analysis layer never changes."""
    import base64

    import cv2
    import numpy as np

    admin, (ann, _), _ = rounds_project
    project = state.get_state().project
    _plain_mask(project)  # a detected fold over cells 45-54, every column
    name = _session(admin)["name"]
    before = exclusions.digest(project, CORE, "6E10")
    current = ann.get(f"/api/annotate/{name}/cores/{CORE}/6E10/areas").json()
    assert current["status"] == "done" and set(current["kinds"]) == {"1"} and (current["x"], current["y"]) == (0, 0)
    assert current["width"] == current["height"] == 1200
    picture = cv2.imdecode(np.frombuffer(base64.b64decode(current["kinds"]["1"].split(",", 1)[1]), np.uint8), cv2.IMREAD_UNCHANGED)
    assert picture.shape[:2] == (150, 150) and picture[50, 10, 3] == 255 and picture[60, 10, 3] == 0
    # add to type 1 at cells 55-64 (touching the fold), take cells 45-54 x 0-39 out of type 1
    edits = {"6E10": {"pngs": {"1": _square(150, (55, 65, 0, 40)), "-1": _square(150, (45, 55, 0, 40))}, "scale": 8}}
    assert ann.post(f"/api/annotate/{name}/cores/{CORE}/save", json={"checked": ["6E10", "AT8", "NeuN"], "edits": edits}).status_code == 200
    final = exclusions.final_say(project, CORE, "6E10")
    assert final.codes[60, 10] == 1 and final.codes[50, 10] == exclusions.COUNTED and final.codes[50, 100] == 1
    folds = [r for r in final.regions() if r["kind"] == "1"]
    assert len(folds) == 1  # the extended fold is one region
    people = exclusions.annotated(project, CORE, "6E10")
    assert people.codes[50, 10] == exclusions.CLEARED  # taken out: what used to be "marked fine"
    after = ann.get(f"/api/annotate/{name}/cores/{CORE}/6E10/areas").json()
    assert set(after["kinds"]) == {"1"} and after["version"] != current["version"]  # the brush starts from the new areas
    assert exclusions.digest(project, CORE, "6E10") == before
    assert ann.get(f"/api/annotate/{name}/cores/{CORE}").json()["pixel_size_um"] == project.pixel_size_um  # the brush in um
