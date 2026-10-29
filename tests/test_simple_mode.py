"""Fix the field (simple): the simple classes as their own vocabulary (configurable per stain), where candidates start
(detection's call in the simple classes), the labels kept as they are and trained through a mapping the Models page
sets, the field saved like the detailed mode's (claimed, finished, read only after), the candidates left in no class
saved as "not an object", the wand's thread segments, and the practice that keeps nothing."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pandas as pd
import pytest
from test_annotation_rounds import OTHER, TILE, _save, rounds_project  # noqa: F401 - the fixture

from stainid.api import state
from stainid.labels import simple, store
from stainid.labels import vocabulary as vocab


def _labels() -> pd.DataFrame:
    return store.current(state.get_state().project.output("reviews"))


def test_the_simple_classes_and_where_candidates_start():
    words = simple.from_config("6E10", None, vocab.defaults("6E10"))
    assert [c.name for c in words.visible] == ["diffuse amyloid beta", "diffuse plaque", "plaque with cores"]
    assert words.start("plaque", "compact") == "cored_plaque" and words.start("plaque", "cored-neuritic") == "cored_plaque"
    assert words.start("plaque", "diffuse") == "diffuse_plaque" and words.start("plaque", "") == "diffuse_plaque"  # untyped: diffuse plaque
    assert words.start("CAA", "") == simple.NONE and words.start("not a plaque", "") == simple.NONE
    assert words.mapped("cored_plaque") == ("plaque", "compact") and words.mapped("diffuse_abeta") is None  # unused until decided
    assert words.mapped(simple.NONE) == ("not a plaque", "")
    tau = simple.from_config("AT8", None, vocab.defaults("AT8"))
    assert [c.name for c in tau.visible] == ["tangles", "neuritic plaque", "neurite"]
    assert tau.get("neurite").kind == simple.WAND and tau.start("tau+ neuron", "pretangle") == "tangle"
    assert tau.mapped("neurite") == ("thread", "")
    # changed in the settings: a renamed class keeps its id, a new one is added, the built-in ones come back when left out
    changed = simple.from_config("6E10", {"classes": [{"id": "cored_plaque", "name": "cored", "key": "3"},
                                                      {"name": "vessel amyloid", "key": "4", "starts_from": ["CAA"]}],
                                          "training": {"vessel_amyloid": "CAA"}}, vocab.defaults("6E10"))
    assert changed.get("cored_plaque").name == "cored" and changed.get("cored_plaque").starts_from == ("plaque/compact", "plaque/cored-neuritic")
    assert changed.start("CAA", "") == "vessel_amyloid" and changed.mapped("vessel_amyloid") == ("CAA", "")
    assert {c.id for c in changed.classes} >= {"diffuse_abeta", "diffuse_plaque"}
    assert simple.problems(changed, vocab.defaults("6E10")) == []
    wrong = simple.from_config("6E10", {"training": {"cored_plaque": "tangle"}}, vocab.defaults("6E10"))
    assert simple.problems(wrong, vocab.defaults("6E10"))


def test_simple_labels_train_through_the_mapping_and_stay_as_they_are():
    words = simple.from_config("6E10", None, vocab.defaults("6E10"))
    labels = pd.DataFrame([
        {"id": f"simple:{TILE}:10:10", "task": "simple", "stain": "6E10", "value": "cored_plaque", "tile_id": TILE, "x": 10.0, "y": 10.0, "candidate": "c1",
         "source": "Ann", "session": "s"},
        {"id": f"simple:{TILE}:20:20", "task": "simple", "stain": "6E10", "value": "diffuse_abeta", "tile_id": TILE, "x": 20.0, "y": 20.0, "candidate": "c2",
         "source": "Ann", "session": "s"},
        {"id": f"simple:{TILE}:30:30", "task": "simple", "stain": "6E10", "value": "none", "tile_id": TILE, "x": 30.0, "y": 30.0, "candidate": "c3",
         "source": "Ann", "session": "s"},
    ])
    rows = simple.training_rows(labels, words)
    assert list(rows.value) == ["plaque", "not a plaque"] and list(rows.task) == ["class", "class"]  # "diffuse amyloid beta": not used
    assert rows.subtype.iloc[0] == "compact" and rows.meta.iloc[0] == {"simple": "cored_plaque"} and rows.id.iloc[0].startswith("class:")
    assert list(labels.task) == ["simple"] * 3  # the simple labels themselves are untouched
    assert simple.counts(labels, "6E10", words) == {"plaque with cores": 1, "diffuse amyloid beta": 1, "not an object": 1}


def test_a_simple_field_starts_from_detection_and_saves_every_candidate(rounds_project):
    admin, (ann, ben), rows = rounds_project
    meta = admin.post("/api/labels/sessions", json={"stain": "6E10", "round": 2, "mode": "simple", "open": True, "tile_ids": [TILE]}).json()
    name = meta["name"]
    assert meta["mode"] == "simple" and "outline" not in meta["tools"]
    detail = ann.get(f"/api/annotate/{name}").json()["meta"]
    assert detail["mode"] == "simple" and [c["id"] for c in detail["simple"]["classes"]] == ["diffuse_abeta", "diffuse_plaque", "cored_plaque"]
    field = ann.get(f"/api/annotate/{name}/fields/{TILE}").json()
    own = [o for o in field["objects"] if o["own"]]
    start = field["simple"]["start"]
    assert {start[o["candidate_id"]] for o in own if o["decision"] == "accepted"} == {"cored_plaque"}  # detection's plaques start lit
    assert {start[o["candidate_id"]] for o in own if o["decision"] == "rejected"} == {"none"}
    # a detailed class or an outline fix is not part of a simple session
    first, second = own[0]["candidate_id"], own[3]["candidate_id"]
    assert _save(ann, name, TILE, [{"task": "class", "candidate": first, "value": "CAA"}]).status_code == 400
    assert _save(ann, name, TILE, [{"task": "simple", "candidate": first, "value": "plaque"}]).status_code == 400  # not a simple class
    edits = [{"task": "simple", "candidate": first, "value": "diffuse_plaque"}, {"task": "simple", "candidate": second, "value": "cored_plaque"},
             {"task": "simple", "x": 60, "y": 70, "value": "diffuse_abeta"}, {"task": "outline", "x": 60, "y": 70, "points": [[55, 65], [65, 65], [65, 75]]}]
    saved = _save(ann, name, TILE, edits)
    assert saved.status_code == 200, saved.text
    live = ben.post(f"/api/annotate/{name}/live", json={"tile_id": TILE, "view": True}).json()
    assert live["simple_checked"][first] == {"by": ["Ann"], "value": "diffuse_plaque", "time": live["simple_checked"][first]["time"]}
    done = _save(ann, name, TILE, [], submit=True).json()["saved"]
    left = [r for r in done if r["task"] == "simple"]
    assert len(left) == 4 and all(r["confirmed"] for r in left)  # the other four, as they started
    assert not [r for r in done if r["task"] == "outline"]  # outlines stay detection's: none recorded as checked
    mine = _labels()
    calls = mine[(mine.task == "simple") & mine.candidate.notna()].set_index("candidate").value.to_dict()
    expected = {o["candidate_id"]: start[o["candidate_id"]] for o in own} | {first: "diffuse_plaque", second: "cored_plaque"}
    assert calls == expected  # every candidate of the field in one class or "not an object"
    assert not (mine.task == "class").any()  # never written as detailed classes
    again = ann.get(f"/api/annotate/{name}/fields/{TILE}").json()
    assert again["read_only"] is True and again["finished"][0]["person"] == "Ann"
    assert ben.get(f"/api/annotate/{name}/fields/{TILE}?own=true").json()["simple"]["checked"] == {}  # an extra annotation starts afresh
    # the Labels page and the training chooser count them in their simple classes
    sets = admin.get("/api/labels/sets?stain=6E10").json()
    mine_set = next(s for s in sets if s["name"] == name)
    assert mine_set["simple"] is True and mine_set["classes"]["plaque with cores"] >= 1 and mine_set["classes"]["diffuse amyloid beta"] == 1
    overview = admin.get("/api/labels/overview").json()
    listed = next(s for s in overview["sessions"] if s["name"] == name)
    assert listed["kind_title"] == "Fix the field (simple)" and listed["simple_classes"]["diffuse plaque"] == 1 and listed["labels"] >= 7
    # training reads them through the mapping
    project = state.get_state().project
    mapped = simple.with_training_rows(mine, project, "6E10")
    assert set(mapped[mapped.task == "class"].value) <= {"plaque", "not a plaque"}
    assert simple.used_mapping(mine, project, "6E10")["training"]["cored_plaque"] == "plaque/compact"


def test_the_mapping_and_the_simple_classes_are_set_per_stain(rounds_project):
    admin, _, _ = rounds_project
    got = admin.get("/api/labels/simple").json()["stains"]["6E10"]
    assert got["training"]["diffuse_abeta"] == "" and got["proposed"]["cored_plaque"] == "plaque/compact"
    assert {"value": "plaque/compact", "name": "plaque · compact (dense-core)", "role": "positive"} in got["targets"]
    changed = admin.put("/api/labels/simple/6E10", json={"training": {**got["training"], "diffuse_abeta": "plaque/diffuse"}})
    assert changed.status_code == 200 and changed.json()["stains"]["6E10"]["training"]["diffuse_abeta"] == "plaque/diffuse"
    assert simple.of(state.get_state().project, "6E10").mapped("diffuse_abeta") == ("plaque", "diffuse")
    assert admin.put("/api/labels/simple/6E10", json={"training": {"diffuse_abeta": "tangle"}}).status_code == 400
    renamed = [{**c, "name": "haze"} if c["id"] == "diffuse_abeta" else c for c in got["classes"]]
    assert admin.put("/api/labels/simple/6E10", json={"classes": renamed}).json()["stains"]["6E10"]["classes"][0]["name"] == "haze"
    back = admin.put("/api/labels/simple/6E10", json={"reset": True}).json()["stains"]["6E10"]
    assert back["classes"][0]["name"] == "diffuse amyloid beta" and back["training"]["diffuse_abeta"] == ""


def test_the_simple_practice_keeps_nothing(rounds_project):
    _, (ann, _), _ = rounds_project
    detail = ann.get("/api/annotate/practice-simple-6E10").json()["meta"]
    assert detail["mode"] == "simple" and detail["practice"] is True
    field = ann.get(f"/api/annotate/practice-simple-6E10/fields/{TILE}").json()
    target = next(o for o in field["objects"] if o["own"])
    assert field["simple"]["start"][target["candidate_id"]] in ("cored_plaque", "none")
    saved = _save(ann, "practice-simple-6E10", TILE, [{"task": "simple", "candidate": target["candidate_id"], "value": "diffuse_plaque"}], submit=True)
    assert saved.status_code == 200 and saved.json()["saved"][-1]["task"] == "done"
    assert _labels().empty
    assert ann.post("/api/session/tutorials", json={"kind": "simple", "done": True}).status_code == 200


def test_candidates_on_the_wand_s_threads_are_in_its_class():
    from stainid.api.routes import rounds as routes

    line = [0, 0, 10, 0, 20, 0]
    assert routes._near_lines(10, 2, [line], 3) and not routes._near_lines(10, 9, [line], 3)


def test_the_neurite_brush_keeps_pieces_of_any_thread_level_and_the_slider_s_factor(rounds_project, monkeypatch):
    from types import SimpleNamespace

    from fastapi import HTTPException

    from stainid.api.routes import rounds as routes
    from stainid.labels import thread_threshold as tt

    wide = tt.WIDE_FACTORS
    assert wide[0] == 0.25 and wide[-1] == 4.0 and len(wide) == 33 and wide[tt.WIDE_DETECTION] == 1.0
    assert wide[tt.DETECTION + tt.WIDE_OFFSET] == tt.FACTORS[tt.DETECTION] and wide[tt.WIDE_OFFSET:tt.WIDE_OFFSET + len(tt.FACTORS)] == tt.FACTORS
    steps = [{"factor": f, "length_um": 1.0, "lines": [[k, 0, k + 5, 0], [k, 9, k + 5, 9]]} for k, f in enumerate(wide)]
    ladder = {"status": "done", "factors": list(wide), "detection": tt.WIDE_DETECTION, "steps": steps, "dab_threshold": 0.2, "ladder": tt.WIDE}
    asked = []
    monkeypatch.setattr(routes, "field_tile", lambda meta, tile_id: {"tile_id": tile_id, "core_id": "C"})
    monkeypatch.setattr(routes.thread_threshold, "field_ladder", lambda project, tile, wide=False: asked.append(wide) or ladder)
    meta = {"stain": "AT8", "name": "s"}

    def mark(**given):
        return SimpleNamespace(task=simple.AREA_TASK, value="neurite", deleted=False, tile_id="T", x=None, y=None, candidate=None, points=None, meta=given)

    low = 0  # 0.25×: at the right end of the slider, the most lines
    record = routes.clean_simple(meta, mark(pieces=[[low, 1], [tt.WIDE_DETECTION, 0]], factor=0.25, ladder="wide"), "Ann")
    assert asked == [True]  # the simple brush reads the wide ladder
    assert record["points"] == [steps[low]["lines"][1], steps[tt.WIDE_DETECTION]["lines"][0]]  # each piece at its own level
    assert record["meta"]["pieces"] == [[low, 1], [tt.WIDE_DETECTION, 0]] and record["meta"]["factor"] == 0.25 and record["meta"]["ladder"] == "wide"
    assert record["meta"]["thread_threshold"] == pytest.approx(tt.RIDGE * 0.2 * 0.25) and record["meta"]["factors"] == [0.25, 1.0]
    standard = routes.clean_simple(meta, mark(pieces=[[0, 1]], factor=0.5), "Ann")  # saved before the wide ladder: its step 0 is 0.5×
    assert standard["meta"]["pieces"] == [[tt.WIDE_OFFSET, 1]] and standard["meta"]["factors"] == [0.5]
    older = routes.clean_simple(meta, mark(segments=[1]), "Ann")  # an earlier page's wand: pieces of detection's step
    assert older["meta"]["pieces"] == [[tt.WIDE_DETECTION, 1]] and older["meta"]["factor"] == 1.0
    with pytest.raises(HTTPException):
        routes.clean_simple(meta, mark(pieces=[[low, 7]], ladder="wide"), "Ann")
    assert routes.clean_simple(meta, mark(pieces=[]), "Ann")["deleted"]
    assert tt.record(ladder, 4.0)["value"] == "4.0000" and tt.record({"factors": list(tt.FACTORS)}, 4.0)["value"] == "2.0000"


def test_the_neurite_pass_starts_from_every_thread_and_keeps_its_strokes(rounds_project, monkeypatch):
    """Every thread at the slider's level starts chosen; the brush's strokes take pieces out and are kept (`edits`);
    taking every piece out still stands as a label; a field finished with the pass untouched counts as every thread at
    detection's level, confirmed."""
    from types import SimpleNamespace

    import pandas as pd

    from stainid.api.routes import rounds as routes
    from stainid.labels import thread_threshold as tt

    wide = tt.WIDE_FACTORS
    steps = [{"factor": f, "length_um": 1.0, "lines": [[k, 0, k + 5, 0], [k, 9, k + 5, 9]]} for k, f in enumerate(wide)]
    ladder = {"status": "done", "factors": list(wide), "detection": tt.WIDE_DETECTION, "steps": steps, "dab_threshold": 0.2, "ladder": tt.WIDE}
    monkeypatch.setattr(routes, "field_tile", lambda meta, tile_id: {"tile_id": tile_id, "core_id": "C"})
    monkeypatch.setattr(routes.thread_threshold, "field_ladder", lambda project, tile, wide=False: ladder)
    meta = {"stain": "AT8", "name": "s"}

    def mark(**given):
        return SimpleNamespace(task=simple.AREA_TASK, value="neurite", deleted=False, tile_id="T", x=None, y=None, candidate=None, points=None, meta=given)

    stroke = {"erase": True, "radius": 4.0, "points": [[1.04, 9], [3, 9]]}
    record = routes.clean_simple(meta, mark(pieces=[[20, 0]], factor=float(wide[20]), ladder="wide", edits=[stroke]), "Ann")
    assert not record["deleted"] and record["points"] == [steps[20]["lines"][0]]
    assert record["meta"]["edits"] == [{"erase": True, "radius": 4.0, "points": [[1.0, 9.0], [3.0, 9.0]]}] and record["meta"]["factor"] == wide[20]
    nothing = routes.clean_simple(meta, mark(pieces=[], factor=1.0, ladder="wide", edits=[stroke]), "Ann")  # every piece taken out
    assert not nothing["deleted"] and nothing["points"] is None and nothing["meta"]["pieces"] == [] and nothing["meta"]["factors"] == [1.0]
    with pytest.raises(Exception):
        routes.clean_simple(meta, mark(pieces=[], edits=[{"erase": True, "radius": 4, "points": []}]), "Ann")

    tile = {"tile_id": "T", "core_id": "C"}
    untouched = routes.neurite_default(meta, tile, "Ann", pd.DataFrame(columns=["task", "value"]))
    assert untouched["confirmed"] and untouched["meta"]["untouched"] and untouched["meta"]["factor"] == 1.0
    assert untouched["meta"]["pieces"] == [[tt.WIDE_DETECTION, 0], [tt.WIDE_DETECTION, 1]] and untouched["points"] == steps[tt.WIDE_DETECTION]["lines"]
    chosen = pd.DataFrame([{"task": simple.AREA_TASK, "value": "neurite"}])
    assert routes.neurite_default(meta, tile, "Ann", chosen) is None  # a choice saved: it stands
    assert routes.neurite_default({**meta, "stain": "6E10"}, tile, "Ann", pd.DataFrame(columns=["task", "value"])) is None


def test_the_wand_logic():
    node = shutil.which("node") or str(Path.home() / ".local/node/bin/node")
    if not Path(node).exists():
        pytest.skip("Node.js is not installed")
    script = Path(__file__).parent / "js" / "wand.test.ts"
    done = subprocess.run([node, "--experimental-strip-types", "--no-warnings", str(script)], capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stdout + done.stderr
