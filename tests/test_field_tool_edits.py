"""The field tool's newer edits: the brush merges into each type's current area, step 1's acceptance threshold (where
untouched objects start, saved with the field), and splitting and merging objects (object labels, never analysis
inputs)."""
# ruff: noqa: F811  (the rounds fixture is shared with test_annotation_rounds)
from __future__ import annotations

import base64
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from test_annotation_rounds import CANDIDATES, OTHER, TILE, _plain_mask, _png, _save, _store_row, rounds_project  # noqa: E402, F401

from stainid import cache, exclusions  # noqa: E402
from stainid.api import state  # noqa: E402
from stainid.labels import candidate_store, compare, decisions, objectedit, store  # noqa: E402

CORE = "LIP-1_B-2"


def _session(admin) -> str:
    return admin.post("/api/labels/sessions", json={"stain": "6E10", "round": 2, "open": True, "tile_ids": [TILE]}).json()["name"]


def _labels() -> pd.DataFrame:
    return store.current(state.get_state().project.output("reviews"))


def test_the_slider_logic_and_its_lock():
    node = shutil.which("node") or str(Path.home() / ".local/node/bin/node")
    if not Path(node).exists():
        pytest.skip("Node.js is not installed")
    script = Path(__file__).parent / "js" / "field_start.test.ts"
    done = subprocess.run([node, "--experimental-strip-types", "--no-warnings", str(script)], capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stdout + done.stderr


def test_brush_strokes_merge_into_each_type_s_current_area(rounds_project):
    """Two overlapping strokes give one region, a stroke touching a detected fold extends that fold, Shift-drag carves it
    (detection's part too), painting another type over it gives the pixels to that type; the analysis layer never
    changes."""
    from stainid.api import store_view

    admin, (ann, _), _ = rounds_project
    project = state.get_state().project
    _plain_mask(project)  # a detected fold over core rows 360-439 (cells 45-54), every column
    cache.forget()
    exclusions.forget()
    auto = exclusions.automatic(project, CORE, "6E10").codes.copy()
    digest = exclusions.digest(project, CORE, "6E10")
    name = _session(admin)
    ann.get(f"/api/annotate/{name}/fields/{TILE}")
    # quarter-scale boxes: (60-90, 10-40) touches the fold (field rows 188-267 = quarter rows 47-66), (80-110, 30-60) overlaps it
    assert _save(ann, name, TILE, [{"task": "exclude", "pngs": {"1": _png((60, 90, 10, 40), (80, 110, 30, 60))}, "scale": 4}]).status_code == 200
    final = exclusions.final_say(project, CORE, "6E10")
    folds = [r for r in final.regions() if r["kind"] == "1"]
    assert len(folds) == 1  # detection's fold and both strokes: one region
    assert final.codes[70, 45] == 1 and auto[70, 45] == 0  # a stroke cell beyond the detected fold now belongs to it
    # Shift-drag (a removal from type 1) inside the detected fold: carved out, a hole in the one region
    assert _save(ann, name, TILE, [{"task": "exclude", "pngs": {"-1": _png((50, 56, 100, 120))}, "scale": 4}]).status_code == 200
    final = exclusions.final_say(project, CORE, "6E10")
    assert final.codes[47, 75] == exclusions.COUNTED and auto[47, 75] == 1
    folds = [r for r in final.regions() if r["kind"] == "1"]
    assert len(folds) == 1 and len(folds[0]["rings"]) >= 2  # still one region, now with a hole
    people = exclusions.annotated(project, CORE, "6E10")
    assert people.codes[47, 75] == exclusions.CLEARED and people.codes[70, 45] == 1
    # painting type 2 over part of the fold gives those pixels to type 2
    assert _save(ann, name, TILE, [{"task": "exclude", "pngs": {"2": _png((85, 95, 40, 50))}, "scale": 4}]).status_code == 200
    final = exclusions.final_say(project, CORE, "6E10")
    assert final.codes[66, 43] == 2
    # the brush starts from these areas: one picture per type over the field
    tile = state.get_state().tile(TILE)
    areas = store_view.field_areas(project, tile)
    assert set(areas["kinds"]) >= {"1", "2"} and areas["status"] == "done"
    raw = base64.b64decode(areas["kinds"]["1"].split(",", 1)[1])
    picture = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
    window = store_view._cells(final, *[*[max(0, int(tile[k]) - project.context_px) for k in ("x_px", "y_px")],
                                        int(tile["x_px"]) + int(tile["width_px"]) + project.context_px,
                                        int(tile["y_px"]) + int(tile["height_px"]) + project.context_px])
    assert int((picture[..., 3] > 127).sum()) == int((final.codes[window] == 1).sum())
    assert ann.get(f"/api/annotate/{name}/fields/{TILE}/areas").json()["kinds"].keys() == areas["kinds"].keys()
    # the analysis layer is detection's alone
    assert np.array_equal(exclusions.automatic(project, CORE, "6E10").codes, auto) and exclusions.digest(project, CORE, "6E10") == digest


def test_the_acceptance_threshold_sets_where_untouched_objects_start_and_is_saved_with_the_field(rounds_project):
    admin, (ann, _), _ = rounds_project
    project = state.get_state().project
    rows = [_store_row(x, y, p, a, threshold=0.5) for x, y, p, a in CANDIDATES]
    rows[-1]["tile_id"] = OTHER
    candidate_store.write_core(project, "6E10", CORE, pd.DataFrame(rows))
    decisions.forget()
    name = _session(admin)
    field = ann.get(f"/api/annotate/{name}/fields/{TILE}").json()
    assert field["detection_threshold"] == pytest.approx(0.5) and field["threshold"] is None
    objects = {o["candidate_id"]: o for o in field["objects"]}
    weak, strong, low = rows[2]["candidate_id"], rows[1]["candidate_id"], rows[3]["candidate_id"]  # 0.45 rejected, 0.55 accepted, 0.2 rejected
    assert objects[weak]["if_accepted"][0] == "plaque" and objects[strong]["if_accepted"] == ["plaque", "compact"]
    store_file = candidate_store.path(project, "6E10", CORE)
    before = (cache.signature(store_file), exclusions.digest(project, CORE, "6E10"), sorted(p.name for p in store_file.parent.iterdir()))
    # step 1 at 0.40, then Save and continue without touching anything
    assert _save(ann, name, TILE, [{"task": "threshold", "value": "0.4"}], submit=True).status_code == 200
    labels = _labels()
    mine = labels[(labels.source == "Ann") & (labels.tile_id == TILE)]
    assert mine[mine.task == "threshold"].value.tolist() == ["0.400"]  # the locked value, with the field
    calls = mine[mine.task == "class"].set_index("candidate")
    assert calls.loc[weak, "value"] == "plaque" and not calls.loc[weak, "confirmed"]  # started accepted at 0.40: not detection's call
    assert calls.loc[strong, "value"] == "plaque" and calls.loc[strong, "confirmed"]
    assert calls.loc[low, "value"] == "not a plaque" and calls.loc[low, "confirmed"]
    assert ann.get(f"/api/annotate/{name}/fields/{TILE}").json()["threshold"] == pytest.approx(0.4)  # opens locked at it
    # no analysis input changes: the store, its folder and the excluded-area digest are as they were
    assert (cache.signature(store_file), exclusions.digest(project, CORE, "6E10"), sorted(p.name for p in store_file.parent.iterdir())) == before
    assert _save(ann, name, TILE, [{"task": "threshold", "value": "3"}]).status_code == 400


def _dumbbell(cx: float, cy: float) -> list[list[float]]:
    mask = np.zeros((60, 100), np.uint8)
    cv2.circle(mask, (25, 30), 18, 1, -1)
    cv2.circle(mask, (70, 30), 18, 1, -1)
    cv2.rectangle(mask, (25, 26), (70, 34), 1, -1)
    contour = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0][0][:, 0, :]
    return (contour + [cx - 47, cy - 30]).astype(float).tolist()


def test_split_and_merge_are_stored_as_object_edits(rounds_project):
    admin, (ann, _), rows = rounds_project
    project = state.get_state().project
    name = _session(admin)
    ann.get(f"/api/annotate/{name}/fields/{TILE}")
    shape = _dumbbell(228, 228)  # over the first candidate (core 400, 400: field 228, 228)
    split = ann.post(f"/api/annotate/{name}/fields/{TILE}/split", json={"points": shape})
    assert split.status_code == 200 and len(split.json()["pieces"]) == 2
    moved = ann.post(f"/api/annotate/{name}/fields/{TILE}/split", json={"points": shape, "seed": [205, 228]}).json()["pieces"]
    assert len(moved) == 2
    far = [[500, 500], [520, 500], [520, 520], [500, 520]]
    cut = ann.post(f"/api/annotate/{name}/fields/{TILE}/split", json={"line": [[228, 180], [228, 280]], "outlines": [shape, far]}).json()["pieces"]
    assert len(cut[0]) == 2 and cut[1] is None  # only the object the line crosses right through
    pieces = split.json()["pieces"]
    group = "g1"
    into = [list(objectedit.centre(p)) for p in pieces]
    records = [{"task": "replaced", "candidate": rows[0]["candidate_id"], "value": "split", "meta": {"group": group, "into": into}}]
    for piece, (x, y) in zip(pieces, into):
        records += [{"task": "class", "x": x, "y": y, "value": "plaque", "subtype": "compact", "meta": {"edit": "split", "group": group}},
                    {"task": "outline", "x": x, "y": y, "points": piece, "meta": {"edit": "split", "group": group}}]
    a, b = [[300, 300], [320, 300], [320, 320], [300, 320]], [[326, 300], [346, 300], [346, 320], [326, 320]]
    merged = ann.post(f"/api/annotate/{name}/fields/{TILE}/merge", json={"outlines": [a, b]})
    assert merged.status_code == 200 and objectedit.area(merged.json()["outline"]) > objectedit.area(a) + objectedit.area(b)
    assert ann.post(f"/api/annotate/{name}/fields/{TILE}/merge", json={"outlines": [a]}).status_code == 400
    x, y = objectedit.centre(merged.json()["outline"])
    records += [{"task": "replaced", "candidate": rows[1]["candidate_id"], "value": "merge", "meta": {"group": "g2", "into": [[x, y]]}},
                {"task": "replaced", "candidate": rows[4]["candidate_id"], "value": "merge", "meta": {"group": "g2", "into": [[x, y]]}},
                {"task": "class", "x": x, "y": y, "value": "plaque", "meta": {"edit": "merge", "group": "g2"}},
                {"task": "outline", "x": x, "y": y, "points": merged.json()["outline"], "meta": {"edit": "merge", "group": "g2"}}]
    assert _save(ann, name, TILE, records, submit=True).status_code == 200
    assert _save(ann, name, TILE, [{"task": "replaced", "candidate": rows[2]["candidate_id"], "value": "vanish"}]).status_code == 400
    labels = _labels()
    mine = labels[labels.source == "Ann"]
    assert set(mine[mine.task == "replaced"].candidate) == {rows[0]["candidate_id"], rows[1]["candidate_id"], rows[4]["candidate_id"]}
    assert mine[mine.task.isin(["class", "outline"])].meta.map(objectedit.is_piece).sum() == 6  # 2 pieces and 1 merged object, each class + outline
    # finishing the field confirms no call on the replaced candidates: their pieces are her answer
    on_replaced = mine[(mine.task == "class") & mine.candidate.isin({rows[0]["candidate_id"], rows[1]["candidate_id"], rows[4]["candidate_id"]})]
    assert on_replaced.empty
    decided = decisions.state(project, "6E10", state.get_state().tiles)
    assert "Ann" not in decided.by(rows[0]["candidate_id"])  # pieces are never matched onto the candidate by position
    edits = compare.object_edits(project, "6E10", state.get_state().tiles, CORE)
    kinds = sorted((e["kind"], len(e["originals"]), len(e["pieces"])) for e in edits)
    assert kinds == [("merge", 2, 1), ("split", 1, 2)]
    split_edit = next(e for e in edits if e["kind"] == "split")
    assert split_edit["originals"][0]["candidate_id"] == rows[0]["candidate_id"] and all(p["points"] for p in split_edit["pieces"])
    added = compare.added_objects(project, "6E10", state.get_state().tiles, CORE)
    assert added.empty  # pieces are not "objects detection missed"
    layer = admin.get(f"/api/cores/{CORE}/6E10/annotations").json()
    assert layer["counts"]["split"] == 1 and layer["counts"]["merge"] == 1 and len(layer["edits"]) == 2


def test_an_automatic_split_gives_two_real_pieces():
    """A dumbbell splits at its neck; a plain oval (one centre) splits across its long axis, never into a sliver."""
    halves = objectedit.split_auto(_dumbbell(100, 100))
    assert halves is not None and len(halves) == 2
    assert all(x < 100 for x, _ in halves[0]) != all(x < 100 for x, _ in halves[1]) or abs(objectedit.area(halves[0]) - objectedit.area(halves[1])) < 0.2 * objectedit.area(_dumbbell(100, 100))
    oval = cv2.ellipse2Poly((200, 200), (90, 40), 20, 0, 360, 4).astype(float).tolist()
    pieces = objectedit.split_auto(oval)
    assert pieces is not None and len(pieces) == 2
    sizes = sorted(objectedit.area(p) for p in pieces)
    assert sizes[0] > 0.3 * sum(sizes)
