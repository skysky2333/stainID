# ruff: noqa: F811  (the rounds fixture is shared with test_annotation_rounds)
"""Outline fixes: the outlines people changed in "Fix the field" (precise edits, splits, merges) against every outline
detection had (the stain outline, Segment Anything small, medium and large, and the one shown first), per stain on the
Models page, with a hint when one outline is clearly closer; the rows keep each fixed outline for training."""
from __future__ import annotations

import io

import pandas as pd
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from test_annotation_rounds import TILE, _save, _store_row, rounds_project  # noqa: E402, F401

from stainid.api import state  # noqa: E402
from stainid.labels import candidate_store, decisions, outline_fixes  # noqa: E402

CORE = "LIP-1_B-2"
ORIGIN = 172  # the field's picture starts at (172, 172) in the core image
EDITED = [(400.0, 400.0), (460.0, 400.0), (520.0, 400.0), (400.0, 480.0), (460.0, 480.0)]
MERGED = [(600.0, 560.0), (624.0, 560.0)]
PICKED = (520.0, 480.0)


def square(x: float, y: float, half: float) -> list[list[float]]:
    return [[x - half, y - half], [x + half, y - half], [x + half, y + half], [x - half, y + half]]


def text(points) -> str:
    return " ".join(f"{v:.0f}" for p in points for v in p)


def _row(x: float, y: float) -> dict:
    """A candidate whose stain outline is 16 px across (shown first: 6E10's default outline is the stain's) and whose
    Segment Anything masks are 12, 24 and 40 px across."""
    masks = [square(x, y, h) for h in (6, 12, 20)]
    return _store_row(x, y, 0.9, True, outline=square(x, y, 8), sam_outline=masks[0], sam_status="ok", sam_masks=masks, sam_scores=[0.9, 0.8, 0.7],
                      sam_default=0)


def field(points) -> list[list[float]]:
    return [[x - ORIGIN, y - ORIGIN] for x, y in points]


def test_outline_fixes_compare_each_of_detections_outlines_with_what_people_drew(rounds_project, monkeypatch):
    admin, (ann, _), _ = rounds_project
    project, tiles = state.get_state().project, state.get_state().tiles
    rows = [_row(x, y) for x, y in [*EDITED, *MERGED, PICKED]]
    candidate_store.write_core(project, "6E10", CORE, pd.DataFrame(rows))
    decisions.forget()
    ids = {(r["x_core"], r["y_core"]): r["candidate_id"] for r in rows}
    name = admin.post("/api/labels/sessions", json={"stain": "6E10", "round": 2, "open": True, "tile_ids": [TILE]}).json()["name"]
    ann.get(f"/api/annotate/{name}/fields/{TILE}")
    records = [{"task": "outline", "candidate": ids[xy], "points": field(square(*xy, 12)), "value": "edited"} for xy in EDITED]  # precise edits
    merged = [[588, 548], [636, 548], [636, 572], [588, 572]]  # one object where detection saw two: the medium masks' union
    meta = {"edit": "merge", "group": "g1"}
    records += [{"task": "class", "x": 612 - ORIGIN, "y": 560 - ORIGIN, "value": "plaque", "subtype": "compact", "meta": meta},
                {"task": "outline", "x": 612 - ORIGIN, "y": 560 - ORIGIN, "points": field(merged), "meta": meta}]
    records += [{"task": "replaced", "candidate": ids[xy], "x": xy[0] - ORIGIN, "y": xy[1] - ORIGIN, "value": "merge",
                 "meta": {"group": "g1", "into": [[612 - ORIGIN, 560 - ORIGIN]]}} for xy in MERGED]
    records.append({"task": "outline", "candidate": ids[PICKED], "points": field(square(*PICKED, 20)), "value": "mask 2"})  # switched to the large mask
    saved = _save(ann, name, TILE, records)
    assert saved.status_code == 200, saved.text

    seen = outline_fixes.summary(project, "6E10", tiles)
    assert (seen["objects"], seen["edited"], seen["merge"], seen["split"]) == (6, 5, 1, 0) and seen["people"] == ["Ann"]
    by = {o["id"]: o for o in seen["options"]}
    assert by["sam_medium"]["iou_median"] == 1.0 and by["sam_medium"]["closest"] == 6 and by["sam_medium"]["area_ratio_median"] == 1.0
    assert by["shown"]["iou_median"] == by["stain"]["iou_median"] == pytest.approx(289 / 625, abs=0.01)  # the stain outline, shown first
    assert by["sam_small"]["iou_median"] < by["sam_large"]["iou_median"] < by["stain"]["iou_median"] and by["sam_large"]["area_ratio_median"] > 2
    assert seen["picked"] == {"sam_large": 1} and seen["default"] == "stain"
    assert seen["hint"]["option"] == "sam_medium" and "SAM medium matches people's fixes better for 6E10 (IoU 1.00 vs 0.46" in seen["hint"]["text"]

    kept = outline_fixes.rows(project, "6E10", tiles)  # each person's outline, for training a better outline
    assert len(kept) == 6 and set(kept.closest) == {"sam_medium"}
    assert kept[kept.kind == "merge"].candidates.iloc[0].split() == [ids[xy] for xy in MERGED]
    assert kept[kept.kind == "edited"].outline.iloc[0] == text(square(*EDITED[0], 12))  # core pixels
    shown = _answer(admin, "/api/models/outline-fixes")["stains"]
    assert [s["stain"] for s in shown] == ["6E10"] and shown[0]["hint"]["option"] == "sam_medium"
    table = pd.read_csv(io.StringIO(admin.get("/api/models/outline-fixes.csv").text))
    assert len(table) == 6 and {"iou_sam_medium", "ratio_shown", "closest"} <= set(table.columns)
    assert ann.get("/api/models/outline-fixes").status_code in (401, 403)  # the admin's page

    # a comparison that takes long (a whole cohort) and no answer kept yet: "preparing" at once, not after seconds
    import threading
    import time

    from stainid import disk_cache
    from stainid.api import work

    disk_cache.answer_file(project.root, "outline-fixes.json").unlink(missing_ok=True)
    with work._COND:
        work._BY_KEY.clear()
        work._BY_ID.clear()
    release, real = threading.Event(), outline_fixes.summary
    monkeypatch.setattr(outline_fixes, "summary", lambda *args: release.wait(20) and real(*args))
    started = time.monotonic()
    assert admin.get("/api/models/outline-fixes").status_code == 202 and time.monotonic() - started < 2
    release.set()


def _answer(client, path: str) -> dict:
    """The page's answer once its background work is done (202 "preparing" until then)."""
    import time

    answer = client.get(path)
    deadline = time.monotonic() + 30
    while answer.status_code == 202 and time.monotonic() < deadline:
        time.sleep(0.1)
        answer = client.get(path)
    return answer.json()
