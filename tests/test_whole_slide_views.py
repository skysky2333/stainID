"""Whole slides put back together for viewing (imageset/slides.py, api/slide_pyramid.py, api/slide_units.py,
api/routes/whole_slides.py): the regions stay the units of the pipeline and of every label, and the Cores view and the
annotation pages show a slide as one picture with everything at its place. A TMA study sees none of it."""
from __future__ import annotations

import base64

import cv2
import numpy as np
import pandas as pd
import pytest
import yaml

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402
from people import annotator  # noqa: E402

from stainid import exclusions  # noqa: E402
from stainid.api import slide_pyramid, slide_units, state  # noqa: E402
from stainid.api.app import create_app  # noqa: E402
from stainid.imageset.slides import CELL, Region, Slide, layouts, split_mask  # noqa: E402
from stainid.labels import areas, matter, store  # noqa: E402

SLIDE = "S1"
FIRST, SECOND = "S1_R01", "S1_R02"
# two regions of one slide (level-0 px of the scan = slide px here, scale 1): one in the first cell of the grid, one in
# the cell below and to the right of it
BOXES = {FIRST: (100, 50, 300, 200), SECOND: (CELL + 56, CELL + 36, 250, 150)}
COLOURS = {FIRST: (40, 80, 160), SECOND: (160, 60, 30)}  # BGR


def _table(boxes=BOXES, image_id=SLIDE, stain="6E10", scale=1.0) -> pd.DataFrame:
    return pd.DataFrame([{"core_id": c, "image_id": image_id, "stain": stain, "tma": image_id, "donor_id": "D1", "region": "MFG",
                          "source_box": " ".join(str(v) for v in box), "scale": scale, "native_width_px": round(box[2] * scale),
                          "native_height_px": round(box[3] * scale), "image_path": f"data/images/{c}.png"} for c, box in boxes.items()])


def test_a_slide_is_laid_out_from_its_regions_and_points_map_both_ways():
    (slide,) = layouts(_table()).values()
    assert slide.slide_id == SLIDE and slide.origin == (0, 0) and slide.cores == [FIRST, SECOND]
    first, second = slide.regions
    assert (first.x, first.y, first.width, first.height) == (100, 50, 300, 200)
    assert (second.x, second.y) == (CELL + 56, CELL + 36) and (slide.width, slide.height) == (CELL + 306, CELL + 186)
    assert slide.to_slide(SECOND, 10, 20) == (CELL + 66, CELL + 56)
    assert slide.to_region(CELL + 66, CELL + 56) == (SECOND, 10, 20)
    assert slide.to_region(5000, 5000) is None  # glass between the regions
    assert [r.core_id for r in slide.overlapping(0, 0, 200, 200)] == [FIRST]
    # a scan read at half size, its regions starting in the second cell: the origin is that cell's corner
    (half,) = layouts(_table({FIRST: (2 * CELL + 400, 600, 600, 400)}, scale=0.5)).values()
    assert half.origin == (CELL, 0) and (half.regions[0].x, half.regions[0].y) == (200, 300)
    assert layouts(pd.DataFrame([{"core_id": "LIP-1_B-2", "stain": "6E10", "tma": 1}])) == {}  # a TMA study: no slides


def _painted(shape, box) -> str:
    rgba = np.zeros((*shape, 4), np.uint8)
    r0, r1, c0, c1 = box
    rgba[r0:r1, c0:c1] = 255
    return "data:image/png;base64," + base64.b64encode(cv2.imencode(".png", rgba)[1].tobytes()).decode()


def test_what_is_painted_over_a_slide_is_saved_as_each_region_s_own_at_its_place():
    # two regions side by side: the second starts 100 px into its cell (not on the mask's grid)
    slide = Slide("S", "6E10", "S", "D", "MFG", "S", (0, 0), CELL + 3100, CELL,
                  (Region("A", 0, 0, CELL, CELL), Region("B", CELL + 100, 0, 3000, CELL)))
    scale = slide_units.mask_scale(slide)
    assert scale == exclusions.SCALE and slide_units.mask_scale(Slide("W", "6E10", "W", "", "", "", (0, 0), 109_403, 78_375, ())) == 64
    shape = (-(-slide.height // scale), -(-slide.width // scale))
    # painted over slide px x 5600–7200, y 800–1600 (mask cells 700–900 across, 100–200 down): across both regions
    parts = slide_units.split_pngs(slide, {"2": _painted(shape, (100, 200, 700, 900))}, scale)
    assert set(parts) == {"A", "B"} and parts["A"][2] == ["2"] * len(parts["A"][0])
    a = exclusions.rasterize(parts["A"][0], parts["A"][1], (CELL // 8, CELL // 8))
    b = exclusions.rasterize(parts["B"][0], parts["B"][1], (CELL // 8, 3000 // 8))
    assert a[150, 710] and a[150, 767] and not a[150, 690] and not a[90, 710]  # A: from x 5600 to its edge
    xs = [p[0] for ring in parts["B"][0] for p in ring]
    assert min(xs) == 0 and max(xs) == pytest.approx(7200 - CELL - 100 - 4, abs=8)  # B: from its edge (the 100 px of glass are no one's)
    assert b[150, 10] and b[150, 115] and not b[150, 130]
    whole = slide_units.split_png(slide, _painted(shape, (100, 200, 0, 10)), scale)  # white matter: every region has its whole say
    assert whole["A"][0] and whole["B"] == ([], [])
    cut = split_mask(np.ones(shape, bool), scale, slide)
    assert cut["B"][1] == (CELL + 100) // scale * scale - (CELL + 100) and cut["A"][0].shape == (CELL // 8, CELL // 8)


def test_codes_of_the_regions_are_put_on_one_grid_of_the_slide():
    slide = Slide("S", "6E10", "S", "", "", "", (0, 0), 2 * CELL, CELL, (Region("A", 0, 0, 64, 64), Region("B", CELL, 0, 64, 32)))
    a, b = np.full((8, 8), 2, np.uint8), np.full((4, 8), 3, np.uint8)
    b[0, 0] = exclusions.OUTSIDE
    out = slide_units.compose(slide, [(slide.regions[0], a, 8), (slide.regions[1], b, 8)], 32)
    assert out.shape == (CELL // 32, 2 * CELL // 32) and out[0, 0] == 2 and out[0, CELL // 32] in (3, exclusions.OUTSIDE)
    assert out[0, CELL // 32 + 1] == 3 and out[10, 10] == exclusions.OUTSIDE


@pytest.fixture()
def slide_project(tmp_path, monkeypatch):
    """A whole-slide project: one 6E10 slide of two regions (each a plain colour with a dark dot), one analysed tile
    each, a stain threshold; an admin."""
    from stainid.api.routes import annotate

    images = tmp_path / "data" / "images"
    images.mkdir(parents=True)
    tiles = []
    for core, (x, y, w, h) in BOXES.items():
        picture = np.full((h, w, 3), COLOURS[core], np.uint8)
        cv2.circle(picture, (w // 2, h // 2), 10, (20, 20, 20), -1)
        cv2.imwrite(str(images / f"{core}.png"), picture)
        tiles.append({"tile_id": f"{core}_6E10_S01", "core_id": core, "tma": SLIDE, "core_label": core, "donor_id": "D1", "sample_region_id": "D1_MFG",
                      "region": "MFG", "disease_group": "", "technical_replicate": 1, "stain": "6E10", "image_path": f"data/images/{core}.png",
                      "pixel_width_um": 0.2738, "pixel_height_um": 0.2738, "selection_role": "whole_core_grid", "selection_order": 1,
                      "nested_sample": "whole_core", "x_px": 0, "y_px": 0, "width_px": 128, "height_px": 96, "preview_tissue_fraction": 1.0})
    analysis = tmp_path / "data" / "analysis"
    analysis.mkdir(parents=True)
    pd.DataFrame(tiles).to_csv(analysis / "cohort_whole_core_tiles.csv", index=False)
    _table().assign(pixel_width_um=0.2738, tissue_status="complete").to_csv(analysis / "core_images.csv", index=False)
    pd.DataFrame([{"tma": SLIDE, "stain": s, "threshold_dab_od": 0.05} for s in ("6E10", "AT8")]).to_csv(analysis / "slide_dab_calibration.csv", index=False)
    (tmp_path / "data" / "annotations").mkdir()
    (tmp_path / "stainid.yaml").write_text(yaml.safe_dump({"name": "slides", "kind": "slides"}))
    monkeypatch.setenv("STAINID_PROJECT", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(annotate, "prepare_job", lambda *args, **kwargs: None)
    state.reset()
    admin = TestClient(create_app(), client=("127.0.0.1", 50000))
    assert admin.post("/api/session/admin", json={"password": "admin"}).status_code == 200
    yield admin
    state.reset()


def _colour(jpeg: bytes, x: int, y: int) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)[y, x].astype(int)


def test_the_slide_picture_is_built_cell_by_cell_resumes_and_serves_the_regions_at_their_places(slide_project):
    s = state.get_state()
    assert slide_pyramid.pending(s) == [SLIDE] and not slide_pyramid.ready(s, SLIDE)
    asked = {"n": 0}

    def stop_after_one() -> bool:
        asked["n"] += 1
        return asked["n"] > 1

    assert slide_pyramid.build(s, SLIDE, stop_after_one) is None  # stopped after the first cell: the work so far is kept
    assert slide_pyramid._paths(s, SLIDE)[2].exists() and not slide_pyramid.ready(s, SLIDE)
    found = slide_pyramid.build(s, SLIDE)
    assert found is not None and slide_pyramid.ready(s, SLIDE) and not slide_pyramid._paths(s, SLIDE)[2].exists()
    assert (found["width"], found["height"]) == (CELL + 306, CELL + 186) and len(found["levels"]) >= 5
    first = slide_pyramid.tile(s, SLIDE, 0, 0, 0)  # the first region at (100, 50)
    assert np.abs(_colour(first, 150, 60) - COLOURS[FIRST]).max() < 25 and _colour(first, 30, 20).min() > 240  # glass around it
    second = slide_pyramid.tile(s, SLIDE, 0, (CELL + 56) // 512, (CELL + 36) // 512)
    assert np.abs(_colour(second, (CELL + 56) % 512 + 30, (CELL + 36) % 512 + 20) - COLOURS[SECOND]).max() < 25
    crop = slide_pyramid.crop(s, SLIDE, (0, 0, found["width"], found["height"]), 256)
    assert max(crop.shape[:2]) <= 256
    assert slide_pyramid.thumbnail(s, SLIDE, 512).exists() and not slide_pyramid.made(s, SLIDE)  # the 2048 px one not yet
    for size in slide_pyramid.THUMBS:
        slide_pyramid.thumbnail(s, SLIDE, size)
    assert slide_pyramid.made(s, SLIDE) and slide_pyramid.pending(s) == []
    assert slide_pyramid.build(s, SLIDE)["version"] == found["version"]  # built once


def test_idle_time_makes_each_slide_picture(slide_project):
    from stainid.api import ahead

    s = state.get_state()
    units, _ = ahead.plan(s)
    (unit,) = [u for u in units if u.kind == "slide"]
    assert unit.tier == 3 and unit.image_path == SLIDE and not ahead.done(unit)
    ahead.make(unit, lambda: False)
    assert ahead.done(unit) and slide_pyramid.made(s, SLIDE)


def test_the_cores_view_shows_the_slide_with_the_regions_drawings_in_slide_px(slide_project, monkeypatch):
    from stainid.api.routes import cohort

    admin = slide_project
    listed = admin.get("/api/whole-slides").json()
    assert listed["unit"] == "slide" and [(s["slide_id"], s["regions"], s["tiles"]) for s in listed["slides"]] == [(SLIDE, 2, 2)]
    info = admin.get(f"/api/whole-slides/{SLIDE}").json()
    assert info["unit"] == "slide" and info["core_label"] == SLIDE and info["images"][0]["native_width_px"] == CELL + 306
    assert [r[0] for r in info["regions"]] == [FIRST, SECOND]
    region = admin.get(f"/api/cores/{SECOND}").json()  # a region opens its slide, at its place
    assert region["slide"] == {"slide_id": SLIDE, "stain": "6E10", "x": CELL + 56, "y": CELL + 36, "width": 250, "height": 150}
    assert admin.get(f"/api/whole-slides/of/{SECOND}?x=10&y=20").json()["x"] == CELL + 66
    slide_pyramid.build(state.get_state(), SLIDE)
    picture = admin.get(f"/api/whole-slides/{SLIDE}/6E10/picture").json()
    assert picture["ready"] and picture["width"] == CELL + 306
    tile = admin.get(f"/api/images/whole-slides/{SLIDE}/6E10/tiles/0/0_0.jpg?v={picture['version']}")
    assert tile.status_code == 200 and tile.headers["content-type"] == "image/jpeg"
    dab = admin.get(f"/api/images/whole-slides/{SLIDE}/6E10/tiles/0/0_0_dab.png")
    assert dab.status_code == 200, dab.text
    assert admin.get(f"/api/images/whole-slides/{SLIDE}/6E10.jpg?size=256").status_code == 200
    assert admin.get(f"/api/images/whole-slides/{SLIDE}/6E10/crop.jpg?x0=0&y0=0&x1=600&y1=400&size=300").status_code == 200
    assert admin.get(f"/api/whole-slides/{SLIDE}/AT8/picture").status_code == 404

    def objects(core_id, stain):  # each region's objects in its own px
        return {"status": "done", "tiles_done": 1, "tiles_total": 1, "columns": ["x", "y", "radius_px"], "rows": [[10.0, 20.0, 5.0]],
                "tiles": [[0, 0, 128, 96]], "outlines": False, "threads": None, "source": "v2", "unproposed": 1, "outline_default": "sam"}

    monkeypatch.setattr(cohort, "core_objects", objects)
    found = admin.get(f"/api/whole-slides/{SLIDE}/6E10/objects").json()
    assert found["status"] == "done" and found["rows"] == [[110.0, 70.0, 5.0], [CELL + 66.0, CELL + 56.0, 5.0]]
    assert found["tiles"] == [[100, 50, 128, 96], [CELL + 56, CELL + 36, 128, 96]] and found["unproposed"] == 2
    status = admin.get(f"/api/whole-slides/{SLIDE}/6E10/status").json()
    assert (status["width"], status["height"]) == (CELL + 306, CELL + 186)
    assert sorted((t["x"], t["y"]) for t in status["tiles"]) == [(100, 50), (CELL + 56, CELL + 36)]
    assert admin.get(f"/api/whole-slides/{SLIDE}/6E10/excluded").json()["status"] == "not_made"
    notes = admin.get(f"/api/whole-slides/{SLIDE}/6E10/annotations").json()
    assert notes["objects"] == [] and notes["areas"]["regions"] == []


def test_the_field_picker_draws_each_slide_as_its_tiles_at_their_places(slide_project):
    from stainid.api.routes import annotate

    fields = [{"tile_id": f"{SECOND}_6E10_S01", "core_id": SECOND, "core": SECOND, "tma": SLIDE, "box": [0, 0, 128, 96]}]
    (placed,) = annotate._on_slides(fields)
    assert placed["core_id"] == SLIDE and placed["region_id"] == SECOND and placed["box"] == [CELL + 56, CELL + 36, 128, 96]
    assert placed["tma_name"] == annotate.SLIDES_GROUP
    layout = annotate._slide_layouts({"name": "x", "stain": "6E10"}, slide_pyramid.slides(state.get_state()), {SLIDE}, state.get_state().tiles,
                                     {placed["tile_id"]})[SLIDE]
    assert (layout["x"], layout["y"]) == (100, 50) and layout["others"] == [[100, 50, 128, 96, ""]]  # the other region's tile, faint
    assert layout["slide"] == [0, 0, CELL + 306, CELL + 186] and layout["picture"].endswith(f"/slides/{SLIDE}.jpg?size=1024")


def test_bad_areas_and_white_matter_are_marked_on_the_whole_slide_and_saved_per_region(slide_project):
    admin = slide_project
    ann = annotator(admin, "Ann", every=True)
    project = state.get_state().project
    meta = admin.post("/api/labels/sessions", json={"stain": "all", "kind": "areas", "open": True, "order": "random", "seed": 1}).json()
    assert meta["cores"] == [SLIDE] and meta["stains_of"] == {SLIDE: ["6E10"]}  # the slide is the unit people check
    name = meta["name"]
    assert ann.get(f"/api/annotate/{name}/areas/next").json()["core_id"] == SLIDE
    unit = ann.get(f"/api/annotate/{name}/cores/{SLIDE}").json()
    (image,) = unit["images"]
    assert unit["unit"] == "slide" and unit["regions"] == 2 and image["mask_scale"] == 8 and image["picture"]["width"] == CELL + 306
    scale = image["mask_scale"]
    shape = (-(-(CELL + 186) // scale), -(-(CELL + 306) // scale))
    # painted "artifact" over slide px 160–240 across, 80–160 down: inside the first region (at 100, 50)
    painted = _painted(shape, (10, 20, 20, 30))
    saved = ann.post(f"/api/annotate/{name}/cores/{SLIDE}/save", json={"checked": ["6E10"], "edits": {"6E10": {"pngs": {"2": painted}, "scale": scale}}})
    assert saved.status_code == 200, saved.text
    labels = store.current(project.output("reviews"))
    (edit,) = labels[labels.task == "exclude"].itertuples()
    assert edit.tile_id == FIRST
    xs, ys = [p[0] for ring in edit.points for p in ring], [p[1] for ring in edit.points for p in ring]
    assert 50 <= min(xs) <= 70 and 130 <= max(xs) <= 150 and 20 <= min(ys) <= 40  # region px: slide px less (100, 50)
    done = labels[(labels.task == "done") & (labels.value == areas.AREAS)]
    assert sorted(done.tile_id) == [FIRST, SECOND]  # every region checked
    assert saved.json()["progress"]["people"] == [{"person": "Ann", "decided": 1}]  # the slide is Ann's done (a second check waits for someone else)
    assert ann.get(f"/api/annotate/{name}/areas/next").json()["core_id"] is None
    people = ann.get(f"/api/annotate/{name}/cores/{SLIDE}/6E10/annotated").json()
    assert people["regions"] and min(p[0] for r in people["regions"] for ring in r["rings"] for p in ring) >= 150  # back in slide px
    assert ann.get(f"/api/annotate/{name}/cores/{SLIDE}/6E10/areas").json()["status"] == "not_made"  # no detection layer yet

    meta = admin.post("/api/labels/sessions", json={"stain": "all", "kind": "matter", "open": True, "seed": 1}).json()
    assert meta["cores"] == [SLIDE] and meta["stains_of"] == {SLIDE: ["6E10"]}
    name = meta["name"]
    opened = ann.get(f"/api/annotate/{name}/matter/cores/{SLIDE}").json()
    assert opened["core"] == SLIDE and opened["images"][0]["mask_scale"] == 8
    assert ann.get(f"/api/annotate/{name}/matter/cores/{SLIDE}/6E10/area").json()["from"] == "new"
    # white matter over the second region's middle (slide px CELL+120 … CELL+200 across)
    cells = (CELL + 36 + 40) // 8, (CELL + 36 + 120) // 8, (CELL + 56 + 64) // 8, (CELL + 56 + 144) // 8
    saved = ann.post(f"/api/annotate/{name}/matter/cores/{SLIDE}/6E10/save", json={"png": _painted(shape, cells), "scale": 8})
    assert saved.status_code == 200, saved.text and saved.json()["finished"]
    labels = store.current(project.output("reviews"))
    assert matter.final(labels, FIRST, "6E10").rings == () and matter.final(labels, SECOND, "6E10").rings  # each region its whole say
    xs = [p[0] for ring in matter.final(labels, SECOND, "6E10").rings for p in ring]
    assert 55 <= min(xs) <= 75 and 135 <= max(xs) <= 155
    back = ann.get(f"/api/annotate/{name}/matter/cores/{SLIDE}/6E10/area").json()
    assert back["from"] == "saved" and back["scale"] == 8 and back["kinds"]["white"]


def test_a_tma_study_sees_no_slides(synthetic_project, monkeypatch):
    from stainid.api import ahead
    from stainid.api.routes import annotate

    monkeypatch.chdir(synthetic_project)
    state.reset()
    admin = TestClient(create_app(), client=("127.0.0.1", 50000))
    assert admin.post("/api/session/admin", json={"password": "admin"}).status_code == 200
    assert admin.get("/api/whole-slides").json() == {"slides": [], "unit": "core"}
    assert admin.get("/api/cores/LIP-1_B-2").json()["slide"] is None
    assert admin.get("/api/whole-slides/LIP-1_B-2").status_code == 404
    fields = [{"tile_id": "LIP-1_B-2_6E10_S01", "core_id": "LIP-1_B-2", "box": [300, 300, 512, 512]}]
    assert annotate._on_slides(fields) == fields
    found = {("LIP-1_B-2", "6E10"): {"Ann"}}
    assert slide_units.with_slides(found) == found and slide_units.unit_stains({"LIP-1_B-2": ["6E10"]}) == {}
    assert slide_units.slide_unit("LIP-1_B-2") is None
    assert not [u for u in ahead.plan(state.get_state())[0] if u.kind == "slide"]
    meta = admin.post("/api/labels/sessions", json={"stain": "all", "kind": "areas", "open": True, "order": "random", "seed": 1}).json()
    assert meta["cores"] == ["LIP-1_B-2"]
    unit = admin.get(f"/api/annotate/{meta['name']}/cores/LIP-1_B-2").json()
    assert unit["unit"] == "core" and {i["mask_scale"] for i in unit["images"]} == {exclusions.SCALE}
    state.reset()


def test_a_field_s_strip_shows_its_whole_slide_and_its_surroundings(slide_project):
    admin = slide_project
    ann = annotator(admin, "Ann", every=True)
    from stainid.labels import candidate_store

    tile = f"{SECOND}_6E10_S01"
    x, y = 60.0, 40.0
    candidate_store.write_core(state.get_state().project, "6E10", SECOND, pd.DataFrame([{
        "candidate_id": candidate_store.candidate_id("6E10", SECOND, x, y), "x_core": x, "y_core": y, "tile_id": tile, "area_um2": 300.0,
        "radius_px": 10.0, "sources": "dab@1", "proposal_threshold": 1.0, "outline": f"{x - 8:.0f} {y:.0f} {x:.0f} {y - 8:.0f} {x + 8:.0f} {y:.0f}",
        "sam_outline": "", "sam_status": "", "probability": 0.9, "model_id": "m@1", "decision": "accepted", "reason": "accepted",
        "class": "plaque", "subtype": "compact", "mean_dab_od": 0.3, "m_context_dab_mean": 1.0}]))
    made = admin.post("/api/labels/sessions", json={"stain": "6E10", "round": 2, "open": True, "tile_ids": [tile]})
    assert made.status_code == 200, made.text
    name = made.json()["name"]
    detail = ann.get(f"/api/annotate/{name}").json()
    assert detail["unit"] == "slide" and detail["fields"][0]["core_id"] == SLIDE and detail["fields"][0]["box"] == [CELL + 56, CELL + 36, 128, 96]
    assert set(detail["cores"]) == {SLIDE}
    before = ann.get(f"/api/annotate/{name}/fields/{tile}/core").json()  # the slide's picture not made yet: its region meanwhile
    assert before["unit"] == "region" and before["core_id"] == SLIDE and before["shift"] == [CELL + 56, CELL + 36]
    slide_pyramid.build(state.get_state(), SLIDE)
    strip = ann.get(f"/api/annotate/{name}/fields/{tile}/core").json()
    assert strip["unit"] == "slide" and (strip["width"], strip["height"]) == (CELL + 306, CELL + 186)
    x0, y0, x1, y1 = strip["mid_box"]
    assert x0 <= CELL + 56 and x1 >= CELL + 56 + 128 and y0 <= CELL + 36 and x1 <= CELL + 306
    assert strip["origin"][0] >= CELL and ann.get(strip["mid"]).status_code == 200
    assert ann.get(strip["thumb"]).status_code == 200
    assert ann.get(f"/api/annotate/{name}/slides/NOPE.jpg").status_code == 404
