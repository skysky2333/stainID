"""Stage 2 keeps every object exactly once, whichever tiles it lies across: a plaque across the edge of two tiles, one on
the corner of four, small objects near a tile edge, and a pale diffuse plaque that only specks of candidates touch is
reported as unproposed."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import yaml

from stainid import candidates as store
from stainid.project import load_project
from stainid.v2 import proposals, run
from stainid.v2.field import Field

PS = 0.2738
BROWN = (110, 70, 35)
TILE = 400


def border_project(tmp_path: Path, draw, size: int = 2 * TILE, context: int = 32, stain: str = "6E10") -> Path:
    """One core image of `size` px, cut into 400 px tiles, with whatever `draw(rgb)` paints on a pale background."""
    core_dir = tmp_path / "data" / "cores" / "LIP-1" / "B-2"
    core_dir.mkdir(parents=True)
    (tmp_path / "data" / "analysis" / "tile_nuclei").mkdir(parents=True)
    rgb = np.full((size, size, 3), 225, np.uint8)
    draw(rgb)
    cv2.imwrite(str(core_dir / f"{stain}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    tiles = []
    for row, y in enumerate(range(0, size, TILE)):
        for col, x in enumerate(range(0, size, TILE)):
            tiles.append({"tile_id": f"LIP-1_B-2_{stain}_W{row:02d}{col:02d}", "core_id": "LIP-1_B-2", "tma": 1, "core_label": "B-2",
                          "donor_id": "0001", "sample_region_id": "BRC_0001_FR", "region": "frontal", "disease_group": "AD",
                          "technical_replicate": 1, "stain": stain, "image_path": f"data/cores/LIP-1/B-2/{stain}.png",
                          "pixel_width_um": PS, "pixel_height_um": PS, "selection_role": "whole_core_grid", "selection_order": len(tiles),
                          "nested_sample": "whole_core", "x_px": x, "y_px": y, "width_px": TILE, "height_px": TILE, "preview_tissue_fraction": 0.9})
    analysis = tmp_path / "data" / "analysis"
    pd.DataFrame(tiles).to_csv(analysis / "cohort_whole_core_tiles.csv", index=False)
    pd.DataFrame([{"tma": 1, "stain": stain, "threshold_dab_od": 0.05}]).to_csv(analysis / "slide_dab_calibration.csv", index=False)
    (tmp_path / "data" / "annotations").mkdir()
    (tmp_path / "data" / "annotations" / "cohort_manual_exclusions.json").write_text(json.dumps({}))
    (tmp_path / "stainid.yaml").write_text(yaml.safe_dump({"name": "borders", "field_context_px": context}))
    return tmp_path


def propose(root: Path, stain: str = "6E10") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    project = load_project(root)
    calibration, manual, nuclei = run.proposal_inputs(project)
    for core, s, tiles, _ in run.todo(project, "proposals", [stain]):
        proposals.propose_core(project, s, core, tiles, calibration[(tiles[0]["tma"], s)], [], nuclei)
    read = lambda kind: store.read_table(store.path(project, stain, "LIP-1_B-2", kind))  # noqa: E731
    return read("proposals"), read("tiles"), read("unproposed")


def covering(frame: pd.DataFrame, x: float, y: float) -> pd.DataFrame:
    """The candidates whose outline holds the point."""
    hits = []
    for row in frame.itertuples():
        flat = np.asarray(store.points(row.outline), np.float32).reshape(-1, 1, 2)
        hits.append(len(flat) >= 3 and cv2.pointPolygonTest(flat, (float(x), float(y)), False) >= 0)
    return frame[np.array(hits, bool)] if len(frame) else frame


def test_plaque_across_two_tiles_is_kept_once_and_whole(tmp_path):
    plaque = (TILE, 200, 60)  # centred on the vertical edge between W0000 and W0001, wider than the 32 px context
    small = (TILE - 20, 600, 10)  # a small deposit near the edge, whole in both tiles' crops
    root = border_project(tmp_path, lambda rgb: [cv2.circle(rgb, (x, y), r, BROWN, -1) for x, y, r in (plaque, small)])
    frame, tiles, _ = propose(root)
    for x, y, r in (plaque, small):
        hits = covering(frame, x, y)
        assert len(hits) == 1, (x, y, hits[["tile_id", "x_core", "y_core", "area_um2"]])
        hit = hits.iloc[0]
        t = tiles.set_index("tile_id").loc[hit.tile_id]  # it belongs to the tile holding its centroid
        assert t.x_px <= hit.x_core < t.x_px + t.width_px and t.y_px <= hit.y_core < t.y_px + t.height_px
        assert hit.x_tile == pytest.approx(hit.x_core - t.x_px) and hit.y_tile == pytest.approx(hit.y_core - t.y_px)
    big = covering(frame, plaque[0], plaque[1]).iloc[0]
    # the kept copy is the more complete one: well over half of the plaque (each crop holds at most 60 + 32 of its 120 px)
    assert big.area_um2 > 0.6 * np.pi * plaque[2] ** 2 * PS ** 2
    assert int(tiles.candidate_count.sum()) == len(frame)


def test_plaque_on_the_corner_of_four_tiles_is_kept_once(tmp_path):
    plaque = (TILE, TILE, 50)
    root = border_project(tmp_path, lambda rgb: cv2.circle(rgb, plaque[:2], plaque[2], BROWN, -1))
    frame, _, _ = propose(root)
    hits = covering(frame, TILE, TILE)
    assert len(hits) == 1
    near = frame[np.hypot(frame.x_core - TILE, frame.y_core - TILE) < plaque[2]]
    assert len(near) == 1  # no second copy anywhere on the plaque


def test_objects_in_the_middle_of_tiles_do_not_change(tmp_path):
    spots = [(120, 150, 14), (260, 300, 20), (600, 120, 16), (650, 650, 25)]
    root = border_project(tmp_path, lambda rgb: [cv2.circle(rgb, (x, y), r, BROWN, -1) for x, y, r in spots])
    frame, _, _ = propose(root)
    for x, y, _ in spots:
        assert len(covering(frame, x, y)) == 1
    assert len(frame) == len(spots)


def test_owned_once_rules():
    def row(tile, x, y, r, own=True, cut=False, area=None):
        pts = cv2.ellipse2Poly((x, y), (r, r), 0, 0, 360, 10)
        return {"tile_id": tile, "x_core": float(x), "y_core": float(y), "bbox": f"{x - r} {y - r} {x + r + 1} {y + r + 1}",
                "outline": store.encode_points(pts), "touches_crop_edge": cut, "_own": own, "_area_px": area or int(np.pi * r * r),
                "x_tile": 0.0, "y_tile": 0.0, "x_field": 0.0, "y_field": 0.0}
    tiles = [{"tile_id": "A", "x_px": 0, "y_px": 0, "width_px": 100, "height_px": 100},
             {"tile_id": "B", "x_px": 100, "y_px": 0, "width_px": 100, "height_px": 100}]
    # two cut copies of one object: the larger stays, moved to the tile holding its centroid
    out = proposals.owned_once([row("A", 103, 50, 30, own=False, cut=True, area=2000), row("B", 98, 50, 30, own=False, cut=True, area=1500)], tiles, 32)
    assert len(out) == 1 and out[0]["tile_id"] == "B" and out[0]["x_tile"] == pytest.approx(3.0) and "_own" not in out[0]
    # a whole copy beats a cut one, whatever their areas
    out = proposals.owned_once([row("A", 95, 50, 20, cut=True, area=5000), row("B", 96, 50, 20, own=False, cut=False, area=1000)], tiles, 32)
    assert len(out) == 1 and out[0]["touches_crop_edge"] is False
    # two whole objects in their own tiles are kept even when their outlines touch
    out = proposals.owned_once([row("A", 90, 50, 12), row("B", 108, 50, 12)], tiles, 32)
    assert len(out) == 2
    # a copy whose centroid lies in no tile of the image is dropped
    assert proposals.owned_once([row("B", 205, 50, 10, own=False)], tiles, 32) == []


def test_pale_diffuse_plaque_touched_by_specks_is_unproposed():
    rng = np.random.default_rng(0)
    rgb = np.full((400, 400, 3), 225, np.uint8)
    yy, xx = np.mgrid[:400, :400]
    halo = (xx - 200) ** 2 + (yy - 200) ** 2 <= 110 ** 2
    rgb[halo] = (195, 170, 150)  # a pale brown halo (DAB 0.046): below the slide threshold, clearly stained over 4 µm
    for x, y in rng.integers(120, 280, size=(6, 2)):
        cv2.circle(rgb, (int(x), int(y)), 4, BROWN, -1)  # dark specks in it, each proposed on its own
    tile = {"tile_id": "T_W0000", "x_px": 0, "y_px": 0, "width_px": 400, "height_px": 400, "pixel_width_um": PS, "pixel_height_um": PS,
            "context_px": 0}
    field = Field(tile, rgb, (slice(0, 400), slice(0, 400)), (0, 0), 0.05, PS, np.zeros((400, 400), bool), None)
    owner = np.zeros((400, 400), np.int32)
    for k, (x, y) in enumerate(rng.integers(120, 280, size=(6, 2)), start=1):
        cv2.circle(owner, (int(x), int(y)), 5, k, -1)
    rule = {"smooth_um": 4.0, "level": 0.75, "contrast": 0.25, "min_um2": 100.0, "min_cover": 0.25}
    found = proposals.unproposed_regions(field, "6E10", owner, rule, None, np.zeros((400, 400), bool))
    assert len(found) == 1 and found[0]["area_um2"] > 0.5 * np.pi * 110 ** 2 * PS ** 2
    # the old rule (DAB at the threshold, touching no candidate) misses it
    assert proposals.unproposed_regions(field, "6E10", owner, 20.0, None, np.zeros((400, 400), bool)) == []
    # covered by one candidate, it is proposed
    owner[halo] = 1
    assert proposals.unproposed_regions(field, "6E10", owner, rule, None, np.zeros((400, 400), bool)) == []


def test_segment_anything_output_shrunk_onto_a_speck_is_not_the_default():
    from stainid.v2.outlines import Output, choose

    def out(area_px):
        return Output(area_px, False, (0, 0, 1, 1), np.ones((1, 1), bool), 0.9, 0.9)

    candidate_px = 4000 / PS ** 2  # a 4000 µm² pale plaque
    dab = np.zeros((10, 10), np.float32)
    speck, part, whole = out(round(45 / PS ** 2)), out(round(600 / PS ** 2)), out(round(3800 / PS ** 2))
    # without the guard the medium output (6E10's rule) is the default even when it is one speck of the plaque
    assert choose([speck, speck, whole], candidate_px, "6E10", PS, dab, 5.0) == (1, "ok")
    # with it, a valid larger output is taken, and with none the candidate keeps its own outline
    assert choose([speck, speck, whole], candidate_px, "6E10", PS, dab, 5.0, 0.25, 100.0) == (2, "ok")
    assert choose([speck, part, part], candidate_px, "6E10", PS, dab, 5.0, 0.25, 100.0) == (-1, "fallback_shrunk")
    # small candidates are not checked: a 60 µm² deposit may well be outlined by a quarter of its area
    assert choose([out(round(10 / PS ** 2)), out(round(14 / PS ** 2) + 50), out(round(20 / PS ** 2))], 60 / PS ** 2, "6E10", PS, dab,
                  5.0, 0.25, 100.0) == (1, "ok")


def test_pale_diffuse_plaque_is_proposed_whole(tmp_path):
    """A pale halo below the threshold with dark specks in it: the specks are proposed, and so is the plaque, as one."""
    rng = np.random.default_rng(1)

    def draw(rgb):
        cv2.circle(rgb, (200, 200), 90, (195, 170, 150), -1)  # DAB 0.046 against a 0.05 threshold
        for x, y in rng.integers(150, 250, size=(5, 2)):
            cv2.circle(rgb, (int(x), int(y)), 4, BROWN, -1)
    frame, _, unproposed = propose(border_project(tmp_path, draw))
    whole = frame[(frame.area_um2 > 0.6 * np.pi * 90 ** 2 * PS ** 2)]
    assert len(covering(whole, 200, 200)) == 1 and "coarse" in whole.sources.iloc[0]
    assert len(unproposed) == 0


def test_a_faint_node_that_merges_kept_deposits_gives_way_to_its_children():
    from stainid.v2.tree import walk_roots

    rgb = np.full((300, 300, 3), 225, np.uint8)
    cv2.rectangle(rgb, (40, 120), (260, 180), (205, 190, 175), -1)  # a faint band (0.031) joining three deposits
    for x in (70, 150, 230):
        cv2.circle(rgb, (x, 150), 20, (190, 150, 120), -1)  # 0.069: above the threshold
    field = Field({"tile_id": "T", "x_px": 0, "y_px": 0, "width_px": 300, "height_px": 300, "pixel_width_um": PS, "pixel_height_um": PS},
                  rgb, (slice(0, 300), slice(0, 300)), (0, 0), 0.05, PS, np.zeros((300, 300), bool), None)
    offered = []

    def keep(region):
        offered.append(region)
        return region.level > 0.5  # the band (level 0.5) is a merge of kept deposits: it gives way

    walk_roots(field, [0.5, 1.0], np.ones((300, 300), bool), 20.0, 50000.0, 20.0, keep)
    assert [r.level for r in offered] == [0.5, 1.0, 1.0, 1.0]
