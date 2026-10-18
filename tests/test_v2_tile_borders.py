"""Stage 2 keeps every object exactly once, whichever tiles it lies across: a plaque across the edge of two tiles, one on
the corner of four, small objects near a tile edge, and a pale diffuse plaque that only specks of candidates touch is
reported as unproposed."""
from __future__ import annotations

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
    (tmp_path / "stainid.yaml").write_text(yaml.safe_dump({"name": "borders", "field_context_px": context}))
    return tmp_path


def propose(root: Path, stain: str = "6E10") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    project = load_project(root)
    calibration, nuclei = run.proposal_inputs(project)
    for core, s, tiles, _ in run.todo(project, "proposals", [stain]):
        proposals.propose_core(project, s, core, tiles, calibration[(tiles[0]["tma"], s)], nuclei)
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
    # one object two tiles segmented slightly differently, whole in both, its centroids either side of the border: once
    out = proposals.owned_once([row("A", 98, 50, 30, area=2800), row("B", 102, 50, 28, area=2500)], tiles, 32)
    assert len(out) == 1 and out[0]["tile_id"] == "A"
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


def _with_window(root: Path, stain: str, context_px: int) -> None:
    config = yaml.safe_load((root / "stainid.yaml").read_text())
    config["v2"] = {"proposals": {stain: {"context_px": context_px}}}
    (root / "stainid.yaml").write_text(yaml.safe_dump(config))


def _corner_plaque(rgb):
    cv2.circle(rgb, (TILE, TILE), 150, BROWN, -1)  # 300 px across, centred where four tiles meet (tiles' context: 32 px)
    cv2.ellipse(rgb, (TILE + 60, TILE - 40), (170, 60), 30, 0, 360, BROWN, -1)  # an irregular lobe across the same corner


def test_a_plaque_far_wider_than_the_tile_context_on_a_four_tile_corner_is_detected_whole_once(tmp_path):
    painted = np.zeros((2 * TILE, 2 * TILE, 3), np.uint8)
    _corner_plaque(painted)
    area = painted.any(axis=2).sum() * PS ** 2
    root = border_project(tmp_path / "whole", _corner_plaque)
    frame, _, _ = propose(root)  # the 6E10 detection window (proposals.6E10.context_px) reaches past the plaque
    hits = covering(frame, TILE, TILE)
    assert len(hits) == 1
    hit = hits.iloc[0]
    assert not bool(hit.touches_crop_edge)
    assert hit.area_um2 >= 0.95 * area  # all of it (with its rim): no straight crop edge
    assert len(frame[np.hypot(frame.x_core - TILE, frame.y_core - TILE) < 200]) == 1  # and no other piece of it

    root = border_project(tmp_path / "narrow", _corner_plaque)  # a window only as wide as the tiles' context cuts it
    _with_window(root, "6E10", 32)
    frame, _, _ = propose(root)
    cut = covering(frame, TILE, TILE)
    assert len(cut) == 0 or bool(cut.iloc[0].touches_crop_edge) or cut.iloc[0].area_um2 < 0.9 * area


def test_window_nuclei_keep_the_tiles_own_nuclei_and_add_the_neighbours():
    size = (800, 800)
    tiles = [{"tile_id": f"T{r}{c}", "x_px": c * TILE, "y_px": r * TILE, "width_px": TILE, "height_px": TILE} for r in (0, 1) for c in (0, 1)]
    context = 32
    labels = {}
    for t in tiles:  # each tile's Cellpose labels over its standard crop: one nucleus in the middle of its own area
        x0, y0, x1, y1 = proposals._crop_box(t, context, size)
        lab = np.zeros((y1 - y0, x1 - x0), np.uint32)
        cx, cy = t["x_px"] + 200 - x0, t["y_px"] + 200 - y0
        lab[cy - 5: cy + 5, cx - 5: cx + 5] = 1
        labels[t["tile_id"]] = lab
    out = proposals.window_nuclei(tiles[0], tiles, (0, 0, 800, 800), context, size, lambda t: labels[t["tile_id"]])
    centres = set()
    for value in np.unique(out[out > 0]):
        ys, xs = np.nonzero(out == value)
        centres.add((int(round(xs.mean())), int(round(ys.mean()))))
    assert centres == {(200, 200), (600, 200), (200, 600), (600, 600)}  # four nuclei, four labels
    assert out[200, 200] == 1  # the tile's own label stands


def _level_fields(wide_context: int, stain: str = "6E10"):
    """A 300 px tile with a 50 px standard margin, cut `wide_context` px wide, where the tissue beyond the standard crop is
    browner: the field as the wide window cuts it, the standard crop of the same tile, and the wide field with its levels
    taken over all of it (as before)."""
    rng = np.random.default_rng(3)
    size = 300 + 2 * wide_context
    rgb = np.clip(rng.normal((200, 190, 195), 6, (size, size, 3)), 0, 255).astype(np.uint8)
    outer = np.ones((size, size), bool)
    outer[wide_context - 50: wide_context + 350, wide_context - 50: wide_context + 350] = False
    rgb[outer] = np.clip(rng.normal((185, 160, 150), 10, (int(outer.sum()), 3)), 0, 255).astype(np.uint8)
    tile = {"tile_id": "T", "stain": stain, "pixel_width_um": PS, "pixel_height_um": PS, "context_px": 50}
    inner = (slice(wide_context, wide_context + 300), slice(wide_context, wide_context + 300))
    a, b = wide_context - 50, wide_context + 350
    wide = Field(tile, rgb, inner, (0, 0), 0.05, PS, np.zeros((size, size), bool), None)
    standard = Field(tile, rgb[a:b, a:b].copy(), (slice(50, 350), slice(50, 350)), (0, 0), 0.05, PS, np.zeros((400, 400), bool), None)
    unscoped = Field({**tile, "context_px": wide_context}, rgb, inner, (0, 0), 0.05, PS, np.zeros((size, size), bool), None)
    return wide, standard, unscoped


def test_a_wider_window_moves_no_level_of_the_tile():
    """The crop-wide levels (6E10 background median and spread, so its seed and extent levels; the stained level; the AT8
    positive level) come from the standard crop: seeing further does not lower the extent level and join neighbouring
    plaques that the standard crop keeps apart."""
    from stainid.stains.tau.network import segment_at8_network
    from stainid.v2.amyloid import _levels
    from stainid.v2.tree import stained_level

    wide, standard, unscoped = _level_fields(250)
    got, want, moved = _levels(wide), _levels(standard), _levels(unscoped)
    for key in ("background_median", "background_sigma", "seed_threshold", "extent_threshold"):
        assert got[key] == pytest.approx(want[key], rel=0.01), key
    assert abs(moved["background_median"] - want["background_median"]) > 0.1 * want["background_median"]  # the test sees a change
    level = lambda f: stained_level(f, f.dab, 0.1, 0.25, f.tissue)  # noqa: E731
    assert level(wide) == pytest.approx(level(standard), rel=0.01)
    assert abs(level(unscoped) - level(standard)) > 0.1 * level(standard)

    def positive(field):  # threshold 0: the positive level is the background median + 5 sigma
        _, _, maps = segment_at8_network(field.rgb, 0.0, PS, PS, None, statistics_mask=field.standard)
        return float(maps["positive"][field.inner].mean())

    assert positive(wide) == pytest.approx(positive(standard), abs=0.005)


def _scattered_plaques(rgb):
    # centred on a corner of four tiles and longer than any tile's first window reaches past it: each crop cuts it,
    # and it is seen whole in a local window
    cv2.ellipse(rgb, (800, 800), (560, 140), 0, 0, 360, BROWN, -1)
    cv2.ellipse(rgb, (1300, 700), (90, 40), 20, 0, 360, (150, 110, 80), -1)  # a paler lobe joined to it
    rng = np.random.default_rng(3)
    for x, y in rng.integers(40, 1560, size=(40, 2)):
        if abs(y - 800) > 200:
            cv2.circle(rgb, (int(x), int(y)), int(rng.integers(6, 25)), BROWN, -1)


def test_objects_cut_by_the_first_window_are_the_whole_windows_objects(tmp_path, monkeypatch):
    """Detecting each tile over a first window and the objects it cuts again in local windows around them gives the
    candidates (outlines and features), tile measures and unproposed regions of detecting each tile over its whole
    window."""
    replaced = []
    original = proposals.whole_objects

    def counting(*args, **kwargs):
        rows, stats = original(*args, **kwargs)
        replaced.append(stats["replaced"])
        return rows, stats

    monkeypatch.setattr(proposals, "whole_objects", counting)
    found = {}
    for whole in (True, False):
        root = border_project(tmp_path / str(whole), _scattered_plaques, size=4 * TILE)
        config = yaml.safe_load((root / "stainid.yaml").read_text())
        config["v2"] = {"proposals": {"whole_objects": whole}}
        (root / "stainid.yaml").write_text(yaml.safe_dump(config))
        found[whole] = propose(root)
    assert sum(replaced) >= 4  # the test sees local windows: every tile around the corner cut the plaque
    for got, want in zip(found[True], found[False]):
        assert len(got) == len(want)
        got, want = got.reset_index(drop=True), want.reset_index(drop=True)
        for column in want.columns:
            if pd.api.types.is_numeric_dtype(want[column]):
                assert np.allclose(got[column].astype(float), want[column].astype(float), rtol=1e-6, equal_nan=True), column
            else:
                assert (got[column].astype(str) == want[column].astype(str)).all(), column
    big = covering(found[True][0], 800, 800)
    assert len(big) == 1 and big.iloc[0].area_um2 >= 0.95 * np.pi * 560 * 140 * PS ** 2
