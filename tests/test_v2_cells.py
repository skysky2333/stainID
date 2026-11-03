"""Stage 2's per-nucleus table and the nuclei every count takes (`stainid.v2.cells`), and stage 5's NeuN cell columns."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from stainid.nuclei import nucleus_centroids
from stainid.v2 import cells
from stainid.v2.measures import MIN_CELLS, cell_columns


def _field(stain_dab: float = 0.5):
    """A 200 x 200 field (tile = its middle 100 x 100, origin (1000, 2000) core px, 0.5 µm pixels) with four nuclei: a
    hematoxylin nucleus, a DAB-dense one (a plaque core or a tangle), one too small, one centred outside the tile."""
    labels = np.zeros((200, 200), np.uint32)
    hematoxylin = np.full((200, 200), 0.02, np.float32)
    dab = np.full((200, 200), 0.01, np.float32)
    labels[60:70, 60:72] = 1  # 120 px = 30 µm²
    hematoxylin[60:70, 60:72] = 0.2
    labels[100:112, 100:110] = 2  # DAB-dense
    hematoxylin[100:112, 100:110] = 0.1
    dab[100:112, 100:110] = stain_dab
    labels[80:83, 80:83] = 3  # 9 px = 2.25 µm²: no cell
    labels[20:30, 20:30] = 4  # 100 px = 25 µm², centred outside the tile
    hematoxylin[20:30, 20:30] = 0.2
    return SimpleNamespace(nuclei=labels, hematoxylin=hematoxylin, dab=dab, pixel_size_um=0.5, memo={},
                           inner=(slice(50, 150), slice(50, 150)), origin=(1000, 2000), tile={"tile_id": "T1"})


def test_dense_dab_nuclei_are_no_cells_on_6e10_and_at8_and_counts_match_the_old_rule_otherwise():
    field = _field()
    valid = np.ones((200, 200), bool)
    old = {tuple(p) for p in nucleus_centroids(field.nuclei, valid, field.pixel_size_um)}
    assert {tuple(p) for p in cells.centroids(field, valid, "NeuN")} == old  # NeuN: the size rule only, as before
    for stain in ("6E10", "AT8"):
        assert {tuple(p) for p in cells.centroids(field, valid, stain)} == old - {(104, 105)}  # the DAB-dense one goes
    assert {tuple(p) for p in cells.centroids(_field(0.05), valid, "6E10")} == old  # more hematoxylin than DAB: a cell
    valid[60:70, 60:72] = False
    assert (65, 64) not in {tuple(p) for p in cells.centroids(field, valid, "NeuN")}  # centre outside the measured tissue


def test_the_tile_table_holds_the_tiles_own_cells_with_shape_stain_and_whether_they_count():
    field = _field()
    table = cells.tile_cells(field, np.ones((200, 200), bool), "AT8")
    assert list(table.columns) == cells.COLUMNS and len(table) == 2  # the small one and the one outside the tile are not there
    one = table.set_index("x_core").loc[1065]
    assert (one.y_core, one.area_um2, bool(one.counted)) == (2064, pytest.approx(30.0), True)
    assert one.hematoxylin_mean == pytest.approx(0.2) and 0 < one.eccentricity < 1 and one.solidity == pytest.approx(1.0)
    assert not bool(table.set_index("x_core").loc[1104].counted)
    assert cells.tile_cells(field, np.ones((200, 200), bool), "NeuN").counted.all()


def test_neuronal_cells_and_the_neun_columns():
    table = pd.DataFrame({"tile_id": ["A", "A", "A", "B"], "x_core": [10, 50, 90, 10], "y_core": [10, 10, 10, 10],
                          "counted": [True, True, False, True]})
    square = [[0, 0, 60, 0, 60, 60, 0, 60]]
    inside = cells.in_outlines(table, [square])
    assert inside.tolist() == [True, True, False, True]
    tiles = pd.DataFrame({"tile_id": ["A", "B", "C"], "nuclei_available": [True, True, False]})
    counted = cells.tile_counts(tiles, table, inside)
    assert counted.cell_count.tolist()[:2] == [2, 1] and np.isnan(counted.cell_count.iloc[2])
    assert counted.nonneuronal_cell_count.tolist()[:2] == [0, 0]
    assert cells.tile_counts(tiles, None, np.zeros(0, bool)).cell_count.isna().all()  # stage 2 ran before the table
    rows = [{"tissue_area_mm2": "0.5", "neun_positive_profile_count": "30", "nonneuronal_cell_count": "70"},
            {"tissue_area_mm2": "0.5", "neun_positive_profile_count": "10", "nonneuronal_cell_count": ""}]  # a tile without nuclei
    out = cell_columns(rows)
    assert out == {"neun_cell_density_mm2": pytest.approx(200.0), "neun_nonneuronal_cell_density_mm2": pytest.approx(140.0),
                   "neun_neuron_fraction_of_cells": pytest.approx(0.3)}
    few = cell_columns([{"tissue_area_mm2": "1", "neun_positive_profile_count": "3", "nonneuronal_cell_count": str(MIN_CELLS - 4)}])
    assert np.isnan(few["neun_neuron_fraction_of_cells"]) and few["neun_cell_density_mm2"] == pytest.approx(MIN_CELLS - 1)
    assert all(np.isnan(v) for v in cell_columns(rows[1:]).values())
