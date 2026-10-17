"""The Cores view's outline choice (`store_view.outline_rows` and /api/cores/{core}/{stain}/outlines?outline=…): the
stain's default, the candidate's own outline, or one of the three Segment Anything sizes (falling back to the own outline)."""
from __future__ import annotations

import pandas as pd
import pytest

from stainid import candidates as store
from stainid.api import store_view
from stainid.project import load_project

CORE = "LIP-1_A-1"


@pytest.fixture()
def project(tmp_path):
    project = load_project(tmp_path)
    frame = pd.DataFrame({
        "candidate_id": [f"AT8:{CORE}:10:10", f"AT8:{CORE}:50:50"], "x_core": [10.0, 50.0], "y_core": [10.0, 50.0], "radius_px": [5.0, 5.0],
        "area_um2": [40.0, 30.0], "probability": [0.9, 0.2], "decision": ["accepted", "rejected"], "reason": ["accepted", "rejected_by_model"],
        "class": ["tau+ neuron", ""], "subtype": ["", ""], "m_from_ring": [0, 0],
        "outline": ["0 0 20 0 20 20", "40 40 60 40 60 60"],
        "sam_outline": ["1 1 19 1 19 19", ""], "sam_status": ["ok", ""],
        "sam_masks": ["2 2 4 2 4 4|1 1 19 1 19 19|0 0 40 0 40 40", ""],
    })
    store.write_table(store.root(project) / "candidates" / "AT8" / f"{CORE}.csv.gz", frame)
    return project


def rings_of(project, choice):
    return [row[3] for row in store_view.outline_rows(project, CORE, "AT8", choice)]


def test_outline_choices(project):
    own = [[0, 0, 20, 0, 20, 20], [40, 40, 60, 40, 60, 60]]
    assert rings_of(project, "default") == [[[1, 1, 19, 1, 19, 19]], [own[1]]]  # SAM where kept, else the own outline
    assert rings_of(project, "stain") == [[own[0]], [own[1]]]
    assert rings_of(project, "sam0") == [[[2, 2, 4, 2, 4, 4]], [own[1]]]  # no masks: the own outline
    assert rings_of(project, "sam2") == [[[0, 0, 40, 0, 40, 40]], [own[1]]]
    small, large = store_view.outline_rows(project, CORE, "AT8", "sam0")[0], store_view.outline_rows(project, CORE, "AT8", "sam2")[0]
    assert small[2] < large[2]  # each size its own area
    assert store_view.outline_rows(project, CORE, "AT8", "stain")[0][2] == 40.0
    with pytest.raises(ValueError):
        store_view.outline_rows(project, CORE, "AT8", "sam3")
