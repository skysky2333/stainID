"""What the Cores view draws as counted is what the analysis counts: the counted cells of the "Excluded from analysis"
view (`stainid.exclusions.view`, everything else shaded: tiles not analysed, no tissue, folds, linear artifacts, the
edge guard) add up to the area every measure divides by (stage 2's `analysis_area_mm2` of the core's tiles: tissue for
areas less the excluded areas, which stage 5 takes as each tile's tissue area), per stain."""
from __future__ import annotations

import pytest
from test_v2_pipeline import PS, _proposals, v2_project  # noqa: F401 - the fixture

from stainid import candidates, exclusions
from stainid.project import load_project
from stainid.v2.proposals import MASK_SCALE

CORE = "LIP-1_B-2"


@pytest.mark.parametrize("stain", ["NeuN", "6E10", "AT8"])
def test_counted_pixels_of_the_view_are_the_area_the_measures_divide_by(v2_project, stain):  # noqa: F811
    project = load_project(v2_project)
    _proposals(project)
    exclusions.forget()
    shown = exclusions.view(project, CORE, stain)
    counted_mm2 = float((shown.codes == exclusions.COUNTED).sum()) * (MASK_SCALE * PS) ** 2 / 1e6
    tiles = candidates.read_table(candidates.path(project, stain, CORE, "tiles"))
    measured = float(tiles.analysis_area_mm2.sum())
    assert measured > 0 and counted_mm2 == pytest.approx(measured, rel=0.03)
    # and the view shades the rest: nothing outside the counted area is drawn as counted
    assert ((shown.codes == exclusions.COUNTED) <= exclusions.automatic(project, CORE, stain).valid()).all()
