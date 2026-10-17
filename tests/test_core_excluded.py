"""The Cores view's "Excluded areas" layer: /api/cores/{core}/{stain}/excluded (types and shares) and excluded.png (the
translucent, hatched picture), drawn from the resolved layer (`stainid.exclusions.view`: the stage-2 mask, one byte of
`stainid.v2.proposals.BITS` per 8 × 8 px, with every edit applied, and the tissue in no analysed tile)."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from stainid import exclusions  # noqa: E402
from stainid.api import state, store_view  # noqa: E402
from stainid.api.app import create_app  # noqa: E402
from stainid.v2.proposals import BITS, MASK_SCALE  # noqa: E402

CORE = "LIP-1_B-2"


@pytest.fixture()
def client(synthetic_project, monkeypatch):
    monkeypatch.chdir(synthetic_project)
    state.reset()
    exclusions.forget()
    admin = TestClient(create_app(), client=("127.0.0.1", 50000))
    assert admin.post("/api/session/admin", json={"password": "admin"}).status_code == 200
    yield admin
    state.reset()
    exclusions.forget()


def write_mask(project_root) -> np.ndarray:
    """A 20 × 20 mask (160 × 160 core px): all analysed tissue, with a fold (part of it also hand-drawn: not read), a
    linear artifact and a hole without tissue, and a column outside every tile."""
    mask = np.full((20, 20), BITS["tile"] | BITS["tissue"], np.uint8)
    mask[0:4, :] |= BITS["fold"]
    mask[0:2, 0:10] |= BITS["manual"]
    mask[10, :] |= BITS["linear"]
    mask[15:20, 15:20] = BITS["tile"]  # a hole: analysed, no tissue
    mask[:, 19] = 0  # outside the analysed tiles
    from stainid.project import load_project

    target = store_view.excluded_path(load_project(project_root), CORE, "AT8")
    target.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(target), mask)
    return mask


def test_a_field_gets_its_part_of_the_excluded_areas(client, synthetic_project):
    """The annotation pages draw the part of the core's layer over their field, in field pixels."""
    write_mask(synthetic_project)
    project = state.get_state().project
    tile = {"tile_id": "F", "core_id": CORE, "stain": "AT8", "x_px": 40, "y_px": 16, "width_px": 64, "height_px": 48}
    info, png = store_view.field_excluded(project, tile)
    x0, y0 = max(0, 40 - project.context_px), max(0, 16 - project.context_px)
    assert info["status"] == "done" and info["x"] == (x0 // 8) * 8 - x0 and info["y"] == (y0 // 8) * 8 - y0
    picture = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_UNCHANGED)
    assert picture.shape[:2] == (info["height"] // 8, info["width"] // 8) and picture[..., 3].max() > 0
    assert "fold" in {k["id"] for k in info["kinds"]}  # the fold at the top of the core reaches this field


def test_excluded_areas(client, synthetic_project):
    assert store_view.EXCLUDED_SCALE == MASK_SCALE
    missing = client.get(f"/api/cores/{CORE}/AT8/excluded").json()
    assert missing["status"] == "not_made" and [k["id"] for k in missing["kinds"]] == [k.id for k in store_view.EXCLUDED_KINDS]
    assert client.get(f"/api/cores/{CORE}/AT8/excluded.png").status_code == 404

    write_mask(synthetic_project)
    info = client.get(f"/api/cores/{CORE}/AT8/excluded").json()
    assert info["status"] == "done" and (info["width"], info["height"], info["scale"]) == (20, 20, MASK_SCALE)
    share = {k["id"]: k["fraction"] for k in info["kinds"]}
    assert set(share) == {"fold", "artifact", "no_tissue", "focus", "edge_guard", "not_analysed"}  # one entry per type, no history layers
    core = 20 * 19  # analysed cells (the synthetic image has no tissue the tiling would find outside its tile)
    assert share["fold"] == round(4 * 19 / core, 4)  # the hand-drawn bit is not read: hand-drawn areas are edits now
    assert share["artifact"] == round(19 / core, 4)
    assert share["no_tissue"] == round(5 * 4 / core, 4)
    assert share["edge_guard"] == 0.0 and share["focus"] == 0.0
    assert all(k["color"].startswith("#") and len(k["color"]) == 7 for k in info["kinds"])

    picture = client.get(f"/api/cores/{CORE}/AT8/excluded.png?v={info['version']}")
    assert picture.status_code == 200 and picture.headers["content-type"] == "image/png"
    rgba = cv2.imdecode(np.frombuffer(picture.content, np.uint8), cv2.IMREAD_UNCHANGED)
    assert rgba.shape == (20, 20, 4)
    assert rgba[12, 5, 3] == 0 and rgba[5, 19, 3] == 0  # plain tissue and outside the tiles (no tissue there): transparent
    assert rgba[3, 15, 3] > 0 and tuple(rgba[3, 15, :3][::-1]) == (255, 176, 0)  # fold (BGRA on disk)
    assert {int(rgba[y, x, 3]) for y in range(2, 4) for x in range(19)} == {70, 210}  # translucent fill and hatching lines
    assert {int(rgba[y, x, 3]) for y in range(15, 19) for x in range(15, 19)} == {40, 120}  # no tissue: lighter, a faint wash with fine dots
    assert tuple(rgba[15, 15, :3][::-1]) == (120, 128, 140)


def test_tissue_in_no_tile_is_drawn_as_not_analysed(client, synthetic_project, monkeypatch):
    """What the analysis does not count is drawn: tissue outside every analysed tile, and a core image with no tiles."""
    write_mask(synthetic_project)
    project = state.get_state().project
    tissue = np.zeros((20, 20), bool)
    tissue[:, 18:] = True  # tissue reaching past the tile into column 19
    monkeypatch.setattr(exclusions, "_tissue", lambda project, image, shape: tissue if shape in (None, (20, 20)) else None)
    shown = exclusions.view(project, CORE, "AT8")
    assert (shown.codes[:, 19] == exclusions.NOT_ANALYSED).all() and shown.codes[12, 18] == 0
    assert exclusions.automatic(project, CORE, "AT8").codes[5, 19] == exclusions.OUTSIDE  # the analysis layer itself is unchanged
    info = client.get(f"/api/cores/{CORE}/AT8/excluded").json()
    assert {k["id"]: k["fraction"] for k in info["kinds"]}["not_analysed"] == round(20 / (20 * 19 + 20), 4)
