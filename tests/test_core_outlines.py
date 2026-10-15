"""The core viewer's Segment Anything outlines: synthetic label images in field-crop pixels, served in core pixels."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from stainid.api import state  # noqa: E402
from stainid.api.app import create_app  # noqa: E402
from stainid.api.outlines import label_rings  # noqa: E402

TILE = "LIP-1_B-2_6E10_S01"
ORIGIN = 300 - 128  # the tile's field crop starts one context margin before the tile


@pytest.fixture()
def client(synthetic_project, monkeypatch):
    monkeypatch.chdir(synthetic_project)
    state.reset()
    admin = TestClient(create_app(), client=("127.0.0.1", 50000))
    assert admin.post("/api/session/admin", json={"password": "admin"}).status_code == 200
    yield admin
    state.reset()


def _write_masks(root, kept: int = 2) -> None:
    """Outlines for the first `kept` plaques (field-crop pixels (100 + 30 i, 200)), as the masks step writes them."""
    out = root / "data" / "analysis" / "cohort_object_masks" / "6E10"
    (out / "labels").mkdir(parents=True, exist_ok=True)
    (out / "parts").mkdir(parents=True, exist_ok=True)
    labels = np.zeros((768, 768), np.uint16)
    labels[195:205, 95:105] = 1  # a 10 x 10 px square on plaque 0
    yy, xx = np.mgrid[:768, :768]
    labels[(xx - 130) ** 2 + (yy - 200) ** 2 <= 64] = 2  # a disc on plaque 1
    np.savez_compressed(out / "labels" / f"{TILE}.npz", labels=labels, inner=np.array([128, 128]))
    rows = [{"tile_id": TILE, "x": 100 + 30 * i, "y": 200, "mask_status": "ok" if i < kept else "size_rejected",
             "label": i + 1 if i < kept else "", "area_um2": 7.5 if i == 0 else 15.25} for i in range(3)]
    pd.DataFrame(rows).to_csv(out / "parts" / "LIP-1_B-2_6E10_objects.csv", index=False)


def test_label_rings_simplify_and_keep_every_piece():
    labels = np.zeros((40, 40), np.int32)
    labels[5:15, 5:15] = 1
    labels[20:24, 20:24] = 2
    labels[30:34, 30:34] = 2  # an outline cut in two by an earlier object
    labels[36, 36] = 2  # and a speck of it
    labels[38, 2:4] = 3  # an outline too thin to draw: its object keeps its circle
    rings = label_rings(labels)
    assert sorted(zip(rings[1][0][::2], rings[1][0][1::2])) == [(5, 5), (5, 14), (14, 5), (14, 14)]
    assert len(rings[2]) == 2 and 3 not in rings


def test_outlines_not_made_yet(client):
    assert client.get("/api/cores/LIP-1_B-2/6E10/objects").json()["outlines"] is False
    assert client.get("/api/cores/LIP-1_B-2/6E10/outlines").json() == {"status": "not_made", "columns": ["x", "y", "area_um2", "rings"], "rows": []}
    assert client.get("/api/cores/LIP-1_B-2/NeuN/objects").json()["outlines"] is False  # detection has not run there


def test_outlines_in_core_pixels_matched_to_detections(client, synthetic_project):
    _write_masks(synthetic_project)
    found = client.get("/api/cores/LIP-1_B-2/6E10/objects").json()
    assert found["outlines"] is True
    detections = {(r[0], r[1]) for r in found["rows"]}
    served = client.get("/api/cores/LIP-1_B-2/6E10/outlines").json()
    assert served["status"] == "done" and len(served["rows"]) == 2
    square, disc = sorted(served["rows"], key=lambda r: r[0])
    assert (square[0], square[1]) == (ORIGIN + 100, ORIGIN + 200) and (disc[0], disc[1]) in detections
    assert (square[0], square[1]) in detections and square[2] == 7.5
    corners = sorted(zip(square[3][0][::2], square[3][0][1::2]))
    assert corners == [(ORIGIN + 95, ORIGIN + 195), (ORIGIN + 95, ORIGIN + 204), (ORIGIN + 104, ORIGIN + 195), (ORIGIN + 104, ORIGIN + 204)]
    xs, ys = np.array(disc[3][0][::2]), np.array(disc[3][0][1::2])
    assert np.allclose(np.hypot(xs - (ORIGIN + 130), ys - (ORIGIN + 200)), 8, atol=1.5)  # simplified, still on the disc's edge
    assert (synthetic_project / "data" / "cache" / "core_outlines" / "LIP-1_B-2_6E10.json").exists()


def test_outlines_rebuilt_when_the_step_rewrites_the_core(client, synthetic_project):
    _write_masks(synthetic_project, kept=2)
    assert len(client.get("/api/cores/LIP-1_B-2/6E10/outlines").json()["rows"]) == 2
    parts = synthetic_project / "data" / "analysis" / "cohort_object_masks" / "6E10" / "parts" / "LIP-1_B-2_6E10_objects.csv"
    _write_masks(synthetic_project, kept=1)
    os.utime(parts, ns=(parts.stat().st_atime_ns, parts.stat().st_mtime_ns + 10**9))
    assert len(client.get("/api/cores/LIP-1_B-2/6E10/outlines").json()["rows"]) == 1


def test_outlines_are_sent_compressed(client, synthetic_project, monkeypatch):
    from stainid.api.routes import cohort

    _write_masks(synthetic_project)
    monkeypatch.setattr(cohort, "GZIP_MIN_BYTES", 0)
    response = client.get("/api/cores/LIP-1_B-2/6E10/outlines", headers={"Accept-Encoding": "gzip"})
    assert response.headers["content-encoding"] == "gzip" and len(response.json()["rows"]) == 2
    plain = client.get("/api/cores/LIP-1_B-2/6E10/outlines", headers={"Accept-Encoding": "identity"})
    assert "content-encoding" not in plain.headers and plain.json() == response.json()
