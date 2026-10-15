"""The core viewer's AT8 neuropil threads: traced again exactly as the detection step measured them, served in core pixels."""
from __future__ import annotations

import time

import cv2
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from stainid.api import state  # noqa: E402
from stainid.api.app import create_app  # noqa: E402
from stainid.api.threads import polylines, trace_tile  # noqa: E402
from stainid.stains.tau.field_pipeline import analyze_at8_v2  # noqa: E402

TISSUE, THREAD = (225, 220, 225), (135, 90, 55)  # RGB
ORIGIN = 300 - 128  # the tile's field crop starts one context margin before the tile


def _threads_field(size: int = 512) -> np.ndarray:
    rgb = np.full((size, size, 3), TISSUE, np.uint8)
    cv2.line(rgb, (40, 80), (470, 420), THREAD, 4)
    cv2.line(rgb, (70, 450), (440, 110), THREAD, 3)  # crosses the first: a junction
    cv2.line(rgb, (60, 250), (200, 260), THREAD, 2)
    return rgb


class _NoNeurons:
    def predict_proba(self, matrix):
        return np.zeros((len(matrix), 2))


def test_traced_skeleton_is_the_one_the_step_measured():
    rgb = _threads_field()
    inner = (slice(64, 448), slice(64, 448))
    nuclei = np.zeros(rgb.shape[:2], np.uint32)
    exclusion = np.zeros(rgb.shape[:2], bool)
    bundle = {"feature_names": ["soma_area_um2", "ring"], "classifier": _NoNeurons(), "nms_radius_um": 5.0,
              "probability_threshold": 0.5, "mature_p90_threshold": 1.0}
    summary, _ = analyze_at8_v2(rgb, inner, nuclei, 0.04, 0.274, exclusion, bundle)
    skeleton, width = trace_tile(rgb, inner, nuclei, 0.04, 0.274, exclusion)
    assert summary["thread_ridge_skeleton_px"] > 300 and int(skeleton.sum()) == summary["thread_ridge_skeleton_px"]
    assert not skeleton[:64].any() and not skeleton[:, 448:].any()  # only inside the tile, as counted

    pieces = polylines(skeleton, width)
    assert len(pieces) >= 4  # the crossing splits the lines into pieces
    drawn = np.zeros(skeleton.shape, np.uint8)
    for w, points in pieces:
        assert 0.3 < w < 3 and len(points) >= 4 and len(points) % 2 == 0
        cv2.polylines(drawn, [np.array(points, np.int32).reshape(-1, 2)], False, 1, 1)
    near = cv2.dilate(drawn, np.ones((3, 3), np.uint8)).astype(bool)
    assert near[skeleton].mean() > 0.99  # every skeleton pixel lies on (within a pixel of) a drawn piece
    assert polylines(np.zeros((8, 8), bool), np.zeros((8, 8))) == []


@pytest.fixture()
def client(synthetic_project, monkeypatch):
    monkeypatch.chdir(synthetic_project)
    state.reset()
    admin = TestClient(create_app(), client=("127.0.0.1", 50000))
    assert admin.post("/api/session/admin", json={"password": "admin"}).status_code == 200
    yield admin
    state.reset()


def _detect_at8(root) -> None:
    """An AT8 core image with threads in its tile, and the step's per-tile results for it."""
    rgb = np.full((1200, 1200, 3), TISSUE, np.uint8)
    rgb[ORIGIN:ORIGIN + 768, ORIGIN:ORIGIN + 768] = _threads_field(768)
    cv2.imwrite(str(root / "data" / "cores" / "LIP-1" / "B-2" / "AT8.png"), rgb[..., ::-1])
    parts = root / "data" / "analysis" / "cohort_v3" / "parts"
    pd.DataFrame([{"tile_id": "LIP-1_B-2_AT8_S01", "core_id": "LIP-1_B-2", "stain": "AT8", "thread_ridge_length_mm": 0.25,
                   "thread_ridge_skeleton_px": 900}]).to_csv(parts / "LIP-1_B-2_AT8_features.csv", index=False)
    columns = ["tile_id", "core_id", "sample_region_id", "donor_id", "region", "disease_group", "centroid_x_px", "centroid_y_px",
               "candidate_source", "soma_area_um2", "tau_neuron_probability", "tau_neuron_mature"]
    pd.DataFrame(columns=columns).to_csv(parts / "LIP-1_B-2_AT8_objects.csv", index=False)  # no tau+ neurons


def _traced(client, timeout: float = 120) -> dict:
    end = time.monotonic() + timeout
    while True:
        served = client.get("/api/cores/LIP-1_B-2/AT8/threads").json()
        if served["status"] not in ("waiting", "tracing") or time.monotonic() > end:
            return served
        time.sleep(0.2)


def test_threads_only_for_at8_and_after_detection(client):
    assert client.get("/api/cores/LIP-1_B-2/NeuN/threads").status_code == 404
    assert client.get("/api/cores/LIP-1_B-2/AT8/threads").json()["status"] == "not_run"
    assert client.get("/api/cores/LIP-1_B-2/6E10/objects").json()["threads"] is None


def test_threads_traced_in_core_pixels_and_cached(client, synthetic_project):
    _detect_at8(synthetic_project)
    assert client.get("/api/cores/LIP-1_B-2/AT8/objects").json()["threads"] == {"length_mm": 0.25, "skeleton_px": 900}
    assert client.get("/api/cores/LIP-1_B-2/AT8/threads?prepare=false").json()["status"] == "not_started"
    served = _traced(client)
    assert served["status"] == "done" and served["tiles_done"] == served["tiles_total"] == 1
    assert served["columns"] == ["width_um", "points"] and served["pieces"] == len(served["rows"]) >= 4
    assert served["measured"] == {"length_mm": 0.25, "skeleton_px": 900} and served["length_mm"] > 0
    xs = np.concatenate([np.array(points[::2]) for _, points in served["rows"]])
    ys = np.concatenate([np.array(points[1::2]) for _, points in served["rows"]])
    assert xs.min() >= 300 and xs.max() < 812 and ys.min() >= 300 and ys.max() < 812  # inside the tile, in core pixels
    # the first line runs from field (40, 80) to (470, 420): its points lie on it, shifted by the crop origin
    on_first = np.abs((ys - ORIGIN - 80) * 430 - (xs - ORIGIN - 40) * 340) / np.hypot(430, 340) < 3
    assert on_first.sum() >= 2
    assert (synthetic_project / "data" / "cache" / "core_threads" / "LIP-1_B-2_AT8.json").exists()
    assert client.get("/api/cores/LIP-1_B-2/AT8/threads?prepare=false").json() == served  # from the cache
