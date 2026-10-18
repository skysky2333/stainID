"""Click-to-outline with Segment Anything: features made ahead and kept on disk, the CPU path with a tiny model, and the
names the features are kept under."""
from __future__ import annotations

import time

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("segment_anything")

from stainid import disk_cache  # noqa: E402
from stainid.masks import clicks  # noqa: E402
from stainid.masks.sam import WindowedSam, window_origin, window_origins  # noqa: E402

TILE = "LIP-1_B-2_6E10_S01"


def tiny_sam(checkpoint=None):
    """A Segment Anything with the real parts but a few thousand random weights: shapes and plumbing, not quality."""
    from segment_anything.modeling import ImageEncoderViT, MaskDecoder, PromptEncoder, Sam, TwoWayTransformer

    torch.manual_seed(0)
    dim, size, patch = 32, 64, 16
    return Sam(image_encoder=ImageEncoderViT(img_size=size, patch_size=patch, embed_dim=dim, depth=1, num_heads=2, out_chans=dim,
                                             use_rel_pos=True, window_size=0, global_attn_indexes=(0,)),
               prompt_encoder=PromptEncoder(embed_dim=dim, image_embedding_size=(size // patch, size // patch), input_image_size=(size, size), mask_in_chans=4),
               mask_decoder=MaskDecoder(transformer_dim=dim, transformer=TwoWayTransformer(depth=1, embedding_dim=dim, mlp_dim=64, num_heads=2),
                                        num_multimask_outputs=3, iou_head_depth=1, iou_head_hidden_dim=16)).eval()


@pytest.fixture()
def on_cpu(monkeypatch):
    """The tiny model, on the CPU whatever the machine has (tests never touch a GPU)."""
    from stainid.api import sam_embeddings

    monkeypatch.setenv(clicks.DEVICE_ENV, "cpu")
    monkeypatch.setattr(clicks, "build_model", tiny_sam)
    monkeypatch.setattr(clicks, "_MODELS", {})
    monkeypatch.setattr(clicks, "_CHOSEN", [None, -clicks.GPU_RETRY_SECONDS])
    monkeypatch.setattr(sam_embeddings, "_MEMORY", type(sam_embeddings._MEMORY)())
    monkeypatch.setattr(sam_embeddings, "_FIELDS", type(sam_embeddings._FIELDS)())
    monkeypatch.setattr(sam_embeddings, "_LOOKED", {})
    with sam_embeddings._COND:
        sam_embeddings._TASKS.clear()
        sam_embeddings._FAILED.clear()


def test_windows_cover_the_field_and_the_nearest_one_is_chosen_as_before():
    assert window_origins((2304, 2304)) == [(x, y) for y in (0, 768, 1280) for x in (0, 768, 1280)]
    assert window_origins((768, 900)) == [(0, 0)]  # smaller than a window: one, the whole field
    sam = WindowedSam.__new__(WindowedSam)
    sam.window, sam.stride, sam.rgb = 1024, 768, np.zeros((2304, 1800, 3), np.uint8)
    for x, y in [(10, 10), (1300, 900), (1799, 2303), (900, 1700)]:
        assert sam._origin(x, y) == window_origin((2304, 1800), x, y)
    assert window_origin((2304, 2304), 1700, 1700) == (1280, 1280)


def _old_polygon(mask, origin=(0, 0)):
    """The outline as it was made from the whole window."""
    import cv2
    from scipy import ndimage

    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if count < 2 or stats[1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA])), cv2.CC_STAT_AREA] < clicks.MIN_PIXELS:
        return None
    piece = ndimage.binary_fill_holes(labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    contours, _ = cv2.findContours(piece, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = cv2.approxPolyDP(max(contours, key=cv2.contourArea), clicks.EPSILON_PX, True)[:, 0, :]
    return (contour + np.asarray(origin)).astype(int).tolist() if len(contour) >= 3 else None


def test_outlines_from_the_object_box_are_those_of_the_whole_window():
    import cv2

    rng = np.random.default_rng(1)
    for _ in range(30):
        mask = np.zeros((300, 260), np.uint8)
        for _ in range(int(rng.integers(1, 4))):
            cv2.ellipse(mask, (int(rng.integers(0, 260)), int(rng.integers(0, 300))), (int(rng.integers(3, 90)), int(rng.integers(3, 90))),
                        float(rng.integers(0, 180)), 0, 360, 1, -1)
        mask[rng.random(mask.shape) < 0.02] = 0  # holes
        assert clicks.polygon(mask.astype(bool), (7, 11)) == _old_polygon(mask.astype(bool), (7, 11))
    assert clicks.polygon(np.zeros((50, 50), bool)) is None


def test_the_cpu_path_encodes_a_window_and_outlines_a_click(on_cpu, tmp_path):
    weights = tmp_path / "sam_vit_b_01ec64.pth"
    weights.write_bytes(b"weights")
    model = clicks.outliner(weights)
    assert model is clicks.outliner(weights) and not model.on_gpu and model.gpu is None  # one model per server
    assert not clicks.encodes_on_gpu(weights)
    rgb = np.full((768, 768, 3), 225, np.uint8)
    rgb[200:300, 200:300] = (110, 70, 30)
    found = model.encode(rgb)
    assert found.features.dtype == np.float16 and found.features.shape == (1, 32, 4, 4)
    assert found.original_size == (768, 768) and found.input_size == (64, 64)
    out = model.outline(found, (0, 0), rgb, "6E10", [(250.0, 250.0, True)])
    assert set(out) == {"options", "chosen"} and 0 <= out["chosen"] < max(1, len(out["options"]))
    assert [o["area_px"] for o in out["options"]] == sorted(o["area_px"] for o in out["options"])
    refined = model.outline(found, (0, 0), rgb, "6E10", [(250.0, 250.0, True), (600.0, 600.0, False), (2000.0, 10.0, True)])  # outside: ignored
    assert len(refined["options"]) <= 1


def test_a_failing_gpu_falls_back_to_the_cpu(on_cpu, tmp_path, monkeypatch):
    weights = tmp_path / "sam_vit_b_01ec64.pth"
    weights.write_bytes(b"weights")
    model = clicks.outliner(weights)

    def broken(_x):
        raise RuntimeError("CUDA out of memory")

    model.gpu, model._gpu_encoder, model._cpu_encoder = "cuda:0", broken, None  # as if loaded on a card that then filled up
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    assert model.on_gpu
    found = model.encode(np.zeros((100, 120, 3), np.uint8))
    assert found.features.shape == (1, 32, 4, 4) and found.original_size == (100, 120)
    assert not model.on_gpu  # tried again after a while


# ---------------------------------------------------------------- through the app

@pytest.fixture()
def app(synthetic_project, monkeypatch, on_cpu):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from people import annotator

    from stainid.api import state
    from stainid.api.app import create_app
    from stainid.api.routes import outline as outline_routes

    monkeypatch.chdir(synthetic_project)
    monkeypatch.setattr(outline_routes, "proposals_job", lambda stain, session: None)
    state.reset()
    for model in ("sam", "amyloid"):
        weights = state.get_state().project.model(model)
        weights.parent.mkdir(parents=True, exist_ok=True)
        weights.write_bytes(b"weights")
    admin = TestClient(create_app(), client=("127.0.0.1", 50000))
    assert admin.post("/api/session/admin", json={"password": "admin"}).status_code == 200
    made = admin.post("/api/segmentation/sessions", json={"stains": ["6E10"], "n_fields": 1})
    assert made.status_code == 200, made.text
    name = made.json()[0]["name"]
    ann = annotator(admin, "Ann", name)
    ann.session_name = name
    yield ann
    state.reset()


def _wait_ready(ann, name: str) -> dict:
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        status = ann.get(f"/api/annotate/{name}/outline/{TILE}/segment").json()
        if status["state"] != "preparing":
            return status
        time.sleep(0.05)
    raise AssertionError("the field's features were never made")


def test_a_field_is_prepared_in_the_background_and_clicks_use_what_was_kept(app):
    from stainid.api import sam_embeddings
    from stainid.api.state import get_state

    name = app.session_name
    work = app.get(f"/api/annotate/{name}/outline/{TILE}").json()
    assert work["segment_anything"] and work["segment_ready"] is False  # asked for at once
    assert _wait_ready(app, name) == {"ready": True, "state": "ready"}
    assert app.get(f"/api/annotate/{name}/outline/{TILE}").json()["segment_ready"]
    state = get_state()
    stem = sam_embeddings.field_stem(state, TILE)
    folder = state.cache_dir("sam_embeddings")
    assert sorted(p.name for p in folder.iterdir()) == [f"{stem}.0-0.npz", f"{stem}.json"]

    made = []
    model = clicks.outliner(sam_embeddings.checkpoint(state))
    original = model.encode
    model.encode = lambda window: made.append(window.shape) or original(window)
    out = app.post(f"/api/annotate/{name}/outline/{TILE}/segment", json={"points": [[250, 250, 1]]})
    assert out.status_code == 200 and set(out.json()) == {"options", "chosen"}
    assert made == []  # from what was made ahead

    sam_embeddings._MEMORY.clear()
    disk_cache.flush()
    disk_cache.evict(state.root, 0, keep_seconds=0)  # cleaned up: made again on the next click
    assert not any(folder.iterdir())
    assert app.post(f"/api/annotate/{name}/outline/{TILE}/segment", json={"points": [[250, 250, 1]]}).status_code == 200
    assert made == [(768, 768, 3)] and (folder / f"{stem}.0-0.npz").exists()


def test_fields_ahead_wait_for_a_gpu_and_the_open_field_comes_first(app):
    from stainid.api import sam_embeddings
    from stainid.api.state import get_state

    state = get_state()
    meta = {"name": "s", "open": True, "fields": [TILE]}
    with sam_embeddings._COND:  # no worker picks these up while we look
        sam_embeddings._TASKS.clear()
        ahead = sam_embeddings._Task(state, "other", sam_embeddings.AHEAD)
        now = sam_embeddings._Task(state, TILE, sam_embeddings.NOW)
        assert sam_embeddings._more_urgent(ahead)() is False
        sam_embeddings._TASKS.update({("r", "other"): ahead, ("r", TILE): now})
        assert sam_embeddings._more_urgent(ahead)() and not sam_embeddings._more_urgent(now)()
        sam_embeddings._TASKS.clear()
    assert sam_embeddings._run(sam_embeddings._Task(state, TILE, sam_embeddings.AHEAD))  # finished without making anything on the CPU
    assert not sam_embeddings.marker_path(state, sam_embeddings.field_stem(state, TILE)).exists()
    sam_embeddings.prepare_session(state, {**meta, "open": False})
    assert not sam_embeddings._TASKS


def test_features_are_named_after_what_they_are_made_from(app):
    import os

    from stainid.api import sam_embeddings
    from stainid.api.state import get_state

    state = get_state()
    stem = sam_embeddings.field_stem(state, TILE)
    assert stem.startswith(f"{TILE}.") and stem == sam_embeddings.field_stem(state, TILE)
    first, second = (sam_embeddings.window_path(state, stem, o) for o in [(0, 0), (768, 0)])
    assert first != second and disk_cache.unit(f"sam_embeddings/{first.name}") == disk_cache.unit(f"sam_embeddings/{second.name}") \
        == disk_cache.unit(f"sam_embeddings/{sam_embeddings.marker_path(state, stem).name}")  # a field's files go together
    assert sam_embeddings.field_stem(state, "LIP-1_B-2_AT8_S01") != stem

    image = state.root / state.tile(TILE)["image_path"]
    stat = image.stat()
    os.utime(image, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10**9))  # the core image changed
    changed = sam_embeddings.field_stem(state, TILE)
    assert changed != stem
    weights = sam_embeddings.checkpoint(state)
    weights.write_bytes(b"other weights")
    assert sam_embeddings.field_stem(state, TILE) not in (stem, changed)
    assert "sam_embeddings" in disk_cache.KINDS
