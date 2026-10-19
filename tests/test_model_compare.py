"""The outline-model panel's scoring and picture grid on made-up outlines (no model weights needed)."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
from PIL import Image

from stainid import candidates
from stainid.masks import model_compare as mc


def _square(x0, y0, side):
    return [x0, y0, x0 + side, y0, x0 + side, y0 + side, x0, y0 + side]


def _frame():
    rows = []
    for n, (stain, decision) in enumerate((("NeuN", "accepted"), ("NeuN", "rejected"), ("6E10", "accepted"))):
        x = 100 + 60 * n
        rows.append({"candidate_id": f"{stain}:core-1:{x}:100", "core_id": "core-1", "tile_id": "t1", "x_core": x, "y_core": 100, "area_um2": 100.0,
                     "outline": candidates.encode_points(_square(x - 10, 90, 20)), "decision": decision, "probability": 0.8,
                     "sam_outline": candidates.encode_points(_square(x - 12, 88, 24)), "sam_status": "ok", "stain": stain, "tma": "1",
                     "group": "fixed (edited)" if n == 0 else "sample (small, accepted)",
                     "truth": candidates.encode_rings([_square(x - 12, 88, 24)]) if n == 0 else "", "person": ""})
    return pd.DataFrame(rows)


def _found(frame, grow):
    out = {}
    for row in frame.itertuples():
        side = 20 + 2 * grow
        polygon = _square(row.x_core - side // 2, 100 - side // 2, side)
        out[row.candidate_id] = {"polygons": [_square(row.x_core - 2, 98, 4), polygon, _square(row.x_core - 30, 70, 60)], "default": 1,
                                 "status": "ok", "pixel_um": 0.5}
    return out


def test_scores_pairs_and_summary():
    frame = _frame()
    found = {"sam_vit_b": _found(frame, 2), "pathosam_vit_b": _found(frame, 0)}
    found["pathosam_vit_b"][frame.candidate_id[1]]["default"] = -1  # stage 3 falls back to the stain outline
    found["pathosam_vit_b"][frame.candidate_id[1]]["status"] = "fallback_size"
    scored = mc.score(frame, found)
    first = scored.iloc[0]
    assert first.has_truth and first["iou_truth_sam_vit_b"] == 1.0  # 24 px squares: the same as the human outline
    assert first["iou_truth_stain outline"] < 1 and first["iou_truth_sam_vit_b best of three"] == 1.0
    assert first.sam_vit_b_ratio_to_stain == 1.44 and first.sam_vit_b_iou_stored_stage3 == 1.0
    assert scored.iloc[1].pathosam_vit_b_ratio_to_stain == 1.0  # its fallback is the stain outline
    summary = mc.summarise(scored, list(found))
    neun = summary[(summary.stain == "NeuN") & (summary.decision == "all") & (summary.method == "pathosam_vit_b")].iloc[0]
    assert neun.stage3_fallback_share == 0.5 and neun.objects == 2
    pairs = mc.pairwise(frame, found)
    assert set(pairs.a) <= {"stain outline", "sam_vit_b"} and len(pairs) == 2 * 3
    truth = mc.truth_table(scored, list(found))
    assert set(truth.method) >= {"stain outline", "sam_vit_b", "sam_vit_b best of three"} and (truth.objects == 1).all()
    timing = pd.DataFrame([{"model": m, "params": 9.4e7, "s_per_object": 0.08, "peak_gpu_gb": 1.4} for m in found])
    head = mc.overview(summary, timing, truth, list(found))  # (the human outlines stand in for the external set here)
    row = head[(head.stain == "NeuN") & (head.model == "sam_vit_b")].iloc[0]
    assert row.params_M == 94.0 and row.external_iou_median == 1.0 and row.external_iou_best_of_three == 1.0
    assert row.summed_area_ratio_accepted == 1.44  # the accepted NeuN candidate: 24 px vs 20 px squares
    text = mc.markdown({}, head, summary, truth, truth, {}, pairs, timing, {}, [], list(found), "Keep SAM ViT-B.")
    assert "## Recommendation\n\nKeep SAM ViT-B." in text and "### NeuN" in text and "| SAM-B |" in text


def test_the_grid_has_one_row_per_candidate(tmp_path, monkeypatch):
    frame = _frame()
    found = {"sam_vit_b": _found(frame, 2)}
    picture = tmp_path / "core.png"
    Image.fromarray(np.full((400, 400, 3), 200, np.uint8)).save(picture)
    tiles = pd.DataFrame([{"tile_id": "t1", "image_path": "core.png"}])
    monkeypatch.setattr("stainid.imaging.core_images.open_image", lambda path: Image.open(path))
    found["sam2.1_hiera_t"] = _found(frame, 1)
    found["pathosam_vit_b"] = _found(frame, 0)
    written = mc.grid(SimpleNamespace(root=tmp_path), tiles, frame, found, mc.score(frame, found), tmp_path)
    assert sorted(p.name for p in written) == ["grid_6E10_01.png", "grid_NeuN_01.png"]
    sheet = Image.open(tmp_path / "grid_NeuN_01.png")
    assert sheet.size == (mc.LABEL_W + mc.CELL * 5, 22 + mc.CELL * 2)  # raw | stain | SAM-B | S2-t | PS-B, two NeuN rows


def test_results_of_an_earlier_pass_are_kept_for_the_same_candidates(tmp_path):
    frame = _frame()
    results = _found(frame, 2)
    ids = frame.candidate_id.tolist()
    mc.save_results(tmp_path, "sam2.1_hiera_b+", "", ids, results, {"model": "sam2.1_hiera_b+", "s_per_object": 0.1})
    kept = mc.load_results(tmp_path, "sam2.1_hiera_b+", "", list(reversed(ids)))
    assert kept is not None and kept[0][ids[0]]["default"] == 1 and kept[1]["s_per_object"] == 0.1
    assert mc.load_results(tmp_path, "sam2.1_hiera_b+", "", ids[:2]) is None  # another panel: run again
    assert mc.load_results(tmp_path, "sam2.1_hiera_b+", "adnp", ids) is None


def test_every_official_size_is_listed_and_the_sheets_split_by_family():
    assert {m["family"] for m in mc.MODELS.values()} == {"SAM", "SAM 2", "PathoSAM"} and len(mc.MODELS) == 10
    assert mc.columns_of(["pathosam_vit_b", "sam2.1_hiera_t", "sam_vit_b"]) == ["sam_vit_b", "sam2.1_hiera_t", "pathosam_vit_b"]
    assert set(mc.COLOURS) == set(mc.MODELS)
