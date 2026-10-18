"""Build a fully synthetic stainID project for demos, screenshots and trying out the web app.

    python tools/demo/make_demo_project.py OUT_DIR [--seed 2026] [--workers 6]

The project "Demo cohort (synthetic)" has three TMAs (DEMO-1 to DEMO-3) of 2 x 5 cores, 15 invented donors in three
groups (Control, ASYMAD, AD) with a frontal and a temporal core each, and NeuN, 6E10 and AT8 core images drawn
procedurally (synthetic_tissue.py). Every table the app reads is written as if the workflow had run: slides table, core
table, TMA map, whole-core tiles, nuclei and outline label images, NeuN / 6E10 / AT8 detections
(placed exactly on the drawn objects, with simulated model errors), candidate pools, a label store with labelling
sessions by three invented annotators, trained stain models with their validation reports, and the results tables.
The cheap stainid steps run for real on the demo: stain calibration, model training and the results tables.

The v2 detection is written as if its stages had run: a candidate store per core image and stain (every candidate the
simulated detector saw, accepted and rejected, with three outline sizes standing in for the Segment Anything masks; the
objects it missed are left for people to add), stage manifests with the digests the stages would record (so the Run
page shows every v2 stage done and up to date), the v2 results tables and finished runs of each step in the job store.
On that store sits an AT8 "Fix the field" session, with Ann's saved work on a field
she has not finished, fields finished by Ben and Carla, and the tutorial done by the three of them (so a new name gets
it). demo.json names the sessions and fields the screenshots open.

Nothing is read from a real study: all values come from the random generator (fixed by --seed). The published model
files (Cellpose-SAM, Segment Anything, Phikon, plaque CNNs) are small placeholder files, and the "slide scans" are
low-resolution overview pictures, so the demo shows every page but the slide-reading and deep-learning steps cannot
be re-run on it. About 1 GB on disk; a few minutes on 6 cores.

This project is already set up; make_setup_demo.py makes one that is not, for trying the guided setup.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from calendar import timegm
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from stainid.labels import pool, sessions, store
from stainid.labels.tasks import FIELD_TAGS, NEGATIVE, ai_value, positive
from stainid.project import DEFAULTS, Project, load_project, save_project
from stainid.sampling.tiles import labelling_tiles
from stainid.training.train import train_model
from stainid.workflows.aggregate import summarize_results
from stainid.workflows.calibrate import calibrate_slides

sys.path.insert(0, str(Path(__file__).resolve().parent))
from synthetic_tissue import PIXEL_UM, draw_core  # noqa: E402

NAME = "Demo cohort (synthetic)"
PREFIX = "DEMO-"
STAINS = ("NeuN", "6E10", "AT8")
TMAS, ROWS, COLUMNS = 3, 2, 5
REGIONS = ("frontal", "temporal")
GROUPS = [{"code": "CT", "label": "Control", "color": 3}, {"code": "ASYM", "label": "ASYMAD", "color": 1},
          {"code": "AD", "label": "AD", "color": 2}]
SEVERITY = {"CT": (0.0, 0.12), "ASYM": (0.28, 0.5), "AD": (0.62, 0.95)}
CORE_PX, FIELD_PX, CONTEXT_PX, MIN_TISSUE = 4800, 2048, 128, 0.10
START = timegm((2026, 8, 31, 8, 30, 0))     # first label of the (invented) labelling rounds
PEOPLE = ("Ann", "Ben", "Carla")
THRESHOLD = {"NeuN": 0.5, "6E10": 0.5, "AT8": 0.4}
TRUTH = {
    "neuron": "NeuN+ neuron", "missed_neuron": "NeuN+ neuron", "glia": "not a neuron", "precipitate": "artifact or unclear",
    "compact": "compact plaque", "diffuse": "diffuse plaque", "small_plaque": "small plaque", "missed_plaque": "diffuse plaque",
    "speck": "not a plaque", "vascular": "not a plaque",
    "tangle": "tau+ neuron: tangle", "pretangle": "tau+ neuron: pretangle", "missed_pretangle": "tau+ neuron: pretangle",
    "tau_dot": "not a tau+ neuron",
}
# The classes the simulated annotators choose from, in the words of the labels made before types existed
# (`stainid.labels.vocabulary.LEGACY` reads them as identity and type).
CLASSES = {
    "NeuN": ["NeuN+ neuron", "not a neuron", "artifact or unclear", "unsure"],
    "6E10": ["compact plaque", "diffuse plaque", "small plaque", "not a plaque", "unsure"],
    "AT8": ["tau+ neuron: tangle", "tau+ neuron: pretangle", "not a tau+ neuron", "unsure"],
}
POSITIVE_KINDS = {"neuron", "missed_neuron", "compact", "diffuse", "small_plaque", "missed_plaque", "tangle", "pretangle", "missed_pretangle"}
META = ("core_id", "tma", "donor_id", "sample_region_id", "region", "disease_group", "technical_replicate")


# ---------------------------------------------------------------- cohort

def design_cohort(rng: np.random.Generator) -> tuple[list[dict], list[dict]]:
    """15 donors (5 per group, every TMA holds all groups) and one frontal and one temporal core per donor."""
    while True:
        groups = rng.permutation(["CT"] * 5 + ["ASYM"] * 5 + ["AD"] * 5).reshape(TMAS, 5)
        if all(len(set(row)) == 3 for row in groups):
            break
    ids = rng.permutation(np.arange(1, 16))
    donors, cores = [], []
    for t in range(TMAS):
        positions = rng.permutation([f"{chr(65 + c)}-{r + 1}" for r in range(ROWS) for c in range(COLUMNS)])
        for k, group in enumerate(groups[t]):
            low, high = SEVERITY[group]
            severity = float(rng.uniform(low, high))
            donor = {"donor_id": f"D{ids[5 * t + k]:03d}", "tma": str(t + 1), "disease_group": str(group),
                     "age": int(rng.integers(62, 92)), "sex": str(rng.choice(["F", "M"])),
                     "apoe_e4_count": int(rng.choice([0, 1, 2], p=[0.6, 0.32, 0.08]) if group == "CT" else rng.choice([0, 1, 2], p=[0.35, 0.45, 0.2])),
                     "pmd_hours": round(float(rng.uniform(3, 20)), 1), "severity": severity}
            donors.append(donor)
            for j, region in enumerate(REGIONS):
                label = positions[2 * k + j]
                tau = min(1.0, severity * (1.15 if region == "temporal" else 0.9))
                cores.append({
                    "core_id": f"{PREFIX}{t + 1}_{label}", "tma": str(t + 1), "core_label": label, "donor_id": donor["donor_id"],
                    "sample_region_id": f"{donor['donor_id']}_{region}", "region": region, "disease_group": str(group), "technical_replicate": 1,
                    "cerad": {"CT": "none", "ASYM": "moderate", "AD": "frequent"}[group] if rng.random() > 0.15 else "sparse",
                    "braak": {"CT": "I", "ASYM": "III", "AD": "VI"}[group] if rng.random() > 0.2 else {"CT": "II", "ASYM": "IV", "AD": "V"}[group],
                    "severity": {"NeuN": severity, "6E10": min(1.0, severity * rng.uniform(0.9, 1.1)), "AT8": tau},
                    "neuron_density": float(rng.normal(374 if region == "frontal" else 340, 28.3) * (1 - 0.42 * severity)),
                    "shape_seed": int(rng.integers(1 << 31)),
                })
    return donors, sorted(cores, key=lambda c: (c["tma"], c["core_label"]))


# ---------------------------------------------------------------- drawing one core image (runs in worker processes)

def _grid(width: int) -> list[tuple[int, int]]:
    out = [(start, FIELD_PX) for start in range(0, width - FIELD_PX + 1, FIELD_PX)]
    rest = width - len(out) * FIELD_PX
    return out + ([(len(out) * FIELD_PX, rest)] if rest >= 256 else [])


def _ai_call(rng: np.random.Generator, stain: str, kind: str) -> dict:
    """The simulated detector: a probability and, if accepted, a class (with a few deliberate mistakes)."""
    beta = {"neuron": (9, 1.6), "glia": (1.4, 9), "precipitate": (2, 5), "compact": (10, 1.5), "diffuse": (8, 1.8),
            "small_plaque": (6, 2.2), "speck": (1.5, 8), "vascular": (3, 4), "tangle": (12, 1.3), "pretangle": (6, 2.2),
            "tau_dot": (1.5, 6)}[kind]
    p = float(rng.beta(*beta))
    call = {"probability": round(p, 4), "accepted": p >= THRESHOLD[stain], "model_class": None, "mature": False}
    if not call["accepted"]:
        return call
    if stain == "NeuN":
        call["model_class"] = "neuron"
    elif stain == "6E10":
        swap = rng.random()
        call["model_class"] = {"compact": "compact" if swap > 0.1 else "diffuse", "diffuse": "diffuse" if swap > 0.1 else "compact",
                               "small_plaque": "small_plaque" if swap > 0.08 else "diffuse"}.get(kind, "small_plaque" if swap > 0.5 else "diffuse")
    else:
        call["mature"] = bool(rng.random() < (0.9 if kind == "tangle" else 0.12 if kind == "pretangle" else 0.3))
        call["model_class"] = "dense_body" if rng.random() < (0.8 if call["mature"] else 0.15) else "nucleus_ring"
    return call


def make_core(job: dict) -> dict:
    """Draw one core image and write it with its per-tile nuclei and outline label images; return what was drawn."""
    rng = np.random.default_rng(job["seed"])
    stain, root = job["stain"], Path(job["root"])
    core = draw_core(rng, stain, job["severity"], job["shape_seed"], CORE_PX, job["neuron_density"])
    image_path = root / job["image_path"]
    image_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(image_path), cv2.cvtColor(core.rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
    thumb = cv2.resize(core.rgb, (480, 480), interpolation=cv2.INTER_AREA)

    tiles = []
    squares = [(x, y, w, h, r, c) for r, (y, h) in enumerate(_grid(CORE_PX)) for c, (x, w) in enumerate(_grid(CORE_PX))]
    for x, y, w, h, r, c in squares:
        fraction = float(core.outline_mask[y:y + h, x:x + w].mean())
        if fraction >= MIN_TISSUE:
            tiles.append({"x_px": x, "y_px": y, "width_px": w, "height_px": h, "row": r, "column": c, "tissue_fraction": round(fraction, 4),
                          "tissue_area_mm2": float(core.tissue[y:y + h, x:x + w].sum() * PIXEL_UM**2 / 1e6),
                          "dab_positive_mm2": float(core.dab_positive[y:y + h, x:x + w].sum() * PIXEL_UM**2 / 1e6)})
    for t in tiles:
        t["tile_id"] = f"{job['core_id']}_{stain}_W{t['row']:02d}{t['column']:02d}"
        t["selection_order"] = 100 + 100 * t["row"] + t["column"]

    def tile_of(x: float, y: float) -> dict | None:
        return next((t for t in tiles if t["x_px"] <= x < t["x_px"] + t["width_px"] and t["y_px"] <= y < t["y_px"] + t["height_px"]), None)

    objects = []
    for i, o in enumerate(core.objects):
        tile = tile_of(o.x, o.y)
        if tile is None:
            continue
        candidate = not o.kind.startswith("missed_")
        call = _ai_call(rng, stain, o.kind) if candidate else {"probability": None, "accepted": False, "model_class": None, "mature": False}
        hull = cv2.convexHull(o.outline.astype(np.float32))
        perimeter = float(cv2.arcLength(o.outline.astype(np.float32), True) * PIXEL_UM)
        objects.append({"uid": i, "kind": o.kind, "x": float(o.x), "y": float(o.y), "tile_id": tile["tile_id"], "candidate": candidate,
                        "area_um2": o.area_um2, "diameter_um": o.diameter_um, "dab": o.dab, "perimeter_um": perimeter,
                        "solidity": float(min(1.0, o.area_um2 / max(cv2.contourArea(hull) * PIXEL_UM**2, 1e-6))),
                        "outline": o.outline.round().astype(int).tolist(), **o.extra, **call})

    nuclei_points = core.nuclei[:, :2]
    for t in tiles:
        x0, y0 = max(0, t["x_px"] - CONTEXT_PX), max(0, t["y_px"] - CONTEXT_PX)
        x1, y1 = min(CORE_PX, t["x_px"] + t["width_px"] + CONTEXT_PX), min(CORE_PX, t["y_px"] + t["height_px"] + CONTEXT_PX)
        t["origin"] = (x0, y0)
        labels = np.zeros((y1 - y0, x1 - x0), np.int32)
        inside = (nuclei_points[:, 0] >= x0 - 30) & (nuclei_points[:, 0] < x1 + 30) & (nuclei_points[:, 1] >= y0 - 30) & (nuclei_points[:, 1] < y1 + 30)
        for n, (x, y, a, b, angle) in enumerate(core.nuclei[inside], start=1):
            cv2.ellipse(labels, (int(x - x0), int(y - y0)), (max(2, int(a)), max(2, int(b))), float(angle), 0, 360, n, -1)
        np.savez_compressed(root / job["nuclei_dir"] / f"{t['tile_id']}.npz", masks=labels)
        in_field = ((nuclei_points[:, 0] >= t["x_px"]) & (nuclei_points[:, 0] < t["x_px"] + t["width_px"])
                    & (nuclei_points[:, 1] >= t["y_px"]) & (nuclei_points[:, 1] < t["y_px"] + t["height_px"]))
        t["nucleus_count"] = int(in_field.sum())
        outlines = np.zeros((y1 - y0, x1 - x0), np.int32)
        label = 0
        for o in objects:
            if o["tile_id"] == t["tile_id"] and o["accepted"]:
                label += 1
                o["mask_label"] = label
                cv2.fillPoly(outlines, [np.asarray(o["outline"], np.int32) - [x0, y0]], label)
        (root / job["masks_dir"]).mkdir(parents=True, exist_ok=True)
        np.savez_compressed(root / job["masks_dir"] / f"{t['tile_id']}.npz", labels=outlines)
        if stain == "AT8":
            length = [float(np.linalg.norm(np.diff(line, axis=0), axis=1).sum()) for line in core.threads
                      if t["x_px"] <= line[0][0] < t["x_px"] + t["width_px"] and t["y_px"] <= line[0][1] < t["y_px"] + t["height_px"]]
            t["thread_count"], t["thread_length_px"] = len(length), float(sum(length))
        if stain == "6E10":
            plaques = [o for o in objects if o["tile_id"] == t["tile_id"] and o["accepted"]]
            near = far = 0
            for o in plaques:
                d = np.hypot(nuclei_points[:, 0] - o["x"], nuclei_points[:, 1] - o["y"])
                r = o["diameter_um"] / 2 / PIXEL_UM
                o["nuclei_inside"], o["nuclei_ring"] = int((d < r).sum()), int(((d >= r) & (d < r + 15 / PIXEL_UM)).sum())
                near += o["nuclei_inside"] + o["nuclei_ring"]
            far = max(0, t["nucleus_count"] - near)
            t["far_nuclei"] = far
    return {"key": (job["core_id"], stain), "tiles": tiles, "objects": objects, "thumb": thumb,
            "tissue_fraction": float(core.outline_mask.mean())}


# ---------------------------------------------------------------- tables

def write_csv(path: Path, rows: list[dict] | pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    (rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)).to_csv(path, index=False)


def crop_xy(o: dict, tile: dict) -> tuple[float, float]:
    """Object position in the field crop (field plus context margin), the coordinate system of every pipeline output."""
    return round(o["x"] - tile["origin"][0], 1), round(o["y"] - tile["origin"][1], 1)


def detection_outputs(root: Path, cores: list[dict], drawn: dict, rng: np.random.Generator) -> None:
    analysis = root / "data" / "analysis"
    for core in cores:
        meta = {k: core[k] for k in META}
        for stain in STAINS:
            result = drawn[(core["core_id"], stain)]
            tiles = {t["tile_id"]: t for t in result["tiles"]}
            objects = [o for o in result["objects"] if o["candidate"]]
            mask_rows = []
            if stain == "NeuN":
                for tile_id, t in tiles.items():
                    mine = [o for o in objects if o["tile_id"] == tile_id]
                    accepted = [o for o in mine if o["accepted"]]
                    area = max(t["tissue_area_mm2"], 1e-6)
                    rows = [{"tile_id": tile_id, **meta, "stain": stain, "selection_order": t["selection_order"],
                             "candidate_class": "positive" if o["accepted"] else "negative", "touches_edge": "false",
                             "tile_centroid_x_px": round(o["x"] - t["x_px"], 1), "tile_centroid_y_px": round(o["y"] - t["y_px"], 1),
                             "area_um2": round(o["area_um2"], 2), "equivalent_diameter_um": round(o["diameter_um"], 2),
                             "mean_dab_od": round(0.08 + 0.12 * o["dab"] + rng.normal(0, 0.01), 4),
                             "context_positive_probability": o["probability"], "source_kind": "cellpose" if rng.random() < 0.3 else "dab_contour"}
                            for o in mine]
                    folder = root / "data" / "analysis" / "cohort_neun" / "tiles"
                    write_csv(folder / f"{tile_id}_objects.csv", rows if rows else pd.DataFrame(columns=["tile_id", "candidate_class"]))
                    write_csv(folder / f"{tile_id}_features.csv", [{
                        "tile_id": tile_id, **meta, "stain": stain, "selection_order": t["selection_order"], "tissue_area_mm2": round(area, 6),
                        "total_artifact_exclusion_area_fraction": round(float(rng.uniform(0.0, 0.04)), 4), "candidate_profile_count": len(mine),
                        "neun_positive_profile_count": len(accepted),
                        "neun_positive_profile_area_fraction": round(sum(o["area_um2"] for o in accepted) / 1e6 / area, 6),
                        "neun_review_profile_count": sum(0.35 <= o["probability"] < 0.65 for o in mine),
                        "neun_union_positive_profile_count": len(accepted) + int(rng.poisson(0.04 * len(accepted) + 0.5)),
                        "neun_local_density_cv_100um": round(float(rng.uniform(0.3, 0.6)), 4),
                        "neun_local_density_window_count": int(area / 0.01)}])
            else:
                features, rows = [], []
                for tile_id, t in tiles.items():
                    accepted = [o for o in objects if o["tile_id"] == tile_id and o["accepted"]]
                    area = max(t["tissue_area_mm2"], 1e-6)
                    base = {"tile_id": tile_id, **meta, "stain": stain, "selection_order": t["selection_order"],
                            "v2_tissue_area_mm2": round(area, 6), "nuclei_available": "true", "nucleus_count": t["nucleus_count"]}
                    if stain == "AT8":
                        length_mm = t["thread_length_px"] * PIXEL_UM / 1000
                        features.append({**base, "tau_neuron_count": len(accepted), "tau_neuron_mature_count": sum(o["mature"] for o in accepted),
                                         "thread_ridge_length_mm": round(length_mm, 5), "thread_ridge_skeleton_px": int(t["thread_length_px"]),
                                         "thread_ridge_width_sum_um": round(t["thread_length_px"] * rng.uniform(0.55, 0.75), 2),
                                         "thread_branchpoint_count": int(t["thread_count"] * rng.uniform(0.2, 0.4)),
                                         "at8_positive_area_mm2": round(t["dab_positive_mm2"], 6)})
                        rows += [{"tile_id": tile_id, **meta, "stain": stain, "selection_order": t["selection_order"],
                                  "centroid_x_px": crop_xy(o, t)[0], "centroid_y_px": crop_xy(o, t)[1], "candidate_source": o["model_class"],
                                  "tau_neuron_probability": o["probability"], "soma_area_um2": round(o["area_um2"], 2),
                                  "tau_neuron_mature": str(o["mature"]).lower()} for o in accepted]
                    else:
                        plaques = [o for o in accepted if o["diameter_um"] > 10]
                        ring = sum(o["nuclei_ring"] for o in plaques)
                        ring_area = sum(np.pi * ((o["diameter_um"] / 2 + 15) ** 2 - (o["diameter_um"] / 2) ** 2) for o in plaques)
                        features.append({**base, "v2_plaque_count": len(plaques), "v2_plaque_area_sum_um2": round(sum(o["area_um2"] for o in plaques), 2),
                                         "v2_plaque_nuclei": sum(o["nuclei_inside"] for o in plaques), "v2_ring_nuclei": ring,
                                         "v2_ring_area_um2": round(ring_area, 2), "v2_far_nuclei": t["far_nuclei"],
                                         "v2_far_area_mm2": round(max(area - ring_area / 1e6, 1e-6), 6),
                                         "v2_vascular_or_edge_count": sum(o["kind"] == "vascular" for o in objects if o["tile_id"] == tile_id),
                                         "v2_speck_count": len(accepted) - len(plaques),
                                         "v2_dense_core_area_sum_um2": round(sum(o["area_um2"] * o.get("dense_core_fraction", 0) for o in plaques), 2)})
                        rows += [{"tile_id": tile_id, **meta, "stain": stain, "selection_order": t["selection_order"],
                                  "centroid_x_px": crop_xy(o, t)[0], "centroid_y_px": crop_xy(o, t)[1], "plaque_class": o["model_class"],
                                  "plaque_probability": o["probability"], "plaque_area_um2": round(o["area_um2"], 2),
                                  "equivalent_diameter_um": round(o["diameter_um"], 2), "dense_core_fraction": round(o.get("dense_core_fraction", 0), 4),
                                  "circularity": round(min(1.0, 4 * np.pi * o["area_um2"] / o["perimeter_um"] ** 2), 4)} for o in plaques]
                parts = analysis / "cohort_v3" / "parts"
                write_csv(parts / f"{core['core_id']}_{stain}_features.csv", features)
                write_csv(parts / f"{core['core_id']}_{stain}_objects.csv", rows if rows else pd.DataFrame(columns=["tile_id", "stain", "centroid_x_px"]))
            for o in objects:
                if o["accepted"]:
                    t = tiles[o["tile_id"]]
                    mask_rows.append({"tile_id": o["tile_id"], **meta, "stain": stain, "label": o["mask_label"], "mask_status": "ok",
                                      "seed_source": o.get("model_class") or "", "area_um2": round(o["area_um2"] * rng.uniform(0.92, 1.05), 2),
                                      "perimeter_um": round(o["perimeter_um"], 2), "solidity": round(o["solidity"], 4),
                                      "circularity": round(min(1.0, 4 * np.pi * o["area_um2"] / o["perimeter_um"] ** 2), 4),
                                      "elongation": round(float(rng.uniform(1.1, 2.4) if o["kind"] == "tangle" else rng.uniform(1.0, 1.6)), 3),
                                      "normalized_mean_dab": round(o["dab"] / 0.3 * rng.uniform(0.9, 1.1), 3),
                                      "dense_core_fraction": round(o.get("dense_core_fraction", 0), 4), "dense_core_count": int(o.get("dense_core_fraction", 0) > 0.05),
                                      "seed_mature": str(o["mature"]).lower(), "centroid_x_px": crop_xy(o, t)[0], "centroid_y_px": crop_xy(o, t)[1]})
            write_csv(analysis / "cohort_object_masks" / stain / "parts" / f"{core['core_id']}_{stain}_objects.csv",
                      mask_rows if mask_rows else pd.DataFrame(columns=["tile_id", "label", "mask_status"]))


# ---------------------------------------------------------------- candidate pools and labels

FEATURES = {
    "NeuN": {
        "mean_dab_od": {"neuron": (0.30, 0.08), "glia": (0.05, 0.03), "precipitate": (0.45, 0.12)},
        "p90_dab_od": {"neuron": (0.48, 0.12), "glia": (0.09, 0.05), "precipitate": (0.70, 0.15)},
        "dab_std": {"neuron": (0.09, 0.03), "glia": (0.03, 0.015), "precipitate": (0.18, 0.05)},
        "mean_hematoxylin_od": {"neuron": (0.22, 0.07), "glia": (0.55, 0.12), "precipitate": (0.12, 0.05)},
        "ring_dab_contrast": {"neuron": (0.18, 0.07), "glia": (0.02, 0.03), "precipitate": (0.30, 0.10)},
        "nucleus_fraction": {"neuron": (0.25, 0.08), "glia": (0.92, 0.06), "precipitate": (0.05, 0.05)},
        "cellpose_overlap": {"neuron": (0.75, 0.15), "glia": (0.90, 0.08), "precipitate": (0.10, 0.10)},
        "context_55um_neun_density": {"neuron": (310, 90), "glia": (260, 90), "precipitate": (240, 90)},
        "context_55um_mean_dab": {"neuron": (0.07, 0.02), "glia": (0.06, 0.02), "precipitate": (0.06, 0.02)},
        "context_55um_candidate_count": {"neuron": (14, 5), "glia": (13, 5), "precipitate": (11, 5)},
        "distance_to_tissue_edge_um": {"neuron": (320, 150), "glia": (300, 150), "precipitate": (220, 150)},
    },
    "6E10": {
        "mean_dab_od": {"compact": (0.42, 0.10), "diffuse": (0.24, 0.07), "small_plaque": (0.33, 0.09), "speck": (0.40, 0.12), "vascular": (0.30, 0.10)},
        "inner_mean_dab_od": {"compact": (0.70, 0.15), "diffuse": (0.26, 0.08), "small_plaque": (0.45, 0.12), "speck": (0.50, 0.14), "vascular": (0.25, 0.10)},
        "normalized_inner_dab": {"compact": (2.3, 0.5), "diffuse": (1.05, 0.3), "small_plaque": (1.6, 0.4), "speck": (1.9, 0.6), "vascular": (1.0, 0.4)},
        "normalized_mean_dab": {"compact": (1.5, 0.3), "diffuse": (0.95, 0.25), "small_plaque": (1.2, 0.3), "speck": (1.6, 0.5), "vascular": (1.1, 0.4)},
        "granularity": {"compact": (0.45, 0.10), "diffuse": (0.62, 0.10), "small_plaque": (0.40, 0.12), "speck": (0.15, 0.08), "vascular": (0.25, 0.10)},
        "dab_texture_std": {"compact": (0.20, 0.05), "diffuse": (0.12, 0.04), "small_plaque": (0.14, 0.05), "speck": (0.10, 0.05), "vascular": (0.12, 0.05)},
        "nuclei_inside": {"compact": (1.2, 1.0), "diffuse": (2.5, 1.5), "small_plaque": (0.3, 0.5), "speck": (0.05, 0.2), "vascular": (2.0, 1.2)},
        "lumen_contact_fraction": {"compact": (0.03, 0.04), "diffuse": (0.04, 0.05), "small_plaque": (0.03, 0.04), "speck": (0.05, 0.06), "vascular": (0.55, 0.2)},
        "edge_contact_fraction": {"compact": (0.01, 0.02), "diffuse": (0.02, 0.03), "small_plaque": (0.01, 0.02), "speck": (0.04, 0.05), "vascular": (0.05, 0.05)},
        "context_140um_dab_fraction": {"compact": (0.08, 0.04), "diffuse": (0.09, 0.04), "small_plaque": (0.06, 0.03), "speck": (0.04, 0.03), "vascular": (0.05, 0.03)},
        "context_140um_plaque_count": {"compact": (4, 2.5), "diffuse": (4, 2.5), "small_plaque": (3, 2), "speck": (2, 2), "vascular": (2, 2)},
        "hematoxylin_mean_od": {"compact": (0.20, 0.05), "diffuse": (0.18, 0.05), "small_plaque": (0.19, 0.05), "speck": (0.17, 0.05), "vascular": (0.26, 0.07)},
    },
    "AT8": {
        "mean_dab_od": {"tangle": (0.62, 0.14), "pretangle": (0.24, 0.07), "tau_dot": (0.40, 0.14)},
        "soma_dab_p90": {"tangle": (0.85, 0.16), "pretangle": (0.33, 0.08), "tau_dot": (0.55, 0.18)},
        "ring_dab_od": {"tangle": (0.55, 0.14), "pretangle": (0.26, 0.07), "tau_dot": (0.15, 0.08)},
        "nucleus_dab_ratio": {"tangle": (0.85, 0.15), "pretangle": (0.35, 0.12), "tau_dot": (0.20, 0.15)},
        "hematoxylin_mean_od": {"tangle": (0.15, 0.05), "pretangle": (0.30, 0.08), "tau_dot": (0.12, 0.05)},
        "local_background_dab_60um": {"tangle": (0.08, 0.03), "pretangle": (0.07, 0.03), "tau_dot": (0.08, 0.03)},
        "contrast_to_background": {"tangle": (7.5, 2.0), "pretangle": (3.2, 1.0), "tau_dot": (4.5, 2.0)},
        "thread_overlap_fraction": {"tangle": (0.05, 0.05), "pretangle": (0.06, 0.05), "tau_dot": (0.30, 0.15)},
        "context_55um_dab_fraction": {"tangle": (0.10, 0.05), "pretangle": (0.08, 0.04), "tau_dot": (0.09, 0.05)},
        "nucleus_area_um2": {"tangle": (45, 15), "pretangle": (58, 14), "tau_dot": (8, 6)},
        "from_dense_body": {"tangle": (0.8, 0.3), "pretangle": (0.2, 0.3), "tau_dot": (0.5, 0.4)},
    },
}
GEOMETRY = ("area_um2", "equivalent_diameter_um", "perimeter_um", "circularity", "solidity")


def feature_names(stain: str) -> list[str]:
    names = list(GEOMETRY) + list(FEATURES[stain])
    return ["soma_area_um2" if (stain == "AT8" and n == "area_um2") else n for n in names]


def pool_row(stain: str, o: dict, tile: dict, rng: np.random.Generator) -> dict:
    x, y = crop_xy(o, tile)
    diameter = o["diameter_um"]
    call = o["model_class"] if o["accepted"] else None
    row = {"x": x, "y": y, "radius_px": round(0.5 * diameter / PIXEL_UM, 1), "ai_value": ai_value(stain, call, o["mature"]),
           "ai_probability": o["probability"], "tma": tile["tma"], "complete": True,
           "candidate_source": {"NeuN": "dab_contour", "6E10": "dab_segment", "AT8": o.get("model_class") or "nucleus_ring"}[stain]}
    geometry = {"area_um2": o["area_um2"], "equivalent_diameter_um": diameter, "perimeter_um": o["perimeter_um"],
                "circularity": min(1.0, 4 * np.pi * o["area_um2"] / o["perimeter_um"] ** 2), "solidity": o["solidity"]}
    for name in GEOMETRY:
        row["soma_area_um2" if (stain == "AT8" and name == "area_um2") else name] = round(float(geometry[name] * rng.lognormal(0, 0.3)), 4)
    for name, by_kind in FEATURES[stain].items():
        mean, sd = by_kind[o["kind"]]
        row[name] = round(float(rng.normal(mean, 3 * sd)), 4)      # wide enough that the models make some mistakes
    if stain == "NeuN" and rng.random() < 0.08:
        row["cellpose_overlap"] = np.nan        # no Cellpose cell at this candidate
    return row


class Labeller:
    """Simulated annotators: mostly right, a little careless, and they mostly keep detection's call when it is plausible."""

    def __init__(self, rng: np.random.Generator):
        self.rng, self.clock, self.records = rng, START, []

    def tick(self, seconds: float = 6.0) -> float:
        self.clock += float(self.rng.exponential(seconds))
        return self.clock

    def value(self, stain: str, kind: str, ai: str) -> str:
        truth = TRUTH[kind]
        if self.rng.random() < 0.025:
            return str(self.rng.choice([c for c in CLASSES[stain] if c != truth]))
        if ai == truth or self.rng.random() < 0.12:
            return ai
        return truth

    def add(self, **record) -> None:
        self.records.append({"confirmed": False, "deleted": False, "session": "", "points": None, "x": None, "y": None, **record, "time": self.tick()})


def make_labels(root: Path, project, cores: list[dict], drawn: dict, tiles: pd.DataFrame, rng: np.random.Generator) -> dict:
    """Candidate pools for the prepared fields, labelling sessions and the label store. Returns the label records split
    into the earlier imported set and the later session labels (models are trained after each)."""
    reviews = root / "data" / "annotations"
    by_tile = {}
    for (core_id, stain), result in drawn.items():
        tile_rows = {t["tile_id"]: t for t in result["tiles"]}
        for o in result["objects"]:
            by_tile.setdefault(o["tile_id"], {"tile": tile_rows[o["tile_id"]], "objects": []})["objects"].append(o)
    core_meta = {c["core_id"]: c for c in cores}

    plan = {"NeuN": {"session": 12, "import": 8}, "6E10": {"session": 8, "import": 10}, "AT8": {"session": 10, "import": 8}}
    chosen = {}
    for stain, counts in plan.items():
        fields = tiles[tiles.stain == stain]
        picked = sessions.pick_fields(fields, set(), counts["session"] + counts["import"], seed=int(rng.integers(1000)))
        chosen[stain] = {"session": picked[:counts["session"]], "import": picked[counts["session"]:]}
        names = feature_names(stain)
        pool.folder(project, stain).mkdir(parents=True, exist_ok=True)
        (pool.folder(project, stain) / "features.json").write_text(json.dumps({"feature_names": names}))
        for tile_id in picked:
            entry = by_tile[tile_id]
            tile = {**entry["tile"], "tma": core_meta[tile_id.rsplit("_", 2)[0]]["tma"]}
            rows = [pool_row(stain, o, tile, rng) for o in entry["objects"] if o["candidate"]]
            frame = pd.DataFrame(rows)
            frame = frame[["x", "y", "radius_px", "ai_value", "ai_probability", "tma", "complete", "candidate_source", *names]]
            frame.to_csv(pool.path(project, stain, tile_id), index=False)

    people = Labeller(rng)

    def review(stain: str, tile_id: str, person: str, session: str, finish: bool = True, limit: int | None = None) -> None:
        entry = by_tile[tile_id]
        tile = entry["tile"]
        listed = [o for o in entry["objects"] if o["candidate"]]
        listed = sorted(listed, key=lambda o: (o["y"], o["x"]))[: limit or len(listed)]
        for o in listed:
            x, y = crop_xy(o, tile)
            ai = ai_value(stain, o["model_class"] if o["accepted"] else None, o["mature"])
            value = people.value(stain, o["kind"], ai)
            if value == ai and not finish:
                continue
            people.add(id=store.object_id("class", tile_id, x, y), source=person, stain=stain, task="class", tile_id=tile_id, x=x, y=y,
                       value=value, session=session, confirmed=value == ai)
            if positive(stain, value) and o["accepted"] and finish:
                points = cv2.approxPolyDP((np.asarray(o["outline"], np.int32) - tile["origin"]).reshape(-1, 1, 2), 2.5, True)[:, 0, :]
                edited = people.rng.random() < 0.05
                if edited:
                    points = points + people.rng.integers(-3, 4, points.shape)
                people.add(id=store.object_id("outline", tile_id, x, y), source=person, stain=stain, task="outline", tile_id=tile_id, x=x, y=y,
                           points=points.tolist(), session=session, confirmed=not edited)
        for o in entry["objects"]:
            if not o["candidate"] and people.rng.random() < 0.55:
                x, y = round(o["x"] - tile["origin"][0]), round(o["y"] - tile["origin"][1])
                people.add(id=store.object_id("class", tile_id, x, y), source=person, stain=stain, task="class", tile_id=tile_id, x=x, y=y,
                           value=TRUTH[o["kind"]], session=session)
        for tag, chance in zip(FIELD_TAGS, (0.08, 0.15, 0.06, 0.04, 0.02)):
            if finish and people.rng.random() < chance:
                people.add(id=f"tag:{tile_id}:{tag}", source=person, stain=stain, task="tag", tile_id=tile_id, value=tag, session=session)
        if finish:
            people.add(id=f"done:{tile_id}", source=person, stain=stain, task="done", tile_id=tile_id, session=session)

    # 1. an earlier label set, imported: random objects of other prepared fields
    for stain in STAINS:
        for tile_id in chosen[stain]["import"]:
            entry = by_tile[tile_id]
            for o in entry["objects"]:
                if o["candidate"] and people.rng.random() < 0.6:
                    x, y = crop_xy(o, entry["tile"])
                    value = TRUTH[o["kind"]] if people.rng.random() > 0.02 else NEGATIVE[stain]
                    if stain == "6E10" and positive(stain, value) and people.rng.random() < 0.3:
                        value = "plaque"
                    people.add(id=store.object_id("class", tile_id, x, y), source="import: pilot set", stain=stain, task="class",
                               tile_id=tile_id, x=x, y=y, value=value)
    imported = len(people.records)

    # 2. labelling sessions
    people.clock += 9 * 86400
    sessions_root = reviews
    created = {}

    def new_session(name: str, title: str, stain: str, kind: str, fields: list[str], per_field: int = 1, rule: str = "uncertain", open_: bool = True) -> None:
        sessions.create(sessions_root, name, title=title, stain=stain, kind=kind, tools=["class", "point", "outline", "tag"] if kind == "fields" else ["class"],
                        ai_view="full", per_field=per_field, rule=rule, fields=fields, instructions="")
        created[name] = people.clock
        sessions.update(sessions_root, name, open=open_, created=people.clock)

    neun = chosen["NeuN"]["session"]
    best = max(range(len(neun)), key=lambda k: by_tile[neun[k]]["tile"]["tissue_fraction"])
    neun[9], neun[best] = neun[best], neun[9]    # the field Ann is working on (the annotation screenshot) is full of tissue
    new_session("neun_review", "NeuN review: round 1", "NeuN", "fields", neun, per_field=2)
    for i, tile_id in enumerate(neun):
        if i < 9:
            review("NeuN", tile_id, "Ann", "neun_review")
        if i < 6:
            review("NeuN", tile_id, "Ben", "neun_review")
    review("NeuN", neun[9], "Ann", "neun_review", finish=False, limit=40)   # Ann's field in progress (the annotation screenshot)

    plaques = chosen["6E10"]["session"]
    new_session("plaque_review", "6E10 plaques: whole fields", "6E10", "fields", plaques, open_=False)
    for tile_id in plaques:
        review("6E10", tile_id, "Ann", "plaque_review")
    new_session("plaque_queue", "6E10: uncertain objects", "6E10", "queue", [], rule="uncertain")
    candidates = [(tile_id, o) for tile_id in chosen["6E10"]["import"] for o in by_tile[tile_id]["objects"] if o["candidate"]]
    order = sorted(range(len(candidates)), key=lambda k: (abs(candidates[k][1]["probability"] - 0.5) > 0.2, people.rng.random()))
    for person, picks in (("Ben", order[:90]), ("Carla", order[50:140])):
        for k in picks:
            tile_id, o = candidates[k]
            x, y = crop_xy(o, by_tile[tile_id]["tile"])
            ai = ai_value("6E10", o["model_class"] if o["accepted"] else None, o["mature"])
            people.add(id=store.object_id("class", tile_id, x, y), source=person, stain="6E10", task="class", tile_id=tile_id, x=x, y=y,
                       value=people.value("6E10", o["kind"], ai), session="plaque_queue")

    tau = chosen["AT8"]["session"]
    new_session("at8_review", "AT8 tangles and pretangles", "AT8", "fields", tau)
    for i, tile_id in enumerate(tau):
        if i < 7:
            review("AT8", tile_id, "Carla", "at8_review")
        elif i < 8:
            review("AT8", tile_id, "Ben", "at8_review")
    return {"imported": people.records[:imported], "sessions": people.records[imported:], "annotate": {"session": "neun_review", "tile_id": neun[9]}}


# ---------------------------------------------------------------- models

PLACEHOLDER = "Placeholder written by tools/demo/make_demo_project.py: the synthetic demo project ships no model weights.\n"


def train_models(root: Path, labels: dict) -> float:
    """Train each stain model twice with stainid's own training (after the imported labels, then after the sessions),
    under fixed folder names, and switch to the models a user would have chosen. Returns when the last one was made."""
    trained = root / "data" / "models" / "trained"
    reviews = root / "data" / "annotations"
    store.append(reviews, labels["imported"])
    use = {}
    for round_, (records, when) in enumerate(((None, START + 4 * 86400), (labels["sessions"], labels["sessions"][-1]["time"] + 3600))):
        if records:
            store.append(reviews, records)
        for i, stain in enumerate(STAINS):
            project = load_project(root)
            folder = train_model(project, stain)
            stamp = when + 600 * i
            final = trained / f"{stain}_{time.strftime('%Y%m%d-%H%M%S', time.gmtime(stamp))}"
            folder.rename(final)
            report = json.loads((final / "report.json").read_text())
            report["created"] = stamp
            (final / "report.json").write_text(json.dumps(report, indent=1))
            if round_ == 0 or stain != "AT8":   # keep the first AT8 model in use: the Models page offers the newer one
                key = {"NeuN": "neun", "6E10": "amyloid", "AT8": "tau"}[stain]
                use[key] = project.relative(final / "bundle.joblib")
                save_project(load_project(root), {"models": {key: use[key]}})
    return stamp


# ---------------------------------------------------------------- pictures of slides, grids and QC

def slide_pictures(root: Path, cores: list[dict], drawn: dict) -> None:
    cell, margin = 240, 60
    for t in range(1, TMAS + 1):
        for stain in STAINS:
            slide = np.full((2 * margin + ROWS * cell + 200, 2 * margin + COLUMNS * cell + 600, 3), (238, 238, 236), np.uint8)
            grid = slide.copy()
            for core in (c for c in cores if c["tma"] == str(t)):
                column, row = ord(core["core_label"][0]) - 65, int(core["core_label"][2:]) - 1
                x, y = margin + 300 + column * cell, margin + 100 + row * cell
                thumb = cv2.resize(drawn[(core["core_id"], stain)]["thumb"], (cell - 40, cell - 40), interpolation=cv2.INTER_AREA)
                for image in (slide, grid):
                    image[y + 20: y + cell - 20, x + 20: x + cell - 20] = thumb
                cv2.circle(grid, (x + cell // 2, y + cell // 2), cell // 2 - 22, (40, 150, 40), 3)
                cv2.putText(grid, core["core_label"], (x + 24, y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 40, 40), 2)
            name = f"{PREFIX}{t} {stain}"
            (root / "data" / "slides").mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(root / "data" / "slides" / f"{name}.tif"), cv2.cvtColor(slide, cv2.COLOR_RGB2BGR))
            (root / "data" / "qc" / "grids").mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(root / "data" / "qc" / "grids" / f"{name}.jpg"), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
    (root / "data" / "qc" / "cores").mkdir(parents=True, exist_ok=True)
    for (core_id, stain), result in drawn.items():
        cv2.imwrite(str(root / "data" / "qc" / "cores" / f"{core_id}_{stain}.jpg"), cv2.cvtColor(result["thumb"], cv2.COLOR_RGB2BGR))


# ---------------------------------------------------------------- the v2 detection: candidate store, stage manifests, results

REJECT_RULES = {  # (stain, kind): the rule that rejects some of these before the model, and how often
    ("6E10", "speck"): ("rule:speck", 0.7), ("6E10", "vascular"): ("rule:vascular_or_edge", 0.6),
    ("AT8", "tau_dot"): ("rule:low_contrast", 0.35), ("NeuN", "glia"): ("rule:pale_nucleus", 0.3), ("NeuN", "precipitate"): ("rule:shape", 0.4),
}
SIZES = (0.8, 1.08, 1.38)  # the three Segment Anything masks of a candidate (its smoothed hull, small to large); the stain outline is the drawn one


def _scaled(outline, x: float, y: float, factor: float, hull: bool = False) -> np.ndarray:
    """An object's outline (or, with `hull`, its convex hull: the smoother look of a Segment Anything mask), simplified,
    grown or shrunk about its centre."""
    ring = np.asarray(outline, float)
    if hull:
        ring = cv2.convexHull(ring.astype(np.float32).reshape(-1, 1, 2))[:, 0, :].astype(float)
    simple = cv2.approxPolyDP(ring.astype(np.float32).reshape(-1, 1, 2), 1.2, True)[:, 0, :].astype(float)
    if len(simple) >= 3:
        ring = simple
    return np.c_[x + (ring[:, 0] - x) * factor, y + (ring[:, 1] - y) * factor]


def _call_class(stain: str, o: dict) -> tuple[str, str]:
    """An accepted call as label-vocabulary ids (identity, type)."""
    if stain == "NeuN":
        return "NeuN+ neuron", "strong" if o["dab"] > 0.55 else "weak"
    if stain == "6E10":
        return "plaque", {"compact": "compact", "diffuse": "diffuse", "small_plaque": "small"}.get(o["model_class"] or "", "diffuse")
    return "tau+ neuron", "mature tangle" if o["mature"] else "pretangle"


def candidate_store_outputs(project, drawn: dict, rng: np.random.Generator) -> dict[tuple[str, str], int]:
    """The v2 candidate store (`stainid.candidates`, read by the annotation rounds and the viewer): every candidate the
    simulated detector saw in each core image, accepted and rejected (not the missed objects, which people add), with
    three outline sizes, the call, its reason and class. Returns the number of candidates per (core, stain)."""
    from stainid import candidates as v2store

    out = {}
    for (core_id, stain), result in sorted(drawn.items()):
        objects = [o for o in result["objects"] if o["candidate"]]
        ids = v2store.unique_ids(stain, core_id, [o["x"] for o in objects], [o["y"] for o in objects])
        rows = []
        for cid, o in zip(ids, objects):
            masks = [_scaled(o["outline"], o["x"], o["y"], f, hull=True) for f in SIZES]
            own = _scaled(o["outline"], o["x"], o["y"], 1.0)  # the stain-based outline
            rule, chance = REJECT_RULES.get((stain, o["kind"]), ("", 0.0))
            if o["accepted"]:
                decision, reason = "accepted", "accepted"
                klass, subtype = _call_class(stain, o)
            else:
                decision, klass, subtype = "rejected", "", ""
                reason = rule if rule and rng.random() < chance else "rejected_by_model"
            level = float(rng.choice([0.5, 0.75, 1.0, 1.5], p=[0.1, 0.2, 0.5, 0.2]))
            rows.append({
                "candidate_id": cid, "x_core": round(o["x"], 1), "y_core": round(o["y"], 1), "tile_id": o["tile_id"],
                "area_um2": round(o["area_um2"], 2), "radius_px": round(0.5 * o["diameter_um"] / PIXEL_UM, 1),
                "sources": {"NeuN": "legacy_profile", "6E10": "legacy_deposit", "AT8": "ring"}[stain] if level >= 1.0 else f"dab@{level:g}",
                "proposal_threshold": level, "equivalent_diameter_um": round(o["diameter_um"], 2), "solidity": round(o["solidity"], 4),
                "perimeter_um": round(o["perimeter_um"], 2), "mean_dab_od": round(0.08 + 0.4 * o["dab"], 4),
                "outline": v2store.encode_points(own), "sam_outline": v2store.encode_points(masks[1]), "sam_status": "ok",
                "sam_masks": "|".join(v2store.encode_points(m) for m in masks),
                "sam_scores": "|".join(f"{v:.3f}" for v in (rng.uniform(0.72, 0.86), rng.uniform(0.86, 0.97), rng.uniform(0.7, 0.88))),
                "sam_stability": "|".join(f"{v:.3f}" for v in rng.uniform(0.8, 0.98, 3)), "sam_default": 1,
                "probability": np.nan if reason.startswith("rule:") else o["probability"], "threshold": THRESHOLD[stain], "model_id": "demo",
                "decision": decision, "reason": reason, "class": klass, "subtype": subtype})
        frame = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["candidate_id", "x_core", "y_core", "tile_id", "decision", "class", "subtype"])
        v2store.write_table(v2store.root(project) / "candidates" / stain / f"{core_id}.csv.gz", frame)
        # and each stage's own table, as the stages write them (proposals, calls, outlines: the store is their join)
        calls = [c for c in ("candidate_id", *v2store.CALL_COLUMNS) if c in frame]
        shapes = [c for c in ("candidate_id", *v2store.OUTLINE_COLUMNS) if c in frame]
        proposed = [c for c in frame.columns if c == "candidate_id" or (c not in calls and c not in shapes)]
        for kind, columns in (("proposals", proposed), ("calls", calls), ("outlines", shapes)):
            v2store.write_table(v2store.root(project) / kind / stain / f"{core_id}.csv.gz", frame[columns])
        out[(core_id, stain)] = len(frame)
    return out


def v2_manifests(project, counts: dict[tuple[str, str], int]) -> None:
    """Stage manifests with the digests the stages would record now (as `stainid.v2.run.status` works them out), so every
    v2 stage shows as done and up to date; and the v2 results tables (the same as the current ones: the demo has one
    detection)."""
    from stainid.v2 import classify, manifest, outlines, proposals
    from stainid.v2.run import groups, proposal_inputs

    calibration, nuclei = proposal_inputs(project)
    models = {stain: classify.load_model(project, stain) for stain in STAINS}
    for (core_id, stain), tiles in groups(project, STAINS).items():
        threshold = calibration[(tiles[0]["tma"], stain)]
        n = counts.get((core_id, stain), 0)
        manifest.write(project, "proposals", stain, core_id,
                       proposals.expected(project, stain, core_id, tiles, threshold, nuclei), candidates=n)
        manifest.write(project, "calls", stain, core_id, classify.expected(project, stain, core_id, models[stain]), candidates=n)
        manifest.write(project, "outlines", stain, core_id, outlines.expected(project, stain, core_id), outlined=n)
        manifest.write(project, "candidates", stain, core_id, classify.assembled_expected(project, stain, core_id), candidates=n)
    measures = project.output("v2") / "measures"
    measures.mkdir(parents=True, exist_ok=True)
    for name in ("results_donor_region.csv", "results_core.csv"):
        shutil.copyfile(project.output("tables") / name, measures / name)


# ---------------------------------------------------------------- the runs of the steps (the Run page's history)

def job_records(project, cores: list[dict], tiles: int, models_at: float) -> None:
    """Finished runs of the steps that "made" the demo's tables, with plausible times, as the app's job store keeps them."""
    from stainid.jobs import APP, Job, JobStore
    from stainid.workflows.steps import BY_ID, to_argv

    jobs = JobStore(project.output("jobs"))
    slides, images = TMAS * len(STAINS), len(cores) * len(STAINS)
    setup = START - 11 * 86400
    plan = [  # step, start, seconds, units done
        ("dearray", setup, 9 * 62, slides), ("export", setup + 700, 9 * 95, slides), ("qc", setup + 1700, 170, images),
        ("tiles", setup + 1900, 95, images), ("calibrate", setup + 2100, 9 * 41, slides),
        ("nuclei", setup + 3000, int(tiles * 4.1), tiles), ("v2_proposals", setup + 86400, int(images * 96), images),
        ("v2_outlines", setup + 2 * 86400, int(images * 31), images), ("train", models_at - 240, 52, 3),
        ("v2_calls", models_at + 900, 140, images), ("v2_measures", models_at + 1100, 210, 1),
    ]
    for n, (step, start, seconds, units) in enumerate(plan):
        options = {o.key: o.default for o in BY_ID[step].options}
        job = Job(id=time.strftime("%Y%m%d-%H%M%S-", time.localtime(start)) + f"de{n:04x}", workflow=step, options=options,
                  argv=to_argv(step, options), status="finished", created=start - 2, started=start, ended=start + seconds, returncode=0, source=APP)
        jobs.save(job)
        lines = [f"[{k}/{units}] {BY_ID[step].title}" for k in sorted({max(1, units * i // 4) for i in range(1, 5)})]
        jobs.log_path(job).write_text("\n".join([f"{BY_ID[step].title}: {units} to do", *lines, "done"]) + "\n")
        jobs.exit_path(job.id).write_text("0")


# ---------------------------------------------------------------- the annotation rounds (over the v2 candidate store)

def round_sessions(project, tiles: pd.DataFrame, drawn: dict, rng: np.random.Generator, when: float) -> dict:
    """A "Fix the field" session of AT8, made as the Labels page makes it (`stainid.api.routes.rounds.create_round_session`),
    with Ann's saved work on one field she has not finished
    (calls changed, candidates removed, a rejected one taken back, one unsure, one object she added), fields finished by
    Ben and Carla, and the tutorial done by the three of them (a new name still gets it)."""
    from stainid.labels import candidate_store, frames, rounds
    from stainid.labels import vocabulary as vocab

    reviews = project.output("reviews")
    stain = "AT8"
    words = vocab.of(project, stain)
    remove = candidate_store.negative_class(words)
    unsure = next(c.id for c in words.classes if c.role == vocab.SKIP)
    mine = tiles[tiles.stain == stain]
    taken = set(sessions.taken(reviews)) | set(store.reviewed_fields(store.current(reviews), stain))
    free = mine[~mine.tile_id.isin(taken)]
    common = {"stain": stain, "instructions": "", "subtypes": True, "ai_view": "full", "source": rounds.SOURCE, "level": None,
              "labels_needed": 1, "per_field": 1}
    missed = {}
    for (core_id, s), result in drawn.items():
        if s == stain:
            for o in result["objects"]:
                if not o["candidate"]:
                    missed.setdefault(o["tile_id"], []).append(o)

    def own(tile_id: str) -> tuple[pd.DataFrame, tuple[int, int]]:
        tile = mine[mine.tile_id == tile_id].iloc[0]
        found = candidate_store.read_core(project, stain, str(tile.core_id))
        found = found[found.tile_id == tile_id].rename(columns={"class": "klass"})
        x0, y0 = frames.field_origin(tile, project.context_px)
        return found.assign(x=(found.x_core - x0).round(1), y=(found.y_core - y0).round(1)), (x0, y0)

    # round 2: six fields; the first (most candidates, and an object detection missed) is Ann's field in progress
    fields = [str(t) for t in sessions.pick_fields(free, set(), 14, seed=int(rng.integers(1000)))]
    fields.sort(key=lambda t: (not missed.get(t), -len(own(t)[0])))
    fix = fields[:6]
    fix_name = "at8_fix_fields"
    sessions.create(reviews, fix_name, kind="fields", tools=["class", "point", "outline", "tag"], rule="random", fields=fix, scope=fix,
                    title="AT8: fix the fields", round=2, **common)
    sessions.update(reviews, fix_name, open=True, created=when)

    records, clock = [], [when + 3600]

    def add(**record) -> None:
        clock[0] += float(rng.exponential(9.0))
        records.append({"confirmed": False, "deleted": False, "points": None, "subtype": None, "candidate": None, "stain": stain, **record,
                        "time": clock[0]})

    def call(person: str, session: str, tile_id: str, row, value: str, subtype: str = "", confirmed: bool = False) -> None:
        x, y = round(float(row.x), 1), round(float(row.y), 1)
        add(id=store.object_id("class", tile_id, x, y), source=person, task="class", session=session, tile_id=tile_id, x=x, y=y,
            value=value, subtype=subtype or None, candidate=row.candidate_id, confirmed=confirmed)

    # Ann, on the first field: her changes are saved, the field is not finished
    ann_field = fix[0]
    found, (x0, y0) = own(ann_field)
    picks = list(found[found.decision == "accepted"].sort_values("probability").itertuples())
    rejected = list(found[found.decision != "accepted"].sort_values("probability", ascending=False, na_position="last").itertuples())
    for row, value in zip(picks[:2], ("thread", "neuritic plaque")):
        call("Ann", fix_name, ann_field, row, value)
    for row in picks[2:4]:
        call("Ann", fix_name, ann_field, row, remove)
    if len(picks) > 4:  # another type, and the larger outline
        row = picks[4]
        call("Ann", fix_name, ann_field, row, "tau+ neuron", "pretangle" if row.subtype == "mature tangle" else "mature tangle")
        add(id=store.object_id("outline", ann_field, row.x, row.y), source="Ann", task="outline", session=fix_name, tile_id=ann_field, x=row.x,
            y=row.y, value="mask 2", points=[[round(px - x0, 1), round(py - y0, 1)] for px, py in row.sam_masks[2]], candidate=row.candidate_id)
    if len(picks) > 5:
        call("Ann", fix_name, ann_field, picks[5], unsure)
    for row in rejected[:1]:  # a rejected candidate taken back
        call("Ann", fix_name, ann_field, row, "tau+ neuron", "pretangle")
    for o in missed.get(ann_field, [])[:1]:  # an object detection missed, added with one click
        x, y = round(o["x"] - x0), round(o["y"] - y0)
        add(id=store.object_id("class", ann_field, x, y), source="Ann", task="class", session=fix_name, tile_id=ann_field, x=x, y=y,
            value="tau+ neuron", subtype="pretangle")
        add(id=store.object_id("outline", ann_field, x, y), source="Ann", task="outline", session=fix_name, tile_id=ann_field, x=x, y=y,
            points=(_scaled(o["outline"], o["x"], o["y"], 1.0) - [x0, y0]).round(1).tolist())

    # Ben and Carla finished one field each: the calls they left as they were count as checked
    for person, tile_id in (("Ben", fix[1]), ("Carla", fix[2])):
        for row in own(tile_id)[0].itertuples():
            identity, subtype = candidate_store.ai_call(words, row.decision, row.klass, row.subtype)
            changed = rng.random() < 0.06 and identity != remove
            call(person, fix_name, tile_id, row, remove if changed else identity, "" if changed else subtype, confirmed=not changed)
        add(id=rounds.done_id({"round": 2, "kind": "fields"}, tile_id), source=person, task="done", session=fix_name, tile_id=tile_id, value=None)

    store.append(reviews, records)
    (reviews / "tutorials.json").write_text(json.dumps({p: when - 3600 + 60 * i for i, p in enumerate(PEOPLE)}, indent=1))
    return {"fix": fix_name, "ann_field": ann_field}


# ---------------------------------------------------------------- main

def build(out: Path, seed: int, workers: int) -> None:
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty; choose a new folder")
    out.mkdir(parents=True, exist_ok=True)
    root = out.resolve()
    rng = np.random.default_rng(seed)
    donors, cores = design_cohort(rng)
    print(f"cohort: {len(donors)} donors, {len(cores)} cores on {TMAS} TMAs", flush=True)

    save_project(Project(root=root, config=DEFAULTS), {
        "name": NAME, "tma": {"prefix": PREFIX, "rows": ROWS, "columns": COLUMNS}, "groups": GROUPS})
    project = load_project(root)
    for folder in ("data/analysis/tile_nuclei", "data/jobs", "data/annotations"):
        (root / folder).mkdir(parents=True, exist_ok=True)
    (root / "data" / "annotations" / "cohort_manual_exclusions.json").write_text("{}\n")

    jobs = [{"seed": int(rng.integers(1 << 31)), "stain": stain, "core_id": core["core_id"], "severity": core["severity"][stain],
             "shape_seed": core["shape_seed"], "neuron_density": core["neuron_density"], "root": str(root),
             "image_path": f"data/cores/{PREFIX}{core['tma']}/{core['core_id']}_{stain}.jpg", "nuclei_dir": "data/analysis/tile_nuclei",
             "masks_dir": f"data/analysis/cohort_object_masks/{stain}/labels"} for core in cores for stain in STAINS]
    drawn = {}
    started = time.time()
    with ProcessPoolExecutor(workers) as executor:
        for number, result in enumerate(executor.map(make_core, jobs), start=1):
            drawn[result["key"]] = result
            print(f"[{number}/{len(jobs)}] {' '.join(result['key'])}: {len(result['objects'])} objects, {len(result['tiles'])} tiles "
                  f"({time.time() - started:.0f} s)", flush=True)
    image_of = {(j["core_id"], j["stain"]): j["image_path"] for j in jobs}

    # set-up tables: slides, core table, TMA map, donors, core images and tiles
    slides = [{"slide_path": str(root / "data" / "slides" / f"{PREFIX}{t} {stain}.tif"), "tma": str(t), "stain": stain}
              for t in range(1, TMAS + 1) for stain in STAINS]
    write_csv(project.input("slides_table"), slides)
    manifest, images, whole = [], [], []
    for core in cores:
        column, row = ord(core["core_label"][0]) - 65, int(core["core_label"][2:]) - 1
        identity = {k: core[k] for k in ("donor_id", "sample_region_id", "region", "disease_group", "technical_replicate")}
        for stain in STAINS:
            result = drawn[(core["core_id"], stain)]
            fraction = result["tissue_fraction"]
            status = "complete" if fraction >= 0.45 else "partial"
            manifest.append({"slide": f"{PREFIX}{core['tma']} {stain}", "slide_path": str(root / "data" / "slides" / f"{PREFIX}{core['tma']} {stain}.tif"),
                             "tma": core["tma"], "stain": stain, "row": row + 1, "column_label": core["core_label"][0], "column": column + 1,
                             "core_label": core["core_label"], "center_x_px": 9000 + column * 6200, "center_y_px": 8000 + row * 6200,
                             "diameter_x_px": CORE_PX, "diameter_y_px": CORE_PX, "provisional_tissue_fraction": round(fraction, 3),
                             "provisional_tissue_status": status, "source_pixel_width_um": PIXEL_UM, "source_pixel_height_um": PIXEL_UM,
                             **identity, "cerad": core["cerad"], "braak": core["braak"], "core_role": "biological", "map_qc": "ok",
                             "output_path": image_of[(core["core_id"], stain)], "export_status": "complete", "export_downsample": 1,
                             "output_width_px": CORE_PX, "output_height_px": CORE_PX, "tissue_fraction": round(fraction, 4),
                             "tissue_status": status, "review_required": "false", "low_focus_fraction": round(float(rng.uniform(0, 0.03)), 4)})
            base = {"core_id": core["core_id"], "tma": core["tma"], "core_label": core["core_label"], **identity, "stain": stain,
                    "image_path": image_of[(core["core_id"], stain)], "pixel_width_um": PIXEL_UM, "pixel_height_um": PIXEL_UM}
            images.append({**{k: base[k] for k in ("core_id", "tma", "core_label", "donor_id", "sample_region_id", "region", "disease_group",
                                                   "technical_replicate", "stain", "image_path")},
                           "native_width_px": CORE_PX, "native_height_px": CORE_PX, "tissue_fraction": round(fraction, 4), "tissue_status": status})
            for t in result["tiles"]:
                t["tma"] = core["tma"]
                box = {"x_px": t["x_px"], "y_px": t["y_px"], "width_px": t["width_px"], "height_px": t["height_px"]}
                whole.append({"tile_id": t["tile_id"], **base, "selection_role": "whole_core_grid", "selection_order": t["selection_order"],
                              "nested_sample": "whole_core", **box, "preview_tissue_fraction": t["tissue_fraction"]})
    write_csv(project.input("core_manifest"), manifest)
    write_csv(project.input("tma_layout"), [{"tma": c["tma"], "core_label": c["core_label"], "donor_id": c["donor_id"], "region": c["region"],
                                             "disease_group": c["disease_group"], "tissue_control": "", "cerad": c["cerad"], "braak": c["braak"]} for c in cores])
    write_csv(project.input("donor_metadata"), [{k: d[k] for k in ("donor_id", "tma", "disease_group", "age", "sex", "apoe_e4_count", "pmd_hours")}
                                                for d in donors])
    write_csv(project.input("core_images"), images)
    write_csv(project.input("whole_core_manifest"), whole)
    tiles = labelling_tiles(project)  # the fields labelling sessions are drawn from

    print("detections, outlines and pictures", flush=True)
    detection_outputs(root, cores, drawn, rng)
    slide_pictures(root, cores, drawn)
    for path, kind in (("data/models/cellpose/cpsam_v2", "file"), ("data/models/sam/sam_vit_b_01ec64.pth", "file"),
                       ("data/models/foundation/hf/README.txt", "file"), ("data/models/plaque_cnn/README.txt", "file")):
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_text(PLACEHOLDER)

    print("candidate pools and labels", flush=True)
    labels = make_labels(root, project, cores, drawn, tiles, rng)

    os.chdir(root)    # stainid steps resolve table paths against the project folder
    print("stain calibration (stainid calibrate)", flush=True)
    calibrate_slides(load_project(root))
    print("model training (stainid train)", flush=True)
    models_at = train_models(root, labels)
    print("results tables (stainid summarize)", flush=True)
    for level in ("sample_region_id", "core_id"):
        summarize_results(load_project(root), level)
    print("v2 detection: candidate store, stage manifests and results tables", flush=True)
    project = load_project(root)
    counts = candidate_store_outputs(project, drawn, rng)
    v2_manifests(project, counts)
    job_records(project, cores, len(whole), models_at)
    print("annotation rounds", flush=True)
    rounds_made = round_sessions(project, labelling_tiles(project), drawn, rng, models_at + 2 * 86400)
    shutil.rmtree(root / "data" / "cache", ignore_errors=True)
    (root / "demo.json").write_text(json.dumps({"seed": seed, "annotate": labels["annotate"], "rounds": rounds_made}, indent=1))
    print(f"done: {root} ({time.time() - started:.0f} s)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("out_dir", type=Path, help="new (empty) folder for the demo project")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--workers", type=int, default=min(6, os.cpu_count() or 1), help="core images drawn at once (about 1.5 GB of memory each)")
    args = parser.parse_args()
    build(args.out_dir, args.seed, args.workers)


if __name__ == "__main__":
    main()
