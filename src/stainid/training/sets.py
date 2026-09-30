"""Training sets: candidate objects from randomly chosen fields, shown blinded for labelling, with their features saved."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load

from stainid.project import Project
from stainid.review import store
from stainid.review.sampling import DEFAULT_FOV_UM
from stainid.tables import read_csv
from stainid.training import candidates

LABELS = {
    "NeuN": ["NeuN+ neuron", "not a neuron", "artifact or unclear", "unsure"],
    "6E10": ["compact plaque", "diffuse plaque", "not a plaque", "unsure"],
    "AT8": ["tau+ neuron: tangle", "tau+ neuron: pretangle", "not a tau+ neuron", "unsure"],
}
POSITIVE = {"NeuN": {"NeuN+ neuron"}, "6E10": {"compact plaque", "diffuse plaque"}, "AT8": {"tau+ neuron: tangle", "tau+ neuron: pretangle"}}
SKIP = {"unsure"}
INSTRUCTIONS = {
    "NeuN": "Look at the object under the cross. NeuN+ neuron: a brown (DAB) neuronal cell body or nucleus. "
            "Not a neuron: blue-only nuclei (glia, vessels) or background. Artifact: folds, edges, stain precipitate.",
    "6E10": "Look at the deposit under the cross. Compact plaque: a dense, well-defined brown core. Diffuse plaque: loose, "
            "fluffy brown deposit without a dense core. Not a plaque: vessel wall, cell, edge, fold or stain precipitate.",
    "AT8": "Look at the cell under the cross. Tangle: a dense brown flame- or globe-shaped neuron. Pretangle: a neuron with "
           "diffuse brown cytoplasm around its nucleus. Not a tau+ neuron: threads, dots, glia, vessels or background.",
}
MODEL_KEY = {"NeuN": "neun", "6E10": "amyloid", "AT8": "tau"}
KEEP = ("centroid_x_px", "centroid_y_px", "candidate_source", "equivalent_diameter_um", "normalized_inner_dab", "soma_dab_p90")


def current_probability(project: Project, stain: str, matrix: np.ndarray, names: list[str]) -> np.ndarray:
    path = project.model(MODEL_KEY[stain])
    if not path.exists():
        return np.full(len(matrix), np.nan)
    bundle = load(path)
    if stain == "NeuN" and names == bundle["feature_names"]:
        return bundle["classifier"].predict_proba(matrix[:, bundle["usable_features"]])[:, 1]
    if stain == "6E10" and names == bundle["identity_feature_names"]:
        return bundle["identity_classifier"].predict_proba(matrix)[:, 1]
    if stain == "AT8" and names == bundle["feature_names"]:
        return bundle["classifier"].predict_proba(matrix)[:, 1]
    return np.full(len(matrix), np.nan)


def with_detections(project: Project, stain: str, tiles: list[dict[str, str]]) -> list[dict[str, str]]:
    """Fields where the current pipeline already found objects (plaques / tau+ neurons are rare in many fields)."""
    column = {"AT8": "tau_neuron_count", "6E10": "v2_plaque_count"}.get(stain)
    parts = project.output("fields") / "parts"
    if column is None or not parts.exists():
        return tiles
    counts = {r["tile_id"]: float(r[column]) for p in parts.glob(f"*_{stain}_features.csv") for r in read_csv(p)}
    return [t for t in tiles if counts.get(t["tile_id"], 0) > 0] or tiles


def unused_fields(project: Project, stain: str, tiles: list[dict[str, str]]) -> list[dict[str, str]]:
    """Prefer fields no earlier training set of this stain has used, so a new set brings new objects."""
    root = project.output("reviews") / "training"
    keys = [pd.read_csv(key, dtype=str) for key in root.glob("*/key.csv")] if root.exists() else []
    used = {t for key in keys for t in key.tile_id[key.stain == stain]}
    return [t for t in tiles if t["tile_id"] not in used] or tiles


def pick_fields(tiles: list[dict[str, str]], n: int, seed: int) -> list[dict[str, str]]:
    """Spread fields over TMAs and diagnostic groups: shuffle, then take one field per (TMA, group) in turn."""
    rng = np.random.default_rng(seed)
    order = [tiles[i] for i in rng.permutation(len(tiles))]
    buckets: dict[tuple[str, str], list] = {}
    for tile in order:
        buckets.setdefault((tile["tma"], tile.get("disease_group", "")), []).append(tile)
    picked = []
    while len(picked) < n and any(buckets.values()):
        for bucket in buckets.values():
            if bucket and len(picked) < n:
                picked.append(bucket.pop())
    return picked


def choose(probability: np.ndarray, k: int, strategy: str, rng: np.random.Generator) -> np.ndarray:
    index = np.arange(len(probability))
    if len(index) <= k:
        return index
    if strategy == "uncertain" and np.isfinite(probability).any():
        uncertain = index[(probability >= 0.2) & (probability <= 0.8)]
        first = rng.choice(uncertain, min(len(uncertain), k // 2), replace=False) if len(uncertain) else np.array([], int)
        rest = rng.choice(np.setdiff1d(index, first), k - len(first), replace=False)
        return np.concatenate([first, rest])
    return rng.choice(index, k, replace=False)


def create_training_set(project: Project, stain: str, name: str, fields: int = 12, per_field: int = 10, strategy: str = "uncertain",
                        enrich: bool = True, seed: int = 0) -> Path:
    folder_name = f"training/{name}"
    if (project.output("reviews") / folder_name).exists():
        raise FileExistsError(f"A training set called {name} already exists")
    tiles = [t for t in read_csv(project.input("tile_manifest")) if t["stain"] == stain]
    tiles = unused_fields(project, stain, with_detections(project, stain, tiles) if enrich else tiles)
    tiles = pick_fields(tiles, fields, seed)
    loader = candidates.FieldLoader(project)
    rng = np.random.default_rng(seed)
    items, features, embeddings, names = [], [], [], None
    for number, tile in enumerate(tiles, start=1):
        field = loader.load(tile)
        rows, matrix, field_names = candidates.EXTRACT[stain](loader, field)
        if rows:
            names = names or field_names
            probability = current_probability(project, stain, matrix, names)
            chosen = choose(probability, per_field, strategy, rng)
            picked = [rows[i] for i in chosen]
            if stain == "6E10":
                emb, wong = candidates.amyloid_extras(field, picked, project)
                if emb is not None:
                    embeddings.append(emb)
                for row, value in zip(picked, wong):
                    row["wong_plaque_probability"] = float(value)
            for i, row in zip(chosen, picked):
                items.append({"tile_id": tile["tile_id"], "stain": stain, "x": round(float(row["centroid_x_px"]), 1), "y": round(float(row["centroid_y_px"]), 1),
                              "tma": tile["tma"], "probability": float(probability[i]), "disease_group": tile.get("disease_group", ""),
                              "donor_id": tile.get("donor_id", ""), "sample_region_id": tile.get("sample_region_id", ""), "region": tile.get("region", "")})
                features.append({**dict(zip(names, matrix[i])), **{k: row[k] for k in KEEP if k in row},
                                 **({"wong_plaque_probability": row["wong_plaque_probability"]} if stain == "6E10" else {})})
        print(f"[{number}/{len(tiles)}] {tile['tile_id']}: {len(rows)} candidates", flush=True)
    if not items:
        raise ValueError(f"No {stain} candidates found in the chosen fields")
    frame = pd.DataFrame(items)
    frame["item"] = range(len(frame))
    folder = store.create_set(project.output("reviews"), folder_name, name, stain, frame, LABELS[stain], INSTRUCTIONS[stain], DEFAULT_FOV_UM[stain], seed)
    key = pd.read_csv(folder / "key.csv")
    table = pd.DataFrame(features).iloc[key["item"].to_numpy()].reset_index(drop=True)
    table.insert(0, "review_id", key["review_id"])
    table.to_csv(folder / "features.csv", index=False)
    if embeddings:
        np.save(folder / "embeddings.npy", np.concatenate(embeddings)[key["item"].to_numpy()])
    meta = json.loads((folder / "meta.json").read_text())
    (folder / "meta.json").write_text(json.dumps({**meta, "purpose": "training", "feature_names": names, "strategy": strategy, "fields": len(tiles)}, indent=1))
    print(f"{len(frame)} candidates to label -> {project.relative(folder)}", flush=True)
    return folder
