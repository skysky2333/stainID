"""Bring labels made before the label store into it, once: training sets, the CNN-outlier adjudication and the
white-matter field review. Originals stay where they are; each set becomes one source (`import: <name>`).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from stainid.labels import pool, store
from stainid.labels.tasks import NEGATIVE
from stainid.project import Project
from stainid.tables import read_csv

WHITE_MATTER_TAGS = {"white_matter": "white matter", "mixed": "mixed grey and white matter", "unusable": "unusable"}


def _done(root, source: str) -> bool:
    labels = store.history(root)
    return bool((labels.source == source).any())


def training_sets(project: Project) -> int:
    root = project.output("reviews")
    tiles = {t["tile_id"]: t for t in read_csv(project.input("tile_manifest"))}
    total = 0
    for meta_file in sorted((root / "training").glob("*/meta.json")):
        folder = meta_file.parent
        source = f"import: {folder.name}"
        if _done(root, source) or not (folder / "labels.csv").exists():
            continue
        meta = json.loads(meta_file.read_text())
        stain, names = meta["stain"], meta["feature_names"]
        key = pd.read_csv(folder / "key.csv", dtype={"review_id": str, "tile_id": str, "tma": str})
        labels = pd.read_csv(folder / "labels.csv", dtype=str).drop_duplicates("review_id", keep="last")
        features = pd.read_csv(folder / "features.csv", dtype={"review_id": str})
        rows = key.merge(features, on="review_id")
        pool._check_names(project, stain, names)
        probability, threshold = pool.model_probability(project, stain, rows[names].to_numpy(float), names)
        for tile_id, group in rows.assign(probability=probability).groupby("tile_id"):
            x, y = group.x.to_numpy(float), group.y.to_numpy(float)
            calls = pool._ai_calls(project, tiles[tile_id], x, y) if tile_id in tiles else pool.generic_calls(stain, group.probability.to_numpy(), threshold)
            frame = pd.DataFrame({"x": x.round(1), "y": y.round(1), "radius_px": [pool._radius(r, project.pixel_size_um) for r in group.to_dict("records")],
                                  "ai_value": calls, "ai_probability": group.probability.round(4).to_numpy(), "tma": group.tma.astype(str).to_numpy(),
                                  "complete": False})
            extra = group[[k for k in pool.KEEP if k in group and k not in names]].reset_index(drop=True)
            pool.add_imported(project, stain, tile_id, pd.concat([frame, extra, group[names].reset_index(drop=True)], axis=1))
        labelled = key.merge(labels[["review_id", "label"]], on="review_id")
        store.append(root, [{"id": store.object_id("class", r.tile_id, r.x, r.y), "source": source, "stain": stain, "task": "class",
                             "tile_id": r.tile_id, "x": float(r.x), "y": float(r.y), "value": r.label} for r in labelled.itertuples()])
        total += len(labelled)
        print(f"{folder.name}: {len(labelled)} labels", flush=True)
    return total


def cnn_adjudication(project: Project) -> int:
    root = project.output("reviews")
    folder = project.root / "data" / "analysis" / "plaque_cnn_disagreements"
    source = "import: CNN outlier adjudication"
    if _done(root, source) or not (folder / "consensus.csv").exists():
        return 0
    consensus = pd.read_csv(folder / "consensus.csv", dtype={"blind_id": str})
    key = pd.read_csv(folder / "key.csv", dtype={"blind_id": str})
    objects = pd.read_csv(folder / "objects.csv")
    rows = consensus[consensus.consensus.isin(["P", "N"])].merge(key[["blind_id", "object_id"]], on="blind_id").merge(objects[["object_id", "tile_id", "x_px", "y_px"]], on="object_id")
    store.append(root, [{"id": store.object_id("class", r.tile_id, r.x_px, r.y_px), "source": source, "stain": "6E10", "task": "class",
                         "tile_id": r.tile_id, "x": float(r.x_px), "y": float(r.y_px), "value": "plaque" if r.consensus == "P" else NEGATIVE["6E10"]}
                        for r in rows.itertuples()])
    print(f"CNN outlier adjudication: {len(rows)} labels", flush=True)
    return len(rows)


def white_matter_review(project: Project) -> int:
    root = project.output("reviews")
    folder = root / "white_matter_field_review"
    source = "import: white-matter field review"
    if _done(root, source) or not (folder / "review_labels.csv").exists():
        return 0
    rows = pd.read_csv(folder / "selection_key.csv").merge(pd.read_csv(folder / "review_labels.csv"), on="review_id")
    rows = rows[rows.label.isin(WHITE_MATTER_TAGS)]
    store.append(root, [{"id": f"tag:{r.tile_id}:{WHITE_MATTER_TAGS[r.label]}", "source": source, "stain": "NeuN", "task": "tag", "tile_id": r.tile_id,
                         "value": WHITE_MATTER_TAGS[r.label]} for r in rows.itertuples()])
    print(f"white-matter field review: {len(rows)} field tags", flush=True)
    return len(rows)


def import_all(project: Project) -> None:
    counts = [training_sets(project), cnn_adjudication(project), white_matter_review(project)]
    print(f"imported {int(np.sum(counts))} labels", flush=True)
