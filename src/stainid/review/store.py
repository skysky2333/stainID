"""Review (annotation) sets: blinded object crops labelled by eye, stored as plain CSV next to the data.

A review set is a folder with `meta.json` (instructions, label options, field of view), `key.csv` (one row per
item: review_id, tile_id, x, y in field-crop pixels, plus hidden columns such as group and model probability) and
`labels.csv` (review_id, label, confidence, notes, reviewer, timestamp). Legacy sets from the morphology study
(`selection_key.csv` + `review_labels.csv`) are listed read-only.
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

HIDDEN_COLUMNS = ("disease_group", "donor_id", "sample_region_id", "probability", "model_class", "region")
LEGACY_LABEL_COLUMNS = ("label", "visual_label", "expert_label", "morphotype_label", "identity_label", "verdict", "acceptable")


def list_sets(root: Path) -> list[dict]:
    out = []
    for key in sorted(root.rglob("key.csv")) + sorted(root.rglob("selection_key.csv")):
        folder = key.parent
        legacy = key.name == "selection_key.csv"
        labels_file = folder / ("review_labels.csv" if legacy else "labels.csv")
        items = sum(1 for _ in open(key)) - 1
        labelled = _label_counts(labels_file, legacy)
        meta = json.loads((folder / "meta.json").read_text()) if (folder / "meta.json").exists() else {}
        out.append({"name": str(folder.relative_to(root)), "legacy": legacy, "items": items, "labelled": sum(labelled.values()),
                    "label_counts": labelled, "title": meta.get("title", folder.name), "stain": meta.get("stain", ""),
                    "created": meta.get("created"), "purpose": meta.get("purpose", "legacy" if legacy else "check")})
    return out


def _label_counts(path: Path, legacy: bool) -> dict[str, int]:
    if not path.exists():
        return {}
    frame = pd.read_csv(path, dtype=str)
    column = "label" if not legacy else next((c for c in LEGACY_LABEL_COLUMNS if c in frame), None)
    if column is None:
        return {}
    return {k: int(v) for k, v in frame[column].dropna().value_counts().items()}


def create_set(root: Path, name: str, title: str, stain: str, items: pd.DataFrame, label_options: list[str], instructions: str,
               fov_um: float, seed: int = 0) -> Path:
    folder = root / name
    if folder.exists():
        raise FileExistsError(folder)
    folder.mkdir(parents=True)
    items = items.sample(frac=1, random_state=seed).reset_index(drop=True)
    items.insert(0, "review_id", [f"R{i + 1:04d}" for i in range(len(items))])
    items.to_csv(folder / "key.csv", index=False)
    (folder / "meta.json").write_text(json.dumps({"title": title, "stain": stain, "label_options": label_options, "instructions": instructions,
                                                  "fov_um": fov_um, "created": time.time(), "blinded_columns": list(HIDDEN_COLUMNS)}, indent=1))
    return folder


def load_set(root: Path, name: str, include_hidden: bool = False) -> dict:
    folder = root / name
    meta = json.loads((folder / "meta.json").read_text())
    key = pd.read_csv(folder / "key.csv", dtype={"review_id": str})
    labels = read_labels(folder)
    items = key if include_hidden else key.drop(columns=[c for c in HIDDEN_COLUMNS if c in key])
    items = items.merge(labels[["review_id", "label", "confidence", "notes"]], on="review_id", how="left") if not labels.empty else items
    return {"meta": meta, "items": items.replace({np.nan: None}).to_dict(orient="records")}


def read_labels(folder: Path) -> pd.DataFrame:
    path = folder / "labels.csv"
    return pd.read_csv(path, dtype=str).drop_duplicates("review_id", keep="last") if path.exists() else pd.DataFrame()


def save_label(root: Path, name: str, review_id: str, label: str, confidence: str = "", notes: str = "", reviewer: str = "") -> None:
    path = root / name / "labels.csv"
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if new:
            writer.writerow(["review_id", "label", "confidence", "notes", "reviewer", "timestamp"])
        writer.writerow([review_id, label, confidence, notes, reviewer, time.strftime("%Y-%m-%dT%H:%M:%S")])


def unblinded_summary(root: Path, name: str) -> dict[str, list[dict]]:
    """Your labels against the hidden columns, once labelling is complete: by diagnostic group and by what the model decided."""
    folder = root / name
    key = pd.read_csv(folder / "key.csv", dtype=str)
    labels = read_labels(folder)
    merged = key.merge(labels[["review_id", "label"]], on="review_id") if not labels.empty else pd.DataFrame()
    return {column: [{"label": label, **{g: int(v) for g, v in row.items()}} for label, row in pd.crosstab(merged.label, merged[column]).iterrows()]
            for column in ("model_class", "disease_group") if column in merged}
