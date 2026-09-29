from __future__ import annotations

import csv
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def used_annotation_ids(manifests: list[Path]) -> set[str]:
    used = set()
    for path in manifests:
        for row in read_csv(path):
            used.add(row["image_id"].rsplit("_", 1)[0])
    return used


def select_validation_triplets(
    selection_rows: list[dict[str, str]], used_ids: set[str]
) -> list[dict[str, str]]:
    by_tma: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in selection_rows:
        if row["annotation_id"] not in used_ids:
            by_tma[row["tma"]].append(row)
    tmas = sorted(by_tma, key=int)
    if not tmas or any(not by_tma[tma] for tma in tmas):
        raise ValueError("Each TMA requires an unused validation candidate")

    best = None
    for combination in itertools.product(*(by_tma[tma] for tma in tmas)):
        groups = Counter(row["disease_group"] for row in combination)
        regions = Counter(row["region"] for row in combination)
        balance = sum(
            (groups[group] - len(tmas) / 3.0) ** 2 for group in ("CT", "ASYMP", "AD")
        ) + sum(
            (regions[region] - len(tmas) / 2.0) ** 2
            for region in ("frontal", "occipital")
        )
        signal_distance = sum(float(row["selection_distance"]) for row in combination)
        tissue = -sum(float(row["minimum_tissue_fraction"]) for row in combination)
        identifiers = tuple(row["annotation_id"] for row in combination)
        score = (round(balance, 12), round(signal_distance, 12), round(tissue, 12), identifiers)
        if best is None or score < best[0]:
            best = (score, combination)
    return list(best[1])


def validation_rows(
    selected: list[dict[str, str]],
    annotation_rows: list[dict[str, str]],
    field_rows: list[dict[str, str]],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    selected_by_id = {row["annotation_id"]: row for row in selected}
    annotations = {
        row["image_id"]: row
        for row in annotation_rows
        if row["annotation_id"] in selected_by_id
    }
    fields = {
        row["image_id"]: row for row in field_rows if row["annotation_id"] in selected_by_id
    }
    expected = {f"{annotation_id}_{stain}" for annotation_id in selected_by_id for stain in ("6E10", "AT8", "NeuN")}
    if set(annotations) != expected or set(fields) != expected:
        raise ValueError("Selected validation triplets are incomplete")

    blinded = []
    for image_id in sorted(expected):
        annotation = annotations[image_id]
        field = fields[image_id]
        selected_row = selected_by_id[annotation["annotation_id"]]
        blinded.append(
            {
                "annotation_id": annotation["annotation_id"],
                "image_id": image_id,
                "tma": selected_row["tma"],
                "stain": annotation["stain"],
                "field_id": field["field_id"],
                "field_path": f"data/annotations/fields/{image_id}.png",
                "target_dab_quantile": field["target_dab_quantile"],
                "validation_role": "locked_holdout",
                "review_status": "not_started",
            }
        )
    key_fields = (
        "annotation_id",
        "tma",
        "core_label",
        "donor_id",
        "sample_region_id",
        "disease_group",
        "region",
        "cerad",
        "braak",
        "technical_replicate",
    )
    key = [{field: row[field] for field in key_fields} for row in selected]
    return blinded, sorted(key, key=lambda row: int(row["tma"]))


def write_csv(path: Path, rows: list[dict[str, str]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_json(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)
        handle.write("\n")
    return path


def subset_by_image_id(
    rows: list[dict[str, object]], image_ids: set[str]
) -> list[dict[str, object]]:
    selected = [row for row in rows if str(row["image_id"]) in image_ids]
    if {str(row["image_id"]) for row in selected} != image_ids:
        raise ValueError("Adjudication image set is incomplete")
    return sorted(selected, key=lambda row: str(row["image_id"]))


__all__ = [
    "read_csv",
    "select_validation_triplets",
    "used_annotation_ids",
    "validation_rows",
    "write_csv",
    "write_json",
    "subset_by_image_id",
]
