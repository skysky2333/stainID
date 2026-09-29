from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

LAYOUT_FIELDS = [
    "donor_id",
    "region",
    "tissue_control",
    "disease_group",
    "cerad",
    "braak",
    "technical_replicate",
    "sample_region_id",
    "core_role",
    "map_source",
    "map_pathology_raw",
    "map_qc",
]


def expected_positions() -> set[tuple[str, str]]:
    return {
        (str(tma), f"{column}-{row}")
        for tma in range(1, 8)
        for row in range(1, 6)
        for column in "ABCDEF"
    }


def read_layout(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    keys = [(row["tma"], row["core_label"]) for row in rows]
    if len(keys) != len(set(keys)):
        duplicates = sorted(key for key, count in Counter(keys).items() if count > 1)
        raise ValueError(f"Duplicate TMA layout positions: {duplicates}")
    expected = expected_positions()
    observed = set(keys)
    if observed != expected:
        raise ValueError(
            f"TMA layout mismatch; missing={sorted(expected - observed)}, "
            f"unexpected={sorted(observed - expected)}"
        )

    replicate_counts: Counter[tuple[str, str, str]] = Counter()
    layout = {}
    for row in rows:
        donor = row["donor_id"]
        region = row["region"]
        if donor:
            replicate_key = (row["tma"], donor, region)
            replicate_counts[replicate_key] += 1
            replicate = str(replicate_counts[replicate_key])
            sample_region_id = f"BRC_{donor}_{'FR' if region == 'frontal' else 'OCC'}"
            core_role = "biological"
        else:
            replicate = ""
            sample_region_id = ""
            core_role = "orientation_control"

        if core_role == "orientation_control":
            map_qc = "orientation_control"
        elif not row["cerad"] or not row["braak"]:
            map_qc = "missing_pathology"
        elif row["disease_group"] == "CT" and (
            row["cerad"] != "0" or row["braak"] not in {"0", "1"}
        ):
            map_qc = "group_pathology_conflict"
        else:
            map_qc = "ok"

        row.update(
            technical_replicate=replicate,
            sample_region_id=sample_region_id,
            core_role=core_role,
            map_qc=map_qc,
        )
        layout[(row["tma"], row["core_label"])] = row
    return layout


def attach_layout(manifest_path: Path, layout_path: Path) -> None:
    layout = read_layout(layout_path)
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = list(reader.fieldnames or [])
    if len(rows) != 630:
        raise ValueError(f"Expected 630 stain-core rows, found {len(rows)}")

    image_counts: Counter[tuple[str, str]] = Counter()
    for row in rows:
        key = (row["tma"], row["core_label"])
        if key not in layout:
            raise ValueError(f"No TMA layout entry for LIP-{key[0]} {key[1]}")
        image_counts[key] += 1
        annotation = layout[key]
        row.update({field: annotation[field] for field in LAYOUT_FIELDS})
    invalid_counts = sorted(key for key, count in image_counts.items() if count != 3)
    if invalid_counts:
        raise ValueError(f"Positions without exactly three stains: {invalid_counts}")

    complete_fields = fields + [field for field in LAYOUT_FIELDS if field not in fields]
    temporary = manifest_path.with_suffix(".tmp.csv")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=complete_fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(manifest_path)


__all__ = [
    "attach_layout",
    "expected_positions",
    "read_layout",
]
