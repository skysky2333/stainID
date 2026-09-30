"""TMA map: which donor, region and diagnostic group sits at each core position."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from stainid.tables import read_csv, write_records

REQUIRED_COLUMNS = ("tma", "core_label", "donor_id", "region", "disease_group")
OPTIONAL_COLUMNS = ("sample_region_id", "cerad", "braak", "tissue_control", "map_source", "map_pathology_raw")
COMPUTED_COLUMNS = ("technical_replicate", "sample_region_id", "core_role", "map_qc")


def read_layout(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    """One row per core position. Positions without a donor are orientation / control cores.

    `sample_region_id` defaults to `<donor_id>_<region>`; replicate numbers count repeated cores of the same
    donor-region on a TMA in file order.
    """
    rows = read_csv(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in (rows[0] if rows else {})]
    if missing:
        raise ValueError(f"TMA map is missing columns: {', '.join(missing)}")
    keys = [(row["tma"], row["core_label"]) for row in rows]
    duplicates = sorted(key for key, count in Counter(keys).items() if count > 1)
    if duplicates:
        raise ValueError(f"Duplicate TMA map positions: {duplicates}")
    replicates: Counter[tuple[str, str, str]] = Counter()
    layout = {}
    for row in rows:
        donor, region = row["donor_id"], row["region"]
        if donor:
            replicates[(row["tma"], donor, region)] += 1
            computed = {"technical_replicate": str(replicates[(row["tma"], donor, region)]),
                        "sample_region_id": row.get("sample_region_id") or f"{donor}_{region}", "core_role": "biological",
                        "map_qc": "missing_pathology" if any(c in row and not row[c] for c in ("cerad", "braak")) else "ok"}
        else:
            computed = {"technical_replicate": "", "sample_region_id": "", "core_role": "orientation_control", "map_qc": "orientation_control"}
        layout[(row["tma"], row["core_label"])] = {**row, **computed}
    return layout


def attach_layout(manifest_path: Path, layout_path: Path) -> None:
    layout = read_layout(layout_path)
    rows = read_csv(manifest_path)
    unmapped = sorted({(row["tma"], row["core_label"]) for row in rows} - set(layout))
    if unmapped:
        raise ValueError(f"No TMA map entry for {len(unmapped)} positions, e.g. {unmapped[:5]}")
    columns = [c for c in REQUIRED_COLUMNS[2:] + OPTIONAL_COLUMNS + COMPUTED_COLUMNS if c in next(iter(layout.values()))]
    for row in rows:
        annotation = layout[(row["tma"], row["core_label"])]
        row.update({column: annotation[column] for column in columns})
    write_records(manifest_path, rows)


__all__ = ["attach_layout", "read_layout"]
