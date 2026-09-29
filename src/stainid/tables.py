from __future__ import annotations

import csv
from pathlib import Path


def write_csv(
    path: Path,
    rows: list[dict[str, object]],
    fieldnames: list[str] | None = None,
) -> None:
    if fieldnames is None:
        if not rows:
            raise ValueError(f"Cannot infer columns for empty CSV: {path}")
        fieldnames = list(rows[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.csv")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def format_csv_value(value: object) -> object:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:.8f}"
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_records(path: Path, rows: list[dict[str, object]], empty_columns: list[str] | None = None) -> None:
    """Write rows with formatted values and the union of their columns (in first-seen order)."""
    columns = list(dict.fromkeys(k for row in rows for k in row)) if rows else (empty_columns or ["id"])
    write_csv(Path(path), [{k: format_csv_value(v) for k, v in row.items()} for row in rows], columns)
