from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from stainid.slides.export import core_bounds, png_matches, validate_png


def audit_export_row(
    manifest_path: Path,
    row: dict[str, str],
    verify: bool,
) -> None:
    output = Path(row["output_path"])
    if not output.is_absolute():
        output = manifest_path.parent / output
    if not output.is_file():
        raise FileNotFoundError(output)
    _, _, width, height = core_bounds(row)
    expected = (width, height)
    if row["export_status"] != "complete" or row["export_downsample"] != "1":
        raise ValueError(f"Incomplete native export: {row['slide']} {row['core_label']}")
    if (int(row["output_width_px"]), int(row["output_height_px"])) != expected:
        raise ValueError(f"Manifest dimensions disagree: {row['slide']} {row['core_label']}")
    if verify:
        validate_png(output, expected)
    elif not png_matches(output, expected):
        raise ValueError(f"PNG header disagrees: {output}")


def audit_exports(manifest_path: Path, verify: bool = False, workers: int = 4) -> int:
    if workers < 1:
        raise ValueError("workers must be at least 1")
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Manifest contains no rows: {manifest_path}")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        list(executor.map(lambda row: audit_export_row(manifest_path, row, verify), rows))
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("data/core_manifest.csv"))
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    count = audit_exports(args.manifest, args.verify, args.workers)
    print(f"Validated {count} native-resolution PNG exports")


if __name__ == "__main__":
    main()
