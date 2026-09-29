from __future__ import annotations

import csv
from pathlib import Path

STAINS = ("6E10", "AT8", "NeuN")


def read_core_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "tma",
        "core_label",
        "stain",
        "core_role",
        "output_path",
        "output_width_px",
        "output_height_px",
        "slide_path",
        "center_x_px",
        "center_y_px",
        "diameter_x_px",
        "diameter_y_px",
        "source_pixel_width_um",
        "source_pixel_height_um",
        "donor_id",
        "region",
        "disease_group",
        "cerad",
        "braak",
        "technical_replicate",
        "sample_region_id",
        "tissue_status",
        "tissue_fraction",
        "review_required",
    }
    missing = sorted(required - set(rows[0] if rows else ()))
    if missing:
        raise ValueError(f"Core manifest is missing columns: {', '.join(missing)}")
    return rows


def analysis_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    biological = [row for row in rows if row["core_role"] == "biological"]
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in biological:
        grouped.setdefault((row["tma"], row["core_label"]), []).append(row)

    records = []
    for (tma, core_label), triplet in sorted(
        grouped.items(), key=lambda item: (int(item[0][0]), item[0][1])
    ):
        by_stain = {row["stain"]: row for row in triplet}
        if set(by_stain) != set(STAINS) or len(triplet) != len(STAINS):
            raise ValueError(f"Invalid stain triplet for LIP-{tma} {core_label}")
        identity_fields = (
            "donor_id",
            "region",
            "disease_group",
            "cerad",
            "braak",
            "technical_replicate",
            "sample_region_id",
        )
        for field in identity_fields:
            if len({row[field] for row in triplet}) != 1:
                raise ValueError(f"Stain metadata disagree for LIP-{tma} {core_label}: {field}")

        core_id = f"LIP-{tma}_{core_label}"
        for stain in STAINS:
            row = by_stain[stain]
            records.append(
                {
                    "core_id": core_id,
                    "tma": tma,
                    "core_label": core_label,
                    "donor_id": row["donor_id"],
                    "sample_region_id": row["sample_region_id"],
                    "region": row["region"],
                    "disease_group": row["disease_group"],
                    "cerad": row["cerad"],
                    "braak": row["braak"],
                    "technical_replicate": row["technical_replicate"],
                    "stain": stain,
                    "image_path": str(Path("data") / row["output_path"]),
                    "slide_path": row["slide_path"],
                    "center_x_px": row["center_x_px"],
                    "center_y_px": row["center_y_px"],
                    "diameter_x_px": row["diameter_x_px"],
                    "diameter_y_px": row["diameter_y_px"],
                    "native_width_px": row["output_width_px"],
                    "native_height_px": row["output_height_px"],
                    "pixel_width_um": row["source_pixel_width_um"],
                    "pixel_height_um": row["source_pixel_height_um"],
                    "tissue_status": row["tissue_status"],
                    "tissue_fraction": row["tissue_fraction"],
                    "review_required": row["review_required"],
                }
            )
    return records


def write_analysis_manifest(core_manifest: Path, output_path: Path) -> Path:
    records = analysis_rows(read_core_manifest(core_manifest))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".tmp.csv")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    temporary.replace(output_path)
    return output_path


__all__ = [
    "analysis_rows",
    "read_core_manifest",
    "write_analysis_manifest",
]
