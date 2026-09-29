from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

SUMMARY_FIELDS = [
    "slide",
    "tma",
    "stain",
    "positions",
    "biological_positions",
    "orientation_controls",
    "locally_centered",
    "lattice_centered",
    "median_center_offset_preview_px",
    "substantial",
    "partial",
    "sparse",
    "empty",
    "review_required",
    "focus_review_required",
    "median_tissue_fraction",
    "median_low_focus_tissue_fraction",
    "median_background_brightness",
    "median_hematoxylin_od_p90",
    "median_dab_od_p90",
    "iqr_dab_od_p90",
]

REVIEW_FIELDS = [
    "slide",
    "tma",
    "stain",
    "core_label",
    "donor_id",
    "region",
    "tissue_status",
    "tissue_fraction",
    "center_source",
    "center_offset_preview_px",
    "low_focus_tissue_fraction",
    "slide_edge_padding_fraction",
    "review_reasons",
]

MATERIAL_PADDING_FRACTION = 0.10


def median(values: list[float]) -> str:
    return f"{np.median(values):.6f}" if values else ""


def summarize_manifest(manifest_path: Path, output_path: Path) -> Path:
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = set(reader.fieldnames or [])
    required = {
        "slide",
        "tma",
        "stain",
        "core_role",
        "center_source",
        "center_offset_preview_px",
        "tissue_status",
        "tissue_fraction",
        "background_brightness",
        "hematoxylin_od_p90",
        "dab_od_p90",
        "review_required",
        "focus_review_required",
        "low_focus_tissue_fraction",
    }
    missing = sorted(required - fields)
    if missing:
        raise ValueError(f"Manifest is missing QC columns: {', '.join(missing)}")
    if any(not row["tissue_status"] for row in rows):
        raise ValueError("Core QC is incomplete")

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["slide"]].append(row)

    summary = []
    for slide in sorted(grouped):
        slide_rows = grouped[slide]
        statuses = Counter(row["tissue_status"] for row in slide_rows)
        biological = [row for row in slide_rows if row["core_role"] == "biological"]
        usable = [
            row
            for row in biological
            if row["tissue_status"] in {"partial", "substantial"}
        ]
        dab_p90 = [float(row["dab_od_p90"]) for row in usable]
        dab_quartiles = (
            np.quantile(dab_p90, [0.25, 0.75])
            if dab_p90
            else [np.nan, np.nan]
        )
        summary.append(
            {
                "slide": slide,
                "tma": slide_rows[0]["tma"],
                "stain": slide_rows[0]["stain"],
                "positions": len(slide_rows),
                "biological_positions": len(biological),
                "orientation_controls": len(slide_rows) - len(biological),
                "locally_centered": sum(
                    row["center_source"] == "component" for row in slide_rows
                ),
                "lattice_centered": sum(
                    row["center_source"] == "lattice" for row in slide_rows
                ),
                "median_center_offset_preview_px": median(
                    [
                        float(row["center_offset_preview_px"])
                        for row in slide_rows
                        if row["center_source"] == "component"
                    ]
                ),
                "substantial": statuses["substantial"],
                "partial": statuses["partial"],
                "sparse": statuses["sparse"],
                "empty": statuses["empty"],
                "review_required": sum(
                    row["review_required"] == "true" for row in slide_rows
                ),
                "focus_review_required": sum(
                    row["focus_review_required"] == "true" for row in slide_rows
                ),
                "median_tissue_fraction": median(
                    [float(row["tissue_fraction"]) for row in usable]
                ),
                "median_low_focus_tissue_fraction": median(
                    [float(row["low_focus_tissue_fraction"]) for row in usable]
                ),
                "median_background_brightness": median(
                    [float(row["background_brightness"]) for row in usable]
                ),
                "median_hematoxylin_od_p90": median(
                    [float(row["hematoxylin_od_p90"]) for row in usable]
                ),
                "median_dab_od_p90": median(dab_p90),
                "iqr_dab_od_p90": (
                    f"{dab_quartiles[1] - dab_quartiles[0]:.6f}"
                    if dab_p90
                    else ""
                ),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary)
    return output_path


def write_review_queue(manifest_path: Path, output_path: Path) -> Path:
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    review_rows = []
    for row in rows:
        reasons = []
        if row["center_source"] == "lattice":
            reasons.append("unresolved_local_center")
        if row["tissue_status"] in {"sparse", "empty"}:
            reasons.append(row["tissue_status"] + "_tissue")
        if row["focus_review_required"] == "true":
            reasons.append("low_focus")
        width = int(row["output_width_px"])
        height = int(row["output_height_px"])
        source_width = width - int(row.get("padding_left_px", "0") or 0) - int(
            row.get("padding_right_px", "0") or 0
        )
        source_height = height - int(row.get("padding_top_px", "0") or 0) - int(
            row.get("padding_bottom_px", "0") or 0
        )
        padding_fraction = 1.0 - source_width * source_height / (width * height)
        if padding_fraction >= MATERIAL_PADDING_FRACTION:
            reasons.append("slide_edge_padding")
        if reasons:
            review_rows.append(
                {
                    **{field: row.get(field, "") for field in REVIEW_FIELDS[:-2]},
                    "slide_edge_padding_fraction": f"{padding_fraction:.6f}",
                    "review_reasons": ";".join(reasons),
                }
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(review_rows)
    return output_path


__all__ = [
    "summarize_manifest",
    "write_review_queue",
]
