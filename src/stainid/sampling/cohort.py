from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np

from stainid.qc.core_qc import tissue_mask
from stainid.qc.focus import low_focus_mask, measure_focus
from stainid.sampling.fields import integral_sum
from stainid.slides.core_sheets import read_core_preview
from stainid.tables import write_csv


def spatially_spread_indices(
    candidates: list[dict[str, float | int]], count: int
) -> list[int]:
    if count <= 0 or not candidates:
        return []
    points = np.asarray(
        [
            [float(row["center_x_fraction"]), float(row["center_y_fraction"])]
            for row in candidates
        ],
        dtype=np.float64,
    )
    chosen = [int(np.argmin(np.square(points - 0.5).sum(axis=1)))]
    while len(chosen) < min(count, len(candidates)):
        eligible = np.ones(len(candidates), dtype=bool)
        for index, row in enumerate(candidates):
            if index in chosen:
                eligible[index] = False
                continue
            for selected_index in chosen:
                selected = candidates[selected_index]
                if all(
                    key in row and key in selected
                    for key in (
                        "x_px",
                        "y_px",
                        "width_px",
                        "height_px",
                    )
                ):
                    separated = (
                        int(row["x_px"]) + int(row["width_px"])
                        <= int(selected["x_px"])
                        or int(selected["x_px"]) + int(selected["width_px"])
                        <= int(row["x_px"])
                        or int(row["y_px"]) + int(row["height_px"])
                        <= int(selected["y_px"])
                        or int(selected["y_px"]) + int(selected["height_px"])
                        <= int(row["y_px"])
                    )
                    if not separated:
                        eligible[index] = False
                        break
        if not eligible.any():
            break
        distances = np.square(points[:, None, :] - points[chosen][None, :, :]).sum(
            axis=2
        )
        minimum = distances.min(axis=1)
        minimum[~eligible] = -1.0
        chosen.append(int(np.argmax(minimum)))
    return chosen


def select_systematic_fields(
    rgb: np.ndarray,
    native_width: int,
    native_height: int,
    field_size_px: int = 2048,
    fields_per_core: int = 8,
) -> tuple[list[dict[str, float | int | str]], dict[str, float | int | str]]:
    preview_height, preview_width = rgb.shape[:2]
    field_width = min(field_size_px, native_width)
    field_height = min(field_size_px, native_height)
    preview_field_width = max(1, round(field_width * preview_width / native_width))
    preview_field_height = max(1, round(field_height * preview_height / native_height))

    tissue, _, _ = tissue_mask(rgb)
    _, focus_tiles = measure_focus(rgb, tissue)
    blurred = low_focus_mask(tissue.shape, focus_tiles) & tissue
    usable = tissue & ~blurred
    tissue_integral = cv2.integral(tissue.astype(np.float64))
    usable_integral = cv2.integral(usable.astype(np.float64))
    blur_integral = cv2.integral(blurred.astype(np.float64))
    candidates: list[dict[str, float | int]] = []
    native_y_positions = range(0, native_height - field_height + 1, field_height)
    native_x_positions = range(0, native_width - field_width + 1, field_width)
    for native_y in native_y_positions:
        y = round(native_y * preview_height / native_height)
        local_preview_height = min(preview_field_height, preview_height - y)
        for native_x in native_x_positions:
            x = round(native_x * preview_width / native_width)
            local_preview_width = min(preview_field_width, preview_width - x)
            area = local_preview_width * local_preview_height
            tissue_area = integral_sum(
                tissue_integral, x, y, local_preview_width, local_preview_height
            )
            usable_area = integral_sum(
                usable_integral, x, y, local_preview_width, local_preview_height
            )
            blur_area = integral_sum(
                blur_integral, x, y, local_preview_width, local_preview_height
            )
            candidates.append(
                {
                    "x_px": native_x,
                    "y_px": native_y,
                    "width_px": field_width,
                    "height_px": field_height,
                    "preview_x_px": x,
                    "preview_y_px": y,
                    "preview_tissue_fraction": tissue_area / area,
                    "preview_usable_tissue_fraction": usable_area / area,
                    "preview_low_focus_tissue_fraction": (
                        blur_area / tissue_area if tissue_area else 0.0
                    ),
                    "center_x_fraction": (native_x + field_width / 2) / native_width,
                    "center_y_fraction": (native_y + field_height / 2) / native_height,
                }
            )

    tiers = (
        (
            "strict_tissue_focus",
            lambda row: row["preview_usable_tissue_fraction"] >= 0.75
            and row["preview_low_focus_tissue_fraction"] <= 0.05,
        ),
        (
            "relaxed_focus",
            lambda row: row["preview_usable_tissue_fraction"] >= 0.75
            and row["preview_low_focus_tissue_fraction"] <= 0.20,
        ),
        (
            "relaxed_tissue",
            lambda row: row["preview_usable_tissue_fraction"] >= 0.50
            and row["preview_low_focus_tissue_fraction"] <= 0.20,
        ),
        (
            "best_available_tissue",
            lambda row: row["preview_usable_tissue_fraction"] >= 0.20,
        ),
    )
    eligible: list[dict[str, float | int]] = []
    selection_status = "no_usable_tissue"
    for status, predicate in tiers:
        tier = [row for row in candidates if predicate(row)]
        if tier:
            eligible = tier
            selection_status = status
        if len(tier) >= fields_per_core:
            break
    if not eligible:
        return [], {
            "selection_status": selection_status,
            "sampling_frame_candidate_count": len(candidates),
            "eligible_candidate_count": 0,
        }

    eligible.sort(
        key=lambda row: (int(row["preview_y_px"]), int(row["preview_x_px"]))
    )
    selected = [
        eligible[index]
        for index in spatially_spread_indices(eligible, fields_per_core)
    ]
    fields = []
    for order, row in enumerate(selected, start=1):
        fields.append(
            {
                "selection_order": order,
                "nested_sample": "primary_four" if order <= 4 else "extended_eight",
                **row,
            }
        )
    return fields, {
        "selection_status": selection_status,
        "sampling_frame_candidate_count": len(candidates),
        "eligible_candidate_count": len(eligible),
    }


def create_systematic_tile_manifest(
    core_manifest: Path,
    output_path: Path,
    summary_output: Path,
    fields_per_core: int = 8,
    field_size_px: int = 2048,
    target_long_side: int = 4500,
    limit: int | None = None,
) -> tuple[Path, Path]:
    with core_manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda row: (row["slide_path"], row["core_id"], row["stain"]))
    if limit is not None:
        rows = rows[:limit]
    records: list[dict[str, object]] = []
    summary_records: list[dict[str, object]] = []
    for image_index, row in enumerate(rows, start=1):
        preview = read_core_preview(row, target_long_side)
        fields, sampling = select_systematic_fields(
            preview,
            int(row["native_width_px"]),
            int(row["native_height_px"]),
            field_size_px,
            fields_per_core,
        )
        summary_records.append(
            {
                "core_id": row["core_id"],
                "tma": row["tma"],
                "core_label": row["core_label"],
                "donor_id": row["donor_id"],
                "sample_region_id": row["sample_region_id"],
                "region": row["region"],
                "disease_group": row["disease_group"],
                "technical_replicate": row["technical_replicate"],
                "stain": row["stain"],
                "source_tissue_status": row["tissue_status"],
                "source_tissue_fraction": row["tissue_fraction"],
                "selected_field_count": len(fields),
                **sampling,
            }
        )
        for field in fields:
            order = int(field["selection_order"])
            records.append(
                {
                    "tile_id": f"{row['core_id']}_{row['stain']}_S{order:02d}",
                    "core_id": row["core_id"],
                    "tma": row["tma"],
                    "core_label": row["core_label"],
                    "donor_id": row["donor_id"],
                    "sample_region_id": row["sample_region_id"],
                    "region": row["region"],
                    "disease_group": row["disease_group"],
                    "technical_replicate": row["technical_replicate"],
                    "stain": row["stain"],
                    "image_path": row["image_path"],
                    "pixel_width_um": row["pixel_width_um"],
                    "pixel_height_um": row["pixel_height_um"],
                    "selection_role": "spatially_balanced_native_field",
                    **field,
                    **sampling,
                }
            )
        if image_index % 25 == 0 or image_index == len(rows):
            print(f"Selected fields for {image_index}/{len(rows)} stain-core images", flush=True)
    if not records:
        raise ValueError("No systematic cohort fields were selected")
    write_csv(output_path, records, list(records[0]))
    write_csv(summary_output, summary_records, list(summary_records[0]))
    return output_path, summary_output


__all__ = [
    "create_systematic_tile_manifest",
    "select_systematic_fields",
    "spatially_spread_indices",
]
