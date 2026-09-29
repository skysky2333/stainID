from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from stainid.imaging.color import rgb_to_hed
from stainid.qc.core_qc import tissue_mask
from stainid.qc.focus import low_focus_mask, measure_focus
from stainid.slides.core_sheets import read_core_preview

TARGET_QUANTILES = (0.15, 0.50, 0.85)


def window_positions(length: int, size: int, stride: int) -> list[int]:
    if size >= length:
        return [0]
    positions = list(range(0, length - size + 1, stride))
    if positions[-1] != length - size:
        positions.append(length - size)
    return positions


def integral_sum(integral: np.ndarray, x: int, y: int, width: int, height: int) -> float:
    return float(
        integral[y + height, x + width]
        - integral[y, x + width]
        - integral[y + height, x]
        + integral[y, x]
    )


def target_quantile(image_id: str, seed: int = 20260923) -> float:
    identifier, separator, stain = image_id.rpartition("_")
    if separator and identifier[1:].isdigit() and stain in {"6E10", "AT8", "NeuN"}:
        stain_offset = {"6E10": 0, "AT8": 1, "NeuN": 2}[stain]
        return TARGET_QUANTILES[(int(identifier[1:]) - 1 + stain_offset) % 3]
    digest = hashlib.sha256(f"{seed}:{image_id}".encode()).digest()
    return TARGET_QUANTILES[digest[0] % len(TARGET_QUANTILES)]


def select_field(
    rgb: np.ndarray,
    native_width: int,
    native_height: int,
    target: float,
    field_size_px: int = 2048,
) -> dict[str, float | int | str]:
    preview_height, preview_width = rgb.shape[:2]
    field_width = min(field_size_px, native_width)
    field_height = min(field_size_px, native_height)
    preview_field_width = max(1, round(field_width * preview_width / native_width))
    preview_field_height = max(1, round(field_height * preview_height / native_height))
    stride_x = max(1, preview_field_width // 2)
    stride_y = max(1, preview_field_height // 2)

    tissue, _, _ = tissue_mask(rgb)
    _, focus_tiles = measure_focus(rgb, tissue)
    blurred = low_focus_mask(tissue.shape, focus_tiles)
    usable_tissue = tissue & ~blurred
    dab = rgb_to_hed(rgb)[..., 2]
    tissue_integral = cv2.integral(usable_tissue.astype(np.float64))
    blur_integral = cv2.integral((blurred & tissue).astype(np.float64))
    dab_integral = cv2.integral(dab * usable_tissue)
    candidates = []
    for y in window_positions(preview_height, preview_field_height, stride_y):
        for x in window_positions(preview_width, preview_field_width, stride_x):
            area = preview_field_width * preview_field_height
            tissue_area = integral_sum(
                tissue_integral, x, y, preview_field_width, preview_field_height
            )
            fraction = tissue_area / area
            mean_dab = (
                integral_sum(dab_integral, x, y, preview_field_width, preview_field_height)
                / tissue_area
                if tissue_area
                else 0.0
            )
            low_focus_area = integral_sum(
                blur_integral, x, y, preview_field_width, preview_field_height
            )
            candidates.append((x, y, fraction, mean_dab, low_focus_area / area))

    high_tissue = [candidate for candidate in candidates if candidate[2] >= 0.75]
    if high_tissue:
        eligible = high_tissue
        selection_status = "tissue_rich"
    else:
        maximum = max(candidate[2] for candidate in candidates)
        eligible = [candidate for candidate in candidates if candidate[2] >= 0.9 * maximum]
        selection_status = "best_available_tissue"

    eligible.sort(key=lambda candidate: (candidate[3], candidate[0], candidate[1]))
    selected_index = round(target * (len(eligible) - 1))
    x, y, fraction, mean_dab, low_focus_fraction = eligible[selected_index]
    native_x = min(
        native_width - field_width, round(x * native_width / preview_width)
    )
    native_y = min(
        native_height - field_height, round(y * native_height / preview_height)
    )
    percentile = selected_index / (len(eligible) - 1) if len(eligible) > 1 else 0.5
    return {
        "x_px": native_x,
        "y_px": native_y,
        "width_px": field_width,
        "height_px": field_height,
        "preview_tissue_fraction": fraction,
        "preview_mean_dab_od": mean_dab,
        "preview_low_focus_fraction": low_focus_fraction,
        "local_dab_percentile": percentile,
        "candidate_count": len(eligible),
        "selection_status": selection_status,
    }


def create_field_manifests(
    annotation_manifest: Path,
    output_csv: Path,
    output_json: Path,
    field_size_px: int = 2048,
    seed: int = 20260923,
) -> tuple[Path, Path]:
    with annotation_manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    records = []
    ordered_rows = sorted(
        rows, key=lambda item: (item.get("slide_path", ""), item["image_id"])
    )
    for row in ordered_rows:
        rgb = read_core_preview(row)
        target = target_quantile(row["image_id"], seed)
        field = select_field(
            rgb,
            int(row["native_width_px"]),
            int(row["native_height_px"]),
            target,
            field_size_px,
        )
        records.append(
            {
                "annotation_id": row["annotation_id"],
                "image_id": row["image_id"],
                "field_id": f"{row['image_id']}_F01",
                "stain": row["stain"],
                "target_dab_quantile": f"{target:.2f}",
                **{
                    key: f"{value:.6f}" if isinstance(value, float) else value
                    for key, value in field.items()
                },
            }
        )
        print(f"{row['image_id']}: selected annotation field", flush=True)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    temporary_csv = output_csv.with_suffix(".tmp.csv")
    with temporary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    temporary_csv.replace(output_csv)

    temporary_json = output_json.with_suffix(".tmp.json")
    with temporary_json.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2)
        handle.write("\n")
    temporary_json.replace(output_json)
    return output_csv, output_json


__all__ = [
    "create_field_manifests",
    "select_field",
    "target_quantile",
]
