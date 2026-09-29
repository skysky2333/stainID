from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from stainid.review.selection import evenly_spaced_candidates


def select_at8_review(
    rows: list[dict[str, object]],
    per_group: int,
    seed: int = 20260924,
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row["source_image_id"]),
                str(row["source_kind"]),
                str(row["candidate_class"]),
            )
        ].append(row)
    selected = [
        row
        for key in sorted(grouped)
        for row in evenly_spaced_candidates(
            grouped[key], per_group, value_key="area_um2"
        )
    ]
    random.Random(seed).shuffle(selected)
    return [
        {**row, "review_id": f"TR{index:03d}"}
        for index, row in enumerate(selected, start=1)
    ]


def crop_with_padding(
    image: np.ndarray,
    center_x: float,
    center_y: float,
    size: int,
    fill: int = 255,
) -> np.ndarray:
    half = size // 2
    x0 = round(center_x) - half
    y0 = round(center_y) - half
    x1 = x0 + size
    y1 = y0 + size
    shape = (size, size) if image.ndim == 2 else (size, size, image.shape[2])
    output = np.full(shape, fill, dtype=image.dtype)
    source_x0 = max(0, x0)
    source_y0 = max(0, y0)
    source_x1 = min(image.shape[1], x1)
    source_y1 = min(image.shape[0], y1)
    if source_x1 > source_x0 and source_y1 > source_y0:
        output[
            source_y0 - y0 : source_y1 - y0,
            source_x0 - x0 : source_x1 - x0,
        ] = image[source_y0:source_y1, source_x0:source_x1]
    return output


def target_overlay(
    bgr: np.ndarray,
    masks: dict[str, np.ndarray],
    row: dict[str, object],
) -> np.ndarray:
    source_kind = str(row["source_kind"])
    label = int(row["label"])
    if source_kind == "compact_profile":
        target = masks["compact_labels"] == label
        color = (35, 35, 220)
    elif source_kind == "thread_cluster":
        target = masks["cluster_labels"] == label
        color = (120, 170, 20)
    else:
        raise ValueError(f"Unsupported AT8 candidate source: {source_kind}")
    overlay = bgr.copy()
    if source_kind == "thread_cluster":
        local_thread = masks["thread"].astype(bool) & target
        overlay[local_thread] = (
            0.3 * overlay[local_thread] + 0.7 * np.asarray([205, 175, 40])
        ).astype(np.uint8)
    contours, _ = cv2.findContours(
        target.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(overlay, contours, -1, color, 4)
    return overlay


def render_review_panel(
    bgr: np.ndarray,
    masks: dict[str, np.ndarray],
    row: dict[str, object],
    context_um: float,
    panel_px: int,
) -> np.ndarray:
    pixel_size_um = np.sqrt(
        float(row["pixel_width_um"]) * float(row["pixel_height_um"])
    )
    crop_size = max(64, round(context_um / pixel_size_um))
    center_x = float(row["centroid_x_px"])
    center_y = float(row["centroid_y_px"])
    raw = crop_with_padding(bgr, center_x, center_y, crop_size)
    overlay = crop_with_padding(
        target_overlay(bgr, masks, row), center_x, center_y, crop_size
    )
    raw = cv2.resize(raw, (panel_px, panel_px), interpolation=cv2.INTER_AREA)
    overlay = cv2.resize(overlay, (panel_px, panel_px), interpolation=cv2.INTER_AREA)
    center = panel_px // 2
    cv2.drawMarker(
        overlay,
        (center, center),
        (180, 50, 190),
        cv2.MARKER_CROSS,
        18,
        2,
        cv2.LINE_AA,
    )
    panel = np.concatenate((raw, overlay), axis=1)
    header = np.full((36, panel.shape[1], 3), 255, dtype=np.uint8)
    title = f"{row['review_id']}  {row['source_kind']} / {row['candidate_class']}"
    cv2.putText(
        header,
        title,
        (8, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )
    return np.concatenate((header, panel), axis=0)


def write_review_images(
    rows: list[dict[str, object]],
    output_dir: Path,
    context_um: float,
    panel_px: int,
    sheet_columns: int,
    sheet_rows: int,
) -> None:
    if not rows:
        raise ValueError("AT8 candidate review contains no objects")
    patches_dir = output_dir / "patches"
    sheets_dir = output_dir / "sheets"
    patches_dir.mkdir(parents=True, exist_ok=True)
    sheets_dir.mkdir(parents=True, exist_ok=True)
    panels: list[np.ndarray] = []
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["raw_path"])].append(row)
    by_id: dict[str, np.ndarray] = {}
    for raw_path, group in grouped.items():
        bgr = cv2.imread(raw_path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"Could not read AT8 field: {raw_path}")
        with np.load(str(group[0]["mask_path"])) as loaded:
            masks = {key: loaded[key] for key in loaded.files}
        for row in group:
            by_id[str(row["review_id"])] = render_review_panel(
                bgr, masks, row, context_um, panel_px
            )
    for row in rows:
        panel = by_id[str(row["review_id"])]
        output_path = patches_dir / f"{row['review_id']}.jpg"
        if not cv2.imwrite(str(output_path), panel, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise OSError(f"Could not write AT8 review patch: {output_path}")
        panels.append(panel)
    page_size = sheet_columns * sheet_rows
    blank = np.full_like(panels[0], 255)
    for start in range(0, len(panels), page_size):
        page = panels[start : start + page_size]
        page.extend([blank] * (page_size - len(page)))
        sheet = np.concatenate(
            [
                np.concatenate(page[index : index + sheet_columns], axis=1)
                for index in range(0, page_size, sheet_columns)
            ],
            axis=0,
        )
        output_path = sheets_dir / f"sheet_{start // page_size + 1:02d}.jpg"
        if not cv2.imwrite(str(output_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 94]):
            raise OSError(f"Could not write AT8 review sheet: {output_path}")


__all__ = [
    "crop_with_padding",
    "render_review_panel",
    "select_at8_review",
    "target_overlay",
    "write_review_images",
]
