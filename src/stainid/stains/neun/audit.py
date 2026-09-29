from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from stainid.imaging.tissue import field_tissue_mask
from stainid.review.selection import evenly_spaced_candidates

COLORS = {
    "positive_profile": (35, 35, 220),
    "review": (25, 145, 245),
    "negative": (230, 140, 30),
}
REVIEW_OUTLINE_COLOR = (40, 190, 40)


def select_candidate_review(
    rows: list[dict[str, object]],
    per_group: int,
    seed: int = 20260925,
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if str(row["touches_edge"]).lower() == "true":
            continue
        grouped[
            (
                str(row["tma"]),
                str(row["source_kind"]),
                str(row["candidate_class"]),
            )
        ].append(row)
    selected = [
        {
            **row,
            "sampling_group": "|".join(key),
            "eligible_group_count": len(grouped[key]),
            "selected_group_count": min(per_group, len(grouped[key])),
            "sampling_weight": len(grouped[key]) / min(per_group, len(grouped[key])),
        }
        for key in sorted(grouped)
        for row in evenly_spaced_candidates(
            grouped[key], per_group, value_key="selection_area_um2"
        )
    ]
    random.Random(seed).shuffle(selected)
    return [
        {**row, "review_id": f"NR{index:03d}"}
        for index, row in enumerate(selected, start=1)
    ]


def crop_with_padding(
    image: np.ndarray,
    center_x: float,
    center_y: float,
    size: int,
) -> np.ndarray:
    half = size // 2
    x0 = round(center_x) - half
    y0 = round(center_y) - half
    x1 = x0 + size
    y1 = y0 + size
    shape = (size, size) if image.ndim == 2 else (size, size, image.shape[2])
    output = np.full(shape, 255, dtype=image.dtype)
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
    profile_labels: np.ndarray,
    cellpose_labels: np.ndarray,
    row: dict[str, object],
) -> np.ndarray:
    labels = (
        profile_labels
        if str(row["source_kind"]) == "dab_profile"
        else cellpose_labels
    )
    target = labels == int(row["source_label"])
    overlay = bgr.copy()
    contours, _ = cv2.findContours(
        target.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(
        overlay,
        contours,
        -1,
        REVIEW_OUTLINE_COLOR,
        4,
    )
    return overlay


def render_candidate_panel(
    bgr: np.ndarray,
    profile_labels: np.ndarray,
    cellpose_labels: np.ndarray,
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
        target_overlay(bgr, profile_labels, cellpose_labels, row),
        center_x,
        center_y,
        crop_size,
    )
    raw = cv2.resize(raw, (panel_px, panel_px), interpolation=cv2.INTER_AREA)
    overlay = cv2.resize(overlay, (panel_px, panel_px), interpolation=cv2.INTER_AREA)
    cv2.drawMarker(
        overlay,
        (panel_px // 2, panel_px // 2),
        (190, 50, 190),
        cv2.MARKER_CROSS,
        18,
        2,
        cv2.LINE_AA,
    )
    panel = np.concatenate((raw, overlay), axis=1)
    header = np.full((36, panel.shape[1], 3), 255, dtype=np.uint8)
    cv2.putText(
        header,
        str(row["review_id"]),
        (8, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )
    return np.concatenate((header, panel), axis=0)


def select_field_windows(
    bgr: np.ndarray,
    objects: list[dict[str, object]],
    exclusion_mask: np.ndarray,
    pixel_size_um: float,
    window_um: float,
    count: int,
) -> list[dict[str, object]]:
    size = max(128, round(window_um / pixel_size_um))
    tissue = field_tissue_mask(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    positive = [
        row
        for row in objects
        if row["source_kind"] == "dab_profile"
        and row["candidate_class"] == "positive_profile"
        and str(row["touches_edge"]).lower() != "true"
    ]
    windows = []
    for y0 in range(0, bgr.shape[0] - size + 1, size):
        for x0 in range(0, bgr.shape[1] - size + 1, size):
            local_tissue = tissue[y0 : y0 + size, x0 : x0 + size]
            local_exclusion = exclusion_mask[y0 : y0 + size, x0 : x0 + size]
            tissue_fraction = float(local_tissue.mean())
            if tissue_fraction < 0.65 or float(local_exclusion.mean()) > 0.05:
                continue
            proposal_count = sum(
                x0 <= float(row["centroid_x_px"]) < x0 + size
                and y0 <= float(row["centroid_y_px"]) < y0 + size
                for row in positive
            )
            tissue_area_mm2 = (
                local_tissue.sum() * pixel_size_um**2 / 1_000_000.0
            )
            windows.append(
                {
                    "candidate_id": f"{x0}_{y0}",
                    "x_px": x0,
                    "y_px": y0,
                    "size_px": size,
                    "tissue_fraction": tissue_fraction,
                    "proposal_count": proposal_count,
                    "proposal_density_mm2": proposal_count / tissue_area_mm2,
                }
            )
    if not windows:
        raise ValueError("NeuN field audit found no eligible tissue windows")
    return evenly_spaced_candidates(
        windows, count, value_key="proposal_density_mm2"
    )


def positive_overlay(
    bgr: np.ndarray,
    profile_labels: np.ndarray,
    objects: list[dict[str, object]],
) -> np.ndarray:
    accepted = {
        int(row["source_label"])
        for row in objects
        if row["source_kind"] == "dab_profile"
        and row["candidate_class"] == "positive_profile"
        and str(row["touches_edge"]).lower() != "true"
    }
    mask = np.isin(profile_labels, list(accepted))
    overlay = bgr.copy()
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(overlay, contours, -1, COLORS["positive_profile"], 3)
    return overlay


def classified_overlay(
    bgr: np.ndarray,
    profile_labels: np.ndarray,
    cellpose_labels: np.ndarray,
    objects: list[dict[str, object]],
    probability_threshold: float = 0.5,
) -> np.ndarray:
    overlay = bgr.copy()
    for row in objects:
        if float(row["positive_probability"]) < probability_threshold:
            continue
        labels = (
            profile_labels
            if row["source_kind"] == "dab_profile"
            else cellpose_labels
        )
        mask = (labels == int(row["source_label"])).astype(np.uint8)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(overlay, contours, -1, COLORS["positive_profile"], 3)
    return overlay


def render_field_panel(
    bgr: np.ndarray,
    profile_labels: np.ndarray,
    objects: list[dict[str, object]],
    row: dict[str, object],
    panel_px: int,
) -> np.ndarray:
    x0 = int(row["x_px"])
    y0 = int(row["y_px"])
    size = int(row["size_px"])
    raw = bgr[y0 : y0 + size, x0 : x0 + size]
    overlay = positive_overlay(bgr, profile_labels, objects)[
        y0 : y0 + size, x0 : x0 + size
    ]
    raw = cv2.resize(raw, (panel_px, panel_px), interpolation=cv2.INTER_AREA)
    overlay = cv2.resize(overlay, (panel_px, panel_px), interpolation=cv2.INTER_AREA)
    panel = np.concatenate((raw, overlay), axis=1)
    header = np.full((36, panel.shape[1], 3), 255, dtype=np.uint8)
    cv2.putText(
        header,
        f"{row['review_id']}  proposed={row['proposal_count']}",
        (8, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )
    return np.concatenate((header, panel), axis=0)


def render_raw_field_panel(
    bgr: np.ndarray,
    row: dict[str, object],
    panel_px: int,
) -> np.ndarray:
    x0 = int(row["x_px"])
    y0 = int(row["y_px"])
    size = int(row["size_px"])
    panel = cv2.resize(
        bgr[y0 : y0 + size, x0 : x0 + size],
        (panel_px, panel_px),
        interpolation=cv2.INTER_AREA,
    )
    header = np.full((36, panel.shape[1], 3), 255, dtype=np.uint8)
    cv2.putText(
        header,
        str(row["review_id"]),
        (8, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )
    return np.concatenate((header, panel), axis=0)


def write_panels(
    panels: list[tuple[str, np.ndarray]],
    output_dir: Path,
    columns: int,
    rows: int,
) -> None:
    if not panels:
        raise ValueError("NeuN review contains no panels")
    patches_dir = output_dir / "patches"
    sheets_dir = output_dir / "sheets"
    patches_dir.mkdir(parents=True, exist_ok=True)
    sheets_dir.mkdir(parents=True, exist_ok=True)
    images = []
    for review_id, panel in panels:
        path = patches_dir / f"{review_id}.jpg"
        if not cv2.imwrite(str(path), panel, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise OSError(f"Could not write NeuN review panel: {path}")
        images.append(panel)
    page_size = columns * rows
    blank = np.full_like(images[0], 255)
    for start in range(0, len(images), page_size):
        page = images[start : start + page_size]
        page.extend([blank] * (page_size - len(page)))
        sheet = np.concatenate(
            [
                np.concatenate(page[index : index + columns], axis=1)
                for index in range(0, page_size, columns)
            ],
            axis=0,
        )
        path = sheets_dir / f"sheet_{start // page_size + 1:02d}.jpg"
        if not cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 94]):
            raise OSError(f"Could not write NeuN review sheet: {path}")


__all__ = [
    "REVIEW_OUTLINE_COLOR",
    "positive_overlay",
    "classified_overlay",
    "render_candidate_panel",
    "render_field_panel",
    "render_raw_field_panel",
    "select_candidate_review",
    "select_field_windows",
    "target_overlay",
    "write_panels",
]
