from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np


def tile_origins(size: int, tile_size: int, overlap: int) -> list[int]:
    if size <= tile_size:
        return [0]
    step = tile_size - overlap
    if step <= 0:
        raise ValueError("Tile overlap must be smaller than tile size")
    origins = list(range(0, size - tile_size + 1, step))
    last = size - tile_size
    if origins[-1] != last:
        origins.append(last)
    return origins


def non_max_suppression(
    boxes: np.ndarray, scores: np.ndarray, threshold: float
) -> list[int]:
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    kept: list[int] = []
    while order.size:
        current = int(order[0])
        kept.append(current)
        if order.size == 1:
            break
        rest = order[1:]
        intersection_width = np.maximum(
            0.0, np.minimum(x2[current], x2[rest]) - np.maximum(x1[current], x1[rest])
        )
        intersection_height = np.maximum(
            0.0, np.minimum(y2[current], y2[rest]) - np.maximum(y1[current], y1[rest])
        )
        intersection = intersection_width * intersection_height
        union = areas[current] + areas[rest] - intersection
        iou = np.divide(
            intersection,
            union,
            out=np.zeros_like(intersection),
            where=union > 0,
        )
        order = rest[iou <= threshold]
    return kept


def tiled_images(
    image: np.ndarray, tile_size: int, overlap: int
) -> tuple[list[np.ndarray], list[tuple[int, int, int, int]]]:
    tiles: list[np.ndarray] = []
    locations: list[tuple[int, int, int, int]] = []
    for y in tile_origins(image.shape[0], tile_size, overlap):
        for x in tile_origins(image.shape[1], tile_size, overlap):
            patch = image[y : y + tile_size, x : x + tile_size]
            tile = np.full((tile_size, tile_size, 3), 255, dtype=np.uint8)
            tile[: patch.shape[0], : patch.shape[1]] = patch
            tiles.append(tile)
            locations.append((x, y, patch.shape[1], patch.shape[0]))
    return tiles, locations


def collect_detections(
    result_batches: list[list[object]],
    location_batches: list[list[tuple[int, int, int, int]]],
    scale: float,
    image_width: int,
    image_height: int,
    nms_threshold: float,
) -> list[dict[str, float]]:
    detections: list[dict[str, float]] = []
    for results, locations in zip(result_batches, location_batches, strict=True):
        for result, (offset_x, offset_y, valid_width, valid_height) in zip(
            results, locations, strict=True
        ):
            boxes = result.boxes.xyxy.detach().cpu().numpy()
            scores = result.boxes.conf.detach().cpu().numpy()
            for box, score in zip(boxes, scores, strict=True):
                center_x = float((box[0] + box[2]) / 2.0)
                center_y = float((box[1] + box[3]) / 2.0)
                if center_x >= valid_width or center_y >= valid_height:
                    continue
                x1 = max(0.0, (float(box[0]) + offset_x) / scale)
                y1 = max(0.0, (float(box[1]) + offset_y) / scale)
                x2 = min(float(image_width), (float(box[2]) + offset_x) / scale)
                y2 = min(float(image_height), (float(box[3]) + offset_y) / scale)
                detections.append(
                    {
                        "x1_px": x1,
                        "y1_px": y1,
                        "x2_px": x2,
                        "y2_px": y2,
                        "confidence": float(score),
                    }
                )
    if not detections:
        return []
    boxes = np.asarray(
        [[row[key] for key in ("x1_px", "y1_px", "x2_px", "y2_px")] for row in detections],
        dtype=np.float64,
    )
    scores = np.asarray([row["confidence"] for row in detections], dtype=np.float64)
    return [detections[index] for index in non_max_suppression(boxes, scores, nms_threshold)]


def enrich_detections(
    detections: list[dict[str, float]],
    tissue_mask: np.ndarray,
    pixel_width_um: float,
    pixel_height_um: float,
    image_id: str,
) -> list[dict[str, float | str | bool]]:
    rows = []
    for index, detection in enumerate(detections, start=1):
        center_x = (detection["x1_px"] + detection["x2_px"]) / 2.0
        center_y = (detection["y1_px"] + detection["y2_px"]) / 2.0
        xi = min(tissue_mask.shape[1] - 1, max(0, round(center_x)))
        yi = min(tissue_mask.shape[0] - 1, max(0, round(center_y)))
        width_um = (detection["x2_px"] - detection["x1_px"]) * pixel_width_um
        height_um = (detection["y2_px"] - detection["y1_px"]) * pixel_height_um
        rows.append(
            {
                "image_id": image_id,
                "candidate_id": f"{image_id}_TT_{index:05d}",
                **detection,
                "centroid_x_px": center_x,
                "centroid_y_px": center_y,
                "width_um": width_um,
                "height_um": height_um,
                "box_area_um2": width_um * height_um,
                "inside_tissue": bool(tissue_mask[yi, xi]),
                "proposed_role": "mature_nft",
                "review_status": "zero_shot_candidate",
            }
        )
    return rows


def write_detection_csv(
    rows: list[dict[str, float | str | bool]], output_path: Path
) -> Path:
    fields = [
        "image_id",
        "candidate_id",
        "confidence",
        "x1_px",
        "y1_px",
        "x2_px",
        "y2_px",
        "centroid_x_px",
        "centroid_y_px",
        "width_um",
        "height_um",
        "box_area_um2",
        "inside_tissue",
        "proposed_role",
        "review_status",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def write_detection_geojson(
    rows: list[dict[str, float | str | bool]], output_path: Path, proposal_source: str
) -> Path:
    features = []
    for row in rows:
        x1 = float(row["x1_px"])
        y1 = float(row["y1_px"])
        x2 = float(row["x2_px"])
        y2 = float(row["y2_px"])
        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[x1, y1], [x2, y1], [x2, y2], [x1, y2], [x1, y1]]],
                },
                "properties": {
                    "objectType": "detection",
                    "name": row["candidate_id"],
                    "classification": {
                        "name": "Ambiguous AT8 object",
                        "color": [245, 145, 25],
                    },
                    "metadata": {
                        "proposal_source": proposal_source,
                        "candidate_id": row["candidate_id"],
                        "proposed_role": row["proposed_role"],
                        "confidence": str(row["confidence"]),
                        "inside_tissue": str(row["inside_tissue"]).lower(),
                        "review_status": row["review_status"],
                        "geometry_role": "zero_shot_bounding_box",
                    },
                },
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump({"type": "FeatureCollection", "features": features}, handle, indent=2)
        handle.write("\n")
    return output_path


def write_detection_overlay(
    bgr: np.ndarray,
    rows: list[dict[str, float | str | bool]],
    output_path: Path,
) -> Path:
    overlay = bgr.copy()
    for row in rows:
        color = (35, 35, 220) if row["inside_tissue"] else (190, 50, 180)
        top_left = (round(float(row["x1_px"])), round(float(row["y1_px"])))
        bottom_right = (round(float(row["x2_px"])), round(float(row["y2_px"])))
        cv2.rectangle(overlay, top_left, bottom_right, color, 3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), overlay, [cv2.IMWRITE_JPEG_QUALITY, 95]):
        raise OSError(f"Could not write Tangle-Tracer overlay: {output_path}")
    return output_path


__all__ = [
    "collect_detections",
    "enrich_detections",
    "non_max_suppression",
    "tile_origins",
    "tiled_images",
    "write_detection_csv",
    "write_detection_geojson",
    "write_detection_overlay",
]
