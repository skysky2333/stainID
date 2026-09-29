from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build_core_exclusions(
    annotation_manifest: Path,
    field_manifest: Path,
    sources: list[tuple[str, Path]],
) -> dict[str, list[dict[str, object]]]:
    annotations = {row["image_id"]: row for row in read_csv(annotation_manifest)}
    fields = {row["image_id"]: row for row in read_csv(field_manifest)}
    output: dict[str, list[dict[str, object]]] = {}
    for stain, path in sources:
        with path.open(encoding="utf-8") as handle:
            exclusions = json.load(handle)
        for image_id, polygons in exclusions.items():
            if image_id not in annotations or image_id not in fields:
                raise ValueError(f"Unknown exclusion field: {image_id}")
            if annotations[image_id]["stain"] != stain:
                raise ValueError(f"Exclusion stain mismatch: {image_id}")
            x = int(fields[image_id]["x_px"])
            y = int(fields[image_id]["y_px"])
            image_path = annotations[image_id]["image_path"]
            for index, polygon in enumerate(polygons, start=1):
                output.setdefault(image_path, []).append(
                    {
                        "source_image_id": image_id,
                        "source_stain": stain,
                        "exclusion_id": f"{image_id}_E{index:02d}",
                        "polygon_core_px": [
                            [x + int(point[0]), y + int(point[1])]
                            for point in polygon
                        ],
                    }
                )
    return output


def rasterize_core_exclusions(
    exclusions: list[dict[str, object]],
    crop_x: int,
    crop_y: int,
    shape: tuple[int, int],
) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    for row in exclusions:
        points = np.asarray(row["polygon_core_px"], dtype=np.int32)
        points[:, 0] -= crop_x
        points[:, 1] -= crop_y
        cv2.fillPoly(mask, [points], 1)
    return mask.astype(bool)


def write_core_exclusions(
    annotation_manifest: Path,
    field_manifest: Path,
    sources: list[tuple[str, Path]],
    output_path: Path,
) -> Path:
    exclusions = build_core_exclusions(annotation_manifest, field_manifest, sources)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(exclusions, handle, indent=2)
        handle.write("\n")
    return output_path


__all__ = [
    "build_core_exclusions",
    "rasterize_core_exclusions",
    "write_core_exclusions",
]
