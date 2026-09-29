from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from stainid.slides.vsi import SlidePreview, read_slide_preview

STATUS_COLORS = {
    "substantial": (45, 145, 45),
    "partial": (0, 170, 240),
    "sparse": (20, 30, 220),
    "empty": (170, 30, 170),
}


def crop_preview(preview: SlidePreview, row: dict[str, str]) -> np.ndarray:
    center_x = float(row["center_x_px"]) / preview.scale_x
    center_y = float(row["center_y_px"]) / preview.scale_y
    width = max(1, round(float(row["diameter_x_px"]) / preview.scale_x))
    height = max(1, round(float(row["diameter_y_px"]) / preview.scale_y))
    x = round(center_x - width / 2)
    y = round(center_y - height / 2)
    output = np.full((height, width, 3), 255, dtype=np.uint8)
    source_x = max(x, 0)
    source_y = max(y, 0)
    source_right = min(x + width, preview.rgb.shape[1])
    source_bottom = min(y + height, preview.rgb.shape[0])
    if source_right > source_x and source_bottom > source_y:
        output[
            source_y - y : source_bottom - y,
            source_x - x : source_right - x,
        ] = preview.rgb[source_y:source_bottom, source_x:source_right]
    return output


@lru_cache(maxsize=3)
def read_cached_slide_preview(path: str, target_long_side: int) -> SlidePreview:
    return read_slide_preview(Path(path), target_long_side)


def read_core_preview(
    row: dict[str, str], target_long_side: int = 4500
) -> np.ndarray:
    geometry = (
        "slide_path",
        "center_x_px",
        "center_y_px",
        "diameter_x_px",
        "diameter_y_px",
    )
    if all(row.get(field) for field in geometry):
        return crop_preview(
            read_cached_slide_preview(row["slide_path"], target_long_side), row
        )
    image_path = Path(row["image_path"])
    if not image_path.is_absolute():
        image_path = Path.cwd() / image_path
    bgr = cv2.imread(str(image_path), cv2.IMREAD_REDUCED_COLOR_8)
    if bgr is None:
        raise ValueError(f"Could not read image: {image_path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def render_core_sheet(
    preview: SlidePreview,
    rows: list[dict[str, str]],
    tile_size: int = 260,
    label_height: int = 32,
) -> np.ndarray:
    if len(rows) != 30:
        raise ValueError(f"Expected 30 positions for {preview.path.stem}, found {len(rows)}")
    positions = {(int(row["row"]), int(row["column"])) for row in rows}
    if positions != {(row, column) for row in range(1, 6) for column in range(1, 7)}:
        raise ValueError(f"Invalid core positions for {preview.path.stem}")

    header_height = 46
    sheet = np.full(
        (header_height + 5 * (tile_size + label_height), 6 * tile_size, 3),
        245,
        dtype=np.uint8,
    )
    cv2.putText(
        sheet,
        preview.path.stem,
        (12, 31),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (25, 25, 25),
        2,
        cv2.LINE_AA,
    )
    for row in rows:
        row_index = int(row["row"]) - 1
        column_index = int(row["column"]) - 1
        x = column_index * tile_size
        y = header_height + row_index * (tile_size + label_height)
        crop = crop_preview(preview, row)
        interpolation = cv2.INTER_AREA if max(crop.shape[:2]) > tile_size else cv2.INTER_CUBIC
        tile = cv2.resize(crop, (tile_size, tile_size), interpolation=interpolation)
        sheet[y : y + tile_size, x : x + tile_size] = cv2.cvtColor(tile, cv2.COLOR_RGB2BGR)
        status = row.get("provisional_tissue_status", "")
        color = STATUS_COLORS.get(status, (80, 80, 80))
        cv2.rectangle(sheet, (x, y), (x + tile_size - 1, y + tile_size - 1), color, 4)
        source = row.get("center_source", "lattice")
        offset = float(row.get("center_offset_preview_px", 0.0) or 0.0)
        label = f"{row['core_label']}  {source}  shift={offset:.0f}px"
        cv2.putText(
            sheet,
            label,
            (x + 7, y + tile_size + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (25, 25, 25),
            1,
            cv2.LINE_AA,
        )
    return sheet


def render_manifest(
    manifest_path: Path,
    output_dir: Path,
    target_long_side: int = 4500,
) -> None:
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["slide"]].append(row)
    output_dir.mkdir(parents=True, exist_ok=True)
    for slide, slide_rows in sorted(grouped.items()):
        preview = read_slide_preview(Path(slide_rows[0]["slide_path"]), target_long_side)
        sheet = render_core_sheet(preview, slide_rows)
        output = output_dir / f"{slide.replace(' ', '_')}.jpg"
        if not cv2.imwrite(str(output), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92]):
            raise OSError(f"Could not write review sheet: {output}")
        print(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("data/core_manifest.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/qc/core_sheets"))
    parser.add_argument("--target-long-side", type=int, default=4500)
    args = parser.parse_args()
    render_manifest(args.manifest, args.output_dir, args.target_long_side)


if __name__ == "__main__":
    main()
