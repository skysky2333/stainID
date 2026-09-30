from __future__ import annotations

import csv
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import numpy as np
from aicsimageio.readers.bioformats_reader import BioFile
from PIL import Image

from stainid.qc.core_qc import QC_FIELDS

EXPORT_FIELDS = [
    "output_path",
    "export_status",
    "export_downsample",
    "output_width_px",
    "output_height_px",
    "source_x_px",
    "source_y_px",
    "source_width_px",
    "source_height_px",
    "padding_left_px",
    "padding_top_px",
    "padding_right_px",
    "padding_bottom_px",
    "file_size_bytes",
]


def read_manifest(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = list(reader.fieldnames or [])
    if not rows:
        raise ValueError(f"Manifest contains no core positions: {path}")
    required = {
        "slide",
        "slide_path",
        "tma",
        "stain",
        "core_label",
        "center_x_px",
        "center_y_px",
        "diameter_x_px",
        "diameter_y_px",
        "full_scene",
    }
    missing = sorted(required - set(fields))
    if missing:
        raise ValueError(f"Manifest is missing columns: {', '.join(missing)}")
    return fields, rows


def write_manifest(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    complete_fields = fields + [field for field in EXPORT_FIELDS + QC_FIELDS if field not in fields]
    temporary = path.with_suffix(".tmp.csv")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=complete_fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def core_bounds(row: dict[str, str]) -> tuple[int, int, int, int]:
    width = round(float(row["diameter_x_px"]))
    height = round(float(row["diameter_y_px"]))
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid dimensions for {row['slide']} {row['core_label']}")
    x = round(float(row["center_x_px"]) - width / 2)
    y = round(float(row["center_y_px"]) - height / 2)
    return x, y, width, height


def read_core(
    biofile: BioFile,
    x: int,
    y: int,
    width: int,
    height: int,
) -> tuple[np.ndarray, tuple[int, int, int, int], tuple[int, int, int, int]]:
    *_, image_height, image_width, rgb_channels = biofile.core_meta.shape
    if rgb_channels != 3 or biofile.core_meta.dtype != np.dtype("uint8"):
        raise ValueError(
            f"Expected interleaved RGB uint8 data, found {biofile.core_meta.shape} "
            f"{biofile.core_meta.dtype}"
        )

    source, padding = source_geometry(image_width, image_height, x, y, width, height)
    source_x, source_y, source_width, source_height = source
    source_right = source_x + source_width
    source_bottom = source_y + source_height
    if source_right <= source_x or source_bottom <= source_y:
        raise ValueError("Core footprint does not intersect the source image")

    patch = biofile._get_plane(
        y=slice(source_y, source_bottom),
        x=slice(source_x, source_right),
    )
    if any(padding):
        output = np.full((height, width, 3), 255, dtype=np.uint8)
        left, top, _, _ = padding
        output[top : top + source_height, left : left + source_width] = patch
        patch = output
    return patch, (source_x, source_y, source_width, source_height), padding


def source_geometry(
    image_width: int,
    image_height: int,
    x: int,
    y: int,
    width: int,
    height: int,
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    source_x = max(0, x)
    source_y = max(0, y)
    source_right = min(image_width, x + width)
    source_bottom = min(image_height, y + height)
    if source_right <= source_x or source_bottom <= source_y:
        raise ValueError("Core footprint does not intersect the source image")
    source = source_x, source_y, source_right - source_x, source_bottom - source_y
    padding = (
        source_x - x,
        source_y - y,
        x + width - source_right,
        y + height - source_bottom,
    )
    return source, padding


def write_png(rgb: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.png")
    Image.fromarray(rgb).save(temporary, format="PNG", compress_level=3)
    temporary.replace(path)


def validate_png(path: Path, expected_size: tuple[int, int]) -> None:
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(path) as image:
        if image.format != "PNG" or image.mode != "RGB" or image.size != expected_size:
            raise ValueError(
                f"Unexpected exported image {path}: {image.format}, {image.mode}, {image.size}"
            )
        image.verify()


def png_matches(path: Path, expected_size: tuple[int, int]) -> bool:
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(path) as image:
        matches = (
            image.format == "PNG"
            and image.mode == "RGB"
            and image.size == expected_size
        )
        return matches


def export_matches(
    row: dict[str, str],
    path: Path,
    expected_size: tuple[int, int],
    source: tuple[int, int, int, int],
    padding: tuple[int, int, int, int],
) -> bool:
    values = source + padding
    fields = (
        "source_x_px",
        "source_y_px",
        "source_width_px",
        "source_height_px",
        "padding_left_px",
        "padding_top_px",
        "padding_right_px",
        "padding_bottom_px",
    )
    return png_matches(path, expected_size) and all(
        row.get(field) == str(value) for field, value in zip(fields, values)
    )


def update_export_fields(
    row: dict[str, str],
    relative_path: Path,
    width: int,
    height: int,
    source: tuple[int, int, int, int],
    padding: tuple[int, int, int, int],
    file_size: int,
) -> None:
    source_x, source_y, source_width, source_height = source
    left, top, right, bottom = padding
    row.update(
        output_path=relative_path.as_posix(),
        export_status="complete",
        export_downsample="1",
        output_width_px=str(width),
        output_height_px=str(height),
        source_x_px=str(source_x),
        source_y_px=str(source_y),
        source_width_px=str(source_width),
        source_height_px=str(source_height),
        padding_left_px=str(left),
        padding_top_px=str(top),
        padding_right_px=str(right),
        padding_bottom_px=str(bottom),
        file_size_bytes=str(file_size),
    )


def export_cores(
    manifest_path: Path,
    selected_slides: set[str] | None = None,
    selected_cores: set[str] | None = None,
    overwrite: bool = False,
    write_workers: int = 2,
    tma_prefix: str = "TMA-",
) -> None:
    if write_workers < 1:
        raise ValueError("write_workers must be at least 1")
    fields, rows = read_manifest(manifest_path)
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if selected_slides and row["slide"] not in selected_slides:
            continue
        if selected_cores and row["core_label"] not in selected_cores:
            continue
        grouped[row["slide"]].append(row)
    if not grouped:
        raise ValueError("No manifest rows matched the requested slides")

    for slide_number, slide in enumerate(sorted(grouped), start=1):
        slide_rows = sorted(
            grouped[slide],
            key=lambda row: (int(row["row"]), int(row["column"])),
        )
        source_path = Path(slide_rows[0]["slide_path"])
        series = int(slide_rows[0]["full_scene"])
        pending: list[
            tuple[
                Future[None],
                int,
                dict[str, str],
                Path,
                int,
                int,
                tuple[int, int, int, int],
                tuple[int, int, int, int],
            ]
        ] = []

        def finish_first() -> None:
            future, index, row, output_path, width, height, source, padding = pending.pop(0)
            future.result()
            if not png_matches(output_path, (width, height)):
                raise ValueError(f"Invalid exported image: {output_path}")
            relative_path = output_path.relative_to(manifest_path.parent)
            row.update({field: "" for field in QC_FIELDS})
            update_export_fields(
                row,
                relative_path,
                width,
                height,
                source,
                padding,
                output_path.stat().st_size,
            )
            print(
                f"[{slide_number}/{len(grouped)}] {slide}: core {index}/{len(slide_rows)} {row['core_label']}",
                flush=True,
            )

        with BioFile(source_path, series=series, meta=False) as biofile, ThreadPoolExecutor(
            max_workers=write_workers
        ) as executor:
            for index, row in enumerate(slide_rows, start=1):
                x, y, width, height = core_bounds(row)
                relative_path = (
                    Path("cores")
                    / f"{tma_prefix}{row['tma']}"
                    / row["core_label"]
                    / f"{row['stain']}.png"
                )
                output_path = manifest_path.parent / relative_path
                image_height = biofile.core_meta.shape[-3]
                image_width = biofile.core_meta.shape[-2]
                source, padding = source_geometry(
                    image_width, image_height, x, y, width, height
                )
                if (
                    output_path.exists()
                    and not overwrite
                    and export_matches(
                        row, output_path, (width, height), source, padding
                    )
                ):
                    update_export_fields(
                        row,
                        relative_path,
                        width,
                        height,
                        source,
                        padding,
                        output_path.stat().st_size,
                    )
                    print(
                        f"[{slide_number}/{len(grouped)}] {slide}: core {index}/{len(slide_rows)} {row['core_label']}",
                        flush=True,
                    )
                else:
                    rgb, source, padding = read_core(biofile, x, y, width, height)
                    pending.append(
                        (
                            executor.submit(write_png, rgb, output_path),
                            index,
                            row,
                            output_path,
                            width,
                            height,
                            source,
                            padding,
                        )
                    )
                    if len(pending) >= write_workers:
                        finish_first()
            while pending:
                finish_first()
        write_manifest(manifest_path, fields, rows)


__all__ = [
    "core_bounds",
    "export_matches",
    "export_cores",
    "png_matches",
    "read_core",
    "read_manifest",
    "source_geometry",
    "validate_png",
]
