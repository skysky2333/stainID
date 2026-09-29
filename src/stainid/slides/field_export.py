from __future__ import annotations

import csv
import shutil
import subprocess
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

from aicsimageio.readers.bioformats_reader import BioFile
from PIL import Image

from stainid.slides.export import read_core, write_png


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Manifest contains no rows: {path}")
    return rows


def resolve_image_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def build_export_rows(
    annotation_manifest: Path,
    field_manifest: Path,
) -> list[dict[str, str | int]]:
    images = {row["image_id"]: row for row in read_rows(annotation_manifest)}
    fields = read_rows(field_manifest)
    if len({row["image_id"] for row in fields}) != len(fields):
        raise ValueError("Field manifest must contain one row per image_id")
    if set(images) != {row["image_id"] for row in fields}:
        raise ValueError("Annotation and field manifest image sets disagree")

    project_root = annotation_manifest.parent.parent.parent
    exports = []
    for field in fields:
        image = images[field["image_id"]]
        x = int(field["x_px"])
        y = int(field["y_px"])
        width = int(field["width_px"])
        height = int(field["height_px"])
        native_width = int(image["native_width_px"])
        native_height = int(image["native_height_px"])
        if min(x, y, width, height) < 0 or width == 0 or height == 0:
            raise ValueError(f"Invalid field geometry for {field['image_id']}")
        if x + width > native_width or y + height > native_height:
            raise ValueError(f"Field is outside image bounds for {field['image_id']}")
        image_path = resolve_image_path(project_root, image["image_path"])
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        exports.append(
            {
                "image_id": field["image_id"],
                "image_path": str(image_path),
                "x": x,
                "y": y,
                "width": width,
                "height": height,
            }
        )
        source_fields = (
            "slide_path",
            "full_scene",
            "center_x_px",
            "center_y_px",
            "diameter_x_px",
            "diameter_y_px",
        )
        if all(image.get(name) for name in source_fields):
            slide_path = resolve_image_path(project_root, image["slide_path"])
            if not slide_path.is_file():
                raise FileNotFoundError(slide_path)
            core_width = round(float(image["diameter_x_px"]))
            core_height = round(float(image["diameter_y_px"]))
            core_x = round(float(image["center_x_px"]) - core_width / 2)
            core_y = round(float(image["center_y_px"]) - core_height / 2)
            exports[-1].update(
                slide_path=str(slide_path),
                full_scene=int(image["full_scene"]),
                slide_x=core_x + x,
                slide_y=core_y + y,
            )
    return exports


def valid_output(path: Path, width: int, height: int) -> bool:
    if not path.is_file():
        return False
    with Image.open(path) as image:
        return image.size == (width, height)


def export_field(
    row: dict[str, str | int],
    output_dir: Path,
    overwrite: bool,
) -> Path:
    output = output_dir / f"{row['image_id']}.png"
    width = int(row["width"])
    height = int(row["height"])
    if not overwrite and valid_output(output, width, height):
        return output
    temporary = output.with_suffix(".tmp.png")
    subprocess.run(
        [
            "vips",
            "crop",
            str(row["image_path"]),
            str(temporary),
            str(row["x"]),
            str(row["y"]),
            str(width),
            str(height),
        ],
        check=True,
    )
    if not valid_output(temporary, width, height):
        raise ValueError(f"Invalid exported field: {temporary}")
    temporary.replace(output)
    return output


def export_fields(
    annotation_manifest: Path,
    field_manifest: Path,
    output_dir: Path,
    workers: int = 2,
    overwrite: bool = False,
) -> list[Path]:
    if workers < 1:
        raise ValueError("workers must be at least 1")
    rows = build_export_rows(annotation_manifest, field_manifest)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_groups: dict[tuple[str, int], list[dict[str, str | int]]] = defaultdict(list)
    fallback = []
    for row in rows:
        if "slide_path" in row:
            source_groups[(str(row["slide_path"]), int(row["full_scene"]))].append(row)
        else:
            fallback.append(row)

    outputs: dict[str, Path] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for (slide_path, series), group in sorted(source_groups.items()):
            pending: list[
                tuple[Future[None], dict[str, str | int], Path]
            ] = []

            def finish_first() -> None:
                future, source_row, output = pending.pop(0)
                future.result()
                if not valid_output(output, int(source_row["width"]), int(source_row["height"])):
                    raise ValueError(f"Invalid exported field: {output}")
                outputs[str(source_row["image_id"])] = output
                print(f"{source_row['image_id']}: {output}", flush=True)

            with BioFile(Path(slide_path), series=series, meta=False) as biofile:
                for row in group:
                    output = output_dir / f"{row['image_id']}.png"
                    if not overwrite and valid_output(
                        output, int(row["width"]), int(row["height"])
                    ):
                        outputs[str(row["image_id"])] = output
                        print(f"{row['image_id']}: {output}", flush=True)
                        continue
                    rgb, _, _ = read_core(
                        biofile,
                        int(row["slide_x"]),
                        int(row["slide_y"]),
                        int(row["width"]),
                        int(row["height"]),
                    )
                    pending.append((executor.submit(write_png, rgb, output), row, output))
                    if len(pending) >= workers:
                        finish_first()
                while pending:
                    finish_first()

        if fallback:
            if shutil.which("vips") is None:
                raise RuntimeError("libvips command-line tools are required")

            def run(row: dict[str, str | int]) -> Path:
                output = export_field(row, output_dir, overwrite)
                print(f"{row['image_id']}: {output}", flush=True)
                return output

            for row, output in zip(fallback, executor.map(run, fallback)):
                outputs[str(row["image_id"])] = output
    return [outputs[str(row["image_id"])] for row in rows]


__all__ = [
    "build_export_rows",
    "export_field",
    "export_fields",
    "valid_output",
]
