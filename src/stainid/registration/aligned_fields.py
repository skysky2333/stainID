from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from stainid.imaging.color import rgb_to_hed
from stainid.qc.core_qc import tissue_mask
from stainid.qc.focus import low_focus_mask, measure_focus
from stainid.sampling.fields import integral_sum, window_positions
from stainid.slides.core_sheets import read_core_preview


@dataclass(frozen=True)
class AlignedField:
    x_px: int
    y_px: int
    width_px: int
    height_px: int
    common_tissue_fraction: float
    pathology_score: float
    candidate_count: int
    selection_status: str


def matrix_from_row(row: dict[str, str]) -> np.ndarray:
    return np.array(
        [
            [float(row["native_m00"]), float(row["native_m01"]), float(row["native_m02"])],
            [float(row["native_m10"]), float(row["native_m11"]), float(row["native_m12"])],
        ],
        dtype=np.float64,
    )


def preview_matrix(
    native_matrix: np.ndarray,
    reference_preview_shape: tuple[int, int],
    reference_native_size: tuple[int, int],
    moving_preview_shape: tuple[int, int],
    moving_native_size: tuple[int, int],
) -> np.ndarray:
    reference_preview_height, reference_preview_width = reference_preview_shape
    reference_native_width, reference_native_height = reference_native_size
    moving_preview_height, moving_preview_width = moving_preview_shape
    moving_native_width, moving_native_height = moving_native_size
    reference_scale = np.diag(
        [
            reference_preview_width / reference_native_width,
            reference_preview_height / reference_native_height,
            1.0,
        ]
    )
    moving_inverse_scale = np.diag(
        [
            moving_native_width / moving_preview_width,
            moving_native_height / moving_preview_height,
            1.0,
        ]
    )
    return (
        reference_scale
        @ np.vstack([native_matrix, [0.0, 0.0, 1.0]])
        @ moving_inverse_scale
    )[:2]


def aligned_preview(
    moving_rgb: np.ndarray,
    matrix: np.ndarray,
    reference_shape: tuple[int, int],
    reference_native_size: tuple[int, int],
    moving_native_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    height, width = reference_shape
    transform = preview_matrix(
        matrix,
        reference_shape,
        reference_native_size,
        moving_rgb.shape[:2],
        moving_native_size,
    )
    moving_tissue, _, _ = tissue_mask(moving_rgb)
    _, focus_tiles = measure_focus(moving_rgb, moving_tissue)
    moving_tissue &= ~low_focus_mask(moving_tissue.shape, focus_tiles)
    moving_dab = rgb_to_hed(moving_rgb)[..., 2]
    aligned_tissue = cv2.warpAffine(
        moving_tissue.astype(np.uint8),
        transform,
        (width, height),
        flags=cv2.INTER_NEAREST,
    ).astype(bool)
    aligned_dab = cv2.warpAffine(
        moving_dab,
        transform,
        (width, height),
        flags=cv2.INTER_LINEAR,
    )
    return aligned_tissue, aligned_dab


def select_aligned_field(
    reference_rgb: np.ndarray,
    reference_native_size: tuple[int, int],
    moving: list[tuple[np.ndarray, np.ndarray, tuple[int, int]]],
    field_size_px: int = 2048,
    target_quantile: float = 0.85,
) -> AlignedField:
    native_width, native_height = reference_native_size
    field_width = min(field_size_px, native_width)
    field_height = min(field_size_px, native_height)
    preview_height, preview_width = reference_rgb.shape[:2]
    preview_field_width = max(1, round(field_width * preview_width / native_width))
    preview_field_height = max(1, round(field_height * preview_height / native_height))
    reference_tissue, _, _ = tissue_mask(reference_rgb)
    _, focus_tiles = measure_focus(reference_rgb, reference_tissue)
    reference_tissue &= ~low_focus_mask(reference_tissue.shape, focus_tiles)
    common = reference_tissue.copy()
    pathology = np.zeros(reference_tissue.shape, dtype=np.float32)
    for moving_rgb, matrix, moving_native_size in moving:
        aligned_tissue, aligned_dab = aligned_preview(
            moving_rgb,
            matrix,
            reference_tissue.shape,
            reference_native_size,
            moving_native_size,
        )
        common &= aligned_tissue
        pathology += aligned_dab
    pathology /= max(len(moving), 1)
    common_integral = cv2.integral(common.astype(np.float64))
    pathology_integral = cv2.integral(pathology.astype(np.float64) * common)
    candidates = []
    for y in window_positions(preview_height, preview_field_height, preview_field_height // 2):
        for x in window_positions(preview_width, preview_field_width, preview_field_width // 2):
            area = preview_field_width * preview_field_height
            tissue_area = integral_sum(
                common_integral, x, y, preview_field_width, preview_field_height
            )
            fraction = tissue_area / area
            score = (
                integral_sum(
                    pathology_integral, x, y, preview_field_width, preview_field_height
                )
                / tissue_area
                if tissue_area
                else 0.0
            )
            candidates.append((x, y, fraction, score))
    eligible = [candidate for candidate in candidates if candidate[2] >= 0.75]
    if eligible:
        status = "common_tissue"
    else:
        maximum = max(candidate[2] for candidate in candidates)
        eligible = [candidate for candidate in candidates if candidate[2] >= 0.9 * maximum]
        status = "best_available_common_tissue"
    eligible.sort(key=lambda candidate: (candidate[3], candidate[0], candidate[1]))
    selected = eligible[round(target_quantile * (len(eligible) - 1))]
    x, y, fraction, score = selected
    native_x = min(native_width - field_width, round(x * native_width / preview_width))
    native_y = min(native_height - field_height, round(y * native_height / preview_height))
    return AlignedField(
        native_x,
        native_y,
        field_width,
        field_height,
        fraction,
        score,
        len(eligible),
        status,
    )


def source_bounds(
    matrix: np.ndarray,
    field: AlignedField,
    moving_native_size: tuple[int, int],
    margin: int = 4,
) -> tuple[int, int, int, int]:
    inverse = np.linalg.inv(np.vstack([matrix, [0.0, 0.0, 1.0]]))
    corners = np.array(
        [
            [field.x_px, field.y_px, 1.0],
            [field.x_px + field.width_px, field.y_px, 1.0],
            [field.x_px, field.y_px + field.height_px, 1.0],
            [field.x_px + field.width_px, field.y_px + field.height_px, 1.0],
        ]
    )
    mapped = (inverse @ corners.T).T[:, :2]
    moving_width, moving_height = moving_native_size
    x = max(0, int(np.floor(mapped[:, 0].min())) - margin)
    y = max(0, int(np.floor(mapped[:, 1].min())) - margin)
    right = min(moving_width, int(np.ceil(mapped[:, 0].max())) + margin)
    bottom = min(moving_height, int(np.ceil(mapped[:, 1].max())) + margin)
    if right <= x or bottom <= y:
        raise ValueError("Aligned field does not intersect the moving image")
    return x, y, right - x, bottom - y


def local_matrix(
    matrix: np.ndarray,
    source_x: int,
    source_y: int,
    field: AlignedField,
) -> np.ndarray:
    local = matrix.copy()
    local[:, 2] = (
        matrix[:, :2] @ np.array([source_x, source_y], dtype=np.float64)
        + matrix[:, 2]
        - np.array([field.x_px, field.y_px], dtype=np.float64)
    )
    return local


def export_aligned_image(
    image_path: Path,
    moving_native_size: tuple[int, int],
    matrix: np.ndarray,
    field: AlignedField,
    output_path: Path,
) -> Path:
    if shutil.which("vips") is None:
        raise RuntimeError("libvips command-line tools are required")
    x, y, width, height = source_bounds(matrix, field, moving_native_size)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as directory:
        crop_path = Path(directory) / "source.png"
        subprocess.run(
            ["vips", "crop", str(image_path), str(crop_path), str(x), str(y), str(width), str(height)],
            check=True,
        )
        bgr = cv2.imread(str(crop_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Could not read temporary crop from {image_path}")
    aligned = cv2.warpAffine(
        bgr,
        local_matrix(matrix, x, y, field),
        (field.width_px, field.height_px),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    temporary = output_path.with_suffix(".tmp.png")
    if not cv2.imwrite(str(temporary), aligned, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
        raise OSError(f"Could not write aligned field: {temporary}")
    temporary.replace(output_path)
    return output_path


def render_montage(paths: dict[str, Path], output_path: Path) -> Path:
    panels = []
    for stain in ("6E10", "AT8", "NeuN"):
        panel = cv2.imread(str(paths[stain]), cv2.IMREAD_COLOR)
        if panel is None:
            raise ValueError(f"Could not read aligned field: {paths[stain]}")
        panel = cv2.resize(panel, (768, 768), interpolation=cv2.INTER_AREA)
        cv2.rectangle(panel, (0, 0), (150, 42), (255, 255, 255), -1)
        cv2.putText(panel, stain, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2)
        panels.append(panel)
    montage = np.concatenate(panels, axis=1)
    if not cv2.imwrite(str(output_path), montage, [cv2.IMWRITE_JPEG_QUALITY, 94]):
        raise OSError(f"Could not write montage: {output_path}")
    return output_path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def export_core_triplet(
    core_id: str,
    analysis_manifest: Path,
    registration_manifest: Path,
    output_dir: Path,
    field_size_px: int = 2048,
    target_quantile: float = 0.85,
    allow_coarse_registration: bool = False,
) -> tuple[Path, Path]:
    images = {
        row["stain"]: row
        for row in read_csv(analysis_manifest)
        if row["core_id"] == core_id
    }
    if set(images) != {"6E10", "AT8", "NeuN"}:
        raise ValueError(f"Core does not contain a complete stain triplet: {core_id}")
    registrations = {
        row["moving_stain"]: row
        for row in read_csv(registration_manifest)
        if row["core_id"] == core_id
    }
    if set(registrations) != {"6E10", "AT8"}:
        raise ValueError(f"Core has no complete registration: {core_id}")
    if any(row["status"] == "fail" for row in registrations.values()):
        raise ValueError(f"Core registration failed: {core_id}")
    if not allow_coarse_registration and any(
        row.get("spatial_status", "not_landmark_validated") != "validated"
        for row in registrations.values()
    ):
        raise ValueError(
            f"Core registration has not passed landmark validation: {core_id}"
        )

    reference = images["NeuN"]
    reference_rgb = read_core_preview(reference)
    reference_size = (int(reference["native_width_px"]), int(reference["native_height_px"]))
    moving_previews = []
    matrices = {}
    for stain in ("6E10", "AT8"):
        row = images[stain]
        size = (int(row["native_width_px"]), int(row["native_height_px"]))
        matrix = matrix_from_row(registrations[stain])
        matrices[stain] = matrix
        moving_previews.append((read_core_preview(row), matrix, size))
    field = select_aligned_field(
        reference_rgb,
        reference_size,
        moving_previews,
        field_size_px,
        target_quantile,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for stain in ("6E10", "AT8", "NeuN"):
        row = images[stain]
        matrix = matrices.get(stain, np.eye(2, 3, dtype=np.float64))
        size = (int(row["native_width_px"]), int(row["native_height_px"]))
        paths[stain] = export_aligned_image(
            Path(row["image_path"]),
            size,
            matrix,
            field,
            output_dir / f"{core_id}_{stain}.png",
        )
    manifest = output_dir / f"{core_id}.json"
    with manifest.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "core_id": core_id,
                "reference_stain": "NeuN",
                "field": asdict(field),
                "images": {stain: str(path) for stain, path in paths.items()},
                "registration_status": {
                    stain: registrations[stain]["status"] for stain in ("6E10", "AT8")
                },
                "spatial_status": {
                    stain: registrations[stain].get(
                        "spatial_status", "not_landmark_validated"
                    )
                    for stain in ("6E10", "AT8")
                },
                "development_only": allow_coarse_registration,
            },
            handle,
            indent=2,
        )
        handle.write("\n")
    montage = render_montage(paths, output_dir / f"{core_id}_montage.jpg")
    return manifest, montage


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("core_id")
    parser.add_argument("--analysis-manifest", type=Path, default=Path("data/analysis/core_images.csv"))
    parser.add_argument(
        "--registration-manifest",
        type=Path,
        default=Path("data/registration/all_triplets.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/pilot/aligned_fields"))
    parser.add_argument("--field-size", type=int, default=2048)
    parser.add_argument("--target-quantile", type=float, default=0.85)
    parser.add_argument("--allow-coarse-registration", action="store_true")
    args = parser.parse_args()
    for path in export_core_triplet(
        args.core_id,
        args.analysis_manifest,
        args.registration_manifest,
        args.output_dir,
        args.field_size,
        args.target_quantile,
        args.allow_coarse_registration,
    ):
        print(path)


if __name__ == "__main__":
    main()
