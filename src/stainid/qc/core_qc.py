from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import cv2
import numpy as np

from stainid.imaging.color import rgb_to_hed
from stainid.qc.focus import FocusTile, measure_focus

QC_FIELDS = [
    "tissue_fraction",
    "largest_fragment_fraction",
    "tissue_fragment_count",
    "tissue_status",
    "tissue_threshold",
    "background_brightness",
    "hematoxylin_od_p50",
    "hematoxylin_od_p90",
    "hematoxylin_od_p99",
    "dab_od_p50",
    "dab_od_p90",
    "dab_od_p99",
    "focus_reference",
    "focus_p10",
    "focus_tile_count",
    "low_focus_tile_count",
    "low_focus_tissue_fraction",
    "focus_review_required",
    "review_required",
]


@dataclass(frozen=True)
class TissueQc:
    tissue_fraction: float
    largest_fragment_fraction: float
    fragment_count: int
    status: str
    threshold: float
    background_brightness: float
    hematoxylin_quantiles: tuple[float, float, float]
    dab_quantiles: tuple[float, float, float]


def circular_footprint(
    shape: tuple[int, int],
    radius_fraction: float = 0.49,
) -> np.ndarray:
    height, width = shape
    yy, xx = np.ogrid[:height, :width]
    radius = min(height, width) * radius_fraction
    return (xx - (width - 1) / 2) ** 2 + (yy - (height - 1) / 2) ** 2 <= radius**2


def tissue_mask(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    footprint = circular_footprint(rgb.shape[:2])
    background_pixels = rgb[~footprint].astype(np.float32) / 255.0
    background = np.quantile(background_pixels, 0.9, axis=0)
    darkness = np.max(
        np.clip(background - rgb.astype(np.float32) / 255.0, 0.0, None),
        axis=2,
    )
    background_darkness = darkness[~footprint]
    median = float(np.median(background_darkness))
    mad = float(np.median(np.abs(background_darkness - median)))
    threshold = max(0.025, median + 8.0 * mad)
    mask = (darkness >= threshold) & footprint
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel_open)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
    return mask.astype(bool), footprint, threshold


def classify_coverage(tissue_fraction: float) -> str:
    if tissue_fraction < 0.02:
        return "empty"
    if tissue_fraction < 0.20:
        return "sparse"
    if tissue_fraction < 0.75:
        return "partial"
    return "substantial"


def measure_tissue(rgb: np.ndarray) -> tuple[TissueQc, np.ndarray, np.ndarray]:
    mask, footprint, threshold = tissue_mask(rgb)
    footprint_area = int(footprint.sum())
    components, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8),
        8,
    )
    minimum_fragment_area = max(16, round(footprint_area * 0.0002))
    kept_labels = [
        index
        for index in range(1, components)
        if stats[index, cv2.CC_STAT_AREA] >= minimum_fragment_area
    ]
    fragment_areas = [int(stats[index, cv2.CC_STAT_AREA]) for index in kept_labels]
    kept = np.isin(labels, kept_labels)
    tissue_fraction = float(kept.sum() / footprint_area)
    largest_fragment_fraction = float(max(fragment_areas, default=0) / footprint_area)
    background_brightness = float(np.median(rgb[~footprint]) / 255.0)
    if kept.any():
        hed = rgb_to_hed(rgb)
        quantiles = [0.5, 0.9, 0.99]
        hematoxylin_quantiles = tuple(
            float(value) for value in np.quantile(hed[..., 0][kept], quantiles)
        )
        dab_quantiles = tuple(
            float(value) for value in np.quantile(hed[..., 2][kept], quantiles)
        )
    else:
        hematoxylin_quantiles = (float("nan"),) * 3
        dab_quantiles = (float("nan"),) * 3
    qc = TissueQc(
        tissue_fraction=tissue_fraction,
        largest_fragment_fraction=largest_fragment_fraction,
        fragment_count=len(fragment_areas),
        status=classify_coverage(tissue_fraction),
        threshold=threshold,
        background_brightness=background_brightness,
        hematoxylin_quantiles=hematoxylin_quantiles,
        dab_quantiles=dab_quantiles,
    )
    return qc, kept, footprint


def read_for_qc(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    bgr = cv2.imread(str(path), cv2.IMREAD_REDUCED_COLOR_8)
    if bgr is None:
        raise ValueError(f"Could not read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def write_overlay(
    rgb: np.ndarray,
    tissue: np.ndarray,
    footprint: np.ndarray,
    focus_tiles: list[FocusTile],
    path: Path,
) -> None:
    overlay = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    tissue_contours, _ = cv2.findContours(
        tissue.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    footprint_contours, _ = cv2.findContours(
        footprint.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(overlay, footprint_contours, -1, (0, 180, 255), 2)
    cv2.drawContours(overlay, tissue_contours, -1, (0, 160, 0), 2)
    for tile in focus_tiles:
        if tile.low_focus:
            cv2.rectangle(
                overlay,
                (tile.x, tile.y),
                (tile.x + tile.width - 1, tile.y + tile.height - 1),
                (0, 0, 255),
                3,
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), overlay):
        raise OSError(f"Could not write overlay: {path}")


def overlay_path(manifest_path: Path, row: dict[str, str]) -> Path:
    relative = Path(row["output_path"])
    if relative.parts[0] == "cores":
        relative = Path(*relative.parts[1:])
    return manifest_path.parent / "qc" / "cores" / relative


def write_manifest(
    manifest_path: Path,
    fields: list[str],
    rows: list[dict[str, str]],
) -> None:
    complete_fields = fields + [field for field in QC_FIELDS if field not in fields]
    temporary = manifest_path.with_suffix(".tmp.csv")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=complete_fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(manifest_path)


def format_metric(value: float) -> str:
    return f"{value:.6f}" if np.isfinite(value) else ""


def measure_manifest_row(
    manifest_path: Path,
    write_overlays: bool,
    row: dict[str, str],
) -> dict[str, str]:
    image_path = Path(row["output_path"])
    if not image_path.is_absolute():
        image_path = manifest_path.parent / image_path
    rgb = read_for_qc(image_path)
    qc, mask, footprint = measure_tissue(rgb)
    focus, focus_tiles = measure_focus(rgb, mask)
    focus_review = focus.low_focus_tissue_fraction >= 0.02
    if write_overlays:
        write_overlay(rgb, mask, footprint, focus_tiles, overlay_path(manifest_path, row))
    return {
        "tissue_fraction": f"{qc.tissue_fraction:.6f}",
        "largest_fragment_fraction": f"{qc.largest_fragment_fraction:.6f}",
        "tissue_fragment_count": str(qc.fragment_count),
        "tissue_status": qc.status,
        "tissue_threshold": f"{qc.threshold:.6f}",
        "background_brightness": f"{qc.background_brightness:.6f}",
        "hematoxylin_od_p50": format_metric(qc.hematoxylin_quantiles[0]),
        "hematoxylin_od_p90": format_metric(qc.hematoxylin_quantiles[1]),
        "hematoxylin_od_p99": format_metric(qc.hematoxylin_quantiles[2]),
        "dab_od_p50": format_metric(qc.dab_quantiles[0]),
        "dab_od_p90": format_metric(qc.dab_quantiles[1]),
        "dab_od_p99": format_metric(qc.dab_quantiles[2]),
        "focus_reference": format_metric(focus.reference),
        "focus_p10": format_metric(focus.p10),
        "focus_tile_count": str(focus.tile_count),
        "low_focus_tile_count": str(focus.low_focus_tile_count),
        "low_focus_tissue_fraction": f"{focus.low_focus_tissue_fraction:.6f}",
        "focus_review_required": str(focus_review).lower(),
        "review_required": str(qc.status != "substantial" or focus_review).lower(),
    }


def analyze_manifest(
    manifest_path: Path,
    write_overlays: bool,
    overwrite: bool = False,
    workers: int = 1,
) -> Path:
    if workers < 1:
        raise ValueError("workers must be at least 1")
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = list(reader.fieldnames or [])
    if not rows:
        raise ValueError(f"Manifest contains no cores: {manifest_path}")

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["slide"]].append(row)

    measure = partial(measure_manifest_row, manifest_path, write_overlays)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for slide, slide_rows in grouped.items():
            pending = [
                row
                for row in slide_rows
                if overwrite
                or not row.get("tissue_status")
                or (write_overlays and not overlay_path(manifest_path, row).exists())
            ]
            for row, metrics in zip(pending, executor.map(measure, pending)):
                row.update(metrics)
            write_manifest(manifest_path, fields, rows)
            print(f"{slide}: {len(slide_rows)} cores", flush=True)
    return manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/core_manifest.csv"),
    )
    parser.add_argument("--overlays", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = analyze_manifest(
        args.manifest,
        args.overlays,
        args.overwrite,
        args.workers,
    )
    print(output_path)


if __name__ == "__main__":
    main()
