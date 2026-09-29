from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d

from stainid.qc.core_qc import classify_coverage
from stainid.slides.vsi import SlidePreview, read_slide_preview

SLIDE_PATTERN = re.compile(r"TMA LIP-(?P<tma>\d+) (?P<stain>6E10|AT8|NeuN)$", re.IGNORECASE)
CROP_RADIUS_SCALE = 1.12
CENTER_MAX_SHIFT_FRACTION = 0.32
CENTER_MIN_AREA_RATIO = 0.08


@dataclass(frozen=True)
class Component:
    x: float
    y: float
    width: int
    height: int
    area: int


@dataclass(frozen=True)
class GridFit:
    origin: np.ndarray
    column_vector: np.ndarray
    row_vector: np.ndarray
    radius: float
    components: tuple[Component, ...]
    inlier_components: tuple[Component, ...]
    rmse: float
    strict_threshold: float

    def center(self, row: int, column: int) -> np.ndarray:
        return self.origin + column * self.column_vector + row * self.row_vector


@dataclass(frozen=True)
class CenterRefinement:
    lattice_center: np.ndarray
    center: np.ndarray
    source: str
    offset: float
    component_area_ratio: float


def parse_slide(path: Path) -> tuple[int, str]:
    match = SLIDE_PATTERN.fullmatch(path.stem)
    if match is None:
        raise ValueError(f"Unexpected slide filename: {path.name}")
    stain = match.group("stain")
    if stain.lower() == "neun":
        stain = "NeuN"
    else:
        stain = stain.upper()
    return int(match.group("tma")), stain


def segment_for_grid(rgb: np.ndarray) -> tuple[np.ndarray, float]:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    darkness = 255 - cv2.medianBlur(gray, 5)
    threshold, mask = cv2.threshold(darkness, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    close_size = max(9, round(min(rgb.shape[:2]) / 90)) | 1
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size)),
    )
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    return mask.astype(bool), float(threshold)


def find_components(mask: np.ndarray) -> tuple[Component, ...]:
    count, _, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    image_area = mask.shape[0] * mask.shape[1]
    minimum_area = image_area * 0.0007
    maximum_area = image_area * 0.06
    components = []
    for index in range(1, count):
        _, _, width, height, area = stats[index]
        if minimum_area <= area <= maximum_area:
            components.append(
                Component(
                    x=float(centroids[index, 0]),
                    y=float(centroids[index, 1]),
                    width=int(width),
                    height=int(height),
                    area=int(area),
                )
            )
    if len(components) < 12:
        raise ValueError(f"Only {len(components)} candidate core components were detected")
    return tuple(components)


def fit_axis(profile: np.ndarray, count: int) -> tuple[float, float]:
    dimension = len(profile)
    sigma = dimension / (count * 25)
    smooth = gaussian_filter1d(profile.astype(float), sigma=sigma)
    smooth = (smooth - smooth.min()) / max(np.ptp(smooth), 1e-8)
    coordinates = np.arange(dimension)
    best_score = -np.inf
    best_origin = 0.0
    best_pitch = 0.0
    for pitch in np.linspace(dimension / (count + 1.2), dimension / (count - 0.8), 280):
        maximum_origin = dimension - (count - 1) * pitch
        for origin in np.linspace(-0.08 * pitch, maximum_origin + 0.08 * pitch, 320):
            centers = origin + np.arange(count) * pitch
            if np.any(centers < 0) or np.any(centers > dimension - 1):
                continue
            center_score = np.interp(centers, coordinates, smooth).sum()
            gaps = origin + (np.arange(count - 1) + 0.5) * pitch
            gap_score = np.interp(gaps, coordinates, smooth).sum()
            score = center_score - 0.3 * gap_score
            if score > best_score:
                best_score = score
                best_origin = float(origin)
                best_pitch = float(pitch)
    if not np.isfinite(best_score):
        raise ValueError("Could not fit a regular grid axis")
    return best_origin, best_pitch


def _weighted_affine(
    rows: np.ndarray,
    columns: np.ndarray,
    points: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    design = np.column_stack([np.ones(len(rows)), columns, rows])
    weighted_design = design * np.sqrt(weights)[:, None]
    weighted_points = points * np.sqrt(weights)[:, None]
    coefficients, _, _, _ = np.linalg.lstsq(weighted_design, weighted_points, rcond=None)
    return coefficients


def fit_grid(mask: np.ndarray, rows: int, columns: int, strict_threshold: float) -> GridFit:
    components = find_components(mask)
    origin_x, pitch_x = fit_axis(mask.mean(axis=0), columns)
    origin_y, pitch_y = fit_axis(mask.mean(axis=1), rows)
    initial_centers = np.array(
        [
            [origin_x + column * pitch_x, origin_y + row * pitch_y]
            for row in range(rows)
            for column in range(columns)
        ]
    )

    areas = np.array([component.area for component in components])
    reference_count = min(len(areas), rows * columns - 4)
    reference_area = float(np.median(np.sort(areas)[-reference_count:]))
    selected: dict[tuple[int, int], Component] = {}
    for component in components:
        distances = np.linalg.norm(initial_centers - np.array([component.x, component.y]), axis=1)
        position = int(np.argmin(distances))
        row, column = divmod(position, columns)
        aspect = component.width / component.height
        if distances[position] > 0.48 * min(pitch_x, pitch_y):
            continue
        if component.area < 0.42 * reference_area or not 0.55 <= aspect <= 1.8:
            continue
        previous = selected.get((row, column))
        if previous is None or component.area > previous.area:
            selected[(row, column)] = component

    if len(selected) < 12:
        raise ValueError(f"Only {len(selected)} reliable components could be assigned to the grid")

    assigned_rows = np.array([position[0] for position in selected], dtype=float)
    assigned_columns = np.array([position[1] for position in selected], dtype=float)
    assigned_components = list(selected.values())
    points = np.array([[component.x, component.y] for component in assigned_components])
    weights = np.sqrt(np.array([component.area for component in assigned_components]) / reference_area)
    keep = np.ones(len(points), dtype=bool)
    coefficients = _weighted_affine(assigned_rows, assigned_columns, points, weights)

    for _ in range(4):
        design = np.column_stack([np.ones(len(points)), assigned_columns, assigned_rows])
        residuals = np.linalg.norm(design @ coefficients - points, axis=1)
        median = float(np.median(residuals))
        mad = float(np.median(np.abs(residuals - median)))
        limit = max(0.10 * min(pitch_x, pitch_y), median + 2.5 * mad)
        next_keep = residuals <= limit
        if next_keep.sum() < 10 or np.array_equal(next_keep, keep):
            break
        keep = next_keep
        coefficients = _weighted_affine(
            assigned_rows[keep],
            assigned_columns[keep],
            points[keep],
            weights[keep],
        )

    design = np.column_stack([np.ones(len(points)), assigned_columns, assigned_rows])
    residuals = np.linalg.norm(design @ coefficients - points, axis=1)
    rmse = float(np.sqrt(np.mean(residuals[keep] ** 2)))
    origin = coefficients[0]
    column_vector = coefficients[1]
    row_vector = coefficients[2]
    minimum_pitch = min(np.linalg.norm(column_vector), np.linalg.norm(row_vector))
    inlier_components = tuple(
        component
        for component, included in zip(assigned_components, keep)
        if included
    )
    radius = float(0.41 * minimum_pitch)
    if len(inlier_components) < 10 or rmse > 0.12 * minimum_pitch:
        raise ValueError(
            f"Unreliable grid fit: {len(inlier_components)} inliers, RMSE {rmse:.1f} pixels"
        )
    return GridFit(
        origin=origin,
        column_vector=column_vector,
        row_vector=row_vector,
        radius=radius,
        components=components,
        inlier_components=inlier_components,
        rmse=rmse,
        strict_threshold=strict_threshold,
    )


def coverage_mask(rgb: np.ndarray) -> tuple[np.ndarray, float]:
    height, width = rgb.shape[:2]
    edge = max(4, round(min(height, width) * 0.025))
    border = np.concatenate(
        [
            rgb[:edge].reshape(-1, 3),
            rgb[-edge:].reshape(-1, 3),
            rgb[:, :edge].reshape(-1, 3),
            rgb[:, -edge:].reshape(-1, 3),
        ]
    ).astype(np.float32) / 255.0
    background = np.quantile(border, 0.9, axis=0)
    darkness = np.max(
        np.clip(background - rgb.astype(np.float32) / 255.0, 0.0, None),
        axis=2,
    )
    border_darkness = np.concatenate(
        [
            darkness[:edge].ravel(),
            darkness[-edge:].ravel(),
            darkness[:, :edge].ravel(),
            darkness[:, -edge:].ravel(),
        ]
    )
    median = float(np.median(border_darkness))
    mad = float(np.median(np.abs(border_darkness - median)))
    threshold = max(0.025, median + 8 * mad)
    mask = (darkness >= threshold).astype(np.uint8)
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
    )
    return mask.astype(bool), threshold


def core_coverage(mask: np.ndarray, center: np.ndarray, radius: float) -> float:
    height, width = mask.shape
    yy, xx = np.ogrid[:height, :width]
    footprint = (xx - center[0]) ** 2 + (yy - center[1]) ** 2 <= radius**2
    return float((mask & footprint).sum() / max(footprint.sum(), 1))


def nearest_component_distance(fit: GridFit, center: np.ndarray) -> float:
    return float(
        min(
            np.hypot(component.x - center[0], component.y - center[1])
            for component in fit.components
        )
    )


def refine_core_centers(
    fit: GridFit,
    rows: int,
    columns: int,
) -> dict[tuple[int, int], CenterRefinement]:
    positions = [(row, column) for row in range(rows) for column in range(columns)]
    lattice = np.array([fit.center(row, column) for row, column in positions])
    components = list(fit.components)
    if not components:
        return {
            position: CenterRefinement(center, center, "lattice", 0.0, 0.0)
            for position, center in zip(positions, lattice)
        }

    points = np.array([[component.x, component.y] for component in components])
    costs = np.linalg.norm(lattice[:, None, :] - points[None, :, :], axis=2)
    pitch = min(np.linalg.norm(fit.column_vector), np.linalg.norm(fit.row_vector))
    area_count = min(len(components), rows * columns - 4)
    reference_area = float(
        np.median(np.sort([component.area for component in components])[-area_count:])
    )
    candidates = []
    for position_index in range(len(positions)):
        for component_index, component in enumerate(components):
            area_ratio = component.area / reference_area
            aspect = component.width / component.height
            if (
                costs[position_index, component_index]
                <= CENTER_MAX_SHIFT_FRACTION * pitch
                and area_ratio >= CENTER_MIN_AREA_RATIO
                and 0.35 <= aspect <= 2.85
            ):
                candidates.append(
                    (costs[position_index, component_index], position_index, component_index)
                )
    assigned = {}
    used_components = set()
    for _, position_index, component_index in sorted(candidates):
        if position_index not in assigned and component_index not in used_components:
            assigned[position_index] = component_index
            used_components.add(component_index)

    refinements = {}
    for index, (position, lattice_center) in enumerate(zip(positions, lattice)):
        component_index = assigned.get(index)
        if component_index is None:
            refinements[position] = CenterRefinement(
                lattice_center, lattice_center, "lattice", 0.0, 0.0
            )
            continue
        component = components[component_index]
        point = points[component_index]
        offset = float(costs[index, component_index])
        area_ratio = component.area / reference_area
        refinements[position] = CenterRefinement(
            lattice_center=lattice_center,
            center=point,
            source="component",
            offset=offset,
            component_area_ratio=area_ratio,
        )
    return refinements


def core_records(
    preview: SlidePreview,
    fit: GridFit,
    rows: int,
    columns: int,
) -> list[dict]:
    tissue, coverage_threshold = coverage_mask(preview.rgb)
    tma, stain = parse_slide(preview.path)
    refinements = refine_core_centers(fit, rows, columns)
    crop_radius = CROP_RADIUS_SCALE * fit.radius
    records = []
    for row in range(rows):
        for column in range(columns):
            refinement = refinements[row, column]
            center = refinement.center
            fraction = core_coverage(tissue, center, crop_radius)
            full_x = center[0] * preview.scale_x
            full_y = center[1] * preview.scale_y
            diameter_x = 2 * crop_radius * preview.scale_x
            diameter_y = 2 * crop_radius * preview.scale_y
            records.append(
                {
                    "slide": preview.path.stem,
                    "slide_path": str(preview.path.resolve()),
                    "tma": tma,
                    "stain": stain,
                    "row": row + 1,
                    "column_label": chr(ord("A") + column),
                    "column": column + 1,
                    "core_label": f"{chr(ord('A') + column)}-{row + 1}",
                    "center_x_px": full_x,
                    "center_y_px": full_y,
                    "lattice_center_x_px": refinement.lattice_center[0] * preview.scale_x,
                    "lattice_center_y_px": refinement.lattice_center[1] * preview.scale_y,
                    "center_source": refinement.source,
                    "center_offset_preview_px": refinement.offset,
                    "center_component_area_ratio": refinement.component_area_ratio,
                    "diameter_x_px": diameter_x,
                    "diameter_y_px": diameter_y,
                    "provisional_tissue_fraction": fraction,
                    "provisional_tissue_status": classify_coverage(fraction),
                    "provisional_empty": fraction < 0.02,
                    "nearest_component_distance_preview_px": nearest_component_distance(
                        fit, refinement.lattice_center
                    ),
                    "preview_scene": preview.preview_scene.index,
                    "full_scene": preview.full_scene.index,
                    "preview_scale_x": preview.scale_x,
                    "preview_scale_y": preview.scale_y,
                    "source_pixel_width_um": preview.pixel_width_um,
                    "source_pixel_height_um": preview.pixel_height_um,
                    "coverage_threshold": coverage_threshold,
                    "grid_rmse_preview_px": fit.rmse,
                    "grid_inlier_count": len(fit.inlier_components),
                    "estimated_core_diameter_um": (
                        2 * fit.radius * preview.scale_x * preview.pixel_width_um
                        + 2 * fit.radius * preview.scale_y * preview.pixel_height_um
                    )
                    / 2,
                    "crop_diameter_um": (
                        diameter_x * preview.pixel_width_um
                        + diameter_y * preview.pixel_height_um
                    )
                    / 2,
                }
            )
    return records


def draw_qc(preview: SlidePreview, fit: GridFit, records: list[dict]) -> np.ndarray:
    image = cv2.cvtColor(preview.rgb, cv2.COLOR_RGB2BGR)
    colors = {
        "substantial": (40, 150, 40),
        "partial": (0, 180, 255),
        "sparse": (0, 0, 255),
        "empty": (180, 0, 180),
    }
    for component in fit.components:
        cv2.circle(image, (round(component.x), round(component.y)), 5, (255, 180, 0), -1)
    for record in records:
        center = np.array(
            [
                float(record["center_x_px"]) / preview.scale_x,
                float(record["center_y_px"]) / preview.scale_y,
            ]
        )
        lattice_center = np.array(
            [
                float(record["lattice_center_x_px"]) / preview.scale_x,
                float(record["lattice_center_y_px"]) / preview.scale_y,
            ]
        )
        point = (round(center[0]), round(center[1]))
        lattice_point = (round(lattice_center[0]), round(lattice_center[1]))
        color = colors[str(record["provisional_tissue_status"])]
        radius = round(float(record["diameter_x_px"]) / preview.scale_x / 2)
        if point != lattice_point:
            cv2.line(image, lattice_point, point, (220, 80, 20), 3)
            cv2.drawMarker(image, lattice_point, (220, 80, 20), cv2.MARKER_TILTED_CROSS, 18, 2)
        cv2.circle(image, point, radius, color, 5)
        cv2.drawMarker(image, point, color, cv2.MARKER_CROSS, 24, 3)
        label = f"{record['core_label']} {float(record['provisional_tissue_fraction']):.2f}"
        cv2.putText(
            image,
            label,
            (point[0] - radius, point[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )
    title = (
        f"{preview.path.stem} | inliers={len(fit.inlier_components)} | "
        f"RMSE={fit.rmse:.1f}px | crop diameter="
        f"{float(records[0]['diameter_x_px']) / preview.scale_x:.1f}px"
    )
    cv2.rectangle(image, (0, 0), (min(image.shape[1], 1500), 45), (255, 255, 255), -1)
    cv2.putText(
        image,
        title,
        (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )
    return image


def write_table(records: list[dict], path: Path, delimiter: str = ",") -> None:
    fields = list(records[0])
    if path.exists():
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter=delimiter)
            existing = list(reader)
            existing_fields = list(reader.fieldnames or [])
        key = lambda row: (str(row["slide"]), str(row["core_label"]))
        existing_by_key = {key(record): record for record in existing}
        if len(existing_by_key) != len(existing):
            raise ValueError(f"Duplicate slide/core rows in {path}")
        updates = {key(record): record for record in records}
        if len(updates) != len(records):
            raise ValueError("Duplicate slide/core rows in detected records")
        unknown = sorted(set(updates) - set(existing_by_key))
        if unknown:
            raise ValueError(f"Detected rows are absent from {path}: {unknown}")
        merged = []
        for existing_record in existing:
            update = updates.get(key(existing_record))
            if update is None:
                merged.append(existing_record)
                continue
            record = dict(update)
            record.update(
                {
                    field: existing_record.get(field, "")
                    for field in existing_fields
                    if field not in record
                }
            )
            merged.append(record)
        records = merged
        fields.extend(field for field in existing_fields if field not in fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            delimiter=delimiter,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(records)


def process_slide(
    path: Path,
    rows: int,
    columns: int,
    target_long_side: int,
    qc_dir: Path | None = None,
) -> list[dict]:
    preview = read_slide_preview(path, target_long_side)
    strict_mask, strict_threshold = segment_for_grid(preview.rgb)
    fit = fit_grid(strict_mask, rows, columns, strict_threshold)
    records = core_records(preview, fit, rows, columns)
    slug = path.stem.replace(" ", "_")

    if qc_dir is not None:
        qc_path = qc_dir / f"{slug}.jpg"
        qc_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(qc_path), draw_qc(preview, fit, records)):
            raise OSError(f"Could not write QC image: {qc_path}")
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("slides_dir", type=Path)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/core_manifest.csv"),
    )
    parser.add_argument("--qc-dir", type=Path)
    parser.add_argument("--rows", type=int, default=5)
    parser.add_argument("--columns", type=int, default=6)
    parser.add_argument("--target-long-side", type=int, default=4500)
    parser.add_argument("--slide", action="append", default=[])
    parser.add_argument("--qc-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    slides = sorted(args.slides_dir.glob("*.vsi"))
    if args.slide:
        requested = set(args.slide)
        slides = [
            slide
            for slide in slides
            if slide.stem in requested or slide.name in requested
        ]
    if not slides:
        raise ValueError(f"No VSI slides found in {args.slides_dir}")
    if args.qc_only and args.qc_dir is None:
        raise ValueError("--qc-only requires --qc-dir")

    all_records = []
    summaries = []
    for slide in slides:
        records = process_slide(
            slide,
            args.rows,
            args.columns,
            args.target_long_side,
            args.qc_dir,
        )
        all_records.extend(records)
        summary = {
            "slide": slide.stem,
            "substantial": sum(
                record["provisional_tissue_status"] == "substantial"
                for record in records
            ),
            "partial": sum(
                record["provisional_tissue_status"] == "partial" for record in records
            ),
            "sparse": sum(
                record["provisional_tissue_status"] == "sparse" for record in records
            ),
            "empty": sum(
                record["provisional_tissue_status"] == "empty" for record in records
            ),
            "grid_inlier_count": records[0]["grid_inlier_count"],
            "grid_rmse_preview_px": records[0]["grid_rmse_preview_px"],
            "estimated_core_diameter_um": records[0]["estimated_core_diameter_um"],
        }
        summaries.append(summary)
        print(f"{slide.name}: {summary}")

    if not args.qc_only:
        write_table(all_records, args.manifest)


if __name__ == "__main__":
    main()
