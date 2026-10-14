from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d

from stainid.qc.core_qc import classify_coverage
from stainid.slides.vsi import SlidePreview, read_slide_preview

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


def _at_least(fraction: float, positions: int) -> int:
    return max(3, int(np.ceil(fraction * positions)))


def tissue_blobs(mask: np.ndarray) -> tuple[Component, ...]:
    """Every connected piece of tissue on the slide preview."""
    count, _, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    return tuple(
        Component(x=float(centroids[i, 0]), y=float(centroids[i, 1]), width=int(stats[i, 2]), height=int(stats[i, 3]), area=int(stats[i, 4]))
        for i in range(1, count)
    )


def find_components(mask: np.ndarray, positions: int = 30, blobs: tuple[Component, ...] | None = None) -> tuple[Component, ...]:
    """Tissue blobs that could be cores; size limits scale with the grid (for a 5 x 6 grid: 0.07-6% of the slide)."""
    cell_area = mask.shape[0] * mask.shape[1] / positions
    minimum_area = cell_area * 0.021
    maximum_area = cell_area * 1.8
    components = [blob for blob in (tissue_blobs(mask) if blobs is None else blobs) if minimum_area <= blob.area <= maximum_area]
    if len(components) < _at_least(0.4, positions):
        raise ValueError(f"Only {len(components)} core-sized tissue pieces were found on the slide preview; "
                         "check the rows and columns in Settings and that the scan shows the whole TMA")
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
        origins = np.linspace(-0.08 * pitch, maximum_origin + 0.08 * pitch, 320)
        centers = origins[:, None] + np.arange(count) * pitch
        inside = ~(np.any(centers < 0, axis=1) | np.any(centers > dimension - 1, axis=1))
        if not inside.any():
            continue
        center_score = np.interp(centers[inside], coordinates, smooth).sum(axis=1)
        gaps = origins[inside][:, None] + (np.arange(count - 1) + 0.5) * pitch
        gap_score = np.interp(gaps, coordinates, smooth).sum(axis=1)
        score = center_score - 0.3 * gap_score
        best = int(np.argmax(score))  # the first of equal scores, as one origin after the other would give
        if score[best] > best_score:
            best_score = float(score[best])
            best_origin = float(origins[inside][best])
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


def fit_grid(
    mask: np.ndarray,
    rows: int,
    columns: int,
    strict_threshold: float,
    blobs: tuple[Component, ...] | None = None,
    axes: dict | None = None,
) -> GridFit:
    """Fit a rows x columns lattice to the tissue mask. `blobs` (from `tissue_blobs`) and `axes` (a dict filled with
    the axis fits) let several grid sizes be tried on one slide without repeating the shared work."""
    positions = rows * columns
    components = find_components(mask, positions, blobs)
    axes = {} if axes is None else axes
    if ("x", columns) not in axes:
        axes["x", columns] = fit_axis(mask.mean(axis=0), columns)
    if ("y", rows) not in axes:
        axes["y", rows] = fit_axis(mask.mean(axis=1), rows)
    origin_x, pitch_x = axes["x", columns]
    origin_y, pitch_y = axes["y", rows]
    initial_centers = np.array(
        [
            [origin_x + column * pitch_x, origin_y + row * pitch_y]
            for row in range(rows)
            for column in range(columns)
        ]
    )

    areas = np.array([component.area for component in components])
    reference_count = max(1, min(len(areas), round(positions * 26 / 30)))
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

    if len(selected) < _at_least(0.4, positions):
        raise ValueError(f"Only {len(selected)} cores could be matched to a {rows} x {columns} grid; check the rows and columns in Settings")

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
        if next_keep.sum() < _at_least(0.33, positions) or np.array_equal(next_keep, keep):
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
    if len(inlier_components) < _at_least(0.33, positions) or rmse > 0.12 * minimum_pitch:
        raise ValueError(
            f"The {rows} x {columns} grid does not fit this slide well ({len(inlier_components)} cores fit, "
            f"error {rmse:.1f} preview pixels); check the rows and columns in Settings"
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


# ---------------------------------------------------------------- choosing the grid size when a slide has none set

GRID_SIZES = range(2, 13)   # rows and columns tried when looking for a slide's grid size
KEEP_DEFAULT_MARGIN = 0.03  # the project's grid size is kept unless another size fits clearly better


@dataclass(frozen=True)
class GridChoice:
    rows: int
    columns: int
    fit: GridFit
    score: float
    source: str  # "set": the size was given; "auto": found by trying sizes
    tried: tuple[tuple[int, int, float | None], ...] = ()


def core_like(blobs: tuple[Component, ...], shape: tuple[int, int]) -> tuple[Component, ...]:
    """The blobs about the size of the larger ones (the cores), whatever the grid: crumbs and smears are left out."""
    pieces = [b for b in blobs if b.area >= 0.0002 * shape[0] * shape[1] and 0.4 <= b.width / b.height <= 2.5]
    if not pieces:
        return ()
    areas = np.sort([b.area for b in pieces])
    reference = float(np.median(areas[len(areas) // 2:]))
    return tuple(b for b in pieces if 0.3 * reference <= b.area <= 3.0 * reference)


def count_lines(values: np.ndarray, size: float) -> int:
    """How many rows (or columns) the core centres fall into: centres closer than half a core belong to one line."""
    if len(values) == 0:
        return 0
    ordered = np.sort(values)
    return int(1 + np.sum(np.diff(ordered) > 0.5 * size))


def estimate_grid(cores: tuple[Component, ...]) -> tuple[int, int]:
    """Rows and columns read off the core positions (a first guess; the fit decides)."""
    if not cores:
        return 0, 0
    size = float(np.median([np.sqrt(c.area) for c in cores]))
    return count_lines(np.array([c.y for c in cores]), size), count_lines(np.array([c.x for c in cores]), size)


def grid_score(fit: GridFit, rows: int, columns: int, cores: int) -> float:
    """How well a grid size explains the slide (0-1): most positions hold a core (too many rows or columns leave
    empty ones), most cores sit on a position (too few leave cores out), and the cores lie on a regular lattice."""
    matched = len(fit.inlier_components)
    filled = matched / (rows * columns)
    explained = matched / max(cores, matched, 1)
    if not filled or not explained:
        return 0.0
    pitch = min(np.linalg.norm(fit.column_vector), np.linalg.norm(fit.row_vector))
    regular = 1 - 0.5 * min(1.0, fit.rmse / (0.12 * pitch)) ** 2
    return float(2 * filled * explained / (filled + explained) * regular)


def tissue_window(cores: tuple[Component, ...], shape: tuple[int, int]) -> tuple[int, int, int, int] | None:
    """The part of the preview the cores occupy, with a margin of about half a core spacing: a grid is fitted on the
    assumption that it fills the picture, which an array scanned with wide empty margins does not."""
    if len(cores) < 4:
        return None
    points = np.array([[c.x, c.y] for c in cores])
    gaps = np.sort(np.linalg.norm(points[:, None] - points[None], axis=2), axis=1)[:, 1]
    margin = 0.7 * float(np.median(gaps))
    x0, y0 = (max(0, int(v - margin)) for v in points.min(axis=0))
    x1, y1 = int(min(shape[1], points[:, 0].max() + margin)), int(min(shape[0], points[:, 1].max() + margin))
    return x0, y0, x1, y1


def _moved(fit: GridFit, dx: float, dy: float) -> GridFit:
    shift = lambda components: tuple(Component(c.x + dx, c.y + dy, c.width, c.height, c.area) for c in components)  # noqa: E731
    return GridFit(fit.origin + np.array([dx, dy]), fit.column_vector, fit.row_vector, fit.radius, shift(fit.components),
                   shift(fit.inlier_components), fit.rmse, fit.strict_threshold)


def choose_grid(
    mask: np.ndarray,
    strict_threshold: float,
    rows: int | None = None,
    columns: int | None = None,
    default: tuple[int, int] = (5, 6),
) -> GridChoice:
    """The slide's grid: the given rows and columns when both are set, otherwise the best-fitting size among the project
    default and the sizes around the number of core rows and columns counted on the slide (2-12 each). Sizes are also
    fitted on the part of the slide the cores occupy, so wide empty margins do not push the fit to a wrong size."""
    blobs = tissue_blobs(mask)
    cores = core_like(blobs, mask.shape)
    if rows and columns:
        fit = fit_grid(mask, rows, columns, strict_threshold, blobs)
        return GridChoice(rows, columns, fit, grid_score(fit, rows, columns, len(cores)), "set")
    rows_seen, columns_seen = estimate_grid(cores)
    candidates = {(r, c) for r in (rows_seen - 1, rows_seen, rows_seen + 1) for c in (columns_seen - 1, columns_seen, columns_seen + 1)}
    candidates = sorted(s for s in candidates | {default} if s[0] in GRID_SIZES and s[1] in GRID_SIZES)
    window = tissue_window(cores, mask.shape)
    views = [(mask, blobs, 0, 0, {})]  # (mask, its blobs, offset x, offset y, axis fits)
    if window and window != (0, 0, mask.shape[1], mask.shape[0]):
        x0, y0, x1, y1 = window
        part = mask[y0:y1, x0:x1]
        views.append((part, tissue_blobs(part), x0, y0, {}))
    fits: dict[tuple[int, int], tuple[GridFit, float]] = {}
    for size in candidates:
        for number, (view, view_blobs, dx, dy, axes) in enumerate(views):
            if number == 0 and size != default and len(views) > 1:
                continue  # the whole slide is only tried for the project's size, as Find cores always did
            try:
                fit = _moved(fit_grid(view, *size, strict_threshold, view_blobs, axes), dx, dy)
            except ValueError:
                continue
            score = grid_score(fit, *size, len(cores))
            if size not in fits or score > fits[size][1] + (KEEP_DEFAULT_MARGIN if number else 0):
                fits[size] = (fit, score)
    tried = tuple((r, c, round(fits[r, c][1], 3) if (r, c) in fits else None) for r, c in candidates)
    if not fits:
        raise ValueError(f"No grid of 2-12 rows and columns fits this slide ({len(cores)} core-sized tissue pieces found); "
                         "check that the scan shows the whole TMA, or set the slide's rows and columns")
    best = max(fits, key=lambda size: (fits[size][1], size == default))
    if default in fits and fits[best][1] - fits[default][1] <= KEEP_DEFAULT_MARGIN:
        best = default
    return GridChoice(*best, *fits[best], "auto", tried)


ORIENTATIONS = {"": "as scanned", "rotate_180": "turned 180°", "flip_left_right": "mirrored left to right",
                "flip_top_bottom": "mirrored top to bottom"}


def oriented(row: int, column: int, rows: int, columns: int, orientation: str = "") -> tuple[int, int]:
    """The map position (row, column) of the fitted grid position, for a slide that lies turned or mirrored on the scan."""
    if orientation not in ORIENTATIONS:
        raise ValueError(f"Unknown slide orientation {orientation!r}: use one of {', '.join(o for o in ORIENTATIONS if o)}")
    flip_rows = orientation in ("rotate_180", "flip_top_bottom")
    flip_columns = orientation in ("rotate_180", "flip_left_right")
    return (rows - 1 - row if flip_rows else row), (columns - 1 - column if flip_columns else column)


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
    tma: str,
    stain: str,
    orientation: str = "",
) -> list[dict]:
    tissue, coverage_threshold = coverage_mask(preview.rgb)
    refinements = refine_core_centers(fit, rows, columns)
    crop_radius = CROP_RADIUS_SCALE * fit.radius
    records = []
    for fitted_row in range(rows):
        for fitted_column in range(columns):
            refinement = refinements[fitted_row, fitted_column]
            row, column = oriented(fitted_row, fitted_column, rows, columns, orientation)
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


def draw_qc(preview: SlidePreview, fit: GridFit, records: list[dict], note: str = "") -> np.ndarray:
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
        f"{preview.path.stem} | {note + ' | ' if note else ''}inliers={len(fit.inlier_components)} | "
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


PREVIEW_LONG_SIDE = 1600  # the plain slide picture kept for core thumbnails (qc/grids/previews)


def slide_slug(path: Path) -> str:
    return Path(path).stem.replace(" ", "_")


def dearray_preview(
    preview: SlidePreview,
    tma: str,
    stain: str,
    rows: int | None = None,
    columns: int | None = None,
    qc_dir: Path | None = None,
    orientation: str = "",
    default: tuple[int, int] = (5, 6),
) -> tuple[list[dict], GridChoice]:
    """Find the cores on one slide preview: the given grid size, or the best-fitting one when rows / columns are not set.
    With `qc_dir`, writes `<slide>.jpg` (the fitted grid drawn on the slide) and `previews/<slide>.jpg` (the plain
    slide, for core thumbnails)."""
    strict_mask, strict_threshold = segment_for_grid(preview.rgb)
    choice = choose_grid(strict_mask, strict_threshold, rows, columns, default)
    records = core_records(preview, choice.fit, choice.rows, choice.columns, tma, stain, orientation)
    if qc_dir is not None:
        slug = slide_slug(preview.path)
        qc_path = qc_dir / f"{slug}.jpg"
        qc_path.parent.mkdir(parents=True, exist_ok=True)
        how = "found automatically" if choice.source == "auto" else "as set"
        note = f"{choice.rows} x {choice.columns} grid ({how})" + (f", {ORIENTATIONS[orientation]}" if orientation else "")
        if not cv2.imwrite(str(qc_path), draw_qc(preview, choice.fit, records, note)):
            raise OSError(f"Could not write QC image: {qc_path}")
        scale = min(1.0, PREVIEW_LONG_SIDE / max(preview.rgb.shape[:2]))
        plain = cv2.resize(preview.rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else preview.rgb
        (qc_dir / "previews").mkdir(exist_ok=True)
        cv2.imwrite(str(qc_dir / "previews" / f"{slug}.jpg"), cv2.cvtColor(plain, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
        (qc_dir / "previews" / f"{slug}.json").write_text(json.dumps({"scale": scale}))  # picture pixels per preview pixel
    return records, choice


def process_slide(
    path: Path,
    tma: str,
    stain: str,
    rows: int | None,
    columns: int | None,
    target_long_side: int,
    qc_dir: Path | None = None,
    orientation: str = "",
    default: tuple[int, int] = (5, 6),
) -> tuple[list[dict], GridChoice]:
    preview = read_slide_preview(path, target_long_side)
    return dearray_preview(preview, tma, stain, rows, columns, qc_dir, orientation, default)
