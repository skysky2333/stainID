from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from stainid.imaging.color import rgb_to_hed
from stainid.qc.core_qc import tissue_mask
from stainid.slides.core_sheets import read_core_preview


@dataclass(frozen=True)
class RegistrationResult:
    matrix: np.ndarray
    method: str
    status: str
    initial_dice: float
    final_dice: float
    reference_overlap: float
    moving_overlap: float
    structural_correlation: float
    scale: float
    rotation_degrees: float


def read_preview(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_REDUCED_COLOR_8)
    if bgr is None:
        raise ValueError(f"Could not read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def read_registration_preview(row: dict[str, str]) -> np.ndarray:
    return read_core_preview(row, 4500)


def structural_image(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    structure = cv2.GaussianBlur(rgb_to_hed(rgb)[..., 0], (0, 0), 1.2)
    values = structure[mask]
    if values.size == 0:
        return np.zeros(mask.shape, dtype=np.uint8)
    lower, upper = np.quantile(values, [0.02, 0.98])
    if upper <= lower:
        return np.zeros(mask.shape, dtype=np.uint8)
    normalized = np.clip((structure - lower) / (upper - lower), 0.0, 1.0)
    return np.rint(normalized * 255).astype(np.uint8)


def dice_score(reference: np.ndarray, moving: np.ndarray) -> tuple[float, float, float]:
    intersection = int((reference & moving).sum())
    reference_area = int(reference.sum())
    moving_area = int(moving.sum())
    dice = 2 * intersection / max(reference_area + moving_area, 1)
    return (
        float(dice),
        float(intersection / max(reference_area, 1)),
        float(intersection / max(moving_area, 1)),
    )


def transform_characteristics(matrix: np.ndarray) -> tuple[float, float]:
    scale = float(np.sqrt(matrix[0, 0] ** 2 + matrix[1, 0] ** 2))
    rotation = float(np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0])))
    return scale, rotation


def plausible_transform(matrix: np.ndarray, shape: tuple[int, int]) -> bool:
    scale, rotation = transform_characteristics(matrix)
    height, width = shape
    translation = float(np.hypot(matrix[0, 2], matrix[1, 2]))
    determinant = float(np.linalg.det(matrix[:, :2]))
    return (
        determinant > 0
        and 0.85 <= scale <= 1.15
        and abs(rotation) <= 8.0
        and translation <= 0.25 * max(height, width)
    )


def registration_status(dice: float) -> str:
    if dice >= 0.75:
        return "pass"
    if dice >= 0.50:
        return "review"
    return "fail"


def masked_correlation(
    reference: np.ndarray, moving: np.ndarray, mask: np.ndarray
) -> float:
    if mask.sum() < 100:
        return 0.0
    reference_values = reference[mask].astype(np.float64)
    moving_values = moving[mask].astype(np.float64)
    reference_values -= reference_values.mean()
    moving_values -= moving_values.mean()
    denominator = np.linalg.norm(reference_values) * np.linalg.norm(moving_values)
    return float(reference_values @ moving_values / denominator) if denominator else 0.0


def mask_centroid(mask: np.ndarray) -> tuple[float, float]:
    moments = cv2.moments(mask.astype(np.uint8))
    if moments["m00"] == 0:
        return ((mask.shape[1] - 1) / 2, (mask.shape[0] - 1) / 2)
    return (moments["m10"] / moments["m00"], moments["m01"] / moments["m00"])


def rigid_candidate(
    angle: float,
    shift: tuple[float, float],
    shape: tuple[int, int],
) -> np.ndarray:
    height, width = shape
    matrix = cv2.getRotationMatrix2D(((width - 1) / 2, (height - 1) / 2), angle, 1.0)
    matrix[:, 2] += shift
    return matrix


def evaluate_matrix(
    matrix: np.ndarray,
    reference_mask: np.ndarray,
    moving_mask: np.ndarray,
    reference_structure: np.ndarray,
    moving_structure: np.ndarray,
) -> tuple[float, float, float, float, float]:
    height, width = reference_mask.shape
    aligned_mask = cv2.warpAffine(
        moving_mask.astype(np.uint8), matrix, (width, height), flags=cv2.INTER_NEAREST
    ).astype(bool)
    dice, reference_overlap, moving_overlap = dice_score(reference_mask, aligned_mask)
    aligned_structure = cv2.warpAffine(
        moving_structure, matrix, (width, height), flags=cv2.INTER_LINEAR
    )
    correlation = masked_correlation(
        reference_structure, aligned_structure, reference_mask & aligned_mask
    )
    scale, rotation = transform_characteristics(matrix)
    translation = float(np.hypot(matrix[0, 2], matrix[1, 2]))
    penalty = 0.001 * (abs(rotation) / 5.0 + translation / max(height, width))
    score = dice + 0.03 * max(correlation, 0.0) - penalty
    return score, dice, reference_overlap, moving_overlap, correlation


def register_previews(reference_rgb: np.ndarray, moving_rgb: np.ndarray) -> RegistrationResult:
    height, width = reference_rgb.shape[:2]
    moving_rgb = cv2.resize(moving_rgb, (width, height), interpolation=cv2.INTER_AREA)
    work_scale = min(1.0, 768 / max(height, width))
    work_size = (round(width * work_scale), round(height * work_scale))
    reference_work = cv2.resize(reference_rgb, work_size, interpolation=cv2.INTER_AREA)
    moving_work = cv2.resize(moving_rgb, work_size, interpolation=cv2.INTER_AREA)
    reference_mask, _, _ = tissue_mask(reference_work)
    moving_mask, _, _ = tissue_mask(moving_work)
    initial_dice, initial_reference_overlap, initial_moving_overlap = dice_score(
        reference_mask, moving_mask
    )
    reference_structure = structural_image(reference_work, reference_mask).astype(np.float32)
    moving_structure = structural_image(moving_work, moving_mask).astype(np.float32)
    identity = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    candidates = [identity]
    reference_center = mask_centroid(reference_mask)
    window = cv2.createHanningWindow(work_size, cv2.CV_32F)
    coarse_angles = (-4.0, -2.0, 0.0, 2.0, 4.0)
    angle_scores = []
    for angle in coarse_angles:
        rotation = rigid_candidate(angle, (0.0, 0.0), reference_mask.shape)
        rotated_mask = cv2.warpAffine(
            moving_mask.astype(np.uint8), rotation, work_size, flags=cv2.INTER_NEAREST
        ).astype(bool)
        rotated_structure = cv2.warpAffine(
            moving_structure, rotation, work_size, flags=cv2.INTER_LINEAR
        )
        moving_center = mask_centroid(rotated_mask)
        centroid_shift = (
            reference_center[0] - moving_center[0],
            reference_center[1] - moving_center[1],
        )
        phase_shift, _ = cv2.phaseCorrelate(
            rotated_structure.copy(), reference_structure.copy(), window
        )
        angle_candidates = []
        for shift in (centroid_shift, phase_shift):
            matrix = rotation.copy()
            matrix[:, 2] += shift
            if plausible_transform(matrix, reference_mask.shape):
                candidates.append(matrix)
                angle_candidates.append(
                    evaluate_matrix(
                        matrix,
                        reference_mask,
                        moving_mask,
                        reference_structure,
                        moving_structure,
                    )[0]
                )
        angle_scores.append((max(angle_candidates, default=-np.inf), angle))

    best_coarse_angle = max(angle_scores)[1]
    for angle in np.arange(best_coarse_angle - 1.5, best_coarse_angle + 1.51, 0.5):
        if angle in coarse_angles or abs(angle) > 5.0:
            continue
        rotation = rigid_candidate(float(angle), (0.0, 0.0), reference_mask.shape)
        rotated_mask = cv2.warpAffine(
            moving_mask.astype(np.uint8), rotation, work_size, flags=cv2.INTER_NEAREST
        ).astype(bool)
        rotated_structure = cv2.warpAffine(
            moving_structure, rotation, work_size, flags=cv2.INTER_LINEAR
        )
        moving_center = mask_centroid(rotated_mask)
        centroid_shift = (
            reference_center[0] - moving_center[0],
            reference_center[1] - moving_center[1],
        )
        phase_shift, _ = cv2.phaseCorrelate(
            rotated_structure.copy(), reference_structure.copy(), window
        )
        for shift in (centroid_shift, phase_shift):
            matrix = rotation.copy()
            matrix[:, 2] += shift
            if plausible_transform(matrix, reference_mask.shape):
                candidates.append(matrix)

    evaluated = [
        (
            evaluate_matrix(
                matrix,
                reference_mask,
                moving_mask,
                reference_structure,
                moving_structure,
            ),
            matrix,
        )
        for matrix in candidates
    ]
    (score, final_dice, reference_overlap, moving_overlap, correlation), fitted = max(
        evaluated, key=lambda item: item[0][0]
    )
    scale, rotation = transform_characteristics(fitted)

    if final_dice + 0.02 < initial_dice:
        return RegistrationResult(
            identity,
            "identity_better_overlap",
            registration_status(initial_dice),
            initial_dice,
            initial_dice,
            initial_reference_overlap,
            initial_moving_overlap,
            float(correlation),
            1.0,
            0.0,
        )

    work_scale_matrix = np.diag(
        [work_size[0] / width, work_size[1] / height, 1.0]
    )
    preview_matrix = (
        np.linalg.inv(work_scale_matrix)
        @ np.vstack([fitted, [0.0, 0.0, 1.0]])
        @ work_scale_matrix
    )[:2]

    return RegistrationResult(
        preview_matrix,
        "mask_structure_rigid" if not np.array_equal(fitted, identity) else "identity",
        registration_status(final_dice),
        initial_dice,
        final_dice,
        reference_overlap,
        moving_overlap,
        float(correlation),
        scale,
        rotation,
    )


def native_matrix(
    preview_matrix: np.ndarray,
    reference_preview_shape: tuple[int, int],
    reference_native_shape: tuple[int, int],
    moving_native_shape: tuple[int, int],
) -> np.ndarray:
    preview_height, preview_width = reference_preview_shape
    reference_height, reference_width = reference_native_shape
    moving_height, moving_width = moving_native_shape
    reference_inverse = np.diag(
        [reference_width / preview_width, reference_height / preview_height, 1.0]
    )
    moving_scale = np.diag(
        [preview_width / moving_width, preview_height / moving_height, 1.0]
    )
    homogeneous = np.vstack([preview_matrix, [0.0, 0.0, 1.0]])
    return (reference_inverse @ homogeneous @ moving_scale)[:2]


def qc_image(
    reference_rgb: np.ndarray, moving_rgb: np.ndarray, result: RegistrationResult
) -> np.ndarray:
    height, width = reference_rgb.shape[:2]
    moving_rgb = cv2.resize(moving_rgb, (width, height), interpolation=cv2.INTER_AREA)
    aligned = cv2.warpAffine(
        moving_rgb,
        result.matrix,
        (width, height),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    overlay = np.rint(
        0.5 * reference_rgb.astype(np.float32) + 0.5 * aligned.astype(np.float32)
    ).astype(np.uint8)
    panel = np.concatenate([reference_rgb, moving_rgb, aligned, overlay], axis=1)
    panel = cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)
    label = (
        f"{result.method} | Dice {result.initial_dice:.3f} -> "
        f"{result.final_dice:.3f} | {result.status}"
    )
    cv2.rectangle(panel, (0, 0), (min(panel.shape[1], 1800), 48), (255, 255, 255), -1)
    cv2.putText(
        panel,
        label,
        (12, 33),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )
    return panel


def format_number(value: float) -> str:
    return f"{value:.8f}" if np.isfinite(value) else ""


def register_triplet_manifest(
    manifest_path: Path,
    output_path: Path,
    identifier_column: str = "annotation_id",
    qc_dir: Path | None = None,
    selected_identifiers: set[str] | None = None,
) -> Path:
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    grouped: dict[str, dict[str, dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row[identifier_column], {})[row["stain"]] = row
    if any(set(stains) != {"6E10", "AT8", "NeuN"} for stains in grouped.values()):
        raise ValueError("Every identifier must contain 6E10, AT8, and NeuN")
    if selected_identifiers:
        unknown = sorted(selected_identifiers - set(grouped))
        if unknown:
            raise ValueError(f"Unknown registration identifiers: {unknown}")
        grouped = {
            identifier: grouped[identifier]
            for identifier in sorted(selected_identifiers)
        }

    fields = [
        identifier_column,
        "moving_stain",
        "reference_stain",
        "moving_path",
        "reference_path",
        "method",
        "status",
        "qc_scope",
        "spatial_status",
        "initial_tissue_dice",
        "registered_tissue_dice",
        "reference_tissue_overlap",
        "moving_tissue_overlap",
        "structural_correlation",
        "preview_scale",
        "preview_rotation_degrees",
        "native_m00",
        "native_m01",
        "native_m02",
        "native_m10",
        "native_m11",
        "native_m12",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".tmp.csv")
    checkpoint_records = []
    if selected_identifiers and output_path.exists():
        with output_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != fields:
                raise ValueError(f"Registration columns disagree: {output_path}")
            checkpoint_records = [
                record
                for record in reader
                if record[identifier_column] not in selected_identifiers
            ]
    elif temporary.exists():
        with temporary.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != fields:
                raise ValueError(f"Checkpoint columns disagree: {temporary}")
            checkpoint_records = list(reader)
    checkpoint_groups: dict[str, set[str]] = {}
    for record in checkpoint_records:
        checkpoint_groups.setdefault(record[identifier_column], set()).add(
            record["moving_stain"]
        )
    completed = {
        identifier
        for identifier, stains in checkpoint_groups.items()
        if stains == {"6E10", "AT8"}
    }
    checkpoint_records = [
        record
        for record in checkpoint_records
        if record[identifier_column] in completed
    ]
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(checkpoint_records)

    for identifier in sorted(grouped):
        if identifier in completed:
            print(f"{identifier}: resumed", flush=True)
            continue
        stain_rows = grouped[identifier]
        reference_path = Path(stain_rows["NeuN"]["image_path"])
        reference_rgb = read_registration_preview(stain_rows["NeuN"])
        reference_native_shape = (
            int(stain_rows["NeuN"]["native_height_px"]),
            int(stain_rows["NeuN"]["native_width_px"]),
        )
        records = []
        for stain in ("6E10", "AT8"):
            moving_path = Path(stain_rows[stain]["image_path"])
            moving_rgb = read_registration_preview(stain_rows[stain])
            moving_native_shape = (
                int(stain_rows[stain]["native_height_px"]),
                int(stain_rows[stain]["native_width_px"]),
            )
            result = register_previews(reference_rgb, moving_rgb)
            matrix = native_matrix(
                result.matrix,
                reference_rgb.shape[:2],
                reference_native_shape,
                moving_native_shape,
            )
            records.append(
                {
                    identifier_column: identifier,
                    "moving_stain": stain,
                    "reference_stain": "NeuN",
                    "moving_path": str(moving_path),
                    "reference_path": str(reference_path),
                    "method": result.method,
                    "status": result.status,
                    "qc_scope": "coarse_tissue_overlap",
                    "spatial_status": "not_landmark_validated",
                    "initial_tissue_dice": f"{result.initial_dice:.6f}",
                    "registered_tissue_dice": f"{result.final_dice:.6f}",
                    "reference_tissue_overlap": f"{result.reference_overlap:.6f}",
                    "moving_tissue_overlap": f"{result.moving_overlap:.6f}",
                    "structural_correlation": format_number(
                        result.structural_correlation
                    ),
                    "preview_scale": f"{result.scale:.8f}",
                    "preview_rotation_degrees": f"{result.rotation_degrees:.8f}",
                    "native_m00": format_number(matrix[0, 0]),
                    "native_m01": format_number(matrix[0, 1]),
                    "native_m02": format_number(matrix[0, 2]),
                    "native_m10": format_number(matrix[1, 0]),
                    "native_m11": format_number(matrix[1, 1]),
                    "native_m12": format_number(matrix[1, 2]),
                }
            )
            if qc_dir is not None:
                qc_dir.mkdir(parents=True, exist_ok=True)
                qc_path = qc_dir / f"{identifier}_{stain}_to_NeuN.jpg"
                if not cv2.imwrite(str(qc_path), qc_image(reference_rgb, moving_rgb, result)):
                    raise OSError(f"Could not write registration QC image: {qc_path}")
        with temporary.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writerows(records)
        print(f"{identifier}: registered 6E10 and AT8 to NeuN", flush=True)

    temporary.replace(output_path)
    return output_path


__all__ = [
    "RegistrationResult",
    "native_matrix",
    "read_registration_preview",
    "register_triplet_manifest",
    "register_previews",
]
