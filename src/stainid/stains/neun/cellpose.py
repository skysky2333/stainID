from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

from stainid.imaging.color import rgb_to_hed
from stainid.imaging.tissue import field_tissue_mask, linear_artifact_mask
from stainid.stains.neun.candidates import write_candidate_overlay


def cellpose_input(rgb: np.ndarray, mode: str) -> np.ndarray:
    if mode == "rgb":
        return rgb
    if mode != "hdab":
        raise ValueError(f"Unsupported Cellpose input mode: {mode}")
    hed = rgb_to_hed(rgb)
    hematoxylin = hed[..., 0].astype(np.float32)
    dab = hed[..., 2].astype(np.float32)
    return np.stack((hematoxylin, dab, np.maximum(hematoxylin, dab)), axis=-1)


def largest_component(mask: np.ndarray) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), 8
    )
    if count <= 1:
        return np.zeros(mask.shape, dtype=bool)
    label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == label


def classify_cellpose_masks(
    raw_masks: np.ndarray,
    rgb: np.ndarray,
    dab_threshold: float,
    pixel_width_um: float,
    pixel_height_um: float,
) -> tuple[
    dict[str, float | int],
    list[dict[str, float | int | bool | str]],
    np.ndarray,
]:
    initial_tissue = field_tissue_mask(rgb)
    artifact = linear_artifact_mask(rgb)
    valid_tissue = initial_tissue & ~artifact
    if valid_tissue.sum() < 100:
        raise ValueError("Cellpose classification requires at least 100 tissue pixels")
    hed = rgb_to_hed(rgb)
    hematoxylin = np.maximum(hed[..., 0], 0.0).astype(np.float32)
    dab = np.maximum(hed[..., 2], 0.0).astype(np.float32)
    pixel_area_um2 = pixel_width_um * pixel_height_um
    image_height, image_width = raw_masks.shape
    labels = np.zeros(raw_masks.shape, dtype=np.int32)
    objects: list[dict[str, float | int | bool | str]] = []
    rejected_area = 0
    rejected_tissue = 0
    for raw_label, bounds in enumerate(ndimage.find_objects(raw_masks), start=1):
        if bounds is None:
            continue
        y_slice, x_slice = bounds
        raw_local = raw_masks[y_slice, x_slice] == raw_label
        tissue_fraction = float(valid_tissue[y_slice, x_slice][raw_local].mean())
        if tissue_fraction < 0.5:
            rejected_tissue += 1
            continue
        local = largest_component(raw_local & valid_tissue[y_slice, x_slice])
        area_um2 = int(local.sum()) * pixel_area_um2
        if not 8.0 <= area_um2 <= 650.0:
            rejected_area += 1
            continue
        label = len(objects) + 1
        labels[y_slice, x_slice][local] = label
        y0 = max(0, y_slice.start - 5)
        y1 = min(image_height, y_slice.stop + 5)
        x0 = max(0, x_slice.start - 5)
        x1 = min(image_width, x_slice.stop + 5)
        local_label = labels[y0:y1, x0:x1] == label
        neighborhood = cv2.dilate(
            local_label.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
        ).astype(bool)
        neighborhood &= valid_tissue[y0:y1, x0:x1]
        values = dab[y_slice, x_slice][local]
        neighborhood_values = dab[y0:y1, x0:x1][neighborhood]
        ys, xs = np.nonzero(local)
        mean_dab = float(values.mean())
        p80_dab = float(np.quantile(neighborhood_values, 0.8))
        positive = mean_dab >= 0.5 * dab_threshold or p80_dab >= dab_threshold
        review = not positive and (
            mean_dab >= 0.4 * dab_threshold or p80_dab >= 0.75 * dab_threshold
        )
        objects.append(
            {
                "label": label,
                "source_label": raw_label,
                "centroid_x_px": float(x_slice.start + xs.mean()),
                "centroid_y_px": float(y_slice.start + ys.mean()),
                "area_um2": area_um2,
                "tissue_fraction": tissue_fraction,
                "mean_hematoxylin_od": float(
                    hematoxylin[y_slice, x_slice][local].mean()
                ),
                "mean_dab_od": mean_dab,
                "neighborhood_dab_p80": p80_dab,
                "candidate_class": (
                    "positive" if positive else "review" if review else "negative"
                ),
                "neun_positive": bool(positive),
                "touches_edge": bool(
                    x_slice.start == 0
                    or y_slice.start == 0
                    or x_slice.stop == image_width
                    or y_slice.stop == image_height
                ),
            }
        )
    complete = [row for row in objects if not row["touches_edge"]]
    positive = [row for row in complete if row["neun_positive"]]
    review = [row for row in complete if row["candidate_class"] == "review"]
    tissue_area_mm2 = valid_tissue.sum() * pixel_area_um2 / 1_000_000.0
    summary = {
        "raw_mask_count": int(raw_masks.max()),
        "candidate_profile_count": len(complete),
        "neun_positive_profile_count": len(positive),
        "neun_review_profile_count": len(review),
        "neun_positive_profile_density_mm2": (
            len(positive) / tissue_area_mm2 if tissue_area_mm2 else float("nan")
        ),
        "edge_profile_count": len(objects) - len(complete),
        "rejected_area_count": rejected_area,
        "rejected_tissue_count": rejected_tissue,
        "tissue_area_mm2": tissue_area_mm2,
        "linear_artifact_area_fraction": float(
            artifact.sum() / initial_tissue.sum() if initial_tissue.sum() else 0.0
        ),
    }
    return summary, objects, labels


def selected_instance_labels(
    labels: np.ndarray,
    objects: list[dict[str, float | int | bool | str]],
    positive_only: bool = False,
) -> np.ndarray:
    selected = [
        row
        for row in objects
        if not bool(row["touches_edge"])
        and (not positive_only or bool(row["neun_positive"]))
    ]
    result = np.zeros(labels.shape, dtype=np.int32)
    for output_label, row in enumerate(selected, start=1):
        result[labels == int(row["label"])] = output_label
    return result


def instance_agreement(
    reference: np.ndarray,
    comparison: np.ndarray,
    minimum_iou: float = 0.3,
) -> dict[str, float | int]:
    reference_count = int(reference.max())
    comparison_count = int(comparison.max())
    if reference_count == 0 or comparison_count == 0:
        return {
            "reference_count": reference_count,
            "comparison_count": comparison_count,
            "matched_count": 0,
            "reference_match_fraction": 0.0,
            "comparison_match_fraction": 0.0,
            "median_matched_iou": float("nan"),
        }
    joint = np.bincount(
        (reference.ravel() * (comparison_count + 1) + comparison.ravel()),
        minlength=(reference_count + 1) * (comparison_count + 1),
    ).reshape(reference_count + 1, comparison_count + 1)
    intersections = joint[1:, 1:].astype(np.float64)
    reference_areas = np.bincount(
        reference.ravel(), minlength=reference_count + 1
    )[1:, None]
    comparison_areas = np.bincount(
        comparison.ravel(), minlength=comparison_count + 1
    )[None, 1:]
    unions = reference_areas + comparison_areas - intersections
    iou = np.divide(
        intersections,
        unions,
        out=np.zeros_like(intersections),
        where=unions > 0,
    )
    reference_indices, comparison_indices = linear_sum_assignment(-iou)
    assigned = iou[reference_indices, comparison_indices]
    matched = assigned >= minimum_iou
    matched_ious = assigned[matched]
    matched_count = int(matched.sum())
    return {
        "reference_count": reference_count,
        "comparison_count": comparison_count,
        "matched_count": matched_count,
        "reference_match_fraction": matched_count / reference_count,
        "comparison_match_fraction": matched_count / comparison_count,
        "median_matched_iou": (
            float(np.median(matched_ious)) if matched_count else float("nan")
        ),
    }


def benchmark_agreement(
    baseline_labels: np.ndarray,
    baseline_objects: list[dict[str, float | int | bool | str]],
    cellpose_labels: np.ndarray,
    cellpose_objects: list[dict[str, float | int | bool | str]],
    minimum_iou: float = 0.3,
) -> dict[str, float | int]:
    result: dict[str, float | int] = {}
    for name, positive_only in (("all", False), ("positive", True)):
        metrics = instance_agreement(
            selected_instance_labels(
                baseline_labels, baseline_objects, positive_only=positive_only
            ),
            selected_instance_labels(
                cellpose_labels, cellpose_objects, positive_only=positive_only
            ),
            minimum_iou,
        )
        result.update({f"{name}_{key}": value for key, value in metrics.items()})
    return result


def write_comparison_panel(
    rgb: np.ndarray,
    baseline_labels: np.ndarray,
    baseline_objects: list[dict[str, float | int | bool | str]],
    cellpose_labels: np.ndarray,
    cellpose_objects: list[dict[str, float | int | bool | str]],
    output_path: Path,
    mode: str,
) -> Path:
    temporary_baseline = output_path.with_name(f".{output_path.stem}_baseline.jpg")
    temporary_cellpose = output_path.with_name(f".{output_path.stem}_cellpose.jpg")
    write_candidate_overlay(rgb, baseline_labels, baseline_objects, temporary_baseline)
    write_candidate_overlay(rgb, cellpose_labels, cellpose_objects, temporary_cellpose)
    baseline = cv2.cvtColor(cv2.imread(str(temporary_baseline)), cv2.COLOR_BGR2RGB)
    cellpose = cv2.cvtColor(cv2.imread(str(temporary_cellpose)), cv2.COLOR_BGR2RGB)
    temporary_baseline.unlink()
    temporary_cellpose.unlink()
    scale = min(1.0, 900.0 / max(rgb.shape[:2]))
    size = (round(rgb.shape[1] * scale), round(rgb.shape[0] * scale))
    panels = [
        cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        for image in (rgb, baseline, cellpose)
    ]
    for panel, title in zip(
        panels, ("Original", "Watershed baseline", f"Cellpose-SAM {mode}")
    ):
        cv2.rectangle(panel, (0, 0), (size[0], 42), (255, 255, 255), -1)
        cv2.putText(
            panel,
            title,
            (12, 29),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (25, 25, 25),
            2,
            cv2.LINE_AA,
        )
    comparison = np.concatenate(panels, axis=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(
        str(output_path),
        cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_JPEG_QUALITY, 95],
    ):
        raise OSError(f"Could not write Cellpose comparison: {output_path}")
    return output_path


__all__ = [
    "benchmark_agreement",
    "cellpose_input",
    "classify_cellpose_masks",
    "instance_agreement",
    "selected_instance_labels",
    "write_comparison_panel",
]
