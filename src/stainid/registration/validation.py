from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import cv2

from stainid.registration.aligned_fields import matrix_from_row, preview_matrix
from stainid.registration.rigid import RegistrationResult, qc_image
from stainid.slides.core_sheets import read_core_preview


def select_validation_cores(
    registrations: list[dict[str, str]],
    annotation_keys: list[dict[str, str]],
) -> dict[str, str]:
    registration_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in registrations:
        registration_groups[row["core_id"]].append(row)
    selected = {
        core_id: "coarse_qc_flag"
        for core_id, rows in registration_groups.items()
        if any(row["status"] != "pass" for row in rows)
    }
    if "LIP-5_C-1" in registration_groups:
        selected.setdefault("LIP-5_C-1", "development_pilot")
    for tma in sorted({int(row["tma"]) for row in annotation_keys}):
        desired = (
            (("ASYMP", "frontal"), ("AD", "occipital"))
            if tma % 2
            else (("ASYMP", "occipital"), ("AD", "frontal"))
        )
        tma_rows = [row for row in annotation_keys if int(row["tma"]) == tma]
        for disease_group, region in desired:
            matches = [
                row
                for row in tma_rows
                if row["disease_group"] == disease_group and row["region"] == region
            ]
            if not matches:
                continue
            row = sorted(matches, key=lambda item: item["annotation_id"])[0]
            core_id = f"LIP-{tma}_{row['core_label']}"
            if all(
                result["status"] == "pass"
                for result in registration_groups.get(core_id, [])
            ):
                selected.setdefault(core_id, "balanced_landmark_sample")
    return selected


def build_validation_rows(
    images: list[dict[str, str]],
    registrations: list[dict[str, str]],
    annotation_keys: list[dict[str, str]],
) -> list[dict[str, str]]:
    selected = select_validation_cores(registrations, annotation_keys)
    references = {
        row["core_id"]: row for row in images if row["stain"] == "NeuN"
    }
    rows = []
    for result in registrations:
        core_id = result["core_id"]
        if core_id not in selected:
            continue
        reference = references[core_id]
        rows.append(
            {
                "core_id": core_id,
                "tma": reference["tma"],
                "core_label": reference["core_label"],
                "donor_id": reference["donor_id"],
                "region": reference["region"],
                "disease_group": reference["disease_group"],
                "moving_stain": result["moving_stain"],
                "review_source": selected[core_id],
                "coarse_status": result["status"],
                "registered_tissue_dice": result["registered_tissue_dice"],
                "structural_correlation": result["structural_correlation"],
                "landmark_status": "not_reviewed",
                "landmark_count": "",
                "median_target_error_um": "",
                "spatial_eligible": "false",
            }
        )
    return sorted(rows, key=lambda row: (int(row["tma"]), row["core_label"], row["moving_stain"]))


def write_rows(path: Path, rows: list[dict[str, str]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.csv")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)
    return path


def render_validation_qc(
    images: list[dict[str, str]],
    registrations: list[dict[str, str]],
    validation_rows: list[dict[str, str]],
    output_dir: Path,
) -> list[Path]:
    image_map = {(row["core_id"], row["stain"]): row for row in images}
    registration_map = {
        (row["core_id"], row["moving_stain"]): row for row in registrations
    }
    outputs = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for validation in validation_rows:
        core_id = validation["core_id"]
        stain = validation["moving_stain"]
        reference_row = image_map[(core_id, "NeuN")]
        moving_row = image_map[(core_id, stain)]
        result_row = registration_map[(core_id, stain)]
        reference = read_core_preview(reference_row, 4800)
        moving = read_core_preview(moving_row, 4800)
        matrix = preview_matrix(
            matrix_from_row(result_row),
            reference.shape[:2],
            (
                int(reference_row["native_width_px"]),
                int(reference_row["native_height_px"]),
            ),
            moving.shape[:2],
            (
                int(moving_row["native_width_px"]),
                int(moving_row["native_height_px"]),
            ),
        )
        result = RegistrationResult(
            matrix=matrix,
            method=result_row["method"],
            status=result_row["status"],
            initial_dice=float(result_row["initial_tissue_dice"]),
            final_dice=float(result_row["registered_tissue_dice"]),
            reference_overlap=float(result_row["reference_tissue_overlap"]),
            moving_overlap=float(result_row["moving_tissue_overlap"]),
            structural_correlation=float(result_row["structural_correlation"] or 0.0),
            scale=float(result_row["preview_scale"]),
            rotation_degrees=float(result_row["preview_rotation_degrees"]),
        )
        output = output_dir / f"{core_id}_{stain}_to_NeuN.jpg"
        if not cv2.imwrite(str(output), qc_image(reference, moving, result)):
            raise OSError(f"Could not write registration validation image: {output}")
        outputs.append(output)
    expected = set(outputs)
    for path in output_dir.glob("*.jpg"):
        if path not in expected:
            path.unlink()
    return outputs


__all__ = [
    "build_validation_rows",
    "render_validation_qc",
    "select_validation_cores",
    "write_rows",
]
