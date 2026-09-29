from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from skimage.color import rgb2hed, rgb2hsv

MORPHOLOGY_FEATURES = (
    "deposit_area_um2",
    "equivalent_diameter_um",
    "stained_area_um2",
    "stain_fill_fraction",
    "seed_area_um2",
    "core_area_um2",
    "core_fraction",
    "normalized_core_offset",
    "mean_dab_od",
    "median_dab_od",
    "p90_dab_od",
    "max_dab_od",
    "inner_mean_dab_od",
    "outer_mean_dab_od",
    "radial_dab_contrast",
    "perimeter_um",
    "circularity",
    "solidity",
    "aspect_ratio",
    "eccentricity",
    "major_axis_um",
    "minor_axis_um",
    "boundary_irregularity",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_review_records(review_dirs: list[Path]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    candidate_ids: set[str] = set()
    for review_dir in review_dirs:
        keys = {row["review_id"]: row for row in read_csv(review_dir / "selection_key.csv")}
        labels = read_csv(review_dir / "review_labels.csv")
        if set(keys) != {row["review_id"] for row in labels}:
            raise ValueError(f"Review IDs do not match in {review_dir}")
        for label in labels:
            visual_label = label["visual_label"]
            if visual_label == "uncertain":
                continue
            if visual_label not in {"plaque", "non_plaque"}:
                raise ValueError(f"Unresolved visual label in {review_dir}: {visual_label}")
            row = keys[label["review_id"]]
            candidate_id = row["candidate_id"]
            if candidate_id in candidate_ids:
                raise ValueError(f"Candidate reviewed more than once: {candidate_id}")
            candidate_ids.add(candidate_id)
            records.append(
                {
                    **row,
                    "visual_label": visual_label,
                    "target": int(visual_label == "plaque"),
                    "review_round": review_dir.name,
                    "patch_path": str(review_dir / "patches" / f"{label['review_id']}.jpg"),
                }
            )
    return records


def raw_review_patch(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read review patch: {path}")
    panel = image.shape[1] // 2
    raw = image[-panel:, :panel]
    return cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)


def _region_values(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    selected = values[mask]
    if selected.size == 0:
        raise ValueError("Empty context-feature region")
    return selected


def context_features(
    rgb: np.ndarray,
    equivalent_diameter_um: float,
    context_um: float = 140.0,
) -> dict[str, float]:
    height, width = rgb.shape[:2]
    yy, xx = np.ogrid[:height, :width]
    distance = np.sqrt((xx - (width - 1) / 2) ** 2 + (yy - (height - 1) / 2) ** 2)
    radius = np.clip(
        equivalent_diameter_um * min(height, width) / context_um / 2,
        3,
        min(height, width) / 4,
    )
    center = distance <= radius
    annulus = (distance > radius) & (distance <= 3 * radius)
    hed = rgb2hed(rgb)
    hematoxylin = hed[:, :, 0]
    dab = hed[:, :, 2]
    hsv = rgb2hsv(rgb)
    center_h = _region_values(hematoxylin, center)
    annulus_h = _region_values(hematoxylin, annulus)
    center_dab = _region_values(dab, center)
    annulus_dab = _region_values(dab, annulus)
    context_h90 = float(np.quantile(hematoxylin, 0.90))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return {
        "context_dab_mean": float(np.mean(dab)),
        "context_dab_p90": float(np.quantile(dab, 0.90)),
        "context_dab_std": float(np.std(dab)),
        "center_dab_mean": float(np.mean(center_dab)),
        "center_dab_p90": float(np.quantile(center_dab, 0.90)),
        "annulus_dab_mean": float(np.mean(annulus_dab)),
        "center_annulus_dab_contrast": float(np.mean(center_dab) - np.mean(annulus_dab)),
        "center_hematoxylin_mean": float(np.mean(center_h)),
        "center_hematoxylin_p90": float(np.quantile(center_h, 0.90)),
        "annulus_hematoxylin_mean": float(np.mean(annulus_h)),
        "center_high_hematoxylin_fraction": float(np.mean(center_h >= context_h90)),
        "context_saturation_mean": float(np.mean(hsv[:, :, 1])),
        "context_laplacian_variance": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
    }


def build_tabular_matrix(
    records: list[dict[str, object]],
    include_context: bool,
) -> tuple[np.ndarray, list[str]]:
    names = list(MORPHOLOGY_FEATURES)
    context_by_path: dict[str, dict[str, float]] = {}
    if include_context:
        for row in records:
            patch_path = str(row["patch_path"])
            context_by_path[patch_path] = context_features(
                raw_review_patch(Path(patch_path)),
                float(row["equivalent_diameter_um"]),
            )
        names += list(next(iter(context_by_path.values())))
    matrix = []
    for row in records:
        values = [float(row[name]) for name in MORPHOLOGY_FEATURES]
        if include_context:
            values += [context_by_path[str(row["patch_path"])][name] for name in names[len(MORPHOLOGY_FEATURES) :]]
        matrix.append(values)
    return np.asarray(matrix, dtype=float), names


def efficientnet_embeddings(
    records: list[dict[str, object]],
    batch_size: int = 32,
) -> np.ndarray:
    import torch
    from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

    weights = EfficientNet_B0_Weights.DEFAULT
    transform = weights.transforms()
    model = efficientnet_b0(weights=weights)
    model.classifier = torch.nn.Identity()
    model.eval()
    batches = []
    with torch.inference_mode():
        for start in range(0, len(records), batch_size):
            tensors = [
                transform(Image.fromarray(raw_review_patch(Path(row["patch_path"]))))
                for row in records[start : start + batch_size]
            ]
            batches.append(model(torch.stack(tensors)).cpu().numpy())
    return np.concatenate(batches)


__all__ = [
    "MORPHOLOGY_FEATURES",
    "build_tabular_matrix",
    "context_features",
    "efficientnet_embeddings",
    "load_review_records",
    "raw_review_patch",
]
