from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from stainid.stains.amyloid.classifier import MORPHOLOGY_FEATURES, build_tabular_matrix, context_features, load_review_records
from stainid.stains.amyloid.published_cnn import cnn_patch, consensus_probabilities
from stainid.stains.neun.audit import crop_with_padding

IDENTITY_CLASSES = {"compact", "diffuse", "review"}
PATCH_CONTEXT_UM = 140.0
PATCH_PX = 180
MORPHOTYPE_MINIMUM_DIAMETER_UM = 15.0


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def identity_model():
    return make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        RandomForestClassifier(
            n_estimators=500,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=20260924,
            n_jobs=-1,
        ),
    )


PHIKON_MODEL = "owkin/phikon"
PHIKON_CROP_UM = 122.7
_PHIKON = {}


def phikon_embeddings(images: list[np.ndarray]) -> np.ndarray:
    import torch
    from transformers import AutoImageProcessor, AutoModel

    if not _PHIKON:
        _PHIKON["processor"] = AutoImageProcessor.from_pretrained(PHIKON_MODEL)
        _PHIKON["model"] = AutoModel.from_pretrained(PHIKON_MODEL).eval()
    features = []
    with torch.inference_mode():
        for start in range(0, len(images), 32):
            batch = _PHIKON["processor"](images=images[start : start + 32], return_tensors="pt")
            features.append(_PHIKON["model"](**batch).last_hidden_state[:, 0].numpy())
    return np.concatenate(features)


def phikon_patch(rgb: np.ndarray, x: float, y: float, pixel_size_um: float) -> np.ndarray:
    crop = crop_with_padding(rgb, x, y, round(PHIKON_CROP_UM / pixel_size_um))
    return cv2.resize(crop, (224, 224), interpolation=cv2.INTER_AREA)


def review_patch(bgr: np.ndarray, x: float, y: float, pixel_size_um: float) -> np.ndarray:
    crop = crop_with_padding(bgr, x, y, max(64, round(PATCH_CONTEXT_UM / pixel_size_um)))
    patch = cv2.resize(crop, (PATCH_PX, PATCH_PX), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", patch, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise OSError("Could not encode amyloid review patch")
    return cv2.cvtColor(cv2.imdecode(encoded, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def morphotype_threshold(values: np.ndarray, compact: np.ndarray) -> float:
    candidates = np.unique(values)
    scores = [
        (values[compact] >= value).mean() + (values[~compact] < value).mean()
        for value in candidates
    ]
    return float(candidates[int(np.argmax(scores))])


def train_amyloid_bundle(
    review_dirs: list[Path],
    morphotype_dir: Path,
    calibration_path: Path,
) -> dict[str, object]:
    records = load_review_records(review_dirs)
    matrix, feature_names = build_tabular_matrix(records, True)
    classifier = identity_model()
    targets = np.asarray([int(row["target"]) for row in records])
    classifier.fit(matrix, targets)
    patches: list[np.ndarray | None] = [None] * len(records)
    loaded, image = None, None
    for index in sorted(range(len(records)), key=lambda i: str(records[i]["raw_path"])):
        row = records[index]
        if row["raw_path"] != loaded:
            loaded = row["raw_path"]
            image = cv2.cvtColor(cv2.imread(str(loaded)), cv2.COLOR_BGR2RGB)
        pixel_size = float(np.sqrt(float(row["pixel_width_um"]) * float(row["pixel_height_um"])))
        patches[index] = phikon_patch(image, float(row["raw_x_px"]), float(row["raw_y_px"]), pixel_size)
    phikon = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=5000, class_weight="balanced"))
    phikon.fit(phikon_embeddings(patches), targets)
    calibration = {
        row["tma"]: float(row["threshold_dab_od"])
        for row in read_csv(calibration_path)
        if row["stain"] == "6E10"
    }
    by_candidate = {str(row["candidate_id"]): row for row in records}
    key = {row["review_id"]: row for row in read_csv(morphotype_dir / "selection_key.csv")}
    resolved = [
        (by_candidate[key[row["review_id"]]["candidate_id"]], row["morphotype_label"])
        for row in read_csv(morphotype_dir / "review_labels.csv")
        if row["morphotype_label"] in {"compact_or_cored", "diffuse"}
    ]
    normalized = np.asarray(
        [float(row["inner_mean_dab_od"]) / calibration[str(row["tma"])] for row, _ in resolved]
    )
    compact = np.asarray([label == "compact_or_cored" for _, label in resolved])
    return {
        "identity_classifier": classifier,
        "phikon_classifier": phikon,
        "identity_rule": "mean of morphology-context random forest, Phikon linear probe, and Wong 2022 consensus CNN plaque score",
        "use_wong_consensus": True,
        "identity_feature_names": feature_names,
        "identity_threshold": 0.50,
        "identity_training_candidate_ids": list(by_candidate),
        "morphotype_feature": "inner_mean_dab_od / slide 6E10 calibration threshold",
        "morphotype_threshold": morphotype_threshold(normalized, compact),
        "morphotype_minimum_diameter_um": MORPHOTYPE_MINIMUM_DIAMETER_UM,
        "morphotype_training_n": int(len(resolved)),
    }


def classify_amyloid_objects(
    rgb: np.ndarray,
    objects: list[dict[str, object]],
    dab_threshold: float,
    pixel_size_um: float,
    bundle: dict[str, object],
) -> list[dict[str, object]]:
    bgr = rgb[:, :, ::-1]
    eligible = [row for row in objects if row["candidate_class"] in IDENTITY_CLASSES]
    caa_scores: dict[int, float] = {}
    probabilities: dict[int, float] = {}
    if eligible:
        contexts = [
            context_features(
                review_patch(
                    bgr,
                    float(row["centroid_x_px"]),
                    float(row["centroid_y_px"]),
                    pixel_size_um,
                ),
                float(row["equivalent_diameter_um"]),
            )
            for row in eligible
        ]
        names = list(MORPHOLOGY_FEATURES) + list(contexts[0])
        if names != bundle["identity_feature_names"]:
            raise ValueError("Amyloid cohort features do not match the frozen model")
        matrix = np.asarray(
            [
                [float(row[name]) for name in MORPHOLOGY_FEATURES]
                + [context[name] for name in names[len(MORPHOLOGY_FEATURES) :]]
                for row, context in zip(eligible, contexts)
            ],
            dtype=float,
        )
        values = bundle["identity_classifier"].predict_proba(matrix)[:, 1]
        if "phikon_classifier" in bundle:
            embeddings = phikon_embeddings([
                phikon_patch(rgb, float(row["centroid_x_px"]), float(row["centroid_y_px"]), pixel_size_um) for row in eligible
            ])
            parts = [values, bundle["phikon_classifier"].predict_proba(embeddings)[:, 1]]
            if bundle.get("use_wong_consensus"):
                consensus = consensus_probabilities([
                    cnn_patch(rgb, float(row["centroid_x_px"]), float(row["centroid_y_px"]), pixel_size_um) for row in eligible
                ])
                parts.append(consensus[:, :2].max(axis=1))
                caa_scores = {id(row): float(score) for row, score in zip(eligible, consensus[:, 2])}
            values = np.mean(parts, axis=0)
        probabilities = {id(row): float(value) for row, value in zip(eligible, values)}
    output = []
    for row in objects:
        probability = probabilities.get(id(row), float("nan"))
        accepted = bool(probability >= float(bundle["identity_threshold"]))
        normalized = float(row["inner_mean_dab_od"]) / dab_threshold
        morphotype_eligible = accepted and float(row["equivalent_diameter_um"]) >= float(
            bundle["morphotype_minimum_diameter_um"]
        )
        if not accepted:
            plaque_class = "rejected"
        elif not morphotype_eligible:
            plaque_class = "small_plaque"
        elif normalized >= float(bundle["morphotype_threshold"]):
            plaque_class = "compact"
        else:
            plaque_class = "diffuse"
        output.append(
            {
                **row,
                "proposal_class": row["candidate_class"],
                "candidate_class": plaque_class,
                "plaque_probability": probability,
                "wong_caa_probability": caa_scores.get(id(row), float("nan")),
                "normalized_inner_dab": normalized,
            }
        )
    return output


def outer_boundary_contact(mask: np.ndarray, non_tissue: np.ndarray) -> float:
    filled = ndimage.binary_fill_holes(mask)
    boundary = filled & ~cv2.erode(filled.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    near = cv2.dilate(non_tissue.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))).astype(bool) & ~filled
    touching = cv2.dilate(near.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    return float((boundary & touching).sum() / max(boundary.sum(), 1))


def flag_vascular_or_edge(
    objects: list[dict[str, object]],
    labels: np.ndarray,
    non_tissue: np.ndarray,
    minimum_contact: float = 0.25,
    minimum_caa: float = 0.8,
) -> list[dict[str, object]]:
    bounds = ndimage.find_objects(labels)
    for row in objects:
        if row["candidate_class"] not in {"compact", "diffuse", "small_plaque"}:
            continue
        box = bounds[int(row["label"]) - 1]
        pad = 12
        y0, y1 = max(0, box[0].start - pad), min(labels.shape[0], box[0].stop + pad)
        x0, x1 = max(0, box[1].start - pad), min(labels.shape[1], box[1].stop + pad)
        contact = outer_boundary_contact(labels[y0:y1, x0:x1] == int(row["label"]), non_tissue[y0:y1, x0:x1])
        row["lumen_contact_fraction"] = contact
        caa = float(row.get("wong_caa_probability", float("nan")))
        vascular_by_cnn = caa >= minimum_caa and float(row["equivalent_diameter_um"]) >= 15.0
        if contact >= minimum_contact or vascular_by_cnn:
            row["candidate_class"] = "vascular_or_edge"
    return objects


__all__ = [
    "flag_vascular_or_edge",
    "classify_amyloid_objects",
    "review_patch",
    "train_amyloid_bundle",
]
