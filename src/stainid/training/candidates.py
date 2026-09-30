"""Every candidate object of a field with the exact features the stain's random forest uses.

The same functions that the detection pipelines call produce the candidates and features, so a model trained on
these rows can be dropped into the pipeline unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np
from PIL import Image

from stainid.imaging.tissue import fold_mask, linear_artifact_mask
from stainid.pipelines.cohort_v1 import context_crop
from stainid.project import Project
from stainid.qc.exclusions import rasterize_core_exclusions, read_manual_exclusions
from stainid.workflows import calibration_table

Image.MAX_IMAGE_PIXELS = None
NEUN_CONTEXT_UM = 55.0


@dataclass
class Field:
    tile: dict[str, str]
    rgb: np.ndarray
    inner: tuple[slice, slice]
    pixel_size_um: float
    threshold: float
    manual: np.ndarray

    @cached_property
    def exclusion(self) -> np.ndarray:
        return linear_artifact_mask(self.rgb) | fold_mask(self.rgb, self.pixel_size_um) | self.manual

    def inside(self, row: dict) -> bool:
        return self.inner[1].start <= float(row["centroid_x_px"]) < self.inner[1].stop and self.inner[0].start <= float(row["centroid_y_px"]) < self.inner[0].stop


class FieldLoader:
    def __init__(self, project: Project):
        self.project = project
        self.calibration = calibration_table(project.input("calibration"))
        self.manual = read_manual_exclusions(project.input("manual_exclusions"))
        self._cellpose = None

    def load(self, tile: dict[str, str]) -> Field:
        x, y = int(tile["x_px"]), int(tile["y_px"])
        with Image.open(self.project.root / tile["image_path"]) as image:
            rgb, inner = context_crop(image, x, y, int(tile["width_px"]), int(tile["height_px"]), self.project.context_px)
        manual = rasterize_core_exclusions(self.manual.get(tile["image_path"], []), x - inner[1].start, y - inner[0].start, rgb.shape[:2])
        pixel_size = float(np.sqrt(float(tile["pixel_width_um"]) * float(tile["pixel_height_um"])))
        return Field(tile, rgb, inner, pixel_size, self.calibration[(tile["tma"], tile["stain"])], manual)

    def cellpose(self, rgb: np.ndarray, cache) -> np.ndarray:
        """Cellpose-SAM labels for a field, read from (or written to) the pipeline's cache so work is never repeated."""
        if cache.exists():
            with np.load(cache) as loaded:
                return loaded["masks"]
        from stainid.nuclei import cellpose_masks, load_cellpose

        if self._cellpose is None:
            self.project.apply_environment()
            self._cellpose = load_cellpose(self.project.model("cellpose"), 4, "cpu")
        masks = cellpose_masks(self._cellpose, rgb)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, masks=masks.astype(np.uint32))
        return masks


def neun(loader: FieldLoader, field: Field) -> tuple[list[dict], np.ndarray, list[str]]:
    from stainid.stains.neun.pipeline import neun_candidates, neun_feature_matrix

    tile = field.tile
    raw = loader.cellpose(field.rgb, loader.project.output("neun") / "cellpose_masks" / f"{tile['tile_id']}.npz")
    rows, _, _ = neun_candidates(field.rgb, raw, field.threshold, float(tile["pixel_width_um"]), float(tile["pixel_height_um"]), field.manual, tile["tile_id"])
    rows = [r for r in rows if field.inside(r)]
    if not rows:
        return [], np.zeros((0, 0)), []
    matrix, names = neun_feature_matrix(field.rgb, rows, field.pixel_size_um, NEUN_CONTEXT_UM)
    return rows, matrix, names


def amyloid(loader: FieldLoader, field: Field) -> tuple[list[dict], np.ndarray, list[str]]:
    from stainid.stains.amyloid.model import IDENTITY_CLASSES, identity_features
    from stainid.stains.amyloid.segmentation import segment_amyloid_candidates

    _, objects, _, _, _ = segment_amyloid_candidates(field.rgb, field.threshold, field.pixel_size_um, field.pixel_size_um, field.exclusion, split_touching=True)
    rows = [r for r in objects if r["candidate_class"] in IDENTITY_CLASSES and field.inside(r)]
    if not rows:
        return [], np.zeros((0, 0)), []
    for row in rows:
        row["normalized_inner_dab"] = float(row["inner_mean_dab_od"]) / field.threshold
    matrix, names = identity_features(field.rgb, rows, field.pixel_size_um)
    return rows, matrix, names


def tau(loader: FieldLoader, field: Field) -> tuple[list[dict], np.ndarray, list[str]]:
    from stainid.nuclei import clean_nuclei
    from stainid.stains.tau.neurons import add_context_features, tau_feature_matrix, tau_neuron_candidates

    nuclei = loader.cellpose(field.rgb, loader.project.output("nuclei") / f"{field.tile['tile_id']}.npz")
    objects, _ = tau_neuron_candidates(field.rgb, clean_nuclei(nuclei, field.rgb), field.threshold, field.pixel_size_um, field.exclusion)
    rows = [r for r in add_context_features(field.rgb, objects, field.pixel_size_um) if field.inside(r)]
    if not rows:
        return [], np.zeros((0, 0)), []
    matrix, names = tau_feature_matrix(rows)
    return rows, matrix, names


EXTRACT = {"NeuN": neun, "6E10": amyloid, "AT8": tau}


def amyloid_extras(field: Field, rows: list[dict], project: Project) -> tuple[np.ndarray | None, np.ndarray]:
    """Phikon embeddings (to train the linear probe) and Wong 2022 consensus plaque scores (fixed ensemble member)."""
    from stainid.stains.amyloid.model import phikon_embeddings, phikon_patch

    project.apply_environment()
    xy = [(float(r["centroid_x_px"]), float(r["centroid_y_px"])) for r in rows]
    embeddings = phikon_embeddings([phikon_patch(field.rgb, x, y, field.pixel_size_um) for x, y in xy]) if project.model("huggingface_home").exists() else None
    wong = np.full(len(rows), np.nan)
    if project.model("plaque_cnn_dir").exists():
        from stainid.stains.amyloid.published_cnn import cnn_patch, consensus_probabilities

        wong = consensus_probabilities([cnn_patch(field.rgb, x, y, field.pixel_size_um) for x, y in xy])[:, :2].max(axis=1)
    return embeddings, wong
