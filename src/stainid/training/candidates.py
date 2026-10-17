"""Every candidate object of a field with the exact features the stain's random forest uses.

The same functions that the detection pipelines call produce the candidates and features, so a model trained on
these rows can be dropped into the pipeline unchanged. `measure_at` measures one object people added where the
detection step proposed nothing, with the same segmentation (restricted to a window around the click) and features.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np
from PIL import Image

from stainid.imaging.core_images import open_image
from stainid.imaging.tissue import fold_mask, linear_artifact_mask
from stainid.pipelines.cohort_v1 import context_crop
from stainid.project import Project
from stainid.tables import save_npz_atomic
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
        self._cellpose = None

    def load(self, tile: dict[str, str]) -> Field:
        x, y = int(tile["x_px"]), int(tile["y_px"])
        with open_image(self.project.root / tile["image_path"]) as image:
            rgb, inner = context_crop(image, x, y, int(tile["width_px"]), int(tile["height_px"]), self.project.context_px)
        manual = np.zeros(rgb.shape[:2], bool)  # detection's own areas only, as stage 2 (people's areas leave labels out instead)
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
        save_npz_atomic(cache, masks=masks.astype(np.uint32))
        return masks


Box = tuple[slice, slice]


def _whole(field: Field) -> Box:
    return slice(0, field.rgb.shape[0]), slice(0, field.rgb.shape[1])


def _moved(rows: list[dict], box: Box) -> list[dict]:
    """Rows segmented in a part of the field, with their centroids in field pixels."""
    dy, dx = box[0].start, box[1].start
    return [{**r, "centroid_x_px": float(r["centroid_x_px"]) + dx, "centroid_y_px": float(r["centroid_y_px"]) + dy} for r in rows] if dx or dy else rows


def _nuclei(loader: FieldLoader, field: Field, compute: bool) -> np.ndarray | None:
    cache = loader.project.output("nuclei") / f"{field.tile['tile_id']}.npz"
    return loader.cellpose(field.rgb, cache) if compute or cache.exists() else None


def _with_masks(rows: list[dict], layers: list[np.ndarray], which) -> tuple[list[dict], list[np.ndarray]]:
    """Rows that name their own mask (`mask_layer`, `mask_label`: the label of the object in `layers[mask_layer]`)."""
    for row in rows:
        row["mask_layer"], row["mask_label"] = which(row)
    return rows, layers


def neun_segment(loader: FieldLoader, field: Field, box: Box, threshold: float, compute_nuclei: bool = True, masks: bool = False):
    """The candidates in a part of the field; with `masks`, also their masks (DAB profiles, then Cellpose cells)."""
    from stainid.stains.neun.pipeline import neun_candidates

    tile = field.tile
    raw = _nuclei(loader, field, compute_nuclei)
    if raw is None:
        raise LookupError("this field has no cell outlines yet")
    rows, profiles, cells = neun_candidates(field.rgb[box], raw[box], threshold, float(tile["pixel_width_um"]), float(tile["pixel_height_um"]), field.manual[box], tile["tile_id"])
    rows = _moved(rows, box)
    return _with_masks(rows, [profiles, cells], lambda r: (0 if r["source_kind"] == "dab_profile" else 1, int(r["source_label"]))) if masks else rows


def neun_features(loader: FieldLoader, field: Field, rows: list[dict]) -> tuple[np.ndarray, list[str]]:
    from stainid.stains.neun.pipeline import neun_feature_matrix

    return neun_feature_matrix(field.rgb, rows, field.pixel_size_um, NEUN_CONTEXT_UM)


def amyloid_segment(loader: FieldLoader, field: Field, box: Box, threshold: float, compute_nuclei: bool = True, masks: bool = False):
    from stainid.stains.amyloid.segmentation import segment_amyloid_candidates

    _, objects, labels, _, _ = segment_amyloid_candidates(field.rgb[box], threshold, field.pixel_size_um, field.pixel_size_um, field.exclusion[box], split_touching=True)
    objects = _moved(objects, box)
    return _with_masks(objects, [labels], lambda r: (0, int(r["label"]))) if masks else objects


def amyloid_features(loader: FieldLoader, field: Field, rows: list[dict]) -> tuple[np.ndarray, list[str]]:
    from stainid.stains.amyloid.model import identity_features

    for row in rows:
        row["normalized_inner_dab"] = float(row["inner_mean_dab_od"]) / field.threshold
    return identity_features(field.rgb, rows, field.pixel_size_um)


def tau_segment(loader: FieldLoader, field: Field, box: Box, threshold: float, compute_nuclei: bool = True, masks: bool = False):
    from stainid.nuclei import clean_nuclei
    from stainid.stains.tau.neurons import tau_neuron_candidates

    nuclei = _nuclei(loader, field, compute_nuclei)
    if nuclei is None:
        raise LookupError("this field has no cell outlines yet")
    rgb = field.rgb[box]
    objects, labels = tau_neuron_candidates(rgb, clean_nuclei(nuclei[box], rgb), threshold, field.pixel_size_um, field.exclusion[box])
    objects = _moved(objects, box)
    return _with_masks(objects, [labels], lambda r: (0, int(r["label"]))) if masks else objects


def tau_features(loader: FieldLoader, field: Field, rows: list[dict]) -> tuple[np.ndarray, list[str]]:
    from stainid.stains.tau.neurons import add_context_features, tau_feature_matrix

    rows[:] = add_context_features(field.rgb, rows, field.pixel_size_um)
    return tau_feature_matrix(rows)


def _candidate_class(stain: str):
    """Which segmented objects the detection step considers (the 6E10 identity model never sees artifacts)."""
    if stain != "6E10":
        return lambda row: True
    from stainid.stains.amyloid.model import IDENTITY_CLASSES

    return lambda row: row["candidate_class"] in IDENTITY_CLASSES


SEGMENT = {"NeuN": neun_segment, "6E10": amyloid_segment, "AT8": tau_segment}
FEATURES = {"NeuN": neun_features, "6E10": amyloid_features, "AT8": tau_features}


def extract(stain: str, loader: FieldLoader, field: Field) -> tuple[list[dict], np.ndarray, list[str]]:
    """Every candidate of the field's tile (centroid inside the tile) with its features, as the detection step finds them."""
    keep = _candidate_class(stain)
    rows = [r for r in SEGMENT[stain](loader, field, _whole(field), field.threshold) if keep(r) and field.inside(r)]
    if not rows:
        return [], np.zeros((0, 0)), []
    matrix, names = FEATURES[stain](loader, field, rows)
    return rows, matrix, names


def segment_with_masks(stain: str, loader: FieldLoader, field: Field) -> tuple[list[dict], list[np.ndarray]]:
    """Every candidate of the whole field (context margin included, unlike `extract`) with its mask, as the detection step
    segments them: rows carry `mask_layer` / `mask_label` into the returned label images (field pixels)."""
    keep = _candidate_class(stain)
    rows, layers = SEGMENT[stain](loader, field, _whole(field), field.threshold, masks=True)
    return [r for r in rows if keep(r)], layers


def neun(loader: FieldLoader, field: Field) -> tuple[list[dict], np.ndarray, list[str]]:
    return extract("NeuN", loader, field)


def amyloid(loader: FieldLoader, field: Field) -> tuple[list[dict], np.ndarray, list[str]]:
    return extract("6E10", loader, field)


def tau(loader: FieldLoader, field: Field) -> tuple[list[dict], np.ndarray, list[str]]:
    return extract("AT8", loader, field)


EXTRACT = {"NeuN": neun, "6E10": amyloid, "AT8": tau}

# An object a person added where the detection step proposed nothing: segmented in a window around the click, first as the
# detection step does it, then with the stain threshold lowered (objects are often missed for being too faint).
WINDOW_UM = {"NeuN": 60.0, "6E10": 150.0, "AT8": 60.0}
REACH_UM = 5.0
PASSES = ((1.0, "as the detection step segments it"), (0.5, "at half the slide's stain threshold"))


def object_radius_um(row: dict) -> float:
    if row.get("equivalent_diameter_um") not in (None, ""):
        return 0.5 * float(row["equivalent_diameter_um"])
    for key in ("soma_area_um2", "selection_area_um2", "area_um2", "deposit_area_um2"):
        if row.get(key) not in (None, ""):
            return float(np.sqrt(float(row[key]) / np.pi))
    return 0.0


def measure_at(stain: str, loader: FieldLoader, field: Field, x: float, y: float) -> tuple[dict | None, np.ndarray | None, list[str], str]:
    """The object at a clicked position (field pixels): the segmented object under or nearest the click (its centre at most
    its own radius, and at least REACH_UM, away), with the stain model's features. Returns (row, features, names, how);
    row is None when nothing could be segmented there, and `how` then says why in plain words."""
    px = field.pixel_size_um
    half = int(round(WINDOW_UM[stain] / px))
    height, width = field.rgb.shape[:2]
    if not (0 <= x < width and 0 <= y < height):
        return None, None, [], "the position is outside the field picture"
    cx, cy = int(x), int(y)
    box = (slice(max(0, cy - half), min(height, cy + half + 1)), slice(max(0, cx - half), min(width, cx + half + 1)))
    if stain in ("6E10", "AT8") and field.exclusion[cy, cx]:
        return None, None, [], "the spot lies in an area left out of the analysis (a fold, a tissue edge or an area marked by hand)"
    for scale, how in PASSES:
        try:
            rows = SEGMENT[stain](loader, field, box, field.threshold * scale, compute_nuclei=False)
        except LookupError as error:
            return None, None, [], str(error)
        except ValueError:
            return None, None, [], "there is too little tissue around the spot to measure it"
        near = [(float(np.hypot(float(r["centroid_x_px"]) - x, float(r["centroid_y_px"]) - y)), i) for i, r in enumerate(rows)]
        near = [(d, i) for d, i in near if d * px <= max(REACH_UM, object_radius_um(rows[i]))]
        if near:
            row = rows[min(near)[1]]
            matrix, names = FEATURES[stain](loader, field, [row])
            return row, matrix[0], names, how
    return None, None, [], "no stained object could be segmented at this spot"


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
