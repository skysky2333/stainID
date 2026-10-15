"""Small synthetic dataset packages, one per kind: tiles with points, regions with boxes, a tiled TIFF whole slide,
image-level labels and a two-chromogen image. Pictures are drawn from stain amounts (Beer-Lambert), so the colours are
those the colour deconvolution expects."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image

from stainid.imaging.chromogens import VECTORS

BACKGROUND = 0.03  # glass is not quite white


def paint(height: int, width: int, spots: list[tuple[float, float, float, str]], counter: float = 0.25, seed: int = 0) -> np.ndarray:
    """RGB of tissue with a hematoxylin counterstain and round spots (x, y, radius, chromogen)."""
    rng = np.random.default_rng(seed)
    od = np.zeros((height, width, 3))
    unit = {k: np.array(v) / np.linalg.norm(v) for k, v in VECTORS.items()}
    od += (counter * (0.8 + 0.4 * rng.random((height, width))))[..., None] * unit["hematoxylin"]
    yy, xx = np.mgrid[:height, :width]
    for x, y, radius, chromogen in spots:
        inside = (xx - x) ** 2 + (yy - y) ** 2 <= radius ** 2
        od[inside] += 0.9 * unit[chromogen]
    od += BACKGROUND
    return np.clip(255 * 10 ** (-od), 0, 255).astype(np.uint8)


def _write(folder: Path, meta: dict, images: list[dict], annotations: list[dict], donors: list[dict] | None = None) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "dataset.yaml").write_text(yaml.safe_dump(meta, sort_keys=False))
    pd.DataFrame(images).to_csv(folder / "images.csv", index=False)
    if annotations:
        pd.DataFrame(annotations).to_csv(folder / "annotations.csv", index=False)
    if donors:
        pd.DataFrame(donors).to_csv(folder / "donors.csv", index=False)
    return folder


def tiles_with_points(root: Path) -> tuple[Path, dict[str, list[tuple[float, float]]]]:
    """Three 600 px DAB tiles at 0.5 µm/px (so they are enlarged 1.83 times), plaques marked as points by two experts
    (expert2 marks one more), and a donor table. Returns the package and each image's true plaque centres."""
    raw = root / "raw" / "tiles"
    raw.mkdir(parents=True)
    images, annotations, truth = [], [], {}
    rng = np.random.default_rng(1)
    for i, donor in enumerate(["D1", "D1", "D2"]):
        centres = [(float(x), float(y)) for x, y in rng.integers(80, 520, size=(4, 2))]
        Image.fromarray(paint(600, 600, [(x, y, 14, "DAB") for x, y in centres], seed=i)).save(raw / f"tile{i}.png")
        image_id = f"tile{i}"
        truth[image_id] = centres
        images.append({"image_id": image_id, "path": f"../raw/tiles/tile{i}.png", "width_px": 600, "height_px": 600, "pixel_size_um": 0.5,
                       "stain": "4G8", "slide_id": f"S{donor}", "donor_id": donor, "region": "temporal", "site": "lab A", "split": "test"})
        for rater in ("expert1", "expert2"):
            for x, y in centres:
                annotations.append({"image_id": image_id, "rater": rater, "kind": "point", "x": x + rng.normal(0, 2), "y": y + rng.normal(0, 2),
                                    "label": "cored" if rater == "expert1" else "Cored plaque", "stainid_class": ""})
        annotations.append({"image_id": image_id, "rater": "expert2", "kind": "point", "x": 300, "y": 40, "label": "diffuse", "stainid_class": ""})
    meta = {"id": "synthetic_points", "title": "Synthetic points", "licence": "CC0", "citation": "None", "image_kind": "tiles", "pixel_size_um": 0.5,
            "stains": [{"stain": "4G8", "antibody": "4G8", "chromogen": "DAB", "counterstain": "hematoxylin"}],
            "class_map": {"cored": "compact plaque", "Cored plaque": "compact plaque", "diffuse": "diffuse plaque"}}
    donors = [{"donor_id": "D1", "diagnosis": "AD", "braak": "VI", "age": 80}, {"donor_id": "D2", "diagnosis": "Control", "braak": "I", "age": 70}]
    return _write(root / "package", meta, images, annotations, donors), truth


def regions_with_boxes(root: Path) -> Path:
    """Two 900 px NovaRed regions already at the analysis scale, with boxes from one rater."""
    raw = root / "raw"
    raw.mkdir(parents=True)
    images, annotations = [], []
    for i in range(2):
        spots = [(200.0 + 300 * k, 300.0 + 150 * i, 20.0, "NovaRed") for k in range(3)]
        Image.fromarray(paint(900, 900, spots, seed=10 + i)).save(raw / f"region{i}.png")
        images.append({"image_id": f"r{i}", "path": str(raw / f"region{i}.png"), "pixel_size_um": 0.2738, "stain": "NAB228", "donor_id": f"P{i}"})
        annotations += [{"image_id": f"r{i}", "rater": "consensus", "kind": "box", "x": x - 25, "y": y - 25, "x2": x + 25, "y2": y + 25,
                         "label": "plaque"} for x, y, _, _ in spots]
        annotations += [{"image_id": f"r{i}", "rater": "expert1", "kind": "polygon", "label": "plaque",
                         "polygon": json_points([(x - 20, y), (x, y - 20), (x + 20, y), (x, y + 20)])} for x, y, _, _ in spots[:2]]
    meta = {"id": "synthetic_boxes", "title": "Synthetic boxes", "image_kind": "regions", "stains": [{"stain": "NAB228", "chromogen": "NovaRed"}],
            "class_map": {"plaque": "plaque"}}
    return _write(root / "package", meta, images, annotations)


def whole_slide(root: Path) -> tuple[Path, list[tuple[float, float]]]:
    """A pyramidal tiled TIFF (3000 x 2400 px at 0.5 µm/px) with two pieces of tissue and AT8 tangles marked as points."""
    import tifffile

    raw = root / "raw"
    raw.mkdir(parents=True)
    rgb = np.full((2400, 3000, 3), 245, np.uint8)
    spots = [(500.0, 500.0, 10.0, "DAB"), (700.0, 650.0, 10.0, "DAB"), (2200.0, 1800.0, 10.0, "DAB")]
    for (x0, y0), seed in (((200, 200), 1), ((1900, 1500), 2)):
        piece = paint(700, 700, [(x - x0, y - y0, r, c) for x, y, r, c in spots if x0 <= x < x0 + 700 and y0 <= y < y0 + 700], seed=seed)
        rgb[y0:y0 + 700, x0:x0 + 700] = piece
    path = raw / "slide.tif"
    with tifffile.TiffWriter(path) as tif:
        options = {"tile": (256, 256), "photometric": "rgb", "compression": "zlib", "resolution": (1e4 / 0.5, 1e4 / 0.5), "resolutionunit": "CENTIMETER"}
        tif.write(rgb, subifds=2, **options)
        for factor in (2, 4):
            tif.write(rgb[::factor, ::factor], subfiletype=1, **options)
    images = [{"image_id": "slide1", "path": str(path), "pixel_size_um": "", "stain": "AT8", "donor_id": "W1", "slide_id": "slide1"}]
    annotations = [{"image_id": "slide1", "rater": "expert1", "kind": "point", "x": x, "y": y, "label": "tangle"} for x, y, _, _ in spots]
    meta = {"id": "synthetic_wsi", "title": "Synthetic slide", "image_kind": "wsi", "stains": [{"stain": "AT8", "chromogen": "DAB"}],
            "class_map": {"tangle": "tau+ neuron: tangle"}}
    return _write(root / "package", meta, images, annotations, [{"donor_id": "W1", "diagnosis": "AD"}]), [(x, y) for x, y, _, _ in spots]


def image_labels(root: Path) -> Path:
    """Four tiles, two with a plaque, labelled per image by three raters (one disagrees on one image)."""
    raw = root / "raw"
    raw.mkdir(parents=True)
    images, annotations = [], []
    for i in range(4):
        has = i < 2
        Image.fromarray(paint(512, 512, [(256.0, 256.0, 30.0, "DAB")] if has else [], seed=20 + i)).save(raw / f"crop{i}.png")
        images.append({"image_id": f"c{i}", "path": str(raw / f"crop{i}.png"), "pixel_size_um": 0.2738, "stain": "6E10", "donor_id": "X"})
        for rater in ("a", "b", "c"):
            said = has if not (rater == "c" and i == 1) else False
            annotations.append({"image_id": f"c{i}", "rater": rater, "kind": "image_label", "label": "plaque" if said else "none"})
    meta = {"id": "synthetic_image_labels", "title": "Image labels", "image_kind": "tiles", "class_map": {"plaque": "plaque", "none": "not a plaque"}}
    return _write(root / "package", meta, images, annotations)


def two_chromogens(root: Path) -> tuple[Path, list[tuple[float, float]], list[tuple[float, float]]]:
    """One image with AT8 in blue and pTDP-43 in brown; the package says AT8 is the stain stainID analyses."""
    raw = root / "raw"
    raw.mkdir(parents=True)
    blue = [(150.0, 150.0), (400.0, 380.0)]
    brown = [(380.0, 120.0), (120.0, 400.0)]
    spots = [(x, y, 18.0, "blue") for x, y in blue] + [(x, y, 18.0, "DAB") for x, y in brown]
    Image.fromarray(paint(512, 512, spots, counter=0.15, seed=5)).save(raw / "at.png")
    images = [{"image_id": "at1", "path": str(raw / "at.png"), "pixel_size_um": 0.2738, "stain": "AT8+pTDP-43", "donor_id": "T1"}]
    meta = {"id": "synthetic_two_colour", "title": "Two chromogens", "image_kind": "tiles",
            "stains": [{"stain": "AT8", "chromogen": "blue"}, {"stain": "pTDP-43", "chromogen": "DAB"}]}
    return _write(root / "package", meta, images, []), blue, brown


def fake_plaques(project_root: Path, tiles: pd.DataFrame, positions: dict[str, list[tuple[float, float]]], stain: str = "6E10",
                 classes: str = "compact") -> None:
    """Pipeline outputs as Detect plaques would write them: one accepted object at each (core-image) position."""
    parts = project_root / "data" / "analysis" / "cohort_v3" / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    for core_id, points in positions.items():
        rows = []
        for x, y in points:
            tile = tiles[(tiles.core_id == core_id) & (tiles.x_px <= x) & (x < tiles.x_px + tiles.width_px) & (tiles.y_px <= y) & (y < tiles.y_px + tiles.height_px)].iloc[0]
            ox, oy = max(0, int(tile.x_px) - 128), max(0, int(tile.y_px) - 128)
            common = {k: tile[k] for k in ("tile_id", "core_id", "tma", "donor_id", "sample_region_id", "region", "disease_group", "stain")}
            if stain == "6E10":
                rows.append({**common, "centroid_x_px": x - ox, "centroid_y_px": y - oy, "plaque_class": classes, "plaque_probability": 0.9,
                             "plaque_area_um2": 300.0, "dense_core_fraction": 0.3, "equivalent_diameter_um": 20.0})
            else:
                rows.append({**common, "centroid_x_px": x - ox, "centroid_y_px": y - oy, "candidate_source": "dense_body", "tau_neuron_probability": 0.8,
                             "soma_area_um2": 150.0, "tau_neuron_mature": True})
        pd.DataFrame(rows).to_csv(parts / f"{core_id}_{stain}_objects.csv", index=False)
        pd.DataFrame([{"tile_id": rows[0]["tile_id"]}]).to_csv(parts / f"{core_id}_{stain}_features.csv", index=False)


def json_points(points: list[tuple[float, float]]) -> str:
    return json.dumps([[x, y] for x, y in points])
