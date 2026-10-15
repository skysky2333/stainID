"""Prepare images (`stainid images`): the image list of an image-set or whole-slide project becomes the core image
table and the tiles, exactly as Find cores, Export cores and Cut cores into tiles make them for a TMA study.

- Image set: every image is one core image.
- Whole slides: every slide is covered by a grid of square regions (`REGION_PX` at the analysis scale, about 1.7 mm,
  the size of a large TMA core); every region that holds tissue, cut down to its tissue, is one core image.

Each core image is brought to the analysis scale (`pixel_size_um`, the models' scale) and to brown-on-blue (DAB)
colours (`stainid.imaging.core_images`): an image that already is both is used where it lies, any other is linked.
Then each core image is cut into ~560 µm tiles (an image smaller than a tile is one tile; larger ones are cut into equal
tiles of at most 2048 px), and the labels of the dataset package the project was made from are brought in
(`stainid.imageset.annotations`).

The core image table names the source of every core image (`source_path`, `source_box` in the source's own pixels,
`scale`), so positions on the source (e.g. a dataset's annotations) can be found on the core image and its tiles.
"""
from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from stainid.imageset.table import read_images, resolve
from stainid.imaging import core_images, settings
from stainid.imaging.sources import MissingReader, Source, open_source
from stainid.imaging.tissue import field_tissue_mask
from stainid.project import Project
from stainid.tables import read_csv, write_records

REGION_PX = 6144
THUMB_UM = 8.0  # whole slides are searched for tissue on a picture with 8 µm pixels
MIN_REGION_TISSUE = 0.05
REGION_MARGIN_UM = 60.0
MIN_TILE_TISSUE = {"images": 0.01, "slides": 0.10}
DIRECT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}  # read as they are when already at scale and in DAB
DIRECT_LIMIT = 16384 * 16384


def batch_of(row: dict) -> str:
    """The group stain thresholds are set for: the staining batch, else the slide, else the site, else one for all."""
    return row.get("batch") or row.get("slide_id") or row.get("site") or "all"


def sample_region(donor: str, region: str) -> str:
    return f"{donor}_{region}" if region else donor


def tissue_regions(source: Source, scale: float, native_um: float, region_px: int = REGION_PX) -> list[tuple[int, int, int, int]]:
    """Boxes (level-0 pixels) of the slide's tissue: a grid of squares of `region_px` analysis pixels, each kept when it
    holds tissue and cut down to the tissue in it (plus a margin), so regions never overlap."""
    thumb_scale = min(native_um / THUMB_UM, 4096 / max(source.width, source.height))
    thumb = source.region((0, 0, source.width, source.height), thumb_scale)
    mask = field_tissue_mask(thumb).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))).astype(bool)
    cell = region_px / scale  # level-0 pixels per region side
    margin = REGION_MARGIN_UM / native_um
    boxes = []
    for y0 in np.arange(0, source.height, cell):
        for x0 in np.arange(0, source.width, cell):
            x1, y1 = min(source.width, x0 + cell), min(source.height, y0 + cell)
            ty0, ty1 = int(y0 * thumb_scale), max(int(y0 * thumb_scale) + 1, int(y1 * thumb_scale))
            tx0, tx1 = int(x0 * thumb_scale), max(int(x0 * thumb_scale) + 1, int(x1 * thumb_scale))
            patch = mask[ty0:ty1, tx0:tx1]
            if patch.size == 0 or patch.mean() < MIN_REGION_TISSUE:
                continue
            ys, xs = np.nonzero(patch)
            bx0 = max(x0, (tx0 + xs.min()) / thumb_scale - margin)
            by0 = max(y0, (ty0 + ys.min()) / thumb_scale - margin)
            bx1 = min(x1, (tx0 + xs.max() + 1) / thumb_scale + margin)
            by1 = min(y1, (ty0 + ys.max() + 1) / thumb_scale + margin)
            boxes.append((int(bx0), int(by0), max(1, int(bx1 - bx0)), max(1, int(by1 - by0))))
    return boxes


def _direct(source: Source, scale: float, recipe) -> bool:
    return (scale == 1.0 and recipe.is_plain_dab and source.reader in ("picture", "tiff") and len(source.levels) == 1
            and source.path.suffix.lower() in DIRECT and source.width * source.height <= DIRECT_LIMIT)


def even_grid(length: int, field: int) -> list[tuple[int, int]]:
    """Equal pieces of at most `field` px covering `length` (one piece when it is shorter)."""
    count = max(1, math.ceil(length / field))
    edges = [round(i * length / count) for i in range(count + 1)]
    return [(edges[i], edges[i + 1] - edges[i]) for i in range(count)]


KEEP = ("core_id", "tma", "core_label", "donor_id", "sample_region_id", "region", "disease_group", "technical_replicate", "stain",
        "antibody", "image_id", "split", "image_path", "pixel_width_um", "pixel_height_um")


def image_tiles(project: Project, core: dict, field: int, kind: str) -> list[dict]:
    """The tiles of one core image: equal squares of at most `field` px, those with some tissue (images) or at least
    10% tissue (slide regions)."""
    width, height = int(core["native_width_px"]), int(core["native_height_px"])
    reduce = 8 if min(width, height) >= 2048 else 4 if min(width, height) >= 512 else 1
    flag = {8: cv2.IMREAD_REDUCED_COLOR_8, 4: cv2.IMREAD_REDUCED_COLOR_4, 1: cv2.IMREAD_COLOR}[reduce]
    bgr = core_images.read_bgr(project.root / core["image_path"], flag)
    if bgr is None:
        raise ValueError(f"could not read {core['image_path']}")
    tissue = field_tissue_mask(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    sy, sx = tissue.shape[0] / height, tissue.shape[1] / width
    out = []
    for r, (y, h) in enumerate(even_grid(height, field)):
        for c, (x, w) in enumerate(even_grid(width, field)):
            patch = tissue[int(y * sy): max(int(y * sy) + 1, int((y + h) * sy)), int(x * sx): max(int(x * sx) + 1, int((x + w) * sx))]
            fraction = float(patch.mean()) if patch.size else 0.0
            if fraction < MIN_TILE_TISSUE[kind]:
                continue
            out.append({"tile_id": f"{core['core_id']}_{core['stain']}_W{r:02d}{c:02d}", **{k: core.get(k, "") for k in KEEP},
                        "selection_role": "whole_core_grid", "selection_order": 100 + r * 100 + c, "nested_sample": "whole_core",
                        "x_px": x, "y_px": y, "width_px": w, "height_px": h, "preview_tissue_fraction": round(fraction, 4)})
    return out


def _groups(project: Project) -> dict[str, str]:
    path = project.input("donor_metadata")
    return {r["donor_id"]: r.get("disease_group", "") for r in read_csv(path)} if path.exists() else {}


def core_rows(project: Project, log=print) -> tuple[list[dict], list[str]]:
    """The core image rows of every analysed image in the image list, and a note for each image that was left out."""
    kind = project.kind
    groups, rows, notes = _groups(project), [], []
    mode = settings.rescale_mode(project)
    replicate: dict[tuple[str, str], int] = defaultdict(int)
    images = read_images(project)
    for number, row in enumerate(images, start=1):
        stain = settings.analysis_stain(project, row.get("stain", ""))
        name = row.get("image_id") or Path(row["path"]).stem
        if not stain:
            notes.append(f"{name}: stain {row.get('stain') or '(none)'} is not analysed (say what it is under Imaging)")
            continue
        if not row.get("donor_id"):
            notes.append(f"{name}: no donor")
            continue
        try:
            source = open_source(resolve(project, row))
        except (MissingReader, FileNotFoundError, OSError, ValueError) as error:
            notes.append(f"{name}: {error}")
            continue
        native = settings.native_pixel_size(project, row.get("pixel_size_um"), source.pixel_size_um)
        if native is None:
            notes.append(f"{name}: pixel size not known, read as {project.pixel_size_um} µm")
            native = project.pixel_size_um
        scale = settings.scale_factor(project, native)
        recipe = settings.recipe_for(project, row["stain"])
        boxes = [(0, 0, source.width, source.height)] if kind == "images" else tissue_regions(source, scale, native)
        if not boxes:
            notes.append(f"{name}: no tissue found")
        for index, box in enumerate(boxes, start=1):
            core_id = core_images.safe_name(name if kind == "images" else f"{name}_R{index:02d}")
            size = (max(1, round(box[2] * scale)), max(1, round(box[3] * scale)))
            if kind == "images" and _direct(source, scale, recipe):
                image_path = project.relative(source.path)
            else:
                link = core_images.write_link(core_images.link_path(project.root, f"{core_id}_{stain}"), source.path, box, scale, recipe,
                                              mode == "on_import", size)
                image_path = project.relative(link)
            region = row.get("region", "")
            sample = sample_region(row["donor_id"], region)
            replicate[(sample, stain)] += 1
            rows.append({"core_id": core_id, "tma": batch_of(row), "core_label": core_id, "donor_id": row["donor_id"], "sample_region_id": sample,
                         "region": region, "disease_group": row.get("disease_group") or groups.get(row["donor_id"], ""),
                         "technical_replicate": replicate[(sample, stain)], "stain": stain, "antibody": row.get("stain", ""),
                         "image_id": name, "slide_id": row.get("slide_id", ""), "site": row.get("site", ""), "split": row.get("split", ""),
                         "image_path": image_path, "native_width_px": size[0], "native_height_px": size[1],
                         "pixel_width_um": project.pixel_size_um, "pixel_height_um": project.pixel_size_um,
                         "source_path": str(source.path), "source_pixel_size_um": native, "source_box": " ".join(map(str, box)),
                         "scale": round(scale, 6), "chromogen": recipe.interest, "tissue_status": "complete", "tissue_fraction": "",
                         "review_required": "false"})
        log(f"[{number}/{len(images)}] {name}: {len(boxes)} {'core image' if len(boxes) == 1 else 'core images'}", flush=True)
    return rows, notes


def prepare_images(project: Project, tile_size: int = 2048, log=print) -> list[Path]:
    """Core image table, tiles and imported labels of an image-set or whole-slide project."""
    if project.is_tma:
        raise ValueError("Prepare images is for image-set and whole-slide projects; a TMA study uses Find cores and Cut cores into tiles")
    if not read_images(project):
        raise ValueError("The image list is empty: add images on Set up → Images first")
    cores, notes = core_rows(project, log)
    for note in notes:
        log(f"left out: {note}", flush=True)
    if not cores:
        raise ValueError("None of the images could be prepared: " + "; ".join(notes[:5]))
    write_records(project.input("core_images"), cores)
    links = [project.root / c["image_path"] for c in cores if core_images.is_link(c["image_path"])]
    if settings.rescale_mode(project) == "on_import" and links:
        for number, link in enumerate(links, start=1):
            core_images.prepare(link)
            if number % 25 == 0 or number == len(links):
                log(f"[{number}/{len(links)}] pictures made at the analysis scale", flush=True)
    tiles = []
    for core in cores:
        tiles += image_tiles(project, core, tile_size, project.kind)
    write_records(project.input("whole_core_manifest"), tiles)
    log(f"{len(cores)} core images, {len(tiles)} tiles -> {project.relative(project.input('whole_core_manifest'))}", flush=True)
    written = [project.input("core_images"), project.input("whole_core_manifest")]
    if project.config.get("dataset", {}).get("package"):
        from stainid.imageset.annotations import import_package_labels

        summary = import_package_labels(project, log=log)
        written.append(summary)
    return written
