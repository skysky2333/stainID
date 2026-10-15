"""Stand-ins for the core grid, there at once: every core of a slide cut from the slide picture Find cores kept.

Making a sharp thumbnail of a core takes a fraction of a second at best, so the grid shows these tiny pictures (a few
dozen px, sent inside the grid's answer and blurred by the page) until the sharp ones arrive. One decode of the slide
picture gives all its cores in a few tens of milliseconds. The picture is `qc/grids/previews/<slide>.jpg` (the plain
slide) when there is one, else `qc/grids/<slide>.jpg`, the slide with the fitted grid drawn on: what
`stainid.slides.dearray.draw_qc` drew there (circles, crosses, labels, the title bar) is painted over first.
"""
from __future__ import annotations

import base64
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from stainid import cache
from stainid.imaging.thumbnails import jpeg
from stainid.project import Project
from stainid.slides.dearray import slide_slug
from stainid.tables import read_csv

SIZE = 32  # px across
SOURCE_PX = 2 * SIZE  # a core is cut from a decode at least this large (slide pictures are decoded reduced)
REDUCED = {8: cv2.IMREAD_REDUCED_COLOR_8, 4: cv2.IMREAD_REDUCED_COLOR_4, 2: cv2.IMREAD_REDUCED_COLOR_2}
LABEL_FONT, LABEL_SCALE, LABEL_THICKNESS = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2  # as draw_qc writes the core labels


def _manifest(project: Project) -> list[dict[str, str]]:
    path = project.input("core_manifest")
    return cache.memo(("manifest rows", str(path)), [path], lambda: read_csv(path) if path.exists() else [])


def _slide_picture(project: Project, slide_path: str) -> tuple[Path, float, bool] | None:
    """(picture, its px per Find cores preview px, whether the fitted grid is drawn on it), or None."""
    grids, slug = project.output("qc") / "grids", slide_slug(Path(slide_path))
    plain, scale_file = grids / "previews" / f"{slug}.jpg", grids / "previews" / f"{slug}.json"
    if plain.exists() and scale_file.exists():
        return plain, float(json.loads(scale_file.read_text())["scale"]), False
    drawn = grids / f"{slug}.jpg"
    return (drawn, 1.0, True) if drawn.exists() else None


def _drawn(shape: tuple[int, int], rows: list[dict[str, str]], scale: float) -> np.ndarray:
    """Where draw_qc drew on the slide picture (in picture px at `scale` per preview px), a little generously."""
    mask = np.zeros(shape, np.uint8)
    at = lambda x, y: (round(x * scale), round(y * scale))  # noqa: E731
    cv2.rectangle(mask, (0, 0), at(1510, 50), 255, -1)  # the title bar
    for row in rows:
        x, y = float(row["center_x_px"]) / float(row["preview_scale_x"]), float(row["center_y_px"]) / float(row["preview_scale_y"])
        lx, ly = float(row["lattice_center_x_px"]) / float(row["preview_scale_x"]), float(row["lattice_center_y_px"]) / float(row["preview_scale_y"])
        radius = float(row["diameter_x_px"]) / float(row["preview_scale_x"]) / 2
        cv2.circle(mask, at(x, y), round(radius * scale), 255, max(2, round(12 * scale)))
        cv2.line(mask, at(lx, ly), at(x, y), 255, max(1, round(6 * scale)))
        for px, py in ((x, y), (lx, ly)):  # crosses, and the dot of the tissue found there
            cv2.circle(mask, at(px, py), max(2, round(18 * scale)), 255, -1)
        label = f"{row['core_label']} {float(row.get('provisional_tissue_fraction') or 0):.2f}"
        (width, height), baseline = cv2.getTextSize(label, LABEL_FONT, LABEL_SCALE, LABEL_THICKNESS)
        cv2.rectangle(mask, at(x - radius - 4, y - height - 5), at(x - radius + width + 6, y + baseline + 5), 255, -1)
    return mask


def _cut(image: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    """image[y0:y1, x0:x1], its edges repeated where the box runs off the picture."""
    height, width = image.shape[:2]
    part = image[max(0, y0):max(0, min(y1, height)), max(0, x0):max(0, min(x1, width))]
    if part.size == 0:
        return part
    return cv2.copyMakeBorder(part, max(0, -y0), max(0, y1 - height), max(0, -x0), max(0, x1 - width), cv2.BORDER_REPLICATE)


def _uri(bgr: np.ndarray) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(jpeg(cv2.resize(bgr, (SIZE, SIZE), interpolation=cv2.INTER_AREA), 75)).decode()


def _cores_of_slide(picture: Path, scale: float, drawn_on: bool, rows: list[dict[str, str]]) -> dict[str, str]:
    diameter = float(rows[0]["diameter_x_px"]) / float(rows[0]["preview_scale_x"]) * scale
    reduce = next((f for f in REDUCED if diameter / f >= SOURCE_PX), 1)
    image = cv2.imread(str(picture), REDUCED.get(reduce, cv2.IMREAD_COLOR))
    if image is None:
        return {}
    scale /= reduce
    mask = _drawn(image.shape[:2], rows, scale) if drawn_on else None
    found = {}
    for row in rows:
        x = float(row["center_x_px"]) / float(row["preview_scale_x"]) * scale
        y = float(row["center_y_px"]) / float(row["preview_scale_y"]) * scale
        radius = float(row["diameter_x_px"]) / float(row["preview_scale_x"]) * scale / 2
        box = (round(x - radius), round(y - radius), round(x + radius), round(y + radius))
        core = _cut(image, *box)
        if core.size == 0:
            continue
        if mask is not None:
            core = cv2.inpaint(core, _cut(mask, *box), 3, cv2.INPAINT_TELEA)
        found[row["core_label"]] = _uri(core)
    return found


def for_tma(project: Project, tma: str, stain: str) -> dict[str, str]:
    """A tiny picture of each core of one TMA and stain (core label -> JPEG data: URI); {} when Find cores kept none."""
    slides: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in _manifest(project):
        if row.get("tma") == tma and row.get("stain") == stain and row.get("preview_scale_x"):
            slides[row["slide_path"]].append(row)
    pictures = {slide: _slide_picture(project, slide) for slide in slides}
    pictures = {slide: found for slide, found in pictures.items() if found is not None}
    if not pictures:
        return {}

    def make() -> dict[str, str]:
        found: dict[str, str] = {}
        for slide, (picture, scale, drawn_on) in pictures.items():
            found.update(_cores_of_slide(picture, scale, drawn_on, slides[slide]))
        return found

    inputs = [project.input("core_manifest"), *(picture for picture, _, _ in pictures.values())]
    return cache.memo(("core placeholders", str(project.root), tma, stain), inputs, make)
