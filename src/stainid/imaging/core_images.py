"""Core images of image-set and whole-slide projects: each image (or tissue region of a slide) as the analysis-scale,
brown-on-blue picture every step reads, made from the lab's own file without copying it.

An image that is already at the analysis scale and in DAB is used where it lies (its path goes straight into the core
image table). Any other becomes a small *link* file, `data/images/links/<id>.json`, saying how to make the picture: the
source file, the box on it (level-0 pixels), the enlargement to the analysis scale and the colour recipe
(`stainid.imaging.chromogens`). Every step opens core images with `open_image` / `read_bgr`, which make the picture from
a link; with `imaging.rescale: on_import` the picture is kept in the project's cache (`data/cache/images/<id>.png`,
written only by `_cached_picture`) and remade whenever it is missing or older than its link, with `on_the_fly` it is
made every time it is read and nothing is stored. Source files are never changed or copied.
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from stainid.imaging.chromogens import Recipe, to_analysis_rgb

Image.MAX_IMAGE_PIXELS = None
LINKS = Path("data") / "images" / "links"
CACHE_KIND = "images"
_LOCKS: dict[str, threading.Lock] = {}
_GUARD = threading.Lock()


def is_link(path: Path | str) -> bool:
    return str(path).endswith(".json")


def link_path(root: Path, image_id: str) -> Path:
    return Path(root) / LINKS / f"{safe_name(image_id)}.json"


def safe_name(text: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in str(text))


def write_link(path: Path, source: Path | str, box: tuple[int, int, int, int], scale: float, recipe: Recipe, cache: bool,
               size: tuple[int, int]) -> Path:
    """Say how a core image is made; `size` is the picture's (width, height) at the analysis scale."""
    data = {"stainid_core_image": 1, "source": str(source), "box": [int(v) for v in box], "scale": float(scale), "recipe": recipe.as_dict(),
            "cache": bool(cache), "width_px": int(size[0]), "height_px": int(size[1])}
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, indent=1)
    if not path.exists() or path.read_text() != text:  # unchanged links keep their time, so their cached pictures stay fresh
        path.write_text(text)
    return path


def read_link(path: Path) -> dict:
    data = json.loads(Path(path).read_text())
    if not data.get("stainid_core_image"):
        raise ValueError(f"{path} is not a stainID core image link")
    return data


def cache_folder(link: Path) -> Path:
    """`data/cache/images` of the project the link belongs to (links live in `data/images/links`)."""
    return Path(link).resolve().parents[2] / "cache" / CACHE_KIND


def render(link: Path) -> np.ndarray:
    """The analysis picture of a link (RGB)."""
    from stainid.imaging.sources import open_source

    data = read_link(link)
    source = open_source(data["source"])
    rgb = source.region(tuple(data["box"]), float(data["scale"]))
    width, height = int(data["width_px"]), int(data["height_px"])
    if (rgb.shape[1], rgb.shape[0]) != (width, height):
        rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_LINEAR)
    return to_analysis_rgb(rgb, Recipe.from_dict(data["recipe"]))


def _cached_picture(link: Path) -> Path | None:
    """The cached picture of a link (made now if it is missing or older than the link); None for on-the-fly links. The
    only place that writes the image cache."""
    data = read_link(link)
    if not data.get("cache"):
        return None
    target = cache_folder(link) / f"{Path(link).stem}.png"
    with _GUARD:
        lock = _LOCKS.setdefault(str(target), threading.Lock())
    with lock:
        if not _fresh(target, link):
            target.parent.mkdir(parents=True, exist_ok=True)
            rgb = render(link)
            with tempfile.NamedTemporaryFile(dir=target.parent, prefix=f".{target.name}.", suffix=".png", delete=False) as handle:
                name = handle.name
            if not cv2.imwrite(name, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 1]):
                raise OSError(f"could not write {target}")
            os.replace(name, target)
    return target


def _fresh(picture: Path, link: Path) -> bool:
    try:
        return picture.stat().st_mtime_ns >= Path(link).stat().st_mtime_ns
    except FileNotFoundError:
        return False


def prepare(link: Path) -> None:
    """Make (or refresh) the cached picture of an on-import link now, so later steps open it at once."""
    _cached_picture(link)


def open_image(path: Path | str) -> Image.Image:
    """A core image as a PIL image: the file itself, or the picture a link makes (use as `with open_image(p) as image`)."""
    path = Path(path)
    if not is_link(path):
        return Image.open(path)
    cached = _cached_picture(path)
    return Image.open(cached) if cached else Image.fromarray(render(path))


def read_bgr(path: Path | str, flag: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    """A core image as OpenCV reads it (BGR; `flag` may ask for a reduced decode); None if it cannot be read."""
    path = Path(path)
    if not is_link(path):
        return cv2.imread(str(path), flag)
    cached = _cached_picture(path)
    if cached:
        return cv2.imread(str(cached), flag)
    bgr = cv2.cvtColor(render(path), cv2.COLOR_RGB2BGR)
    reduce = {cv2.IMREAD_REDUCED_COLOR_2: 2, cv2.IMREAD_REDUCED_COLOR_4: 4, cv2.IMREAD_REDUCED_COLOR_8: 8}.get(flag, 1)
    return cv2.resize(bgr, (max(1, bgr.shape[1] // reduce), max(1, bgr.shape[0] // reduce)), interpolation=cv2.INTER_AREA) if reduce > 1 else bgr
