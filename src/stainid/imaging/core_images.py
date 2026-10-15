"""Core images of image-set and whole-slide projects: each image (or tissue region of a slide) as the analysis-scale,
brown-on-blue picture every step reads, made from the lab's own file without copying it.

An image that is already at the analysis scale and in DAB is used where it lies (its path goes straight into the core
image table). Any other becomes a small *link* file, `data/images/links/<id>.json`, saying how to make the picture: the
source file, the box on it (level-0 pixels), the enlargement to the analysis scale and the colour recipe
(`stainid.imaging.chromogens`). Every step opens core images with `open_image` / `read_bgr`, which make the picture from
a link. With `imaging.rescale: on_import` the picture is kept as a cache file (kind `images`, through
`stainid.disk_cache`, written only by `_kept_picture`): made again whenever it is missing (the cache may delete it to
stay within its size limit) or older than its link. With `on_the_fly` it is made every time it is read and nothing is
stored. The link files are study data, not cache; source files are never changed or copied.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from stainid import disk_cache
from stainid.imaging.chromogens import Recipe, to_analysis_rgb

Image.MAX_IMAGE_PIXELS = None
LINKS = Path("data") / "images" / "links"
KIND = "images"  # in disk_cache.KINDS
_LOCKS: dict[str, threading.Lock] = {}
_GUARD = threading.Lock()


def is_link(path: Path | str) -> bool:
    return str(path).endswith(".json")


def link_path(root: Path, image_id: str) -> Path:
    return Path(root) / LINKS / f"{safe_name(image_id)}.json"


def safe_name(text: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in str(text))


def write_link(path: Path, source: Path | str, box: tuple[int, int, int, int], scale: float, recipe: Recipe, keep: bool,
               size: tuple[int, int]) -> Path:
    """Say how a core image is made; `size` is the picture's (width, height) at the analysis scale; `keep`: keep the
    picture once made (on_import) instead of making it at every read (on_the_fly)."""
    data = {"stainid_core_image": 1, "source": str(source), "box": [int(v) for v in box], "scale": float(scale), "recipe": recipe.as_dict(),
            "keep_picture": bool(keep), "width_px": int(size[0]), "height_px": int(size[1])}
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, indent=1)
    if not path.exists() or path.read_text() != text:  # unchanged links keep their time, so their kept pictures stay fresh
        path.write_text(text)
    return path


def read_link(path: Path) -> dict:
    data = json.loads(Path(path).read_text())
    if not data.get("stainid_core_image"):
        raise ValueError(f"{path} is not a stainID core image link")
    return data


def project_root(link: Path) -> Path:
    """The project a link belongs to (links live in `<project>/data/images/links`)."""
    return Path(link).resolve().parents[3]


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


def _kept_picture(link: Path) -> Path | None:
    """The kept picture of an on-import link, made now if it is missing or older than the link; None for on-the-fly
    links. The only place that writes this cache."""
    if not read_link(link).get("keep_picture"):
        return None
    target = disk_cache.path(project_root(link), KIND, f"{Path(link).stem}.png")
    with _GUARD:
        lock = _LOCKS.setdefault(str(target), threading.Lock())
    with lock:
        if not _fresh(target, link):
            ok, encoded = cv2.imencode(".png", cv2.cvtColor(render(link), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 1])
            if not ok:
                raise OSError(f"could not encode {target.name}")
            disk_cache.write(target, encoded.tobytes())
        else:
            disk_cache.used(target)
    return target


def _fresh(picture: Path, link: Path) -> bool:
    try:
        return picture.stat().st_mtime_ns >= Path(link).stat().st_mtime_ns
    except FileNotFoundError:
        return False


def prepare(link: Path) -> None:
    """Make (or refresh) the kept picture of an on-import link now, so later steps open it at once."""
    _kept_picture(link)


def _from_kept(link: Path, read):
    """`read(path)` of the kept picture; made again if the cache removed it in between (None for on-the-fly links)."""
    for attempt in (1, 2):
        kept = _kept_picture(link)
        if kept is None:
            return None
        try:
            return read(kept)
        except FileNotFoundError:
            if attempt == 2:
                raise
    return None


def open_image(path: Path | str) -> Image.Image:
    """A core image as a PIL image: the file itself, or the picture a link makes (use as `with open_image(p) as image`)."""
    path = Path(path)
    if not is_link(path):
        return Image.open(path)
    image = _from_kept(path, Image.open)
    return image if image is not None else Image.fromarray(render(path))


def _imread(flag: int):
    def read(path: Path) -> np.ndarray:
        bgr = cv2.imread(str(path), flag)
        if bgr is None and not path.exists():
            raise FileNotFoundError(path)
        return bgr
    return read


def read_bgr(path: Path | str, flag: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    """A core image as OpenCV reads it (BGR; `flag` may ask for a reduced decode); None if it cannot be read."""
    path = Path(path)
    if not is_link(path):
        return cv2.imread(str(path), flag)
    bgr = _from_kept(path, _imread(flag))
    if bgr is not None:
        return bgr
    bgr = cv2.cvtColor(render(path), cv2.COLOR_RGB2BGR)
    reduce = {cv2.IMREAD_REDUCED_COLOR_2: 2, cv2.IMREAD_REDUCED_COLOR_4: 4, cv2.IMREAD_REDUCED_COLOR_8: 8}.get(flag, 1)
    return cv2.resize(bgr, (max(1, bgr.shape[1] // reduce), max(1, bgr.shape[0] // reduce)), interpolation=cv2.INTER_AREA) if reduce > 1 else bgr
