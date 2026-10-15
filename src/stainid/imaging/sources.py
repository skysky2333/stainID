"""Reading image files and whole slides of any lab: plain pictures (PNG, JPEG, TIFF), whole-slide scans (.svs, .ndpi,
.tif, .mrxs, .scn, .czi) and OME-Zarr, with their size, levels and pixel size, one region at a time.

Whole slides need the `slides` extra (`pip install "stainid[slides]"`: OpenSlide, Zarr, Bio-Formats). Without it a
tiled TIFF is still read with tifffile, one level at a time, and the error for anything else says what to install.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
PICTURES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
TIFFS = {".tif", ".tiff"}
SLIDES = {".svs", ".ndpi", ".mrxs", ".scn", ".vms", ".vmu", ".bif", ".svslide", ".czi", ".vsi"}
IMAGE_SUFFIXES = PICTURES | TIFFS | SLIDES
INSTALL = 'pip install "stainid[slides]"'
WHOLE_LEVEL_LIMIT = 30_000 * 30_000  # without OpenSlide or Zarr a TIFF level is read whole: only up to this many pixels


class MissingReader(RuntimeError):
    """The file needs a reader that is not installed."""


def available() -> dict[str, bool]:
    """Which optional readers work in this installation (shown on the setup page)."""
    out = {}
    for name in ("openslide", "zarr", "tifffile", "aicsimageio"):
        try:
            __import__(name)
            out[name] = True
        except Exception:  # noqa: BLE001 - a broken install (e.g. zarr with a newer numcodecs) is as good as none
            out[name] = False
    return out


def is_zarr(path: Path) -> bool:
    path = Path(path)
    return path.suffix.lower() == ".zarr" or (path.is_dir() and any((path / name).exists() for name in (".zattrs", ".zgroup", "zarr.json")))


def is_image(path: Path) -> bool:
    return Path(path).suffix.lower() in IMAGE_SUFFIXES or is_zarr(path)


@dataclass
class Source:
    """One image or slide: level-0 size, the size of each level, and the pixel size the file states (None if it does not)."""
    path: Path
    width: int
    height: int
    levels: list[tuple[int, int]]
    pixel_size_um: float | None
    reader: str
    _handle: object = field(default=None, repr=False)

    def downsample(self, level: int) -> float:
        return self.width / self.levels[level][0]

    def best_level(self, downsample: float) -> int:
        """The smallest level that is still at least as fine as `downsample`."""
        good = [i for i in range(len(self.levels)) if self.downsample(i) <= downsample * 1.001]
        return max(good, key=self.downsample) if good else 0

    def read(self, x: int, y: int, width: int, height: int, level: int = 0) -> np.ndarray:
        """RGB pixels of a level-0 box (x, y, width, height) read at `level` (so about box / downsample pixels)."""
        return _READERS[self.reader](self, x, y, width, height, level)

    def region(self, box: tuple[int, int, int, int], scale: float) -> np.ndarray:
        """A level-0 box enlarged (scale > 1) or reduced by `scale`, read from the best level."""
        x, y, w, h = box
        target = (max(1, round(w * scale)), max(1, round(h * scale)))
        level = self.best_level(1.0 / scale) if scale < 1 else 0
        rgb = self.read(x, y, w, h, level)
        if (rgb.shape[1], rgb.shape[0]) != target:
            rgb = cv2.resize(rgb, target, interpolation=cv2.INTER_AREA if target[0] < rgb.shape[1] else cv2.INTER_LINEAR)
        return rgb

    def thumbnail(self, longest: int = 2048) -> tuple[np.ndarray, float]:
        """A small picture of the whole image and its pixels per level-0 pixel."""
        scale = min(1.0, longest / max(self.width, self.height))
        return self.region((0, 0, self.width, self.height), scale), scale


def _rgb(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=2)
    if array.dtype != np.uint8:
        top = float(array.max()) if array.size else 1.0
        array = (array.astype(np.float32) * (255.0 / (65535.0 if top > 255 else 255.0))).clip(0, 255).astype(np.uint8)
    return np.ascontiguousarray(array[..., :3])


# --------------------------------------------------------------------------- plain pictures

def _open_picture(path: Path) -> Source:
    with Image.open(path) as image:
        width, height = image.size
        info = image.info
    dpi = info.get("dpi")
    pixel = 25400.0 / float(dpi[0]) if dpi and float(dpi[0]) > 1000 else None  # only a real microscope resolution, not 72/300 dpi
    return Source(Path(path), width, height, [(width, height)], pixel, "picture")


def _read_picture(source: Source, x: int, y: int, width: int, height: int, level: int) -> np.ndarray:
    with Image.open(source.path) as image:
        return _rgb(np.asarray(image.crop((x, y, x + width, y + height)).convert("RGB")))


# --------------------------------------------------------------------------- OpenSlide

def _openslide_pixel(properties) -> float | None:
    for key in ("openslide.mpp-x", "aperio.MPP", "tiff.XResolution"):
        value = properties.get(key)
        if value:
            try:
                number = float(value)
            except ValueError:
                continue
            if key == "tiff.XResolution":
                unit = properties.get("tiff.ResolutionUnit", "")
                number = {"centimeter": 1e4 / number, "inch": 25400 / number}.get(unit, 0)
            if 0.01 < number < 50:
                return number
    return None


def _open_openslide(path: Path) -> Source:
    import openslide

    slide = openslide.OpenSlide(str(path))
    width, height = slide.dimensions
    return Source(Path(path), width, height, list(slide.level_dimensions), _openslide_pixel(slide.properties), "openslide", slide)


def _read_openslide(source: Source, x: int, y: int, width: int, height: int, level: int) -> np.ndarray:
    down = source.downsample(level)
    size = (max(1, round(width / down)), max(1, round(height / down)))
    region = source._handle.read_region((int(x), int(y)), level, size)
    rgba = np.asarray(region)
    rgb = rgba[..., :3].copy()
    rgb[rgba[..., 3] == 0] = 255  # outside the scanned area: white glass
    return rgb


# --------------------------------------------------------------------------- TIFF without OpenSlide

def _tiff_pixel(page) -> float | None:
    description = str(getattr(page, "description", "") or "")
    found = re.search(r"MPP\s*=\s*([0-9.]+)", description)
    if found:
        return float(found.group(1))
    try:
        if "PhysicalSizeX" in description:
            match = re.search(r'PhysicalSizeX="([0-9.eE+-]+)"', description)
            unit = re.search(r'PhysicalSizeXUnit="([^"]+)"', description)
            if match:
                value = float(match.group(1))
                return value * {"nm": 1e-3, "mm": 1e3, "µm": 1.0, "um": 1.0}.get(unit.group(1) if unit else "µm", 1.0)
        tags = page.tags
        if "XResolution" in tags:
            numerator, denominator = tags["XResolution"].value
            unit = tags["ResolutionUnit"].value if "ResolutionUnit" in tags else 2
            per_unit = numerator / denominator if denominator else 0
            if per_unit:
                um = {3: 1e4, 2: 25400.0}.get(int(unit), 0) / per_unit
                if 0.01 < um < 50:
                    return um
    except Exception:  # noqa: BLE001 - odd tags mean "not stated"
        return None
    return None


def _open_tiff(path: Path) -> Source:
    import tifffile

    handle = tifffile.TiffFile(str(path))
    series = handle.series[0]
    levels = [(int(level.shape[level.axes.index("X")]), int(level.shape[level.axes.index("Y")])) for level in series.levels]
    width, height = levels[0]
    return Source(Path(path), width, height, levels, _tiff_pixel(handle.pages[0]), "tiff", handle)


def _tiff_level(source: Source, level: int) -> np.ndarray:
    cache = source.__dict__.setdefault("_levels", {})
    if level not in cache:
        series = source._handle.series[0]
        try:
            import zarr

            store = series.aszarr(level=level)
            cache[level] = zarr.open(store, mode="r")
        except Exception:  # noqa: BLE001 - no working zarr: read the whole level, if it is small enough
            w, h = source.levels[level]
            if w * h > WHOLE_LEVEL_LIMIT:
                raise MissingReader(f"{source.path.name} is too large to read without OpenSlide or Zarr: {INSTALL}") from None
            cache[level] = series.levels[level].asarray()
    return cache[level]


def _read_tiff(source: Source, x: int, y: int, width: int, height: int, level: int) -> np.ndarray:
    down = source.downsample(level)
    array = _tiff_level(source, level)
    x0, y0 = int(x / down), int(y / down)
    x1, y1 = x0 + max(1, round(width / down)), y0 + max(1, round(height / down))
    axes = source._handle.series[0].levels[level].axes
    index = tuple(slice(y0, y1) if a == "Y" else slice(x0, x1) if a == "X" else slice(None) if a == "S" else 0 for a in axes)
    return _pad(_rgb(np.asarray(array[index])), y1 - y0, x1 - x0)


def _pad(rgb: np.ndarray, height: int, width: int) -> np.ndarray:
    """A box that runs past the image edge is filled with white glass."""
    if rgb.shape[0] == height and rgb.shape[1] == width:
        return rgb
    out = np.full((height, width, 3), 255, np.uint8)
    out[:rgb.shape[0], :rgb.shape[1]] = rgb[:height, :width]
    return out


# --------------------------------------------------------------------------- OME-Zarr

def _open_zarr(path: Path) -> Source:
    try:
        import zarr
    except Exception as error:  # noqa: BLE001
        raise MissingReader(f"OME-Zarr needs a working Zarr: {INSTALL} ({error})") from None
    group = zarr.open(str(path), mode="r")
    attributes = dict(group.attrs)
    multiscales = (attributes.get("multiscales") or attributes.get("ome", {}).get("multiscales") or [{}])[0]
    datasets = multiscales.get("datasets") or [{"path": "0"}]
    axes = [a["name"] if isinstance(a, dict) else a for a in multiscales.get("axes", ["t", "c", "z", "y", "x"])]
    arrays = [group[d["path"]] for d in datasets]
    levels = [(int(a.shape[axes.index("x")]), int(a.shape[axes.index("y")])) for a in arrays]
    pixel = None
    for transform in datasets[0].get("coordinateTransformations", []):
        if transform.get("type") == "scale":
            scale = transform["scale"][axes.index("x")]
            unit = next((a.get("unit") for a in multiscales.get("axes", []) if isinstance(a, dict) and a.get("name") == "x"), "micrometer")
            pixel = float(scale) * {"nanometer": 1e-3, "millimeter": 1e3}.get(unit or "micrometer", 1.0)
    source = Source(Path(path), levels[0][0], levels[0][1], levels, pixel, "zarr", arrays)
    source.__dict__["_axes"] = axes
    return source


def _read_zarr(source: Source, x: int, y: int, width: int, height: int, level: int) -> np.ndarray:
    down = source.downsample(level)
    axes = source.__dict__["_axes"]
    x0, y0 = int(x / down), int(y / down)
    x1, y1 = x0 + max(1, round(width / down)), y0 + max(1, round(height / down))
    index = tuple(slice(y0, y1) if a == "y" else slice(x0, x1) if a == "x" else slice(0, 3) if a == "c" else 0 for a in axes)
    array = np.asarray(source._handle[level][index])
    if "c" in axes and axes.index("c") < axes.index("y"):
        array = np.moveaxis(array, 0, -1)  # (c, y, x) -> (y, x, c)
    return _pad(_rgb(array), y1 - y0, x1 - x0)


# --------------------------------------------------------------------------- Bio-Formats (czi, vsi)

def _open_bioformats(path: Path) -> Source:
    try:
        from aicsimageio import AICSImage
    except Exception as error:  # noqa: BLE001
        raise MissingReader(f"{Path(path).suffix} slides need Bio-Formats: {INSTALL} ({error})") from None
    image = AICSImage(str(path))
    width, height = int(image.dims.X), int(image.dims.Y)
    pixel = image.physical_pixel_sizes.X
    return Source(Path(path), width, height, [(width, height)], float(pixel) if pixel else None, "bioformats", image)


def _read_bioformats(source: Source, x: int, y: int, width: int, height: int, level: int) -> np.ndarray:
    data = source._handle.get_image_dask_data("YXS")[y:y + height, x:x + width]
    return _pad(_rgb(np.asarray(data.compute())), height, width)


_READERS = {"picture": _read_picture, "openslide": _read_openslide, "tiff": _read_tiff, "zarr": _read_zarr, "bioformats": _read_bioformats}


def open_source(path: Path | str) -> Source:
    """Open an image or slide with the best reader installed."""
    path = Path(path)
    suffix = path.suffix.lower()
    if is_zarr(path):
        return _open_zarr(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if suffix in PICTURES:
        return _open_picture(path)
    if suffix in {".czi", ".vsi"}:
        return _open_bioformats(path)
    if available()["openslide"]:
        import openslide

        if openslide.OpenSlide.detect_format(str(path)):
            return _open_openslide(path)
    if suffix in TIFFS or suffix in {".svs", ".ndpi"}:
        try:
            return _open_tiff(path)
        except MissingReader:
            raise
        except Exception as error:  # noqa: BLE001
            if suffix in TIFFS:
                try:
                    return _open_picture(path)
                except Exception:  # noqa: BLE001
                    pass
            raise MissingReader(f"{path.name} could not be read ({error}); whole slides read best with OpenSlide: {INSTALL}") from None
    raise MissingReader(f"{path.name}: no reader for {suffix or 'this file'} is installed; {INSTALL}")


def describe(path: Path | str) -> dict:
    """Size, levels and pixel size of an image or slide, for the setup page (an `error` in plain words if unreadable)."""
    try:
        source = open_source(path)
    except (MissingReader, FileNotFoundError, OSError, ValueError) as error:
        return {"path": str(path), "readable": False, "error": str(error)}
    return {"path": str(path), "readable": True, "width_px": source.width, "height_px": source.height, "levels": len(source.levels),
            "pixel_size_um": source.pixel_size_um, "reader": source.reader}
