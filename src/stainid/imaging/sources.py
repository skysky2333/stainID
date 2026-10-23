"""Reading image files and whole slides of any lab: plain pictures (PNG, JPEG, TIFF), whole-slide scans (.svs, .ndpi,
.tif, .mrxs, .scn, .czi), OME-Zarr, and plain Zarr v2 arrays (e.g. the Tangle-Tracer slides: one RGB array in 5000 px
chunks, each a JPEG-XR picture), with their size, levels and pixel size, one region at a time.

Whole slides need the `slides` extra (`pip install "stainid[slides]"`: OpenSlide, Zarr, Bio-Formats, imagecodecs).
Without it a tiled TIFF is still read with tifffile, one level at a time, and the error for anything else says what to
install. A plain Zarr array is read chunk by chunk without the zarr package (imagecodecs decodes the chunks), so it
works with any numcodecs.
"""
from __future__ import annotations

import json
import math
import re
import threading
from concurrent.futures import ThreadPoolExecutor
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
    for name in ("openslide", "zarr", "tifffile", "aicsimageio", "imagecodecs"):
        try:
            __import__(name)
            out[name] = True
        except Exception:  # noqa: BLE001 - a broken install (e.g. zarr with a newer numcodecs) is as good as none
            out[name] = False
    return out


def is_zarr(path: Path) -> bool:
    path = Path(path)
    return path.suffix.lower() == ".zarr" or (path.is_dir() and any((path / name).exists() for name in (".zattrs", ".zgroup", ".zarray", "zarr.json")))


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


# --------------------------------------------------------------------------- plain Zarr v2 arrays (Tangle-Tracer)

ARRAY_MIN_LEVEL_PX = 256  # a plain array's reduced levels (made while reading) go down to about this size
ARRAY_THREADS = 4  # chunks decoded at once (a 5000 px JPEG-XR chunk takes ~0.3 s and 75 MB)
# An RGB array whose fill value is 0 stores the unscanned parts of its chunks as black: pixels this dark are glass (white);
# the darkest stain in brightfield is far lighter.
BLACK_GLASS = 10


def _chunk_decoder(compressor: dict | None):
    """A function from a chunk file's bytes to its pixels (an array, or bytes to be shaped)."""
    if not compressor:
        return lambda data: data
    name = str(compressor.get("id", ""))
    if name.startswith("imagecodecs_"):
        try:
            import imagecodecs
        except Exception as error:  # noqa: BLE001
            raise MissingReader(f"Zarr chunks in {name[12:].upper()} need imagecodecs: {INSTALL} ({error})") from None
        decode = getattr(imagecodecs, f"{name[12:]}_decode", None)
        codec = getattr(imagecodecs, name[12:].upper(), None)
        if decode is None or (codec is not None and not getattr(codec, "available", True)):
            raise MissingReader(f"this imagecodecs cannot decode {name[12:].upper()} Zarr chunks: {INSTALL}")
        return decode
    try:
        import numcodecs

        return numcodecs.get_codec(dict(compressor)).decode
    except Exception as error:  # noqa: BLE001
        raise MissingReader(f"Zarr chunks compressed with {name or 'an unknown codec'} need numcodecs: {INSTALL} ({error})") from None


def _open_zarr_array(path: Path) -> Source:
    """A plain Zarr v2 array (`.zarray`, no OME metadata): (y, x), (y, x, c) or (c, y, x) with up to 4 channels, one level.
    The reader makes reduced levels as it reads (each chunk decoded and shrunk), so overviews need little memory."""
    meta = json.loads((Path(path) / ".zarray").read_text())
    shape, chunks = [int(v) for v in meta["shape"]], [int(v) for v in meta["chunks"]]
    if meta.get("filters"):
        raise MissingReader(f"{Path(path).name}: Zarr filters are not supported ({meta['filters']})")
    if len(shape) == 2:
        axes = "yx"
    elif len(shape) == 3 and shape[2] <= 4:
        axes = "yxc"
    elif len(shape) == 3 and shape[0] <= 4:
        axes = "cyx"
    else:
        raise ValueError(f"{Path(path).name}: a Zarr array of shape {shape} is not an image")
    height, width = shape[axes.index("y")], shape[axes.index("x")]
    levels, factor = [(width, height)], 2
    while min(width, height) / factor >= ARRAY_MIN_LEVEL_PX:
        levels.append((math.ceil(width / factor), math.ceil(height / factor)))
        factor *= 2
    pixel = None
    attributes = Path(path) / ".zattrs"
    if attributes.exists():
        value = (json.loads(attributes.read_text()) or {}).get("pixel_size_um")
        pixel = float(value) if isinstance(value, (int, float)) and value > 0 else None
    source = Source(Path(path), width, height, levels, pixel, "zarr_array", _chunk_decoder(meta.get("compressor")))
    dtype = np.dtype(meta["dtype"])
    source.__dict__.update(_axes=axes, _shape=shape, _chunks=chunks, _dtype=dtype, _order=meta.get("order", "C"),
                           _separator=meta.get("dimension_separator") or ".",
                           _black_glass=meta.get("fill_value") == 0 and dtype == np.uint8 and axes != "yx")
    return source


def _array_chunk(source: Source, cy: int, cx: int) -> np.ndarray | None:
    """One chunk as (y, x, c) pixels (cut to the array's edge); None when it is not stored (empty glass)."""
    axes, chunks, sep = source.__dict__["_axes"], source.__dict__["_chunks"], source.__dict__["_separator"]
    key = {"yx": (cy, cx), "yxc": (cy, cx, 0), "cyx": (0, cy, cx)}[axes]
    file = source.path.joinpath(*map(str, key)) if sep == "/" else source.path / sep.join(map(str, key))
    try:
        data = file.read_bytes()
    except FileNotFoundError:
        return None
    array = source._handle(data)
    if not isinstance(array, np.ndarray) or array.ndim == 1:  # raw or byte-compressed (zlib, blosc …): shape the bytes
        raw = array.tobytes() if isinstance(array, np.ndarray) else bytes(array)
        array = np.frombuffer(raw, dtype=source.__dict__["_dtype"]).reshape(chunks, order=source.__dict__["_order"])
    if axes == "cyx" and array.ndim == 3 and array.shape[0] <= 4 < array.shape[-1]:
        array = np.moveaxis(array, 0, -1)
    return array


def _read_zarr_array(source: Source, x: int, y: int, width: int, height: int, level: int) -> np.ndarray:
    down = 2 ** level
    out_w, out_h = max(1, round(width / down)), max(1, round(height / down))
    out = np.full((out_h, out_w, 3), 255, np.uint8)  # chunks that are not stored are empty glass
    chunks = source.__dict__["_chunks"]
    ch, cw = (chunks[1], chunks[2]) if source.__dict__["_axes"] == "cyx" else (chunks[0], chunks[1])
    x0, y0, x1, y1 = max(0, x), max(0, y), min(source.width, x + width), min(source.height, y + height)
    jobs = [(cy, cx) for cy in range(y0 // ch, (y1 - 1) // ch + 1) for cx in range(x0 // cw, (x1 - 1) // cw + 1)] if x1 > x0 and y1 > y0 else []
    blank_is_black = source.__dict__["_black_glass"]

    def piece(job: tuple[int, int]):
        cy, cx = job
        top, left = cy * ch, cx * cw
        a0, a1, b0, b1 = max(y0, top), min(y1, top + ch), max(x0, left), min(x1, left + cw)
        oy0, oy1 = round((a0 - y) / down), round((a1 - y) / down)
        ox0, ox1 = round((b0 - x) / down), round((b1 - x) / down)
        oy1, ox1 = min(out_h, max(oy1, oy0 + 1)), min(out_w, max(ox1, ox0 + 1))
        if oy0 >= out_h or ox0 >= out_w:
            return None
        chunk = _array_chunk(source, cy, cx)
        if chunk is None:
            return None
        rgb = _pad(_rgb(chunk[a0 - top:a1 - top, b0 - left:b1 - left]), a1 - a0, b1 - b0)
        if blank_is_black:
            rgb = rgb.copy() if not rgb.flags.writeable else rgb
            rgb[rgb.max(axis=2) <= BLACK_GLASS] = 255
        if rgb.shape[:2] != (oy1 - oy0, ox1 - ox0):
            rgb = cv2.resize(rgb, (ox1 - ox0, oy1 - oy0), interpolation=cv2.INTER_AREA if down > 1 else cv2.INTER_LINEAR)
        return oy0, ox0, rgb

    with ThreadPoolExecutor(max_workers=min(ARRAY_THREADS, max(1, len(jobs)))) as pool:
        for found in pool.map(piece, jobs):
            if found is not None:
                oy0, ox0, rgb = found
                out[oy0:oy0 + rgb.shape[0], ox0:ox0 + rgb.shape[1]] = rgb
    return out


# --------------------------------------------------------------------------- Bio-Formats (czi, vsi)

@dataclass(frozen=True)
class Series:
    """One image series of a Bio-Formats file (an Olympus .vsi holds the label, the slide overview, the macro picture
    and each scanned area with its reduced copies as series of their own)."""
    index: int
    name: str
    width: int
    height: int
    pixel_size_um: float | None


def scan_series(series: list[Series]) -> list[Series]:
    """The series of the scan itself, finest first: the largest of the series with the finest pixels (not the label, the
    overview or the macro picture; of a slide scanned in two areas, the larger, e.g. not a strip over the marker ink), then its reduced copies (`<name> resolution N`, as Bio-Formats names a .vsi pyramid), largest first.
    The study's TMA steps choose the same series (`stainid.slides.vsi`: the largest scene)."""
    if not series:
        raise ValueError("the file holds no image")
    calibrated = [s for s in series if s.pixel_size_um]
    if calibrated:  # the finest pixels are the scan's (the label and overview are camera pictures at ~2.7 µm)
        finest = min(s.pixel_size_um for s in calibrated)
        calibrated = [s for s in calibrated if s.pixel_size_um <= finest * 1.05]
    full = max(calibrated or list(series), key=lambda s: (s.width * s.height, -s.index))
    reduced = [s for s in series if s.name.startswith(f"{full.name} resolution ") and s.width < full.width]
    return [full, *sorted(reduced, key=lambda s: s.width * s.height, reverse=True)]


class _BioHandle:
    """An open Bio-Formats reader and the series of its levels. One reader per file and process, shared by every Source
    of that file (opening one takes about a second); its reads are serialised (a reader is not thread-safe)."""

    def __init__(self, path: Path):
        try:
            from aicsimageio.readers.bioformats_reader import BioFile
        except Exception as error:  # noqa: BLE001
            raise MissingReader(f"{Path(path).suffix} slides need Bio-Formats: {INSTALL} ({error})") from None
        self.lock = threading.Lock()
        self.file = BioFile(str(path), series=0, meta=True)
        reader = self.file._r
        store = reader.getMetadataStore()
        found = []
        for index in range(int(reader.getSeriesCount())):
            self.file.set_series(index)
            *_, height, width, _ = self.file.core_meta.shape
            size = store.getPixelsPhysicalSizeX(index)
            pixel = float(size.value()) if size is not None and size.value() is not None else None
            name = store.getImageName(index)
            found.append(Series(index, str(name) if name is not None else f"series {index}", int(width), int(height), pixel))
        self.levels = scan_series(found)
        self.current = -1

    def read(self, level: int, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
        series = self.levels[level]
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(series.width, x1), min(series.height, y1)
        if x1 <= x0 or y1 <= y0:
            return np.full((0, 0, 3), 255, np.uint8)
        with self.lock:
            if self.current != series.index:
                self.file.set_series(series.index)
                self.current = series.index
            return _rgb(np.asarray(self.file._get_plane(y=slice(y0, y1), x=slice(x0, x1))))


_BIO_HANDLES: dict[str, tuple[int, _BioHandle]] = {}
_BIO_GUARD = threading.Lock()


def _bio_handle(path: Path) -> _BioHandle:
    key, stamp = str(Path(path).resolve()), Path(path).stat().st_mtime_ns
    with _BIO_GUARD:
        known = _BIO_HANDLES.get(key)
        if known is None or known[0] != stamp:
            _BIO_HANDLES[key] = (stamp, _BioHandle(Path(path)))
        return _BIO_HANDLES[key][1]


def _open_bioformats(path: Path) -> Source:
    handle = _bio_handle(path)
    full = handle.levels[0]
    return Source(Path(path), full.width, full.height, [(s.width, s.height) for s in handle.levels], full.pixel_size_um, "bioformats", handle)


def _read_bioformats(source: Source, x: int, y: int, width: int, height: int, level: int) -> np.ndarray:
    down = source.downsample(level)
    x0, y0 = int(x / down), int(y / down)
    x1, y1 = x0 + max(1, round(width / down)), y0 + max(1, round(height / down))
    rgb = source._handle.read(level, x0, y0, x1, y1)
    if x0 >= 0 and y0 >= 0:
        return _pad(rgb, y1 - y0, x1 - x0)
    out = np.full((y1 - y0, x1 - x0, 3), 255, np.uint8)  # a box starting before the image: glass on that side
    out[max(0, -y0):max(0, -y0) + rgb.shape[0], max(0, -x0):max(0, -x0) + rgb.shape[1]] = rgb
    return out


_READERS = {"picture": _read_picture, "openslide": _read_openslide, "tiff": _read_tiff, "zarr": _read_zarr, "zarr_array": _read_zarr_array,
            "bioformats": _read_bioformats}


def open_source(path: Path | str) -> Source:
    """Open an image or slide with the best reader installed."""
    path = Path(path)
    suffix = path.suffix.lower()
    if is_zarr(path):
        if not path.exists():
            raise FileNotFoundError(path)
        return _open_zarr_array(path) if (path / ".zarray").exists() else _open_zarr(path)
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
