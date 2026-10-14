"""A small picture of a whole slide, label included, so people can recognise a scan by what is written on it.

Most scanners keep a few small images beside the scan itself: Olympus .vsi files hold the slide overview (label and
cores) and the label alone as ordinary TIFF pages, Aperio .svs files a "macro" and a "label" page, pyramidal TIFFs a
thumbnail. These are read with tifffile, in milliseconds and without starting the slide reader. When a file has none,
the plain slide picture Find cores keeps (`qc/grids/previews/<slide>.jpg`) is used if it exists.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from stainid.slides.dearray import slide_slug

MAX_PIXELS = 3_000_000  # pages larger than this are the scan itself, not a picture of the slide
MAX_PAGES = 16


def _page_size(page) -> tuple[int, int, int]:
    shape = tuple(page.shape)
    if len(shape) == 3 and shape[-1] in (3, 4):
        return shape[0], shape[1], shape[2]
    if len(shape) == 2:
        return shape[0], shape[1], 1
    return 0, 0, 0


def embedded_picture(path: Path) -> np.ndarray | None:
    """The overview (or label) image a scan file keeps beside the scan, as RGB; None when there is none or it cannot be read."""
    try:
        import tifffile
    except ImportError:  # pragma: no cover - tifffile comes with scikit-image
        return None
    try:
        with tifffile.TiffFile(str(path)) as tiff:
            pages = []
            for index, page in enumerate(tiff.pages[:MAX_PAGES]):
                height, width, channels = _page_size(page)
                if not height or height * width > MAX_PIXELS or min(height, width) < 32:
                    continue
                text = (getattr(page, "description", "") or "").lower()
                rank = 0 if "macro" in text else 1 if "label" in text else 2
                pages.append((rank, index, page, channels))
            if not pages:
                return None
            _, _, page, channels = min(pages, key=lambda p: (p[0], p[1]))
            image = np.asarray(page.asarray())
    except Exception:  # an unreadable or unusual file simply has no picture
        return None
    if image.dtype != np.uint8:
        image = cv2.normalize(image.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    return np.ascontiguousarray(image[..., :3])


def slide_picture(path: Path, qc_dir: Path | None = None, long_side: int = 480) -> np.ndarray | None:
    """A picture of the slide no longer than `long_side`: the one in the scan file, else the Find cores preview."""
    image = embedded_picture(path)
    if image is None and qc_dir is not None:
        preview = Path(qc_dir) / "grids" / "previews" / f"{slide_slug(Path(path))}.jpg"
        bgr = cv2.imread(str(preview), cv2.IMREAD_COLOR) if preview.exists() else None
        image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if bgr is not None else None
    if image is None:
        return None
    scale = min(1.0, long_side / max(image.shape[:2]))
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else image
