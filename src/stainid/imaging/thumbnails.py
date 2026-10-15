"""Small whole-core pictures (`data/cache/thumbs`), made without decoding a whole core image whenever possible.

A core image is a ~16 000 px PNG of about 300 MB: decoding one takes seconds and ~750 MB of memory. Its thumbnail base
(at most `THUMB_BASE` px across, every smaller size is made from it) is therefore written by core export while the core
is still in memory, or made from the ~2000 px picture Check core quality keeps of every core, once the outlines drawn
on that picture are taken out again (they are drawn in exact colours, without smoothing, so they can be found exactly).
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

import cv2
import numpy as np

from stainid.qc.core_qc import OUTLINE_COLOURS

THUMB_BASE = 2048


def cache_folder(project_root: Path) -> Path:
    """Where the thumbnails of a project are kept (the API's `ProjectState.cache_dir("thumbs")`)."""
    return Path(project_root) / "data" / "cache" / "thumbs"


def base_path(folder: Path, core_id: str, stain: str) -> Path:
    return Path(folder) / f"{core_id}_{stain}_base.jpg"


def write_atomic(path: Path, data: bytes) -> None:
    """Write a file so that readers see the old or the new version, never half of one (also from parallel writers)."""
    temporary = path.with_name(f"{path.name}.{os.getpid()}-{threading.get_ident()}.tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def jpeg(bgr: np.ndarray, quality: int = 85) -> bytes:
    ok, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("could not encode a JPEG")
    return encoded.tobytes()


def shrink(image: np.ndarray, size: int) -> np.ndarray:
    """The image at most `size` px across (unchanged when it is already that small)."""
    scale = size / max(image.shape[:2])
    if scale >= 1:
        return image
    width, height = max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def write_base(path: Path, bgr: np.ndarray) -> Path:
    """Save the thumbnail base of a core image from the image (or any reduced copy of it), BGR."""
    path.parent.mkdir(parents=True, exist_ok=True)
    write_atomic(path, jpeg(shrink(bgr, THUMB_BASE), 90))
    return path


def without_outlines(bgr: np.ndarray) -> np.ndarray:
    """A Check core quality picture with the outlines drawn on it painted over from the pixels around them."""
    drawn = np.zeros(bgr.shape[:2], np.uint8)
    for colour in OUTLINE_COLOURS:
        drawn |= cv2.inRange(bgr, colour, colour)
    if not drawn.any():
        return bgr
    return cv2.inpaint(bgr, drawn, 2, cv2.INPAINT_TELEA)
