from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from aicsimageio import AICSImage


@dataclass(frozen=True)
class SceneInfo:
    index: int
    name: str
    width: int
    height: int


@dataclass(frozen=True)
class SlidePreview:
    path: Path
    rgb: np.ndarray
    full_scene: SceneInfo
    preview_scene: SceneInfo
    scale_x: float
    scale_y: float
    pixel_width_um: float
    pixel_height_um: float


def _scene_info(image: AICSImage) -> list[SceneInfo]:
    scenes = []
    for index, name in enumerate(image.scenes):
        image.set_scene(name)
        scenes.append(SceneInfo(index, str(name), int(image.dims.X), int(image.dims.Y)))
    return scenes


def _select_scenes(scenes: list[SceneInfo], target_long_side: int) -> tuple[SceneInfo, SceneInfo | None]:
    """The full-resolution scene and the largest stored preview no longer than `target_long_side` (None if the file has none)."""
    full = max(scenes, key=lambda scene: scene.width * scene.height)
    full_aspect = full.width / full.height
    pyramid = [
        scene
        for scene in scenes
        if max(scene.width, scene.height) <= target_long_side
        and abs(np.log((scene.width / scene.height) / full_aspect)) < 0.02
    ]
    return full, max(pyramid, key=lambda scene: scene.width * scene.height) if pyramid else None


def _to_rgb(image: AICSImage, step: int = 1) -> np.ndarray:
    """RGB of the current scene; `step` > 1 reads every step-th pixel (a preview for files without a stored one)."""
    if int(image.dims.S) >= 3:
        array = image.get_image_dask_data("YXS", T=0, Z=0, C=0)
    elif int(image.dims.C) >= 3:
        array = image.get_image_dask_data("YXC", T=0, Z=0)
    else:
        array = image.get_image_dask_data("YX", T=0, Z=0, C=0)[..., None].repeat(3, axis=2)
    array = np.asarray(array[::step, ::step].compute())
    if array.dtype == np.uint16:
        array = (array / 257).astype(np.uint8)
    elif np.issubdtype(array.dtype, np.floating):
        scale = 255.0 if float(array.max()) <= 1.0 else 1.0
        array = np.clip(array * scale, 0, 255).astype(np.uint8)
    elif array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return array[..., :3]


def read_slide_preview(path: Path, target_long_side: int = 4500) -> SlidePreview:
    image = AICSImage(str(path))
    scenes = _scene_info(image)
    full, preview = _select_scenes(scenes, target_long_side)

    image.set_scene(image.scenes[full.index])
    pixel_sizes = image.physical_pixel_sizes
    if pixel_sizes.X is None or pixel_sizes.Y is None:
        raise ValueError(f"Full-resolution scene has no pixel calibration: {path}")

    if preview is None:
        step = int(np.ceil(max(full.width, full.height) / target_long_side))
        rgb = _to_rgb(image, step)
        preview = SceneInfo(full.index, full.name, rgb.shape[1], rgb.shape[0])
    else:
        image.set_scene(image.scenes[preview.index])
        rgb = _to_rgb(image)
    return SlidePreview(
        path=path,
        rgb=rgb,
        full_scene=full,
        preview_scene=preview,
        scale_x=full.width / preview.width,
        scale_y=full.height / preview.height,
        pixel_width_um=float(pixel_sizes.X),
        pixel_height_um=float(pixel_sizes.Y),
    )
