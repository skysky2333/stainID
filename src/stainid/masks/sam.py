from __future__ import annotations

from pathlib import Path

import numpy as np


class WindowedSam:
    """Point- and box-prompted SAM on native-resolution windows of a large field."""

    def __init__(self, checkpoint: Path, device: str = "cpu", window: int = 1024, stride: int = 768):
        import torch
        from segment_anything import SamPredictor, sam_model_registry

        model = sam_model_registry["vit_b"](checkpoint=str(checkpoint))
        model.to(torch.device(device))
        self.predictor = SamPredictor(model)
        self.window = window
        self.stride = stride
        self.rgb: np.ndarray | None = None
        self.current: tuple[int, int] | None = None

    def set_field(self, rgb: np.ndarray) -> None:
        self.rgb = rgb
        self.current = None

    def _origin(self, x: float, y: float) -> tuple[int, int]:
        height, width = self.rgb.shape[:2]
        starts_x = sorted({min(max(0, s), max(0, width - self.window)) for s in range(0, width, self.stride)})
        starts_y = sorted({min(max(0, s), max(0, height - self.window)) for s in range(0, height, self.stride)})
        return (
            min(starts_x, key=lambda s: abs(s + self.window / 2 - x)),
            min(starts_y, key=lambda s: abs(s + self.window / 2 - y)),
        )

    def predict(
        self,
        x: float,
        y: float,
        box: tuple[float, float, float, float] | None = None,
        multimask: bool = True,
    ) -> tuple[list[np.ndarray], list[float], tuple[int, int]]:
        origin = self._origin(x, y)
        if origin != self.current:
            x0, y0 = origin
            self.predictor.set_image(self.rgb[y0 : y0 + self.window, x0 : x0 + self.window])
            self.current = origin
        x0, y0 = origin
        masks, scores, _ = self.predictor.predict(
            point_coords=np.array([[x - x0, y - y0]], dtype=float),
            point_labels=np.array([1]),
            box=None if box is None else np.array([box[0] - x0, box[1] - y0, box[2] - x0, box[3] - y0], dtype=float),
            multimask_output=multimask,
        )
        return [mask.astype(bool) for mask in masks], [float(s) for s in scores], origin


def paste(mask: np.ndarray, origin: tuple[int, int], shape: tuple[int, int]) -> np.ndarray:
    full = np.zeros(shape, dtype=bool)
    x0, y0 = origin
    h, w = mask.shape
    full[y0 : y0 + h, x0 : x0 + w] = mask
    return full


__all__ = [
    "WindowedSam",
    "paste",
]
