from __future__ import annotations

from pathlib import Path

import numpy as np

WINDOW, STRIDE = 1024, 768  # SAM sees a field in native-resolution windows this big, this far apart


def _starts(length: int, window: int, stride: int) -> list[int]:
    return sorted({min(max(0, s), max(0, length - window)) for s in range(0, length, stride)})


def window_origins(shape: tuple[int, ...], window: int = WINDOW, stride: int = STRIDE) -> list[tuple[int, int]]:
    """The top-left corners (x, y) of every window of a field of this (height, width)."""
    height, width = shape[:2]
    return [(x, y) for y in _starts(height, window, stride) for x in _starts(width, window, stride)]


def window_origin(shape: tuple[int, ...], x: float, y: float, window: int = WINDOW, stride: int = STRIDE) -> tuple[int, int]:
    """The window whose centre is nearest to (x, y) (top-left corner)."""
    height, width = shape[:2]
    return (min(_starts(width, window, stride), key=lambda s: abs(s + window / 2 - x)),
            min(_starts(height, window, stride), key=lambda s: abs(s + window / 2 - y)))


class WindowedSam:
    """Point- and box-prompted SAM on native-resolution windows of a large field."""

    def __init__(self, checkpoint: Path, device: str = "cpu", window: int = WINDOW, stride: int = STRIDE, precision: str = "fp32"):
        """`precision` applies on CUDA only (see `Project.precision`): bf16 runs the image encoder (the expensive part) in
        bfloat16 and the small mask decoder in fp32; tf32 uses TF32 tensor cores throughout."""
        import torch
        from segment_anything import SamPredictor, sam_model_registry

        cuda = device == "cuda"
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = cuda and precision in ("tf32", "bf16")
        self.bf16 = cuda and precision == "bf16"
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
        return window_origin(self.rgb.shape[:2], x, y, self.window, self.stride)

    def predict(
        self,
        x: float,
        y: float,
        box: tuple[float, float, float, float] | None = None,
        multimask: bool = True,
    ) -> tuple[list[np.ndarray], list[float], tuple[int, int]]:
        origin = self._origin(x, y)
        if origin != self.current:
            import torch

            x0, y0 = origin
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.bf16):
                self.predictor.set_image(self.rgb[y0 : y0 + self.window, x0 : x0 + self.window])
            self.predictor.features = self.predictor.features.float()
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
    "STRIDE",
    "WINDOW",
    "WindowedSam",
    "paste",
    "window_origin",
    "window_origins",
]
