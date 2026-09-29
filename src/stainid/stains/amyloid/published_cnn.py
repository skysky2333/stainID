from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

from stainid.stains.neun.audit import crop_with_padding

MODEL_DIR = Path(os.environ.get("STAINID_PLAQUE_CNN_DIR", "data/models/plaque_cnn"))
MODELS = {
    "plaquebox_tang2019": ("plaquebox_model.pkl", "plaquebox_normalization.npy"),
    **{f"consensus_wong2022_fold{i}": (f"ensemble_model_all_fold_{i}_thresholding_2_l2.pkl", "consensus_normalization.npy") for i in range(4)},
}
CROP_UM = 128.0
_CACHE: dict[str, tuple] = {}


def load(name: str):
    import torch

    import __main__

    if name not in _CACHE:
        import sys

        from stainid.stains.amyloid import wong_consensus

        sys.modules.setdefault("core", wong_consensus)
        for cls in (wong_consensus.CustomizedLinearFunction, wong_consensus.EnsembleNet, wong_consensus.EquallyWeightedEnsembleNet, wong_consensus.Net):
            setattr(__main__, cls.__name__, cls)
        weights, norm = MODELS[name]
        model = torch.load(MODEL_DIR / weights, map_location="cpu", weights_only=False)
        model = getattr(model, "module", model).eval()
        stats = np.load(MODEL_DIR / norm, allow_pickle=True).item()
        _CACHE[name] = (model, np.asarray(stats["mean"], dtype=np.float32), np.asarray(stats["std"], dtype=np.float32))
    return _CACHE[name]


def predict(name: str, images: list[np.ndarray]) -> np.ndarray:
    import torch

    model, mean, std = load(name)
    x = (np.stack(images).astype(np.float32) / 255.0 - mean) / std
    with torch.inference_mode():
        return torch.sigmoid(model(torch.from_numpy(np.ascontiguousarray(x.transpose(0, 3, 1, 2)))).contiguous()).numpy()


def consensus_probabilities(images: list[np.ndarray]) -> np.ndarray:
    return np.mean([predict(f"consensus_wong2022_fold{i}", images) for i in range(4)], axis=0)


def cnn_patch(rgb: np.ndarray, x: float, y: float, pixel_size_um: float) -> np.ndarray:
    crop = crop_with_padding(rgb, x, y, round(CROP_UM / pixel_size_um))
    return cv2.resize(crop, (256, 256), interpolation=cv2.INTER_AREA)


__all__ = [
    "cnn_patch",
    "consensus_probabilities",
    "predict",
]
