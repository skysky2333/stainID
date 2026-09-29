from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import ndimage

from stainid.imaging.color import rgb_to_hed


def clean_nuclei(nuclei: np.ndarray, rgb: np.ndarray, minimum_hematoxylin: float = 0.03) -> np.ndarray:
    hematoxylin = np.maximum(rgb_to_hed(rgb)[..., 0], 0)
    means = ndimage.mean(hematoxylin, nuclei, index=np.arange(1, int(nuclei.max()) + 1))
    keep = np.concatenate(([False], np.asarray(means) >= minimum_hematoxylin))
    return np.where(keep[nuclei], nuclei, 0)


def nucleus_centroids(nuclei: np.ndarray, valid: np.ndarray, pixel_size_um: float) -> np.ndarray:
    points = []
    for label, bounds in enumerate(ndimage.find_objects(nuclei), start=1):
        if bounds is None:
            continue
        local = nuclei[bounds] == label
        area = local.sum() * pixel_size_um**2
        if not 10.0 <= area <= 200.0:
            continue
        ys, xs = np.nonzero(local)
        y, x = int(bounds[0].start + ys.mean()), int(bounds[1].start + xs.mean())
        if valid[y, x]:
            points.append((x, y))
    return np.asarray(points, dtype=int).reshape(-1, 2)


CELLPOSE_SETTINGS = {
    "diameter": 30.0,
    "flow_threshold": 0.4,
    "cellprob_threshold": 0.0,
    "min_size": 50,
    "bsize": 256,
    "batch_size": 1,
}


def load_cellpose(model_path: Path, threads: int, device: str):
    import torch
    from cellpose import models

    torch.set_num_threads(threads)
    return models.CellposeModel(
        device=torch.device(device),
        pretrained_model=str(model_path),
        use_bfloat16=False,
    )


def cellpose_masks(model, rgb: np.ndarray) -> np.ndarray:
    masks, _, _ = model.eval(
        rgb,
        batch_size=CELLPOSE_SETTINGS["batch_size"],
        channel_axis=2,
        normalize=True,
        diameter=CELLPOSE_SETTINGS["diameter"],
        flow_threshold=CELLPOSE_SETTINGS["flow_threshold"],
        cellprob_threshold=CELLPOSE_SETTINGS["cellprob_threshold"],
        min_size=CELLPOSE_SETTINGS["min_size"],
        bsize=CELLPOSE_SETTINGS["bsize"],
    )
    return masks
