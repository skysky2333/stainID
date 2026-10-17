"""The models' neighbourhood features (`stainid.stains.amyloid.classifier.context_features`) for many objects of one field
at once: the stain separation, saturation and grey picture are made once for the field and cropped per object, instead
of once per object crop. Values equal the per-crop function's (same crops, same white padding beyond the field)."""
from __future__ import annotations

import cv2
import numpy as np

CONTEXT_NAMES = (
    "context_dab_mean", "context_dab_p90", "context_dab_std", "center_dab_mean", "center_dab_p90", "annulus_dab_mean",
    "center_annulus_dab_contrast", "center_hematoxylin_mean", "center_hematoxylin_p90", "annulus_hematoxylin_mean",
    "center_high_hematoxylin_fraction", "context_saturation_mean", "context_laplacian_variance",
)


def padded_crop(image: np.ndarray, center_x: float, center_y: float, size: int, fill) -> np.ndarray:
    """`stainid.stains.neun.audit.crop_with_padding` for any array: a size × size crop centred on the point, `fill` beyond."""
    half = size // 2
    x0, y0 = round(center_x) - half, round(center_y) - half
    x1, y1 = x0 + size, y0 + size
    out = np.full((size, size) + image.shape[2:], fill, dtype=image.dtype)
    sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(image.shape[1], x1), min(image.shape[0], y1)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 - y0: sy1 - y0, sx0 - x0: sx1 - x0] = image[sy0:sy1, sx0:sx1]
    return out


def context_features_many(field, centers: list[tuple[float, float]], diameters_um: list[float], context_um: float = 55.0,
                          crop_size: int | None = None) -> list[dict[str, float]]:
    """context_features of a `crop_size` crop (default: max(64, context_um / pixel size)) around each centre. Pixels in
    the field's excluded areas (`Field.excluded`) count in none of them, as if they were not there."""
    size = crop_size or max(64, round(context_um / field.pixel_size_um))
    yy, xx = np.ogrid[:size, :size]
    distance = np.sqrt((xx - (size - 1) / 2) ** 2 + (yy - (size - 1) / 2) ** 2)
    hed, saturation, gray = field.skimage_hed, field.saturation, field.gray
    excluded = getattr(field, "excluded", None)
    excluded = excluded if excluded is not None and excluded.any() else None
    everywhere = np.ones((size, size), bool)
    out = []
    for (cx, cy), diameter in zip(centers, diameters_um):
        radius = np.clip(diameter * size / context_um / 2, 3, size / 4)
        keep = everywhere if excluded is None else ~padded_crop(excluded, cx, cy, size, False)
        if not keep.any():
            keep = everywhere
        center_all = distance <= radius
        annulus_all = (distance > radius) & (distance <= 3 * radius)
        center = center_all & keep if (center_all & keep).any() else center_all
        annulus = annulus_all & keep if (annulus_all & keep).any() else annulus_all
        crop = padded_crop(hed, cx, cy, size, 0.0)
        hematoxylin, dab = crop[:, :, 0][keep], crop[:, :, 2][keep]
        center_h, annulus_h = crop[:, :, 0][center], crop[:, :, 0][annulus]
        center_dab, annulus_dab = crop[:, :, 2][center], crop[:, :, 2][annulus]
        context_h90 = float(np.quantile(hematoxylin, 0.90))
        grey = padded_crop(gray, cx, cy, size, 255)
        sharp = keep if keep is everywhere else cv2.erode(keep.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        sharp = sharp if sharp.any() else keep  # the Laplacian of pixels whose 3 x 3 neighbourhood is all kept
        out.append({
            "context_dab_mean": float(np.mean(dab)),
            "context_dab_p90": float(np.quantile(dab, 0.90)),
            "context_dab_std": float(np.std(dab)),
            "center_dab_mean": float(np.mean(center_dab)),
            "center_dab_p90": float(np.quantile(center_dab, 0.90)),
            "annulus_dab_mean": float(np.mean(annulus_dab)),
            "center_annulus_dab_contrast": float(np.mean(center_dab) - np.mean(annulus_dab)),
            "center_hematoxylin_mean": float(np.mean(center_h)),
            "center_hematoxylin_p90": float(np.quantile(center_h, 0.90)),
            "annulus_hematoxylin_mean": float(np.mean(annulus_h)),
            "center_high_hematoxylin_fraction": float(np.mean(center_h >= context_h90)),
            "context_saturation_mean": float(np.mean(padded_crop(saturation, cx, cy, size, 0.0)[keep])),
            "context_laplacian_variance": float(cv2.Laplacian(grey, cv2.CV_64F)[sharp].var()),
        })
    return out
