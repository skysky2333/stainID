from typing import Dict, Mapping, Tuple

import numpy as np
import pandas as pd

from .cross_channel_features import compute_cross_channel_features
from .common import format_length_um
from .mask_features import (
    compute_mask_topology_features,
    compute_object_score_features,
    compute_pixel_intensity_features,
    compute_site_summary,
    compute_tile_heterogeneity_features,
)
from .object_features import measure_objects
from .spatial_features import compute_spatial_features
from .texture_features import compute_granularity_features, compute_texture_features


def measure_channel_sites(mask: np.ndarray,
                          norm_image: np.ndarray,
                          raw_image: np.ndarray,
                          pixel_width_um: float,
                          pixel_height_um: float,
                          labeled_in: np.ndarray = None) -> Tuple[np.ndarray, pd.DataFrame]:
    return measure_objects(
        mask=mask,
        norm_image=norm_image,
        raw_image=raw_image,
        pixel_width_um=pixel_width_um,
        pixel_height_um=pixel_height_um,
        labeled_in=labeled_in,
    )


def compute_channel_summary(channel_name: str,
                            mask: np.ndarray,
                            valid_mask: np.ndarray,
                            norm_image: np.ndarray,
                            raw_image: np.ndarray,
                            df_sites: pd.DataFrame,
                            masks_by_channel: Mapping[str, np.ndarray],
                            labels_by_channel: Mapping[str, np.ndarray],
                            site_tables: Mapping[str, pd.DataFrame],
                            norm_by_channel: Mapping[str, np.ndarray],
                            raw_by_channel: Mapping[str, np.ndarray],
                            pixel_width_um: float,
                            pixel_height_um: float) -> Dict[str, float]:
    out: Dict[str, float] = {}
    points_um = df_sites[["centroid_row_um", "centroid_col_um"]].values if not df_sites.empty else np.zeros((0, 2), dtype=np.float64)
    img_shape_um = (mask.shape[0] * float(pixel_height_um), mask.shape[1] * float(pixel_width_um))

    out["num_sites"] = int(df_sites.shape[0])
    out.update(compute_site_summary(df_sites))
    out.update(compute_pixel_intensity_features(mask, valid_mask, norm_image, raw_image))
    out.update(compute_object_score_features(df_sites))
    out.update(
        compute_mask_topology_features(
            mask,
            valid_mask,
            pixel_width_um,
            pixel_height_um,
            labeled=labels_by_channel.get(channel_name),
        )
    )
    out.update(compute_tile_heterogeneity_features(mask, valid_mask, norm_image))
    out.update(compute_spatial_features(points_um, img_shape_um))

    tissue_mask = np.asarray(valid_mask, dtype=bool)
    positive_mask = np.asarray(mask, dtype=bool) & tissue_mask
    px_area_um2 = float(pixel_width_um) * float(pixel_height_um)
    mean_pixel_um = float(np.sqrt(px_area_um2))
    tissue_area_um2 = float(np.sum(tissue_mask)) * px_area_um2
    out["tissue_area_um2"] = tissue_area_um2
    out["positive_area_um2"] = float(np.sum(positive_mask)) * px_area_um2
    out["num_sites_per_mm2_tissue"] = (
        float(df_sites.shape[0]) / (tissue_area_um2 / 1_000_000.0)
        if tissue_area_um2 > 0
        else float("nan")
    )
    texture_scales = (1, 2, 4)
    texture_labels = tuple(f"s{format_length_um(scale * mean_pixel_um)}um" for scale in texture_scales)
    granularity_scales = (1, 2, 4, 8)
    granularity_labels = tuple(f"s{format_length_um(scale * mean_pixel_um)}um" for scale in granularity_scales)
    out.update(compute_texture_features(norm_image, tissue_mask, prefix="texture_tissue_norm", distances=texture_scales, distance_labels=texture_labels))
    out.update(compute_texture_features(raw_image, tissue_mask, prefix="texture_tissue_raw", distances=texture_scales, distance_labels=texture_labels))
    out.update(compute_texture_features(norm_image, positive_mask, prefix="texture_positive_norm", distances=texture_scales, distance_labels=texture_labels))
    out.update(compute_texture_features(raw_image, positive_mask, prefix="texture_positive_raw", distances=texture_scales, distance_labels=texture_labels))
    out.update(compute_granularity_features(norm_image, tissue_mask, prefix="granularity_tissue_norm", radii=granularity_scales, radius_labels=granularity_labels))
    out.update(compute_granularity_features(norm_image, positive_mask, prefix="granularity_positive_norm", radii=granularity_scales, radius_labels=granularity_labels))
    out.update(
        compute_cross_channel_features(
            channel_name=channel_name,
            masks_by_channel=masks_by_channel,
            labels_by_channel=labels_by_channel,
            site_tables=site_tables,
            norm_by_channel=norm_by_channel,
            raw_by_channel=raw_by_channel,
            valid_mask=valid_mask,
            pixel_width_um=pixel_width_um,
            pixel_height_um=pixel_height_um,
        )
    )
    return out
