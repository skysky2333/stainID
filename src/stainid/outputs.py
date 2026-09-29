"""Uniform readers for pipeline outputs: detected objects per field in field-crop pixel coordinates.

Every object row has: tile_id, stain, x, y (field crop incl. context halo), radius_px, model_class, probability,
plus stain-specific measurements.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from stainid.project import Project

META = ["core_id", "sample_region_id", "donor_id", "region", "disease_group"]


def _origin(tile: pd.Series | dict, context_px: int) -> tuple[int, int]:
    return max(0, int(tile["x_px"]) - context_px), max(0, int(tile["y_px"]) - context_px)


def neun_objects(project: Project, tiles: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for tile in tiles.itertuples():
        path = project.output("neun") / "tiles" / f"{tile.tile_id}_objects.csv"
        if not path.exists():
            continue
        o = pd.read_csv(path)
        if "candidate_class" not in o:
            continue
        x0, y0 = _origin(tile._asdict(), project.context_px)
        frames.append(pd.DataFrame({
            "tile_id": tile.tile_id, "stain": "NeuN",
            "x": o.tile_centroid_x_px + int(tile.x_px) - x0, "y": o.tile_centroid_y_px + int(tile.y_px) - y0,
            "radius_px": 0.5 * o.equivalent_diameter_um.fillna(2 * np.sqrt(o.area_um2 / np.pi)) / project.pixel_size_um,
            "model_class": np.where(o.candidate_class == "positive", "neuron", "rejected"),
            "probability": o.context_positive_probability, "source": o.source_kind, "area_um2": o.area_um2,
            **{k: getattr(tile, k) for k in META},
        }))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def field_objects(project: Project, tiles: pd.DataFrame, stain: str) -> pd.DataFrame:
    frames = []
    parts = project.output("fields") / "parts"
    for core_id in tiles.core_id.unique():
        path = parts / f"{core_id}_{stain}_objects.csv"
        if not path.exists():
            continue
        o = pd.read_csv(path)
        if "centroid_x_px" not in o:
            continue
        o = o[o.tile_id.isin(tiles.tile_id)]
        if stain == "6E10":
            radius = 0.5 * o.equivalent_diameter_um / project.pixel_size_um
            frame = pd.DataFrame({"model_class": o.plaque_class, "probability": o.plaque_probability, "area_um2": o.plaque_area_um2,
                                  "dense_core_fraction": o.dense_core_fraction})
        else:
            radius = np.sqrt(o.soma_area_um2 / np.pi) / project.pixel_size_um
            frame = pd.DataFrame({"model_class": np.where(o.candidate_source == "nucleus_ring", "tau_neuron_ring", "tau_neuron_dense"),
                                  "probability": o.tau_neuron_probability, "area_um2": o.soma_area_um2})
        frame.insert(0, "radius_px", radius.to_numpy())
        frame.insert(0, "y", o.centroid_y_px.to_numpy())
        frame.insert(0, "x", o.centroid_x_px.to_numpy())
        frame.insert(0, "stain", stain)
        frame.insert(0, "tile_id", o.tile_id.to_numpy())
        for k in META:
            frame[k] = o[k].to_numpy()
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def objects(project: Project, tiles: pd.DataFrame, stain: str) -> pd.DataFrame:
    tiles = tiles[tiles.stain == stain]
    return neun_objects(project, tiles) if stain == "NeuN" else field_objects(project, tiles, stain)


def field_path(project: Project, tile_id: str) -> Path:
    return project.root / "data" / "cache" / "fields" / f"{tile_id}.jpg"
