"""The tiles labels and the viewers refer to, and the ones new labelling fields are drawn from.

Detection reads the whole-core tiles (`Project.analysis_tiles()`). Labels, labelling sessions and training sets name a
tile by its id, and in projects analysed before whole-core analysis many of those ids are sampled fields (…_S01) from
the old sampled-field tables (`project.LEGACY_INPUTS`). Most of them are whole-core tiles too (the tiler keeps their
ids); the few that are not are still found here, so earlier labels keep working. New labelling fields always come from
the whole-core tiles.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from stainid import cache
from stainid.project import LEGACY_INPUTS, Project

LABEL_TISSUE = 0.10  # a labelling field is a full-size whole-core tile with at least this share of tissue (its preview)

# identifiers stay text: a stain column holding only "6E10" would otherwise be read as the number 6e10
TEXT_COLUMNS = {c: str for c in ("tile_id", "core_id", "tma", "core_label", "donor_id", "sample_region_id", "region", "disease_group", "stain", "image_path")}


def _paths(project: Project) -> list[Path]:
    return [project.analysis_tiles(), *(project.input(key) for key in LEGACY_INPUTS)]


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=TEXT_COLUMNS) if path.exists() else pd.DataFrame()


def table(path: Path) -> pd.DataFrame:
    """A tile table, read again only when the file changes (shared between requests: never change it in place)."""
    return cache.memo(("frame", str(path)), [path], lambda: _read(path))


def known_tiles(project: Project) -> pd.DataFrame:
    """Every tile with an id: the whole-core tiles, then the older sampled fields that are not among them. Shared
    between requests (kept until a table changes): never change it in place."""
    paths = _paths(project)

    def make() -> pd.DataFrame:
        frames = [frame for frame in map(table, paths) if not frame.empty]
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True).drop_duplicates("tile_id").reset_index(drop=True)

    return cache.memo(("known tiles", *map(str, paths)), paths, make)


def known_tile_rows(project: Project) -> dict[str, dict[str, str]]:
    """The same tiles as rows of text, by tile id (as `stainid.tables.read_csv` gives them)."""
    out: dict[str, dict[str, str]] = {}
    for path in _paths(project):
        for row in cache.rows(path):
            out.setdefault(row["tile_id"], row)
    return out


def labelling_tiles(project: Project) -> pd.DataFrame:
    """The whole-core tiles new labelling fields are drawn from: full-size squares with at least `LABEL_TISSUE` tissue
    (rim tiles are labelled too, nearly empty ones are not), so every field shows the same area; the cut-off tiles at a
    core's right and bottom edge stay out. In an image set every image with that much tissue (the images are the size
    the lab chose)."""
    path = project.analysis_tiles()
    whole_squares = project.is_tma

    def make() -> pd.DataFrame:
        tiles = table(path)
        if tiles.empty:
            return tiles
        tissue = tiles.preview_tissue_fraction.fillna(1.0) if "preview_tissue_fraction" in tiles else 1.0
        size = tiles.width_px.max()
        full = (tiles.width_px == size) & (tiles.height_px == size) if whole_squares else True
        return tiles[full & (tissue >= LABEL_TISSUE)].reset_index(drop=True)

    return cache.memo(("labelling tiles", str(path), whole_squares), [path], make)
