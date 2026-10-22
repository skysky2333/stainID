"""The tiles labels and the viewers refer to, and the ones new labelling fields are drawn from.

Detection reads the whole-core tiles (`Project.analysis_tiles()`). Labels, labelling sessions and training sets name a
tile by its id, and in projects analysed before whole-core analysis many of those ids are sampled fields (…_S01) from
the old sampled-field tables (`project.LEGACY_INPUTS`). Most of them are whole-core tiles too (the tiler keeps their
ids); the few that are not are still found here, so earlier labels keep working. New labelling fields always come from
the whole-core tiles.
"""
from __future__ import annotations

import hashlib
import pickle
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd

from stainid import cache
from stainid.project import LEGACY_INPUTS, Project

LABEL_TISSUE = 0.10  # a labelling field is a full-size whole-core tile with at least this share of tissue
MASK_TISSUE = 1  # the tissue bit of detection's per-core mask (stainid.v2.proposals.BITS; 32 = inside a tile, 16 = AT8 edge guard)
MASK_TILE = 32

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


def mask_tissue(path: Path, boxes: tuple[tuple[str, int, int, int, int], ...], scale: int = 8) -> dict[str, float]:
    """Each tile's share of tissue in detection's own tissue mask of its core (`path`: 1/`scale` of the core image, bit
    flags; a cell is tissue when it has `MASK_TISSUE`), over the tile's box (`boxes`: id, x, y, width, height in core
    px), as the analysis counts it. A tile the mask has no cell of (outside every analysed tile) is left out."""
    import cv2  # noqa: PLC0415

    bits = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if bits is None:
        return {}
    bits = bits if bits.ndim == 2 else bits[..., 0]
    out = {}
    for tile_id, x, y, w, h in boxes:
        cells = bits[y // scale: (y + h) // scale, x // scale: (x + w) // scale]
        inside = (cells & MASK_TILE) > 0
        if inside.any():
            out[tile_id] = round(float(((cells & MASK_TISSUE) > 0)[inside].mean()), 4)
    return out


# each core's tissue shares, kept with the mask file's signature (and the tiles' boxes) in one file per project, so a
# restart or a new version of the app does not read every mask again: only the cores whose mask changed
_SHARES: dict[str, dict] = {}
_SHARES_LOCK = threading.Lock()
SHARES_FORMAT = 1


def _shares_file(project: Project) -> Path:
    from stainid import disk_cache  # noqa: PLC0415

    return disk_cache.answer_file(project.root, "mask-tissue.pkl")


def _load_shares(project: Project) -> dict:
    key = str(project.root)
    if key not in _SHARES:
        try:
            with open(_shares_file(project), "rb") as handle:
                saved = pickle.load(handle)  # noqa: S301 - a file this app wrote in its own cache
            _SHARES[key] = saved["cores"] if saved.get("format") == SHARES_FORMAT else {}
        except Exception:  # noqa: BLE001 - missing or unreadable: worked out again
            _SHARES[key] = {}
    return _SHARES[key]


def _save_shares(project: Project, shares: dict) -> None:
    from stainid import disk_cache  # noqa: PLC0415

    target = _shares_file(project)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        disk_cache.write(target, pickle.dumps({"format": SHARES_FORMAT, "cores": shares}, protocol=pickle.HIGHEST_PROTOCOL))
    except OSError:  # only a quicker start lost
        pass


def tissue_fractions(project: Project, tiles: pd.DataFrame) -> pd.Series:
    """Each tile's share of tissue: from detection's tissue mask of its core where there is one (`mask_tissue`), else
    the preview's estimate (`preview_tissue_fraction`, which undercounts pale cores), else 1."""
    from stainid import exclusions  # noqa: PLC0415

    if "preview_tissue_fraction" in tiles:
        preview = pd.to_numeric(tiles.preview_tissue_fraction, errors="coerce")
    else:
        preview = pd.Series(np.nan, index=tiles.index)
    found: dict[str, float] = {}
    if {"stain", "core_id", "x_px", "y_px", "width_px", "height_px"} <= set(tiles.columns) and len(tiles):
        with _SHARES_LOCK:
            shares = _load_shares(project)
            changed = False
            for (stain, core_id), group in tiles.groupby(["stain", "core_id"], sort=False):
                path = exclusions.mask_path(project, str(core_id), str(stain))
                stamp = cache.signature(path)
                if stamp is None:
                    continue
                boxes = tuple((str(r.tile_id), int(r.x_px), int(r.y_px), int(r.width_px), int(r.height_px)) for r in group.itertuples())
                digest = hashlib.sha1(repr(boxes).encode()).hexdigest()
                kept = shares.get(str(path))
                if kept is None or kept[0] != stamp or kept[1] != digest:
                    kept = (stamp, digest, mask_tissue(path, boxes))
                    if stamp[0] / 1e9 < time.time() - cache.RACY_SECONDS:  # a mask being written: looked at again next time
                        shares[str(path)] = kept
                        changed = True
                found.update(kept[2])
            if changed:
                _save_shares(project, shares)
    from_mask = tiles.tile_id.astype(str).map(found)
    return from_mask.fillna(preview).fillna(1.0)


def labelling_tiles(project: Project) -> pd.DataFrame:
    """The whole-core tiles new labelling fields are drawn from: full-size squares with at least `LABEL_TISSUE` tissue
    (rim tiles are labelled too, nearly empty ones are not), so every field shows the same area; the cut-off tiles at a
    core's right and bottom edge stay out. In an image set every image with that much tissue (the images are the size
    the lab chose). The share of tissue is that of detection's tissue mask (`tissue_fractions`)."""
    path = project.analysis_tiles()
    whole_squares = project.is_tma

    def make() -> pd.DataFrame:
        tiles = table(path)
        if tiles.empty:
            return tiles
        tissue = tissue_fractions(project, tiles)
        size = tiles.width_px.max()
        full = (tiles.width_px == size) & (tiles.height_px == size) if whole_squares else True
        return tiles[full & (tissue >= LABEL_TISSUE)].reset_index(drop=True)

    from stainid import exclusions  # noqa: PLC0415

    listed = table(path)
    stains = sorted({str(v) for v in listed.stain.dropna()}) if "stain" in listed else []
    masks = [exclusions.masks_folder(project, stain) for stain in stains]  # a core's mask (re)written: its tissue again
    return cache.memo(("labelling tiles", str(path), whole_squares), [path, *masks], make, folders=True)
