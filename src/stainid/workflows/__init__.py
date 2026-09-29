"""End-to-end workflows driven by a `Project` (used by the CLI and the web app job runner).

Each workflow prints one progress line per unit of work (`[i/n] ...`) so the job runner can report progress.
All workflows are resumable: finished outputs are skipped.
"""
from __future__ import annotations

from collections import defaultdict

from stainid.tables import read_csv


def tiles_for(tile_manifest, stains: set[str], shard_index: int = 0, shard_count: int = 1, core_ids: set[str] | None = None):
    """Tiles of the requested stains, grouped per (core_id, stain) and sharded by core."""
    rows = [r for r in read_csv(tile_manifest) if r["stain"] in stains and (core_ids is None or r["core_id"] in core_ids)]
    keys = sorted({(r["core_id"], r["stain"]) for r in rows})[shard_index::shard_count]
    wanted = set(keys)
    grouped = defaultdict(list)
    for row in rows:
        if (row["core_id"], row["stain"]) in wanted:
            grouped[(row["core_id"], row["stain"])].append(row)
    return {key: sorted(tiles, key=lambda t: int(t["selection_order"])) for key, tiles in sorted(grouped.items())}


def calibration_table(path) -> dict[tuple[str, str], float]:
    return {(r["tma"], r["stain"]): float(r["threshold_dab_od"]) for r in read_csv(path)}
