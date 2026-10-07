"""End-to-end workflows driven by a `Project` (used by the CLI and the web app job runner).

Each workflow prints one progress line per unit of work (`[i/n] ...`) so the job runner can report progress.
All workflows are resumable: finished outputs are skipped.
"""
from __future__ import annotations

import time
from collections import defaultdict

from stainid.tables import file_names, read_csv


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


def follow_nuclei(project, stains: set[str], run, shard_index: int = 0, shard_count: int = 1, poll_s: float = 60,
                  patience_s: float = 3600) -> None:
    """Run a detection step on each core image as soon as Find nuclei has written all of its tiles, until every core image
    of this shard is done, so detection on the CPU keeps pace with Cellpose on the GPU instead of waiting for it.

    `run(stain, core_ids)` detects the given core images of one stain (resumable, so a repeat is harmless). Stops with an
    error when no new core image has become ready for `patience_s` seconds (Find nuclei has stopped)."""
    waiting = tiles_for(project.analysis_tiles(), stains, shard_index, shard_count)
    print(f"Following Find nuclei for {len(waiting)} core images", flush=True)
    last_ready = time.time()
    while waiting:
        nuclei = file_names(project.output("nuclei"))
        ready = [key for key, tiles in waiting.items() if all(f"{t['tile_id']}.npz" in nuclei for t in tiles)]
        for stain in sorted({stain for _, stain in ready}):
            run(stain, {core for core, s in ready if s == stain})
        for key in ready:
            del waiting[key]
        if ready:
            last_ready = time.time()
        elif time.time() - last_ready > patience_s:
            raise RuntimeError(f"{len(waiting)} core images still lack nuclei after {patience_s / 60:.0f} minutes without progress; run Find nuclei")
        elif waiting:
            time.sleep(poll_s)


def calibration_table(path) -> dict[tuple[str, str], float]:
    return {(r["tma"], r["stain"]): float(r["threshold_dab_od"]) for r in read_csv(path)}
