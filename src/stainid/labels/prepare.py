"""`stainid labels prepare|import`: build candidate pools for fields people will label, or bring in earlier label sets."""
from __future__ import annotations

import pandas as pd

from stainid.labels import importer, pool, sessions
from stainid.project import Project


def run(project: Project, action: str, stain: str | None, fields: int, session: str, seed: int) -> None:
    if action == "import":
        importer.import_all(project)
        return
    if session:
        meta = sessions.meta(project.output("reviews"), session)
        pool.prepare(project, meta["stain"], meta["fields"])
        return
    if not stain:
        raise ValueError("choose a stain (--stain) or a session (--session)")
    tiles = pd.read_csv(project.input("tile_manifest"), dtype={"tma": str})
    tiles = tiles[tiles.stain == stain]
    chosen = sessions.pick_fields(tiles, pool.prepared(project, stain), fields, seed)
    pool.prepare(project, stain, chosen)
