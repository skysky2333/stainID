"""`stainid labels prepare|import`: build candidate pools for fields people will label, or bring in earlier label sets."""
from __future__ import annotations

from stainid.labels import importer, pool, readiness, sessions
from stainid.project import Project
from stainid.sampling.tiles import labelling_tiles


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
    tiles = labelling_tiles(project)
    ready = readiness.ready(project, tiles[tiles.stain == stain])  # fields whose core is detected, calibrated and exported
    chosen = sessions.pick_fields(ready, pool.prepared(project, stain), fields, seed)
    pool.prepare(project, stain, chosen)
