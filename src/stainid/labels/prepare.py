"""`stainid labels prepare|import|store-from-pool`: build candidate pools for fields people will label, bring in earlier label sets, or
write a v2 candidate store from the pool (to try the annotation rounds before the v2 detection has run)."""
from __future__ import annotations

from stainid.labels import importer, pool, readiness, sessions
from stainid.project import Project
from stainid.sampling.tiles import labelling_tiles


def run(project: Project, action: str, stain: str | None, fields: int, session: str, seed: int) -> None:
    if action == "import":
        importer.import_all(project)
        return
    if action == "store-from-pool":
        from stainid.labels import candidate_store
        from stainid.sampling.tiles import known_tiles

        for one in [stain] if stain else ("NeuN", "6E10", "AT8"):
            written = candidate_store.from_pool(project, one, known_tiles(project))
            print(f"{one}: {sum(written.values())} candidates on {len(written)} cores -> {project.relative(candidate_store.folder(project, one))}", flush=True)
        return
    if session:
        meta = sessions.meta(project.output("reviews"), session)
        if sessions.label_set_only(meta):  # an AI review round: read-only labels, nobody opens its fields
            print(f"{session} is a read-only label set: no fields to prepare", flush=True)
            return
        pool.prepare(project, meta["stain"], meta["fields"])
        return
    if not stain:
        raise ValueError("choose a stain (--stain) or a session (--session)")
    tiles = labelling_tiles(project)
    ready = readiness.ready(project, tiles[tiles.stain == stain])  # fields whose core is detected, calibrated and exported
    chosen = sessions.pick_fields(ready, pool.prepared(project, stain), fields, seed)
    pool.prepare(project, stain, chosen)
