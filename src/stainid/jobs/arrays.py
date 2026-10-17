"""Job arrays: a big step split into chunks of core images under one parent job.

A step over many core images (Find nuclei, the v2 proposals, calls and outlines) runs as one chunk per TMA (or per
`CHUNK_CORES` core images for image sets and slides, and a TMA split further when it holds more than `MAX_CORES`).
Each chunk is the same command with `--core <the chunk's list>`, so a pre-emption costs at most one chunk's units in
flight, progress and estimates are per chunk, and other work interleaves between chunks. The parent is never run: its
state, progress and estimate come from its chunks, and Pause, Resume, Cancel and Move act on all of them.
"""
from __future__ import annotations

from pathlib import Path

ARRAY_STEPS = ("nuclei", "v2_proposals", "v2_calls", "v2_outlines")
CHUNK_CORES = 50
MAX_CORES = 200
STAINS = ("NeuN", "6E10", "AT8")


def _cores_in(words: list[str], root: Path) -> set[str] | None:
    found: set[str] = set()
    for i, word in enumerate(words[:-1]):
        if word == "--core":
            path = Path(words[i + 1])
            path = path if path.is_absolute() else root / path
            found |= {line.strip() for line in path.read_text().splitlines() if line.strip()} if path.is_file() else {words[i + 1]}
    return found or None


def without_cores(words: list[str]) -> list[str]:
    out, skip = [], False
    for word in words:
        if skip:
            skip = False
            continue
        if word == "--core":
            skip = True
            continue
        out.append(word)
    return out


def plan(project, workflow: str, options: dict, argv: list[str], by: str | int | None = None) -> list[dict] | None:
    """The chunks of a step ({"label", "cores", "units"}), or None when it is not worth splitting (not an array step,
    run in shards already, or fewer than two chunks). `by`: "tma", a number of core images per chunk, or None (per
    TMA for a TMA study, else per `CHUNK_CORES`)."""
    from stainid import cache

    shards = argv[argv.index("--shard-count") + 1] if "--shard-count" in argv[:-1] else "1"
    if workflow not in ARRAY_STEPS or int(options.get("shard_count") or 1) > 1 or shards != "1":
        return None
    stains = options.get("stain") or list(STAINS)
    stains = set(stains if isinstance(stains, list) else [stains])
    wanted = _cores_in(argv, Path(project.root))
    rows = cache.rows(project.analysis_tiles(), ("core_id", "tma", "stain"))
    per_core: dict[str, dict] = {}
    for row in rows:
        if row["stain"] not in stains or (wanted is not None and row["core_id"] not in wanted):
            continue
        info = per_core.setdefault(row["core_id"], {"tma": row["tma"], "tiles": 0, "stains": set()})
        info["tiles"] += 1
        info["stains"].add(row["stain"])
    if not per_core:
        return None
    tmas = sorted({info["tma"] for info in per_core.values()}, key=lambda t: (len(t), t))
    groups: list[tuple[str, list[str]]] = []
    if (by == "tma" or (by is None and project.is_tma)) and len(tmas) > 1:
        for tma in tmas:
            cores = sorted(c for c, info in per_core.items() if info["tma"] == tma)
            parts = [cores[i:i + MAX_CORES] for i in range(0, len(cores), MAX_CORES)]
            groups += [(f"TMA {tma}" + (f" ({n}/{len(parts)})" if len(parts) > 1 else ""), part) for n, part in enumerate(parts, start=1)]
    else:
        size = int(by) if isinstance(by, int) or (isinstance(by, str) and by.isdigit()) else CHUNK_CORES
        cores = sorted(per_core)
        groups = [(f"core images {i + 1}–{min(i + size, len(cores))}", cores[i:i + size]) for i in range(0, len(cores), size)]
    if len(groups) < 2:
        return None
    unit = (lambda c: per_core[c]["tiles"]) if workflow == "nuclei" else (lambda c: len(per_core[c]["stains"]))
    return [{"label": label, "cores": cores, "units": sum(map(unit, cores))} for label, cores in groups]
