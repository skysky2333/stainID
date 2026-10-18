"""Stage manifests: what each stage output of a core image was made from, so a run redoes exactly what is out of date.

Every stage writes, per stain and core image, `manifests/<stage>/<stain>/<core_id>.json` after its outputs (it marks the
core done): the stage's code version, its parameters and a fingerprint of each input (the core image, the nuclei, the
slide threshold, the hand-drawn exclusions, the model bundle, and the digests of the upstream stages' manifests). Its
`digest` hashes all of that, and downstream manifests record it, so a change anywhere upstream propagates: a re-run
proposal stage makes the classification of that core outdated, a new model or rule only the classification.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

from stainid import cache
from stainid.tables import write_text_atomic
from stainid.v2.params import digest as params_digest

# Bump a stage's version when its code changes what it writes: every output of that stage becomes outdated.
READ = object()  # "read the manifest from its file": the default of the `expected` functions' known-manifest arguments
VERSIONS = {"proposals": 5, "outlines": 2, "calls": 2, "candidates": 2}
STAGES = ("proposals", "outlines", "calls", "candidates")


def root(project) -> Path:
    return project.output("v2")


def path(project, stage: str, stain: str, core_id: str) -> Path:
    return root(project) / "manifests" / stage / stain / f"{core_id}.json"


def file_signature(file: Path | str) -> str:
    try:
        stat = os.stat(file)
    except OSError:
        return "missing"
    return f"{stat.st_size}:{stat.st_mtime_ns}"


def sha256(file: Path) -> str:
    """The sha256 of a file, remembered until the file changes (model bundles are read once per run)."""
    def make() -> str:
        h = hashlib.sha256()
        with Path(file).open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                h.update(block)
        return h.hexdigest()

    return cache.memo(("sha256", str(file)), [file], make)


def make(stage: str, params: dict, inputs: dict) -> dict:
    content = {"stage": stage, "version": VERSIONS[stage], "params": params, "inputs": inputs}
    content["digest"] = params_digest(content)
    return content


def write(project, stage: str, stain: str, core_id: str, manifest: dict, **extra) -> dict:
    target = path(project, stage, stain, core_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    write_text_atomic(target, json.dumps({**manifest, **extra, "written": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=1, default=str))
    return manifest


def read(project, stage: str, stain: str, core_id: str) -> dict | None:
    target = path(project, stage, stain, core_id)
    try:
        return json.loads(target.read_text())
    except (FileNotFoundError, ValueError):
        return None


def up_to_date(recorded: dict | None, expected: dict) -> bool:
    return recorded is not None and recorded.get("digest") == expected["digest"]


def read_all(project, stage: str, stain: str) -> dict[str, dict]:
    """Every core manifest of one stage and stain (kept until the folder changes)."""
    folder = root(project) / "manifests" / stage / stain

    def load() -> dict[str, dict]:
        """Read again only the manifests that changed since the last look (a running stage adds one per core)."""
        kept = _FILES.setdefault(str(folder), {})
        out, seen = {}, set()
        try:
            entries = list(os.scandir(folder))
        except (FileNotFoundError, NotADirectoryError):
            entries = []
        for entry in entries:
            name = entry.name
            if not name.endswith(".json"):
                continue
            try:
                stat = entry.stat()
            except OSError:
                continue
            seen.add(name)
            signature = (stat.st_mtime_ns, stat.st_size)
            known = kept.get(name)
            if known is None or known[0] != signature:
                try:
                    known = (signature, json.loads(Path(entry.path).read_text()))
                except (OSError, ValueError):
                    continue
                kept[name] = known
            out[name[:-5]] = known[1]
        for name in set(kept) - seen:
            del kept[name]
        return out

    return cache.memo(("v2 manifests", str(folder)), [folder], load, folders=True)


_FILES: dict[str, dict[str, tuple[tuple[int, int], dict]]] = {}  # folder -> file name -> (its mtime and size, its manifest)
