"""Applying people's area edits to the results: the core images whose resolved excluded areas (`stainid.exclusions`)
changed since their calls were made are decided again (stage 4, which takes each candidate's excluded share from the
layer) and the results tables made again (stage 5, valid areas from the layer). Nothing else is redone: proposals
(stage 2) and outlines (stage 3) do not depend on the edits.

A core's calls record the layer's digest they were made with (`inputs.exclusions` of the stage-4 manifest); a core
whose manifest has none was made before the layer existed and is left to the next full run.
"""
from __future__ import annotations

from stainid import exclusions
from stainid.v2 import manifest
from stainid.v2.params import STAINS

KEY = "exclusions"  # the input of the stage-4 manifest holding the layer digest


def changed(project) -> dict[str, list[str]]:
    """Per stain, the core images whose layer changed since their calls were made."""
    out: dict[str, list[str]] = {}
    for stain in STAINS:
        for core_id, recorded in sorted(manifest.read_all(project, "calls", stain).items()):
            made_with = ((recorded or {}).get("inputs") or {}).get(KEY)
            if made_with is None:
                continue
            if exclusions.digest(project, core_id, stain) != made_with:
                out.setdefault(stain, []).append(core_id)
    return out


def status(project) -> dict:
    found = changed(project)
    return {"cores": sum(len(v) for v in found.values()), "by_stain": {k: len(v) for k, v in found.items()}}


def apply(project, log=print) -> dict:
    """Decide the changed cores again and remake the results tables; {} when nothing changed."""
    from stainid.v2 import run  # noqa: PLC0415

    found = changed(project)
    if not found:
        log("no area edits to apply: every core's calls were made with its current excluded areas", flush=True)
        return {}
    for stain, cores in found.items():
        log(f"{stain}: deciding {len(cores)} core image(s) again with their edited areas", flush=True)
        run.run_calls(project, [stain], set(cores))
    run.run_measures(project)
    return found
