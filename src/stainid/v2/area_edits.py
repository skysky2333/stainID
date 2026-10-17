"""Applying people's area edits to the results ("re-detecting edited cores"): stage 2 takes the edits of a core image as
an input (excluded pixels are taken out of the tissue before detection; `exclusions.edit_codes`), so an edit puts that
core image's stages 2–5 out of date. This redoes them for the edited core images only: proposals (2), calls (4),
outlines (3), then the results tables (5).

A core's proposals record the digest of its edits they were made with (`inputs.exclusions` of the stage-2 manifest,
`exclusions.edits_digest`); a core whose manifest has none was made before and is left to the next full run.
"""
from __future__ import annotations

from stainid import exclusions
from stainid.v2 import manifest
from stainid.v2.params import STAINS

KEY = "exclusions"  # the input of the stage-2 manifest holding the edits digest
STAGE = "proposals"


def changed(project) -> dict[str, list[str]]:
    """Per stain, the core images whose edits changed since their proposals were made."""
    out: dict[str, list[str]] = {}
    for stain in STAINS:
        for core_id, recorded in sorted(manifest.read_all(project, STAGE, stain).items()):
            made_with = ((recorded or {}).get("inputs") or {}).get(KEY)
            if made_with is None:
                continue
            if exclusions.edits_digest(project, core_id, stain) != made_with:
                out.setdefault(stain, []).append(core_id)
    return out


def status(project) -> dict:
    found = changed(project)
    return {"cores": sum(len(v) for v in found.values()), "by_stain": {k: len(v) for k, v in found.items()}}


def apply(project, log=print) -> dict:
    """Re-detect the edited core images (stages 2, 4, 3) and remake the results tables (5); {} when nothing changed."""
    from stainid.v2 import run  # noqa: PLC0415

    found = changed(project)
    if not found:
        log("no area edits to apply: every core was detected with its current excluded areas", flush=True)
        return {}
    for stain, cores in found.items():
        log(f"{stain}: re-detecting {len(cores)} edited core image(s)", flush=True)
        run.run_proposals(project, [stain], set(cores))
        run.run_calls(project, [stain], set(cores))
        run.run_outlines(project, [stain], set(cores))
    run.run_measures(project)
    return found
