"""Whether a field can be labelled yet, and if not, which step comes first.

Preparing a field (`pool.build`) reads the core image, the slide's stain threshold, the stain's model and the AI's
calls (the detection results of that core); for NeuN and AT8 it also needs Cellpose outlines, read from Find nuclei
or made on the spot with the Cellpose weights. A field missing any of these cannot be offered for labelling: it is
shown with the first missing thing in plain words and the step that makes it. This is per field (in practice per
core), so the fields that are ready can always be chosen.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from stainid import cache
from stainid.labels import pool
from stainid.labels.tasks import MODEL_KEY
from stainid.project import Project
from stainid.tables import read_csv

DETECTION = {"NeuN": ("neun", "Detect NeuN neurons"), "6E10": ("fields", "Detect plaques and tau"), "AT8": ("fields", "Detect plaques and tau")}


def _blocker(text: str, step: str, title: str, link: str = "") -> dict:
    return {"reason": text, "step": step, "step_title": title, "link": link or f"/workflow#{step}"}


def _calibrated(project: Project) -> set[tuple[str, str]]:
    path = project.input("calibration")
    return cache.memo(("calibrated", str(path)), [path], lambda: {(r["tma"], r["stain"]) for r in read_csv(path)} if path.exists() else set())


def _prepared(project: Project, stain: str) -> set[str]:
    """Fields whose candidate pool is built (shared with the step status, kept until the pool folder changes)."""
    folder = pool.folder(project, stain)
    return cache.memo(("prepared", str(folder)), [folder], lambda: pool.prepared(project, stain), folders=True)


def blockers(project: Project, tiles: pd.DataFrame) -> list[dict | None]:
    """For each tile (row of `tiles`, one stain or several): None when it can be labelled, else what is missing first
    ({reason, step, step_title, link})."""
    if tiles.empty:
        return []
    calibrated = _calibrated(project)
    images = {path: os.path.exists(project.root / path) for path in set(tiles.image_path)}
    nuclei = cache.names(project.output("nuclei"))
    neun_done = cache.names(project.output("neun") / "tiles")
    field_done = cache.names(project.output("fields") / "parts")
    models = {stain: project.model(key).exists() for stain, key in MODEL_KEY.items()}
    cellpose = Path(project.model("cellpose")).exists()
    prepared = {stain: _prepared(project, stain) for stain in set(tiles.stain)}
    out: list[dict | None] = []
    for tile in tiles[["tile_id", "core_id", "tma", "stain", "image_path"]].itertuples(index=False):
        stain = tile.stain
        detection, title = DETECTION.get(stain, ("", ""))
        if tile.tile_id in prepared[stain]:  # its objects are found already: nothing more is needed
            out.append(None)
        elif not models.get(stain):
            out.append(_blocker(f"There is no {stain} model yet: choose or train one on the Models page first.", "", "Models", "/models"))
        elif not images[tile.image_path]:
            out.append(_blocker("The picture of this core has not been cut out yet: run “Export cores” first.", "export", "Export cores"))
        elif (str(tile.tma), stain) not in calibrated:
            out.append(_blocker("This slide has no stain threshold yet: run “Calibrate stain thresholds” first.", "calibrate", "Calibrate stain thresholds"))
        elif (f"{tile.tile_id}_objects.csv" not in neun_done) if stain == "NeuN" else (f"{tile.core_id}_{stain}_objects.csv" not in field_done):
            out.append(_blocker(f"The AI has not looked at this core yet, so there are no calls to check: run “{title}” first.", detection, title))
        elif stain in ("NeuN", "AT8") and not cellpose and f"{tile.tile_id}.npz" not in nuclei:
            out.append(_blocker("This field has no cell outlines yet and the Cellpose weights are missing: run “Find nuclei” or download the public "
                                "models first.", "nuclei", "Find nuclei"))
        else:
            out.append(None)
    return out


def ready(project: Project, tiles: pd.DataFrame) -> pd.DataFrame:
    """Only the tiles that can be labelled now."""
    return tiles[[b is None for b in blockers(project, tiles)]] if not tiles.empty else tiles
