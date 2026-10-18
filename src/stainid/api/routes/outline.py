"""The segmentation check's scores for the admin (`segmentation.report`): polygons a dataset's raters drew (imported
sessions, the public-dataset projects) and the outline sessions ("Outline every object", `labels.outline`) of older
projects, scored against detection's candidates and counted objects (`stainid score-segmentation`).

Outline sessions are no longer made and cannot be worked on: outline accuracy in a study comes from the outlines people
fix in "Fix the field" (`labels.outline_fixes`, on the Models page). Their labels stay in the label log, readable and
scored here.
"""
from __future__ import annotations

import json
import time

from fastapi import APIRouter, Depends

from stainid import cache
from stainid.api.routes.annotate import annotate_path, root
from stainid.api.session import require_admin
from stainid.api.state import get_state
from stainid.labels import outline, store
from stainid.labels.tasks import STAINS
from stainid.segmentation import proposals, report

router = APIRouter(tags=["segmentation check"])
admin = [Depends(require_admin)]
RETIRED = ("“Outline every object” sessions are no longer made: people fix detection's outlines in “Fix the field”, "
           "and the Models page compares those fixes with each of detection's outlines.")


@router.get("/segmentation", dependencies=admin)
def overview() -> dict:
    """The segmentation check: the older outline sessions of each stain and how far they are (read only), the latest
    scores with the files to download, and how many imported sessions (a dataset's raters) can be scored."""
    state = get_state()
    labels = store.current(root())
    present = [s for s in STAINS if not state.tiles.empty and s in set(state.tiles.stain)]
    checks = []
    for meta in report.outline_sessions(state.project):
        done = outline.finished(labels, meta["name"])
        checks.append({"name": meta["name"], "title": meta.get("title", meta["name"]), "stain": meta["stain"], "open": meta["open"],
                       "created": meta["created"], "fields": len(meta["fields"]), "complete": sum(bool(done.get(t)) for t in meta["fields"]),
                       "objects": int(len(outline.objects(labels, meta["name"]))), "people": sorted({p for ps in done.values() for p in ps}),
                       "link": annotate_path(meta["name"]), "tmas": meta.get("benchmark", {}).get("tmas", [])})
    path = report.report_path(state.project)
    try:
        scores = cache.memo(("segmentation report", str(path)), [path], lambda: json.loads(path.read_text()))
    except FileNotFoundError:
        scores = None
    folder = proposals.folder(state.project)
    files = [{"label": label, "path": state.project.relative(folder / name), "exists": (folder / name).exists()}
             for label, name in (("Scores (JSON)", "report.json"), ("One row per outlined object", "objects.csv"),
                                 ("One row per counted object", "detections.csv"), ("One row per field", "fields.csv"))]
    return {"stains": present, "checks": checks, "report": scores, "files": files, "segment_anything": state.project.model("sam").exists(),
            "tma_study": state.project.is_tma, "now": time.time(), "imported_sessions": len(report.imported_sessions(state.project))}
