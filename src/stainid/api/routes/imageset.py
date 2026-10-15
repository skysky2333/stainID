"""Image-set and whole-slide projects: the Images screen of the guided setup (the image list and the imaging settings),
the imported labels of a dataset package, and scoring the models on them. Thin routes over `stainid.imageset`."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import cv2
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

from stainid.api.state import get_state, reload_project
from stainid.imageset import annotations, package, scoring, table
from stainid.imaging import chromogens, settings, sources
from stainid.imaging.thumbnails import jpeg
from stainid.project import KINDS, save_project
from stainid.setup.files import read_tables

router = APIRouter(tags=["image sets"])
SHOWN = 500  # image list rows sent to the page at once
PICTURE_PX = 640
# header words that name the image list's columns in a lab's own spreadsheet
HEADERS = {"path": ("path", "file", "filename", "file name", "image", "image path", "slide", "slide path", "location"),
           "stain": ("stain", "antibody", "marker", "immunostain"), "donor_id": ("donor", "donor id", "case", "case id", "subject", "brain id", "id"),
           "region": ("region", "brain region", "area"), "image_id": ("image id", "name"), "slide_id": ("slide id", "slide name", "block"),
           "site": ("site", "lab", "centre", "center"), "batch": ("batch", "run", "staining batch"), "pixel_size_um": ("pixel size", "mpp", "um per pixel", "µm per pixel"),
           "disease_group": ("group", "diagnosis", "disease group"), "split": ("split", "set", "subset")}


class Rows(BaseModel):
    rows: list[dict]
    mode: str = "replace"  # or "add": keep the images already listed


class Folder(BaseModel):
    folder: str


class Imaging(BaseModel):
    native_pixel_size_um: float | None = None
    rescale: str = "on_import"
    stains: dict[str, dict] = {}


class ClassMap(BaseModel):
    class_map: dict[str, str]


def _project():
    project = get_state().project
    if project.is_tma:
        raise HTTPException(400, "This is a TMA study: its images are the cores found on the slides")
    return project


def _stains(project, rows: list[dict]) -> list[dict]:
    used = Counter(r.get("stain", "") for r in rows)
    out = []
    for stain in sorted(set(used) | set(settings.stain_settings(project))):
        own = settings.setting(project, stain)
        out.append({"stain": stain, "images": used.get(stain, 0), "family": own["family"], "chromogen": own["chromogen"], "others": own["others"],
                    "analysed_as": settings.analysis_stain(project, stain), "set": stain in settings.stain_settings(project),
                    "guess": settings.family_of(stain)})
    return out


@router.get("/setup/images")
def overview() -> dict:
    project = _project()
    rows = table.read_images(project)
    imaging = project.config.get("imaging", {})
    return {"kind": project.kind, "kind_label": KINDS[project.kind], "total": len(rows), "rows": rows[:SHOWN],
            "donors": len({r["donor_id"] for r in rows if r.get("donor_id")}), "stains": _stains(project, rows),
            "problems": table.problems(project, rows), "families": settings.FAMILIES, "chromogens": chromogens.chromogen_choices(),
            "imaging": {"native_pixel_size_um": imaging.get("native_pixel_size_um"), "rescale": settings.rescale_mode(project),
                        "analysis_pixel_size_um": project.pixel_size_um},
            "pixel_sizes": dict(Counter(r.get("pixel_size_um") or "" for r in rows)), "readers": sources.available(),
            "path": project.relative(project.input("image_list")), "dataset": project.config.get("dataset") or {}}


@router.post("/setup/images/scan")
def scan(request: Folder) -> dict:
    project = _project()
    known = sorted(set(settings.stain_settings(project)) | set(project.stains))
    try:
        rows = table.scan_folder(Path(request.folder), known, project.kind == "slides")
    except ValueError as error:
        raise HTTPException(400, str(error))
    return {"rows": rows, "found": len(rows), "limit": table.SCAN_LIMIT}


def _column(header: str) -> str:
    words = " ".join(header.strip().lower().replace("_", " ").split())
    return next((column for column, names in HEADERS.items() if words in names), "")


@router.post("/setup/images/read")
async def read_list(file: UploadFile = File(...)) -> dict:
    """An image list from the lab's own spreadsheet (CSV, Excel): its columns matched by their names."""
    try:
        tables = read_tables(file.filename or "file", await file.read())
    except ValueError as error:
        raise HTTPException(400, str(error))
    best = max(tables, key=lambda t: len(t.rows), default=None)
    if best is None or len(best.rows) < 2:
        raise HTTPException(400, "The file holds no table with a header row and at least one image")
    header, body = best.rows[0], best.rows[1:]
    matched = {i: _column(h) for i, h in enumerate(header)}
    if "path" not in matched.values():
        raise HTTPException(400, "No column says where each image is (call it path or file)")
    rows = [{matched[i]: value for i, value in enumerate(row) if i in matched and matched[i]} for row in body]
    return {"rows": [r for r in rows if r.get("path")], "columns": {header[i]: c for i, c in matched.items()}}


@router.put("/setup/images")
def save(request: Rows) -> dict:
    project = _project()
    rows = request.rows if request.mode == "replace" else table.read_images(project) + request.rows
    try:
        count = table.save_images(project, rows)
    except ValueError as error:
        raise HTTPException(400, str(error))
    reload_project(project.root)
    return {"saved": count}


@router.put("/setup/images/imaging")
def save_imaging(request: Imaging) -> dict:
    project = _project()
    if request.rescale not in ("on_import", "on_the_fly"):
        raise HTTPException(400, "rescale must be on_import or on_the_fly")
    stains = {}
    for stain, own in request.stains.items():
        family = own.get("family") or ""
        if family and family not in settings.FAMILIES:
            raise HTTPException(400, f"{stain}: unknown kind of stain {family}")
        chromogen = chromogens.canonical(own.get("chromogen")) or "DAB"
        others = [c for c in (chromogens.canonical(o) for o in own.get("others") or []) if c and c != chromogen]
        try:
            chromogens.recipe(chromogen, others)
        except ValueError as error:
            raise HTTPException(400, f"{stain}: {error}")
        stains[stain] = {k: v for k, v in {"family": family, "chromogen": chromogen, "others": others, "antibody": own.get("antibody"),
                                           "vectors": own.get("vectors")}.items() if v}
    imaging = {"native_pixel_size_um": request.native_pixel_size_um, "rescale": request.rescale, "stains": stains}
    config = {**project.config, "imaging": {}}  # replace the section as a whole, so a stain taken out is gone
    from stainid.project import Project

    reload_project(save_project(Project(root=project.root, config=config), {"imaging": imaging}).root)
    return overview()


def _image(project, image_id: str) -> dict:
    row = next((r for r in table.read_images(project) if r.get("image_id") == image_id), None)
    if row is None:
        raise HTTPException(404, f"no image {image_id} in the image list")
    return row


@router.get("/setup/images/{image_id}/picture.jpg")
def picture(image_id: str, analysis: bool = False) -> Response:
    """A small picture of an image (or whole slide) as it is, or as stainID reads it (its stain of interest as brown)."""
    project = _project()
    row = _image(project, image_id)
    try:
        source = sources.open_source(table.resolve(project, row))
        rgb, _ = source.thumbnail(PICTURE_PX)
    except (sources.MissingReader, FileNotFoundError, OSError, ValueError) as error:
        raise HTTPException(404, str(error))
    if analysis:
        rgb = chromogens.to_analysis_rgb(rgb, settings.recipe_for(project, row.get("stain", "")))
    return Response(jpeg(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), 85), media_type="image/jpeg", headers={"Cache-Control": "private, max-age=60"})


@router.get("/setup/images/{image_id}/file")
def file_info(image_id: str) -> dict:
    project = _project()
    return sources.describe(table.resolve(project, _image(project, image_id)))


# ---------------------------------------------------------------- dataset packages and imported labels

@router.get("/labels/imported")
def imported_labels() -> dict:
    """What was imported from the project's dataset package: raters, sessions, notes and the class map."""
    project = get_state().project
    dataset = project.config.get("dataset") or {}
    if not dataset.get("package"):
        return {"dataset": None}
    folder = annotations.folder(project)
    summary = folder / "import.json"
    placed = annotations.imported(project)
    try:
        preview = package.preview(dataset["package"])
    except (ValueError, OSError) as error:
        preview = {"error": str(error)}
    return {"dataset": {k: dataset.get(k, "") for k in ("id", "title", "licence", "citation", "doi", "package")},
            "class_map": dataset.get("class_map") or {}, "labels": preview.get("labels", {}), "classes": package.class_choices(),
            "import": json.loads(summary.read_text()) if summary.exists() else None,
            "placed": {r: int(n) for r, n in placed.rater.value_counts().items()} if len(placed) else {},
            "scores": scoring.saved(project)}


@router.put("/labels/imported/class-map")
def set_class_map(request: ClassMap) -> dict:
    """Change which stainID class each of the package's labels means, and place the labels again."""
    project = get_state().project
    dataset = project.config.get("dataset") or {}
    if not dataset.get("package"):
        raise HTTPException(400, "This project was not made from a dataset package")
    project = save_project(project, {"dataset": {**dataset, "class_map": request.class_map}})
    reload_project(project.root)
    if project.analysis_tiles().exists():
        try:
            annotations.import_package_labels(project, log=lambda *a, **k: None)
        except (ValueError, OSError) as error:
            raise HTTPException(400, str(error))
    return imported_labels()


@router.get("/models/score")
def saved_scores() -> dict:
    project = get_state().project
    return {"scores": scoring.saved(project), "has_labels": not annotations.imported(project).empty}


@router.post("/models/score")
def score_models() -> dict:
    """Score the models in use on the imported labels (takes seconds to a minute)."""
    try:
        return {"scores": scoring.score(get_state().project), "has_labels": True}
    except ValueError as error:
        raise HTTPException(400, str(error))


@router.get("/models/score.csv")
def scores_file() -> FileResponse:
    path = scoring.scores_csv(get_state().project)
    if not path.exists():
        raise HTTPException(404, "No scores yet: score the models first")
    return FileResponse(path, media_type="text/csv", filename=f"scores_{get_state().project.config.get('dataset', {}).get('id', 'labels')}.csv")
