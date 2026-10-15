"""The label classes of each stain (`stainid.labels.vocabulary`): read and change them in the project settings, and the
example picture of each class.

Example pictures are kept with the labels, `<reviews>/class_examples/<stain>/<file>.jpg` (they are part of the
project's labelling instructions, not a cache), and the people labelling see them, so they are served under the
public /api/annotate addresses.
"""
from __future__ import annotations

import copy
import io
import re
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from stainid.api import label_crops
from stainid.api.routes.annotate import viewer
from stainid.api.session import require_admin
from stainid.api.state import get_state, reload_project
from stainid.labels import vocabulary as vocab
from stainid.project import Project, save_project

router = APIRouter(tags=["label classes"])
admin = [Depends(require_admin)]
EXAMPLE_PX = 480  # example pictures are kept at most this large
MAX_UPLOAD = 20 * 1024 * 1024
FILE = re.compile(r"[A-Za-z0-9_\-]+\.jpg")


class ClassesChange(BaseModel):
    classes: list[dict] | None = None  # None: back to the built-in classes
    subtype_wait_s: float | None = None


class ObjectSpot(BaseModel):
    tile_id: str
    x: float
    y: float


def _stain(stain: str) -> str:
    if stain not in get_state().project.stains:
        raise HTTPException(404, f"this project has no stain {stain}")
    return stain


def examples_folder(root: Path, stain: str) -> Path:
    return root / "class_examples" / stain


def all_classes() -> dict:
    project = get_state().project
    return {"stains": {stain: vocab.of(project, stain).to_api() for stain in project.stains},
            "subtype_wait_s": float((project.config.get("labels") or {}).get("subtype_wait_s", vocab.DEFAULT_SUBTYPE_WAIT_S)),
            "builtin": {stain: vocab.defaults(stain).to_api() for stain in project.stains}}


@router.get("/labels/classes", dependencies=admin)
def get_classes() -> dict:
    """Every stain's classes (as used now, and as built in), and how long a type may follow a class key."""
    return all_classes()


@router.put("/labels/classes/{stain}", dependencies=admin)
def set_classes(stain: str, body: ClassesChange) -> dict:
    """Save a stain's classes (checked first; built-in classes keep their role and come back if left out), or go
    back to the built-in ones (`classes` null)."""
    _stain(stain)
    project = get_state().project
    settings = copy.deepcopy(project.config.get("labels") or {})
    stored = dict(settings.get("classes") or {})
    if body.classes is not None:
        entries = vocab.entry_of(body.classes)
        chosen = vocab.from_config(stain, entries)
        issues = vocab.problems(chosen)
        if issues:
            raise HTTPException(400, " ".join(issues))
        if chosen == vocab.defaults(stain):
            stored.pop(stain, None)
        else:
            stored[stain] = chosen.to_config()
    else:
        stored.pop(stain, None)
    settings["classes"] = stored
    if body.subtype_wait_s is not None:
        if not 0 <= body.subtype_wait_s <= 10:
            raise HTTPException(400, "The time to pick a type must be between 0 and 10 seconds.")
        settings["subtype_wait_s"] = float(body.subtype_wait_s)
    reload_project(save_project(Project(project.root, {**project.config, "labels": settings}), {}).root)
    return all_classes()


def _save_example(stain: str, image: Image.Image) -> dict:
    folder = examples_folder(get_state().project.output("reviews"), stain)
    folder.mkdir(parents=True, exist_ok=True)
    name = f"{uuid.uuid4().hex[:12]}.jpg"
    image = image.convert("RGB")
    image.thumbnail((EXAMPLE_PX, EXAMPLE_PX))
    tmp = folder / f".{name}.tmp"
    image.save(tmp, "JPEG", quality=90)
    tmp.replace(folder / name)
    return {"file": name}


@router.post("/labels/classes/{stain}/examples", dependencies=admin)
async def upload_example(stain: str, request: Request) -> dict:
    """An example picture sent as the request body (a JPEG, PNG or TIFF file); answers its file name, which the
    class's `example` then names."""
    _stain(stain)
    data = await request.body()
    if not data or len(data) > MAX_UPLOAD:
        raise HTTPException(400, "Choose a picture file (JPEG, PNG or TIFF) of at most 20 MB.")
    try:
        with Image.open(io.BytesIO(data)) as image:
            return _save_example(stain, image)
    except (UnidentifiedImageError, OSError) as error:
        raise HTTPException(400, "This file is not a picture stainID can read; use a JPEG, PNG or TIFF.") from error


@router.post("/labels/classes/{stain}/examples/from-object", dependencies=admin)
def example_from_object(stain: str, body: ObjectSpot) -> dict:
    """A labelled object's crop (as the labels view shows it) as an example picture."""
    _stain(stain)
    try:
        crop = label_crops.get(get_state(), label_crops.Spot(stain, body.tile_id, body.x, body.y))
    except label_crops.NotReady as error:
        raise HTTPException(409, "The picture of this object is still being made; try again in a moment.") from error
    except (KeyError, FileNotFoundError) as error:
        raise HTTPException(404, "There is no picture of this object.") from error
    with Image.open(crop) as image:
        return _save_example(stain, image)


@router.get("/annotate/class-examples/{stain}/{file}")
def example(stain: str, file: str, person: str | None = Depends(viewer)) -> FileResponse:
    """An example picture of a class (public: the people labelling see them)."""
    path = examples_folder(get_state().project.output("reviews"), stain) / file
    if not FILE.fullmatch(file) or stain not in get_state().project.stains or not path.is_file():
        raise HTTPException(404, "no such example picture")
    return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=86400"})
