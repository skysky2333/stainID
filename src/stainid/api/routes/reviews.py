from __future__ import annotations

import re
from pathlib import Path

import cv2
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

from stainid.api import render
from stainid.api.state import get_state
from stainid.review import store
from stainid.review.sampling import DEFAULT_FOV_UM, DEFAULT_LABELS, sample_items

router = APIRouter(tags=["reviews"])


def root() -> Path:
    return get_state().project.output("reviews")


def safe(name: str) -> str:
    if ".." in name or name.startswith("/"):
        raise HTTPException(400, "invalid review name")
    return name


class NewReview(BaseModel):
    name: str
    title: str = ""
    stain: str
    n: int = 60
    strategy: str = "random"
    model_class: str | None = None
    low: float = 0.3
    high: float = 0.7
    per_group: bool = True
    label_options: list[str] | None = None
    instructions: str = ""
    seed: int = 0


class Label(BaseModel):
    review_id: str
    label: str
    confidence: str = ""
    notes: str = ""
    reviewer: str = ""


@router.get("/reviews")
def list_reviews() -> list[dict]:
    return store.list_sets(root())


@router.get("/reviews/defaults")
def defaults() -> dict:
    return {"labels": DEFAULT_LABELS, "fov_um": DEFAULT_FOV_UM}


@router.post("/reviews")
def create(request: NewReview) -> dict:
    if not re.fullmatch(r"[A-Za-z0-9_\-]+", request.name):
        raise HTTPException(400, "name may contain letters, digits, - and _ only")
    state = get_state()
    items = sample_items(state.project, state.tiles, request.stain, request.n, request.strategy, request.model_class, request.low, request.high,
                         request.per_group, request.seed)
    if items.empty:
        raise HTTPException(400, "no objects match; run the pipeline for this stain first")
    try:
        store.create_set(root(), f"reviews/{request.name}", request.title or request.name, request.stain, items,
                         request.label_options or DEFAULT_LABELS[request.stain], request.instructions, DEFAULT_FOV_UM[request.stain], request.seed)
    except FileExistsError:
        raise HTTPException(409, "a review set with this name exists")
    return {"name": f"reviews/{request.name}", "items": len(items)}


@router.get("/reviews/{name:path}/items")
def items(name: str) -> dict:
    return store.load_set(root(), safe(name))


@router.post("/reviews/{name:path}/labels")
def label(name: str, request: Label) -> dict:
    store.save_label(root(), safe(name), request.review_id, request.label, request.confidence, request.notes, request.reviewer)
    return {"ok": True}


@router.get("/reviews/{name:path}/summary")
def summary(name: str) -> dict:
    return store.unblinded_summary(root(), safe(name))


@router.get("/reviews/{name:path}/image/{review_id}.jpg")
def item_image(name: str, review_id: str) -> Response:
    folder = root() / safe(name)
    legacy = folder / "patches" / f"{review_id}.jpg"
    if legacy.exists():
        return FileResponse(legacy)
    data = store.load_set(root(), name, include_hidden=True)
    item = next((i for i in data["items"] if i["review_id"] == review_id), None)
    if item is None:
        raise HTTPException(404, review_id)
    state = get_state()
    rgb = render.field_rgb(state, item["tile_id"])
    half = int(data["meta"]["fov_um"] / state.project.pixel_size_um / 2)
    x, y = int(item["x"]), int(item["y"])
    crop = rgb[max(0, y - half): y + half, max(0, x - half): x + half]
    ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(crop, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
    return Response(encoded.tobytes(), media_type="image/jpeg")
