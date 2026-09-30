"""FastAPI application: JSON API under /api and the built React app at /."""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from stainid.api.routes import analysis, cohort, images, jobs, models, project, reviews, slides

STATIC = Path(__file__).parent / "static"


def create_app() -> FastAPI:
    app = FastAPI(title="stainID", version="2.0")
    for module in (project, slides, cohort, images, jobs, reviews, analysis, models):
        app.include_router(module.router, prefix="/api")
    if (STATIC / "index.html").exists():
        app.mount("/assets", StaticFiles(directory=STATIC / "assets"), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str) -> FileResponse:
            target = STATIC / path
            return FileResponse(target if path and target.is_file() else STATIC / "index.html")

    return app
