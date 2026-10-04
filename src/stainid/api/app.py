"""FastAPI application: JSON API under /api and the built React app at /."""
from __future__ import annotations

import re
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from stainid.api.routes import analysis, annotate, cohort, images, jobs, models, project, reviews, session, slides
from stainid.api.session import is_admin

STATIC = Path(__file__).parent / "static"
PUBLIC_API = re.compile(r"^/api/(session|annotate)(/|$)")


def create_app() -> FastAPI:
    app = FastAPI(title="stainID", version="2.0", docs_url=None, redoc_url=None, openapi_url=None)
    for module in (project, slides, cohort, images, jobs, reviews, analysis, models, session, annotate):
        app.include_router(module.router, prefix="/api")

    @app.middleware("http")
    async def admin_only(request: Request, call_next):
        """Everything but sign-in and the annotation pages needs the admin password."""
        if request.url.path.startswith("/api") and not PUBLIC_API.match(request.url.path) and not is_admin(request):
            return JSONResponse({"detail": "admin sign-in required"}, status_code=401)
        return await call_next(request)

    if (STATIC / "index.html").exists():
        app.mount("/assets", StaticFiles(directory=STATIC / "assets"), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str) -> FileResponse:
            target = (STATIC / path).resolve()
            return FileResponse(target if path and target.is_file() and target.is_relative_to(STATIC.resolve()) else STATIC / "index.html")

    return app
