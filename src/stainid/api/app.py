"""FastAPI application: JSON API under /api and the built React app at /.

Several projects can be open at once. A request names its project in the address: `/api/projects/<id>/summary` is
`/api/summary` for that project (`ProjectScope` takes the id off and sets it for the request; the routes call
`state.get_state()`). An address without a project (`/api/summary`, as older web apps send) uses the project opened
last. `/api/session` and `/api/projects` (the projects list) belong to the server as a whole.
"""
from __future__ import annotations

import contextlib
import re
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from stainid.api import jobs as job_queue
from stainid.api import state
from stainid.api.routes import analysis, annotate, cohort, images, imageset, jobs, models, project, projects, session, setup, slides
from stainid.api.routes import areas as area_routes
from stainid.api.routes import cache as cache_routes
from stainid.api.routes import classes as label_classes
from stainid.api.routes import outline as outline_routes
from stainid.api.routes import rounds as round_routes
from stainid.api.session import allowed_origins, is_admin

STATIC = Path(__file__).parent / "static"
PUBLIC_API = re.compile(r"^/api/(session|annotate)(/|$)")
SCOPED = re.compile(r"^/api/projects/([A-Za-z0-9][A-Za-z0-9_.-]*)(/.+)$")


class ProjectScope:
    """Turns `/api/projects/<id>/<route>` into `/api/<route>` handled for project <id> (see `state.CURRENT`)."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        match = SCOPED.match(scope.get("path", "")) if scope["type"] in ("http", "websocket") else None
        if match is None:
            await self.app(scope, receive, send)
            return
        path = "/api" + match.group(2)
        with state.use_project(match.group(1)):
            await self.app({**scope, "path": path, "raw_path": path.encode()}, receive, send)


REQUEST_THREADS = 64  # requests answered at once (sync routes run on threads; slow work goes to `work`'s pool)


@contextlib.asynccontextmanager
async def _lifespan(app: FastAPI):
    import anyio.to_thread  # noqa: PLC0415

    anyio.to_thread.current_default_thread_limiter().total_tokens = REQUEST_THREADS
    from stainid.api import area_batch  # noqa: PLC0415

    area_batch.start()  # "Apply area edits" by itself, at most every 10 minutes (`api.area_batch`)
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="stainID", version="2.0", docs_url=None, redoc_url=None, openapi_url=None, lifespan=_lifespan)
    for module in (projects, project, slides, setup, imageset, cohort, images, jobs, analysis, models, session, annotate, cache_routes, label_classes, outline_routes, round_routes, area_routes):
        app.include_router(module.router, prefix="/api")
    job_queue.SLOTS.elsewhere = state.running_elsewhere  # heavy steps of known projects that are not open count too

    @app.exception_handler(state.UnknownProject)
    async def unknown_project(request: Request, error: state.UnknownProject) -> JSONResponse:
        return JSONResponse({"detail": "This project is not on this server's list (it may have been removed). Open it from the Projects page."},
                            status_code=404)

    @app.middleware("http")
    async def admin_only(request: Request, call_next):
        """Everything but sign-in and the annotation pages needs the admin password."""
        if request.url.path.startswith("/api") and not PUBLIC_API.match(request.url.path) and not is_admin(request):
            return JSONResponse({"detail": "admin sign-in required"}, status_code=401)
        return await call_next(request)

    origins = allowed_origins()
    if origins:  # a web app served from these addresses may use this server (`stainid serve --allow-origin`)
        app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"],
                           allow_headers=["Authorization", "Content-Type"])
    app.add_middleware(ProjectScope)  # outermost: every other part sees the address without the project

    if (STATIC / "index.html").exists():
        app.mount("/assets", StaticFiles(directory=STATIC / "assets"), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str) -> FileResponse:
            target = (STATIC / path).resolve()
            return FileResponse(target if path and target.is_file() and target.is_relative_to(STATIC.resolve()) else STATIC / "index.html")

    return app
