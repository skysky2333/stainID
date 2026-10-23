"""FastAPI application: JSON API under /api and the built React app at /.

Several projects can be open at once. A request names its project in the address: `/api/projects/<id>/summary` is
`/api/summary` for that project (`ProjectScope` takes the id off and sets it for the request; the routes call
`state.get_state()`). An address without a project (`/api/summary`, as older web apps send) uses the project opened
last. `/api/session` and `/api/projects` (the projects list) belong to the server as a whole.
"""
from __future__ import annotations

import contextlib
import logging
import re
import secrets
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from stainid import palette, practice, tutorial
from stainid.api import ahead, state, work
from stainid.api.routes import accounts as account_routes
from stainid.api.routes import analysis, annotate, cohort, images, imageset, jobs, models, project, projects, session, setup, slides
from stainid.api.routes import areas as area_routes
from stainid.api.routes import matter as matter_routes
from stainid.api.routes import cache as cache_routes
from stainid.api.routes import classes as label_classes
from stainid.api.routes import outline as outline_routes
from stainid.api.routes import rounds as round_routes
from stainid.api.routes import users as user_routes
from stainid.api.session import GUEST_COOKIE, GUEST_DAYS, allowed_origins, current_account, guest_id, is_admin
from stainid.labels import store

STATIC = Path(__file__).parent / "static"
log = logging.getLogger(__name__)
PUBLIC_API = re.compile(r"^/api/(session|annotate|tutorial|palette)(/|$)")
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

    from stainid.jobs import queue  # noqa: PLC0415

    anyio.to_thread.current_default_thread_limiter().total_tokens = REQUEST_THREADS
    queue.shared().start_thread()  # the shared job queue runs here unless `stainid jobs serve` (or another app) runs it
    ahead.start()  # idle-time preparation of the open sessions' fields (the lowest priority of all)
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="stainID", version="2.0", docs_url=None, redoc_url=None, openapi_url=None, lifespan=_lifespan)
    for module in (projects, project, slides, setup, imageset, cohort, images, jobs, analysis, models, session, annotate, cache_routes, label_classes, outline_routes, round_routes, area_routes, matter_routes, account_routes, user_routes):
        app.include_router(module.router, prefix="/api")

    @app.get("/api/palette")
    def colours() -> dict:
        """Every colour the pages draw with (`stainid.palette`; the web app bundles the same file)."""
        return palette.palette()

    @app.exception_handler(state.UnknownProject)
    async def unknown_project(request: Request, error: state.UnknownProject) -> JSONResponse:
        return JSONResponse({"detail": "This project is not on this server's list (it may have been removed). Open it from the Projects page."},
                            status_code=404)

    @app.exception_handler(practice.NotKept)
    async def not_kept(request: Request, error: practice.NotKept) -> JSONResponse:
        """A store write on a practice session (`stainid.practice`): refused, nothing changed."""
        log.warning("%s %s: %s", request.method, request.url.path, error)
        return JSONResponse({"detail": practice.NOT_KEPT}, status_code=403)

    @app.exception_handler(Exception)
    async def unexpected(request: Request, error: Exception) -> JSONResponse:
        """An error nobody expected: logged with its detail; the admin sees what it was, annotators a plain message."""
        log.error("%s %s failed", request.method, request.url.path, exc_info=error)
        detail = f"{type(error).__name__}: {error}" if is_admin(request) else work.PLAIN_ERROR
        return JSONResponse({"detail": detail}, status_code=500)

    @app.middleware("http")
    async def admin_only(request: Request, call_next):
        """Everything but sign-in and the annotation pages needs an admin account (annotators: 403). Labels saved while
        answering carry the account (`store.ATTRIBUTION`)."""
        if not request.url.path.startswith("/api"):
            return await call_next(request)
        account = current_account(request)
        if not PUBLIC_API.match(request.url.path):
            if account is None:
                return JSONResponse({"detail": "sign in first"}, status_code=401)
            if not account.admin or account.must_change:
                return JSONResponse({"detail": "choose a new password first" if account.must_change else "only admins can do this"}, status_code=403)
        token = store.ATTRIBUTION.set(account.id if account is not None else None)
        new_guest = secrets.token_hex(16) if account is None and guest_id(request) is None else None  # a guest's own id
        asking = work.ASKER.set(account.id if account is not None else f"guest:{new_guest or guest_id(request)}")
        on_practice = bool(practice.PATH.match(request.url.path))
        practising = practice.PRACTISING.set(on_practice)  # the tutorial: no store is written
        in_flight = ahead.request_started()  # a request that takes a while: the idle-time preparation waits
        try:
            if on_practice and tutorial.built():  # the one tutorial of the server, whatever the project (`stainid.tutorial`)
                with state.use_project(state.TUTORIAL):
                    response = await call_next(request)
            else:
                response = await call_next(request)
        finally:
            ahead.request_ended(in_flight)
            practice.PRACTISING.reset(practising)
            work.ASKER.reset(asking)
            store.ATTRIBUTION.reset(token)
        if new_guest:
            secure = request.url.scheme == "https" or request.headers.get("x-forwarded-proto", "") == "https"
            response.set_cookie(GUEST_COOKIE, new_guest, max_age=GUEST_DAYS * 86400, httponly=True, samesite="lax", secure=secure)
        return response

    origins = allowed_origins()
    if origins:  # a web app served from these addresses may use this server (`stainid serve --allow-origin`)
        app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"],
                           allow_headers=["Authorization", "Content-Type"])
    app.add_middleware(ProjectScope)  # outermost: every other part sees the address without the project

    if (STATIC / "index.html").exists():
        app.mount("/assets", StaticFiles(directory=STATIC / "assets"), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str) -> FileResponse:
            if path == "api" or path.startswith("api/"):  # an address of the API that does not exist: not the page
                raise HTTPException(404, "not found")
            target = (STATIC / path).resolve()
            return FileResponse(target if path and target.is_file() and target.is_relative_to(STATIC.resolve()) else STATIC / "index.html")

    return app
