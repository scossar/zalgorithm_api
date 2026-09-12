"""HTML API, with explicit app lifecycle and replaceable backend factory."""

import logging
from contextlib import asynccontextmanager
from html import escape
from typing import Annotated, Literal

import aiosqlite
from fastapi import Depends, FastAPI, Form, Path, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from starlette.concurrency import run_in_threadpool

from .backend import Backend, sqlite_uri
from .config import Settings
from .keyword import KeywordIndexUnavailable, SearchInputError

log = logging.getLogger(__name__)
CLOSE_BUTTON = '<button type="button" aria-label="Close fragment" onclick=\'this.parentNode.classList.toggle("hidden");\'>x</button>'


async def get_db_connection(request: Request):
    async with aiosqlite.connect(
        sqlite_uri(request.app.state.backend.db_path), uri=True
    ) as db:
        yield db


Database = Annotated[aiosqlite.Connection, Depends(get_db_connection)]


def create_app(settings: Settings | None = None, backend_factory=Backend) -> FastAPI:
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.backend = await run_in_threadpool(backend_factory, settings)
        try:
            yield
        finally:
            await run_in_threadpool(app.state.backend.close)

    app = FastAPI(title="Zalgorithm API", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def no_cache(request: Request, call_next):
        response = await call_next(request)
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(aiosqlite.Error)
    async def database_error(request: Request, error: aiosqlite.Error):
        log.error(
            "Fragment database unavailable",
            exc_info=(type(error), error, error.__traceback__),
        )
        return HTMLResponse(
            "<p>Fragment database is temporarily unavailable.</p>", status_code=503
        )

    @app.post("/api/query", response_class=HTMLResponse)
    async def query_collection(
        request: Request, db: Database, query: Annotated[str, Form(max_length=4096)],
        mode: Annotated[Literal["semantic", "keyword", "hybrid"], Form()] = "semantic",
        keyword_query: Annotated[str | None, Form(max_length=4096)] = None,
        include: Annotated[str | None, Form(max_length=4096)] = None,
        exclude: Annotated[str | None, Form(max_length=4096)] = None,
        keyword_syntax: Annotated[Literal["simple", "fts5"], Form()] = "simple",
    ):
        query = query.strip()
        try:
            if not query:
                return ""
            db_ids = await run_in_threadpool(
                request.app.state.backend.search, query, mode=mode,
                keyword_query=keyword_query, include=include, exclude=exclude,
                keyword_syntax=keyword_syntax,
            )
        except SearchInputError as error:
            return HTMLResponse(f"<p>{escape(str(error))}</p>", status_code=422)
        except KeywordIndexUnavailable:
            return HTMLResponse(
                "<p>Keyword search is unavailable until the fragment index is rebuilt.</p>",
                status_code=503,
            )
        except Exception:
            log.exception("Search query failed")
            return HTMLResponse(
                "<p>Search is temporarily unavailable.</p>", status_code=503
            )
        if not db_ids:
            return ""
        placeholders = ",".join("?" for _ in db_ids)
        async with db.execute(
            f"SELECT id, html_heading, html_fragment FROM sections WHERE id IN ({placeholders})",
            db_ids,
        ) as cursor:
            rows = {row[0]: row[1] + row[2] for row in await cursor.fetchall()}
        if any(db_id not in rows for db_id in db_ids):
            log.error(
                "Search index/SQLite mismatch: a result references a missing fragment"
            )
            return HTMLResponse(
                "<p>Search data is temporarily unavailable.</p>", status_code=503
            )
        return "".join(rows[db_id] for db_id in db_ids)

    @app.get("/api/fragment/{row_id}", response_class=HTMLResponse)
    async def get_fragment(row_id: Annotated[int, Path(gt=0)], db: Database):
        async with db.execute(
            "SELECT html_heading, html_fragment FROM sections WHERE id = ?", (row_id,)
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return HTMLResponse(
                "<p>This fragment is no longer available.</p>", status_code=404
            )
        return CLOSE_BUTTON + row[0] + row[1]

    return app


app = create_app()
