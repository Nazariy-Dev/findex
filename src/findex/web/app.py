"""FastAPI-застосунок: API + сторінка Jinja2. Індекс завантажується один раз у lifespan."""

from __future__ import annotations

import asyncio
import html
import logging
import math
import time
import uuid
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from findex import __version__
from findex.query import QuerySyntaxError, parse
from findex.ranking import (
    BM25,
    Index,
    Scorer,
    SearchResult,
    TfIdf,
    load_index,
    make_snippet,
    rank,
    render_snippet,
)
from findex.semantic import HybridSearcher
from findex.web.schemas import DocumentOut, Health, SearchResponse, SearchResultOut, StatsOut
from findex.web.settings import Settings

log = logging.getLogger("findex.web")
HERE = Path(__file__).parent
SCORERS: dict[str, Scorer] = {"bm25": BM25(), "tfidf": TfIdf()}
PAGE_SIZE_MAX = 50


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_hybrid(request: Request) -> HybridSearcher | None:
    return getattr(request.app.state, "hybrid", None)


def get_index(request: Request) -> Index:
    index: Index | None = getattr(request.app.state, "index", None)
    if index is None:
        raise HTTPException(503, "index is not loaded")
    return index


IndexDep = Annotated[Index, Depends(get_index)]
HybridDep = Annotated[HybridSearcher | None, Depends(get_hybrid)]
QParam = Annotated[str, Query(min_length=1, max_length=200)]
KParam = Annotated[int, Query(ge=1, le=PAGE_SIZE_MAX)]
PageParam = Annotated[int, Query(ge=1, le=100)]
ScorerParam = Literal["bm25", "tfidf"]
ModeParam = Literal["keyword", "semantic", "hybrid"]


def run_search(
    index: Index,
    q: str,
    scorer: str,
    k: int,
    page: int,
    mode: str = "keyword",
    hybrid: HybridSearcher | None = None,
) -> SearchResponse:
    """Чиста функція поверх ЛР3: пошук, посторінковий зріз, сніпети тільки для сторінки."""
    t0 = time.perf_counter()
    try:
        node = parse(q)
        ids = node.evaluate(index)
    except QuerySyntaxError as e:
        raise HTTPException(422, f"bad query: {e}") from None
    terms = node.terms()
    ranked = rank(index, terms, ids, SCORERS[scorer], max(k * page, 50))
    if mode != "keyword":
        if hybrid is None:
            raise HTTPException(503, "semantic index is not available")
        pairs = (
            hybrid.semantic(q, k * page)
            if mode == "semantic"
            else hybrid.hybrid([r.doc_id for r in ranked], q, k * page)
        )
        ranked = [SearchResult(sc, -d, d, index.meta(d).title) for d, sc in pairs]
        ids = {d for d, _ in pairs}  # total = довжина злитого / семантичного рейтингу
    chunk = ranked[(page - 1) * k : page * k]
    results = [
        SearchResultOut(
            doc_id=r.doc_id,
            title=r.title,
            path=index.meta(r.doc_id).path,
            score=round(r.score, 4),
            # HTML-безпечний сніпет: текст екранований, знайдені терми в <mark>
            snippet=render_snippet(
                make_snippet(index.text(r.doc_id), terms), "<mark>", "</mark>", html.escape
            ),
        )
        for r in chunk
    ]
    return SearchResponse(
        query=q,
        scorer=scorer if mode == "keyword" else mode,
        total=len(ids),
        page=page,
        pages=max(1, math.ceil(len(ids) / k)),
        took_ms=round((time.perf_counter() - t0) * 1000, 2),
        results=results,
    )


def create_app(settings: Settings | None = None, load: bool = True) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
        app.state.settings = settings
        app.state.started = time.time()
        if load:
            # завантаження блокує на ~1 с: виносимо в потік, цикл подій лишається вільним
            app.state.index = await asyncio.to_thread(load_index, settings.index_path)
            log.info("index loaded from %s: %r", settings.index_path, app.state.index)
            app.state.hybrid = await asyncio.to_thread(HybridSearcher.open, settings.index_path)
        yield
        # закриваємо лише той індекс, який завантажили самі (не підставлений ззовні в тестах)
        if load:
            index: Index | None = getattr(app.state, "index", None)
            if index is not None:
                index.close()

    app = FastAPI(title="findex", version=__version__, lifespan=lifespan)
    app.state.settings = settings
    templates = Jinja2Templates(directory=HERE / "templates")
    app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")

    @app.middleware("http")
    async def request_id(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        try:
            response = await call_next(request)
        except Exception:
            # стек — у лог разом з id; назовні — лише чиста 500
            log.exception("unhandled error, request_id=%s path=%s", rid, request.url.path)
            response = JSONResponse({"detail": "internal server error", "request_id": rid}, 500)
        response.headers["x-request-id"] = rid
        return response

    if settings.search_mode == "async":
        # варіант для заміру: CPU-робота прямо в корутині блокує цикл подій
        @app.get("/search", response_model=SearchResponse)
        async def search_async(
            index: IndexDep,
            hybrid: HybridDep,
            q: QParam,
            k: KParam = 10,
            scorer: ScorerParam = "bm25",
            page: PageParam = 1,
            mode: ModeParam = "keyword",
        ) -> SearchResponse:
            return run_search(index, q, scorer, k, page, mode, hybrid)
    else:
        # звичайний def: FastAPI виконує його в пулі потоків, цикл подій не блокується
        @app.get("/search", response_model=SearchResponse)
        def search(
            index: IndexDep,
            hybrid: HybridDep,
            q: QParam,
            k: KParam = 10,
            scorer: ScorerParam = "bm25",
            page: PageParam = 1,
            mode: ModeParam = "keyword",
        ) -> SearchResponse:
            return run_search(index, q, scorer, k, page, mode, hybrid)

    @app.get("/docs/{doc_id}", response_model=DocumentOut)
    def get_document(doc_id: int, index: IndexDep) -> DocumentOut:
        if doc_id not in index.raw.doc_meta:
            raise HTTPException(404, f"document {doc_id} not found")
        m = index.meta(doc_id)
        return DocumentOut(
            doc_id=doc_id,
            title=m.title,
            path=m.path,
            length=index.doc_length(doc_id),
            text=index.text(doc_id),
        )

    @app.get("/stats", response_model=StatsOut)
    def stats(index: IndexDep, request: Request) -> StatsOut:
        return StatsOut(
            documents=index.num_docs,
            terms=len(index),
            tokens=sum(index.raw.doc_lengths.values()),
            avg_doc_length=round(index.avg_doc_length, 1),
            positions=index.has_positions,
            uptime_s=round(time.time() - request.app.state.started, 1),
            version=__version__,
        )

    @app.get("/health", response_model=Health)
    def health(request: Request) -> Health | JSONResponse:
        if getattr(request.app.state, "index", None) is None:
            return JSONResponse({"status": "loading"}, status_code=503)
        return Health(status="ok")

    @app.get("/", response_class=HTMLResponse)
    def home(
        request: Request,
        q: str = "",
        scorer: ScorerParam = "bm25",
        page: PageParam = 1,
        mode: ModeParam = "keyword",
    ) -> HTMLResponse:
        index = get_index(request)
        ctx: dict[str, object] = {
            "q": q,
            "scorer": scorer,
            "mode": mode,
            "has_semantic": get_hybrid(request) is not None,
            "res": None,
            "error": None,
            "docs": index.num_docs,
            "terms": len(index),
        }
        if q.strip():
            try:
                res = run_search(index, q, scorer, 10, page, mode, get_hybrid(request))
                ctx["res"] = res
            except HTTPException as e:
                ctx["error"] = e.detail
        return templates.TemplateResponse(request, "index.html", ctx)

    @app.get("/doc/{doc_id}", response_class=HTMLResponse)
    def doc_page(doc_id: int, request: Request, q: str = "") -> HTMLResponse:
        index = get_index(request)
        doc = get_document(doc_id, index)
        return templates.TemplateResponse(request, "doc.html", {"doc": doc, "q": q})

    return app


app = create_app()
