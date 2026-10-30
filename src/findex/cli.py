"""Точка входу CLI: findex index | search | stats.

Логер налаштовується тут один раз; бібліотечні модулі лише беруть getLogger(__name__).
Діагностика йде в stderr, дані — в stdout (щоб `findex search ... --json | jq` працював).
"""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.markup import escape
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TimeElapsedColumn
from rich.table import Table

from findex import __version__
from findex.corpus import Document, iter_documents, iter_paths
from findex.index import build_index
from findex.parallel import ExecutorName, build_parallel
from findex.query import QuerySyntaxError
from findex.ranking import (
    BM25,
    Index,
    Scorer,
    SearchResult,
    TfIdf,
    load_index,
    make_snippet,
    render_snippet,
    search,
)
from findex.store import save
from findex.tokenize import tokenize

app = typer.Typer(add_completion=False, no_args_is_help=True, help="findex — пошук по корпусу")
out = Console()  # stdout: дані
err = Console(stderr=True)  # stderr: діагностика і прогрес
log = logging.getLogger("findex")

SCORERS: dict[str, Scorer] = {"bm25": BM25(), "tfidf": TfIdf()}


def fail(msg: str, code: int = 1) -> typer.Exit:
    err.print(f"[red]error:[/red] {msg}", highlight=False)
    return typer.Exit(code)


def version_cb(value: bool) -> None:
    if value:
        out.print(f"findex {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    verbose: Annotated[
        int, typer.Option("-v", "--verbose", count=True, help="-v INFO, -vv DEBUG")
    ] = 0,
    version: Annotated[
        bool, typer.Option("--version", callback=version_cb, is_eager=True, help="версія")
    ] = False,
) -> None:
    level = logging.WARNING - 10 * min(verbose, 2)
    logging.basicConfig(
        level=level,
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )


def _load(path: Path) -> Index:
    try:
        return load_index(path)
    except (FileNotFoundError, ValueError) as e:
        raise fail(str(e)) from None


@app.command("index")
def index_cmd(
    corpus: Annotated[Path, typer.Argument(help="каталог або .jsonl")],
    out_path: Annotated[Path, typer.Option("--out", "-o", help="файл індексу")] = Path("index.bin"),
    positions: Annotated[bool, typer.Option(help="зберігати позиції (для фраз)")] = True,
    limit: Annotated[int | None, typer.Option(help="лише перші N документів")] = None,
    fmt: Annotated[str, typer.Option("--format", help="pickle або binary")] = "binary",
    workers: Annotated[int, typer.Option("--workers", "-w", min=1, help="число воркерів")] = 2,
    executor: Annotated[str, typer.Option(help="serial | threads | processes")] = "serial",
) -> None:
    """Побудувати індекс і зберегти його."""
    if not corpus.exists():
        raise fail(f"corpus not found: {corpus}")
    if fmt not in ("pickle", "binary"):
        raise fail(f"unknown format {fmt!r}", 2)
    if executor not in ("serial", "threads", "processes"):
        raise fail(f"unknown executor {executor!r}", 2)
    if limit is None and executor != "serial":
        ex: ExecutorName = "threads" if executor == "threads" else "processes"
        with err.status(f"indexing with {workers} {ex}…"):
            ix, tm = build_parallel(corpus, workers, ex, positions=positions)
        log.info("map %.2f s, merge %.2f s", tm.map_s, tm.merge_s)
    else:
        total = sum(1 for _ in iter_paths(corpus))
        if limit is not None:
            total = min(total, limit)
        with Progress(
            "[progress.description]{task.description}",
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=err,
            transient=True,
        ) as bar:
            task = bar.add_task("indexing", total=total)

            def tracked() -> Iterator[Document]:
                for i, d in enumerate(iter_documents(corpus)):
                    if limit is not None and i >= limit:
                        return
                    bar.advance(task)
                    yield d

            ix = build_index(tracked(), positions=positions)
        ix.root = str(corpus.resolve())
    save(ix, out_path, "binary" if fmt == "binary" else "pickle")
    log.info("saved %s", out_path)
    err.print(f"indexed {ix.num_docs} documents, {len(ix.postings)} terms -> {out_path}")


@app.command("search")
def search_cmd(
    index_path: Annotated[Path, typer.Argument(metavar="INDEX")],
    query: Annotated[str, typer.Argument()],
    k: Annotated[int, typer.Option("--k", "-k", min=1, max=100)] = 10,
    scorer: Annotated[str, typer.Option(help="bm25 | tfidf")] = "bm25",
    as_json: Annotated[bool, typer.Option("--json", help="JSON lines у stdout")] = False,
    mode: Annotated[str, typer.Option(help="keyword | semantic | hybrid")] = "keyword",
) -> None:
    """Ранжований пошук з підсвіченими сніпетами."""
    if scorer not in SCORERS:
        raise fail(f"unknown scorer {scorer!r}; use bm25 or tfidf", 2)
    if mode not in ("keyword", "semantic", "hybrid"):
        raise fail(f"unknown mode {mode!r}", 2)
    ix = _load(index_path)
    try:
        results = search_mode(ix, index_path, query, mode, SCORERS[scorer], k)
    except QuerySyntaxError as e:
        raise fail(f"bad query: {e}", 2) from None
    except FileNotFoundError as e:
        raise fail(str(e)) from None
    if as_json:
        for r in results:
            rec = {
                "doc_id": r.doc_id,
                "score": round(r.score, 4),
                "title": r.title,
                "path": ix.meta(r.doc_id).path,
                "snippet": render_snippet(r.snippet, "**", "**"),
            }
            sys.stdout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return
    table = Table(title=f"{query!r} — {len(results)} results ({scorer})", show_lines=True)
    table.add_column("#", justify="right")
    table.add_column("score", justify="right")
    table.add_column("document")
    table.add_column("snippet")
    for i, r in enumerate(results, 1):
        snip = render_snippet(r.snippet, "[bold yellow]", "[/bold yellow]", escape)
        table.add_row(
            str(i),
            f"{r.score:.2f}",
            f"{escape(r.title)}\n[dim]{escape(ix.meta(r.doc_id).path)}[/dim]",
            snip,
        )
    out.print(table)


def search_mode(
    ix: Index, index_path: Path, query: str, mode: str, scorer: Scorer, k: int
) -> list[SearchResult]:
    """keyword — ЛР3; semantic — найкращий фрагмент; hybrid — RRF двох рейтингів."""
    if mode == "keyword":
        return search(ix, query, scorer, k=k)
    from findex.query import parse
    from findex.semantic import HybridSearcher

    hs = HybridSearcher.open(index_path)
    if hs is None:
        raise FileNotFoundError(f"no embeddings for {index_path}; run: findex embed {index_path}")
    terms = parse(query).terms() if mode == "hybrid" else list(tokenize(query))
    if mode == "semantic":
        ranked = hs.semantic(query, k)
    else:
        kw = [r.doc_id for r in search(ix, query, scorer, k=50, snippets=False)]
        ranked = hs.hybrid(kw, query, k)
    return [
        SearchResult(sc, -d, d, ix.meta(d).title, make_snippet(ix.text(d), terms))
        for d, sc in ranked
    ]


@app.command("embed")
def embed_cmd(index_path: Annotated[Path, typer.Argument(metavar="INDEX")]) -> None:
    """Порахувати ембединги фрагментів і зберегти поруч з індексом (.emb.npy)."""
    from findex.semantic import WordLlamaEmbedder, build_embeddings

    ix = _load(index_path)
    if not ix.raw.root:
        raise fail("index has no corpus root; rebuild it with findex index")
    emb_model = WordLlamaEmbedder()
    with Progress(
        "[progress.description]{task.description}",
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=err,
        transient=True,
    ) as bar:
        task = bar.add_task("embedding", total=ix.num_docs)
        emb = build_embeddings(
            iter_documents(Path(ix.raw.root)),
            emb_model,
            WordLlamaEmbedder.name,
            progress=lambda n: bar.advance(task, n),
        )
    emb.save(index_path)
    err.print(
        f"{emb.E.shape[0]} chunks x {emb.E.shape[1]} dims -> {index_path.with_suffix('.emb.npy')}"
    )


@app.command("stats")
def stats_cmd(index_path: Annotated[Path, typer.Argument(metavar="INDEX")]) -> None:
    """Статистика індексу."""
    ix = _load(index_path)
    total = sum(ix.doc_length(d) for d in ix.doc_ids)
    top = sorted(ix, key=lambda t: sum(p.tf for p in ix[t]), reverse=True)[:10]
    t = Table(show_header=False)
    for name, val in (
        ("documents", ix.num_docs),
        ("tokens", total),
        ("vocabulary", len(ix)),
        ("avg doc length", f"{ix.avg_doc_length:.1f}"),
        ("positions", ix.has_positions),
        ("top-10 terms", ", ".join(top)),
    ):
        t.add_row(name, str(val))
    out.print(t)


@app.command("crawl")
def crawl_cmd(
    seed: Annotated[str, typer.Argument(help="початковий URL")],
    out_path: Annotated[Path, typer.Option("--out", "-o")] = Path("data/crawl.jsonl"),
    max_pages: Annotated[int, typer.Option(min=1)] = 500,
    concurrency: Annotated[int, typer.Option(min=1)] = 10,
    per_host: Annotated[int, typer.Option(min=1)] = 2,
    min_delay: Annotated[
        float, typer.Option(help="мінімальна пауза між запитами до хоста, с")
    ] = 0.0,
    log_file: Annotated[Path, typer.Option("--log")] = Path("crawl.log"),
    debug: Annotated[bool, typer.Option(help="asyncio debug: попередження про блокування")] = False,
) -> None:
    """Асинхронний обхід сайту з повагою до robots.txt; JSONL пишеться потоково."""
    import asyncio

    from findex.crawler.cli_crawl import run_crawl

    st = asyncio.run(
        run_crawl(
            seed,
            out_path,
            log_file,
            max_pages,
            concurrency,
            per_host,
            min_delay,
            err,
            live=err.is_terminal,
            debug=debug,
        ),
        debug=debug,
    )
    err.print(f"crawled {st.fetched} pages ({st.errors} errors) -> {out_path}; log: {log_file}")


@app.command("serve")
def serve_cmd(
    index_path: Annotated[Path | None, typer.Option("--index", help="INDEX_PATH")] = None,
    host: Annotated[str, typer.Option()] = "127.0.0.1",
    port: Annotated[int, typer.Option()] = 8000,
    workers: Annotated[int, typer.Option(min=1)] = 1,
) -> None:
    """HTTP API і сторінка пошуку (uvicorn)."""
    import os

    import uvicorn

    if index_path is not None:
        os.environ["INDEX_PATH"] = str(index_path)  # конфіг лише з оточення, як у контейнері
    uvicorn.run("findex.web:app", host=host, port=port, workers=workers, log_level="info")


if __name__ == "__main__":
    app()
