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
from findex.query import QuerySyntaxError
from findex.ranking import BM25, Index, Scorer, TfIdf, load_index, render_snippet, search
from findex.store import save

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
) -> None:
    """Побудувати індекс і зберегти його."""
    if not corpus.exists():
        raise fail(f"corpus not found: {corpus}")
    if fmt not in ("pickle", "binary"):
        raise fail(f"unknown format {fmt!r}", 2)
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
) -> None:
    """Ранжований пошук з підсвіченими сніпетами."""
    if scorer not in SCORERS:
        raise fail(f"unknown scorer {scorer!r}; use bm25 or tfidf", 2)
    ix = _load(index_path)
    try:
        results = search(ix, query, SCORERS[scorer], k=k)
    except QuerySyntaxError as e:
        raise fail(f"bad query: {e}", 2) from None
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


if __name__ == "__main__":
    app()
