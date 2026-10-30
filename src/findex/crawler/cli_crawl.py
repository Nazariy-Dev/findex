"""findex crawl: потоковий запис JSONL, живі лічильники rich, crawl.log зі статусом кожного URL."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path

from rich.console import Console
from rich.live import Live
from rich.table import Table

from findex.crawler.crawl import CrawlStats, crawl
from findex.crawler.fetch import FetchResult


def _panel(st: CrawlStats) -> Table:
    t = Table(show_header=False, title="findex crawl")
    for k, v in (
        ("pages", st.fetched),
        ("pages/s", f"{st.pages_per_s:.1f}"),
        ("in flight", st.in_flight),
        ("errors", st.errors),
        ("queue", st.queued),
        ("robots skipped", st.skipped_robots),
    ):
        t.add_row(k, str(v))
    return t


async def run_crawl(
    seed: str,
    out: Path,
    log_path: Path,
    max_pages: int,
    concurrency: int,
    per_host: int,
    min_delay: float,
    console: Console,
    live: bool = True,
    debug: bool = False,
) -> CrawlStats:
    st = CrawlStats()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fout, log_path.open("w", encoding="utf-8") as flog:

        def on_result(r: FetchResult) -> None:
            flog.write(
                f"{time.strftime('%H:%M:%S')} {r.status} {len(r.body)} "
                f"{r.elapsed:.3f}s attempts={r.attempts} {r.error} {r.url}\n"
            )

        async def consume() -> None:
            async for page in crawl(
                [seed],
                max_pages=max_pages,
                concurrency=concurrency,
                per_host=per_host,
                min_delay=min_delay,
                stats=st,
                on_result=on_result,
            ):
                # одна сторінка = один рядок, одразу на диск; у пам'яті сторінки не накопичуються
                fout.write(
                    json.dumps(
                        {"url": page.url, "title": page.title, "text": page.text},
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        if live:
            with Live(_panel(st), console=console, refresh_per_second=4) as lv:
                task = asyncio.create_task(consume())
                while not task.done():
                    lv.update(_panel(st))
                    await asyncio.sleep(0.25)
                await task
                lv.update(_panel(st))
        else:
            await consume()
    logging.getLogger(__name__).info("crawl done: %d pages, %d errors", st.fetched, st.errors)
    return st
