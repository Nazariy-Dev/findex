"""Фронтир на asyncio.Queue + воркери в TaskGroup. crawl() — асинхронний генератор сторінок."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass, field
from html.parser import HTMLParser
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit

import httpx

from findex.crawler.fetch import USER_AGENT, FetchResult, fetch
from findex.crawler.robots import RobotsCache

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Page:
    url: str
    title: str
    text: str
    fetched_at: float
    status: int
    nbytes: int


@dataclass
class CrawlStats:
    fetched: int = 0
    errors: int = 0
    in_flight: int = 0
    queued: int = 0
    skipped_robots: int = 0
    started: float = field(default_factory=time.perf_counter)

    @property
    def pages_per_s(self) -> float:
        return self.fetched / max(time.perf_counter() - self.started, 1e-9)


def canonicalize(url: str) -> str:
    """Схема і хост у нижньому регістрі, без #фрагмента, стандартний порт прибрано,
    порожній шлях -> '/', параметри query відсортовані."""
    p = urlsplit(url.strip())
    scheme, host = p.scheme.lower(), (p.hostname or "").lower()
    port = p.port
    netloc = (
        host
        if port is None or (scheme, port) in (("http", 80), ("https", 443))
        else (f"{host}:{port}")
    )
    path = p.path or "/"
    query = urlencode(sorted(parse_qsl(p.query, keep_blank_values=True)))
    return urlunsplit((scheme, netloc, path, query, ""))


class _Extractor(HTMLParser):
    """Потоковий розбір HTML: заголовок, видимий текст, посилання (stdlib, без залежностей)."""

    SKIP = frozenset({"script", "style", "nav", "footer", "noscript"})

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title = ""
        self.links: list[str] = []
        self.parts: list[str] = []
        self._in_title = False
        self._skip = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self.SKIP:
            self._skip += 1
        elif tag == "title":
            self._in_title = True
        elif tag == "a":
            href = dict(attrs).get("href")
            if href:
                self.links.append(href)

    def handle_endtag(self, tag: str) -> None:
        if tag in self.SKIP and self._skip:
            self._skip -= 1
        elif tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title += data
        elif not self._skip:
            self.parts.append(data)


def extract(html: bytes, base: str) -> tuple[str, str, list[str]]:
    """Заголовок, видимий текст і абсолютні посилання сторінки."""
    p = _Extractor()
    p.feed(html.decode("utf-8", errors="replace"))
    p.close()
    links = [urljoin(base, h) for h in p.links if not h.startswith(("mailto:", "javascript:", "#"))]
    text = " ".join(" ".join(p.parts).split())
    return (" ".join(p.title.split()) or base), text, links


class HostLimiter:
    """Окремий семафор і мінімальна пауза між запитами для кожного хоста."""

    def __init__(self, per_host: int, min_delay: float) -> None:
        self.per_host = per_host
        self.min_delay = min_delay
        self._sems: dict[str, asyncio.Semaphore] = {}
        self._next_slot: dict[str, float] = {}

    def sem(self, host: str) -> asyncio.Semaphore:
        return self._sems.setdefault(host, asyncio.Semaphore(self.per_host))

    async def wait_turn(self, host: str) -> None:
        loop = asyncio.get_running_loop()
        now = loop.time()
        slot = max(now, self._next_slot.get(host, now))
        self._next_slot[host] = slot + self.min_delay  # бронюємо слот до await: без гонитви
        if slot > now:
            await asyncio.sleep(slot - now)


async def crawl(
    seeds: Iterable[str],
    *,
    max_pages: int = 200,
    concurrency: int = 10,
    per_host: int = 2,
    min_delay: float = 0.0,
    allowed_domains: Iterable[str] | None = None,
    client: httpx.AsyncClient | None = None,
    stats: CrawlStats | None = None,
    on_result: object = None,
) -> AsyncIterator[Page]:
    seeds = [canonicalize(s) for s in seeds]
    domains = (
        {d.lower() for d in allowed_domains}
        if allowed_domains
        else {urlsplit(s).hostname or "" for s in seeds}
    )
    st = stats or CrawlStats()
    own_client = client is None
    cl = client or httpx.AsyncClient(
        headers={"User-Agent": USER_AGENT}, limits=httpx.Limits(max_connections=concurrency)
    )
    robots = RobotsCache(cl)
    hosts = HostLimiter(per_host, min_delay)
    global_sem = asyncio.Semaphore(concurrency)
    frontier: asyncio.Queue[str] = asyncio.Queue()
    out: asyncio.Queue[Page | None] = asyncio.Queue()
    seen: set[str] = set()
    budget = max_pages  # скільки ще адрес можна взяти в роботу

    def enqueue(url: str) -> None:
        u = canonicalize(url)
        if (
            urlsplit(u).scheme not in ("http", "https")
            or (urlsplit(u).hostname or "") not in domains
        ):
            return
        if u in seen:
            return
        seen.add(u)
        frontier.put_nowait(u)
        st.queued = frontier.qsize()

    async def worker() -> None:
        nonlocal budget
        while True:
            url = await frontier.get()
            try:
                if budget <= 0:
                    continue
                if not await robots.allowed(url):
                    st.skipped_robots += 1
                    log.info("robots.txt disallows %s", url)
                    continue
                budget -= 1
                host = urlsplit(url).hostname or ""
                async with global_sem, hosts.sem(host):
                    await hosts.wait_turn(host)
                    st.in_flight += 1
                    try:
                        res: FetchResult = await fetch(cl, url)
                    finally:
                        st.in_flight -= 1
                log.info(
                    "%s %d %d bytes %.3fs attempts=%d %s",
                    url,
                    res.status,
                    len(res.body),
                    res.elapsed,
                    res.attempts,
                    res.error,
                )
                if callable(on_result):
                    on_result(res)
                if not res.ok or "html" not in res.content_type:
                    if not res.ok:
                        st.errors += 1
                    continue
                title, text, links = extract(res.body, res.url)
                for link in links:
                    enqueue(link)
                st.fetched += 1
                await out.put(Page(res.url, title, text, time.time(), res.status, len(res.body)))
            finally:
                st.queued = frontier.qsize()
                frontier.task_done()

    async def supervisor() -> None:
        async with asyncio.TaskGroup() as tg:
            tasks = [tg.create_task(worker()) for _ in range(concurrency)]
            await frontier.join()  # фронтир порожній і всі взяті адреси оброблені
            for t in tasks:
                t.cancel()  # воркери висять на frontier.get(); скасування їх завершує
        await out.put(None)

    for s in seeds:
        enqueue(s)
    sup = asyncio.create_task(supervisor())
    try:
        while (page := await out.get()) is not None:
            yield page
    finally:
        if not sup.done():
            sup.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await sup
        if own_client:
            await cl.aclose()
