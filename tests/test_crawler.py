"""ЛР6: fetch, robots і crawl через httpx.MockTransport — без живої мережі."""

from __future__ import annotations

import asyncio
from collections.abc import Callable

import httpx
import pytest

from findex.crawler import fetch as fetch_mod
from findex.crawler.crawl import canonicalize, crawl, extract
from findex.crawler.fetch import fetch, retry_after_seconds
from findex.crawler.robots import RobotsCache

Handler = Callable[[httpx.Request], httpx.Response]


def client(handler: Handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


@pytest.fixture(autouse=True)
def fast_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fetch_mod, "backoff_delay", lambda attempt, base=0.5, cap=10.0: 0.0)


def run[T](coro: object) -> T:
    return asyncio.run(coro)  # type: ignore[arg-type]


def test_fetch_ok() -> None:
    async def go() -> fetch_mod.FetchResult:
        async with client(lambda r: httpx.Response(200, text="hi")) as c:
            return await fetch(c, "http://x/")

    res: fetch_mod.FetchResult = run(go())
    assert res.ok and res.body == b"hi" and res.attempts == 1


def test_retry_on_503_then_success() -> None:
    calls = {"n": 0}

    def h(r: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(503) if calls["n"] < 3 else httpx.Response(200, text="ok")

    async def go() -> fetch_mod.FetchResult:
        async with client(h) as c:
            return await fetch(c, "http://x/")

    res: fetch_mod.FetchResult = run(go())
    assert res.status == 200 and res.attempts == 3


def test_no_retry_on_404() -> None:
    calls = {"n": 0}

    def h(r: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(404)

    async def go() -> fetch_mod.FetchResult:
        async with client(h) as c:
            return await fetch(c, "http://x/")

    res: fetch_mod.FetchResult = run(go())
    assert res.status == 404 and calls["n"] == 1


def test_gives_up_after_three_retries() -> None:
    calls = {"n": 0}

    def h(r: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(500)

    async def go() -> fetch_mod.FetchResult:
        async with client(h) as c:
            return await fetch(c, "http://x/")

    res: fetch_mod.FetchResult = run(go())
    assert res.status == 500 and calls["n"] == 4  # 1 спроба + 3 повтори


def test_retry_after_header_used() -> None:
    calls = {"n": 0}
    slept: list[float] = []

    def h(r: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"Retry-After": "0"})
        return httpx.Response(200)

    async def go() -> fetch_mod.FetchResult:
        real_sleep = asyncio.sleep

        async def spy(d: float) -> None:
            slept.append(d)
            await real_sleep(0)

        fetch_mod.asyncio.sleep = spy  # type: ignore[assignment]
        try:
            async with client(h) as c:
                return await fetch(c, "http://x/")
        finally:
            fetch_mod.asyncio.sleep = real_sleep  # type: ignore[assignment]

    res: fetch_mod.FetchResult = run(go())
    assert res.status == 200 and slept == [0.0]


def test_timeout_is_retried_then_reported() -> None:
    async def h(r: httpx.Request) -> httpx.Response:
        await asyncio.sleep(1)
        return httpx.Response(200)

    async def go() -> fetch_mod.FetchResult:
        async with httpx.AsyncClient(transport=httpx.MockTransport(h)) as c:
            return await fetch(c, "http://x/", timeout=0.02, max_retries=1)

    res: fetch_mod.FetchResult = run(go())
    assert res.status == 0 and res.error == "TimeoutError" and res.attempts == 2


def test_retry_after_parsing() -> None:
    assert retry_after_seconds("5") == 5.0
    assert retry_after_seconds("Wed, 21 Oct 2015 07:28:00 GMT", now=1445412480.0) == 0.0
    assert retry_after_seconds("garbage") is None and retry_after_seconds(None) is None


def test_robots_cached_and_obeyed() -> None:
    calls = {"robots": 0}

    def h(r: httpx.Request) -> httpx.Response:
        if r.url.path == "/robots.txt":
            calls["robots"] += 1
            return httpx.Response(200, text="User-agent: *\nDisallow: /private/\n")
        return httpx.Response(200)

    async def go() -> list[bool]:
        async with client(h) as c:
            rc = RobotsCache(c)
            return await asyncio.gather(
                *(rc.allowed(u) for u in ["http://h/a", "http://h/private/x", "http://h/b"])
            )

    assert run(go()) == [True, False, True]
    assert calls["robots"] == 1


@pytest.mark.parametrize(
    ("raw", "canon"),
    [
        ("HTTP://Example.COM/a#frag", "http://example.com/a"),
        ("https://example.com", "https://example.com/"),
        ("https://example.com:443/x?b=2&a=1", "https://example.com/x?a=1&b=2"),
        ("http://example.com:8080/", "http://example.com:8080/"),
    ],
)
def test_canonicalize(raw: str, canon: str) -> None:
    assert canonicalize(raw) == canon


def test_extract() -> None:
    html = (
        b"<html><head><title> T </title><script>x=1</script></head><body>"
        b"<a href='/b'>B</a> text <a href='mailto:x'>m</a><a href='#top'>t</a></body></html>"
    )
    title, text, links = extract(html, "http://h/a/")
    assert title == "T" and "x=1" not in text and "text" in text
    assert links == ["http://h/b"]


def site(pages: dict[str, list[str]]) -> Handler:
    def h(r: httpx.Request) -> httpx.Response:
        if r.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nDisallow: /secret\n")
        links = pages.get(r.url.path)
        if links is None:
            return httpx.Response(404)
        body = "".join(f"<a href='{x}'>{x}</a>" for x in links)
        return httpx.Response(200, html=f"<title>{r.url.path}</title><body>{body}</body>")

    return h


def test_crawl_no_duplicates_and_robots() -> None:
    pages = {
        "/": ["/a", "/b", "/a#x", "/secret", "http://other.org/"],
        "/a": ["/", "/b"],
        "/b": ["/c", "/missing"],
        "/c": [],
    }
    seen_urls: list[str] = []

    async def go() -> list[str]:
        async with client(site(pages)) as c:
            return [
                p.url
                async for p in crawl(
                    ["http://h/"],
                    client=c,
                    max_pages=50,
                    concurrency=4,
                    on_result=lambda r: seen_urls.append(r.url),
                )
            ]

    got = run(go())
    assert sorted(got) == ["http://h/", "http://h/a", "http://h/b", "http://h/c"]
    assert len(seen_urls) == len(set(seen_urls)) == 5  # 4 сторінки + /missing (404)
    assert not any("secret" in u or "other.org" in u for u in seen_urls)


def test_crawl_stops_at_max_pages() -> None:
    pages = {f"/{i}": [f"/{i + 1}", f"/{i + 2}"] for i in range(100)}
    pages["/"] = ["/0"]

    async def go() -> list[str]:
        async with client(site(pages)) as c:
            return [
                p.url async for p in crawl(["http://h/"], client=c, max_pages=10, concurrency=5)
            ]

    assert len(run(go())) == 10


def test_consumer_break_closes_cleanly() -> None:
    pages = {f"/{i}": [f"/{i + 1}"] for i in range(50)}
    pages["/"] = ["/0"]

    async def go() -> int:
        n = 0
        async with client(site(pages)) as c:
            async for _ in crawl(["http://h/"], client=c, max_pages=40, concurrency=3):
                n += 1
                if n == 3:
                    break
        return n

    assert run(go()) == 3
