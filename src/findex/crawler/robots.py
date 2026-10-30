"""Кеш robots.txt: один запит на хост, далі відповідь з пам'яті."""

from __future__ import annotations

import asyncio
import logging
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

import httpx

from findex.crawler.fetch import USER_AGENT, fetch

log = logging.getLogger(__name__)


class RobotsCache:
    def __init__(self, client: httpx.AsyncClient, user_agent: str = USER_AGENT) -> None:
        self.client = client
        self.agent = user_agent
        self._parsers: dict[str, RobotFileParser] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    async def _parser(self, url: str) -> RobotFileParser:
        parts = urlsplit(url)
        host = f"{parts.scheme}://{parts.netloc}"
        if host in self._parsers:
            return self._parsers[host]
        # замок на хост: 20 воркерів, що одночасно прийшли на новий хост, роблять 1 запит
        lock = self._locks.setdefault(host, asyncio.Lock())
        async with lock:
            if host in self._parsers:
                return self._parsers[host]
            rp = RobotFileParser()
            res = await fetch(self.client, f"{host}/robots.txt", max_retries=1)
            if res.ok:
                rp.parse(res.body.decode("utf-8", errors="replace").splitlines())
            elif 400 <= res.status < 500:
                rp.parse([])  # немає robots.txt => усе дозволено
            else:
                rp.parse(["User-agent: *", "Disallow: /"])  # 5xx/мережа: обережно, нічого
            log.info("robots.txt for %s: status %d", host, res.status)
            self._parsers[host] = rp
            return rp

    async def allowed(self, url: str) -> bool:
        return (await self._parser(url)).can_fetch(self.agent, url)

    async def crawl_delay(self, url: str) -> float | None:
        d = (await self._parser(url)).crawl_delay(self.agent)
        return float(d) if d is not None else None
