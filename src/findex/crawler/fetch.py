"""Один ввічливий запит: таймаут, повтори з exponential backoff + jitter, Retry-After."""

from __future__ import annotations

import asyncio
import email.utils
import logging
import random
import time
from dataclasses import dataclass

import httpx

log = logging.getLogger(__name__)

USER_AGENT = "findex-crawler/0.6 (student project; +https://github.com/nazarpoptsov/findex)"
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})


@dataclass(frozen=True, slots=True)
class FetchResult:
    url: str
    status: int  # 0 = мережева помилка / таймаут після всіх спроб
    content_type: str
    body: bytes
    elapsed: float
    attempts: int
    error: str = ""

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300


def retry_after_seconds(value: str | None, now: float | None = None) -> float | None:
    """Retry-After буває числом секунд або HTTP-датою."""
    if not value:
        return None
    value = value.strip()
    if value.isdigit():
        return float(value)
    try:
        dt = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, dt.timestamp() - (now if now is not None else time.time()))


def backoff_delay(attempt: int, base: float = 0.5, cap: float = 10.0) -> float:
    """min(cap, base * 2**attempt) + jitter: розводить повтори багатьох воркерів у часі."""
    return min(cap, base * 2**attempt) + random.uniform(0, base)


async def fetch(
    client: httpx.AsyncClient,
    url: str,
    *,
    timeout: float = 10.0,
    max_retries: int = 3,
    base_delay: float = 0.5,
) -> FetchResult:
    t0 = time.perf_counter()
    attempt = 0
    while True:
        attempt += 1
        try:
            async with asyncio.timeout(timeout):
                resp = await client.get(url, follow_redirects=True)
        except (TimeoutError, httpx.TransportError) as e:
            err = type(e).__name__
            if attempt > max_retries:
                log.warning("giving up %s after %d attempts: %s", url, attempt, err)
                return FetchResult(url, 0, "", b"", time.perf_counter() - t0, attempt, err)
            delay = backoff_delay(attempt - 1, base_delay)
            log.info("retry %s in %.2fs (attempt %d, %s)", url, delay, attempt, err)
            await asyncio.sleep(delay)
            continue
        if resp.status_code in RETRY_STATUSES and attempt <= max_retries:
            ra = retry_after_seconds(resp.headers.get("retry-after"))
            delay = ra if ra is not None else backoff_delay(attempt - 1, base_delay)
            log.info(
                "retry %s in %.2fs (attempt %d, status %d%s)",
                url,
                delay,
                attempt,
                resp.status_code,
                ", Retry-After" if ra is not None else "",
            )
            await asyncio.sleep(delay)
            continue
        # інші 4xx (404, 403, ...) не повторюються: відповідь не зміниться
        return FetchResult(
            str(resp.url),
            resp.status_code,
            resp.headers.get("content-type", ""),
            resp.content,
            time.perf_counter() - t0,
            attempt,
        )
