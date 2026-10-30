"""Навантажувальний тест у стилі oha: C одночасних клієнтів протягом T секунд.

oha і hey в оточенні виконання недоступні (заборонене завантаження бінарників), тому
генератор написано на httpx.AsyncClient. Він працює в окремому процесі від сервера.
Запуск: python scripts/loadtest.py URL [concurrency] [seconds]
"""

from __future__ import annotations

import asyncio
import statistics
import sys
import time

import httpx


async def main(url: str, conc: int, secs: float) -> None:
    lat: list[float] = []
    codes: dict[int, int] = {}
    deadline = time.perf_counter() + secs
    limits = httpx.Limits(max_connections=conc, max_keepalive_connections=conc)
    async with httpx.AsyncClient(limits=limits, timeout=30) as client:

        async def worker() -> None:
            while time.perf_counter() < deadline:
                t0 = time.perf_counter()
                r = await client.get(url)
                lat.append(time.perf_counter() - t0)
                codes[r.status_code] = codes.get(r.status_code, 0) + 1

        t_start = time.perf_counter()
        async with asyncio.TaskGroup() as tg:
            for _ in range(conc):
                tg.create_task(worker())
        wall = time.perf_counter() - t_start
    q = statistics.quantiles(lat, n=100, method="inclusive")
    print(
        f"requests={len(lat)} wall={wall:.1f}s rps={len(lat) / wall:.1f} "
        f"p50={q[49] * 1000:.1f}ms p95={q[94] * 1000:.1f}ms p99={q[98] * 1000:.1f}ms "
        f"codes={codes}"
    )


if __name__ == "__main__":
    asyncio.run(
        main(
            sys.argv[1],
            int(sys.argv[2]) if len(sys.argv) > 2 else 50,
            float(sys.argv[3]) if len(sys.argv) > 3 else 20,
        )
    )
