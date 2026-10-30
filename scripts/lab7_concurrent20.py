"""ЛР7, M2: 20 одночасних запитів /search — def проти async def (сервер уже запущений)."""

from __future__ import annotations

import asyncio
import sys
import time

import httpx

URL = sys.argv[1]


async def main() -> None:
    async with httpx.AsyncClient(timeout=60) as c:
        await c.get(URL)  # прогрів
        for _ in range(3):
            t0 = time.perf_counter()
            done: list[float] = []

            async def one(t0: float = t0, done: list[float] = done) -> None:
                await c.get(URL)
                done.append(time.perf_counter() - t0)

            health_ms: list[float] = []

            async def health(health_ms: list[float] = health_ms) -> None:
                await asyncio.sleep(0.05)  # пошуки вже в роботі
                h0 = time.perf_counter()
                await c.get(URL.split("/search")[0] + "/health")
                health_ms.append((time.perf_counter() - h0) * 1000)

            async with asyncio.TaskGroup() as tg:
                for _ in range(20):
                    tg.create_task(one())
                tg.create_task(health())
            # для async def: /health теж стоїть у черзі за блокуючим пошуком
            print(
                f"20 concurrent: total={time.perf_counter() - t0:.2f}s "
                f"first={min(done) * 1000:.0f}ms last={max(done) * 1000:.0f}ms "
                f"/health during load={health_ms[0]:.0f}ms"
            )


asyncio.run(main())
