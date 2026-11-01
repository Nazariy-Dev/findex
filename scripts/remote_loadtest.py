"""ЛР7: навантаження на розгорнутий сервіс. Free-інстанс Render засинає, тому спершу
він «будиться» (опитування /health до 200, до 3 хв) і прогрівається 20 запитами,
а вже потім іде вимір scripts/loadtest.py.

Запуск: python scripts/remote_loadtest.py https://findex-xxxx.onrender.com [concurrency] [seconds]
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
import time

import httpx

BASE = sys.argv[1].rstrip("/")
CONC = sys.argv[2] if len(sys.argv) > 2 else "20"
SECS = sys.argv[3] if len(sys.argv) > 3 else "20"
QUERY = "/search?q=asyncio%20%22event%20loop%22%20OR%20thread&k=10"


async def wake() -> None:
    t0 = time.perf_counter()
    async with httpx.AsyncClient(timeout=60) as c:
        while time.perf_counter() - t0 < 180:
            try:
                r = await c.get(BASE + "/health")
                if r.status_code == 200:
                    print(f"awake after {time.perf_counter() - t0:.1f}s: {r.text}", flush=True)
                    break
            except httpx.HTTPError as e:
                print(f"waiting: {type(e).__name__}", flush=True)
            await asyncio.sleep(3)
        else:
            raise SystemExit("service did not wake up in 180 s")
        for _ in range(20):  # прогрів: кеш ОС, lru_cache
            await c.get(BASE + QUERY)


asyncio.run(wake())
subprocess.run([sys.executable, "scripts/loadtest.py", BASE + QUERY, CONC, SECS], check=True)
