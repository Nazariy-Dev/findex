"""ЛР5, бонус: робота, обмежена вводом-виводом. 50 URL з локального дзеркала (затримка 50 мс)
через ThreadPoolExecutor: потік чекає на сокет і відпускає GIL, тож потоки масштабуються."""

from __future__ import annotations

import statistics
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8765"
NAMES = [
    "asyncio-task",
    "json",
    "pickle",
    "re",
    "threading",
    "multiprocessing",
    "functools",
    "itertools",
    "dataclasses",
    "collections",
    "logging",
    "unittest",
    "sqlite3",
    "typing",
    "pathlib",
    "unicodedata",
    "tracemalloc",
    "heapq",
    "contextlib",
    "argparse",
    "csv",
    "datetime",
    "venv",
    "asyncio-queue",
    "asyncio-sync",
]
urls = [f"{BASE}/pydocs/library/{n}.html" for n in NAMES] * 2


def get(u: str) -> int:
    with urllib.request.urlopen(u, timeout=10) as r:  # noqa: S310 — локальний сервер
        return len(r.read())


print(f"{len(urls)} URLs, server latency 50 ms")
base = None
for w in (1, 2, 4, 8, 16):
    ts = []
    for _ in range(3):
        t0 = time.perf_counter()
        with ThreadPoolExecutor(w) as pool:
            total = sum(pool.map(get, urls))
        ts.append(time.perf_counter() - t0)
    m = statistics.median(ts)
    base = base or m
    print(f"threads={w:2}  wall={m:.2f}s  speedup={base / m:.1f}x  bytes={total}")
