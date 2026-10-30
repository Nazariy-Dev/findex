"""ЛР7: піднімає uvicorn у трьох конфігураціях і міряє (1) 20 одночасних запитів і
(2) навантаження 50 клієнтів × 20 с. Результати — в stdout.

Запуск: uv run python scripts/lab7_bench.py ../idx/web.bin
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
import urllib.request

INDEX = sys.argv[1]
QUERY = "/search?q=asyncio%20%22event%20loop%22%20OR%20thread&k=10"
CONFIGS = [
    ("1 worker, async def", "async", 1),
    ("1 worker, def", "def", 1),
    ("4 workers, def", "def", 4),
]


def wait_ready(port: int) -> None:
    for _ in range(200):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1) as r:
                if r.status == 200:
                    return
        except OSError:
            time.sleep(0.25)
    raise RuntimeError("server did not start")


def main() -> None:
    env = {**os.environ, "INDEX_PATH": INDEX, "NO_PROXY": "127.0.0.1"}
    for label, mode, workers in CONFIGS:
        port = 8010 + workers + (0 if mode == "def" else 5)
        srv = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "findex.web:app",
                "--port",
                str(port),
                "--workers",
                str(workers),
                "--log-level",
                "warning",
            ],
            env={**env, "SEARCH_MODE": mode},
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
        try:
            wait_ready(port)
            time.sleep(1 if workers == 1 else 6)  # усі воркери завантажили індекс
            url = f"http://127.0.0.1:{port}{QUERY}"
            print(f"### {label}", flush=True)
            if workers == 1:
                subprocess.run(
                    [sys.executable, "scripts/lab7_concurrent20.py", url], env=env, check=True
                )
            if "--no-load" not in sys.argv:
                subprocess.run(
                    [sys.executable, "scripts/loadtest.py", url, "50", "20"], env=env, check=True
                )
        finally:
            srv.terminate()
            srv.wait(timeout=30)


if __name__ == "__main__":
    main()
