"""ЛР6: той самий seed і max_pages при concurrency 1, 5, 20 (кожен запуск — окремий процес).

per_host = concurrency: увесь сайт на одному хості, тож саме per_host обмежує паралельність;
окремий рядок 20/2 показує ціну ввічливого ліміту на хост.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

SEED = "http://127.0.0.1:8765/index.html"
ONE = r"""
import asyncio, json, resource, sys, time
from pathlib import Path
from rich.console import Console
from findex.crawler.cli_crawl import run_crawl
c, ph, n = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
t0 = time.perf_counter()
st = asyncio.run(run_crawl(sys.argv[4], Path("/tmp/b.jsonl"), Path("/tmp/b.log"), n, c, ph, 0.0,
                           Console(stderr=True), live=False))
wall = time.perf_counter() - t0
print(json.dumps({"pages": st.fetched, "errors": st.errors, "wall": wall,
                  "rss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}))
"""


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 250
    env = {**os.environ, "NO_PROXY": "127.0.0.1"}
    print("concurrency,per_host,pages,wall_s,pages_per_s,errors,peak_rss_mb")
    for c, ph in ((1, 1), (5, 5), (20, 20), (20, 2)):
        r = subprocess.run(
            [sys.executable, "-c", ONE, str(c), str(ph), str(n), SEED],
            capture_output=True,
            text=True,
            check=True,
            env=env,
        )
        d = json.loads(r.stdout.strip().splitlines()[-1])
        print(
            f"{c},{ph},{d['pages']},{d['wall']:.2f},{d['pages'] / d['wall']:.1f},"
            f"{d['errors']},{d['rss']:.0f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
