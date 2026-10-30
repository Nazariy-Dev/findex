"""ЛР5: бенчмарк паралельної індексації. Кожна конфігурація — окремий процес Python,
щоб пікова RSS батька і CPU time не змішувались між запусками.

Запуск:  uv run python scripts/lab5_bench.py [python-executable] > results.csv
"""

from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys

ONE_RUN = r"""
import json, resource, sys, time
from pathlib import Path
from findex.parallel import build_parallel
ex, w = sys.argv[1], int(sys.argv[2])
c0 = time.process_time(); t0 = time.perf_counter()
ix, tm = build_parallel(Path("data"), w, ex, positions=True)
wall = time.perf_counter() - t0
self_cpu = time.process_time() - c0
ch = resource.getrusage(resource.RUSAGE_CHILDREN)
gil = sys._is_gil_enabled() if hasattr(sys, "_is_gil_enabled") else True
print(json.dumps({"wall": wall, "cpu": self_cpu + ch.ru_utime + ch.ru_stime,
  "rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
  "merge": tm.merge_s, "terms": len(ix.postings), "gil": gil}))
"""


def run(py: str, ex: str, w: int) -> dict[str, float]:
    out = subprocess.run(
        [py, "-c", ONE_RUN, ex, str(w)],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "PYTHONPATH": "src"},
    )
    return json.loads(out.stdout.strip().splitlines()[-1])


def main() -> None:
    py = sys.argv[1] if len(sys.argv) > 1 else sys.executable
    configs = [("serial", 1)]
    cpu = os.cpu_count() or 1
    counts = sorted({1, 2, 4, 8, cpu})
    for ex in ("threads", "processes"):
        configs += [(ex, w) for w in counts]
    if len(sys.argv) > 2:  # лише потоки (для збірки 3.13t)
        configs = [("serial", 1)] + [("threads", w) for w in counts]
    base = None
    print("executor,workers,wall_s,cpu_s,rss_mb,merge_s,speedup,gil,runs_wall")
    for ex, w in configs:
        run(py, ex, w)  # прогрів: кеш файлової системи, імпорти
        rs = [run(py, ex, w) for _ in range(3)]
        med = {k: statistics.median(r[k] for r in rs) for k in ("wall", "cpu", "rss_mb", "merge")}
        if base is None:
            base = med["wall"]
        print(
            f"{ex},{w},{med['wall']:.2f},{med['cpu']:.2f},{med['rss_mb']:.0f},"
            f"{med['merge']:.2f},{base / med['wall']:.2f},{rs[0]['gil']},"
            f"{'/'.join(f'{r["wall"]:.2f}' for r in rs)}",
            flush=True,
        )


if __name__ == "__main__":
    main()
