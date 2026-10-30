"""ЛР5: гонитва на спільному стані. Потоки рахують терми в ОДИН спільний dict.

Без замка: d[t] = d.get(t, 0) + 1 — це читання, додавання і запис; перемикання між
потоками посередині губить оновлення. Під GIL перемикання рідкісні (кожні ~5 мс), без GIL
потоки справді паралельні. Фікс 1 — threading.Lock, фікс 2 — без спільного стану
(кожен потік має свій Counter, потім злиття), саме так зроблено build_partial/merge.
"""

from __future__ import annotations

import sys
import threading
import time
from collections import Counter
from pathlib import Path

from findex.corpus import iter_paths, read_text_file
from findex.tokenize import tokenize

texts = [read_text_file(p) for p in list(iter_paths(Path("data")))[:300]]
tokens = [list(tokenize(t)) for t in texts]
expected = sum(len(t) for t in tokens)
N = 4


def run(mode: str) -> tuple[int, float]:
    shared: dict[str, int] = {}
    lock = threading.Lock()
    locals_: list[Counter[str]] = []

    def worker(part: list[list[str]]) -> None:
        if mode == "local":
            c: Counter[str] = Counter()
            for toks in part:
                c.update(toks)
            locals_.append(c)  # list.append атомарний і в 3.13t (внутрішній замок об'єкта)
            return
        for toks in part:
            for t in toks:
                if mode == "lock":
                    with lock:
                        shared[t] = shared.get(t, 0) + 1
                else:
                    shared[t] = shared.get(t, 0) + 1

    parts = [tokens[i::N] for i in range(N)]
    th = [threading.Thread(target=worker, args=(p,)) for p in parts]
    t0 = time.perf_counter()
    for t in th:
        t.start()
    for t in th:
        t.join()
    dt = time.perf_counter() - t0
    total = sum(sum(c.values()) for c in locals_) if mode == "local" else sum(shared.values())
    return total, dt


gil = sys._is_gil_enabled() if hasattr(sys, "_is_gil_enabled") else True
print(f"python {sys.version.split()[0]}, sys._is_gil_enabled() = {gil}, threads = {N}")
print(f"expected token count: {expected}")
for mode in ("unsafe", "lock", "local"):
    for attempt in range(3):
        total, dt = run(mode)
        lost = expected - total
        print(f"{mode:7} run {attempt + 1}: counted {total:8}  lost {lost:7}  ({dt:.2f} s)")
