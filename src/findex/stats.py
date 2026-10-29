"""Статистика корпусу за один прохід: python -m findex.stats data/ [--limit N] [--eager].

Лінивий варіант — ланцюжок генераторів, що сходиться в Counter.
Жадібний варіант (--eager) навмисно матеріалізує всі документи і всі токени у списках,
щоб виміряти різницю в піковій пам'яті.
"""

from __future__ import annotations

import argparse
import itertools
import time
import tracemalloc
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from findex.corpus import Document, iter_documents
from findex.tokenize import tokenize


@dataclass
class CorpusStats:
    documents: int
    tokens: int
    vocabulary: int
    top: list[tuple[str, int]]


def lazy_stats(docs: Iterable[Document], top_n: int = 50) -> CorpusStats:
    counts: Counter[str] = Counter()  # єдина структура, що росте: таблиця термів
    n_docs = 0
    for doc in docs:
        n_docs += 1
        counts.update(tokenize(doc.text))
    return CorpusStats(n_docs, counts.total(), len(counts), counts.most_common(top_n))


def eager_stats(docs: Iterable[Document], top_n: int = 50) -> CorpusStats:
    all_docs = list(docs)  # усі тексти в пам'яті
    all_tokens = [list(tokenize(d.text)) for d in all_docs]  # список списків токенів
    counts: Counter[str] = Counter()
    for toks in all_tokens:
        counts.update(toks)
    return CorpusStats(len(all_docs), counts.total(), len(counts), counts.most_common(top_n))


def measure(
    fn: Callable[[Iterable[Document], int], CorpusStats], docs: Iterable[Document], top_n: int
) -> tuple[CorpusStats, float, int]:
    tracemalloc.start()
    t0 = time.perf_counter()
    result = fn(docs, top_n)
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, elapsed, peak


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m findex.stats")
    ap.add_argument("root", type=Path)
    ap.add_argument("--limit", type=int, default=None, help="лише перші N документів")
    ap.add_argument("--eager", action="store_true", help="жадібна версія через списки")
    ap.add_argument("--top", type=int, default=50)
    a = ap.parse_args(argv)
    if not a.root.exists():
        ap.exit(1, f"error: corpus path not found: {a.root}\n")

    docs = iter_documents(a.root)
    if a.limit is not None:
        docs = itertools.islice(docs, a.limit)  # islice лінивий: решту файлів навіть не відкриємо
    fn = eager_stats if a.eager else lazy_stats
    st, elapsed, peak = measure(fn, docs, a.top)

    print(f"mode:        {'eager (lists)' if a.eager else 'lazy (generators)'}")
    print(f"documents:   {st.documents}")
    print(f"tokens:      {st.tokens}")
    print(f"vocabulary:  {st.vocabulary}")
    print(f"elapsed:     {elapsed:.2f} s")
    print(f"peak memory: {peak / 2**20:.1f} MiB")
    print(f"top-{a.top} terms:")
    for i in range(0, len(st.top), 5):
        print("  " + "  ".join(f"{t}={c}" for t, c in st.top[i : i + 5]))


if __name__ == "__main__":
    main()
