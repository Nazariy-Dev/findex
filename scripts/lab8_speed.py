"""ЛР8, M2: скорер ЛР3 проти NumPy на частому, рідкісному терміні і запиті з трьох термів."""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

from findex.fast import ArrayIndex, rank_np
from findex.query import parse
from findex.ranking import BM25, load_index, rank


def bench(fn, n: int) -> float:  # type: ignore[no-untyped-def]
    fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1000


def main() -> None:
    ix = load_index(Path(sys.argv[1]))
    t0 = time.perf_counter()
    ai = ArrayIndex.from_index(ix)
    print(f"ArrayIndex build: {time.perf_counter() - t0:.2f} s, terms={len(ai.doc_ids)}")
    by_df = sorted(ix, key=ix.df)
    rare = next(t for t in by_df if ix.df(t) >= 3)
    queries = {
        "common term": "the",
        "rare term": rare,
        "mid term": "coroutine",
        "3-term query": "the OR python OR function",
    }
    bm = BM25()
    # перевірка на повному корпусі: однаковий порядок для 2000 запитів з найчастіших термів
    mism = 0
    for t in by_df[-2000:]:
        r1 = [r.doc_id for r in rank(ix, [t], None, bm, 10)]
        r2 = [r.doc_id for r in rank_np(ai, ix, [t], None, bm, 10)]
        mism += r1 != r2
    print(f"identical top-10 for 2000 most frequent terms: mismatches={mism}")
    print(f"{'query':14} {'term(s)':28} {'docs':>5} {'lab3 ms':>9} {'numpy ms':>9} {'speedup':>8}")
    for label, q in queries.items():
        node = parse(q)
        terms = node.terms()
        ids = node.evaluate(ix)
        n = 30 if len(ids) > 500 else 300
        a = bench(lambda terms=terms, ids=ids: rank(ix, terms, ids, bm, 10), n)
        b = bench(lambda terms=terms, ids=ids: rank_np(ai, ix, terms, ids, bm, 10), n)
        assert [r.doc_id for r in rank(ix, terms, ids, bm, 10)] == [
            r.doc_id for r in rank_np(ai, ix, terms, ids, bm, 10)
        ]
        print(f"{label:14} {q:28} {len(ids):5} {a:9.3f} {b:9.3f} {a / b:7.1f}x")


if __name__ == "__main__":
    main()
