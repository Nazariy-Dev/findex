"""Вимірювання ЛР2: merge vs set, save/load, пам'ять трьох представлень постінгів.

Запуск: uv run python scripts/lab2_bench.py
"""

from __future__ import annotations

import gc
import pickle
import statistics
import sys
import tempfile
import time
import timeit
import tracemalloc
from array import array
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from findex.corpus import iter_documents
from findex.index import build_index
from findex.model import InvertedIndex
from findex.search import OPS
from findex.store import load, save
from findex.tokenize import tokenize

DATA = Path("data")


def median_time(fn: Callable[[], object], n: int = 3) -> float:
    ts = []
    for _ in range(n):
        gc.collect()
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


@dataclass(frozen=True)
class PlainPosting:
    doc_id: int
    tf: int


@dataclass(frozen=True, slots=True)
class SlotPosting:
    doc_id: int
    tf: int


def build_objects(cls: type) -> dict[str, list[Any]]:
    post: defaultdict[str, list[Any]] = defaultdict(list)
    for doc in iter_documents(DATA):
        for term, tf in Counter(tokenize(doc.text)).items():
            post[term].append(cls(doc.doc_id, tf))
    return dict(post)


def build_arrays() -> dict[str, tuple[array[int], array[int]]]:
    post: dict[str, tuple[array[int], array[int]]] = {}
    for doc in iter_documents(DATA):
        for term, tf in Counter(tokenize(doc.text)).items():
            pair = post.get(term)
            if pair is None:
                pair = post[term] = (array("I"), array("I"))
            pair[0].append(doc.doc_id)
            pair[1].append(tf)
    return post


def bench_merge(idx: InvertedIndex) -> None:
    by_df = sorted(idx.postings, key=lambda t: len(idx.postings[t]))
    pairs = {
        "2 most common": by_df[-2:],
        "2 mid-frequency": [t for t in by_df if 40 <= len(idx.postings[t]) <= 60][:2],
        "2 rarest": by_df[:2],
    }
    print("== merge vs set (best of 10 x 200 calls, microseconds per call) ==")
    print(f"{'pair':36} {'df':>11} {'op':>4} {'merge':>9} {'set':>9}")
    for name, (a, b) in pairs.items():
        la = [p.doc_id for p in idx.postings[a]]
        lb = [p.doc_id for p in idx.postings[b]]
        for op in ("AND", "OR"):
            assert OPS["merge"][op](la, lb) == OPS["set"][op](la, lb)
            r = {}
            for eng in ("merge", "set"):
                f = OPS[eng][op]
                t = timeit.repeat(lambda f=f, la=la, lb=lb: f(la, lb), number=200, repeat=10)
                r[eng] = min(t) / 200 * 1e6
            label = f"{name} {a}/{b}"
            print(f"{label:36} {len(la):>5}/{len(lb):<5} {op:>4} {r['merge']:9.2f} {r['set']:9.2f}")


def bench_store(idx: InvertedIndex) -> None:
    print("\n== save/load (median of 3) ==")
    print(f"{'format':22} {'size MiB':>9} {'save s':>8} {'load s':>8}")
    with tempfile.TemporaryDirectory() as tmp:
        for positions in (False, True):
            ix = idx if not positions else build_index(iter_documents(DATA), positions=True)
            for fmt in ("pickle", "binary"):
                p = Path(tmp) / f"ix_{fmt}_{positions}"
                ts = median_time(lambda ix=ix, p=p, fmt=fmt: save(ix, p, fmt))
                tl = median_time(lambda p=p: load(p))
                assert load(p) == ix  # формат без втрат
                label = f"{fmt}{' + positions' if positions else ''}"
                print(f"{label:22} {p.stat().st_size / 2**20:9.2f} {ts:8.2f} {tl:8.2f}")


def bench_memory() -> None:
    print("\n== postings memory under tracemalloc; pickle file size and load time ==")
    print(f"{'representation':32} {'peak MiB':>9} {'kept MiB':>9} {'file MiB':>9} {'load s':>7}")
    variants: list[tuple[str, Callable[[], dict[str, Any]]]] = [
        ("list[Posting] plain @dataclass", lambda: build_objects(PlainPosting)),
        ("list[Posting] slots=True", lambda: build_objects(SlotPosting)),
        ("array('I') pairs (doc_ids, tfs)", build_arrays),
    ]
    for label, fn in variants:
        gc.collect()
        tracemalloc.start()
        obj = fn()
        cur, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        blob = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
        sample = next(iter(obj.values()))
        detail = (
            f"itemsize={sample[0].itemsize}"
            if isinstance(sample, tuple)
            else f"getsizeof={sys.getsizeof(sample[0])}, "
            f"__dict__={sys.getsizeof(sample[0].__dict__) if hasattr(sample[0], '__dict__') else 0}"
        )
        n = sum(len(v[0]) if isinstance(v, tuple) else len(v) for v in obj.values())
        del obj
        tl = median_time(lambda blob=blob: pickle.loads(blob))
        print(
            f"{label:32} {peak / 2**20:9.1f} {cur / 2**20:9.1f} {len(blob) / 2**20:9.2f} "
            f"{tl:7.2f}   postings={n} {detail}"
        )


def main() -> None:
    idx = build_index(iter_documents(DATA))
    bench_merge(idx)
    bench_store(idx)
    del idx
    bench_memory()


if __name__ == "__main__":
    main()
