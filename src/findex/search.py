"""Boolean-пошук: python -m findex.search index.bin "query" [--engine merge|set].

Синтаксис (ліворуч направо, без дужок — справжній парсер з'явиться в ЛР3):
    a b        -> a AND b (неявний AND)
    a OR b     -> об'єднання
    a NOT b    -> a AND NOT b
"""

from __future__ import annotations

import argparse
import time
import tracemalloc
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal

from findex.model import InvertedIndex
from findex.tokenize import tokenize

Engine = Literal["merge", "set"]


# ---------- двовказівникові злиття відсортованих списків ----------
def merge_and(a: Sequence[int], b: Sequence[int]) -> list[int]:
    i = j = 0
    out: list[int] = []
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            out.append(a[i])
            i += 1
            j += 1
        elif a[i] < b[j]:
            i += 1
        else:
            j += 1
    return out


def merge_or(a: Sequence[int], b: Sequence[int]) -> list[int]:
    i = j = 0
    out: list[int] = []
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            out.append(a[i])
            i += 1
            j += 1
        elif a[i] < b[j]:
            out.append(a[i])
            i += 1
        else:
            out.append(b[j])
            j += 1
    out.extend(a[i:])
    out.extend(b[j:])
    return out


def merge_not(a: Sequence[int], b: Sequence[int]) -> list[int]:
    """a AND NOT b."""
    i = j = 0
    out: list[int] = []
    while i < len(a):
        if j >= len(b) or a[i] < b[j]:
            out.append(a[i])
            i += 1
        elif a[i] == b[j]:
            i += 1
            j += 1
        else:
            j += 1
    return out


def set_and(a: Sequence[int], b: Sequence[int]) -> list[int]:
    return sorted(set(a) & set(b))


def set_or(a: Sequence[int], b: Sequence[int]) -> list[int]:
    return sorted(set(a) | set(b))


def set_not(a: Sequence[int], b: Sequence[int]) -> list[int]:
    return sorted(set(a) - set(b))


type MergeFn = Callable[[Sequence[int], Sequence[int]], list[int]]
OPS: dict[str, dict[str, MergeFn]] = {
    "merge": {"AND": merge_and, "OR": merge_or, "NOT": merge_not},
    "set": {"AND": set_and, "OR": set_or, "NOT": set_not},
}


def doc_ids_for(index: InvertedIndex, word: str) -> list[int]:
    toks = list(tokenize(word))
    if not toks:
        return []
    ids = [p.doc_id for p in index.postings.get(toks[0], [])]
    for t in toks[1:]:  # "event-loop" дає два токени — шукаємо обидва
        ids = merge_and(ids, [p.doc_id for p in index.postings.get(t, [])])
    return ids


def boolean_search(index: InvertedIndex, query: str, engine: Engine = "merge") -> list[int]:
    words = query.split()
    if not words:
        return []
    ops = OPS[engine]
    if words[0] in ("AND", "OR", "NOT"):
        raise ValueError(f"query cannot start with operator {words[0]!r}")
    result = doc_ids_for(index, words[0])
    op = "AND"
    for w in words[1:]:
        if w in ("AND", "OR", "NOT"):
            op = w
            continue
        result = ops[op](result, doc_ids_for(index, w))
        op = "AND"
    return result


def main(argv: list[str] | None = None) -> None:
    from findex.store import load

    ap = argparse.ArgumentParser(prog="python -m findex.search")
    ap.add_argument("index", type=Path)
    ap.add_argument("query")
    ap.add_argument("--engine", choices=["merge", "set"], default="merge")
    ap.add_argument("--k", type=int, default=10)
    a = ap.parse_args(argv)
    tracemalloc.start()
    t0 = time.perf_counter()
    try:
        index = load(a.index)
    except (FileNotFoundError, ValueError) as e:
        ap.exit(1, f"error: {e}\n")
    t_load = time.perf_counter() - t0
    t0 = time.perf_counter()
    try:
        ids = boolean_search(index, a.query, a.engine)
    except ValueError as e:
        ap.exit(2, f"error: {e}\n")
    t_search = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"query: {a.query!r}  engine: {a.engine}  matched: {len(ids)} documents")
    for d in ids[: a.k]:
        m = index.doc_meta[d]
        print(f"  [{d:4}] {m.title}  ({m.path})")
    print(
        f"load: {t_load * 1000:.0f} ms, search: {t_search * 1000:.2f} ms, "
        f"peak memory: {peak / 2**20:.1f} MiB"
    )


if __name__ == "__main__":
    main()
