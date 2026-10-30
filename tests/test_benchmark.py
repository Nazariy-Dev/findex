"""ЛР8: pytest-benchmark для search і build_index на фікстурному корпусі (видно регресії в CI)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from findex.corpus import iter_documents
from findex.fast import ArrayIndex, rank_np
from findex.index import build_index
from findex.ranking import BM25, Index, rank


def test_bench_build_index(benchmark: Any, corpus_dir: Path) -> None:
    ix = benchmark(lambda: build_index(iter_documents(corpus_dir), positions=True))
    assert ix.num_docs == 5


def test_bench_search_python(benchmark: Any, small_index: Index) -> None:
    res = benchmark(rank, small_index, ["event", "loop", "the"], None, BM25(), 10)
    assert res


def test_bench_search_numpy(benchmark: Any, small_index: Index) -> None:
    ai = ArrayIndex.from_index(small_index)
    res = benchmark(rank_np, ai, small_index, ["event", "loop", "the"], None, BM25(), 10)
    assert res
