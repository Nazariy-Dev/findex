"""ЛР8: NumPy-скорер дає той самий рейтинг, що й скорер ЛР3."""

from __future__ import annotations

import pytest

from findex.fast import ArrayIndex, rank_np
from findex.query import parse
from findex.ranking import BM25, Index, TfIdf, rank


@pytest.mark.parametrize("scorer", [BM25(), TfIdf(), BM25(k1=1.2, b=0.0)])
@pytest.mark.parametrize(
    "q", ["event", "event OR loop", "the OR a OR event", "queryset NOT event", "missing", "nfc"]
)
@pytest.mark.parametrize("k", [1, 3, 10])
def test_same_ranking(small_index: Index, scorer: object, q: str, k: int) -> None:
    ai = ArrayIndex.from_index(small_index)
    node = parse(q)
    ids = node.evaluate(small_index)
    ref = rank(small_index, node.terms(), ids, scorer, k)  # type: ignore[arg-type]
    got = rank_np(ai, small_index, node.terms(), ids, scorer, k)  # type: ignore[arg-type]
    assert [r.doc_id for r in got] == [r.doc_id for r in ref]
    assert [r.score for r in got] == pytest.approx([r.score for r in ref])


def test_ties_break_by_doc_id(small_index: Index) -> None:
    from findex.corpus import Document
    from findex.index import build_index

    ix = Index(build_index([Document(i, str(i), str(i), "same words") for i in range(6)]))
    ai = ArrayIndex.from_index(ix)
    assert [r.doc_id for r in rank_np(ai, ix, ["same"], None, BM25(), 3)] == [0, 1, 2]
    assert [r.doc_id for r in rank(ix, ["same"], None, BM25(), 3)] == [0, 1, 2]


def test_ties_at_cutoff_keep_lowest_doc_ids() -> None:
    """Регресія: при рівних балах на межі k argpartition брав довільний з рівних."""
    from findex.corpus import Document
    from findex.index import build_index

    docs = [Document(i, str(i), str(i), "x pad" if i % 3 else "x pad pad") for i in range(30)]
    ix = Index(build_index(docs))
    ai = ArrayIndex.from_index(ix)
    for k in range(1, 12):
        ref = [r.doc_id for r in rank(ix, ["x"], None, BM25(), k)]
        assert [r.doc_id for r in rank_np(ai, ix, ["x"], None, BM25(), k)] == ref


def test_rrf_and_chunks() -> None:
    from findex.semantic import chunk_words, rrf

    fused = rrf([1, 2, 3], [3, 1, 4])
    assert [d for d, _ in fused] == [1, 3, 2, 4]
    assert fused[0][1] == pytest.approx(1 / 61 + 1 / 62)
    chunks = list(chunk_words(" ".join(str(i) for i in range(450)), size=200, overlap=40))
    assert len(chunks) == 3 and chunks[1].split()[0] == "160"
    assert list(chunk_words("")) == []
