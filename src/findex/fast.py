"""ЛР8: векторизований скорер на NumPy.

Постінги терма — два масиви int32 (doc_ids, tfs), довжини документів — один int32[N].
BM25/TF-IDF для всіх документів терма рахуються одним виразом над масивами, бали
накопичуються scatter-add (np.add.at), топ-k — np.argpartition за O(n).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from findex.ranking import BM25, Index, Scorer, SearchResult, TfIdf

I32 = npt.NDArray[np.int32]
F64 = npt.NDArray[np.float64]


@dataclass(slots=True)
class ArrayIndex:
    doc_ids: dict[str, I32]
    tfs: dict[str, I32]
    doc_len: I32  # індекс масиву = doc_id
    avg_len: float
    num_docs: int

    @classmethod
    def from_index(cls, index: Index) -> ArrayIndex:
        n = max(index.doc_ids) + 1 if index.num_docs else 0
        doc_len = np.zeros(n, dtype=np.int32)
        for d in index.doc_ids:
            doc_len[d] = index.doc_length(d)
        ids: dict[str, I32] = {}
        tfs: dict[str, I32] = {}
        for term, plist in index.items():
            ids[term] = np.fromiter((p.doc_id for p in plist), dtype=np.int32, count=len(plist))
            tfs[term] = np.fromiter((p.tf for p in plist), dtype=np.int32, count=len(plist))
        return cls(ids, tfs, doc_len, index.avg_doc_length, index.num_docs)

    def df(self, term: str) -> int:
        a = self.doc_ids.get(term)
        return 0 if a is None else int(a.shape[0])


def term_weights(ai: ArrayIndex, term: str, scorer: Scorer) -> tuple[I32, F64]:
    ids, tf = ai.doc_ids[term], ai.tfs[term].astype(np.float64)
    df = ids.shape[0]
    if isinstance(scorer, BM25):
        idf = np.log(1 + (ai.num_docs - df + 0.5) / (df + 0.5))
        norm = scorer.k1 * (1 - scorer.b + scorer.b * ai.doc_len[ids] / ai.avg_len)
        return ids, idf * tf * (scorer.k1 + 1) / (tf + norm)
    if isinstance(scorer, TfIdf):
        return ids, (1 + np.log(tf)) * np.log(ai.num_docs / df)
    raise TypeError(f"no vectorized implementation for {scorer!r}")


def rank_np(
    ai: ArrayIndex,
    index: Index,
    terms: list[str],
    candidates: set[int] | None,
    scorer: Scorer,
    k: int,
) -> list[SearchResult]:
    scores = np.zeros(ai.doc_len.shape[0], dtype=np.float64)
    touched = np.zeros(ai.doc_len.shape[0], dtype=bool)
    for term in dict.fromkeys(terms):
        if term not in ai.doc_ids:
            continue
        ids, w = term_weights(ai, term, scorer)
        np.add.at(scores, ids, w)  # doc_ids у межах терма унікальні: можна й scores[ids] += w
        touched[ids] = True
    if candidates is not None:
        mask = np.zeros_like(touched)
        if candidates:
            mask[np.fromiter(candidates, dtype=np.int64, count=len(candidates))] = True
        touched &= mask
    idx = np.flatnonzero(touched)
    if idx.size == 0:
        return []
    kk = min(k, idx.size)
    sc = scores[idx]
    # argpartition знаходить k-й найбільший бал за O(n). Беремо всіх, хто не нижчий за нього
    # (зокрема рівних), і сортуємо лише цю малу частину — так при однакових балах вище
    # стоїть менший doc_id, як у heapq.nlargest з ЛР3.
    if kk < idx.size:
        kth = sc[np.argpartition(-sc, kk - 1)[kk - 1]]
        top = idx[sc >= kth]
    else:
        top = idx
    order = np.lexsort((top, -scores[top]))[:kk]
    return [
        SearchResult(float(scores[d]), -int(d), int(d), index.meta(int(d)).title)
        for d in top[order]
    ]
