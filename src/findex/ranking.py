"""Клас Index (data model), скорери TF-IDF/BM25, ранжований пошук і сніпети."""

from __future__ import annotations

import functools
import heapq
import logging
import math
import re
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from findex.model import DocMeta, InvertedIndex, Posting
from findex.timing import timed

log = logging.getLogger(__name__)


class Index(Mapping[str, list[Posting]]):
    """Інвертований індекс, що поводиться як вбудований Mapping.

    Mapping дає безкоштовно keys/items/get/__contains__/__eq__; реалізуємо лише
    __getitem__, __len__, __iter__ (+ власний швидкий __contains__ і __repr__).
    """

    def __init__(self, raw: InvertedIndex) -> None:
        self._raw = raw
        self._postings = raw.postings
        self.closed = False

    # ---- протокол Mapping ----
    def __getitem__(self, term: str) -> list[Posting]:
        return self._postings[term]  # KeyError, якщо терма немає

    def __len__(self) -> int:
        return len(self._postings)

    def __iter__(self) -> Iterator[str]:
        return iter(self._postings)

    def __contains__(self, term: object) -> bool:
        return term in self._postings

    def __repr__(self) -> str:
        return f"Index(terms={len(self):_}, docs={self.num_docs:_})"

    # ---- обчислювані атрибути ----
    @property
    def num_docs(self) -> int:
        return len(self._raw.doc_lengths)

    @functools.cached_property
    def avg_doc_length(self) -> float:
        # рахується один раз на екземпляр, а не для кожного документа в BM25
        return sum(self._raw.doc_lengths.values()) / max(self.num_docs, 1)

    def doc_length(self, doc_id: int) -> int:
        return self._raw.doc_lengths[doc_id]

    def df(self, term: str) -> int:
        return len(self._postings.get(term, ()))

    def meta(self, doc_id: int) -> DocMeta:
        return self._raw.doc_meta[doc_id]

    @property
    def doc_ids(self) -> list[int]:
        return list(self._raw.doc_lengths)

    @property
    def has_positions(self) -> bool:
        return self._raw.has_positions

    @property
    def raw(self) -> InvertedIndex:
        return self._raw

    def text(self, doc_id: int) -> str:
        """Повний текст документа з корпусу (для сніпетів і перегляду)."""
        root = Path(self._raw.root)
        p = root / self.meta(doc_id).path
        return p.read_text(encoding="utf-8", errors="replace") if p.is_file() else ""

    def close(self) -> None:
        self._postings = {}
        self.closed = True


@timed
def load_index(path: Path) -> Index:
    from findex.store import load

    return Index(load(path))


@contextmanager
def open_index(path: Path) -> Iterator[Index]:
    """Контекстний менеджер: індекс закривається і при винятку в тілі with."""
    index = load_index(path)
    try:
        yield index
    finally:
        index.close()
        log.debug("index %s closed", path)


# ---------------- скорери ----------------
class Scorer(Protocol):
    """Структурний інтерфейс: будь-який об'єкт з таким методом — скорер."""

    def score(self, term: str, posting: Posting, index: Index) -> float: ...


class TfIdf:
    def score(self, term: str, posting: Posting, index: Index) -> float:
        idf = math.log(index.num_docs / index.df(term))
        return (1 + math.log(posting.tf)) * idf

    def __repr__(self) -> str:
        return "TfIdf()"


@dataclass(frozen=True)
class BM25:
    k1: float = 1.5
    b: float = 0.75

    def idf(self, term: str, index: Index) -> float:
        n, df = index.num_docs, index.df(term)
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def score(self, term: str, posting: Posting, index: Index) -> float:
        tf = posting.tf
        norm = self.k1 * (
            1 - self.b + self.b * index.doc_length(posting.doc_id) / index.avg_doc_length
        )
        return self.idf(term, index) * tf * (self.k1 + 1) / (tf + norm)


@dataclass(frozen=True, order=True)
class SearchResult:
    # порядок полів = порядок сортування: спершу бал, потім (обернено) doc_id
    score: float
    neg_doc_id: int = field(repr=False)
    doc_id: int = field(compare=False)
    title: str = field(compare=False)
    snippet: str = field(default="", compare=False)


def rank(
    index: Index, terms: list[str], candidates: set[int] | None, scorer: Scorer, k: int
) -> list[SearchResult]:
    scores: dict[int, float] = {}
    for term in dict.fromkeys(terms):  # унікальні, у порядку запиту
        for p in index.get(term, ()):
            if candidates is not None and p.doc_id not in candidates:
                continue
            scores[p.doc_id] = scores.get(p.doc_id, 0.0) + scorer.score(term, p, index)
    # O(n log k) замість повного сортування; при рівних балах менший doc_id вище
    top = heapq.nlargest(k, scores.items(), key=lambda kv: (kv[1], -kv[0]))
    return [SearchResult(s, -d, d, index.meta(d).title) for d, s in top]


# ---------------- сніпети ----------------
def make_snippet(
    text: str, terms: list[str], width: int = 80, mark: tuple[str, str] = ("[", "]")
) -> str:
    """Вікно ±width символів навколо місця з найбільшою кількістю термів запиту."""
    if not text or not terms:
        return ""
    from findex.tokenize import TOKEN_RE, normalize

    norm = normalize(text)
    # normalize може змінити довжину (ß -> ss); тоді шукаємо в нормалізованому тексті
    src = text if len(norm) == len(text) else norm
    hits = [m for m in TOKEN_RE.finditer(norm if src is text else src) if m.group() in terms]
    if not hits:
        return re.sub(r"\s+", " ", src[: 2 * width]).strip()
    best, best_cnt = hits[0], 0
    for h in hits:  # місце з найбільшою кількістю різних термів у вікні
        cnt = len({x.group() for x in hits if abs(x.start() - h.start()) <= width})
        if cnt > best_cnt:
            best, best_cnt = h, cnt
    lo, hi = max(0, best.start() - width), min(len(src), best.end() + width)
    out, pos = [], lo
    for h in hits:
        if h.start() >= lo and h.end() <= hi:
            out.append(src[pos : h.start()])
            out.append(mark[0] + src[h.start() : h.end()] + mark[1])
            pos = h.end()
    out.append(src[pos:hi])
    s = re.sub(r"\s+", " ", "".join(out)).strip()
    return ("…" if lo > 0 else "") + s + ("…" if hi < len(src) else "")


# ---------------- ранжований пошук ----------------
_INDEXES: dict[int, Index] = {}


@functools.lru_cache(maxsize=256)
def _matching_ids(index_key: int, query: str) -> frozenset[int]:
    """Шлях запит -> id документів кешується (аргументи хешовані: int і str)."""
    from findex.query import parse

    log.debug("cache miss: evaluating %r", query)
    return frozenset(parse(query).evaluate(_INDEXES[index_key]))


@timed
def search(
    index: Index, query: str, scorer: Scorer | None = None, k: int = 10, snippets: bool = True
) -> list[SearchResult]:
    from findex.query import parse

    scorer = scorer or BM25()
    _INDEXES[id(index)] = index
    before = _matching_ids.cache_info().hits
    ids = _matching_ids(id(index), query)
    if _matching_ids.cache_info().hits > before:
        log.debug("cache hit for %r", query)
    terms = parse(query).terms()
    results = rank(index, terms, set(ids), scorer, k)
    if snippets:
        results = [
            SearchResult(
                r.score, r.neg_doc_id, r.doc_id, r.title, make_snippet(index.text(r.doc_id), terms)
            )
            for r in results
        ]
    return results
