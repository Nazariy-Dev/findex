"""ЛР8, M3: семантичний і гібридний пошук.

Модель: WordLlama `l2_supercat` (256 вимірів, ваги й токенізатор вкладені в пакет wordllama,
тож працює без доступу до мережі). all-MiniLM-L6-v2 із завдання недоступна: huggingface.co
заблоковано в оточенні виконання.

findex embed ділить документи на фрагменти (~200 слів з перекриттям 40), рахує ембединги,
нормалізує до одиничної довжини і зберігає E: float32[N_chunks, d] та chunk_doc: int32[N_chunks].
Семантичний бал документа = найкращий фрагмент: max(E[chunks_of_doc] @ q).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
import numpy.typing as npt

from findex.corpus import Document

log = logging.getLogger(__name__)
F32 = npt.NDArray[np.float32]
I32 = npt.NDArray[np.int32]
Mode = Literal["keyword", "semantic", "hybrid"]
RRF_K = 60


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> F32: ...


class WordLlamaEmbedder:
    name = "wordllama-l2_supercat-256"

    def __init__(self) -> None:
        import wordllama  # pyright: ignore[reportMissingTypeStubs]
        from safetensors.numpy import load_file  # pyright: ignore[reportUnknownVariableType]  # без анотацій у пакеті
        from tokenizers import Tokenizer
        from wordllama.inference import (  # pyright: ignore[reportMissingTypeStubs]
            WordLlamaInference,
        )

        base = Path(wordllama.__file__).parent
        tok = Tokenizer.from_file(str(base / "tokenizers" / "l2_supercat_tokenizer_config.json"))
        weights = load_file(str(base / "weights" / "l2_supercat_256.safetensors"))
        emb = weights["embedding.weight"]  # матриця 32000 x 256: вектор на кожен токен
        self._model: Any = WordLlamaInference(emb, tok)

    def embed(self, texts: list[str]) -> F32:
        out: Any = self._model.embed(texts)
        return np.asarray(out, dtype=np.float32)


def chunk_words(text: str, size: int = 200, overlap: int = 40) -> Iterator[str]:
    words = text.split()
    if not words:
        return
    step = size - overlap
    for start in range(0, max(1, len(words) - overlap), step):
        yield " ".join(words[start : start + size])


def l2_normalize(m: F32) -> F32:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (m / norms).astype(np.float32)


@dataclass(slots=True)
class Embeddings:
    E: F32
    chunk_doc: I32
    model: str

    def save(self, prefix: Path) -> None:
        np.save(prefix.with_suffix(".emb.npy"), self.E)
        np.save(prefix.with_suffix(".chunks.npy"), self.chunk_doc)
        prefix.with_suffix(".model.txt").write_text(self.model)

    @classmethod
    def load(cls, prefix: Path) -> Embeddings:
        return cls(
            np.load(prefix.with_suffix(".emb.npy")),
            np.load(prefix.with_suffix(".chunks.npy")),
            prefix.with_suffix(".model.txt").read_text(),
        )


def build_embeddings(
    docs: Iterable[Document],
    embedder: Embedder,
    model: str,
    batch: int = 256,
    progress: Callable[[int], None] | None = None,
) -> Embeddings:
    vecs: list[F32] = []
    owners: list[int] = []
    buf: list[str] = []
    buf_owner: list[int] = []

    def flush() -> None:
        if buf:
            vecs.append(l2_normalize(embedder.embed(buf)))
            owners.extend(buf_owner)
            buf.clear()
            buf_owner.clear()

    for doc in docs:
        for ch in chunk_words(f"{doc.title}. {doc.text}"):
            buf.append(ch)
            buf_owner.append(doc.doc_id)
            if len(buf) >= batch:
                flush()
        if progress:
            progress(1)
    flush()
    E = np.vstack(vecs) if vecs else np.zeros((0, 1), dtype=np.float32)
    return Embeddings(E, np.asarray(owners, dtype=np.int32), model)


def semantic_rank(
    emb: Embeddings, qvec: F32, k: int, candidates: set[int] | None = None
) -> list[tuple[int, float]]:
    q = l2_normalize(qvec.reshape(1, -1))[0]
    sims = emb.E @ q  # косинус = скалярний добуток одиничних векторів, один матвек
    n_docs = int(emb.chunk_doc.max()) + 1 if emb.chunk_doc.size else 0
    best = np.full(n_docs, -np.inf, dtype=np.float32)
    np.maximum.at(best, emb.chunk_doc, sims)  # найкращий фрагмент для кожного документа
    if candidates is not None:
        mask = np.zeros(n_docs, dtype=bool)
        mask[[c for c in candidates if c < n_docs]] = True
        best[~mask] = -np.inf
    valid = np.flatnonzero(np.isfinite(best))
    if valid.size == 0:
        return []
    kk = min(k, valid.size)
    part = valid[np.argpartition(-best[valid], kk - 1)[:kk]]
    order = np.lexsort((part, -best[part]))
    return [(int(d), float(best[d])) for d in part[order]]


def rrf(*rankings: list[int], k: int = RRF_K) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion: score(d) = Σ 1 / (k + rank_i(d)), ранги з 1."""
    fused: dict[int, float] = {}
    for ranking in rankings:
        for r, d in enumerate(ranking, 1):
            fused[d] = fused.get(d, 0.0) + 1.0 / (k + r)
    return sorted(fused.items(), key=lambda x: (-x[1], x[0]))


@dataclass(slots=True)
class HybridSearcher:
    """Тримає ембединги і модель запиту; три режими пошуку поверх одного Index."""

    emb: Embeddings
    embedder: Embedder

    @classmethod
    def open(cls, index_path: Path) -> HybridSearcher | None:
        if not index_path.with_suffix(".emb.npy").exists():
            return None
        return cls(Embeddings.load(index_path), WordLlamaEmbedder())

    def semantic(self, query: str, k: int) -> list[tuple[int, float]]:
        return semantic_rank(self.emb, self.embedder.embed([query])[0], k)

    def hybrid(
        self, keyword_ids: list[int], query: str, k: int, depth: int = 50
    ) -> list[tuple[int, float]]:
        sem = [d for d, _ in self.semantic(query, depth)]
        return rrf(keyword_ids[:depth], sem)[:k]
