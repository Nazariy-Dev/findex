"""Інвертований індекс: терм -> відсортований за doc_id список постінгів.

Будується одним проходом по лінивому пайплайну з ЛР1. Корпус і далі читається
по одному документу; росте лише сам індекс.

CLI: python -m findex.index data/ --out index.bin [--positions] [--format pickle|binary]
"""

from __future__ import annotations

import argparse
import time
import tracemalloc
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path

from findex.corpus import Document, iter_documents
from findex.model import DocMeta, InvertedIndex, Posting
from findex.tokenize import tokenize

__all__ = ["DocMeta", "InvertedIndex", "Posting", "build_index"]


def build_index(docs: Iterable[Document], positions: bool = False) -> InvertedIndex:
    """Один прохід: Counter поточного документа скидається в defaultdict після документа.

    doc_id ідуть зростаючи, тому append одразу дає відсортовані списки постінгів.
    """
    postings: defaultdict[str, list[Posting]] = defaultdict(list)
    lengths: dict[int, int] = {}
    meta: dict[int, DocMeta] = {}
    for doc in docs:
        if positions:
            pos: defaultdict[str, list[int]] = defaultdict(list)
            n = 0
            for n, tok in enumerate(tokenize(doc.text), 1):
                pos[tok].append(n - 1)
            for term, ps in pos.items():
                postings[term].append(Posting(doc.doc_id, len(ps), tuple(ps)))
        else:
            counts = Counter(tokenize(doc.text))
            n = counts.total()
            for term, tf in counts.items():
                postings[term].append(Posting(doc.doc_id, tf))
        lengths[doc.doc_id] = n
        meta[doc.doc_id] = DocMeta(doc.doc_id, doc.path, doc.title)
    return InvertedIndex(dict(postings), lengths, meta, positions)


def main(argv: list[str] | None = None) -> None:
    from findex.store import save

    ap = argparse.ArgumentParser(prog="python -m findex.index")
    ap.add_argument("root", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--positions", action="store_true", help="зберігати позиції токенів")
    ap.add_argument("--format", choices=["pickle", "binary"], default="pickle")
    a = ap.parse_args(argv)
    if not a.root.exists():
        ap.exit(1, f"error: corpus path not found: {a.root}\n")

    tracemalloc.start()
    t0 = time.perf_counter()
    index = build_index(iter_documents(a.root), positions=a.positions)
    index.root = str(a.root.resolve())
    t_build = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    t0 = time.perf_counter()
    save(index, a.out, fmt=a.format)
    t_save = time.perf_counter() - t0
    print(
        f"documents: {index.num_docs}, terms: {len(index.postings)}, "
        f"positions: {index.has_positions}"
    )
    print(f"build: {t_build:.2f} s, peak memory: {peak / 2**20:.1f} MiB")
    print(f"save ({a.format}): {t_save:.2f} s, file: {a.out.stat().st_size / 2**20:.1f} MiB")


if __name__ == "__main__":
    main()
