"""Спільні фікстури: маленький корпус на диску і зібраний з нього індекс."""

from __future__ import annotations

from pathlib import Path

import pytest

from findex.corpus import iter_documents
from findex.index import build_index
from findex.ranking import Index
from findex.store import save

CORPUS = {
    "asyncio.rst": "Asyncio\n=======\n\nThe event loop runs coroutines. Use await in async def.\n",
    "threads.rst": "Threads\n=======\n\nA thread pool executor runs blocking code; GIL.\n",
    "django.txt": "Querysets\n=========\n\nA queryset is lazy. filter() and exclude().\n",
    "unicode.rst": "Unicode\n=======\n\nNormalize with NFC: caf\u00e9 and cafe\u0301 are equal.\n",
    "long.txt": "Long\n====\n\n" + "filler text " * 400 + "event\n",
}


@pytest.fixture(scope="session")
def corpus_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("corpus")
    for name, text in CORPUS.items():
        (root / name).write_text(text, encoding="utf-8")
    return root


@pytest.fixture(scope="session")
def small_index(corpus_dir: Path) -> Index:
    raw = build_index(iter_documents(corpus_dir), positions=True)
    raw.root = str(corpus_dir)
    return Index(raw)


@pytest.fixture(scope="session")
def index_file(
    corpus_dir: Path, small_index: Index, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    p = tmp_path_factory.mktemp("idx") / "index.bin"
    save(small_index.raw, p, "binary")
    return p
