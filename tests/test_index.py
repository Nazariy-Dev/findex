"""Перевірки ЛР2: збірка індексу, Boolean-пошук, збереження."""

import pickle
from pathlib import Path

from findex.corpus import Document
from findex.index import build_index
from findex.model import DocMeta, Posting
from findex.search import boolean_search, merge_and, merge_not, merge_or
from findex.store import load, save

DOCS = [
    Document(0, "a.txt", "A", "python asyncio event loop"),
    Document(1, "b.txt", "B", "python django orm queryset"),
    Document(2, "c.txt", "C", "asyncio and django async views python python"),
]


def test_postings_sorted_and_tf() -> None:
    ix = build_index(DOCS)
    assert [p.doc_id for p in ix.postings["python"]] == [0, 1, 2]
    assert ix.postings["python"][2].tf == 2
    assert ix.doc_lengths == {0: 4, 1: 4, 2: 7}


def test_records_hashable() -> None:
    assert len({Posting(1, 2), Posting(1, 2), DocMeta(0, "p", "t")}) == 2


def test_merges() -> None:
    a, b = [1, 3, 5, 7], [3, 4, 5, 8]
    assert merge_and(a, b) == [3, 5]
    assert merge_or(a, b) == [1, 3, 4, 5, 7, 8]
    assert merge_not(a, b) == [1, 7]
    assert merge_and([], b) == [] and merge_or([], b) == b


def test_boolean_engines_agree() -> None:
    ix = build_index(DOCS)
    for q in ["python asyncio", "django OR loop", "python NOT django", "missingterm"]:
        assert boolean_search(ix, q, "merge") == boolean_search(ix, q, "set")
    assert boolean_search(ix, "python NOT django") == [0]


def test_store_roundtrip(tmp_path: Path) -> None:
    for positions in (False, True):
        ix = build_index(DOCS, positions=positions)
        for fmt in ("pickle", "binary"):
            p = tmp_path / f"i.{fmt}"
            save(ix, p, fmt)
            assert load(p) == ix


def test_load_rejects_foreign_pickle(tmp_path: Path) -> None:
    p = tmp_path / "x.bin"
    p.write_bytes(pickle.dumps({"not": "an index"}))
    try:
        load(p)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")
