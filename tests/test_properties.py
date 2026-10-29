"""Властивості (Hypothesis): round-trip, відсортованість постінгів, merge == операції над set."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from findex.corpus import Document
from findex.index import build_index
from findex.search import merge_and, merge_not, merge_or
from findex.store import load, save
from findex.tokenize import tokenize

words = st.sampled_from(["python", "loop", "event", "кіт", "café", "a", "b", "don't", "x_y"])
texts = st.lists(words, max_size=30).map(" ".join)
docs_st = st.lists(texts, min_size=1, max_size=8).map(
    lambda ts: [Document(i, f"{i}.txt", str(i), t) for i, t in enumerate(ts)]
)
sorted_ids = st.lists(st.integers(0, 200), max_size=60, unique=True).map(sorted)


@settings(max_examples=60, deadline=None)
@given(docs_st, st.booleans(), st.sampled_from(["pickle", "binary"]))
def test_save_load_roundtrip(docs: list[Document], positions: bool, fmt: str) -> None:
    ix = build_index(docs, positions=positions)
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "i"
        save(ix, p, "binary" if fmt == "binary" else "pickle")
        assert load(p) == ix


@given(docs_st)
def test_postings_always_sorted(docs: list[Document]) -> None:
    ix = build_index(docs)
    for plist in ix.postings.values():
        ids = [p.doc_id for p in plist]
        assert ids == sorted(set(ids))


@given(sorted_ids, sorted_ids)
def test_merge_matches_set_ops(a: list[int], b: list[int]) -> None:
    assert merge_and(a, b) == sorted(set(a) & set(b))
    assert merge_or(a, b) == sorted(set(a) | set(b))
    assert merge_not(a, b) == sorted(set(a) - set(b))


@pytest.mark.slow
@given(st.text())
def test_tokenize_idempotent_and_nonempty(s: str) -> None:
    toks = list(tokenize(s))
    assert all(toks)
    assert list(tokenize(" ".join(toks))) == toks
