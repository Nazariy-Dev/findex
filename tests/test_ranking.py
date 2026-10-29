"""ЛР3: data model Index, парсер (порівняння дерев), скорери."""

from pathlib import Path

import pytest

from findex.corpus import Document
from findex.index import build_index
from findex.query import And, Not, Or, Phrase, QuerySyntaxError, Term, parse
from findex.ranking import BM25, Index, TfIdf, open_index, rank, search
from findex.store import save

DOCS = [
    Document(0, "a", "A", "python asyncio event loop runs coroutines"),
    Document(1, "b", "B", "python django orm queryset"),
    Document(2, "c", "C", "loop event python java"),
]


@pytest.fixture
def ix() -> Index:
    return Index(build_index(DOCS, positions=True))


def test_data_model(ix: Index) -> None:
    assert len(ix) == 10  # унікальні терми трьох документів
    assert "python" in ix and "ruby" not in ix
    assert [p.doc_id for p in ix["python"]] == [0, 1, 2]
    with pytest.raises(KeyError):
        ix["ruby"]
    assert set(ix) >= {"python", "java"}
    assert repr(ix) == "Index(terms=10, docs=3)"
    assert ix.df("event") == 2 and ix.num_docs == 3
    assert ix.avg_doc_length == pytest.approx(14 / 3)


@pytest.mark.parametrize(
    ("q", "tree"),
    [
        ("a OR b c", Or(Term("a"), And(Term("b"), Term("c")))),
        ("a b OR c", Or(And(Term("a"), Term("b")), Term("c"))),
        ("a AND (b OR c)", And(Term("a"), Or(Term("b"), Term("c")))),
        ("a NOT b", And(Term("a"), Not(Term("b")))),
        ('"event loop" x', And(Phrase(("event", "loop")), Term("x"))),
        ("NOT NOT a", Not(Not(Term("a")))),
    ],
)
def test_parser_trees(q: str, tree: object) -> None:
    assert parse(q) == tree


def test_operators_build_same_tree() -> None:
    assert Term("a") & (Term("b") | Term("c")) & ~Term("d") == parse("a (b OR c) NOT d")


@pytest.mark.parametrize("bad", ["", "a AND", "(a OR b", "a )", "OR a", '"!!!"'])
def test_parser_errors(bad: str) -> None:
    with pytest.raises(QuerySyntaxError):
        parse(bad)


def test_phrase_needs_order(ix: Index) -> None:
    assert parse('"event loop"').evaluate(ix) == {0}
    assert parse('"loop event"').evaluate(ix) == {2}
    assert parse("python NOT java").evaluate(ix) == {0, 1}


def test_rare_term_beats_common() -> None:
    docs = [Document(i, str(i), str(i), "common filler words here") for i in range(10)]
    docs[0] = Document(0, "0", "0", "common filler words rare")
    docs[1] = Document(1, "1", "1", "common filler words common")
    ix = Index(build_index(docs))
    for s in (TfIdf(), BM25()):
        res = rank(ix, ["rare", "common"], None, s, 10)
        assert res[0].doc_id == 0


def test_bm25_saturation() -> None:
    def score(tf: int) -> float:
        docs = [
            Document(0, "0", "0", " ".join(["x"] * tf + ["pad"] * (40 - tf))),
            Document(1, "1", "1", "pad " * 40),
        ]
        ix = Index(build_index(docs))
        return rank(ix, ["x"], None, BM25(), 1)[0].score

    gain_2nd = score(2) - score(1)
    gain_20th = score(20) - score(19)
    assert gain_20th < gain_2nd / 10


def test_short_doc_beats_long() -> None:
    docs = [
        Document(0, "0", "0", "match " + "pad " * 5),
        Document(1, "1", "1", "match " + "pad " * 500),
        Document(2, "2", "2", "other"),
    ]
    ix = Index(build_index(docs))
    assert rank(ix, ["match"], None, BM25(), 2)[0].doc_id == 0


def test_open_index_closes_on_error(tmp_path: Path) -> None:
    p = tmp_path / "i.bin"
    save(build_index(DOCS), p)
    holder = []
    with pytest.raises(RuntimeError), open_index(p) as ix:
        holder.append(ix)
        raise RuntimeError("boom")
    assert holder[0].closed


def test_search_results_sorted(ix: Index) -> None:
    res = search(ix, "python OR loop", k=3, snippets=False)
    assert res == sorted(res, reverse=True)
    assert all(r.score > 0 for r in res)
