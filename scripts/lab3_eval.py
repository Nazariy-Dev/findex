"""ЛР3: три перевірки рейтингу на корпусі, сліди @timed/кешу і precision@5 TF-IDF vs BM25.

Запуск: uv run python scripts/lab3_eval.py <index-with-positions>
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from findex.model import Posting
from findex.ranking import BM25, Index, TfIdf, open_index, rank, render_snippet, search


def p_at_5(ix: Index, q: str, rel: set[str], scorer: object) -> float:
    res = search(ix, q.replace(" ", " OR "), scorer=scorer, k=5, snippets=False)  # type: ignore[arg-type]
    return sum(ix.meta(r.doc_id).path in rel for r in res) / 5


def sanity(ix: Index) -> None:
    bm = BM25()
    print("== sanity check 1: rare term outranks common term ==")
    rare, common = "zoneinfo", "the"
    print(f"df({rare})={ix.df(rare)}, df({common})={ix.df(common)}")
    print(f"BM25 idf({rare})={bm.idf(rare, ix):.3f}  idf({common})={bm.idf(common, ix):.3f}")
    top = rank(ix, [rare, common], None, bm, 3)
    for r in top:
        m = ix.meta(r.doc_id)
        tfs = {t: next((p.tf for p in ix[t] if p.doc_id == r.doc_id), 0) for t in (rare, common)}
        print(f"  {r.score:6.2f} {m.path}  tf={tfs}")

    print("\n== sanity check 2: BM25 saturation (one term, avg-length document) ==")
    term = "asyncio"
    avg_doc = min(ix.doc_ids, key=lambda d: abs(ix.doc_length(d) - ix.avg_doc_length))
    for tf in (1, 2, 5, 10, 19, 20):
        s = bm.score(term, Posting(avg_doc, tf), ix)
        print(
            f"  tf={tf:2}  bm25={s:.3f}  tfidf={TfIdf().score(term, Posting(avg_doc, tf), ix):.3f}"
        )
    s19 = bm.score(term, Posting(avg_doc, 19), ix)
    s20 = bm.score(term, Posting(avg_doc, 20), ix)
    s1 = bm.score(term, Posting(avg_doc, 1), ix)
    s2 = bm.score(term, Posting(avg_doc, 2), ix)
    print(f"  gain of 2nd occurrence={s2 - s1:.3f}, gain of 20th={s20 - s19:.4f}")

    print("\n== sanity check 3: short doc with one match vs very long doc with one match ==")
    term = "deprecated"
    once = [p for p in ix[term] if p.tf == 1]
    short = min(once, key=lambda p: ix.doc_length(p.doc_id))
    long_ = max(once, key=lambda p: ix.doc_length(p.doc_id))
    for p in (short, long_):
        print(
            f"  len={ix.doc_length(p.doc_id):6} tf=1 bm25={bm.score(term, p, ix):.3f} "
            f"{ix.meta(p.doc_id).path}"
        )


def main() -> None:
    path = Path(sys.argv[1])
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout, format="%(name)s: %(message)s")
    with open_index(path) as ix:
        print(repr(ix))
        print("\n== @timed + lru_cache trace ==")
        q = 'python AND (async OR await) NOT java "event loop"'
        for _ in range(2):
            res = search(ix, q, k=3)
        for r in res:
            print(f"  {r.score:6.2f} {r.title} | {render_snippet(r.snippet, '[', ']')}")
        logging.getLogger().setLevel(logging.WARNING)
        print()
        sanity(ix)
        data = json.loads(Path("eval/queries.json").read_text())
        print("\n== precision@5 (query terms joined by OR, ranked) ==")
        print(f"{'query':40} {'TF-IDF':>7} {'BM25':>7}")
        tot = {"tfidf": 0.0, "bm25": 0.0}
        for item in data["keyword"]:
            a = p_at_5(ix, item["q"], set(item["rel"]), TfIdf())
            b = p_at_5(ix, item["q"], set(item["rel"]), BM25())
            tot["tfidf"] += a
            tot["bm25"] += b
            print(f"{item['q']:40} {a:7.1f} {b:7.1f}")
        n = len(data["keyword"])
        print(f"{'mean':40} {tot['tfidf'] / n:7.2f} {tot['bm25'] / n:7.2f}")


if __name__ == "__main__":
    main()
