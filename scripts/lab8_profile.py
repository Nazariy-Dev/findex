"""ЛР8, M1: профіль findex search (запит із трьох термів, 200 повторів) і findex index.

cProfile дає рейтинг функцій; py-spy записується окремо (див. README) у flame graph.
"""

from __future__ import annotations

import cProfile
import pstats
import sys
from pathlib import Path

from findex.corpus import iter_documents
from findex.index import build_index
from findex.query import parse
from findex.ranking import BM25, load_index, rank


def search_loop(path: Path, n: int = 200) -> None:
    ix = load_index(path)
    node = parse("the OR python OR function")  # три терми, один з них у 94% документів
    terms = node.terms()
    for _ in range(n):
        ids = node.evaluate(ix)
        rank(ix, terms, ids, BM25(), 10)


def index_once() -> None:
    build_index(iter_documents(Path("data")), positions=True)


def show(label: str, prof: cProfile.Profile, n: int = 12) -> None:
    print(f"===== {label} =====")
    st = pstats.Stats(prof, stream=sys.stdout)
    st.strip_dirs().sort_stats("tottime").print_stats(n)


def main() -> None:
    which = sys.argv[1]
    prof = cProfile.Profile()
    if which == "search":
        path = Path(sys.argv[2])
        prof.runcall(search_loop, path)
    else:
        prof.runcall(index_once)
    show(which, prof)


if __name__ == "__main__":
    main()
