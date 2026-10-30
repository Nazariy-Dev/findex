"""Ціль для Scalene (без cProfile): завантаження індексу і 200 пошуків з трьох термів."""

import sys
from pathlib import Path

from findex.query import parse
from findex.ranking import BM25, load_index, rank


def main() -> None:
    ix = load_index(Path(sys.argv[1]))
    node = parse("the OR python OR function")
    terms = node.terms()
    for _ in range(200):
        rank(ix, terms, node.evaluate(ix), BM25(), 10)


if __name__ == "__main__":
    main()
