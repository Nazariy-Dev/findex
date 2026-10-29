"""Мова запитів і парсер рекурсивного спуску.

Граматика:
    or_expr  := and_expr ('OR' and_expr)*
    and_expr := not_expr (['AND'] not_expr)*      -- AND може бути неявним
    not_expr := 'NOT' not_expr | atom
    atom     := WORD | '"' PHRASE '"' | '(' or_expr ')'

Приклад: python AND (async OR await) NOT java "event loop"
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from findex.search import merge_and
from findex.tokenize import tokenize

if TYPE_CHECKING:
    from findex.ranking import Index


class QuerySyntaxError(ValueError):
    pass


class Node:
    """Базові оператори: дерево запиту можна будувати й без рядка: Term('a') & ~Term('b')."""

    def __and__(self, other: Node) -> Node:
        return And(self, other)

    def __or__(self, other: Node) -> Node:
        return Or(self, other)

    def __invert__(self) -> Node:
        return Not(self)

    def evaluate(self, index: Index) -> set[int]:
        raise NotImplementedError

    def terms(self) -> list[str]:
        """Позитивні терми (для ранжування й підсвітки); терми під NOT не враховуються."""
        return []


@dataclass(frozen=True)
class Term(Node):
    text: str

    def evaluate(self, index: Index) -> set[int]:
        return {p.doc_id for p in index.get(self.text, ())}

    def terms(self) -> list[str]:
        return [self.text]


@dataclass(frozen=True)
class Phrase(Node):
    words: tuple[str, ...]

    def evaluate(self, index: Index) -> set[int]:
        if not index.has_positions:
            raise QuerySyntaxError("phrase queries need an index built with --positions")
        if any(w not in index for w in self.words):
            return set()
        lists = [index[w] for w in self.words]
        common = [p.doc_id for p in lists[0]]
        for lst in lists[1:]:
            common = merge_and(common, [p.doc_id for p in lst])
        by_doc = [{p.doc_id: p.positions for p in lst} for lst in lists]
        out: set[int] = set()
        for d in common:
            # позиція i першого слова; слово j має стояти на i + j
            first = by_doc[0][d]
            rest = [set(bd[d]) for bd in by_doc[1:]]
            if any(all(i + j + 1 in s for j, s in enumerate(rest)) for i in first):
                out.add(d)
        return out

    def terms(self) -> list[str]:
        return list(self.words)


@dataclass(frozen=True)
class And(Node):
    left: Node
    right: Node

    def evaluate(self, index: Index) -> set[int]:
        if isinstance(self.right, Not):  # a NOT b: не будуємо доповнення до всього корпусу
            return self.left.evaluate(index) - self.right.child.evaluate(index)
        return self.left.evaluate(index) & self.right.evaluate(index)

    def terms(self) -> list[str]:
        return self.left.terms() + self.right.terms()


@dataclass(frozen=True)
class Or(Node):
    left: Node
    right: Node

    def evaluate(self, index: Index) -> set[int]:
        return self.left.evaluate(index) | self.right.evaluate(index)

    def terms(self) -> list[str]:
        return self.left.terms() + self.right.terms()


@dataclass(frozen=True)
class Not(Node):
    child: Node

    def evaluate(self, index: Index) -> set[int]:
        return set(index.doc_ids) - self.child.evaluate(index)


TOKEN = re.compile(r'\s*(?:(\()|(\))|"([^"]*)"|([^\s()"]+))')
OPERATORS = {"AND", "OR", "NOT"}


def lex(s: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    pos = 0
    s = s.strip()
    while pos < len(s):
        m = TOKEN.match(s, pos)
        if not m or m.end() == pos:
            raise QuerySyntaxError(f"unexpected character at {pos}: {s[pos : pos + 10]!r}")
        lp, rp, phrase, word = m.groups()
        if lp:
            out.append(("(", lp))
        elif rp:
            out.append((")", rp))
        elif phrase is not None:
            out.append(("PHRASE", phrase))
        elif word in OPERATORS:
            out.append((word, word))
        else:
            out.append(("WORD", word))
        pos = m.end()
        while pos < len(s) and s[pos].isspace():
            pos += 1
    return out


class Parser:
    def __init__(self, text: str) -> None:
        self.toks = lex(text)
        self.i = 0

    def peek(self) -> str | None:
        return self.toks[self.i][0] if self.i < len(self.toks) else None

    def take(self) -> tuple[str, str]:
        tok = self.toks[self.i]
        self.i += 1
        return tok

    def parse(self) -> Node:
        if not self.toks:
            raise QuerySyntaxError("empty query")
        node = self.or_expr()
        if self.peek() is not None:
            raise QuerySyntaxError(f"unexpected {self.toks[self.i][1]!r}")
        return node

    def or_expr(self) -> Node:
        node = self.and_expr()
        while self.peek() == "OR":
            self.take()
            node = Or(node, self.and_expr())
        return node

    def and_expr(self) -> Node:
        node = self.not_expr()
        while self.peek() in ("AND", "NOT", "WORD", "PHRASE", "("):
            if self.peek() == "AND":
                self.take()
            node = And(node, self.not_expr())
        return node

    def not_expr(self) -> Node:
        if self.peek() == "NOT":
            self.take()
            return Not(self.not_expr())
        return self.atom()

    def atom(self) -> Node:
        kind = self.peek()
        if kind is None:
            raise QuerySyntaxError("query ends unexpectedly")
        _, val = self.take()
        if kind == "(":
            node = self.or_expr()
            if self.peek() != ")":
                raise QuerySyntaxError("missing ')'")
            self.take()
            return node
        if kind in ("WORD", "PHRASE"):
            words = tuple(tokenize(val))
            if not words:
                raise QuerySyntaxError(f"no searchable words in {val!r}")
            # "event-loop" без лапок дає два токени — трактуємо як фразу
            return Term(words[0]) if len(words) == 1 else Phrase(words)
        raise QuerySyntaxError(f"unexpected {val!r}")


def parse(text: str) -> Node:
    return Parser(text).parse()
