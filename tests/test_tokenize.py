"""Перевірки токенізатора (assert-стиль; у ЛР4 переїдуть на pytest.parametrize)."""

import inspect

from findex.corpus import iter_documents
from findex.tokenize import tokenize


def toks(s: str) -> list[str]:
    return list(tokenize(s))


def test_mixed_case() -> None:
    assert toks("Python PYTHON python") == ["python"] * 3


def test_casefold_not_lower() -> None:
    assert toks("Straße") == toks("STRASSE") == ["strasse"]


def test_cyrillic() -> None:
    assert toks("Генератор повертає ІТЕРАТОР") == ["генератор", "повертає", "ітератор"]


def test_combining_mark() -> None:
    assert toks("café") == toks("café") == ["café"]


def test_punctuation() -> None:
    assert toks("Hello, world! (yes)...") == ["hello", "world", "yes"]


def test_empty() -> None:
    assert toks("") == [] and toks("  \n\t ") == []


def test_apostrophes() -> None:
    assert toks("don't") == ["don't"]
    assert toks("п'ять сім’я") == ["п'ять", "сім'я"]
    assert toks("'quoted'") == ["quoted"]


def test_hyphen_and_digits() -> None:
    assert toks("event-loop") == ["event", "loop"]
    assert toks("Python 3.13") == ["python", "3", "13"]


def test_identifiers_keep_underscore() -> None:
    assert toks("asyncio.get_event_loop() __init__") == ["asyncio", "get_event_loop", "__init__"]


def test_are_generators() -> None:
    assert inspect.isgeneratorfunction(tokenize)
    assert inspect.isgeneratorfunction(iter_documents)
