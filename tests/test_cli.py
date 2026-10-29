"""CLI через typer.testing.CliRunner."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from findex.cli import app

runner = CliRunner()


def test_help() -> None:
    r = runner.invoke(app, ["--help"])
    assert r.exit_code == 0 and "search" in r.output


def test_index_command(corpus_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "i.bin"
    r = runner.invoke(app, ["index", str(corpus_dir), "--out", str(out)])
    assert r.exit_code == 0, r.output
    assert out.exists()


def test_search_json(index_file: Path) -> None:
    r = runner.invoke(app, ["search", str(index_file), "event loop", "--json"])
    assert r.exit_code == 0
    lines = [json.loads(x) for x in r.stdout.splitlines()]
    assert lines[0]["path"] == "asyncio.rst"
    assert "**event**" in lines[0]["snippet"]


def test_search_table(index_file: Path) -> None:
    r = runner.invoke(app, ["search", str(index_file), "queryset", "--scorer", "tfidf"])
    assert r.exit_code == 0 and "django.txt" in r.stdout


def test_stats(index_file: Path) -> None:
    r = runner.invoke(app, ["stats", str(index_file)])
    assert r.exit_code == 0 and "5" in r.stdout


def test_missing_index_is_one_line(tmp_path: Path) -> None:
    r = runner.invoke(app, ["search", str(tmp_path / "nope.bin"), "x"])
    assert r.exit_code == 1
    assert "Traceback" not in r.output and "not found" in r.output


def test_bad_query_exit_code(index_file: Path) -> None:
    r = runner.invoke(app, ["search", str(index_file), "(a OR"])
    assert r.exit_code == 2 and "bad query" in r.output


def test_bad_scorer(index_file: Path) -> None:
    r = runner.invoke(app, ["search", str(index_file), "a", "--scorer", "xx"])
    assert r.exit_code == 2
