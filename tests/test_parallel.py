"""ЛР5: усі три виконавці дають той самий індекс, що й послідовна збірка ЛР4."""

from __future__ import annotations

from pathlib import Path

import pytest

from findex.corpus import iter_documents
from findex.index import build_index
from findex.parallel import build_parallel, build_partial, chunk, merge


@pytest.mark.parametrize("positions", [False, True])
def test_serial_merge_equals_lab4(corpus_dir: Path, positions: bool) -> None:
    ref = build_index(iter_documents(corpus_dir), positions=positions)
    rels = sorted(p.name for p in corpus_dir.iterdir())
    got = merge([build_partial(str(corpus_dir), rels, 0, positions)])
    assert got.postings == ref.postings
    assert list(got.postings) == list(ref.postings)
    assert got.doc_lengths == ref.doc_lengths and got.doc_meta == ref.doc_meta


@pytest.mark.parametrize("executor", ["serial", "threads", "processes"])
@pytest.mark.parametrize("workers", [1, 2, 3])
def test_executors_identical(corpus_dir: Path, executor: str, workers: int) -> None:
    ref = build_index(iter_documents(corpus_dir), positions=True)
    ref.root = str(corpus_dir.resolve())
    ix, _ = build_parallel(corpus_dir, workers, executor, positions=True)  # type: ignore[arg-type]
    assert ix == ref


def test_chunk_ranges_are_contiguous() -> None:
    parts = chunk([str(i) for i in range(10)], 3)
    assert [s for s, _ in parts] == [0, 4, 7]
    assert sum((ps for _, ps in parts), []) == [str(i) for i in range(10)]
    assert chunk(["a"], 8) == [(0, ["a"])]


def test_worker_failure_propagates(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("ok")
    with pytest.raises(FileNotFoundError):
        # неіснуючий шлях усередині воркера: виняток має дійти до батька, без зависання
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(2, mp_context=mp.get_context("spawn")) as pool:
            pool.submit(build_partial, str(tmp_path), ["missing.txt"], 0).result()
