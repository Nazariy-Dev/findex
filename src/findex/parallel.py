"""Паралельна збірка індексу: map (build_partial) + reduce (merge).

build_partial — функція рівня модуля, тому її можна передати процесу (pickle посилається
на неї за ім'ям findex.parallel.build_partial). У воркер передаються ШЛЯХИ і початковий
doc_id, а не прочитані тексти: інакше весь корпус пройшов би через pickle.
Шматки — суцільні діапазони відсортованих шляхів, тож doc_id глобально унікальні
і збігаються з послідовною збіркою.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import time
from array import array
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from concurrent.futures import Executor, ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from findex.corpus import guess_title, iter_paths, read_text_file
from findex.model import DocMeta, InvertedIndex, Posting
from findex.tokenize import tokenize

log = logging.getLogger(__name__)
ExecutorName = Literal["serial", "threads", "processes"]


@dataclass(slots=True)
class PartialIndex:
    """Компактний результат воркера: масиви замість мільйонів об'єктів Posting."""

    start_id: int
    # term -> (doc_ids, tfs, pos_counts, positions)
    postings: dict[str, tuple[array[int], array[int], array[int], array[int]]]
    lengths: list[int]
    meta: list[tuple[str, str]]  # (relative path, title)
    positions: bool


def build_partial(
    root: str, rel_paths: Sequence[str], start_id: int, positions: bool = False
) -> PartialIndex:
    base = Path(root)
    post: dict[str, tuple[array[int], array[int], array[int], array[int]]] = {}
    lengths: list[int] = []
    meta: list[tuple[str, str]] = []
    for offset, rel in enumerate(rel_paths):
        doc_id = start_id + offset
        p = base / rel if rel else base
        text = read_text_file(p)
        if positions:
            pos: defaultdict[str, list[int]] = defaultdict(list)
            n = 0
            for n, tok in enumerate(tokenize(text), 1):
                pos[tok].append(n - 1)
            items: list[tuple[str, int, list[int]]] = [(t, len(ps), ps) for t, ps in pos.items()]
        else:
            counts = Counter(tokenize(text))
            n = counts.total()
            items = [(t, tf, list[int]()) for t, tf in counts.items()]
        for term, tf, ps in items:
            entry = post.get(term)
            if entry is None:
                entry = post[term] = (array("I"), array("I"), array("I"), array("I"))
            entry[0].append(doc_id)
            entry[1].append(tf)
            if positions:
                entry[2].append(len(ps))
                entry[3].extend(ps)
        lengths.append(n)
        meta.append((rel or p.name, guess_title(text, p.stem)))
    return PartialIndex(start_id, post, lengths, meta, positions)


def merge(partials: Iterable[PartialIndex]) -> InvertedIndex:
    """Послідовний крок reduce. Частини сортуються за start_id, тому для кожного терма
    списки просто дописуються один за одним і лишаються відсортованими за doc_id."""
    parts = sorted(partials, key=lambda p: p.start_id)
    postings: dict[str, list[Posting]] = {}
    lengths: dict[int, int] = {}
    meta: dict[int, DocMeta] = {}
    has_pos = bool(parts) and parts[0].positions
    for part in parts:
        for term, (ids, tfs, pcs, pos) in part.postings.items():
            lst = postings.get(term)
            if lst is None:
                lst = postings[term] = []
            if has_pos:
                cur = 0
                for d, tf, c in zip(ids, tfs, pcs, strict=True):
                    lst.append(Posting(d, tf, tuple(pos[cur : cur + c])))
                    cur += c
            else:
                lst.extend(Posting(d, tf) for d, tf in zip(ids, tfs, strict=True))
        for i, (n, (path, title)) in enumerate(zip(part.lengths, part.meta, strict=True)):
            d = part.start_id + i
            lengths[d] = n
            meta[d] = DocMeta(d, path, title)
    # порядок ключів як у послідовній збірці: за першою появою терма
    order = sorted(postings, key=lambda t: (postings[t][0].doc_id, _first_pos(postings[t][0])))
    return InvertedIndex({t: postings[t] for t in order}, lengths, meta, has_pos)


def _first_pos(p: Posting) -> int:
    return p.positions[0] if p.positions else 0


def chunk(paths: Sequence[str], n_chunks: int) -> list[tuple[int, list[str]]]:
    """Суцільні шматки приблизно однакового розміру з наперед відомим start_id."""
    n_chunks = max(1, min(n_chunks, len(paths)))
    size, extra = divmod(len(paths), n_chunks)
    out: list[tuple[int, list[str]]] = []
    start = 0
    for i in range(n_chunks):
        end = start + size + (1 if i < extra else 0)
        out.append((start, list(paths[start:end])))
        start = end
    return out


@dataclass(slots=True)
class BuildTimings:
    map_s: float
    merge_s: float


def build_parallel(
    root: Path,
    workers: int = 1,
    executor: ExecutorName = "serial",
    positions: bool = False,
    chunks_per_worker: int = 4,
) -> tuple[InvertedIndex, BuildTimings]:
    root_s = str(root.resolve())
    rels = [str(p.relative_to(root)) if root.is_dir() else "" for p in iter_paths(root)]
    t0 = time.perf_counter()
    if executor == "serial":
        partials = [build_partial(root_s, rels, 0, positions)]
    else:
        # більше шматків, ніж воркерів: вирівнює навантаження (файли різного розміру)
        pieces = chunk(rels, workers * chunks_per_worker)
        pool: Executor = (
            ThreadPoolExecutor(max_workers=workers)
            if executor == "threads"
            # spawn: новий інтерпретатор без успадкованих потоків і блокувань батька
            # (fork у багатопотоковому процесі може зависнути; у 3.14 fork уже не типовий)
            else ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"))
        )
        with pool:
            futures = [pool.submit(build_partial, root_s, ps, s, positions) for s, ps in pieces]
            # .result() перевикидає виняток воркера з його трасою: збірка падає, а не зависає
            partials = [f.result() for f in futures]
    t1 = time.perf_counter()
    index = merge(partials)
    t2 = time.perf_counter()
    index.root = root_s
    log.info("map %.2fs, merge %.2fs, workers=%d, executor=%s", t1 - t0, t2 - t1, workers, executor)
    return index, BuildTimings(t1 - t0, t2 - t1)


def default_workers() -> int:
    return os.cpu_count() or 1
