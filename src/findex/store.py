"""Збереження і завантаження індексу у двох форматах.

1. pickle — один виклик, швидко. УВАГА: pickle.load виконує довільний код, закладений
   у файл (через __reduce__), тому завантажувати можна лише власні індекси.
   Чужий index.bin, отриманий з мережі, — це віддалене виконання коду.
2. binary — власний формат без виконання коду: JSON-заголовок (терміни, метадані,
   зсуви) + суцільні буфери array('I') з doc_id, tf і (опційно) позиціями.
"""

from __future__ import annotations

import json
import pickle
import struct
from array import array
from pathlib import Path
from typing import Literal

from findex.model import DocMeta, InvertedIndex, Posting

MAGIC = b"FIDX1\0"
Fmt = Literal["pickle", "binary"]


def detect_format(path: Path) -> Fmt:
    with path.open("rb") as f:
        return "binary" if f.read(len(MAGIC)) == MAGIC else "pickle"


def save(index: InvertedIndex, path: Path, fmt: Fmt = "pickle") -> None:
    if fmt == "pickle":
        with path.open("wb") as f:
            pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)
        return
    terms = list(index.postings)
    doc_ids, tfs, pos_counts, positions = array("I"), array("I"), array("I"), array("I")
    offsets: list[int] = []
    for t in terms:
        offsets.append(len(doc_ids))
        for p in index.postings[t]:
            doc_ids.append(p.doc_id)
            tfs.append(p.tf)
            if index.has_positions:
                pos_counts.append(len(p.positions))
                positions.extend(p.positions)
    offsets.append(len(doc_ids))
    header = {
        "terms": terms,
        "offsets": offsets,
        "has_positions": index.has_positions,
        "root": index.root,
        "docs": [
            [m.doc_id, m.path, m.title, index.doc_lengths[m.doc_id]]
            for m in index.doc_meta.values()
        ],
        "sizes": [len(doc_ids), len(positions)],
    }
    hb = json.dumps(header, ensure_ascii=False).encode()
    with path.open("wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        for arr in (doc_ids, tfs, pos_counts, positions):
            arr.tofile(f)


def load(path: Path) -> InvertedIndex:
    if not path.exists():
        raise FileNotFoundError(f"index file not found: {path}")
    if detect_format(path) == "pickle":
        with path.open("rb") as f:
            obj = pickle.load(f)  # noqa: S301 — лише власні файли, див. докстрінг модуля
        if not isinstance(obj, InvertedIndex):
            raise ValueError(f"{path} does not contain an index")
        return obj
    with path.open("rb") as f:
        f.read(len(MAGIC))
        (hlen,) = struct.unpack("<Q", f.read(8))
        h = json.loads(f.read(hlen))
        n_post, n_pos = h["sizes"]
        doc_ids, tfs, pos_counts, positions = array("I"), array("I"), array("I"), array("I")
        doc_ids.fromfile(f, n_post)
        tfs.fromfile(f, n_post)
        if h["has_positions"]:
            pos_counts.fromfile(f, n_post)
            positions.fromfile(f, n_pos)
    postings: dict[str, list[Posting]] = {}
    off = h["offsets"]
    pcur = 0
    for i, t in enumerate(h["terms"]):
        lst = []
        for j in range(off[i], off[i + 1]):
            if h["has_positions"]:
                c = pos_counts[j]
                ps = tuple(positions[pcur : pcur + c])
                pcur += c
                lst.append(Posting(doc_ids[j], tfs[j], ps))
            else:
                lst.append(Posting(doc_ids[j], tfs[j]))
        postings[t] = lst
    meta = {d: DocMeta(d, p, title) for d, p, title, _ in h["docs"]}
    lengths = {d: n for d, _, _, n in h["docs"]}
    return InvertedIndex(postings, lengths, meta, h["has_positions"], h.get("root", ""))
