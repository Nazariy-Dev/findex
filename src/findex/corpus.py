"""Ліниве читання корпусу: один документ у пам'яті за раз.

Підтримуються два формати:
- каталог з текстовими файлами (.rst, .txt, .md) — обхід через Path.rglob;
- файл JSON Lines (.jsonl), де кожен рядок — {"title": ..., "text": ..., "url": ...}.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple

log = logging.getLogger(__name__)

TEXT_SUFFIXES = (".rst", ".txt", ".md")


class Document(NamedTuple):
    doc_id: int
    path: str
    title: str
    text: str


def guess_title(text: str, fallback: str) -> str:
    """Заголовок reST: перший непорожній рядок, під яким іде лінія з '=', '-', '*' тощо."""
    lines = text.splitlines()
    for i in range(len(lines) - 1):
        cur, nxt = lines[i].strip(), lines[i + 1].strip()
        is_underline = bool(nxt) and len(set(nxt)) == 1 and nxt[0] in "=-*#~^"
        if cur and is_underline and len(nxt) >= len(cur) and not cur.startswith(".."):
            return cur
    return fallback


def iter_paths(root: Path) -> Iterator[Path]:
    """Шляхи до текстових файлів у стабільному (відсортованому) порядку."""
    if root.is_file():
        yield root
        return
    # sorted() матеріалізує лише список шляхів (~1 тис. об'єктів Path), не тексти
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix in TEXT_SUFFIXES + (".jsonl",):
            yield p


def read_text_file(path: Path) -> str:
    # errors="replace": один зіпсований файл не повинен зупиняти обробку корпусу
    raw = path.read_bytes()
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        log.warning("bad encoding in %s, undecodable bytes replaced", path)
        return raw.decode("utf-8", errors="replace")


def iter_jsonl(path: Path, start_id: int = 0) -> Iterator[Document]:
    with path.open(encoding="utf-8", errors="replace") as f:
        doc_id = start_id
        for lineno, line in enumerate(f, 1):  # файл — ітератор рядків, читається ліниво
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                log.warning("%s:%d: invalid JSON line skipped", path, lineno)
                continue
            text = rec.get("text", "")
            yield Document(
                doc_id,
                rec.get("url", f"{path}:{lineno}"),
                rec.get("title") or rec.get("url", ""),
                text,
            )
            doc_id += 1


def iter_documents(root: Path, start_id: int = 0) -> Iterator[Document]:
    """Генератор документів корпусу. Відкриває один файл, віддає документ, переходить далі."""
    doc_id = start_id
    for path in iter_paths(root):
        if path.suffix == ".jsonl":
            for doc in iter_jsonl(path, doc_id):
                yield doc
                doc_id = doc.doc_id + 1
            continue
        text = read_text_file(path)
        rel = str(path.relative_to(root)) if root.is_dir() else path.name
        yield Document(doc_id, rel, guess_title(text, path.stem), text)
        doc_id += 1
