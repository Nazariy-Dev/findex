"""Записи індексу. Винесені в окремий модуль, щоб pickle посилався на findex.model,
а не на __main__ (інакше файл, збережений через python -m findex.index, не завантажиться
з іншої точки входу)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Posting:
    doc_id: int
    tf: int
    positions: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class DocMeta:
    doc_id: int
    path: str
    title: str


@dataclass
class InvertedIndex:
    postings: dict[str, list[Posting]] = field(default_factory=dict)
    doc_lengths: dict[int, int] = field(default_factory=dict)
    doc_meta: dict[int, DocMeta] = field(default_factory=dict)
    has_positions: bool = False
    root: str = ""  # корпус, з якого збудовано (для сніпетів)

    @property
    def num_docs(self) -> int:
        return len(self.doc_lengths)
