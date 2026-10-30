"""Pydantic-моделі лише на межі: вхід і вихід API. Всередині — dataclass-и з ЛР2–3."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SearchResultOut(BaseModel):
    doc_id: int
    title: str
    path: str
    score: float
    snippet: str


class SearchResponse(BaseModel):
    query: str
    scorer: str
    total: int = Field(description="усіх документів, що задовольняють запит")
    page: int
    pages: int
    took_ms: float
    results: list[SearchResultOut]


class DocumentOut(BaseModel):
    doc_id: int
    title: str
    path: str
    length: int
    text: str


class StatsOut(BaseModel):
    documents: int
    terms: int
    tokens: int
    avg_doc_length: float
    positions: bool
    uptime_s: float
    version: str


class Health(BaseModel):
    status: str
