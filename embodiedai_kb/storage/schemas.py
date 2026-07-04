from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class PaperMetadata:
    paper_id: str
    title: str
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    venue: str | None = None
    abstract: str | None = None
    paper_url: str | None = None
    pdf_url: str | None = None
    code_url: str | None = None
    project_url: str | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    semantic_scholar_id: str | None = None
    citation_count: int | None = None
    influential_citation_count: int | None = None
    fields_of_study: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    source_queries: list[str] = field(default_factory=list)
    relevance_score: float = 0.0
    relevance_reasons: list[str] = field(default_factory=list)
    decision: str = "unscored"
    raw_metadata: dict[str, Any] = field(default_factory=dict)
    collected_at: str = field(default_factory=now_iso)
    updated_at: str = field(default_factory=now_iso)

