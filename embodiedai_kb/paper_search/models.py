from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any


@dataclass(slots=True)
class AcademicPaper:
    """Standardized paper record returned by all academic source connectors."""

    paper_id: str
    title: str
    authors: list[str]
    abstract: str
    doi: str
    published_date: datetime | None
    pdf_url: str
    url: str
    source: str
    updated_date: datetime | None = None
    categories: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    citations: int = 0
    influential_citations: int = 0
    references: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def year(self) -> int | None:
        return self.published_date.year if self.published_date else None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["published_date"] = (
            self.published_date.isoformat() if self.published_date else ""
        )
        data["updated_date"] = self.updated_date.isoformat() if self.updated_date else ""
        data["year"] = self.year
        return data
