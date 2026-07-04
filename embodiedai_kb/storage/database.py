from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Any

from embodiedai_kb.storage.schemas import PaperMetadata, now_iso


def normalize_title(title: str) -> str:
    text = title.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _json_loads(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def _merge_unique(*items: list[str]) -> list[str]:
    seen: set[str] = set()
    merged: list[str] = []
    for values in items:
        for value in values:
            clean = str(value).strip()
            key = clean.lower()
            if clean and key not in seen:
                seen.add(key)
                merged.append(clean)
    return merged


def _prefer_longer(a: str | None, b: str | None) -> str | None:
    if not a:
        return b
    if not b:
        return a
    return b if len(b) > len(a) else a


class PaperDatabase:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, timeout=60)
        self.conn.row_factory = sqlite3.Row
        self.init_schema()

    def close(self) -> None:
        self.conn.close()

    def init_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS papers (
                paper_id TEXT PRIMARY KEY,
                normalized_title TEXT UNIQUE,
                title TEXT NOT NULL,
                authors TEXT NOT NULL DEFAULT '[]',
                year INTEGER,
                venue TEXT,
                abstract TEXT,
                paper_url TEXT,
                pdf_url TEXT,
                code_url TEXT,
                project_url TEXT,
                doi TEXT,
                arxiv_id TEXT,
                semantic_scholar_id TEXT,
                citation_count INTEGER,
                influential_citation_count INTEGER,
                fields_of_study TEXT NOT NULL DEFAULT '[]',
                keywords TEXT NOT NULL DEFAULT '[]',
                categories TEXT NOT NULL DEFAULT '[]',
                sources TEXT NOT NULL DEFAULT '[]',
                source_queries TEXT NOT NULL DEFAULT '[]',
                relevance_score REAL NOT NULL DEFAULT 0,
                relevance_reasons TEXT NOT NULL DEFAULT '[]',
                decision TEXT NOT NULL DEFAULT 'unscored',
                raw_metadata TEXT NOT NULL DEFAULT '{}',
                collected_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_papers_year ON papers(year);
            CREATE INDEX IF NOT EXISTS idx_papers_score ON papers(relevance_score);
            CREATE INDEX IF NOT EXISTS idx_papers_decision ON papers(decision);
            CREATE INDEX IF NOT EXISTS idx_papers_arxiv_id ON papers(arxiv_id);
            CREATE INDEX IF NOT EXISTS idx_papers_s2_id ON papers(semantic_scholar_id);
            CREATE INDEX IF NOT EXISTS idx_papers_doi ON papers(doi);
            """
        )
        self.conn.commit()

    def find_existing(self, paper: PaperMetadata) -> sqlite3.Row | None:
        normalized = normalize_title(paper.title)
        checks = [
            ("semantic_scholar_id", paper.semantic_scholar_id),
            ("arxiv_id", paper.arxiv_id),
            ("doi", paper.doi),
            ("normalized_title", normalized),
        ]
        for column, value in checks:
            if not value:
                continue
            row = self.conn.execute(
                f"SELECT * FROM papers WHERE {column} = ? LIMIT 1", (value,)
            ).fetchone()
            if row is not None:
                return row
        return None

    def upsert_paper(self, paper: PaperMetadata) -> str:
        existing = self.find_existing(paper)
        if existing is None:
            self._insert_paper(paper)
            return paper.paper_id
        merged = self._merge_existing(existing, paper)
        self._update_paper(merged)
        return merged.paper_id

    def _insert_paper(self, paper: PaperMetadata) -> None:
        self.conn.execute(
            """
            INSERT INTO papers (
                paper_id, normalized_title, title, authors, year, venue, abstract,
                paper_url, pdf_url, code_url, project_url, doi, arxiv_id,
                semantic_scholar_id, citation_count, influential_citation_count,
                fields_of_study, keywords, categories, sources, source_queries,
                relevance_score, relevance_reasons, decision, raw_metadata,
                collected_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            self._to_row_values(paper),
        )
        self.conn.commit()

    def _update_paper(self, paper: PaperMetadata) -> None:
        self.conn.execute(
            """
            UPDATE papers SET
                normalized_title = ?,
                title = ?,
                authors = ?,
                year = ?,
                venue = ?,
                abstract = ?,
                paper_url = ?,
                pdf_url = ?,
                code_url = ?,
                project_url = ?,
                doi = ?,
                arxiv_id = ?,
                semantic_scholar_id = ?,
                citation_count = ?,
                influential_citation_count = ?,
                fields_of_study = ?,
                keywords = ?,
                categories = ?,
                sources = ?,
                source_queries = ?,
                relevance_score = ?,
                relevance_reasons = ?,
                decision = ?,
                raw_metadata = ?,
                updated_at = ?
            WHERE paper_id = ?
            """,
            (
                normalize_title(paper.title),
                paper.title,
                _json_dumps(paper.authors),
                paper.year,
                paper.venue,
                paper.abstract,
                paper.paper_url,
                paper.pdf_url,
                paper.code_url,
                paper.project_url,
                paper.doi,
                paper.arxiv_id,
                paper.semantic_scholar_id,
                paper.citation_count,
                paper.influential_citation_count,
                _json_dumps(paper.fields_of_study),
                _json_dumps(paper.keywords),
                _json_dumps(paper.categories),
                _json_dumps(paper.sources),
                _json_dumps(paper.source_queries),
                paper.relevance_score,
                _json_dumps(paper.relevance_reasons),
                paper.decision,
                _json_dumps(paper.raw_metadata),
                paper.updated_at,
                paper.paper_id,
            ),
        )
        self.conn.commit()

    def _merge_existing(
        self, existing: sqlite3.Row, incoming: PaperMetadata
    ) -> PaperMetadata:
        existing_score = float(existing["relevance_score"] or 0)
        incoming_score = float(incoming.relevance_score or 0)
        raw_existing = _json_loads(existing["raw_metadata"], {})
        raw_sources = raw_existing.get("sources", [])
        if not isinstance(raw_sources, list):
            raw_sources = [raw_sources]
        raw_sources.append(incoming.raw_metadata)
        merged_raw = raw_existing | {"sources": raw_sources}

        return PaperMetadata(
            paper_id=existing["paper_id"],
            title=incoming.title if len(incoming.title) > len(existing["title"]) else existing["title"],
            authors=_merge_unique(
                _json_loads(existing["authors"], []), incoming.authors
            ),
            year=existing["year"] or incoming.year,
            venue=existing["venue"] or incoming.venue,
            abstract=_prefer_longer(existing["abstract"], incoming.abstract),
            paper_url=existing["paper_url"] or incoming.paper_url,
            pdf_url=existing["pdf_url"] or incoming.pdf_url,
            code_url=existing["code_url"] or incoming.code_url,
            project_url=existing["project_url"] or incoming.project_url,
            doi=existing["doi"] or incoming.doi,
            arxiv_id=existing["arxiv_id"] or incoming.arxiv_id,
            semantic_scholar_id=existing["semantic_scholar_id"]
            or incoming.semantic_scholar_id,
            citation_count=max(
                [v for v in (existing["citation_count"], incoming.citation_count) if v is not None],
                default=None,
            ),
            influential_citation_count=max(
                [
                    v
                    for v in (
                        existing["influential_citation_count"],
                        incoming.influential_citation_count,
                    )
                    if v is not None
                ],
                default=None,
            ),
            fields_of_study=_merge_unique(
                _json_loads(existing["fields_of_study"], []),
                incoming.fields_of_study,
            ),
            keywords=_merge_unique(
                _json_loads(existing["keywords"], []), incoming.keywords
            ),
            categories=_merge_unique(
                _json_loads(existing["categories"], []), incoming.categories
            ),
            sources=_merge_unique(
                _json_loads(existing["sources"], []), incoming.sources
            ),
            source_queries=_merge_unique(
                _json_loads(existing["source_queries"], []), incoming.source_queries
            ),
            relevance_score=max(existing_score, incoming_score),
            relevance_reasons=_merge_unique(
                _json_loads(existing["relevance_reasons"], []),
                incoming.relevance_reasons,
            ),
            decision=(
                incoming.decision
                if incoming_score >= existing_score
                else existing["decision"]
            ),
            raw_metadata=merged_raw,
            collected_at=existing["collected_at"],
            updated_at=now_iso(),
        )

    def _to_row_values(self, paper: PaperMetadata) -> tuple[Any, ...]:
        return (
            paper.paper_id,
            normalize_title(paper.title),
            paper.title,
            _json_dumps(paper.authors),
            paper.year,
            paper.venue,
            paper.abstract,
            paper.paper_url,
            paper.pdf_url,
            paper.code_url,
            paper.project_url,
            paper.doi,
            paper.arxiv_id,
            paper.semantic_scholar_id,
            paper.citation_count,
            paper.influential_citation_count,
            _json_dumps(paper.fields_of_study),
            _json_dumps(paper.keywords),
            _json_dumps(paper.categories),
            _json_dumps(paper.sources),
            _json_dumps(paper.source_queries),
            paper.relevance_score,
            _json_dumps(paper.relevance_reasons),
            paper.decision,
            _json_dumps(paper.raw_metadata),
            paper.collected_at,
            paper.updated_at,
        )

    def export_jsonl(self, output_path: str | Path, min_score: float = 0) -> int:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        rows = self.conn.execute(
            """
            SELECT * FROM papers
            WHERE relevance_score >= ?
            ORDER BY relevance_score DESC, year DESC, citation_count DESC
            """,
            (min_score,),
        ).fetchall()
        with output_path.open("w", encoding="utf-8") as f:
            for row in rows:
                record = dict(row)
                for key in (
                    "authors",
                    "fields_of_study",
                    "keywords",
                    "categories",
                    "sources",
                    "source_queries",
                    "relevance_reasons",
                    "raw_metadata",
                ):
                    record[key] = _json_loads(record.get(key), [] if key != "raw_metadata" else {})
                f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        return len(rows)

    def summary(self) -> dict[str, Any]:
        total = self.conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
        accepted = self.conn.execute(
            "SELECT COUNT(*) FROM papers WHERE relevance_score >= 4"
        ).fetchone()[0]
        by_source = self.conn.execute(
            """
            SELECT sources, COUNT(*) AS n
            FROM papers
            GROUP BY sources
            ORDER BY n DESC
            LIMIT 20
            """
        ).fetchall()
        top = self.conn.execute(
            """
            SELECT title, year, relevance_score, sources
            FROM papers
            ORDER BY relevance_score DESC, year DESC
            LIMIT 10
            """
        ).fetchall()
        return {
            "total": total,
            "score_gte_4": accepted,
            "by_source": [dict(row) for row in by_source],
            "top": [dict(row) for row in top],
        }
