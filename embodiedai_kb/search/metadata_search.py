from __future__ import annotations

import json
import math
import re
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from embodiedai_kb.storage.database import normalize_title


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TOPCONF_DB = PROJECT_ROOT / "data/db/topconf_papers.sqlite"
DEFAULT_ARXIV_DB = PROJECT_ROOT / "data/db/papers_recent_3y.sqlite"
DEFAULT_FRONTIER_DB = PROJECT_ROOT / "data/db/frontier_papers.sqlite"

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "using",
    "via",
    "what",
    "with",
}


@dataclass(frozen=True, slots=True)
class SearchCorpus:
    name: str
    db_path: Path


@dataclass(frozen=True, slots=True)
class SearchFilters:
    min_score: float = 4.0
    year_from: int | None = None
    year_to: int | None = None
    venues: tuple[str, ...] = ()
    require_abstract: bool = True
    require_pdf: bool = False


@dataclass(slots=True)
class SearchResult:
    rank: int
    hybrid_score: float
    retrieval_score: float
    metadata_score: float
    paper_id: str
    corpus: str
    title: str
    authors: list[str]
    year: int | None
    venue: str | None
    abstract: str | None
    paper_url: str | None
    pdf_url: str | None
    code_url: str | None
    project_url: str | None
    doi: str | None
    arxiv_id: str | None
    citation_count: int | None
    influential_citation_count: int | None
    frontier_score: float
    relevance_score: float
    decision: str
    keywords: list[str]
    categories: list[str]
    sources: list[str]
    quality_signals: list[str]
    relevance_reasons: list[str]
    matched_terms: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def default_corpora(
    include_arxiv: bool = False,
    include_frontier: bool = False,
) -> list[SearchCorpus]:
    corpora = [SearchCorpus("topconf", DEFAULT_TOPCONF_DB)]
    if include_frontier:
        corpora.append(SearchCorpus("frontier", DEFAULT_FRONTIER_DB))
    if include_arxiv:
        corpora.append(SearchCorpus("arxiv", DEFAULT_ARXIV_DB))
    return corpora


def _json_loads(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def _tokenize_query(query: str) -> list[str]:
    tokens: list[str] = []
    seen: set[str] = set()
    for token in re.findall(r"[A-Za-z0-9]+", query.lower()):
        if len(token) < 2 or token in STOPWORDS or token in seen:
            continue
        seen.add(token)
        tokens.append(token)
    return tokens


def _fts_query(tokens: Iterable[str]) -> str:
    parts = []
    for token in tokens:
        if len(token) >= 4:
            parts.append(f"{token}*")
        else:
            parts.append(token)
    return " OR ".join(parts)


def _clean_list(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    return [str(value).strip() for value in values if str(value).strip()]


def _year_boost(year: int | None) -> float:
    if year is None:
        return 0.0
    return max(0.0, min((year - 2020) * 0.08, 0.5))


def _field_hit_score(text: str, tokens: list[str], weight: float) -> float:
    lowered = text.lower()
    return sum(weight for token in tokens if re.search(rf"\b{re.escape(token)}", lowered))


def _metadata_score(row: sqlite3.Row, query: str, tokens: list[str]) -> float:
    title = row["title"] or ""
    abstract = row["abstract"] or ""
    keywords = row["keywords"] or ""
    categories = row["categories"] or ""
    venue = row["venue"] or ""
    query_lower = query.lower().strip()

    score = 0.0
    score += float(row["relevance_score"] or 0.0) * 0.18
    score += float(row["frontier_score"] or 0.0) * 0.12
    score += _year_boost(row["year"])
    score += 0.25 if row["pdf_url"] else 0.0
    score += 0.15 if row["code_url"] or row["project_url"] else 0.0
    score += 0.35 if row["corpus"] == "topconf" else 0.0
    score += 0.3 if row["corpus"] == "frontier" else 0.0
    score += min(math.log1p(float(row["citation_count"] or 0)), 4.0) * 0.18
    score += min(math.log1p(float(row["influential_citation_count"] or 0)), 3.0) * 0.3
    score += _field_hit_score(title, tokens, 0.45)
    score += _field_hit_score(keywords, tokens, 0.35)
    score += _field_hit_score(categories, tokens, 0.25)
    score += _field_hit_score(abstract, tokens, 0.08)
    score += _field_hit_score(venue, tokens, 0.2)
    if query_lower and query_lower in title.lower():
        score += 1.5
    elif query_lower and query_lower in abstract.lower():
        score += 0.75
    return score


def _matched_terms(row: sqlite3.Row, tokens: list[str]) -> list[str]:
    haystack = " ".join(
        str(row[key] or "")
        for key in ("title", "abstract", "keywords", "categories", "venue")
    ).lower()
    return [token for token in tokens if re.search(rf"\b{re.escape(token)}", haystack)]


class MetadataSearchEngine:
    """Lightweight metadata search over one or more paper SQLite databases."""

    def __init__(
        self,
        corpora: Iterable[SearchCorpus] | None = None,
        filters: SearchFilters | None = None,
    ) -> None:
        self.corpora = list(corpora or default_corpora())
        self.filters = filters or SearchFilters()
        self.conn = sqlite3.connect(":memory:")
        self.conn.row_factory = sqlite3.Row
        self._closed = False
        self._init_index()
        self._load_corpora()
        self._build_fts()

    def close(self) -> None:
        if not self._closed:
            self.conn.close()
            self._closed = True

    def __enter__(self) -> MetadataSearchEngine:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _init_index(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE paper_meta (
                rowid INTEGER PRIMARY KEY,
                corpus TEXT NOT NULL,
                db_path TEXT NOT NULL,
                paper_id TEXT NOT NULL,
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
                citation_count INTEGER,
                influential_citation_count INTEGER,
                frontier_score REAL NOT NULL DEFAULT 0,
                relevance_score REAL NOT NULL DEFAULT 0,
                decision TEXT NOT NULL DEFAULT 'unscored',
                keywords TEXT NOT NULL DEFAULT '[]',
                categories TEXT NOT NULL DEFAULT '[]',
                sources TEXT NOT NULL DEFAULT '[]',
                quality_signals TEXT NOT NULL DEFAULT '[]',
                relevance_reasons TEXT NOT NULL DEFAULT '[]',
                normalized_key TEXT NOT NULL
            );

            CREATE VIRTUAL TABLE metadata_fts USING fts5(
                title,
                abstract,
                keywords,
                categories,
                venue,
                content='paper_meta',
                content_rowid='rowid',
                tokenize='unicode61 remove_diacritics 2'
            );
            """
        )

    def _load_corpora(self) -> None:
        insert_sql = """
            INSERT INTO paper_meta (
                corpus, db_path, paper_id, title, authors, year, venue, abstract,
                paper_url, pdf_url, code_url, project_url, doi, arxiv_id,
                citation_count, influential_citation_count, frontier_score,
                relevance_score, decision, keywords, categories, sources,
                quality_signals, relevance_reasons, normalized_key
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        for corpus in self.corpora:
            db_path = corpus.db_path
            if not db_path.exists():
                raise FileNotFoundError(f"Metadata DB not found: {db_path}")
            source = sqlite3.connect(db_path)
            source.row_factory = sqlite3.Row
            try:
                rows = source.execute(
                    self._source_query(),
                    self._source_params(),
                ).fetchall()
                for row in rows:
                    raw_metadata = _json_loads(row["raw_metadata"], {})
                    frontier_metadata = raw_metadata.get("frontier", {})
                    if not isinstance(frontier_metadata, dict):
                        frontier_metadata = {}
                    quality_signals = frontier_metadata.get("quality_signals", [])
                    if not isinstance(quality_signals, list):
                        quality_signals = []
                    normalized_key = self._dedupe_key(row)
                    self.conn.execute(
                        insert_sql,
                        (
                            corpus.name,
                            str(db_path),
                            row["paper_id"],
                            row["title"],
                            row["authors"],
                            row["year"],
                            row["venue"],
                            row["abstract"],
                            row["paper_url"],
                            row["pdf_url"],
                            row["code_url"],
                            row["project_url"],
                            row["doi"],
                            row["arxiv_id"],
                            row["citation_count"],
                            row["influential_citation_count"],
                            float(frontier_metadata.get("frontier_score") or 0.0),
                            row["relevance_score"],
                            row["decision"],
                            row["keywords"],
                            row["categories"],
                            row["sources"],
                            json.dumps(quality_signals, ensure_ascii=False),
                            row["relevance_reasons"],
                            normalized_key,
                        ),
                    )
            finally:
                source.close()
        self.conn.commit()

    def _source_query(self) -> str:
        where = ["relevance_score >= ?"]
        if self.filters.year_from is not None:
            where.append("year >= ?")
        if self.filters.year_to is not None:
            where.append("year <= ?")
        if self.filters.require_abstract:
            where.append("abstract IS NOT NULL AND length(trim(abstract)) > 0")
        if self.filters.require_pdf:
            where.append("pdf_url IS NOT NULL AND length(trim(pdf_url)) > 0")
        if self.filters.venues:
            venue_clauses = " OR ".join("venue LIKE ?" for _ in self.filters.venues)
            where.append(f"({venue_clauses})")
        return f"SELECT * FROM papers WHERE {' AND '.join(where)}"

    def _source_params(self) -> tuple[Any, ...]:
        params: list[Any] = [self.filters.min_score]
        if self.filters.year_from is not None:
            params.append(self.filters.year_from)
        if self.filters.year_to is not None:
            params.append(self.filters.year_to)
        for venue in self.filters.venues:
            params.append(venue if "%" in venue else f"{venue}%")
        return tuple(params)

    def _build_fts(self) -> None:
        self.conn.execute(
            """
            INSERT INTO metadata_fts(rowid, title, abstract, keywords, categories, venue)
            SELECT rowid, title, coalesce(abstract, ''), keywords, categories, coalesce(venue, '')
            FROM paper_meta
            """
        )
        self.conn.commit()

    @staticmethod
    def _dedupe_key(row: sqlite3.Row) -> str:
        doi = str(row["doi"] or "").strip().lower()
        if doi:
            return f"doi:{doi}"
        arxiv_id = str(row["arxiv_id"] or "").strip().lower()
        if arxiv_id:
            return f"arxiv:{arxiv_id}"
        return f"title:{normalize_title(row['title'])}"

    def search(
        self,
        query: str,
        candidate_k: int = 30,
        pool_multiplier: int = 12,
        dedupe: bool = True,
    ) -> list[SearchResult]:
        candidate_k = max(1, candidate_k)
        tokens = _tokenize_query(query)
        rows = self._retrieve_rows(tokens, candidate_k, pool_multiplier)
        if not rows:
            return []

        scored: list[tuple[float, sqlite3.Row, float, float]] = []
        total = len(rows)
        for index, row in enumerate(rows):
            retrieval_score = (total - index) / total * 10.0
            metadata_score = _metadata_score(row, query, tokens)
            hybrid_score = retrieval_score + metadata_score
            scored.append((hybrid_score, row, retrieval_score, metadata_score))
        scored.sort(key=lambda item: item[0], reverse=True)

        results: list[SearchResult] = []
        seen: set[str] = set()
        for hybrid_score, row, retrieval_score, metadata_score in scored:
            keys = {
                row["normalized_key"],
                f"title:{normalize_title(row['title'])}",
            }
            if dedupe and seen.intersection(keys):
                continue
            seen.update(keys)
            results.append(
                self._to_result(
                    row=row,
                    rank=len(results) + 1,
                    hybrid_score=hybrid_score,
                    retrieval_score=retrieval_score,
                    metadata_score=metadata_score,
                    tokens=tokens,
                )
            )
            if len(results) >= candidate_k:
                break
        return results

    def _retrieve_rows(
        self,
        tokens: list[str],
        candidate_k: int,
        pool_multiplier: int,
    ) -> list[sqlite3.Row]:
        pool_size = max(candidate_k * max(pool_multiplier, 1), candidate_k)
        if not tokens:
            return self.conn.execute(
                """
                SELECT *
                FROM paper_meta
                ORDER BY relevance_score DESC, year DESC
                LIMIT ?
                """,
                (pool_size,),
            ).fetchall()

        match_query = _fts_query(tokens)
        return self.conn.execute(
            """
            SELECT
                paper_meta.*,
                bm25(metadata_fts, 5.0, 2.0, 1.5, 1.2, 1.0) AS bm25_score
            FROM metadata_fts
            JOIN paper_meta ON paper_meta.rowid = metadata_fts.rowid
            WHERE metadata_fts MATCH ?
            ORDER BY bm25_score
            LIMIT ?
            """,
            (match_query, pool_size),
        ).fetchall()

    def _to_result(
        self,
        row: sqlite3.Row,
        rank: int,
        hybrid_score: float,
        retrieval_score: float,
        metadata_score: float,
        tokens: list[str],
    ) -> SearchResult:
        return SearchResult(
            rank=rank,
            hybrid_score=round(hybrid_score, 4),
            retrieval_score=round(retrieval_score, 4),
            metadata_score=round(metadata_score, 4),
            paper_id=row["paper_id"],
            corpus=row["corpus"],
            title=row["title"],
            authors=_clean_list(_json_loads(row["authors"], [])),
            year=row["year"],
            venue=row["venue"],
            abstract=row["abstract"],
            paper_url=row["paper_url"],
            pdf_url=row["pdf_url"],
            code_url=row["code_url"],
            project_url=row["project_url"],
            doi=row["doi"],
            arxiv_id=row["arxiv_id"],
            citation_count=row["citation_count"],
            influential_citation_count=row["influential_citation_count"],
            frontier_score=float(row["frontier_score"] or 0.0),
            relevance_score=float(row["relevance_score"] or 0.0),
            decision=row["decision"],
            keywords=_clean_list(_json_loads(row["keywords"], [])),
            categories=_clean_list(_json_loads(row["categories"], [])),
            sources=_clean_list(_json_loads(row["sources"], [])),
            quality_signals=_clean_list(_json_loads(row["quality_signals"], [])),
            relevance_reasons=_clean_list(_json_loads(row["relevance_reasons"], [])),
            matched_terms=_matched_terms(row, tokens),
        )


def reciprocal_rank_fusion(
    result_groups: Iterable[list[SearchResult]],
    k: int = 60,
    top_k: int = 30,
) -> list[SearchResult]:
    """Merge multiple ranked result lists without using an LLM."""

    by_key: dict[str, tuple[float, SearchResult]] = {}
    for group in result_groups:
        for result in group:
            key = result.doi or result.arxiv_id or normalize_title(result.title)
            score = 1.0 / (k + result.rank)
            previous = by_key.get(key)
            if previous is None or result.hybrid_score > previous[1].hybrid_score:
                by_key[key] = (score + (previous[0] if previous else 0.0), result)
            else:
                by_key[key] = (previous[0] + score, previous[1])

    merged = sorted(by_key.values(), key=lambda item: item[0], reverse=True)
    results = [result for _, result in merged[:top_k]]
    for rank, result in enumerate(results, start=1):
        result.rank = rank
    return results


def score_distribution(results: list[SearchResult]) -> dict[str, float]:
    if not results:
        return {"count": 0, "min": 0.0, "max": 0.0, "mean": 0.0}
    values = [result.hybrid_score for result in results]
    return {
        "count": len(values),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
        "mean": round(math.fsum(values) / len(values), 4),
    }
