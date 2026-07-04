from __future__ import annotations

import asyncio
import math
import os
import re
import threading
import time
import urllib.error
import urllib.request
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from textwrap import shorten
from typing import Any

from embodiedai_kb.paper_search.academic_platforms import (
    ArxivSearcher,
    CrossrefSearcher,
    OpenAlexSearcher,
    OpenReviewSearcher,
    UnpaywallResolver,
)
from embodiedai_kb.paper_search.academic_platforms.common import (
    arxiv_pdf_url,
    doi_from_url,
    title_key,
)
from embodiedai_kb.paper_search.academic_platforms.base import PaperSource
from embodiedai_kb.paper_search.models import AcademicPaper
from embodiedai_kb.search.metadata_search import SearchResult
from scripts.ask_literature import SelectedPaper


DEFAULT_SOURCES = ("arxiv", "openalex", "openreview")
SOURCE_ALIASES = {
    "openalex": "openalex",
    "arxiv": "arxiv",
    "openreview": "openreview",
    "crossref": "crossref",
}


@dataclass(slots=True)
class AcademicSearchConfig:
    sources: tuple[str, ...] = DEFAULT_SOURCES
    max_queries: int = 6
    results_per_source: int = 8
    top_k: int = 6
    min_score: float = 2.5
    year_from: int | None = None
    year_to: int | None = None
    resolve_pdf: bool = True
    request_delay: float = 1.5
    max_workers: int = 2


def _source_instances() -> dict[str, PaperSource]:
    return {
        "openalex": OpenAlexSearcher(),
        "arxiv": ArxivSearcher(),
        "openreview": OpenReviewSearcher(),
        "crossref": CrossrefSearcher(),
    }


def _parse_sources(value: str | None) -> tuple[str, ...]:
    if not value:
        return DEFAULT_SOURCES
    parsed: list[str] = []
    seen: set[str] = set()
    for raw in value.split(","):
        clean = raw.strip().lower().replace("-", "_")
        source = SOURCE_ALIASES.get(clean)
        if source and source not in seen:
            seen.add(source)
            parsed.append(source)
    return tuple(parsed or DEFAULT_SOURCES)


def _year_filter(config: AcademicSearchConfig) -> str | None:
    if config.year_from and config.year_to:
        return f"{config.year_from}-{config.year_to}"
    if config.year_from:
        return f"{config.year_from}-"
    if config.year_to:
        return f"-{config.year_to}"
    return None


def _dedupe_key(paper: AcademicPaper) -> str:
    doi = (paper.doi or "").strip().lower()
    if doi:
        return f"doi:{doi}"
    arxiv_id = str(paper.extra.get("arxiv_id") or "").strip().lower()
    if arxiv_id:
        return f"arxiv:{arxiv_id}"
    openreview_id = str(paper.extra.get("openreview_id") or "").strip().lower()
    if openreview_id:
        return f"openreview:{openreview_id}"
    if paper.pdf_url:
        parsed = urllib.parse.urlparse(paper.pdf_url)
        return f"pdf:{parsed.netloc.lower()}{parsed.path.lower()}"
    return f"title:{title_key(paper.title)}"


def _merge_paper(base: AcademicPaper, incoming: AcademicPaper) -> AcademicPaper:
    """Merge metadata for duplicate papers, preferring richer fields."""

    base_sources = set(str(item) for item in base.extra.get("merged_sources", [base.source]))
    base_sources.add(incoming.source)
    if incoming.abstract and len(incoming.abstract) > len(base.abstract or ""):
        base.abstract = incoming.abstract
    if incoming.pdf_url and not base.pdf_url:
        base.pdf_url = incoming.pdf_url
    if incoming.doi and not base.doi:
        base.doi = incoming.doi
    if incoming.url and not base.url:
        base.url = incoming.url
    if incoming.published_date and (
        not base.published_date or incoming.published_date > base.published_date
    ):
        base.published_date = incoming.published_date
    if incoming.authors and len(incoming.authors) > len(base.authors):
        base.authors = incoming.authors
    base.categories = sorted(set([*base.categories, *incoming.categories]))
    base.keywords = sorted(set([*base.keywords, *incoming.keywords]))
    base.citations = max(base.citations or 0, incoming.citations or 0)
    base.influential_citations = max(
        base.influential_citations or 0,
        incoming.influential_citations or 0,
    )
    base.extra = {**incoming.extra, **base.extra, "merged_sources": sorted(base_sources)}
    return base


def search_papers(
    queries: list[str],
    config: AcademicSearchConfig,
    source_queries: dict[str, list[str]] | None = None,
) -> tuple[list[AcademicPaper], dict[str, Any]]:
    """Run multi-source concurrent paper search and deduplicate results."""

    source_map = _source_instances()
    selected_sources = [source for source in config.sources if source in source_map]
    generic_queries = [query for query in queries if str(query).strip()]
    queries_by_source: dict[str, list[str]] = {}
    for source in selected_sources:
        raw_queries = (source_queries or {}).get(source) or generic_queries
        queries_by_source[source] = [
            query for query in raw_queries[: config.max_queries] if str(query).strip()
        ]
    jobs: list[tuple[str, str, int]] = []
    max_source_queries = max((len(values) for values in queries_by_source.values()), default=0)
    for query_index in range(max_source_queries):
        for source in selected_sources:
            values = queries_by_source[source]
            if query_index < len(values):
                jobs.append((values[query_index], source, len(jobs)))
    errors: list[dict[str, str]] = []
    source_results: dict[str, int] = {source: 0 for source in selected_sources}
    raw: list[AcademicPaper] = []
    source_locks = {source: threading.Lock() for source in selected_sources}
    last_request_at = {source: 0.0 for source in selected_sources}

    def run_job(query: str, source_name: str, index: int) -> list[AcademicPaper]:
        if config.request_delay > 0:
            lock = source_locks[source_name]
            with lock:
                now = time.monotonic()
                wait = last_request_at[source_name] + config.request_delay - now
                if wait > 0:
                    time.sleep(wait)
                last_request_at[source_name] = time.monotonic()
        searcher = source_map[source_name]
        kwargs: dict[str, object] = {}
        return searcher.search(query, max_results=config.results_per_source, **kwargs)

    with ThreadPoolExecutor(max_workers=max(1, min(len(jobs), config.max_workers))) as pool:
        future_map = {
            pool.submit(run_job, query, source, index): (query, source)
            for query, source, index in jobs
        }
        for future in as_completed(future_map):
            query, source = future_map[future]
            try:
                papers = future.result()
            except Exception as exc:
                errors.append(
                    {
                        "query": query,
                        "source": source,
                        "error": shorten(str(exc), width=240, placeholder="..."),
                    }
                )
                continue
            source_results[source] = source_results.get(source, 0) + len(papers)
            raw.extend(papers)

    deduped: dict[str, AcademicPaper] = {}
    for paper in raw:
        key = _dedupe_key(paper)
        if key in deduped:
            deduped[key] = _merge_paper(deduped[key], paper)
        else:
            paper.extra.setdefault("merged_sources", [paper.source])
            deduped[key] = paper

    trace = {
        "mode": "academic_paper_search",
        "sources_requested": list(config.sources),
        "sources_used": selected_sources,
        "queries": generic_queries[: config.max_queries],
        "queries_by_source": queries_by_source,
        "query_count": len(jobs),
        "request_delay": config.request_delay,
        "request_delay_mode": "per_source_min_interval",
        "max_workers": config.max_workers,
        "source_results": source_results,
        "raw_total": len(raw),
        "deduped_total": len(deduped),
        "errors": errors,
    }
    return list(deduped.values()), trace


class OAFallbackResolver:
    """OA-first resolver inspired by paper-search-mcp download_with_fallback."""

    def __init__(self) -> None:
        self.openalex = OpenAlexSearcher()
        self.unpaywall = UnpaywallResolver()

    def resolve_pdf_url(self, paper: AcademicPaper) -> tuple[str, list[str]]:
        attempts: list[str] = []
        if paper.pdf_url:
            attempts.append("direct_pdf_url")
            return paper.pdf_url, attempts

        arxiv_id = str(paper.extra.get("arxiv_id") or "").strip()
        if arxiv_id:
            attempts.append("source_native_arxiv")
            return arxiv_pdf_url(arxiv_id), attempts

        if paper.source == "openreview" and paper.extra.get("openreview_id"):
            attempts.append("source_native_openreview")
            return f"https://openreview.net/pdf?id={paper.extra['openreview_id']}", attempts

        doi = paper.doi or doi_from_url(paper.url)
        if doi:
            attempts.append("openalex_oa_by_doi")
            try:
                oa_paper = self.openalex.lookup_doi(doi)
                if oa_paper and oa_paper.pdf_url:
                    return oa_paper.pdf_url, attempts
            except Exception as exc:
                attempts.append(f"openalex_error:{type(exc).__name__}")

            attempts.append("unpaywall")
            try:
                pdf_url = self.unpaywall.resolve_best_pdf_url(doi)
                if pdf_url:
                    return pdf_url, attempts
            except Exception as exc:
                attempts.append(f"unpaywall_error:{type(exc).__name__}")
        else:
            attempts.append("doi_missing")

        return "", attempts


def _safe_download_stem(paper: AcademicPaper) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", paper.paper_id).strip("_")
    if not stem:
        stem = title_key(paper.title).replace(" ", "_")[:80]
    return stem or "paper"


def download_with_fallback(
    paper: AcademicPaper,
    save_path: str | Path,
    *,
    timeout: float = 60.0,
    max_pdf_mb: float = 80.0,
    retries: int = 2,
) -> tuple[str, dict[str, Any]]:
    """Resolve an OA PDF URL and download it.

    This mirrors paper-search-mcp's high-level download_with_fallback idea, but
    intentionally stops after lawful OA resolution. It does not implement
    Sci-Hub or restricted-source bypasses.
    """

    resolver = OAFallbackResolver()
    pdf_url, attempts = resolver.resolve_pdf_url(paper)
    trace = {"attempts": attempts, "pdf_url": pdf_url, "status": "pending"}
    if not pdf_url:
        trace["status"] = "missing_pdf_url"
        return "", trace

    output_dir = Path(save_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{_safe_download_stem(paper)}.pdf"
    if output_path.exists() and output_path.stat().st_size > 1024:
        trace["status"] = "cache_hit"
        return str(output_path), trace

    last_error: Exception | None = None
    for attempt in range(max(retries, 0) + 1):
        try:
            request = urllib.request.Request(
                pdf_url,
                headers={"User-Agent": "EmbodiedAI-KB/0.1 academic-paper-search"},
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = response.read()
            if len(payload) > max_pdf_mb * 1024 * 1024:
                raise ValueError(f"PDF too large: {len(payload) / 1024 / 1024:.1f} MB")
            if not payload.startswith(b"%PDF"):
                raise ValueError("download did not return a PDF")
            tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
            tmp_path.write_bytes(payload)
            tmp_path.replace(output_path)
            trace["status"] = "downloaded"
            return str(output_path), trace
        except (OSError, TimeoutError, urllib.error.URLError, ValueError) as exc:
            last_error = exc
            if attempt >= retries:
                break
            time.sleep(1.5 * (attempt + 1))
    trace["status"] = "failed"
    trace["error"] = str(last_error) if last_error else "unknown download error"
    return "", trace


LOW_VALUE_QUERY_TOKENS = {
    "2020",
    "2021",
    "2022",
    "2023",
    "2024",
    "2025",
    "2026",
    "2027",
    "arxiv",
    "openreview",
    "paper",
    "papers",
    "representative",
    "latest",
    "recent",
    "survey",
    "code",
    "github",
    "icra",
    "iclr",
    "cvpr",
    "neurips",
    "conference",
    "journal",
    "study",
    "sequence",
    "milestones",
}


def _tokenize(text: str, *, drop_low_value: bool = False) -> list[str]:
    tokens = [
        token.lower()
        for token in re.findall(r"[\u4e00-\u9fff]+|[a-zA-Z0-9][a-zA-Z0-9-]+", text)
        if len(token) > 1
    ]
    if drop_low_value:
        tokens = [
            token
            for token in tokens
            if token not in LOW_VALUE_QUERY_TOKENS and not token.isdigit()
        ]
    return tokens


def _paper_score(paper: AcademicPaper, queries: list[str]) -> float:
    query_tokens = set(_tokenize(" ".join(queries), drop_low_value=True))
    title_tokens = set(_tokenize(paper.title))
    abstract_tokens = set(_tokenize(paper.abstract))
    author_tokens = set(_tokenize(" ".join(paper.authors)))
    category_tokens = set(_tokenize(" ".join(paper.categories + paper.keywords)))
    title_hits = query_tokens & title_tokens
    abstract_hits = query_tokens & abstract_tokens
    author_hits = query_tokens & author_tokens
    category_hits = query_tokens & category_tokens
    overlap = (
        len(title_hits) * 1.2
        + len(abstract_hits) * 0.10
        + len(author_hits) * 0.8
        + len(category_hits) * 0.35
    )
    if query_tokens and overlap < 0.8:
        return 0.0
    query_text = " ".join(queries).lower()
    haystack = " ".join([paper.title, paper.abstract, " ".join(paper.categories)]).lower()
    if "vla" in query_text and not (
        "vla" in haystack
        or ("vision" in haystack and "language" in haystack and "action" in haystack)
    ):
        return 0.0
    source_weight = {
        "openalex": 2.6,
        "arxiv": 2.4,
        "openreview": 2.8,
        "crossref": 1.8,
    }.get(paper.source, 1.0)
    year_boost = 0.0
    if paper.year:
        year_boost = max(0.0, min((paper.year - 2020) * 0.2, 1.4))
    return round(
        source_weight
        + overlap * 1.35
        + (2.0 if paper.pdf_url else 0.0)
        + (0.8 if paper.abstract else 0.0)
        + min(math.log1p(paper.citations or 0), 5.0) * 0.35
        + min(math.log1p(paper.influential_citations or 0), 4.0) * 0.55
        + year_boost,
        4,
    )


def _paper_to_search_result(
    paper: AcademicPaper,
    *,
    rank: int,
    score: float,
    query: str,
) -> SearchResult:
    merged_sources = [
        str(item)
        for item in paper.extra.get("merged_sources", [paper.source])
        if str(item).strip()
    ]
    venue = str(paper.extra.get("venue") or "")
    if not venue and paper.categories:
        venue = paper.categories[0]
    return SearchResult(
        rank=rank,
        hybrid_score=score,
        retrieval_score=score,
        metadata_score=score,
        paper_id=paper.paper_id,
        corpus="academic",
        title=paper.title,
        authors=paper.authors,
        year=paper.year,
        venue=venue or paper.source,
        abstract=paper.abstract or None,
        paper_url=paper.url or None,
        pdf_url=paper.pdf_url or None,
        code_url=None,
        project_url=None,
        doi=paper.doi or None,
        arxiv_id=str(paper.extra.get("arxiv_id") or "") or None,
        citation_count=paper.citations or None,
        influential_citation_count=paper.influential_citations or None,
        frontier_score=0.0,
        relevance_score=score,
        decision="academic_search",
        keywords=paper.keywords,
        categories=paper.categories,
        sources=["academic_paper_search", *merged_sources],
        quality_signals=[
            signal
            for signal, present in (
                ("direct_pdf_or_oa_pdf", bool(paper.pdf_url)),
                ("citation_count", bool(paper.citations)),
                ("doi", bool(paper.doi)),
                ("abstract", bool(paper.abstract)),
            )
            if present
        ],
        relevance_reasons=[
            f"academic multi-source result for query: {query}",
            f"sources: {', '.join(merged_sources)}",
        ],
        matched_terms=sorted(
            set(_tokenize(query, drop_low_value=True))
            & set(_tokenize(paper.title + " " + paper.abstract))
        )[:12],
    )


def _config_from_args(args: Any) -> AcademicSearchConfig:
    return AcademicSearchConfig(
        sources=_parse_sources(getattr(args, "academic_paper_sources", None)),
        max_queries=int(getattr(args, "academic_paper_max_queries", 6)),
        results_per_source=int(getattr(args, "academic_paper_results_per_source", 8)),
        top_k=int(getattr(args, "academic_paper_k", 6)),
        min_score=float(getattr(args, "academic_paper_min_score", 2.5)),
        year_from=getattr(args, "year_from", None),
        year_to=getattr(args, "year_to", None),
        resolve_pdf=not bool(getattr(args, "disable_oa_pdf_resolution", False)),
        request_delay=float(
            getattr(
                args,
                "academic_paper_request_delay",
                getattr(args, "request_delay", 1.5),
            )
            or 0.0
        ),
        max_workers=int(getattr(args, "academic_paper_max_workers", 2) or 2),
    )


async def run_academic_paper_search(
    *,
    question: str,
    queries: list[str],
    args: Any,
    source_queries: dict[str, list[str]] | None = None,
) -> tuple[list[SelectedPaper], dict[str, Any]]:
    """High-level academic paper search tool with OA-first PDF resolution."""

    if getattr(args, "disable_academic_paper_search", False):
        return [], {"mode": "disabled", "selected_count": 0, "candidate_count": 0}

    config = _config_from_args(args)
    search_queries = [query for query in queries if str(query).strip()]
    if question and question not in search_queries:
        search_queries.append(question)
    scoring_queries = search_queries
    if source_queries:
        scoring_queries = [
            str(query).strip()
            for source in config.sources
            for query in (source_queries.get(source) or [])
            if str(query).strip()
        ] or search_queries

    papers, trace = await asyncio.to_thread(search_papers, search_queries, config, source_queries)
    resolver = OAFallbackResolver()
    resolved_count = 0
    fallback_attempts: list[dict[str, Any]] = []
    if config.resolve_pdf:
        for paper in papers:
            before = paper.pdf_url
            pdf_url, attempts = await asyncio.to_thread(resolver.resolve_pdf_url, paper)
            if pdf_url and not before:
                paper.pdf_url = pdf_url
                resolved_count += 1
            if attempts:
                fallback_attempts.append(
                    {
                        "title": paper.title,
                        "source": paper.source,
                        "attempts": attempts,
                        "resolved": bool(pdf_url),
                    }
                )

    scored_all = [(_paper_score(paper, scoring_queries), paper) for paper in papers]
    scored_all.sort(key=lambda item: item[0], reverse=True)
    scored = [(score, paper) for score, paper in scored_all if score > config.min_score]

    selected: list[SelectedPaper] = []
    candidate_dicts: list[dict[str, Any]] = []
    for rank, (score, paper) in enumerate(scored_all, start=1):
        result = _paper_to_search_result(
            paper,
            rank=rank,
            score=score,
            query=scoring_queries[0] if scoring_queries else question,
        )
        candidate_dicts.append(result.to_dict())
        if score > config.min_score and result.pdf_url and len(selected) < config.top_k:
            selected.append(SelectedPaper(result=result, selection_score=score))

    trace.update(
        {
            "candidate_count": len(scored_all),
            "scored_candidate_count": len(scored),
            "selected_count": len(selected),
            "resolved_pdf_count": resolved_count,
            "fallback_attempt_count": len(fallback_attempts),
            "fallback_attempts": fallback_attempts[:20],
            "scoring_queries": scoring_queries[: config.max_queries * max(1, len(config.sources))],
            "selected_titles": [item.result.title for item in selected],
            "candidate_titles": [paper.title for _, paper in scored[:20]],
            "unpaywall_enabled": bool(os.getenv("PAPER_SEARCH_MCP_UNPAYWALL_EMAIL") or os.getenv("UNPAYWALL_EMAIL")),
            "candidates": candidate_dicts[:50],
        }
    )
    return selected, trace
