from __future__ import annotations

import html
import asyncio
import re
import time
from dataclasses import dataclass
from html.parser import HTMLParser
from textwrap import shorten
from typing import Any
from urllib.parse import parse_qs, urljoin, urlparse

import requests

from embodiedai_kb.search.metadata_search import SearchResult
from embodiedai_kb.storage.database import normalize_title
from scripts.ask_literature import SelectedPaper

from .web_search import DEFAULT_USER_AGENT, run_web_search_agent


PAPER_PAGE_DOMAINS = (
    "arxiv.org",
    "openreview.net",
    "thecvf.com",
    "openaccess.thecvf.com",
    "proceedings.mlr.press",
    "aclanthology.org",
)

PAGE_FETCH_DOMAINS = (
    "openreview.net",
    "arxiv.org",
    "openaccess.thecvf.com",
    "thecvf.com",
    "proceedings.mlr.press",
    "aclanthology.org",
)


@dataclass(slots=True)
class WebPaperCandidate:
    title: str
    paper_url: str
    pdf_url: str
    source_url: str
    source_title: str
    source_type: str
    authors: list[str]
    year: int | None
    venue: str | None
    project_url: str | None = None
    arxiv_id: str | None = None
    score: float = 0.0
    reasons: list[str] | None = None


class _LinkExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[tuple[str, str]] = []
        self.title_chunks: list[str] = []
        self._current_href: str | None = None
        self._current_text: list[str] = []
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = {key: value or "" for key, value in attrs}
        if tag == "a":
            self._current_href = attrs_dict.get("href", "")
            self._current_text = []
        elif tag == "title":
            self._in_title = True

    def handle_data(self, data: str) -> None:
        if self._current_href is not None:
            self._current_text.append(data)
        elif self._in_title:
            self.title_chunks.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._current_href is not None:
            text = _clean_text(" ".join(self._current_text))
            self.links.append((self._current_href, text))
            self._current_href = None
            self._current_text = []
        elif tag == "title":
            self._in_title = False


def _clean_text(value: str) -> str:
    value = html.unescape(value or "")
    value = re.sub(r"<[^>]+>", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _domain(url: str) -> str:
    return urlparse(url).netloc.lower().removeprefix("www.")


def _is_allowed_paper_domain(url: str) -> bool:
    domain = _domain(url)
    return any(domain == item or domain.endswith("." + item) for item in PAPER_PAGE_DOMAINS)


def _is_fetchable_domain(url: str) -> bool:
    domain = _domain(url)
    return any(domain == item or domain.endswith("." + item) for item in PAGE_FETCH_DOMAINS)


def _is_likely_source_page(url: str, title: str, snippet: str = "") -> bool:
    """Allow generic scholarly/profile pages discovered by web search.

    This is intentionally not tied to a specific person or lab. It lets the
    paper discovery tool inspect pages that are likely to contain publication
    links, such as university faculty pages, personal homepages, lab/project
    pages, or publication lists.
    """

    if _is_allowed_paper_domain(url) or _is_fetchable_domain(url):
        return True
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return False
    text = f"{title} {snippet} {url}".lower()
    positive_markers = (
        "publication",
        "publications",
        "paper",
        "papers",
        "faculty",
        "profile",
        "homepage",
        "personal",
        "research",
        "project page",
        "lab",
        "teacher",
        "google scholar",
        "个人主页",
        "主页",
        "教师",
        "师资",
        "论文",
        "课题组",
        "研究方向",
    )
    negative_markers = (
        "csdn.net",
        "zhihu.com",
        "github.com",
        "wikipedia.org",
        "baike.baidu.com",
        "sadscv.com",
    )
    if any(marker in text for marker in negative_markers):
        return False
    return any(marker in text for marker in positive_markers)


def _extract_urls(text: str) -> list[str]:
    urls = re.findall(r"https?://[^\s)）\"'<>]+", text or "")
    return [url.rstrip(".,;:，。；：") for url in urls]


def _arxiv_id_from_url(url: str) -> str | None:
    parsed = urlparse(url)
    if "arxiv.org" not in parsed.netloc:
        return None
    match = re.search(r"/(?:abs|pdf|html)/([0-9]{4}\.[0-9]{4,5}(?:v[0-9]+)?)", parsed.path)
    return match.group(1) if match else None


def _paper_urls_from_url(url: str) -> tuple[str | None, str | None, str | None]:
    """Return (paper_url, pdf_url, arxiv_id) when a URL points to a known paper page."""

    parsed = urlparse(url)
    domain = parsed.netloc.lower()
    arxiv_id = _arxiv_id_from_url(url)
    if arxiv_id:
        clean_id = arxiv_id.removesuffix(".pdf")
        return (
            f"https://arxiv.org/abs/{clean_id}",
            f"https://arxiv.org/pdf/{clean_id}",
            clean_id,
        )

    if "openreview.net" in domain:
        query = parse_qs(parsed.query)
        paper_id = (query.get("id") or [""])[0]
        if paper_id:
            return (
                f"https://openreview.net/forum?id={paper_id}",
                f"https://openreview.net/pdf?id={paper_id}",
                None,
            )

    if parsed.path.lower().endswith(".pdf"):
        return (url, url, None)

    return (None, None, None)


def _year_from_text(text: str) -> int | None:
    years = [int(value) for value in re.findall(r"\b(20[0-9]{2})\b", text or "")]
    if not years:
        return None
    return max(years)


def _strip_venue_prefix(title: str) -> tuple[str, str | None]:
    match = re.match(r"^\[([^\]]+)\]\s*(.+)$", title.strip())
    if not match:
        return title.strip(), None
    return match.group(2).strip(), match.group(1).strip()


def _metadata_from_link_context(
    html_text: str,
    href: str,
    *,
    page_title: str,
) -> tuple[str, list[str], int | None, str | None]:
    """Infer title/authors/venue from text before a nearby PDF link."""

    idx = html_text.find(href)
    if idx < 0:
        idx = html_text.find(html.escape(href, quote=True))
    if idx < 0:
        return page_title, [], _year_from_text(page_title), None

    context = _clean_text(html_text[max(0, idx - 1400) : idx])
    # Keep only the last publication-looking segment on long pages.
    bracket_matches = list(re.finditer(r"\[([^\]]{2,40})\]", context))
    venue = None
    segment = context
    if bracket_matches:
        last = bracket_matches[-1]
        venue = last.group(1).strip()
        segment = context[last.end() :].strip()

    # Most academic homepages use: [VENUE YEAR] Title. Author, Author, ...
    title = segment
    author_text = ""
    boundary = re.search(r"(.+?(?:\?|\.))\s+([A-Z][^。]{2,500})$", segment)
    if boundary:
        title = boundary.group(1).strip()
        author_text = boundary.group(2).strip()
    else:
        title = segment.strip()

    title = re.sub(r"\s*(PDF|Code|Project Page|Project|Paper)\s*$", "", title, flags=re.I)
    title = re.sub(r'^[^[]*">\s*', "", title)
    title = re.sub(r"\b(?:font|vertical-align|baseline|style|color)\s*:[^]]+", "", title)
    if len(title) > 260:
        title = title[-260:]
        title = re.sub(r"^.*?(?=[A-Z][A-Za-z0-9-]+:)", "", title).strip() or title
    if not title or title.lower() in {"pdf", "paper", "arxiv", "openreview"}:
        title = page_title

    authors = [
        _clean_text(value)
        for value in re.split(r",|\band\b", author_text)
        if _clean_text(value) and len(_clean_text(value)) <= 80
    ]
    year = _year_from_text(venue or segment or page_title)
    return title, authors, year, venue


def _fetch_html(url: str, *, timeout: float) -> str:
    response = requests.get(
        url,
        headers={"User-Agent": DEFAULT_USER_AGENT},
        timeout=timeout,
    )
    response.raise_for_status()
    content_type = response.headers.get("Content-Type", "")
    if "text/html" not in content_type and "application/xhtml" not in content_type:
        return ""
    return response.text


def _extract_publication_items(
    html_text: str,
    *,
    page_url: str,
    source_title: str,
) -> list[WebPaperCandidate]:
    """Parse Hugo Academic publication cards, especially personal lab pages."""

    candidates: list[WebPaperCandidate] = []
    blocks = re.split(r'<div class="pub-list-item"', html_text)
    for block in blocks[1:]:
        title_match = re.search(
            r'<h3 class="article-title"[^>]*>.*?<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
            block,
            flags=re.DOTALL,
        )
        if not title_match:
            continue
        paper_url = urljoin(page_url, html.unescape(title_match.group(1)))
        raw_title = _clean_text(title_match.group(2))
        title, venue_from_prefix = _strip_venue_prefix(raw_title)
        authors_match = re.search(
            r'<div class="pub-authors"[^>]*>(.*?)</div>',
            block,
            flags=re.DOTALL,
        )
        authors = []
        if authors_match:
            authors = [
                _clean_text(value)
                for value in re.split(r",|\band\b", _clean_text(authors_match.group(1)))
                if _clean_text(value)
            ]
        publication_match = re.search(
            r'<div class="pub-publication"[^>]*>(.*?)</div>',
            block,
            flags=re.DOTALL,
        )
        publication = _clean_text(publication_match.group(1)) if publication_match else ""
        venue = venue_from_prefix or publication or None
        year = _year_from_text(publication or raw_title)

        links = re.findall(r'<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>', block, flags=re.DOTALL)
        pdf_url = ""
        project_url = None
        for href, label in links:
            absolute = urljoin(page_url, html.unescape(href))
            label_text = _clean_text(label).lower()
            paper_page, pdf_candidate, arxiv_id = _paper_urls_from_url(absolute)
            if label_text == "project":
                project_url = absolute
            if label_text == "pdf" or pdf_candidate:
                pdf_url = pdf_candidate or absolute
                if paper_page:
                    paper_url = paper_page
                break
        if not pdf_url:
            continue
        arxiv_id = _arxiv_id_from_url(pdf_url)
        candidates.append(
            WebPaperCandidate(
                title=title,
                paper_url=paper_url,
                pdf_url=pdf_url,
                source_url=page_url,
                source_title=source_title,
                source_type="publication_page",
                authors=authors,
                year=year,
                venue=venue,
                project_url=project_url,
                arxiv_id=arxiv_id,
                reasons=["publication_card", f"source={_domain(page_url)}"],
            )
        )
    return candidates


def _generic_link_candidates(
    html_text: str,
    *,
    page_url: str,
    source_title: str,
) -> list[WebPaperCandidate]:
    parser = _LinkExtractor()
    parser.feed(html_text)
    page_title = _clean_text(" ".join(parser.title_chunks)) or source_title
    candidates: list[WebPaperCandidate] = []
    for href, label in parser.links:
        absolute = urljoin(page_url, href)
        if not _is_allowed_paper_domain(absolute):
            continue
        paper_url, pdf_url, arxiv_id = _paper_urls_from_url(absolute)
        if not pdf_url:
            continue
        title = _clean_text(label) or page_title
        title, venue_from_prefix = _strip_venue_prefix(title)
        if title.lower() in {"pdf", "paper", "arxiv", "openreview"}:
            inferred_title, inferred_authors, inferred_year, inferred_venue = (
                _metadata_from_link_context(
                    html_text,
                    href,
                    page_title=page_title,
                )
            )
            title = inferred_title
            authors = inferred_authors
            year = inferred_year
            venue = inferred_venue or venue_from_prefix
        else:
            authors = []
            year = _year_from_text(f"{title} {page_title}")
            venue = venue_from_prefix
        candidates.append(
            WebPaperCandidate(
                title=title,
                paper_url=paper_url or absolute,
                pdf_url=pdf_url,
                source_url=page_url,
                source_title=source_title or page_title,
                source_type="page_link",
                authors=authors,
                year=year,
                venue=venue,
                arxiv_id=arxiv_id,
                reasons=["paper_link", f"source={_domain(page_url)}"],
            )
        )
    return candidates


def _candidate_from_search_result(item: dict[str, Any]) -> WebPaperCandidate | None:
    url = str(item.get("url") or "")
    if not _is_allowed_paper_domain(url):
        return None
    paper_url, pdf_url, arxiv_id = _paper_urls_from_url(url)
    if not pdf_url:
        return None
    title = _clean_text(str(item.get("title") or ""))
    title, venue_from_prefix = _strip_venue_prefix(title)
    return WebPaperCandidate(
        title=title or paper_url or url,
        paper_url=paper_url or url,
        pdf_url=pdf_url,
        source_url=url,
        source_title=title or str(item.get("title") or ""),
        source_type="search_result",
        authors=[],
        year=_year_from_text(f"{title} {item.get('snippet', '')}"),
        venue=venue_from_prefix,
        arxiv_id=arxiv_id,
        reasons=["search_result", f"source={_domain(url)}"],
    )


def _score_candidate(candidate: WebPaperCandidate, question: str, queries: list[str]) -> float:
    text = " ".join(
        [
            candidate.title,
            " ".join(candidate.authors),
            candidate.venue or "",
            candidate.paper_url,
            candidate.source_url,
            candidate.source_title,
        ]
    ).lower()
    query_text = " ".join([question, *queries]).lower()
    score = 18.0
    reasons = list(candidate.reasons or [])

    source_domain = _domain(candidate.source_url)
    if source_domain in {
        "arxiv.org",
        "openreview.net",
        "openaccess.thecvf.com",
        "thecvf.com",
        "proceedings.mlr.press",
        "aclanthology.org",
    }:
        score += 8.0
        reasons.append("academic_paper_site")
    source_context = f"{candidate.source_title} {candidate.source_url}".lower()
    if any(
        marker in source_context
        for marker in (
            "publication",
            "publications",
            "faculty",
            "profile",
            "homepage",
            "teacher",
            "personal",
            "research",
            "教师",
            "师资",
            "个人主页",
            "论文",
            "研究方向",
        )
    ):
        score += 10.0
        reasons.append("scholarly_source_page")

    query_tokens = {
        token
        for token in re.findall(r"[a-zA-Z][a-zA-Z0-9-]{1,}", query_text)
        if token
        not in {
            "paper",
            "papers",
            "publication",
            "publications",
            "research",
            "recent",
            "teacher",
            "professor",
            "university",
            "2024",
            "2025",
            "2026",
        }
    }
    overlap = sorted(token for token in query_tokens if token in text)
    if overlap:
        score += min(len(overlap) * 2.0, 10.0)
        reasons.append("query_text_overlap")
    if "上海交通大学" in question and ("sjtu" in text or "shanghai jiao tong" in text):
        score += 4.0
        reasons.append("query_affiliation_overlap")

    if "openreview.net" in candidate.pdf_url:
        score += 4.0
        reasons.append("openreview_pdf")
    if "arxiv.org" in candidate.pdf_url:
        score += 3.0
        reasons.append("arxiv_pdf")
    if candidate.year and candidate.year >= 2025:
        score += 3.0
        reasons.append("recent")
    if any(token in text for token in ("robot", "embodied", "vla", "motion", "agent", "reinforcement")):
        score += 2.0
        reasons.append("topic_match")
    if candidate.source_type == "publication_page":
        score += 5.0
        reasons.append("structured_publication_page")

    candidate.reasons = reasons
    candidate.score = round(score, 4)
    return candidate.score


def _dedupe_candidates(candidates: list[WebPaperCandidate]) -> list[WebPaperCandidate]:
    by_key: dict[str, WebPaperCandidate] = {}
    for candidate in candidates:
        key = (
            candidate.arxiv_id
            or candidate.pdf_url.lower()
            or candidate.paper_url.lower()
            or normalize_title(candidate.title)
        )
        previous = by_key.get(key)
        if previous is None or candidate.score > previous.score:
            by_key[key] = candidate
    return list(by_key.values())


def _candidate_to_selected(candidate: WebPaperCandidate, rank: int) -> SelectedPaper:
    paper_id = candidate.arxiv_id or normalize_title(candidate.title).replace(" ", "-")[:80]
    if not paper_id:
        paper_id = f"web:{rank}"
    if not paper_id.startswith("web:") and candidate.source_type:
        paper_id = f"web:{paper_id}"
    result = SearchResult(
        rank=rank,
        hybrid_score=round(candidate.score, 4),
        retrieval_score=round(candidate.score, 4),
        metadata_score=0.0,
        paper_id=paper_id,
        corpus="web",
        title=candidate.title,
        authors=candidate.authors,
        year=candidate.year,
        venue=candidate.venue or "web-discovered",
        abstract=None,
        paper_url=candidate.paper_url,
        pdf_url=candidate.pdf_url,
        code_url=None,
        project_url=candidate.project_url or candidate.source_url,
        doi=None,
        arxiv_id=candidate.arxiv_id,
        citation_count=None,
        influential_citation_count=None,
        frontier_score=0.0,
        relevance_score=0.0,
        decision="web_discovered",
        keywords=[],
        categories=[],
        sources=["web_paper_discovery", candidate.source_type],
        quality_signals=["web_pdf_url", *(candidate.reasons or [])],
        relevance_reasons=[
            f"Discovered from {candidate.source_title or candidate.source_url}",
            f"source_url={candidate.source_url}",
        ],
        matched_terms=[],
    )
    return SelectedPaper(result=result, selection_score=round(candidate.score, 4))


def _discovery_queries(question: str, queries: list[str], max_queries: int) -> list[str]:
    seeds: list[str] = []
    seen: set[str] = set()

    def add(query: str) -> None:
        clean = re.sub(r"\s+", " ", query).strip()
        key = clean.lower()
        if clean and key not in seen:
            seen.add(key)
            seeds.append(clean)

    add(f"{question} site:arxiv.org/abs")
    add(f"{question} site:openreview.net/forum")
    add(f"{question} site:openaccess.thecvf.com")
    add(f"{question} site:proceedings.mlr.press")
    for query in queries[:3]:
        add(f"{query} PDF arxiv openreview")
    return seeds[:max_queries]


def _collect_seed_urls(
    web_evidence: list[dict[str, Any]],
    discovery_evidence: list[dict[str, Any]],
) -> list[tuple[str, str]]:
    seed_urls: list[tuple[str, str]] = []
    seen: set[str] = set()

    def add(url: str, title: str, snippet: str = "") -> None:
        clean = url.strip()
        if not clean or clean in seen:
            return
        if not _is_likely_source_page(clean, title, snippet):
            return
        seen.add(clean)
        seed_urls.append((clean, title))

    for item in [*web_evidence, *discovery_evidence]:
        title = str(item.get("title") or "")
        snippet = str(item.get("snippet") or "")
        add(str(item.get("url") or ""), title, snippet)
        for url in _extract_urls(snippet):
            add(url, title, snippet)
    return seed_urls


def _discover_sync(
    *,
    question: str,
    queries: list[str],
    web_evidence: list[dict[str, Any]],
    discovery_evidence: list[dict[str, Any]],
    max_pages: int,
    timeout: float,
    request_delay: float,
    top_k: int,
    min_score: float,
) -> tuple[list[SelectedPaper], dict[str, Any]]:
    candidates: list[WebPaperCandidate] = []
    fetched_pages: list[str] = []
    errors: list[dict[str, str]] = []

    for item in [*web_evidence, *discovery_evidence]:
        candidate = _candidate_from_search_result(item)
        if candidate is not None:
            candidates.append(candidate)

    seed_urls = _collect_seed_urls(
        web_evidence,
        discovery_evidence,
    )
    for idx, (url, title) in enumerate(seed_urls[:max_pages]):
        if idx and request_delay > 0:
            time.sleep(request_delay)
        try:
            html_text = _fetch_html(url, timeout=timeout)
        except requests.RequestException as exc:
            errors.append({"url": url, "error": shorten(str(exc), width=180, placeholder="...")})
            continue
        if not html_text:
            continue
        fetched_pages.append(url)
        candidates.extend(
            _extract_publication_items(
                html_text,
                page_url=url,
                source_title=title,
            )
        )
        candidates.extend(
            _generic_link_candidates(
                html_text,
                page_url=url,
                source_title=title,
            )
        )

    for candidate in candidates:
        _score_candidate(candidate, question, queries)

    candidates = _dedupe_candidates(candidates)
    candidates = [candidate for candidate in candidates if candidate.score >= min_score]
    candidates.sort(key=lambda item: item.score, reverse=True)
    selected = [
        _candidate_to_selected(candidate, rank)
        for rank, candidate in enumerate(candidates[:top_k], start=1)
    ]
    trace = {
        "candidate_count": len(candidates),
        "selected_count": len(selected),
        "fetched_pages": fetched_pages,
        "errors": errors,
        "selected_titles": [item.result.title for item in selected],
    }
    return selected, trace


async def run_web_paper_discovery(
    *,
    question: str,
    queries: list[str],
    web_evidence: list[dict[str, Any]],
    planning_web_evidence: list[dict[str, Any]],
    args: Any,
) -> tuple[list[SelectedPaper], dict[str, Any]]:
    """Discover paper PDFs from authoritative web results and project pages."""

    if getattr(args, "disable_web_paper_discovery", False):
        return [], {"mode": "disabled", "candidate_count": 0, "selected_count": 0}
    if getattr(args, "disable_web_search", False):
        base_evidence = [*planning_web_evidence, *web_evidence]
        discovery_evidence: list[dict[str, Any]] = []
        discovery_trace = {"provider": "disabled", "query_count": 0, "result_count": 0}
    else:
        discovery_queries = _discovery_queries(
            question,
            queries,
            int(getattr(args, "web_paper_discovery_max_queries", 4)),
        )
        discovery_args = type("DiscoveryArgs", (), {})()
        discovery_args.disable_web_search = False
        discovery_args.web_max_queries = len(discovery_queries)
        discovery_args.web_results_per_query = int(
            getattr(args, "web_paper_discovery_results_per_query", 5)
        )
        discovery_args.web_search_timeout = float(getattr(args, "web_search_timeout", 12.0))
        discovery_args.web_search_delay = float(getattr(args, "web_search_delay", 0.4))
        discovery_args.web_search_provider = str(getattr(args, "web_search_provider", "auto"))
        discovery_args.tavily_api_key_env = str(
            getattr(args, "tavily_api_key_env", "TAVILY_API_KEY")
        )
        discovery_args.tavily_search_depth = str(
            getattr(args, "tavily_search_depth", "basic")
        )
        discovery_evidence, discovery_trace = await run_web_search_agent(
            discovery_queries,
            discovery_args,
        )
        base_evidence = [*planning_web_evidence, *web_evidence]

    selected, trace = await asyncio.to_thread(
        _discover_sync,
        question=question,
        queries=queries,
        web_evidence=base_evidence,
        discovery_evidence=discovery_evidence,
        max_pages=int(getattr(args, "web_paper_discovery_max_pages", 8)),
        timeout=float(getattr(args, "web_search_timeout", 12.0)),
        request_delay=float(getattr(args, "web_search_delay", 0.4)),
        top_k=int(getattr(args, "web_paper_discovery_k", 6)),
        min_score=float(getattr(args, "web_paper_discovery_min_score", 30.0)),
    )
    trace.update(
        {
            "mode": "web_paper_discovery",
            "search_trace": discovery_trace,
            "discovery_result_count": len(discovery_evidence),
        }
    )
    return selected, trace
