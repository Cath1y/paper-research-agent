from __future__ import annotations

import asyncio
import html
import json
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass
from html.parser import HTMLParser
from textwrap import shorten
from typing import Any

from embodiedai_kb.paper_search import run_academic_paper_search
from embodiedai_kb.paper_search.academic_platforms.common import (
    arxiv_id_from_url,
    arxiv_pdf_url,
    parse_date,
    stable_id,
    title_key,
)
from embodiedai_kb.paper_search.models import AcademicPaper
from embodiedai_kb.search.metadata_search import (
    MetadataSearchEngine,
    SearchCorpus,
    SearchFilters,
    SearchResult,
    reciprocal_rank_fusion,
)
from scripts.ask_literature import (
    SelectedPaper,
    normalize_openai_compatible_model,
    openai_compatible_api_key,
    parse_venues,
    select_for_paperqa,
)

from .progress import emit_progress
from .web_search import DEFAULT_USER_AGENT, run_web_search_agent


ACADEMIC_PLATFORMS = ("arxiv", "openalex", "openreview", "crossref")
ACADEMIC_TOOL_TO_PLATFORM = {
    "arxiv_search": "arxiv",
    "openalex_search": "openalex",
    "openreview_search": "openreview",
    "crossref_search": "crossref",
}
WEB_SEARCH_TOOLS = (
    "web_search",
    "author_homepage",
    "publication_page",
    "google_scholar_profile",
    "dblp_profile",
    "openreview_profile",
    "project_page",
)
PAPER_SEARCH_TOOLS = (*ACADEMIC_TOOL_TO_PLATFORM, *WEB_SEARCH_TOOLS)


@dataclass(slots=True)
class PaperSearchPlan:
    """Composable paper-search plan produced by the query-planning LLM."""

    task_signals: dict[str, bool]
    authors: list[str]
    content_query: str
    search_queries: list[str]
    tool_queries: dict[str, list[str]]
    platform_queries: dict[str, list[str]]
    web_queries: list[str]
    reasoning: str = ""


class LLMJSONError(ValueError):
    """Raised when an LLM response cannot be parsed as a JSON object."""

    def __init__(self, message: str, raw_text: str = "") -> None:
        super().__init__(message)
        self.raw_text = raw_text


class _LinkParser(HTMLParser):
    """Extract readable text and links from publication/profile pages."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.text_chunks: list[str] = []
        self.links: list[dict[str, str]] = []
        self._href = ""
        self._anchor_chunks: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag_name = tag.lower()
        if tag_name in {"script", "style", "noscript", "template", "svg"}:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if tag_name != "a":
            return
        attrs_dict = {key.lower(): value or "" for key, value in attrs}
        self._href = attrs_dict.get("href", "")
        self._anchor_chunks = []

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        clean = re.sub(r"\s+", " ", html.unescape(data)).strip()
        if not clean:
            return
        self.text_chunks.append(clean)
        if self._href:
            self._anchor_chunks.append(clean)

    def handle_endtag(self, tag: str) -> None:
        tag_name = tag.lower()
        if tag_name in {"script", "style", "noscript", "template", "svg"} and self._skip_depth:
            self._skip_depth -= 1
            return
        if self._skip_depth:
            return
        if tag_name != "a" or not self._href:
            return
        text = re.sub(r"\s+", " ", " ".join(self._anchor_chunks)).strip()
        self.links.append({"text": text, "href": self._href})
        self._href = ""
        self._anchor_chunks = []


def _paper_search_model(args: Any) -> str | None:
    model = (
        getattr(args, "paper_search_llm", None)
        or getattr(args, "agent_llm", None)
        or getattr(args, "router_llm", None)
        or getattr(args, "llm", None)
    )
    if not model:
        return None
    if getattr(args, "openai_base_url", None) and not getattr(
        args,
        "disable_openai_compatible_config",
        False,
    ):
        return normalize_openai_compatible_model(model, args, args.openai_base_url)
    return model


def _json_from_text(text: str) -> dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise LLMJSONError("No JSON object found.", text)

    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise LLMJSONError(str(exc), text) from exc
    if not isinstance(payload, dict):
        raise LLMJSONError("Expected JSON object.", text)
    return payload


async def _llm_json(
    *,
    model: str,
    messages: list[dict[str, str]],
    args: Any,
    max_tokens: int = 1000,
    json_mode: bool = True,
) -> dict[str, Any]:
    import litellm

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
        "timeout": float(getattr(args, "llm_timeout", 180.0)),
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    if getattr(args, "openai_base_url", None) and not getattr(
        args,
        "disable_openai_compatible_config",
        False,
    ):
        kwargs["api_base"] = args.openai_base_url
        api_key = openai_compatible_api_key(
            args,
            args.openai_base_url,
            args.openai_api_key_env,
        )
        if api_key:
            kwargs["api_key"] = api_key
    response = await litellm.acompletion(**kwargs)
    content = (response.choices[0].message.content or "").strip()
    try:
        return _json_from_text(content)
    except LLMJSONError as first_exc:
        if not json_mode:
            raise
        repair_kwargs = {
            **kwargs,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Repair the user's malformed JSON into one valid JSON object. "
                        "Preserve the same keys and values as much as possible. "
                        "Remove comments, markdown, trailing commas, unescaped quotes, "
                        "and incomplete list items. Return only valid JSON."
                    ),
                },
                {"role": "user", "content": content[:12000]},
            ],
            "max_tokens": max(max_tokens, 2200),
            "response_format": {"type": "json_object"},
        }
        repair_response = await litellm.acompletion(**repair_kwargs)
        repair_content = (repair_response.choices[0].message.content or "").strip()
        try:
            return _json_from_text(repair_content)
        except LLMJSONError as repair_exc:
            raise LLMJSONError(
                f"{first_exc}; JSON repair failed: {repair_exc}",
                f"{content}\n\n--- JSON REPAIR RESPONSE ---\n{repair_content}",
            ) from repair_exc


def _fallback_plan(question: str, queries: list[str]) -> PaperSearchPlan:
    authors: list[str] = []
    match = re.search(r"([\u4e00-\u9fff]{2,4})(?:老师|教授|副教授|研究员)", question)
    if match:
        name = match.group(1)
        if len(name) > 3:
            name = name[-2:]
        authors.append(name)
    english_by = re.search(r"\b(?:papers?|work|publications?)\s+(?:by|from)\s+([A-Z][A-Za-z-]+(?:\s+[A-Z][A-Za-z-]+){0,2})", question)
    if english_by:
        authors.append(english_by.group(1).strip())

    tool_queries: dict[str, list[str]] = {"web_search": [*queries[:4], question]}
    if authors:
        author_terms = [f"{author} publications" for author in authors]
        tool_queries.update(
            {
                "author_homepage": author_terms,
                "publication_page": author_terms,
                "google_scholar_profile": [f"{author} Google Scholar" for author in authors],
                "dblp_profile": [f"{author} DBLP" for author in authors],
            }
        )
    content = question
    return PaperSearchPlan(
        task_signals={
            "needs_author_disambiguation": bool(authors),
            "needs_specific_paper_lookup": False,
            "needs_topic_survey": not bool(authors),
            "needs_recent_publications": bool(authors),
            "needs_pdf_download": True,
            "needs_profile_pages": bool(authors),
        },
        authors=list(dict.fromkeys(authors)),
        content_query=content,
        search_queries=[*queries[:6], question],
        tool_queries=tool_queries,
        platform_queries={},
        web_queries=_web_queries_from_tool_queries(tool_queries),
        reasoning="fallback query analysis",
    )


def _error_plan(question: str) -> PaperSearchPlan:
    return PaperSearchPlan(
        task_signals={},
        authors=[],
        content_query=question.strip(),
        search_queries=[],
        tool_queries={},
        platform_queries={},
        web_queries=[],
        reasoning="query analyzer failed; academic platform search intentionally skipped",
    )


def _clean_query_list(values: Any, limit: int = 8) -> list[str]:
    if not isinstance(values, list):
        return []
    cleaned: list[str] = []
    for item in values:
        text = re.sub(r"\s+", " ", str(item or "")).strip()
        if text and text not in cleaned:
            cleaned.append(text)
        if len(cleaned) >= limit:
            break
    return cleaned


def _coerce_task_signals(value: Any) -> dict[str, bool]:
    if not isinstance(value, dict):
        return {}
    return {
        str(key).strip(): bool(raw)
        for key, raw in value.items()
        if str(key).strip()
    }


def _coerce_tool_queries(payload: dict[str, Any]) -> dict[str, list[str]]:
    tool_queries: dict[str, list[str]] = {}
    raw_tool_queries = payload.get("tool_queries") or {}
    if isinstance(raw_tool_queries, dict):
        for tool in PAPER_SEARCH_TOOLS:
            values = _clean_query_list(raw_tool_queries.get(tool))
            if values:
                tool_queries[tool] = values

    # Backward-compatible parser for older prompt outputs.
    raw_platform_queries = payload.get("platform_queries") or {}
    if isinstance(raw_platform_queries, dict):
        for tool, source in ACADEMIC_TOOL_TO_PLATFORM.items():
            values = _clean_query_list(raw_platform_queries.get(source))
            if values and tool not in tool_queries:
                tool_queries[tool] = values

    legacy_web_queries = _clean_query_list(payload.get("web_queries"), limit=12)
    if legacy_web_queries and "web_search" not in tool_queries:
        tool_queries["web_search"] = legacy_web_queries

    return tool_queries


def _platform_queries_from_tool_queries(tool_queries: dict[str, list[str]]) -> dict[str, list[str]]:
    platform_queries: dict[str, list[str]] = {}
    for tool, source in ACADEMIC_TOOL_TO_PLATFORM.items():
        values = [query for query in tool_queries.get(tool, []) if query.strip()]
        if values:
            platform_queries[source] = values
    return platform_queries


def _web_queries_from_tool_queries(tool_queries: dict[str, list[str]]) -> list[str]:
    web_queries: list[str] = []
    for tool in WEB_SEARCH_TOOLS:
        for query in tool_queries.get(tool, []):
            if query.strip() and query not in web_queries:
                web_queries.append(query)
    return web_queries


def _merge_unique(values: list[str], incoming: list[str], limit: int | None = None) -> list[str]:
    merged = list(values)
    for value in incoming:
        clean = re.sub(r"\s+", " ", str(value or "")).strip()
        if clean and clean not in merged:
            merged.append(clean)
        if limit is not None and len(merged) >= limit:
            break
    return merged


def _merge_tool_queries(*plans: PaperSearchPlan) -> dict[str, list[str]]:
    merged: dict[str, list[str]] = {}
    for plan in plans:
        for tool, queries in plan.tool_queries.items():
            merged[tool] = _merge_unique(merged.get(tool, []), queries)
    return merged


def _merge_web_evidence(
    existing: list[dict[str, Any]],
    incoming: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged = list(existing)
    seen = {str(item.get("url") or "") for item in merged if item.get("url")}
    for item in incoming:
        url = str(item.get("url") or "")
        if url and url in seen:
            continue
        if url:
            seen.add(url)
        merged.append(item)
    return merged


def _paper_search_plan_from_payload(
    payload: dict[str, Any],
    *,
    question: str,
    queries: list[str],
) -> PaperSearchPlan:
    authors = [str(item).strip() for item in payload.get("authors") or [] if str(item).strip()]
    search_queries = _clean_query_list(payload.get("search_queries"), limit=12)
    task_signals = _coerce_task_signals(payload.get("task_signals"))
    tool_queries = _coerce_tool_queries(payload)
    platform_queries = _platform_queries_from_tool_queries(tool_queries)
    web_queries = _web_queries_from_tool_queries(tool_queries)
    return PaperSearchPlan(
        task_signals=task_signals,
        authors=authors,
        content_query=str(payload.get("content_query") or question).strip(),
        search_queries=search_queries or [*queries[:6], question],
        tool_queries=tool_queries,
        platform_queries=platform_queries,
        web_queries=web_queries,
        reasoning=str(payload.get("reasoning") or ""),
    )


def _candidate_snapshot(candidates: list[SearchResult], limit: int = 12) -> list[dict[str, Any]]:
    return [
        {
            "title": item.title,
            "authors": item.authors[:6],
            "year": item.year,
            "venue": item.venue,
            "pdf": bool(item.pdf_url),
            "url": item.paper_url or item.pdf_url,
            "sources": item.sources,
        }
        for item in candidates[:limit]
    ]


def _search_loop_observation(
    *,
    candidates: list[SearchResult],
    web_evidence: list[dict[str, Any]],
    iteration_traces: list[dict[str, Any]],
) -> dict[str, Any]:
    pdf_candidates = [item for item in candidates if item.pdf_url]
    return {
        "candidate_count": len(candidates),
        "pdf_candidate_count": len(pdf_candidates),
        "web_evidence_count": len(web_evidence),
        "candidate_snapshot": _candidate_snapshot(candidates),
        "recent_iteration_summaries": [
            {
                "iteration": item.get("iteration"),
                "called_tools": item.get("called_tools"),
                "workflow_candidate_count": item.get("workflow_candidate_count"),
                "pdf_candidate_count": item.get("pdf_candidate_count"),
                "web_result_count": item.get("web_search_trace", {}).get("result_count"),
                "source_results": item.get("academic", {}).get("source_results"),
                "web_profile_extract": item.get("web_profile", {}).get("web_page_extract"),
            }
            for item in iteration_traces[-3:]
        ],
    }


def _should_refine_paper_search(
    *,
    candidates: list[SearchResult],
    iteration: int,
    max_iterations: int,
    args: Any,
) -> tuple[bool, str]:
    if iteration >= max_iterations:
        return False, "max_iterations_reached"
    pdf_count = len([item for item in candidates if item.pdf_url])
    default_min_pdf = max(3, min(int(getattr(args, "paperqa_k", 8) or 8), 5))
    min_pdf = int(getattr(args, "paper_search_min_pdf_candidates", None) or default_min_pdf)
    min_total = int(getattr(args, "paper_search_min_candidates", None) or max(min_pdf, 8))
    if pdf_count < min_pdf:
        return True, f"pdf_candidates_below_target:{pdf_count}<{min_pdf}"
    if len(candidates) < min_total:
        return True, f"total_candidates_below_target:{len(candidates)}<{min_total}"
    return False, "candidate_targets_met"


async def analyze_paper_search_query(
    *,
    question: str,
    queries: list[str],
    research_plan: list[dict[str, Any]],
    args: Any,
) -> tuple[PaperSearchPlan, dict[str, Any]]:
    """Plan which paper-search tools should be used before retrieval.

    The planner can emit many tool queries at once. Task signals are kept as
    hints for debugging and later supervision, not as hard routing labels.
    """

    model = _paper_search_model(args)
    if not model:
        plan = _fallback_plan(question, queries)
        return plan, {"mode": "fallback_no_llm", "model": None}

    plan_context = "\n".join(
        f"- {item.get('perspective') or item.get('agent') or 'plan'}: "
        f"{item.get('research_question') or item.get('objective') or item.get('description') or ''}"
        for item in research_plan[:8]
    )
    try:
        payload = await _llm_json(
            model=model,
            args=args,
            max_tokens=1600,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are PaperSearchAgent.SearchPlanner. Build a composable search plan "
                        "for finding academic papers. Do not force the request into one of a few "
                        "classes; complex requests can need several tools at once. "
                        "Return exactly one JSON object. Do not use markdown, comments, trailing commas, "
                        "or any explanatory text outside JSON. "
                        "Use task_signals only as boolean hints, not as exclusive routes. Supported "
                        "signals include needs_author_disambiguation, needs_specific_paper_lookup, "
                        "needs_topic_survey, needs_recent_publications, needs_pdf_download, and "
                        "needs_profile_pages. "
                        "Generate tool_queries for any useful tools. Available tools are: "
                        "arxiv_search, openalex_search, openreview_search, crossref_search, "
                        "web_search, author_homepage, publication_page, google_scholar_profile, "
                        "dblp_profile, openreview_profile, project_page. "
                        "Keep the plan compact: at most 8 search_queries, at most 2 queries per tool, "
                        "and only include tools that are useful for this request. "
                        "Generate arXiv queries using arXiv API syntax only: au:, ti:, abs:, "
                        "all:, cat:, co:, AND, OR, ANDNOT, and quoted phrases. "
                        "For arXiv author requests, combine author and topic/title constraints, for example "
                        "au:\"Jane Doe\" AND all:\"robot learning\" or au:\"Jane Doe\" AND ti:\"example paper\". "
                        "Do not put web-only terms like Google Scholar, homepage, site:, URL fragments, "
                        "or institution pages in arxiv_search/openalex_search/openreview_search/"
                        "crossref_search queries. Use google_scholar_profile only for web search "
                        "queries that locate a scholar profile; do not assume direct Google Scholar "
                        "API access. Use author_homepage/publication_page/dblp_profile/openreview_profile "
                        "when identity disambiguation or author publication lists matter. Treat "
                        "crossref_search as a fallback for exact titles or DOI-like lookup, not as a "
                        "primary broad author/topic search source. For topic surveys, combine academic "
                        "database queries with broad web/project-page queries."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"User question:\n{question}\n\n"
                        f"Research plan/context:\n{plan_context or 'N/A'}\n\n"
                        f"Existing planned queries:\n"
                        + "\n".join(f"- {query}" for query in queries[:12])
                        + "\n\nReturn a JSON object with this exact shape. Values below are examples only; "
                        "replace them with values for this request:\n"
                        "{\n"
                        '  "task_signals": {\n'
                        '    "needs_author_disambiguation": true,\n'
                        '    "needs_specific_paper_lookup": false,\n'
                        '    "needs_topic_survey": true,\n'
                        '    "needs_recent_publications": true,\n'
                        '    "needs_pdf_download": true,\n'
                        '    "needs_profile_pages": true\n'
                        "  },\n"
                        '  "authors": ["Jane Doe"],\n'
                        '  "content_query": "recent publications and research themes",\n'
                        '  "search_queries": ["Jane Doe robot learning recent publications", "Jane Doe embodied AI"],\n'
                        '  "tool_queries": {\n'
                        '    "arxiv_search": ["au:\\"Jane Doe\\" AND all:\\"robot learning\\"", "au:\\"Jane Doe\\" AND all:embodied"],\n'
                        '    "openalex_search": ["Jane Doe robot learning", "Jane Doe embodied AI"],\n'
                        '    "openreview_search": ["Jane Doe ICLR robot learning"],\n'
                        '    "crossref_search": ["Jane Doe university robot learning"],\n'
                        '    "author_homepage": ["Jane Doe university homepage publications"],\n'
                        '    "publication_page": ["Jane Doe publications"],\n'
                        '    "google_scholar_profile": ["Jane Doe Google Scholar"],\n'
                        '    "dblp_profile": ["Jane Doe DBLP"],\n'
                        '    "web_search": ["Jane Doe recent papers robot learning"]\n'
                        "  },\n"
                        '  "reasoning": "short reason"\n'
                        "}"
                    ),
                },
            ],
        )
        plan = _paper_search_plan_from_payload(payload, question=question, queries=queries)
        return plan, {"mode": "llm", "model": model, "json_mode": True}
    except Exception as exc:
        plan = _error_plan(question)
        raw_preview = ""
        if isinstance(exc, LLMJSONError):
            raw_preview = shorten(exc.raw_text, width=1600, placeholder="...")
        return plan, {
            "mode": "llm_error_no_platform_queries",
            "model": model,
            "json_mode": True,
            "error": shorten(str(exc), width=400, placeholder="..."),
            "raw_response_preview": raw_preview,
        }


async def plan_next_paper_search_iteration(
    *,
    question: str,
    queries: list[str],
    research_plan: list[dict[str, Any]],
    previous_plan: PaperSearchPlan,
    executed_tool_queries: dict[str, list[str]],
    observation: dict[str, Any],
    reason: str,
    args: Any,
) -> tuple[PaperSearchPlan | None, dict[str, Any]]:
    """Ask the planner LLM for incremental search actions after observing results."""

    model = _paper_search_model(args)
    if not model:
        return None, {
            "mode": "skipped_no_llm",
            "reason": "PaperSearchAgent loop refinement requires an LLM.",
        }

    plan_context = "\n".join(
        f"- {item.get('perspective') or item.get('agent') or 'plan'}: "
        f"{item.get('research_question') or item.get('objective') or item.get('description') or ''}"
        for item in research_plan[:8]
    )
    prompt_payload = {
        "user_question": question,
        "research_plan": plan_context or "N/A",
        "initial_search_queries": queries[:12],
        "previous_task_signals": previous_plan.task_signals,
        "previous_authors": previous_plan.authors,
        "previous_tool_queries": previous_plan.tool_queries,
        "already_executed_tool_queries": executed_tool_queries,
        "observation": observation,
        "refinement_reason": reason,
    }
    try:
        payload = await _llm_json(
            model=model,
            args=args,
            max_tokens=1400,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are PaperSearchAgent.SearchLoopPlanner. You inspect paper-search "
                        "results and decide whether one more targeted search round is needed. "
                        "Return exactly one JSON object. Do not use markdown or comments. "
                        "Only propose NEW tool queries that were not already executed. If the "
                        "existing candidates are enough or you cannot improve the search, return "
                        "continue_search=false and an empty tool_queries object. "
                        "Keep follow-up compact: at most 2 tools and at most 2 queries per tool. "
                        "Available tools: arxiv_search, openalex_search, openreview_search, "
                        "crossref_search, web_search, author_homepage, publication_page, "
                        "google_scholar_profile, dblp_profile, openreview_profile, project_page. "
                        "Use profile/homepage tools for author identity and publication-list gaps; "
                        "use academic tools for title/topic/author paper lookup. Use Google Scholar "
                        "only as a web/profile query, not as a direct API. Avoid repeating broad "
                        "queries that already failed; prefer precise names, paper titles, venues, "
                        "years, official pages, or disambiguating institutions. Use crossref_search "
                        "only for exact title/DOI-style fallback queries."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Search state:\n"
                        f"{json.dumps(prompt_payload, ensure_ascii=False)[:9000]}\n\n"
                        "Return JSON schema:\n"
                        "{\n"
                        '  "continue_search": true,\n'
                        '  "task_signals": {"needs_author_disambiguation": true},\n'
                        '  "authors": ["..."],\n'
                        '  "content_query": "what gap this round addresses",\n'
                        '  "search_queries": ["short human-readable query"],\n'
                        '  "tool_queries": {\n'
                        '    "author_homepage": ["..."],\n'
                        '    "publication_page": ["..."],\n'
                        '    "arxiv_search": ["au:\\"...\\" AND all:\\"...\\""]\n'
                        "  },\n"
                        '  "reasoning": "why these new searches should fix the gap"\n'
                        "}"
                    ),
                },
            ],
        )
        if not bool(payload.get("continue_search", True)):
            return None, {
                "mode": "llm_stop",
                "model": model,
                "reasoning": str(payload.get("reasoning") or ""),
            }
        plan = _paper_search_plan_from_payload(payload, question=question, queries=queries)

        filtered_tool_queries: dict[str, list[str]] = {}
        for tool, proposed_queries in plan.tool_queries.items():
            executed = set(executed_tool_queries.get(tool, []))
            new_queries = [query for query in proposed_queries if query not in executed]
            if new_queries:
                filtered_tool_queries[tool] = new_queries
        if not filtered_tool_queries:
            return None, {
                "mode": "llm_no_new_queries",
                "model": model,
                "reasoning": plan.reasoning,
            }
        plan.tool_queries = filtered_tool_queries
        plan.platform_queries = _platform_queries_from_tool_queries(filtered_tool_queries)
        plan.web_queries = _web_queries_from_tool_queries(filtered_tool_queries)
        return plan, {
            "mode": "llm_continue",
            "model": model,
            "reasoning": plan.reasoning,
        }
    except Exception as exc:
        raw_preview = ""
        if isinstance(exc, LLMJSONError):
            raw_preview = shorten(exc.raw_text, width=1600, placeholder="...")
        return None, {
            "mode": "llm_error",
            "model": model,
            "error": shorten(str(exc), width=400, placeholder="..."),
            "raw_response_preview": raw_preview,
        }


def _fetch_page(url: str, *, timeout: float) -> tuple[str, str]:
    request = urllib.request.Request(url, headers={"User-Agent": DEFAULT_USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        content_type = response.headers.get("Content-Type", "")
        if "text" not in content_type and "html" not in content_type:
            return "", ""
        raw = response.read(1_500_000)
    html_text = raw.decode("utf-8", errors="replace")
    parser = _LinkParser()
    parser.feed(html_text)
    text = re.sub(r"\s+", " ", " ".join(parser.text_chunks)).strip()
    links = "\n".join(
        f"- {item['text']} | {urllib.parse.urljoin(url, item['href'])}"
        for item in parser.links[:180]
        if item.get("href")
    )
    return text[:18000], links[:18000]


def _profile_url_priority(url: str) -> int:
    parsed = urllib.parse.urlparse(url)
    host = parsed.netloc.lower()
    path = parsed.path.lower()
    if path.endswith(".pdf") or ".pdf" in path:
        return 100
    if any(marker in path for marker in ("publication", "publications", "papers")):
        return -1
    if any(marker in path for marker in ("faculty", "people", "profile", "teacher", "jiaoshi")):
        return 0
    if "openreview.net/profile" in url or "arxiv.org/a/" in url:
        return 1
    if "dblp" in host or "orcid" in host:
        return 2
    if "github.io" in host or ".edu" in host or ".ac." in host:
        return 3
    if "scholar.google" in host:
        return 4
    if "github.com" in host:
        return 7
    return 5


def _interesting_profile_links(base_url: str, links: str, limit: int = 8) -> list[str]:
    """Find likely publication/profile pages linked from an author or lab page."""

    candidates: list[str] = []
    seen: set[str] = set()
    patterns = (
        "publication",
        "publications",
        "papers",
        "selected-publications",
        "research",
        "google scholar",
        "scholar.google",
        "dblp",
        "openreview.net/profile",
        "arxiv.org/a/",
        "orcid",
    )
    for raw_line in links.splitlines():
        line = raw_line.strip()
        if "|" not in line:
            continue
        text, href = line.rsplit("|", 1)
        text = text.lstrip("- ").strip()
        href = href.strip()
        if not href or href.startswith(("mailto:", "javascript:")):
            continue
        combined = f"{text} {href}".lower()
        if not any(pattern in combined for pattern in patterns):
            continue
        url = urllib.parse.urljoin(base_url, href)
        if url.lower().endswith((".jpg", ".jpeg", ".png", ".gif", ".svg", ".zip")):
            continue
        if url not in seen:
            seen.add(url)
            candidates.append(url)
        if len(candidates) >= limit:
            break
    return sorted(candidates, key=_profile_url_priority)


def _source_result_to_search_result(
    paper: AcademicPaper,
    *,
    rank: int,
    source_label: str,
    score: float,
) -> SearchResult:
    venue = str(paper.extra.get("venue") or "")
    if not venue and paper.categories:
        venue = paper.categories[0]
    return SearchResult(
        rank=rank,
        hybrid_score=score,
        retrieval_score=score,
        metadata_score=score,
        paper_id=paper.paper_id,
        corpus="paper_search",
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
        citation_count=paper.citations,
        influential_citation_count=paper.influential_citations,
        frontier_score=0.0,
        relevance_score=score,
        decision="candidate",
        keywords=paper.keywords,
        categories=paper.categories,
        sources=["paper_search_agent", source_label, paper.source],
        quality_signals=[
            signal
            for signal, present in (
                ("pdf", bool(paper.pdf_url)),
                ("abstract", bool(paper.abstract)),
                ("doi", bool(paper.doi)),
                ("citations", bool(paper.citations)),
            )
            if present
        ],
        relevance_reasons=[f"discovered by PaperSearchAgent via {source_label}"],
        matched_terms=[],
    )


def _dedupe_results(items: list[SearchResult]) -> list[SearchResult]:
    seen: set[str] = set()
    deduped: list[SearchResult] = []
    for item in items:
        key = (
            item.doi
            or item.arxiv_id
            or item.pdf_url
            or item.paper_url
            or title_key(item.title)
        )
        key = str(key or "").lower()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    for rank, item in enumerate(deduped, start=1):
        item.rank = rank
    return deduped


def _selected_from_results(results: list[SearchResult], *, top_k: int) -> list[SelectedPaper]:
    return [
        SelectedPaper(result=item, selection_score=float(item.hybrid_score or item.relevance_score or 0.0))
        for item in results
        if item.pdf_url
    ][:top_k]


async def _llm_extract_papers_from_pages(
    *,
    question: str,
    page_payloads: list[dict[str, str]],
    model: str | None,
    args: Any,
) -> tuple[list[AcademicPaper], dict[str, Any]]:
    if not model or not page_payloads:
        return [], {"mode": "skipped_no_llm_or_pages", "extracted_count": 0}

    page_text = "\n\n".join(
        f"PAGE {idx}: {item['url']}\nTEXT:\n{item['text'][:5500]}\nLINKS:\n{item['links'][:3500]}"
        for idx, item in enumerate(page_payloads[:6], start=1)
    )
    try:
        payload = await _llm_json(
            model=model,
            args=args,
            max_tokens=3500,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You extract publication metadata from official author/profile/project pages. "
                        "Only extract real papers visible in the supplied page text/links. "
                        "For author/person questions, extract papers by the requested person or "
                        "papers listed on that person's official publication/profile page; ignore "
                        "unrelated search results, issue pages, lab pages for different people, or "
                        "same-name profiles. Prefer 2025-2026 papers when the user asks for latest "
                        "directions. If an arXiv/OpenReview/DOI/PDF link is visible, include it. "
                        "Prefer recent papers and include URLs/PDF URLs if visible. Return ONLY JSON."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"User question:\n{question}\n\n"
                        f"Fetched pages:\n{page_text}\n\n"
                        "Return JSON schema:\n"
                        "{\n"
                        '  "papers": [\n'
                        '    {"title": "...", "authors": ["..."], "year": 2026, '
                        '"venue": "...", "abstract": "", "paper_url": "...", "pdf_url": "..."}\n'
                        "  ]\n"
                        "}"
                    ),
                },
            ],
        )
        papers: list[AcademicPaper] = []
        for raw in payload.get("papers") or []:
            if not isinstance(raw, dict):
                continue
            title = str(raw.get("title") or "").strip()
            if len(title) < 6:
                continue
            pdf_url = str(raw.get("pdf_url") or "").strip()
            paper_url = str(raw.get("paper_url") or "").strip()
            arxiv_id = arxiv_id_from_url(pdf_url) or arxiv_id_from_url(paper_url)
            if arxiv_id and not pdf_url:
                pdf_url = arxiv_pdf_url(arxiv_id)
            year = str(raw.get("year") or "")
            papers.append(
                AcademicPaper(
                    paper_id=stable_id("web", arxiv_id, pdf_url, paper_url, title),
                    title=title,
                    authors=[str(a).strip() for a in raw.get("authors") or [] if str(a).strip()],
                    abstract=str(raw.get("abstract") or ""),
                    doi="",
                    published_date=parse_date(year),
                    pdf_url=pdf_url,
                    url=paper_url or pdf_url,
                    source="web_profile",
                    categories=[str(raw.get("venue") or "").strip()] if raw.get("venue") else [],
                    extra={"arxiv_id": arxiv_id, "venue": str(raw.get("venue") or ""), "source_label": "web_profile_llm"},
                )
            )
        return papers, {"mode": "llm", "model": model, "extracted_count": len(papers)}
    except Exception as exc:
        return [], {
            "mode": "error",
            "model": model,
            "extracted_count": 0,
            "error": shorten(str(exc), width=400, placeholder="..."),
        }


def _source_query_view(plan: PaperSearchPlan) -> dict[str, list[str]]:
    source_queries: dict[str, list[str]] = {}
    for source in ACADEMIC_PLATFORMS:
        values = [query for query in plan.platform_queries.get(source, []) if query.strip()]
        if values:
            source_queries[source] = values
    return source_queries


async def _web_profile_workflow(
    *,
    question: str,
    web_evidence: list[dict[str, Any]],
    planning_web_evidence: list[dict[str, Any]],
    args: Any,
) -> tuple[list[SearchResult], dict[str, Any]]:
    trace: dict[str, Any] = {
        "mode": "web_profile_llm_extract",
        "web_page_extract": {},
        "errors": [],
    }
    results: list[SearchResult] = []
    model = _paper_search_model(args)
    page_urls: list[str] = []
    for item in [*planning_web_evidence, *web_evidence]:
        url = str(item.get("url") or "")
        if not url or url in page_urls:
            continue
        page_urls.append(url)
    page_urls = sorted(page_urls, key=_profile_url_priority)

    page_payloads: list[dict[str, str]] = []
    max_pages = int(getattr(args, "paper_search_profile_pages", 8))
    pending_urls = list(page_urls)
    fetched_urls: set[str] = set()
    discovered_urls: list[str] = []
    while pending_urls and len(page_payloads) < max_pages:
        url = pending_urls.pop(0)
        if url in fetched_urls:
            continue
        fetched_urls.add(url)
        try:
            text, links = await asyncio.to_thread(
                _fetch_page,
                url,
                timeout=float(getattr(args, "web_search_timeout", 12.0)),
            )
            if text or links:
                page_payloads.append({"url": url, "text": text, "links": links})
                for linked_url in _interesting_profile_links(url, links):
                    if linked_url not in fetched_urls and linked_url not in pending_urls:
                        pending_urls.append(linked_url)
                        discovered_urls.append(linked_url)
                pending_urls.sort(key=_profile_url_priority)
        except Exception as exc:
            trace["errors"].append({"stage": "fetch_profile_page", "url": url, "error": str(exc)})

    extracted, extract_trace = await _llm_extract_papers_from_pages(
        question=question,
        page_payloads=page_payloads,
        model=model,
        args=args,
    )
    trace["web_page_extract"] = {
        **extract_trace,
        "fetched_pages": [item["url"] for item in page_payloads],
        "discovered_pages": discovered_urls,
    }
    for idx, paper in enumerate(extracted, start=1):
        results.append(
            _source_result_to_search_result(
                paper,
                rank=len(results) + 1,
                source_label="web_profile_llm",
                score=28.0 - idx * 0.1,
            )
        )

    return _dedupe_results(results), trace


async def _academic_workflow(
    *,
    question: str,
    plan: PaperSearchPlan,
    queries: list[str],
    args: Any,
) -> tuple[list[SearchResult], dict[str, Any]]:
    source_queries = _source_query_view(plan)
    if not source_queries:
        return [], {
            "mode": "skipped_no_platform_queries",
            "source_queries": {},
            "reason": (
                "PaperSearchAgent requires LLM-generated platform_queries for academic "
                "connectors; generic planner/web queries are not sent to paper platforms."
            ),
        }
    generic_queries = []
    selected, trace = await run_academic_paper_search(
        question=question,
        queries=generic_queries,
        args=args,
        source_queries=source_queries or None,
    )
    candidates = []
    min_score = float(getattr(args, "academic_paper_min_score", 2.5) or 2.5)
    for item in trace.get("candidates") or []:
        try:
            result = SearchResult(**item)
        except TypeError:
            continue
        score = float(result.hybrid_score or result.relevance_score or 0.0)
        if score > min_score:
            candidates.append(result)
    if not candidates:
        candidates = [item.result for item in selected]
    return _dedupe_results(candidates), {
        "mode": "platform_query_workflow",
        "source_queries": source_queries,
        **trace,
    }


def _called_tools_for_iteration(
    *,
    plan: PaperSearchPlan,
    web_queries: list[str],
    web_profile_trace: dict[str, Any],
    args: Any,
) -> list[str]:
    called_tools: list[str] = []
    if web_queries and not getattr(args, "disable_web_search", False):
        called_tools.extend(
            tool for tool in WEB_SEARCH_TOOLS if plan.tool_queries.get(tool)
        )
        if "web_search" not in called_tools:
            called_tools.append("web_search")
    called_tools.extend(
        tool for tool in ACADEMIC_TOOL_TO_PLATFORM if plan.tool_queries.get(tool)
    )
    if web_profile_trace.get("web_page_extract", {}).get("fetched_pages"):
        called_tools.append("web_profile_extract")
    return called_tools


def _mark_executed_tool_queries(
    executed: dict[str, list[str]],
    plan: PaperSearchPlan,
) -> None:
    for tool, tool_queries in plan.tool_queries.items():
        executed[tool] = _merge_unique(executed.get(tool, []), tool_queries)


async def _run_paper_search_iteration(
    *,
    question: str,
    plan: PaperSearchPlan,
    planning_web_evidence: list[dict[str, Any]],
    accumulated_web_evidence: list[dict[str, Any]],
    searched_web_queries: set[str],
    iteration: int,
    reason: str,
    args: Any,
) -> tuple[list[SearchResult], list[dict[str, Any]], dict[str, Any]]:
    max_web_queries = int(getattr(args, "paper_search_web_queries", 5))
    raw_web_queries = list(dict.fromkeys([
        *([question] if iteration == 1 else []),
        *plan.web_queries[:max_web_queries],
    ]))
    web_queries = [query for query in raw_web_queries if query not in searched_web_queries]
    searched_web_queries.update(web_queries)

    web_evidence: list[dict[str, Any]] = []
    web_trace: dict[str, Any] = {
        "provider": "skipped",
        "query_count": 0,
        "result_count": 0,
        "errors": [],
    }
    if web_queries and not getattr(args, "disable_web_search", False):
        emit_progress(
            args,
            "PaperSearchAgent",
            "iteration web search start",
            iteration=iteration,
            queries=len(web_queries),
        )
        web_evidence, web_trace = await run_web_search_agent(web_queries, args)
        emit_progress(
            args,
            "PaperSearchAgent",
            "iteration web search done",
            iteration=iteration,
            results=web_trace.get("result_count"),
            errors=len(web_trace.get("errors", [])),
        )

    expanded_web_evidence = _merge_web_evidence(accumulated_web_evidence, web_evidence)
    emit_progress(
        args,
        "PaperSearchAgent",
        "academic connectors start",
        iteration=iteration,
        sources=",".join(getattr(args, "academic_paper_sources", "").split(",")),
    )
    academic_results, academic_trace = await _academic_workflow(
        question=question,
        plan=plan,
        queries=[],
        args=args,
    )
    emit_progress(
        args,
        "PaperSearchAgent",
        "academic connectors done",
        iteration=iteration,
        candidates=len(academic_results),
        errors=len(academic_trace.get("errors", [])),
        mode=academic_trace.get("mode"),
    )
    emit_progress(
        args,
        "PaperSearchAgent",
        "profile/page extraction start",
        iteration=iteration,
        web_evidence=len(expanded_web_evidence),
    )
    web_results, web_profile_trace = await _web_profile_workflow(
        question=question,
        web_evidence=expanded_web_evidence,
        planning_web_evidence=planning_web_evidence,
        args=args,
    )
    emit_progress(
        args,
        "PaperSearchAgent",
        "profile/page extraction done",
        iteration=iteration,
        candidates=len(web_results),
        fetched_pages=len((web_profile_trace.get("web_page_extract") or {}).get("fetched_pages", [])),
    )
    workflow_results = _dedupe_results([*academic_results, *web_results])
    trace = {
        "iteration": iteration,
        "reason": reason,
        "tool_queries": plan.tool_queries,
        "platform_queries": plan.platform_queries,
        "web_queries": web_queries,
        "called_tools": _called_tools_for_iteration(
            plan=plan,
            web_queries=web_queries,
            web_profile_trace=web_profile_trace,
            args=args,
        ),
        "workflow_candidate_count": len(workflow_results),
        "pdf_candidate_count": len([item for item in workflow_results if item.pdf_url]),
        "web_search_trace": web_trace,
        "academic": academic_trace,
        "web_profile": web_profile_trace,
    }
    return workflow_results, web_evidence, trace


def _metadata_candidates(queries: list[str], args: Any) -> tuple[list[SearchResult], list[SelectedPaper], dict[str, Any]]:
    corpora = [SearchCorpus("topconf", args.topconf_db)]
    if args.include_frontier:
        corpora.append(SearchCorpus("frontier", args.frontier_db))
    if args.include_arxiv:
        corpora.append(SearchCorpus("arxiv", args.arxiv_db))
    filters = SearchFilters(
        min_score=args.min_score,
        year_from=args.year_from,
        year_to=args.year_to,
        venues=parse_venues(args.venues),
        require_abstract=not args.allow_missing_abstract,
        require_pdf=not args.allow_no_pdf,
    )
    with MetadataSearchEngine(corpora=corpora, filters=filters) as engine:
        groups = [engine.search(query, candidate_k=args.per_query_k) for query in queries]
    candidates = reciprocal_rank_fusion(groups, top_k=args.candidate_k)
    selected = select_for_paperqa(candidates, top_k=args.paperqa_k)
    return candidates, selected, {
        "mode": "metadata_search",
        "query_count": len(queries),
        "candidate_count": len(candidates),
        "selected_count": len(selected),
        "corpora": [corpus.name for corpus in corpora],
    }


async def run_paper_search_agent(
    *,
    question: str,
    queries: list[str],
    research_plan: list[dict[str, Any]],
    planning_web_evidence: list[dict[str, Any]],
    args: Any,
) -> tuple[list[SearchResult], list[SelectedPaper], list[dict[str, Any]], dict[str, Any]]:
    """Unified paper discovery agent.

    This is intentionally the single entry point used by ExpertResearchAgent.
    It first asks the planner LLM for tool-specific queries, then runs the
    requested academic and profile/web-page workflows. Metadata DB remains a
    supplemental source for our project-specific corpora.
    """

    emit_progress(args, "PaperSearchAgent", "query analyzer start")
    plan, analyzer_trace = await analyze_paper_search_query(
        question=question,
        queries=queries,
        research_plan=research_plan,
        args=args,
    )
    emit_progress(
        args,
        "PaperSearchAgent",
        "query analyzer done",
        mode=analyzer_trace.get("mode"),
        authors=",".join(plan.authors[:4]),
        tools=",".join(sorted(plan.tool_queries.keys())[:8]),
    )

    web_evidence: list[dict[str, Any]] = []
    workflow_results: list[SearchResult] = []
    iteration_traces: list[dict[str, Any]] = []
    reflection_traces: list[dict[str, Any]] = []
    executed_tool_queries: dict[str, list[str]] = {}
    searched_web_queries: set[str] = set()
    max_iterations = max(1, int(getattr(args, "paper_search_loop_iterations", 2)))
    if getattr(args, "disable_paper_search_loop", False):
        max_iterations = 1

    current_plan = plan
    all_plans = [plan]
    stop_reason = "not_started"
    for iteration in range(1, max_iterations + 1):
        reason = "initial_search_plan" if iteration == 1 else stop_reason
        emit_progress(
            args,
            "PaperSearchAgent",
            "iteration start",
            iteration=iteration,
            reason=reason,
        )
        iteration_results, new_web_evidence, iteration_trace = await _run_paper_search_iteration(
            question=question,
            plan=current_plan,
            planning_web_evidence=planning_web_evidence,
            accumulated_web_evidence=web_evidence,
            searched_web_queries=searched_web_queries,
            iteration=iteration,
            reason=reason,
            args=args,
        )
        workflow_results = _dedupe_results([*workflow_results, *iteration_results])
        web_evidence = _merge_web_evidence(web_evidence, new_web_evidence)
        _mark_executed_tool_queries(executed_tool_queries, current_plan)
        iteration_traces.append(iteration_trace)
        emit_progress(
            args,
            "PaperSearchAgent",
            "iteration done",
            iteration=iteration,
            candidates=len(workflow_results),
            pdf_candidates=len([item for item in workflow_results if item.pdf_url]),
        )

        should_refine, stop_reason = _should_refine_paper_search(
            candidates=workflow_results,
            iteration=iteration,
            max_iterations=max_iterations,
            args=args,
        )
        if not should_refine:
            break

        emit_progress(
            args,
            "PaperSearchAgent",
            "reflection start",
            iteration=iteration,
            reason=stop_reason,
        )
        observation = _search_loop_observation(
            candidates=workflow_results,
            web_evidence=web_evidence,
            iteration_traces=iteration_traces,
        )
        emit_progress(
            args,
            "PaperSearchAgent",
            "reflection done",
            iteration=iteration,
            mode=reflection_trace.get("mode"),
        )
        next_plan, reflection_trace = await plan_next_paper_search_iteration(
            question=question,
            queries=queries,
            research_plan=research_plan,
            previous_plan=current_plan,
            executed_tool_queries=executed_tool_queries,
            observation=observation,
            reason=stop_reason,
            args=args,
        )
        reflection_trace = {
            "after_iteration": iteration,
            "trigger": stop_reason,
            **reflection_trace,
        }
        reflection_traces.append(reflection_trace)
        if not next_plan:
            stop_reason = reflection_trace.get("mode", "no_refinement_plan")
            break
        current_plan = next_plan
        all_plans.append(current_plan)

    merged_tool_queries = _merge_tool_queries(*all_plans)
    merged_platform_queries = _platform_queries_from_tool_queries(merged_tool_queries)
    merged_web_queries = _web_queries_from_tool_queries(merged_tool_queries)
    merged_search_queries: list[str] = []
    for item in [*queries, *[query for search_plan in all_plans for query in search_plan.search_queries]]:
        merged_search_queries = _merge_unique(merged_search_queries, [item])

    provider_sequence: list[str] = []
    web_errors: list[dict[str, Any]] = []
    web_query_count = 0
    academic_source_results: dict[str, int] = {}
    academic_errors: list[dict[str, Any]] = []
    fetched_pages: list[str] = []
    discovered_pages: list[str] = []
    for item in iteration_traces:
        web_trace = item.get("web_search_trace") or {}
        web_query_count += int(web_trace.get("query_count") or 0)
        for provider in web_trace.get("provider_sequence") or []:
            if provider not in provider_sequence:
                provider_sequence.append(provider)
        web_errors.extend(web_trace.get("errors") or [])
        academic_trace = item.get("academic") or {}
        for source, count in (academic_trace.get("source_results") or {}).items():
            academic_source_results[source] = academic_source_results.get(source, 0) + int(count or 0)
        academic_errors.extend(academic_trace.get("errors") or [])
        profile_extract = (item.get("web_profile") or {}).get("web_page_extract") or {}
        fetched_pages = _merge_unique(fetched_pages, profile_extract.get("fetched_pages") or [])
        discovered_pages = _merge_unique(discovered_pages, profile_extract.get("discovered_pages") or [])

    workflow_trace = {
        "mode": "tool_search_loop",
        "max_iterations": max_iterations,
        "iteration_count": len(iteration_traces),
        "stop_reason": stop_reason,
        "called_tools": _merge_unique(
            [],
            [tool for item in iteration_traces for tool in item.get("called_tools", [])],
        ),
        "iterations": iteration_traces,
        "reflections": reflection_traces,
        "academic": {
            "mode": "loop_aggregate",
            "source_results": academic_source_results,
            "errors": academic_errors,
        },
        "web_profile": {
            "mode": "loop_aggregate",
            "web_page_extract": {
                "fetched_pages": fetched_pages,
                "discovered_pages": discovered_pages,
            },
            "errors": [
                error
                for item in iteration_traces
                for error in ((item.get("web_profile") or {}).get("errors") or [])
            ],
        },
    }
    web_search_trace = {
        "mode": "loop_aggregate",
        "provider_sequence": provider_sequence,
        "query_count": web_query_count,
        "result_count": len(web_evidence),
        "errors": web_errors,
    }

    metadata_results: list[SearchResult] = []
    metadata_selected: list[SelectedPaper] = []
    metadata_trace: dict[str, Any] = {"mode": "skipped"}
    if not getattr(args, "disable_metadata_paper_search", False):
        metadata_results, metadata_selected, metadata_trace = _metadata_candidates(
            merged_search_queries or queries,
            args,
        )

    candidates = _dedupe_results([*workflow_results, *metadata_results])
    trace = {
        "mode": "paper_search_agent",
        "query_analysis": {
            "task_signals": plan.task_signals,
            "authors": _merge_unique([], [author for search_plan in all_plans for author in search_plan.authors]),
            "content_query": plan.content_query,
            "search_queries": merged_search_queries,
            "tool_queries": merged_tool_queries,
            "platform_queries": merged_platform_queries,
            "web_queries": merged_web_queries,
            "reasoning": plan.reasoning,
            **analyzer_trace,
        },
        "web_search_trace": web_search_trace,
        "workflow_trace": workflow_trace,
        "metadata_trace": metadata_trace,
        "candidate_count": len(candidates),
        "workflow_candidate_count": len(workflow_results),
        "metadata_candidate_count": len(metadata_results),
        "selected_count": len(_selected_from_results(candidates, top_k=int(getattr(args, "paperqa_k", 8)))),
        "candidate_titles": [item.title for item in candidates[:30]],
        "candidates": [item.to_dict() for item in candidates[: int(getattr(args, "paper_triage_candidate_limit", 60))]],
    }
    selected = _selected_from_results(candidates, top_k=int(getattr(args, "paperqa_k", 8)))
    return candidates, selected, web_evidence, trace
