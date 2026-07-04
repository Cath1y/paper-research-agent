from __future__ import annotations

import asyncio
import html
import os
import re
import time
from dataclasses import asdict, dataclass
from html.parser import HTMLParser
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import requests


DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)


@dataclass(slots=True)
class WebEvidence:
    """One web search result used as lightweight external context."""

    query: str
    rank: int
    title: str
    url: str
    snippet: str
    source: str = "duckduckgo_html"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class _DuckDuckGoParser(HTMLParser):
    """Small parser for DuckDuckGo's no-JS HTML results page."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.results: list[dict[str, str]] = []
        self._in_title = False
        self._in_snippet = False
        self._title_chunks: list[str] = []
        self._snippet_chunks: list[str] = []
        self._pending_href = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = {key: value or "" for key, value in attrs}
        classes = attrs_dict.get("class", "")
        if tag == "a" and "result__a" in classes:
            self._in_title = True
            self._pending_href = attrs_dict.get("href", "")
            self._title_chunks = []
        elif "result__snippet" in classes:
            self._in_snippet = True
            self._snippet_chunks = []

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self._title_chunks.append(data)
        elif self._in_snippet:
            self._snippet_chunks.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._in_title:
            title = _clean_text(" ".join(self._title_chunks))
            url = _decode_duckduckgo_url(self._pending_href)
            if title and url:
                self.results.append({"title": title, "url": url, "snippet": ""})
            self._in_title = False
            self._title_chunks = []
            self._pending_href = ""
        elif self._in_snippet:
            snippet = _clean_text(" ".join(self._snippet_chunks))
            if snippet and self.results:
                for result in reversed(self.results):
                    if not result.get("snippet"):
                        result["snippet"] = snippet
                        break
            self._in_snippet = False
            self._snippet_chunks = []


def _clean_text(value: str) -> str:
    value = html.unescape(value)
    return re.sub(r"\s+", " ", value).strip()


def _decode_duckduckgo_url(url: str) -> str:
    if not url:
        return ""
    if url.startswith("//"):
        url = "https:" + url
    parsed = urlparse(url)
    if "duckduckgo.com" in parsed.netloc and parsed.path.startswith("/l/"):
        target = parse_qs(parsed.query).get("uddg", [""])[0]
        return unquote(target)
    return url


def _search_duckduckgo_html(
    query: str,
    *,
    max_results: int,
    timeout: float,
) -> list[WebEvidence]:
    response = requests.get(
        "https://html.duckduckgo.com/html/",
        params={"q": query},
        headers={"User-Agent": DEFAULT_USER_AGENT},
        timeout=timeout,
    )
    response.raise_for_status()

    parser = _DuckDuckGoParser()
    parser.feed(response.text)

    results: list[WebEvidence] = []
    seen_urls: set[str] = set()
    for item in parser.results:
        url = item.get("url", "")
        if not url or url in seen_urls or "duckduckgo.com" in urlparse(url).netloc:
            continue
        seen_urls.add(url)
        results.append(
            WebEvidence(
                query=query,
                rank=len(results) + 1,
                title=item.get("title", ""),
                url=url,
                snippet=item.get("snippet", ""),
            )
        )
        if len(results) >= max_results:
            break
    return results


def _search_tavily(
    query: str,
    *,
    api_key: str,
    max_results: int,
    timeout: float,
    search_depth: str,
) -> list[WebEvidence]:
    response = requests.post(
        "https://api.tavily.com/search",
        json={
            "api_key": api_key,
            "query": query,
            "search_depth": search_depth,
            "max_results": max_results,
            "include_answer": False,
            "include_raw_content": False,
            "include_images": False,
        },
        headers={
            "Content-Type": "application/json",
            "User-Agent": DEFAULT_USER_AGENT,
        },
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    raw_results = payload.get("results") or []

    results: list[WebEvidence] = []
    seen_urls: set[str] = set()
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url") or "").strip()
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        title = str(item.get("title") or url).strip()
        snippet = str(
            item.get("content")
            or item.get("snippet")
            or item.get("raw_content")
            or ""
        ).strip()
        results.append(
            WebEvidence(
                query=query,
                rank=len(results) + 1,
                title=_clean_text(title),
                url=url,
                snippet=_clean_text(snippet),
                source="tavily",
            )
        )
        if len(results) >= max_results:
            break
    return results


def _score_web_result(result: WebEvidence) -> float:
    """Prefer project/code/paper pages without making this a hard filter."""

    text = f"{result.title} {result.url} {result.snippet}".lower()
    score = 1.0 / max(result.rank, 1)
    boosts = {
        "arxiv.org": 0.35,
        "openreview.net": 0.3,
        "github.com": 0.28,
        "huggingface.co": 0.25,
        "project": 0.12,
        "benchmark": 0.1,
        "dataset": 0.08,
        "leaderboard": 0.08,
        "robot": 0.06,
        "vla": 0.06,
    }
    for token, boost in boosts.items():
        if token in text:
            score += boost
    return score


def search_web(
    queries: list[str],
    *,
    max_queries: int,
    results_per_query: int,
    timeout: float,
    request_delay: float,
    provider: str,
    tavily_api_key: str | None,
    tavily_search_depth: str,
) -> tuple[list[WebEvidence], dict[str, Any]]:
    """Run lightweight web search for the first few planned queries.

    This intentionally returns snippets and URLs only. Full webpage fetching can
    be added later as a separate evidence-expansion step, but snippets are enough
    for the current literature-routing demo to discover project/code/freshness
    signals without slowing down PaperQA.
    """

    web_evidence: list[WebEvidence] = []
    errors: list[dict[str, str]] = []
    provider_sequence: list[str] = []
    seen_urls: set[str] = set()
    searched_queries = [query for query in queries if query.strip()][:max_queries]

    for index, query in enumerate(searched_queries):
        if index and request_delay > 0:
            time.sleep(request_delay)

        search_order: list[str]
        if provider == "tavily":
            search_order = ["tavily"]
        elif provider == "duckduckgo":
            search_order = ["duckduckgo"]
        else:
            search_order = ["tavily", "duckduckgo"] if tavily_api_key else ["duckduckgo"]

        results: list[WebEvidence] = []
        for current_provider in search_order:
            if current_provider not in provider_sequence:
                provider_sequence.append(current_provider)
            try:
                if current_provider == "tavily":
                    if not tavily_api_key:
                        raise RuntimeError("Tavily API key is not configured.")
                    results = _search_tavily(
                        query,
                        api_key=tavily_api_key,
                        max_results=results_per_query,
                        timeout=timeout,
                        search_depth=tavily_search_depth,
                    )
                else:
                    results = _search_duckduckgo_html(
                        query,
                        max_results=results_per_query,
                        timeout=timeout,
                    )
            except (requests.RequestException, RuntimeError, ValueError) as exc:
                errors.append(
                    {
                        "provider": current_provider,
                        "query": query,
                        "error": str(exc),
                    }
                )
                continue
            if results or provider != "auto":
                break

        for result in results:
            if result.url in seen_urls:
                continue
            seen_urls.add(result.url)
            web_evidence.append(result)

    web_evidence.sort(key=_score_web_result, reverse=True)
    trace = {
        "provider": provider,
        "provider_sequence": provider_sequence,
        "query_count": len(searched_queries),
        "result_count": len(web_evidence),
        "errors": errors,
    }
    return web_evidence, trace


async def run_web_search_agent(queries: list[str], args: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Async wrapper so LangGraph can run the blocking web search safely."""

    if getattr(args, "disable_web_search", False):
        return [], {
            "provider": "disabled",
            "query_count": 0,
            "result_count": 0,
            "errors": [],
        }

    evidence, trace = await asyncio.to_thread(
        search_web,
        queries,
        max_queries=int(getattr(args, "web_max_queries", 4)),
        results_per_query=int(getattr(args, "web_results_per_query", 4)),
        timeout=float(getattr(args, "web_search_timeout", 12.0)),
        request_delay=float(getattr(args, "web_search_delay", 0.4)),
        provider=str(getattr(args, "web_search_provider", "auto")),
        tavily_api_key=os.environ.get(
            str(getattr(args, "tavily_api_key_env", "TAVILY_API_KEY"))
        ),
        tavily_search_depth=str(getattr(args, "tavily_search_depth", "basic")),
    )
    return [item.to_dict() for item in evidence], trace
