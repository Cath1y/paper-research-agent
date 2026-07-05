from __future__ import annotations

import urllib.parse

from embodiedai_kb.paper_search.models import AcademicPaper

from .base import PaperSource
from .common import (
    arxiv_id_from_url,
    doi_from_url,
    env_first,
    is_probable_pdf_url,
    open_json,
    parse_date,
    stable_id,
)


def _abstract_from_inverted_index(index: dict[str, list[int]] | None) -> str:
    if not index:
        return ""
    words: list[tuple[int, str]] = []
    for word, positions in index.items():
        for position in positions:
            words.append((int(position), word))
    return " ".join(word for _, word in sorted(words))


def _best_pdf_url(item: dict) -> str:
    primary = item.get("primary_location") or {}
    if primary.get("pdf_url") and is_probable_pdf_url(primary.get("pdf_url")):
        return primary["pdf_url"]
    for location in item.get("locations") or []:
        if (
            isinstance(location, dict)
            and location.get("pdf_url")
            and is_probable_pdf_url(location.get("pdf_url"))
        ):
            return location["pdf_url"]
    return ""


class OpenAlexSearcher(PaperSource):
    source_name = "openalex"
    base_url = "https://api.openalex.org/works"

    def search(
        self,
        query: str,
        max_results: int = 10,
        timeout: float = 30.0,
        retries: int = 2,
        request_delay: float = 1.0,
        **_: object,
    ) -> list[AcademicPaper]:
        params = {
            "search": query,
            "per-page": max(1, min(max_results, 200)),
            "sort": "relevance_score:desc",
            "filter": "type:article|preprint",
        }
        mailto = env_first("PAPER_SEARCH_MCP_OPENALEX_MAILTO", "OPENALEX_MAILTO")
        if mailto:
            params["mailto"] = mailto
        payload = open_json(
            f"{self.base_url}?{urllib.parse.urlencode(params)}",
            timeout=timeout,
            retries=retries,
            delay=request_delay,
        )
        return [paper for item in payload.get("results", []) if (paper := self._parse_item(item))]

    def lookup_doi(
        self,
        doi: str,
        *,
        timeout: float = 30.0,
        retries: int = 1,
        request_delay: float = 1.0,
    ) -> AcademicPaper | None:
        clean = doi.strip()
        if not clean:
            return None
        params = {"filter": f"doi:{clean}", "per-page": 1}
        mailto = env_first("PAPER_SEARCH_MCP_OPENALEX_MAILTO", "OPENALEX_MAILTO")
        if mailto:
            params["mailto"] = mailto
        payload = open_json(
            f"{self.base_url}?{urllib.parse.urlencode(params)}",
            timeout=timeout,
            retries=retries,
            delay=request_delay,
        )
        for item in payload.get("results", []):
            paper = self._parse_item(item)
            if paper:
                return paper
        return None

    def _parse_item(self, item: dict) -> AcademicPaper | None:
        title = item.get("title") or item.get("display_name") or ""
        if not title:
            return None
        doi = doi_from_url(item.get("doi"))
        primary = item.get("primary_location") or {}
        landing_url = primary.get("landing_page_url") or item.get("id") or ""
        pdf_url = _best_pdf_url(item)
        arxiv_id = ""
        for location in item.get("locations") or []:
            if not isinstance(location, dict):
                continue
            arxiv_id = arxiv_id or arxiv_id_from_url(location.get("landing_page_url"))
            arxiv_id = arxiv_id or arxiv_id_from_url(location.get("pdf_url"))
        source = primary.get("source") or {}
        venue = source.get("display_name") if isinstance(source, dict) else ""
        concepts = [
            concept.get("display_name", "")
            for concept in item.get("concepts") or []
            if isinstance(concept, dict) and concept.get("display_name")
        ]
        topics = [
            topic.get("display_name", "")
            for topic in item.get("topics") or []
            if isinstance(topic, dict) and topic.get("display_name")
        ]
        return AcademicPaper(
            paper_id=stable_id("openalex", item.get("id"), arxiv_id, doi, title),
            title=title,
            authors=[
                authorship.get("author", {}).get("display_name", "").strip()
                for authorship in item.get("authorships") or []
                if isinstance(authorship, dict)
                and authorship.get("author", {}).get("display_name")
            ],
            abstract=_abstract_from_inverted_index(item.get("abstract_inverted_index")),
            doi=doi,
            published_date=parse_date(item.get("publication_date") or str(item.get("publication_year") or "")),
            pdf_url=pdf_url,
            url=landing_url,
            source=self.source_name,
            categories=sorted(set(concepts + topics)),
            citations=int(item.get("cited_by_count") or 0),
            extra={
                "openalex_id": item.get("id"),
                "arxiv_id": arxiv_id,
                "venue": venue,
                "is_oa": (item.get("open_access") or {}).get("is_oa"),
            },
        )
