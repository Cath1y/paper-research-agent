from __future__ import annotations

import urllib.parse

from embodiedai_kb.paper_search.models import AcademicPaper

from .base import PaperSource
from .common import env_first, extract_doi, is_probable_pdf_url, open_json, parse_date, stable_id


def _date_parts_to_date(parts: list[list[int]] | None) -> str:
    if not parts or not parts[0]:
        return ""
    values = parts[0]
    if len(values) >= 3:
        return f"{values[0]:04d}-{values[1]:02d}-{values[2]:02d}"
    if len(values) == 2:
        return f"{values[0]:04d}-{values[1]:02d}"
    return f"{values[0]:04d}"


def _author_name(author: dict) -> str:
    given = str(author.get("given") or "").strip()
    family = str(author.get("family") or "").strip()
    return f"{given} {family}".strip()


class CrossrefSearcher(PaperSource):
    source_name = "crossref"
    base_url = "https://api.crossref.org/works"

    def search(self, query: str, max_results: int = 10, **_: object) -> list[AcademicPaper]:
        params = {
            "query.bibliographic": query,
            "rows": max(1, min(max_results, 50)),
            "sort": "relevance",
            "order": "desc",
        }
        mailto = env_first("PAPER_SEARCH_MCP_CROSSREF_MAILTO", "CROSSREF_MAILTO")
        if mailto:
            params["mailto"] = mailto
        payload = open_json(f"{self.base_url}?{urllib.parse.urlencode(params)}", retries=2)
        papers: list[AcademicPaper] = []
        for item in (payload.get("message") or {}).get("items", []) or []:
            paper = self._parse_item(item)
            if paper:
                papers.append(paper)
        return papers

    def _parse_item(self, item: dict) -> AcademicPaper | None:
        title_values = item.get("title") or []
        title = str(title_values[0]).strip() if title_values else ""
        if not title:
            return None
        doi = str(item.get("DOI") or extract_doi(item.get("URL") or "")).strip()
        link_pdf = ""
        for link in item.get("link") or []:
            if not isinstance(link, dict):
                continue
            url = str(link.get("URL") or "")
            content_type = str(link.get("content-type") or "")
            lowered_url = url.lower()
            if url and ("pdf" in content_type.lower() or lowered_url.endswith(".pdf")) and is_probable_pdf_url(url):
                link_pdf = url
                break
        published = (
            _date_parts_to_date((item.get("published-print") or {}).get("date-parts"))
            or _date_parts_to_date((item.get("published-online") or {}).get("date-parts"))
            or _date_parts_to_date((item.get("created") or {}).get("date-parts"))
        )
        container = item.get("container-title") or []
        venue = str(container[0]).strip() if container else ""
        return AcademicPaper(
            paper_id=stable_id("crossref", doi, item.get("URL"), title),
            title=title,
            authors=[
                name for author in item.get("author") or [] if (name := _author_name(author))
            ],
            abstract=str(item.get("abstract") or ""),
            doi=doi,
            published_date=parse_date(published),
            pdf_url=link_pdf,
            url=item.get("URL") or (f"https://doi.org/{doi}" if doi else ""),
            source=self.source_name,
            categories=[venue] if venue else [],
            citations=int(item.get("is-referenced-by-count") or 0),
            extra={"venue": venue, "type": item.get("type")},
        )
