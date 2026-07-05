from __future__ import annotations

import urllib.parse
from datetime import datetime

from embodiedai_kb.paper_search.models import AcademicPaper

from .base import PaperSource
from .common import content_value, extract_doi, open_json, parse_date, stable_id


class OpenReviewSearcher(PaperSource):
    source_name = "openreview"
    base_url = "https://api2.openreview.net/notes/search"

    def search(
        self,
        query: str,
        max_results: int = 10,
        timeout: float = 30.0,
        retries: int = 2,
        request_delay: float = 1.0,
        **_: object,
    ) -> list[AcademicPaper]:
        params = {"term": query, "limit": max(1, min(max_results, 100))}
        payload = open_json(
            f"{self.base_url}?{urllib.parse.urlencode(params)}",
            timeout=timeout,
            retries=retries,
            delay=request_delay,
        )
        papers: list[AcademicPaper] = []
        for note in payload.get("notes", []) or []:
            paper = self._parse_note(note)
            if paper:
                papers.append(paper)
        return papers

    def _parse_note(self, note: dict) -> AcademicPaper | None:
        content = note.get("content") or {}
        title = str(content_value(content, "title") or "").strip()
        if not title:
            return None
        authors_raw = content_value(content, "authors") or []
        authors = [str(item).strip() for item in authors_raw if str(item).strip()] if isinstance(authors_raw, list) else []
        abstract = str(content_value(content, "abstract") or "").strip()
        venue = str(content_value(content, "venue") or content_value(content, "venueid") or "").strip()
        keywords_raw = content_value(content, "keywords") or []
        keywords = [str(item).strip() for item in keywords_raw if str(item).strip()] if isinstance(keywords_raw, list) else []
        note_id = str(note.get("id") or "").strip()
        cdate = note.get("pdate") or note.get("cdate")
        published_date = None
        if isinstance(cdate, (int, float)) and cdate > 0:
            published_date = parse_date(str(datetime.fromtimestamp(cdate / 1000).date()))
        doi = extract_doi(" ".join([title, abstract, str(content_value(content, "pdf") or "")]))
        paper_url = f"https://openreview.net/forum?id={note_id}" if note_id else ""
        pdf_url = f"https://openreview.net/pdf?id={note_id}" if note_id else ""
        return AcademicPaper(
            paper_id=stable_id("openreview", note_id, doi, title),
            title=title,
            authors=authors,
            abstract=abstract,
            doi=doi,
            published_date=published_date,
            pdf_url=pdf_url,
            url=paper_url,
            source=self.source_name,
            categories=[venue] if venue else [],
            keywords=keywords,
            extra={"openreview_id": note_id, "venue": venue},
        )
