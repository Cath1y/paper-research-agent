from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from hashlib import sha1
from typing import Any

from embodiedai_kb.storage.database import normalize_title
from embodiedai_kb.storage.schemas import PaperMetadata


S2_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
S2_FIELDS = ",".join(
    [
        "paperId",
        "title",
        "abstract",
        "year",
        "authors",
        "venue",
        "publicationVenue",
        "externalIds",
        "openAccessPdf",
        "url",
        "citationCount",
        "influentialCitationCount",
        "fieldsOfStudy",
        "s2FieldsOfStudy",
        "publicationTypes",
        "publicationDate",
    ]
)


def _paper_id(s2_id: str | None, arxiv_id: str | None, doi: str | None, title: str) -> str:
    if s2_id:
        return f"s2:{s2_id}"
    if arxiv_id:
        return f"arxiv:{arxiv_id}"
    if doi:
        return f"doi:{doi.lower()}"
    digest = sha1(normalize_title(title).encode("utf-8")).hexdigest()[:16]
    return f"title:{digest}"


def _clean_external_id(value: Any) -> str | None:
    if value is None:
        return None
    value = str(value).strip()
    return value or None


class SemanticScholarClient:
    def __init__(self, request_delay: float = 2.0, max_retries: int = 3) -> None:
        self.request_delay = request_delay
        self.max_retries = max_retries
        self._last_request = 0.0
        self.api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")

    def search(self, query: str, limit: int = 50) -> list[PaperMetadata]:
        self._sleep_if_needed()
        params = {
            "query": query,
            "limit": min(limit, 100),
            "fields": S2_FIELDS,
        }
        url = f"{S2_SEARCH_URL}?{urllib.parse.urlencode(params)}"
        request = urllib.request.Request(url)
        request.add_header("User-Agent", "EmbodiedAI-KB/0.1 (metadata collection)")
        if self.api_key:
            request.add_header("x-api-key", self.api_key)
        payload = self._open_json_with_retries(request)
        papers: list[PaperMetadata] = []
        for item in payload.get("data", []):
            title = item.get("title") or ""
            if not title:
                continue
            external_ids = item.get("externalIds") or {}
            arxiv_id = _clean_external_id(external_ids.get("ArXiv"))
            doi = _clean_external_id(external_ids.get("DOI"))
            s2_id = _clean_external_id(item.get("paperId"))
            open_pdf = item.get("openAccessPdf") or {}
            pdf_url = open_pdf.get("url")
            venue = item.get("venue")
            publication_venue = item.get("publicationVenue") or {}
            if not venue and publication_venue:
                venue = publication_venue.get("name")
            fields = [str(v) for v in item.get("fieldsOfStudy") or [] if v]
            for field in item.get("s2FieldsOfStudy") or []:
                category = field.get("category")
                if category:
                    fields.append(str(category))
            papers.append(
                PaperMetadata(
                    paper_id=_paper_id(s2_id, arxiv_id, doi, title),
                    title=title,
                    authors=[
                        author.get("name", "").strip()
                        for author in item.get("authors") or []
                        if author.get("name")
                    ],
                    year=item.get("year"),
                    venue=venue,
                    abstract=item.get("abstract"),
                    paper_url=item.get("url"),
                    pdf_url=pdf_url,
                    doi=doi,
                    arxiv_id=arxiv_id,
                    semantic_scholar_id=s2_id,
                    citation_count=item.get("citationCount"),
                    influential_citation_count=item.get("influentialCitationCount"),
                    fields_of_study=sorted(set(fields)),
                    sources=["semantic_scholar"],
                    source_queries=[query],
                    raw_metadata={"source": "semantic_scholar", "query": query, "item": item},
                )
            )
        return papers

    def _open_json_with_retries(self, request: urllib.request.Request) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            self._sleep_if_needed()
            try:
                with urllib.request.urlopen(request, timeout=45) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                last_error = exc
                if exc.code != 429 or attempt >= self.max_retries:
                    raise
                retry_after = exc.headers.get("Retry-After")
                try:
                    delay = float(retry_after) if retry_after else 5.0 * (attempt + 1)
                except ValueError:
                    delay = 5.0 * (attempt + 1)
                time.sleep(delay)
        if last_error is not None:
            raise last_error
        raise RuntimeError("Semantic Scholar request failed without an exception.")

    def _sleep_if_needed(self) -> None:
        elapsed = time.monotonic() - self._last_request
        if elapsed < self.request_delay:
            time.sleep(self.request_delay - elapsed)
        self._last_request = time.monotonic()
