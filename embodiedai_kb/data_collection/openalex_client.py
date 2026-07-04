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


OPENALEX_WORKS_URL = "https://api.openalex.org/works"


def _paper_id(openalex_id: str | None, arxiv_id: str | None, doi: str | None, title: str) -> str:
    if openalex_id:
        return f"openalex:{openalex_id.rstrip('/').split('/')[-1]}"
    if arxiv_id:
        return f"arxiv:{arxiv_id}"
    if doi:
        return f"doi:{doi.lower()}"
    digest = sha1(normalize_title(title).encode("utf-8")).hexdigest()[:16]
    return f"title:{digest}"


def _doi(value: str | None) -> str | None:
    if not value:
        return None
    value = value.strip()
    return value.removeprefix("https://doi.org/") or None


def _arxiv_id_from_locations(item: dict[str, Any]) -> str | None:
    for location in item.get("locations") or []:
        source_url = (location or {}).get("landing_page_url") or ""
        pdf_url = (location or {}).get("pdf_url") or ""
        for url in (source_url, pdf_url):
            if "arxiv.org/abs/" in url:
                return url.rstrip("/").split("/abs/")[-1]
            if "arxiv.org/pdf/" in url:
                return url.rstrip("/").split("/pdf/")[-1].removesuffix(".pdf")
    return None


def _abstract_from_inverted_index(index: dict[str, list[int]] | None) -> str | None:
    if not index:
        return None
    words: list[tuple[int, str]] = []
    for word, positions in index.items():
        for position in positions:
            words.append((int(position), word))
    return " ".join(word for _, word in sorted(words))


class OpenAlexClient:
    def __init__(self, request_delay: float = 1.0, max_retries: int = 3) -> None:
        self.request_delay = request_delay
        self.max_retries = max_retries
        self._last_request = 0.0
        self.mailto = os.getenv("OPENALEX_MAILTO")

    def search(self, query: str, limit: int = 50) -> list[PaperMetadata]:
        self._sleep_if_needed()
        params = {
            "search": query,
            "per-page": min(limit, 200),
            "sort": "relevance_score:desc",
            "filter": "from_publication_date:2019-01-01,type:article|preprint",
        }
        if self.mailto:
            params["mailto"] = self.mailto
        url = f"{OPENALEX_WORKS_URL}?{urllib.parse.urlencode(params)}"
        request = urllib.request.Request(url)
        request.add_header("User-Agent", "EmbodiedAI-KB/0.1 (metadata collection)")
        payload = self._open_json_with_retries(request)

        papers: list[PaperMetadata] = []
        for item in payload.get("results", []):
            title = item.get("title") or item.get("display_name") or ""
            if not title:
                continue
            openalex_id = item.get("id")
            doi = _doi(item.get("doi"))
            arxiv_id = _arxiv_id_from_locations(item)
            primary_location = item.get("primary_location") or {}
            source = primary_location.get("source") or {}
            pdf_url = primary_location.get("pdf_url")
            if not pdf_url:
                for location in item.get("locations") or []:
                    if location.get("pdf_url"):
                        pdf_url = location.get("pdf_url")
                        break
            paper_url = (
                primary_location.get("landing_page_url")
                or item.get("ids", {}).get("openalex")
                or openalex_id
            )
            venue = source.get("display_name")
            concepts = [
                concept.get("display_name", "")
                for concept in item.get("concepts") or []
                if concept.get("display_name")
            ]
            topics = [
                topic.get("display_name", "")
                for topic in item.get("topics") or []
                if topic.get("display_name")
            ]
            papers.append(
                PaperMetadata(
                    paper_id=_paper_id(openalex_id, arxiv_id, doi, title),
                    title=title,
                    authors=[
                        authorship.get("author", {}).get("display_name", "").strip()
                        for authorship in item.get("authorships") or []
                        if authorship.get("author", {}).get("display_name")
                    ],
                    year=item.get("publication_year"),
                    venue=venue,
                    abstract=_abstract_from_inverted_index(
                        item.get("abstract_inverted_index")
                    ),
                    paper_url=paper_url,
                    pdf_url=pdf_url,
                    doi=doi,
                    arxiv_id=arxiv_id,
                    citation_count=item.get("cited_by_count"),
                    fields_of_study=sorted(set(concepts + topics)),
                    sources=["openalex"],
                    source_queries=[query],
                    raw_metadata={"source": "openalex", "query": query, "item": item},
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
        raise RuntimeError("OpenAlex request failed without an exception.")

    def _sleep_if_needed(self) -> None:
        elapsed = time.monotonic() - self._last_request
        if elapsed < self.request_delay:
            time.sleep(self.request_delay - elapsed)
        self._last_request = time.monotonic()
