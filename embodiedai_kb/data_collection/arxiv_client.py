from __future__ import annotations

import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime
from hashlib import sha1
from http.client import RemoteDisconnected
from ssl import SSLError
from urllib.error import URLError

from embodiedai_kb.storage.database import normalize_title
from embodiedai_kb.storage.schemas import PaperMetadata


ARXIV_API_URL = "https://export.arxiv.org/api/query"
ATOM_NS = {
    "a": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}


def _text(entry: ET.Element, path: str) -> str | None:
    value = entry.find(path, ATOM_NS)
    if value is None or value.text is None:
        return None
    return " ".join(value.text.split())


def _arxiv_id_from_url(url: str) -> str:
    return url.rstrip("/").split("/abs/")[-1]


def _paper_id(arxiv_id: str | None, title: str) -> str:
    if arxiv_id:
        return f"arxiv:{arxiv_id}"
    digest = sha1(normalize_title(title).encode("utf-8")).hexdigest()[:16]
    return f"title:{digest}"


class ArxivClient:
    def __init__(self, request_delay: float = 3.1, max_retries: int = 4) -> None:
        self.request_delay = request_delay
        self.max_retries = max_retries
        self._last_request = 0.0

    def search(
        self,
        query: str,
        limit: int = 50,
        date_from: str | None = None,
        date_to: str | None = None,
        sort_by: str = "relevance",
        sort_order: str = "descending",
        page_size: int = 100,
    ) -> list[PaperMetadata]:
        papers: list[PaperMetadata] = []
        page_size = max(1, min(page_size, 300))
        search_query = self._build_search_query(query, date_from, date_to)
        for start in range(0, limit, page_size):
            batch_size = min(page_size, limit - start)
            self._sleep_if_needed()
            params = {
                "search_query": search_query,
                "start": start,
                "max_results": batch_size,
                "sortBy": sort_by,
                "sortOrder": sort_order,
            }
            url = f"{ARXIV_API_URL}?{urllib.parse.urlencode(params)}"
            payload = self._open_with_retries(url)
            root = ET.fromstring(payload)
            entries = root.findall("a:entry", ATOM_NS)
            if not entries:
                break
            for entry in entries:
                paper = self._entry_to_metadata(
                    entry,
                    query=query,
                    date_from=date_from,
                    date_to=date_to,
                    search_query=search_query,
                )
                if paper is not None:
                    papers.append(paper)
            if len(entries) < batch_size:
                break
        return papers

    @staticmethod
    def _build_search_query(
        query: str, date_from: str | None, date_to: str | None
    ) -> str:
        search_query = f"all:{query}"
        if date_from or date_to:
            start = _compact_arxiv_date(date_from or "1900-01-01")
            end = _compact_arxiv_date(date_to or "2999-12-31", end_of_day=True)
            search_query = f"({search_query}) AND submittedDate:[{start} TO {end}]"
        return search_query

    @staticmethod
    def _entry_to_metadata(
        entry: ET.Element,
        query: str,
        date_from: str | None,
        date_to: str | None,
        search_query: str,
    ) -> PaperMetadata | None:
        title = _text(entry, "a:title") or ""
        if not title:
            return None
        abstract = _text(entry, "a:summary")
        arxiv_url = _text(entry, "a:id")
        arxiv_id = _arxiv_id_from_url(arxiv_url) if arxiv_url else None
        published = _text(entry, "a:published")
        year = None
        if published:
            try:
                year = datetime.fromisoformat(published.replace("Z", "+00:00")).year
            except ValueError:
                year = int(published[:4]) if published[:4].isdigit() else None
        authors = [
            name.text.strip()
            for author in entry.findall("a:author", ATOM_NS)
            if (name := author.find("a:name", ATOM_NS)) is not None and name.text
        ]
        categories = [
            category.attrib["term"]
            for category in entry.findall("a:category", ATOM_NS)
            if category.attrib.get("term")
        ]
        doi = _text(entry, "arxiv:doi")
        pdf_url = None
        for link in entry.findall("a:link", ATOM_NS):
            if (
                link.attrib.get("title") == "pdf"
                or link.attrib.get("type") == "application/pdf"
            ):
                pdf_url = link.attrib.get("href")
                break
        if arxiv_id and not pdf_url:
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"
        return PaperMetadata(
            paper_id=_paper_id(arxiv_id, title),
            title=title,
            authors=authors,
            year=year,
            venue="arXiv",
            abstract=abstract,
            paper_url=arxiv_url,
            pdf_url=pdf_url,
            doi=doi,
            arxiv_id=arxiv_id,
            fields_of_study=categories,
            sources=["arxiv"],
            source_queries=[query],
            raw_metadata={
                "source": "arxiv",
                "query": query,
                "search_query": search_query,
                "date_from": date_from,
                "date_to": date_to,
                "categories": categories,
                "published": published,
            },
        )

    def _sleep_if_needed(self) -> None:
        elapsed = time.monotonic() - self._last_request
        if elapsed < self.request_delay:
            time.sleep(self.request_delay - elapsed)
        self._last_request = time.monotonic()

    def _open_with_retries(self, url: str) -> bytes:
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            self._sleep_if_needed()
            try:
                request = urllib.request.Request(url)
                request.add_header("User-Agent", "EmbodiedAI-KB/0.1 (metadata collection)")
                with urllib.request.urlopen(request, timeout=60) as response:
                    return response.read()
            except urllib.error.HTTPError as exc:
                last_error = exc
                if exc.code != 429 or attempt >= self.max_retries:
                    raise
                retry_after = exc.headers.get("Retry-After")
                try:
                    delay = float(retry_after) if retry_after else 20.0 * (attempt + 1)
                except ValueError:
                    delay = 20.0 * (attempt + 1)
                time.sleep(delay)
            except (URLError, RemoteDisconnected, SSLError) as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    raise
                time.sleep(10.0 * (attempt + 1))
        if last_error is not None:
            raise last_error
        raise RuntimeError("arXiv request failed without an exception.")


def _compact_arxiv_date(value: str, end_of_day: bool = False) -> str:
    value = value.strip()
    if len(value) == 8 and value.isdigit():
        suffix = "2359" if end_of_day else "0000"
        return f"{value}{suffix}"
    if len(value) >= 10 and value[4] == "-" and value[7] == "-":
        suffix = "2359" if end_of_day else "0000"
        return value[:10].replace("-", "") + suffix
    if len(value) == 12 and value.isdigit():
        return value
    raise ValueError(
        f"Expected arXiv date as YYYY-MM-DD, YYYYMMDD, or YYYYMMDDHHMM, got {value!r}."
    )
