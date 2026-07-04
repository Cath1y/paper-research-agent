from __future__ import annotations

import html
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import gzip
import json
import xml.etree.ElementTree as ET
from hashlib import sha1
from typing import Iterable

from embodiedai_kb.data_collection.arxiv_client import (
    ATOM_NS,
    _arxiv_id_from_url,
    _text,
)
from embodiedai_kb.storage.database import normalize_title
from embodiedai_kb.storage.schemas import PaperMetadata


USER_AGENT = "EmbodiedAI-KB/0.1 (top conference metadata collection)"
DEFAULT_TIMEOUT = 30


def _get(url: str, timeout: int = DEFAULT_TIMEOUT, max_retries: int = 3) -> str:
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = response.read()
                if response.headers.get("Content-Encoding") == "gzip" or payload.startswith(
                    b"\x1f\x8b"
                ):
                    payload = gzip.decompress(payload)
                return payload.decode("utf-8", "ignore")
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code not in {429, 503} or attempt >= max_retries:
                raise
            retry_after = exc.headers.get("Retry-After")
            try:
                delay = float(retry_after) if retry_after else 10.0 * (attempt + 1)
            except ValueError:
                delay = 10.0 * (attempt + 1)
            time.sleep(delay)
        except urllib.error.URLError as exc:
            last_error = exc
            if attempt >= max_retries:
                raise
            time.sleep(5.0 * (attempt + 1))
    if last_error:
        raise last_error
    raise RuntimeError(f"GET failed without an exception: {url}")


def _clean_text(value: str) -> str:
    value = re.sub(r"<[^>]+>", " ", value)
    value = html.unescape(value)
    return re.sub(r"\s+", " ", value).strip()


def _split_authors(value: str) -> list[str]:
    value = _clean_text(value).replace("\xa0", " ")
    parts = re.split(r"\s*,\s*|\s+;\s+", value)
    return [part.strip().strip("*") for part in parts if part.strip()]


def _stable_id(prefix: str, *parts: str) -> str:
    body = "|".join(normalize_title(part) for part in parts if part)
    digest = sha1(body.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}:{digest}"


def _strip_arxiv_version(arxiv_id: str | None) -> str | None:
    if not arxiv_id:
        return None
    return re.sub(r"v\d+$", "", arxiv_id.strip())


def _normalize_doi(doi: str | None) -> str | None:
    if not doi:
        return None
    doi = doi.strip()
    doi = re.sub(r"^https?://(dx\.)?doi\.org/", "", doi, flags=re.I)
    return doi.lower() or None


def _abstract_from_openalex_index(index: dict[str, list[int]] | None) -> str | None:
    if not index:
        return None
    words: list[tuple[int, str]] = []
    for word, positions in index.items():
        for position in positions:
            words.append((int(position), word))
    abstract = " ".join(word for _, word in sorted(words))
    return abstract if len(abstract) >= 80 else None


def _content_value(value: object) -> object:
    if isinstance(value, dict) and "value" in value:
        return value["value"]
    return value


def _content_text(content: dict[str, object], key: str) -> str | None:
    value = _content_value(content.get(key))
    if value is None:
        return None
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    return str(value)


def _content_list(content: dict[str, object], key: str) -> list[str]:
    value = _content_value(content.get(key))
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return _split_authors(str(value))


class CVFClient:
    base_url = "https://openaccess.thecvf.com"

    def __init__(self, request_delay: float = 1.0) -> None:
        self.request_delay = request_delay
        self._last_request = 0.0

    def collect(self, years: Iterable[int]) -> list[PaperMetadata]:
        papers: list[PaperMetadata] = []
        for year in years:
            for conf in self._confs_for_year(year):
                try:
                    papers.extend(self.collect_conference(conf, year))
                except urllib.error.HTTPError as exc:
                    if exc.code != 404:
                        raise
                except urllib.error.URLError:
                    continue
        return papers

    def collect_conference(self, conf: str, year: int) -> list[PaperMetadata]:
        venue_key = f"{conf}{year}"
        url = f"{self.base_url}/{venue_key}?day=all"
        self._sleep()
        page = _get(url)
        blocks = re.findall(
            r'<dt class="ptitle"><br><a href="([^"]+)">(.*?)</a></dt>\s*'
            r"<dd>(.*?)</dd>\s*<dd>(.*?)</dd>",
            page,
            flags=re.DOTALL | re.IGNORECASE,
        )
        papers: list[PaperMetadata] = []
        for html_path, title_html, authors_html, links_html in blocks:
            title = _clean_text(title_html)
            if not title:
                continue
            author_values = re.findall(
                r'name="query_author"\s+value="([^"]+)"', authors_html
            )
            authors = [_clean_text(value) for value in author_values] or _split_authors(
                authors_html
            )
            html_url = urllib.parse.urljoin(self.base_url, html_path)
            pdf_url = None
            pdf_match = re.search(r'href="([^"]+\.pdf)"', links_html, re.I)
            if pdf_match:
                pdf_url = urllib.parse.urljoin(self.base_url, pdf_match.group(1))
            paper_id = _stable_id(f"cvf:{venue_key.lower()}", html_path, title)
            papers.append(
                PaperMetadata(
                    paper_id=paper_id,
                    title=title,
                    authors=authors,
                    year=year,
                    venue=f"{conf} {year}",
                    paper_url=html_url,
                    pdf_url=pdf_url,
                    sources=["topconf", "cvf"],
                    source_queries=[venue_key],
                    raw_metadata={
                        "source": "cvf",
                        "conference": conf,
                        "year": year,
                        "list_url": url,
                        "html_path": html_path,
                    },
                )
            )
        return papers

    @staticmethod
    def _confs_for_year(year: int) -> list[str]:
        confs = ["CVPR"]
        if year % 2 == 1:
            confs.append("ICCV")
        else:
            confs.append("ECCV")
        return confs

    def _sleep(self) -> None:
        elapsed = time.monotonic() - self._last_request
        if elapsed < self.request_delay:
            time.sleep(self.request_delay - elapsed)
        self._last_request = time.monotonic()


class PMLRClient:
    index_url = "https://proceedings.mlr.press/"

    def __init__(self, request_delay: float = 1.0) -> None:
        self.request_delay = request_delay
        self._last_request = 0.0

    def collect(self, years: Iterable[int]) -> list[PaperMetadata]:
        volume_specs = self.discover_volumes(years)
        papers: list[PaperMetadata] = []
        for conf, year, volume in volume_specs:
            try:
                papers.extend(self.collect_volume(conf, year, volume))
            except urllib.error.HTTPError as exc:
                if exc.code != 404:
                    raise
        return papers

    def discover_volumes(self, years: Iterable[int]) -> list[tuple[str, int, str]]:
        index = _get(self.index_url)
        return self._discover_volumes(index, years)

    def _discover_volumes(
        self, index: str, years: Iterable[int]
    ) -> list[tuple[str, int, str]]:
        wanted_years = set(years)
        specs: list[tuple[str, int, str]] = []
        pattern = re.compile(
            r'<li><a href="(v\d+)"><b>Volume \d+</b></a>\s*Proceedings of ([^<]+)</li>',
            re.I,
        )
        for volume, title in pattern.findall(index):
            title_clean = _clean_text(title)
            year_match = re.search(r"\b(20\d{2})\b", title_clean)
            if not year_match:
                continue
            year = int(year_match.group(1))
            if year not in wanted_years:
                continue
            if re.fullmatch(r"ICML\s+20\d{2}", title_clean, re.I):
                specs.append(("ICML", year, volume))
            elif re.fullmatch(r"CoRL\s+20\d{2}", title_clean, re.I):
                specs.append(("CoRL", year, volume))
        return specs

    def collect_volume(self, conf: str, year: int, volume: str) -> list[PaperMetadata]:
        url = urllib.parse.urljoin(self.index_url, f"{volume}/")
        self._sleep()
        page = _get(url)
        blocks = re.findall(
            r'<div class="paper">(.*?)</div>', page, flags=re.DOTALL | re.IGNORECASE
        )
        papers: list[PaperMetadata] = []
        for block in blocks:
            title_match = re.search(
                r'<p class="title">(.*?)</p>', block, re.DOTALL | re.I
            )
            if not title_match:
                continue
            title = _clean_text(title_match.group(1))
            authors_match = re.search(
                r'<span class="authors">(.*?)</span>', block, re.DOTALL | re.I
            )
            authors = _split_authors(authors_match.group(1)) if authors_match else []
            abs_url = None
            pdf_url = None
            abs_match = re.search(r'<a href="([^"]+)">abs</a>', block, re.I)
            if abs_match:
                abs_url = urllib.parse.urljoin(url, abs_match.group(1))
            pdf_match = re.search(r'<a href="([^"]+)"[^>]*>Download PDF</a>', block, re.I)
            if pdf_match:
                pdf_url = urllib.parse.urljoin(url, pdf_match.group(1))
            paper_id = _stable_id(f"pmlr:{volume}", abs_url or "", title)
            papers.append(
                PaperMetadata(
                    paper_id=paper_id,
                    title=title,
                    authors=authors,
                    year=year,
                    venue=f"{conf} {year}",
                    paper_url=abs_url,
                    pdf_url=pdf_url,
                    sources=["topconf", "pmlr"],
                    source_queries=[f"{conf} {year}", volume],
                    raw_metadata={
                        "source": "pmlr",
                        "conference": conf,
                        "year": year,
                        "volume": volume,
                        "list_url": url,
                    },
                )
            )
        return papers

    def _sleep(self) -> None:
        elapsed = time.monotonic() - self._last_request
        if elapsed < self.request_delay:
            time.sleep(self.request_delay - elapsed)
        self._last_request = time.monotonic()


class ECVAClient:
    base_url = "https://www.ecva.net/"
    papers_url = urllib.parse.urljoin(base_url, "papers.php")

    def collect(self, years: Iterable[int]) -> list[PaperMetadata]:
        page = _get(self.papers_url)
        papers: list[PaperMetadata] = []
        for year in years:
            if year % 2 == 1:
                continue
            papers.extend(self.collect_year_from_page(page, year))
        return papers

    def collect_year(self, year: int) -> list[PaperMetadata]:
        if year % 2 == 1:
            return []
        page = _get(self.papers_url)
        return self.collect_year_from_page(page, year)

    def collect_year_from_page(self, page: str, year: int) -> list[PaperMetadata]:
        section_match = re.search(
            rf"ECCV\s+{year}\s+Papers.*?<div class=\"accordion-content\">(.*?)(?:<button class=\"accordion\"|</main>)",
            page,
            flags=re.DOTALL | re.IGNORECASE,
        )
        if not section_match:
            return []
        section = section_match.group(1)
        blocks = re.findall(
            r'<dt class="ptitle"><br>\s*'
            r"<a href=['\"]?([^'\">\s]+)['\"]?\s*>(.*?)</a>\s*</dt>\s*"
            r"<dd>(.*?)</dd>\s*<dd>(.*?)</dd>",
            section,
            flags=re.DOTALL | re.IGNORECASE,
        )
        papers: list[PaperMetadata] = []
        for html_path, title_html, authors_html, links_html in blocks:
            title = _clean_text(title_html)
            if not title:
                continue
            pdf_url = None
            pdf_match = re.search(r"href=['\"]([^'\"]+\.pdf)['\"]", links_html, re.I)
            if pdf_match:
                pdf_url = urllib.parse.urljoin(self.base_url, pdf_match.group(1))
            doi = None
            doi_match = re.search(r"https://link\.springer\.com/chapter/([^'\"]+)", links_html)
            if doi_match:
                doi = doi_match.group(1)
            paper_url = urllib.parse.urljoin(self.base_url, html_path)
            paper_id = _stable_id(f"ecva:eccv{year}", html_path, title)
            papers.append(
                PaperMetadata(
                    paper_id=paper_id,
                    title=title,
                    authors=_split_authors(authors_html),
                    year=year,
                    venue=f"ECCV {year}",
                    paper_url=paper_url,
                    pdf_url=pdf_url,
                    doi=doi,
                    sources=["topconf", "ecva"],
                    source_queries=[f"ECCV {year}"],
                    raw_metadata={
                        "source": "ecva",
                        "conference": "ECCV",
                        "year": year,
                        "list_url": self.papers_url,
                        "html_path": html_path,
                    },
                )
            )
        return papers


class NeurIPSClient:
    base_url = "https://papers.nips.cc"

    def collect(self, years: Iterable[int]) -> list[PaperMetadata]:
        papers: list[PaperMetadata] = []
        for year in years:
            try:
                papers.extend(self.collect_year(year))
            except urllib.error.HTTPError as exc:
                if exc.code != 404:
                    raise
            except urllib.error.URLError:
                continue
        return papers

    def collect_year(self, year: int) -> list[PaperMetadata]:
        url = f"{self.base_url}/paper_files/paper/{year}"
        page = _get(url)
        blocks = re.findall(
            r'<li class="([^"]*)"[^>]*data-track="([^"]*)"[^>]*>(.*?)</li>',
            page,
            flags=re.DOTALL | re.IGNORECASE,
        )
        papers: list[PaperMetadata] = []
        for _, track, block in blocks:
            title_match = re.search(
                r'<a title="paper title" href="([^"]+)">(.*?)</a>',
                block,
                flags=re.DOTALL | re.I,
            )
            if not title_match:
                continue
            paper_path, title_html = title_match.groups()
            title = _clean_text(title_html)
            authors_match = re.search(
                r'<span class="paper-authors">(.*?)</span>',
                block,
                flags=re.DOTALL | re.I,
            )
            authors = _split_authors(authors_match.group(1)) if authors_match else []
            paper_url = urllib.parse.urljoin(self.base_url, paper_path)
            pdf_path = paper_path.replace("-Abstract-", "-Paper-").replace(".html", ".pdf")
            pdf_url = urllib.parse.urljoin(self.base_url, pdf_path)
            badge_match = re.search(
                r'<span class="paper-track-badge">(.*?)</span>', block, re.DOTALL | re.I
            )
            track_name = _clean_text(badge_match.group(1)) if badge_match else track
            paper_id = _stable_id(f"neurips:{year}", paper_path, title)
            papers.append(
                PaperMetadata(
                    paper_id=paper_id,
                    title=title,
                    authors=authors,
                    year=year,
                    venue=f"NeurIPS {year}",
                    paper_url=paper_url,
                    pdf_url=pdf_url,
                    sources=["topconf", "neurips"],
                    source_queries=[f"NeurIPS {year}"],
                    raw_metadata={
                        "source": "neurips",
                        "year": year,
                        "track": track,
                        "track_name": track_name,
                        "list_url": url,
                    },
                )
            )
        return papers


class RSSClient:
    base_url = "https://www.roboticsproceedings.org/"
    accepted_papers_url = "https://roboticsconference.org/program/papers/"
    year_to_rss_number = {2023: 19, 2024: 20, 2025: 21}

    def __init__(self, request_delay: float = 0.2) -> None:
        self.request_delay = request_delay
        self._last_request = 0.0

    def collect(self, years: Iterable[int]) -> list[PaperMetadata]:
        papers: list[PaperMetadata] = []
        for year in years:
            rss_number = self.year_to_rss_number.get(year)
            if rss_number is None and year == 2026:
                papers.extend(self.collect_accepted_year(year))
                continue
            if rss_number is None:
                continue
            try:
                papers.extend(self.collect_year(year, rss_number))
            except urllib.error.HTTPError as exc:
                if exc.code != 404:
                    raise
            except urllib.error.URLError:
                continue
        return papers

    def collect_year(self, year: int, rss_number: int) -> list[PaperMetadata]:
        rss_dir = f"rss{rss_number}/"
        url = urllib.parse.urljoin(self.base_url, f"{rss_dir}index.html")
        page = _get(url)
        rows = re.findall(
            r'<tr><td[^>]*>\s*<a href="([^"]+\.html)">(.*?)</a><br>\s*'
            r'<i>(.*?)</i><br>.*?<a href="([^"]+\.pdf)"',
            page,
            flags=re.DOTALL | re.IGNORECASE,
        )
        papers: list[PaperMetadata] = []
        for html_path, title_html, authors_html, pdf_path in rows:
            title = _clean_text(title_html)
            authors = _split_authors(authors_html)
            paper_url = urllib.parse.urljoin(url, html_path)
            pdf_url = urllib.parse.urljoin(url, pdf_path)
            paper_id = _stable_id(f"rss:rss{rss_number}", html_path, title)
            papers.append(
                PaperMetadata(
                    paper_id=paper_id,
                    title=title,
                    authors=authors,
                    year=year,
                    venue=f"RSS {year}",
                    paper_url=paper_url,
                    pdf_url=pdf_url,
                    sources=["topconf", "rss"],
                    source_queries=[f"RSS {year}", f"rss{rss_number}"],
                    raw_metadata={
                        "source": "rss",
                        "year": year,
                        "rss_number": rss_number,
                        "list_url": url,
                    },
                )
            )
        return papers

    def collect_accepted_year(self, year: int) -> list[PaperMetadata]:
        self._sleep()
        page = _get(self.accepted_papers_url)
        page = re.sub(r"<!--.*?-->", " ", page, flags=re.DOTALL)
        rows = re.findall(
            r'<tr\s+session="([^"]+)">(.*?)</tr>',
            page,
            flags=re.DOTALL | re.IGNORECASE,
        )
        papers: list[PaperMetadata] = []
        for session, row in rows:
            paper_id_match = re.search(r"<td[^>]*>\s*(\d+)\s*</td>", row, re.I)
            title_match = re.search(
                r'<a\s+href="([^"]+)">\s*<b>(.*?)</b>\s*</a>',
                row,
                flags=re.DOTALL | re.I,
            )
            if not paper_id_match or not title_match:
                continue
            paper_number = paper_id_match.group(1)
            detail_path, title_html = title_match.groups()
            title = _clean_text(title_html)
            if not title:
                continue
            cells = re.findall(r"<td[^>]*>(.*?)</td>", row, flags=re.DOTALL | re.I)
            authors_html = cells[3] if len(cells) >= 4 else ""
            authors_html = re.sub(
                r'<div\s+class="content".*',
                " ",
                authors_html,
                flags=re.DOTALL | re.I,
            )
            paper_url = urllib.parse.urljoin(
                self.accepted_papers_url,
                detail_path,
            )
            try:
                abstract = self._accepted_detail_abstract(paper_url)
            except (TimeoutError, urllib.error.URLError):
                abstract = None
            stable_id = _stable_id(f"rss:accepted{year}", paper_number, title)
            papers.append(
                PaperMetadata(
                    paper_id=stable_id,
                    title=title,
                    authors=_split_authors(authors_html),
                    year=year,
                    venue=f"RSS {year}",
                    abstract=abstract,
                    paper_url=paper_url,
                    keywords=[_clean_text(session)],
                    sources=["topconf", "rss_accepted"],
                    source_queries=[f"RSS {year}", "accepted papers"],
                    raw_metadata={
                        "source": "rss_accepted",
                        "year": year,
                        "paper_number": paper_number,
                        "session": _clean_text(session),
                        "list_url": self.accepted_papers_url,
                    },
                )
            )
        return papers

    def _accepted_detail_abstract(self, paper_url: str) -> str | None:
        self._sleep()
        page = _get(paper_url)
        match = re.search(
            r"<b[^>]*>\s*Abstract:\s*</b>\s*(.*?)</p>",
            page,
            flags=re.DOTALL | re.I,
        )
        if not match:
            return None
        abstract = _clean_text(match.group(1))
        return abstract if len(abstract) >= 80 else None

    def _sleep(self) -> None:
        elapsed = time.monotonic() - self._last_request
        if elapsed < self.request_delay:
            time.sleep(self.request_delay - elapsed)
        self._last_request = time.monotonic()


class OpenReviewClient:
    api1_url = "https://api.openreview.net/notes"
    api2_url = "https://api2.openreview.net/notes"
    forum_url = "https://openreview.net/forum"
    pdf_url = "https://openreview.net/pdf"

    def __init__(self, request_delay: float = 0.5, page_size: int = 1000) -> None:
        self.request_delay = request_delay
        self.page_size = page_size
        self._last_request = 0.0

    def collect_iclr(self, years: Iterable[int]) -> list[PaperMetadata]:
        papers: list[PaperMetadata] = []
        for year in years:
            papers.extend(self.collect_iclr_year(year))
        return papers

    def collect_iclr_year(self, year: int) -> list[PaperMetadata]:
        api_version = 1 if year <= 2023 else 2
        notes = self._fetch_notes(
            self.api1_url if api_version == 1 else self.api2_url,
            {"content.venueid": f"ICLR.cc/{year}/Conference"},
        )
        papers: list[PaperMetadata] = []
        for note in notes:
            content = note.get("content", {})
            if not isinstance(content, dict):
                continue
            venue = _content_text(content, "venue")
            if not self._is_accepted_iclr_venue(venue, year):
                continue
            title = _content_text(content, "title")
            if not title:
                continue
            note_id = str(note.get("id") or note.get("forum") or "")
            if not note_id:
                continue
            venue_label = f"ICLR {year}"
            authors = _content_list(content, "authors")
            keywords = _content_list(content, "keywords")
            paper_url = f"{self.forum_url}?id={urllib.parse.quote(note_id)}"
            pdf_url = f"{self.pdf_url}?id={urllib.parse.quote(note_id)}"
            paper_id = _stable_id(f"openreview:iclr{year}", note_id, title)
            papers.append(
                PaperMetadata(
                    paper_id=paper_id,
                    title=title,
                    authors=authors,
                    year=year,
                    venue=venue_label,
                    abstract=_content_text(content, "abstract"),
                    paper_url=paper_url,
                    pdf_url=pdf_url,
                    keywords=keywords,
                    sources=["topconf", "openreview"],
                    source_queries=[f"ICLR {year}", f"ICLR.cc/{year}/Conference"],
                    raw_metadata={
                        "source": "openreview",
                        "conference": "ICLR",
                        "year": year,
                        "api_version": api_version,
                        "note_id": note_id,
                        "forum": note.get("forum"),
                        "venue": venue,
                        "venueid": _content_text(content, "venueid"),
                    },
                )
            )
        return papers

    def _fetch_notes(self, api_url: str, params: dict[str, object]) -> list[dict[str, object]]:
        notes: list[dict[str, object]] = []
        offset = 0
        while True:
            page_params = dict(params)
            page_params["limit"] = self.page_size
            page_params["offset"] = offset
            query = urllib.parse.urlencode(page_params)
            self._sleep()
            page = _get(f"{api_url}?{query}")
            payload = json.loads(page)
            page_notes = payload.get("notes", [])
            if not isinstance(page_notes, list):
                break
            notes.extend(note for note in page_notes if isinstance(note, dict))
            if len(page_notes) < self.page_size:
                break
            offset += self.page_size
        return notes

    @staticmethod
    def _is_accepted_iclr_venue(venue: str | None, year: int) -> bool:
        if not venue:
            return False
        venue_l = venue.lower()
        if f"iclr {year}" not in venue_l:
            return False
        blocked = ("submitted", "withdrawn", "rejected", "desk rejected")
        return not any(term in venue_l for term in blocked)

    def _sleep(self) -> None:
        elapsed = time.monotonic() - self._last_request
        if elapsed < self.request_delay:
            time.sleep(self.request_delay - elapsed)
        self._last_request = time.monotonic()


class PaperceptIROSClient:
    base_url = "https://ras.papercept.net/conferences/conferences/"
    program_days = {2025: (1, 2, 3)}

    def __init__(self, request_delay: float = 0.5) -> None:
        self.request_delay = request_delay
        self._last_request = 0.0

    def collect(self, years: Iterable[int]) -> list[PaperMetadata]:
        papers: list[PaperMetadata] = []
        for year in years:
            papers.extend(self.collect_year(year))
        return papers

    def collect_year(self, year: int) -> list[PaperMetadata]:
        days = self.program_days.get(year, ())
        papers: list[PaperMetadata] = []
        for day in days:
            url = (
                f"{self.base_url}IROS{str(year)[-2:]}/program/"
                f"IROS{str(year)[-2:]}_ContentListWeb_{day}.html"
            )
            self._sleep()
            page = _get(url)
            papers.extend(self._parse_page(page, year, day, url))
        return papers

    def _parse_page(
        self, page: str, year: int, day: int, list_url: str
    ) -> list[PaperMetadata]:
        blocks = re.findall(
            r'(<tr class="pHdr".*?)(?=<tr class="pHdr"|</table>)',
            page,
            flags=re.DOTALL | re.IGNORECASE,
        )
        papers: list[PaperMetadata] = []
        for block in blocks:
            header_match = re.search(
                r'<a name="([^"]+)">(.*?)</a>', block, flags=re.DOTALL | re.I
            )
            title_match = re.search(
                r'<span class="pTtl">.*?<a[^>]*>(.*?)</a>\s*</span>',
                block,
                flags=re.DOTALL | re.I,
            )
            if not title_match:
                continue
            title = _clean_text(title_match.group(1))
            if not title:
                continue
            paper_code = _clean_text(header_match.group(2)) if header_match else ""
            anchor = header_match.group(1) if header_match else ""
            authors = [
                _clean_text(author)
                for author in re.findall(
                    r'IROS25_AuthorIndexWeb\.html#[^"]+"[^>]*>(.*?)</a>',
                    block,
                    flags=re.DOTALL | re.I,
                )
            ]
            keyword_match = re.search(
                r"<strong>Keywords:</strong>(.*?)</span>",
                block,
                flags=re.DOTALL | re.I,
            )
            keywords = (
                [
                    _clean_text(keyword)
                    for keyword in re.findall(
                        r"<a[^>]*>(.*?)</a>", keyword_match.group(1), re.DOTALL | re.I
                    )
                ]
                if keyword_match
                else []
            )
            abstract_match = re.search(
                r"<strong>Abstract:</strong>\s*(.*?)\s*</div>",
                block,
                flags=re.DOTALL | re.I,
            )
            abstract = _clean_text(abstract_match.group(1)) if abstract_match else None
            paper_id = _stable_id(f"papercept:iros{year}", anchor, title)
            papers.append(
                PaperMetadata(
                    paper_id=paper_id,
                    title=title,
                    authors=authors,
                    year=year,
                    venue=f"IROS {year}",
                    abstract=abstract,
                    paper_url=f"{list_url}#{anchor}" if anchor else list_url,
                    keywords=keywords,
                    sources=["topconf", "papercept_iros"],
                    source_queries=[f"IROS {year}", f"IROS{str(year)[-2:]}"],
                    raw_metadata={
                        "source": "papercept_iros",
                        "conference": "IROS",
                        "year": year,
                        "day": day,
                        "paper_code": paper_code,
                        "anchor": anchor,
                        "list_url": list_url,
                    },
                )
            )
        return papers

    def _sleep(self) -> None:
        elapsed = time.monotonic() - self._last_request
        if elapsed < self.request_delay:
            time.sleep(self.request_delay - elapsed)
        self._last_request = time.monotonic()


class HuggingFaceICRAClient:
    rows_url = "https://datasets-server.huggingface.co/rows"
    arxiv_api_url = "https://export.arxiv.org/api/query"

    def __init__(self, request_delay: float = 0.5, page_size: int = 100) -> None:
        self.request_delay = request_delay
        self.page_size = page_size
        self._last_request = 0.0

    def collect(self, years: Iterable[int]) -> list[PaperMetadata]:
        papers: list[PaperMetadata] = []
        for year in years:
            papers.extend(self.collect_year(year))
        return papers

    def collect_year(self, year: int) -> list[PaperMetadata]:
        rows = self._fetch_dataset_rows(f"ai-conferences/ICRA{year}")
        arxiv_ids = [
            str(row.get("arxiv_id")).strip()
            for row in rows
            if row.get("arxiv_id") and str(row.get("arxiv_id")).strip()
        ]
        arxiv_by_id = self._fetch_arxiv_by_id(arxiv_ids)
        papers: list[PaperMetadata] = []
        for row in rows:
            title = str(row.get("title") or "").strip()
            if not title:
                continue
            arxiv_id = _strip_arxiv_version(str(row.get("arxiv_id") or "").strip()) or None
            arxiv_record = arxiv_by_id.get(arxiv_id or "") or {}
            paper_url = str(row.get("paper_url") or "").strip() or None
            doi = str(row.get("doi") or "").strip() or None
            authors_value = row.get("authors")
            authors = (
                [str(author).strip() for author in authors_value if str(author).strip()]
                if isinstance(authors_value, list)
                else []
            )
            paper_id = _stable_id(f"hf:icra{year}", doi or arxiv_id or paper_url or "", title)
            raw_metadata = {
                "source": "huggingface_ai_conferences",
                "conference": "ICRA",
                "year": year,
                "dataset": f"ai-conferences/ICRA{year}",
                "dblp_key": row.get("dblp_key"),
                "arxiv_id_source": row.get("arxiv_id_source"),
            }
            if arxiv_record:
                raw_metadata["abstract_source"] = "arxiv_api"
            papers.append(
                PaperMetadata(
                    paper_id=paper_id,
                    title=title,
                    authors=authors or arxiv_record.get("authors", []),
                    year=year,
                    venue=f"ICRA {year}",
                    abstract=arxiv_record.get("abstract"),
                    paper_url=paper_url or arxiv_record.get("paper_url"),
                    pdf_url=arxiv_record.get("pdf_url"),
                    doi=doi or arxiv_record.get("doi"),
                    arxiv_id=arxiv_id,
                    fields_of_study=arxiv_record.get("categories", []),
                    sources=["topconf", "hf_icra"],
                    source_queries=[f"ICRA {year}", f"ai-conferences/ICRA{year}"],
                    raw_metadata=raw_metadata,
                )
            )
        return papers

    def _fetch_dataset_rows(self, dataset: str) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        offset = 0
        total: int | None = None
        while total is None or offset < total:
            params = {
                "dataset": dataset,
                "config": "default",
                "split": "train",
                "offset": offset,
                "length": self.page_size,
            }
            url = f"{self.rows_url}?{urllib.parse.urlencode(params)}"
            self._sleep()
            payload = json.loads(_get(url))
            total = int(payload.get("num_rows_total") or 0)
            page_rows = payload.get("rows", [])
            if not isinstance(page_rows, list) or not page_rows:
                break
            for item in page_rows:
                if isinstance(item, dict) and isinstance(item.get("row"), dict):
                    row = dict(item["row"])
                    row.pop("embedding", None)
                    rows.append(row)
            offset += len(page_rows)
        return rows

    def _fetch_arxiv_by_id(self, arxiv_ids: list[str]) -> dict[str, dict[str, object]]:
        records: dict[str, dict[str, object]] = {}
        unique_ids = sorted({arxiv_id for arxiv_id in arxiv_ids if arxiv_id})
        for start in range(0, len(unique_ids), 100):
            batch = unique_ids[start : start + 100]
            params = {"id_list": ",".join(batch), "max_results": len(batch)}
            url = f"{self.arxiv_api_url}?{urllib.parse.urlencode(params)}"
            self._sleep()
            root = ET.fromstring(_get(url))
            for entry in root.findall("a:entry", ATOM_NS):
                title = _text(entry, "a:title") or ""
                arxiv_url = _text(entry, "a:id")
                arxiv_id = _strip_arxiv_version(_arxiv_id_from_url(arxiv_url)) if arxiv_url else None
                if not arxiv_id:
                    continue
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
                pdf_url = None
                for link in entry.findall("a:link", ATOM_NS):
                    if (
                        link.attrib.get("title") == "pdf"
                        or link.attrib.get("type") == "application/pdf"
                    ):
                        pdf_url = link.attrib.get("href")
                        break
                records[arxiv_id] = {
                    "title": title,
                    "abstract": _text(entry, "a:summary"),
                    "authors": authors,
                    "paper_url": arxiv_url,
                    "pdf_url": pdf_url or f"https://arxiv.org/pdf/{arxiv_id}",
                    "doi": _text(entry, "arxiv:doi"),
                    "categories": categories,
                }
        return records

    def _sleep(self) -> None:
        elapsed = time.monotonic() - self._last_request
        if elapsed < self.request_delay:
            time.sleep(self.request_delay - elapsed)
        self._last_request = time.monotonic()


class CrossrefIROSClient:
    crossref_url = "https://api.crossref.org/works"
    openalex_url = "https://api.openalex.org/works"

    def __init__(
        self,
        request_delay: float = 0.5,
        page_size: int = 1000,
        openalex_batch_size: int = 50,
    ) -> None:
        self.request_delay = request_delay
        self.page_size = page_size
        self.openalex_batch_size = openalex_batch_size
        self._last_request = 0.0

    def collect(self, years: Iterable[int]) -> list[PaperMetadata]:
        papers: list[PaperMetadata] = []
        for year in years:
            papers.extend(self.collect_year(year))
        return papers

    def collect_year(self, year: int) -> list[PaperMetadata]:
        items = self._fetch_crossref_items(year)
        doi_values = [
            doi
            for item in items
            if (doi := _normalize_doi(str(item.get("DOI") or "")))
        ]
        openalex_by_doi = self._fetch_openalex_by_doi(doi_values)
        papers: list[PaperMetadata] = []
        for item in items:
            title_values = item.get("title") or []
            title = _clean_text(str(title_values[0] if title_values else ""))
            doi = _normalize_doi(str(item.get("DOI") or ""))
            if not title or not doi:
                continue
            openalex_item = openalex_by_doi.get(doi) or {}
            authors = []
            for author in item.get("author") or []:
                if not isinstance(author, dict):
                    continue
                given = str(author.get("given") or "").strip()
                family = str(author.get("family") or "").strip()
                name = " ".join(part for part in (given, family) if part).strip()
                if name:
                    authors.append(name)
            concepts = [
                concept.get("display_name", "")
                for concept in openalex_item.get("concepts") or []
                if concept.get("display_name")
            ]
            topics = [
                topic.get("display_name", "")
                for topic in openalex_item.get("topics") or []
                if topic.get("display_name")
            ]
            primary_location = openalex_item.get("primary_location") or {}
            pdf_url = primary_location.get("pdf_url")
            if not pdf_url:
                for location in openalex_item.get("locations") or []:
                    if location.get("pdf_url"):
                        pdf_url = location.get("pdf_url")
                        break
            abstract = _abstract_from_openalex_index(
                openalex_item.get("abstract_inverted_index")
            )
            paper_id = _stable_id(f"crossref:iros{year}", doi, title)
            papers.append(
                PaperMetadata(
                    paper_id=paper_id,
                    title=title,
                    authors=authors,
                    year=year,
                    venue=f"IROS {year}",
                    abstract=abstract,
                    paper_url=f"https://doi.org/{doi}",
                    pdf_url=pdf_url,
                    doi=doi,
                    citation_count=openalex_item.get("cited_by_count"),
                    fields_of_study=sorted(set(concepts + topics)),
                    sources=["topconf", "crossref_iros"],
                    source_queries=[
                        f"IROS {year}",
                        (
                            f"{year} IEEE/RSJ International Conference on "
                            "Intelligent Robots and Systems (IROS)"
                        ),
                    ],
                    raw_metadata={
                        "source": "crossref_iros",
                        "conference": "IROS",
                        "year": year,
                        "crossref_item": {
                            "DOI": item.get("DOI"),
                            "container-title": item.get("container-title"),
                            "published-print": item.get("published-print"),
                            "published-online": item.get("published-online"),
                        },
                        "openalex_id": openalex_item.get("id"),
                        "abstract_source": "openalex" if abstract else None,
                    },
                )
            )
        return papers

    def _fetch_crossref_items(self, year: int) -> list[dict[str, object]]:
        container_title = (
            f"{year} IEEE/RSJ International Conference on "
            "Intelligent Robots and Systems (IROS)"
        )
        cursor = "*"
        items: list[dict[str, object]] = []
        seen_dois: set[str] = set()
        while True:
            params = {
                "filter": f"prefix:10.1109,container-title:{container_title}",
                "rows": self.page_size,
                "cursor": cursor,
                "select": "DOI,title,author,published-print,published-online,container-title",
            }
            url = f"{self.crossref_url}?{urllib.parse.urlencode(params)}"
            self._sleep()
            payload = json.loads(_get(url))
            message = payload.get("message") or {}
            page_items = message.get("items") or []
            if not isinstance(page_items, list) or not page_items:
                break
            for item in page_items:
                if not isinstance(item, dict):
                    continue
                doi = _normalize_doi(str(item.get("DOI") or ""))
                containers = item.get("container-title") or []
                if (
                    doi
                    and doi.startswith("10.1109/iros")
                    and container_title in containers
                    and doi not in seen_dois
                ):
                    seen_dois.add(doi)
                    items.append(item)
            next_cursor = message.get("next-cursor")
            if not next_cursor or next_cursor == cursor or len(page_items) < self.page_size:
                break
            cursor = next_cursor
        return items

    def _fetch_openalex_by_doi(
        self, doi_values: list[str]
    ) -> dict[str, dict[str, object]]:
        records: dict[str, dict[str, object]] = {}
        unique_dois = sorted({doi for doi in doi_values if doi})
        for start in range(0, len(unique_dois), self.openalex_batch_size):
            batch = unique_dois[start : start + self.openalex_batch_size]
            doi_filter = "|".join(f"https://doi.org/{doi}" for doi in batch)
            params = {
                "filter": f"doi:{doi_filter}",
                "per-page": len(batch),
            }
            url = f"{self.openalex_url}?{urllib.parse.urlencode(params)}"
            self._sleep()
            payload = json.loads(_get(url))
            for item in payload.get("results") or []:
                if not isinstance(item, dict):
                    continue
                doi = _normalize_doi(str(item.get("doi") or ""))
                if doi:
                    records[doi] = item
        return records

    def _sleep(self) -> None:
        elapsed = time.monotonic() - self._last_request
        if elapsed < self.request_delay:
            time.sleep(self.request_delay - elapsed)
        self._last_request = time.monotonic()


class CrossrefAAAIClient:
    crossref_url = "https://api.crossref.org/works"
    openalex_url = "https://api.openalex.org/works"
    container_title = "Proceedings of the AAAI Conference on Artificial Intelligence"

    def __init__(
        self,
        request_delay: float = 1.0,
        page_size: int = 1000,
        openalex_batch_size: int = 100,
    ) -> None:
        self.request_delay = request_delay
        self.page_size = page_size
        self.openalex_batch_size = openalex_batch_size
        self._last_request = 0.0

    def collect(self, years: Iterable[int]) -> list[PaperMetadata]:
        papers: list[PaperMetadata] = []
        for year in years:
            papers.extend(self.collect_year(year))
        return papers

    def collect_year(self, year: int) -> list[PaperMetadata]:
        items = self._fetch_crossref_items(year)
        papers: list[PaperMetadata] = []
        for item in items:
            title_values = item.get("title") or []
            title = _clean_text(str(title_values[0] if title_values else ""))
            doi = _normalize_doi(str(item.get("DOI") or ""))
            if not title or not doi:
                continue
            openalex_item: dict[str, object] = {}
            authors = []
            for author in item.get("author") or []:
                if not isinstance(author, dict):
                    continue
                given = str(author.get("given") or "").strip()
                family = str(author.get("family") or "").strip()
                name = " ".join(part for part in (given, family) if part).strip()
                if name:
                    authors.append(name)
            concepts = [
                concept.get("display_name", "")
                for concept in openalex_item.get("concepts") or []
                if concept.get("display_name")
            ]
            topics = [
                topic.get("display_name", "")
                for topic in openalex_item.get("topics") or []
                if topic.get("display_name")
            ]
            abstract = _abstract_from_openalex_index(
                openalex_item.get("abstract_inverted_index")
            )
            paper_url = f"https://doi.org/{doi}"
            primary_location = openalex_item.get("primary_location") or {}
            if primary_location.get("landing_page_url"):
                paper_url = primary_location["landing_page_url"]
            pdf_url = primary_location.get("pdf_url")
            paper_id = _stable_id(f"crossref:aaai{year}", doi, title)
            papers.append(
                PaperMetadata(
                    paper_id=paper_id,
                    title=title,
                    authors=authors,
                    year=year,
                    venue=f"AAAI {year}",
                    abstract=abstract,
                    paper_url=paper_url,
                    pdf_url=pdf_url,
                    doi=doi,
                    citation_count=openalex_item.get("cited_by_count"),
                    fields_of_study=sorted(set(concepts + topics)),
                    sources=["topconf", "crossref_aaai"],
                    source_queries=[f"AAAI {year}", self.container_title],
                    raw_metadata={
                        "source": "crossref_aaai",
                        "conference": "AAAI",
                        "year": year,
                        "crossref_item": {
                            "DOI": item.get("DOI"),
                            "container-title": item.get("container-title"),
                            "published-print": item.get("published-print"),
                            "published-online": item.get("published-online"),
                        },
                        "openalex_id": openalex_item.get("id"),
                        "abstract_source": "openalex" if abstract else None,
                    },
                )
            )
        return papers

    def _fetch_crossref_items(self, year: int) -> list[dict[str, object]]:
        cursor = "*"
        items: list[dict[str, object]] = []
        seen_dois: set[str] = set()
        while True:
            params = {
                "filter": (
                    f"prefix:10.1609,container-title:{self.container_title},"
                    f"from-pub-date:{year}-01-01,until-pub-date:{year}-12-31"
                ),
                "rows": self.page_size,
                "cursor": cursor,
                "select": "DOI,title,author,published-print,published-online,container-title",
            }
            url = f"{self.crossref_url}?{urllib.parse.urlencode(params)}"
            self._sleep()
            payload = json.loads(_get(url))
            message = payload.get("message") or {}
            page_items = message.get("items") or []
            if not isinstance(page_items, list) or not page_items:
                break
            for item in page_items:
                if not isinstance(item, dict):
                    continue
                doi = _normalize_doi(str(item.get("DOI") or ""))
                containers = item.get("container-title") or []
                title_values = item.get("title") or []
                title = str(title_values[0] if title_values else "")
                if (
                    doi
                    and doi.startswith("10.1609/aaai")
                    and self.container_title in containers
                    and doi not in seen_dois
                    and "student abstract" not in title.lower()
                ):
                    seen_dois.add(doi)
                    items.append(item)
            next_cursor = message.get("next-cursor")
            if not next_cursor or next_cursor == cursor or len(page_items) < self.page_size:
                break
            cursor = next_cursor
        return items

    def _fetch_openalex_by_doi(
        self, doi_values: list[str]
    ) -> dict[str, dict[str, object]]:
        records: dict[str, dict[str, object]] = {}
        unique_dois = sorted({doi for doi in doi_values if doi})
        for start in range(0, len(unique_dois), self.openalex_batch_size):
            batch = unique_dois[start : start + self.openalex_batch_size]
            print(
                f"      OpenAlex abstracts {start + len(batch)}/{len(unique_dois)}",
                flush=True,
            )
            doi_filter = "|".join(f"https://doi.org/{doi}" for doi in batch)
            params = {
                "filter": f"doi:{doi_filter}",
                "per-page": len(batch),
            }
            url = f"{self.openalex_url}?{urllib.parse.urlencode(params)}"
            self._sleep()
            payload = json.loads(_get(url))
            for item in payload.get("results") or []:
                if not isinstance(item, dict):
                    continue
                doi = _normalize_doi(str(item.get("doi") or ""))
                if doi:
                    records[doi] = item
        return records

    def _sleep(self) -> None:
        elapsed = time.monotonic() - self._last_request
        if elapsed < self.request_delay:
            time.sleep(self.request_delay - elapsed)
        self._last_request = time.monotonic()


class AAAIOAIClient:
    archive_url = "https://ojs.aaai.org/index.php/AAAI/issue/archive"
    oai_url = "https://ojs.aaai.org/index.php/AAAI/oai"
    oai_ns = {
        "oai": "http://www.openarchives.org/OAI/2.0/",
        "dc": "http://purl.org/dc/elements/1.1/",
    }

    def __init__(self, request_delay: float = 0.5) -> None:
        self.request_delay = request_delay
        self._last_request = 0.0

    def collect(self, years: Iterable[int]) -> list[PaperMetadata]:
        wanted = set(years)
        papers: list[PaperMetadata] = []
        for year, issue_url, label in self._discover_issues(wanted):
            self._sleep()
            issue_page = _get(issue_url)
            article_ids = self._article_ids(issue_page)
            for article_id in article_ids:
                record = self._get_record(article_id)
                paper = self._record_to_paper(record, wanted)
                if paper:
                    papers.append(paper)
        return papers

    def _discover_issues(self, years: set[int]) -> list[tuple[int, str, str]]:
        issues: list[tuple[int, str, str]] = []
        seen: set[str] = set()
        page_num = 1
        while page_num <= 12:
            url = self.archive_url if page_num == 1 else f"{self.archive_url}/{page_num}"
            self._sleep()
            page = _get(url)
            links = re.findall(
                r'<a[^>]+href="([^"]+/issue/view/[^"]+)"[^>]*>\s*'
                r'([^<]*AAAI-(\d{2})\s+Technical Tracks[^<]*)</a>',
                page,
                flags=re.DOTALL | re.I,
            )
            found_relevant = False
            for href, label, yy in links:
                year = 2000 + int(yy)
                if year not in years:
                    continue
                found_relevant = True
                if href not in seen:
                    seen.add(href)
                    issues.append((year, href, _clean_text(label)))
            has_next = f"/issue/archive/{page_num + 1}" in page
            if not has_next:
                break
            if issues and not found_relevant and page_num > 6:
                break
            page_num += 1
        return sorted(issues, key=lambda item: (item[0], item[2]))

    @staticmethod
    def _article_ids(issue_page: str) -> list[str]:
        seen: set[str] = set()
        ids: list[str] = []
        for article_id in re.findall(r'id="article-(\d+)"', issue_page):
            if article_id not in seen:
                seen.add(article_id)
                ids.append(article_id)
        return ids

    def _get_record(self, article_id: str) -> ET.Element:
        params = {
            "verb": "GetRecord",
            "metadataPrefix": "oai_dc",
            "identifier": f"oai:ojs.aaai.org:article/{article_id}",
        }
        url = f"{self.oai_url}?{urllib.parse.urlencode(params)}"
        self._sleep()
        root = ET.fromstring(_get(url))
        record = root.find(".//oai:record", self.oai_ns)
        if record is None:
            raise RuntimeError(f"AAAI OAI record not found for article {article_id}")
        return record

    def _record_to_paper(
        self, record: ET.Element, wanted_years: set[int]
    ) -> PaperMetadata | None:
        metadata = record.find(".//oai:metadata", self.oai_ns)
        if metadata is None:
            return None
        titles = self._texts(metadata, "dc:title")
        if not titles:
            return None
        title = _clean_text(titles[0])
        sources = self._texts(metadata, "dc:source")
        source_text = " ".join(sources)
        year_match = re.search(r"AAAI-(\d{2})\s+Technical Tracks", source_text)
        if not year_match:
            return None
        year = 2000 + int(year_match.group(1))
        if year not in wanted_years:
            return None
        identifiers = self._texts(metadata, "dc:identifier")
        relations = self._texts(metadata, "dc:relation")
        doi = None
        paper_url = None
        for value in identifiers:
            clean = _clean_text(value)
            if clean.startswith("10."):
                doi = clean
            elif "/article/view/" in clean:
                paper_url = clean
        pdf_url = None
        for value in relations:
            if "/article/view/" in value:
                pdf_url = _clean_text(value)
                break
        description = self._texts(metadata, "dc:description")
        abstract = _clean_text(description[0]) if description else None
        authors = [_clean_text(value) for value in self._texts(metadata, "dc:creator")]
        record_id = self._texts(record, "oai:header/oai:identifier")
        paper_id = _stable_id(
            f"aaai-oai:{year}", doi or paper_url or (record_id[0] if record_id else ""), title
        )
        return PaperMetadata(
            paper_id=paper_id,
            title=title,
            authors=authors,
            year=year,
            venue=f"AAAI {year}",
            abstract=abstract,
            paper_url=paper_url,
            pdf_url=pdf_url,
            doi=_normalize_doi(doi),
            sources=["topconf", "aaai_oai"],
            source_queries=[f"AAAI {year}", "AAAI Technical Tracks"],
            raw_metadata={
                "source": "aaai_oai",
                "conference": "AAAI",
                "year": year,
                "oai_identifier": record_id[0] if record_id else None,
                "dc_source": sources,
            },
        )

    def _texts(self, root: ET.Element, path: str) -> list[str]:
        return [
            elem.text.strip()
            for elem in root.findall(f".//{path}", self.oai_ns)
            if elem.text and elem.text.strip()
        ]

    def _sleep(self) -> None:
        elapsed = time.monotonic() - self._last_request
        if elapsed < self.request_delay:
            time.sleep(self.request_delay - elapsed)
        self._last_request = time.monotonic()


class IJCAIClient:
    base_url = "https://www.ijcai.org"
    openalex_url = "https://api.openalex.org/works"

    def __init__(self, request_delay: float = 0.25) -> None:
        self.request_delay = request_delay
        self._last_request = 0.0

    def collect(self, years: Iterable[int]) -> list[PaperMetadata]:
        papers: list[PaperMetadata] = []
        for year in years:
            papers.extend(self.collect_year(year))
        return papers

    def collect_year(self, year: int) -> list[PaperMetadata]:
        url = f"{self.base_url}/proceedings/{year}/"
        self._sleep()
        page = _get(url)
        blocks = re.findall(
            r'<div id="paper(\d+)" class="paper_wrapper">'
            r'<div class="title">(.*?)</div>'
            r'<div class="authors">(.*?)</div>'
            r'<div class="details">\(<a href="([^"]+)">PDF</a>\s*\|\s*'
            r'<a href="([^"]+)">\s*Details</a>\)</div></div>',
            page,
            flags=re.DOTALL | re.I,
        )
        doi_values = [f"10.24963/ijcai.{year}/{paper_number}" for paper_number, *_ in blocks]
        openalex_by_doi: dict[str, dict[str, object]] = {}
        papers: list[PaperMetadata] = []
        for paper_number, title_html, authors_html, pdf_path, detail_path in blocks:
            title = _clean_text(title_html)
            detail_url = urllib.parse.urljoin(self.base_url, detail_path)
            pdf_url = urllib.parse.urljoin(url, pdf_path)
            doi = f"10.24963/ijcai.{year}/{paper_number}"
            openalex_item = openalex_by_doi.get(doi) or {}
            abstract = _abstract_from_openalex_index(
                openalex_item.get("abstract_inverted_index")
            )
            concepts = [
                concept.get("display_name", "")
                for concept in openalex_item.get("concepts") or []
                if concept.get("display_name")
            ]
            topics = [
                topic.get("display_name", "")
                for topic in openalex_item.get("topics") or []
                if topic.get("display_name")
            ]
            paper_id = _stable_id(f"ijcai:{year}", doi or paper_number, title)
            papers.append(
                PaperMetadata(
                    paper_id=paper_id,
                    title=title,
                    authors=_split_authors(authors_html),
                    year=year,
                    venue=f"IJCAI {year}",
                    abstract=abstract,
                    paper_url=detail_url,
                    pdf_url=pdf_url,
                    doi=doi,
                    citation_count=openalex_item.get("cited_by_count"),
                    fields_of_study=sorted(set(concepts + topics)),
                    sources=["topconf", "ijcai"],
                    source_queries=[f"IJCAI {year}"],
                    raw_metadata={
                        "source": "ijcai",
                        "conference": "IJCAI",
                        "year": year,
                        "paper_number": paper_number,
                        "list_url": url,
                        "abstract_source": "openalex" if abstract else None,
                        "openalex_id": openalex_item.get("id"),
                    },
                )
            )
        return papers

    def _fetch_openalex_by_doi(
        self, doi_values: list[str], batch_size: int = 50
    ) -> dict[str, dict[str, object]]:
        records: dict[str, dict[str, object]] = {}
        unique_dois = sorted({_normalize_doi(doi) or "" for doi in doi_values if doi})
        unique_dois = [doi for doi in unique_dois if doi]
        for start in range(0, len(unique_dois), batch_size):
            batch = unique_dois[start : start + batch_size]
            doi_filter = "|".join(f"https://doi.org/{doi}" for doi in batch)
            params = {
                "filter": f"doi:{doi_filter}",
                "per-page": len(batch),
            }
            url = f"{self.openalex_url}?{urllib.parse.urlencode(params)}"
            self._sleep()
            payload = json.loads(_get(url))
            for item in payload.get("results") or []:
                if not isinstance(item, dict):
                    continue
                doi = _normalize_doi(str(item.get("doi") or ""))
                if doi:
                    records[doi] = item
        return records

    def _sleep(self) -> None:
        elapsed = time.monotonic() - self._last_request
        if elapsed < self.request_delay:
            time.sleep(self.request_delay - elapsed)
        self._last_request = time.monotonic()
