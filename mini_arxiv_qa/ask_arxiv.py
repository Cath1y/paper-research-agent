#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from datetime import datetime
from hashlib import sha1
from pathlib import Path
from textwrap import shorten
from typing import Any


APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent

PAPERQA_SRC = PROJECT_ROOT / "third_party/paper-qa/src"
if str(PAPERQA_SRC) not in sys.path:
    sys.path.insert(0, str(PAPERQA_SRC))

DEFAULT_CACHE_DIR = APP_DIR / "cache/pdfs"
DEFAULT_RUN_JSON = APP_DIR / "runs/last.json"
USER_AGENT = "MiniArxivQA/0.1"
ATOM_NS = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
ARXIV_FIELD_PREFIX_RE = re.compile(r"(?<![\w.])(?:all|ti|au|abs|cat|co):")


@dataclass(slots=True)
class AcademicPaper:
    paper_id: str
    title: str
    authors: list[str]
    abstract: str
    doi: str
    published_date: datetime | None
    pdf_url: str
    url: str
    source: str
    updated_date: datetime | None = None
    categories: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    citations: int = 0
    influential_citations: int = 0
    references: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def year(self) -> int | None:
        return self.published_date.year if self.published_date else None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["published_date"] = self.published_date.isoformat() if self.published_date else ""
        data["updated_date"] = self.updated_date.isoformat() if self.updated_date else ""
        data["year"] = self.year
        return data


def extract_doi(text: str | None) -> str:
    if not text:
        return ""
    match = re.search(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", text, re.IGNORECASE)
    return match.group(0).rstrip(".,;)") if match else ""


def stable_id(prefix: str, *values: str | None) -> str:
    for value in values:
        clean = (value or "").strip()
        if clean:
            return f"{prefix}:{clean}"
    digest = sha1(" ".join(v or "" for v in values).encode("utf-8")).hexdigest()[:16]
    return f"{prefix}:title:{digest}"


def parse_date(value: str | None) -> datetime | None:
    value = (value or "").strip()
    if not value:
        return None
    for fmt, width in (("%Y-%m-%d", 10), ("%Y-%m", 7), ("%Y", 4)):
        try:
            return datetime.strptime(value[:width], fmt)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return None


def arxiv_id_from_url(url: str | None) -> str:
    value = (url or "").strip()
    match = re.search(
        r"arxiv\.org/(?:abs|pdf|html)/([0-9]{4}\.[0-9]{4,5}(?:v[0-9]+)?)",
        value,
        flags=re.I,
    )
    if match:
        return match.group(1).removesuffix(".pdf")
    return ""


def arxiv_pdf_url(arxiv_id: str | None) -> str:
    clean = (arxiv_id or "").strip().removesuffix(".pdf")
    return f"https://arxiv.org/pdf/{clean}" if clean else ""


def open_bytes(url: str, *, timeout: float, retries: int, delay: float) -> bytes:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code not in {429, 500, 502, 503, 504} or attempt >= retries:
                raise
            time.sleep(delay * (attempt + 1))
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
            if attempt >= retries:
                raise
            time.sleep(delay * (attempt + 1))
    if last_error:
        raise last_error
    raise RuntimeError("request failed without an exception")


def _entry_text(entry: ET.Element, path: str) -> str:
    value = entry.find(path, ATOM_NS)
    if value is None or value.text is None:
        return ""
    return " ".join(value.text.split())


def _format_arxiv_query(query: str) -> str:
    clean = " ".join(str(query or "").split())
    if ARXIV_FIELD_PREFIX_RE.search(clean):
        return clean
    return f"all:{clean}"


class ArxivSearcher:
    base_url = "https://export.arxiv.org/api/query"

    def search(
        self,
        query: str,
        max_results: int = 10,
        sort_by: str = "relevance",
        sort_order: str = "descending",
        timeout: float = 30.0,
        retries: int = 2,
        request_delay: float = 1.0,
        **_: object,
    ) -> list[AcademicPaper]:
        params = {
            "search_query": _format_arxiv_query(query),
            "max_results": max(1, min(max_results, 100)),
            "sortBy": sort_by,
            "sortOrder": sort_order,
        }
        payload = open_bytes(
            f"{self.base_url}?{urllib.parse.urlencode(params)}",
            timeout=timeout,
            retries=retries,
            delay=request_delay,
        )
        root = ET.fromstring(payload)
        papers: list[AcademicPaper] = []
        for entry in root.findall("a:entry", ATOM_NS):
            title = _entry_text(entry, "a:title")
            if not title:
                continue
            url = _entry_text(entry, "a:id")
            arxiv_id = url.rstrip("/").split("/")[-1]
            abstract = _entry_text(entry, "a:summary")
            pdf_url = ""
            for link in entry.findall("a:link", ATOM_NS):
                if link.attrib.get("title") == "pdf" or link.attrib.get("type") == "application/pdf":
                    pdf_url = link.attrib.get("href", "")
                    break
            papers.append(
                AcademicPaper(
                    paper_id=stable_id("arxiv", arxiv_id, title),
                    title=title,
                    authors=[
                        name.text.strip()
                        for author in entry.findall("a:author", ATOM_NS)
                        if (name := author.find("a:name", ATOM_NS)) is not None and name.text
                    ],
                    abstract=abstract,
                    doi=_entry_text(entry, "arxiv:doi") or extract_doi(abstract),
                    published_date=parse_date(_entry_text(entry, "a:published")),
                    updated_date=parse_date(_entry_text(entry, "a:updated")),
                    pdf_url=pdf_url or arxiv_pdf_url(arxiv_id),
                    url=url or f"https://arxiv.org/abs/{arxiv_id}",
                    source="arxiv",
                    categories=[
                        item.attrib["term"]
                        for item in entry.findall("a:category", ATOM_NS)
                        if item.attrib.get("term")
                    ],
                    extra={"arxiv_id": arxiv_id},
                )
            )
        return papers


LOW_VALUE_ARXIV_TERMS = {
    "latest",
    "recent",
    "trend",
    "trends",
    "hot",
    "survey",
    "representative",
    "overview",
    "direction",
    "directions",
    "paper",
    "papers",
}


def merge_queries(*groups: list[str], limit: int = 8) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for query in group:
            clean = re.sub(r"\s+", " ", query).strip()
            key = clean.lower()
            if clean and key not in seen:
                seen.add(key)
                merged.append(clean)
            if len(merged) >= limit:
                return merged
    return merged


def fallback_arxiv_queries(question: str) -> list[str]:
    """Generate broad arXiv-friendly fallback queries from common paper-QA wording.

    These queries intentionally avoid making words like "latest", "trends", or
    "representative" mandatory, because arXiv abstracts rarely contain them.
    """
    q_lower = question.lower()
    queries: list[str] = []

    named_papers = [
        token
        for token in re.findall(r"\b[A-Z][A-Za-z0-9-]{2,}(?:VLA|VLM|LLM|QA|RT|PI|Open|Zero|Bot)?\b", question)
        if token.lower() not in {"paperqa", "arxiv"}
    ]
    for token in named_papers[:3]:
        queries.extend([f'ti:"{token}"', f'all:"{token}"'])

    if "openvla" in q_lower:
        queries.extend(['ti:"OpenVLA"', 'all:"OpenVLA"'])

    has_vla = "vla" in q_lower or "vision-language-action" in q_lower or "vision language action" in q_lower
    has_robot = any(term in q_lower for term in ["robot", "robotic", "机器人", "具身", "embodied"])
    has_manip = any(term in q_lower for term in ["manipulation", "操作", "操控", "抓取", "机械臂"])
    if has_vla or (has_robot and has_manip):
        queries.extend(
            [
                'all:"vision-language-action" AND all:"robot"',
                'all:"vision language action" AND all:"robot"',
                'all:"VLA" AND all:"robot"',
                'all:"robot manipulation" AND all:"vision-language"',
                'all:"robot foundation model" AND all:"manipulation"',
            ]
        )

    if "diffusion" in q_lower or "扩散" in question:
        queries.extend(
            [
                'all:"diffusion policy" AND all:"robot"',
                'all:"diffusion" AND all:"robot manipulation"',
            ]
        )

    english_terms = [
        token.lower()
        for token in re.findall(r"[A-Za-z][A-Za-z0-9-]{2,}", question)
        if token.lower() not in LOW_VALUE_ARXIV_TERMS
    ]
    if english_terms:
        phrase = " ".join(english_terms[:5])
        queries.append(f'all:"{phrase}"')
        if len(english_terms) >= 2:
            queries.append(" AND ".join(f'all:"{term}"' for term in english_terms[:3]))

    return merge_queries(queries, limit=8)


def expand_author_order_queries(queries: list[str]) -> list[str]:
    expanded: list[str] = []
    for query in queries:
        expanded.append(query)
        for author in re.findall(r'au:"([^"]+)"', query):
            parts = author.split()
            if len(parts) == 2:
                expanded.append(query.replace(f'au:"{author}"', f'au:"{parts[1]} {parts[0]}"'))
    return merge_queries(expanded, limit=max(len(expanded), len(queries)))


def sanitize_generated_queries(question: str, queries: list[str]) -> list[str]:
    q_lower = question.lower()
    mentions_vla_or_robot = any(
        term in q_lower
        for term in [
            "vla",
            "vision-language-action",
            "vision language action",
            "robot",
            "robotic",
            "机器人",
            "机械臂",
            "具身",
            "操作",
            "操控",
            "抓取",
        ]
    )
    sanitized: list[str] = []
    for query in queries:
        query_lower = query.lower()
        example_domain_leak = any(
            term in query_lower
            for term in [
                "vla",
                "vision-language-action",
                "vision language action",
                "robot",
                "robotic",
            ]
        )
        if example_domain_leak and not mentions_vla_or_robot:
            continue
        sanitized.append(query)
    return expand_author_order_queries(sanitized)


@dataclass(slots=True)
class SelectedArxivPaper:
    paper: AcademicPaper
    score: float
    cache_path: str | None = None
    cache_status: str = "not_downloaded"
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["paper"] = self.paper.to_dict()
        return data


def load_dotenv_files(*paths: Path) -> None:
    for path in paths:
        if not path.exists():
            continue
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def parse_args() -> argparse.Namespace:
    load_dotenv_files(
        APP_DIR / ".env",
        APP_DIR / ".env.local",
        PROJECT_ROOT / ".env",
        PROJECT_ROOT / ".env.local",
    )
    parser = argparse.ArgumentParser(
        description="Mini arXiv QA: search arXiv PDFs and answer with PaperQA."
    )
    parser.add_argument("question", help="Question to answer from arXiv papers.")
    parser.add_argument(
        "--query",
        action="append",
        default=[],
        help="arXiv query. Can be repeated. Supports au:, ti:, abs:, all:, AND/OR.",
    )
    parser.add_argument("--max-results", type=int, default=8)
    parser.add_argument("--paper-k", type=int, default=4)
    parser.add_argument("--year-from", type=int)
    parser.add_argument("--year-to", type=int)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--run-json", type=Path, default=DEFAULT_RUN_JSON)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--download-timeout", type=float, default=60.0)
    parser.add_argument("--download-retries", type=int, default=2)
    parser.add_argument("--request-delay", type=float, default=2.0)
    parser.add_argument("--max-pdf-mb", type=float, default=80.0)
    parser.add_argument("--llm", default=os.environ.get("LITERATURE_LLM", "gpt-4o-mini"))
    parser.add_argument("--summary-llm", default=os.environ.get("LITERATURE_SUMMARY_LLM"))
    parser.add_argument("--embedding", default=os.environ.get("LITERATURE_EMBEDDING", "sparse"))
    parser.add_argument("--openai-base-url", default=os.environ.get("OPENAI_BASE_URL"))
    parser.add_argument("--openai-api-key-env", default=os.environ.get("OPENAI_API_KEY_ENV", "OPENAI_API_KEY"))
    parser.add_argument("--disable-openai-compatible-config", action="store_true")
    parser.add_argument("--llm-timeout", type=float, default=180.0)
    parser.add_argument("--evidence-k", type=int, default=10)
    parser.add_argument("--answer-max-sources", type=int, default=5)
    parser.add_argument("--answer-length", default="about 600 words")
    parser.add_argument(
        "--quiet-progress",
        action="store_true",
        help="Suppress progress logs.",
    )
    return parser.parse_args()


def progress(args: argparse.Namespace, message: str) -> None:
    if not args.quiet_progress:
        print(f"[MiniArxivQA] {message}", flush=True)


def model_has_provider_prefix(model: str) -> bool:
    return "/" in model and not model.startswith(("http://", "https://"))


def normalize_openai_model(model: str, args: argparse.Namespace) -> str:
    clean = model.strip()
    if (
        not clean
        or args.disable_openai_compatible_config
        or not args.openai_base_url
        or model_has_provider_prefix(clean)
    ):
        return clean
    return f"openai/{clean}"


def api_key_for(args: argparse.Namespace) -> str | None:
    return os.getenv(args.openai_api_key_env) or os.getenv("OPENAI_API_KEY")


async def generate_arxiv_queries(question: str, args: argparse.Namespace) -> list[str]:
    if args.query:
        return expand_author_order_queries([q.strip() for q in args.query if q.strip()])

    fallback_queries = fallback_arxiv_queries(question)
    try:
        import litellm

        model = normalize_openai_model(args.llm, args)
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Generate compact arXiv API queries for the user's paper QA task. "
                        "Return JSON only. Use arXiv syntax such as au:, ti:, abs:, all:, "
                        "AND, OR, and quoted phrases. Do not use site:, Google Scholar, "
                        "URLs, institution-only filters, or Chinese terms. Prefer exact "
                        "title queries for named papers and broad topic queries for surveys. "
                        "For author or professor questions, generate author-name queries first, "
                        "and include both likely English name orders if the original question is "
                        "Chinese. Do not invent a technical field unless it appears in the user "
                        "question. Do not copy schema examples as real queries. "
                        "For latest/trend/representative-paper questions, search the core "
                        "technical concepts instead of making words like latest, trends, hot, "
                        "or representative mandatory. Always include at least one broad recall "
                        "query that can match titles or abstracts."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Question:\n{question}\n\n"
                        "Return schema:\n"
                        '{"queries": ["au:\\"First Last\\"", "au:\\"Last First\\"", '
                        '"ti:\\"Exact Paper Title\\"", "all:\\"core technical phrase\\""]}'
                    ),
                },
            ],
            "temperature": 0,
            "max_tokens": 500,
            "timeout": args.llm_timeout,
            "response_format": {"type": "json_object"},
        }
        if args.openai_base_url and not args.disable_openai_compatible_config:
            kwargs["api_base"] = args.openai_base_url
            api_key = api_key_for(args)
            if api_key:
                kwargs["api_key"] = api_key
        response = await litellm.acompletion(**kwargs)
        payload = json.loads(response.choices[0].message.content or "{}")
        queries = [str(q).strip() for q in payload.get("queries", []) if str(q).strip()]
        queries = sanitize_generated_queries(question, queries)
        if queries:
            return merge_queries(queries, fallback_queries, limit=8)
    except Exception as exc:
        progress(args, f"query planner failed, using fallback query | {type(exc).__name__}: {exc}")

    fallback = re.sub(r"\s+", " ", question).strip()
    return fallback_queries or ([fallback] if fallback else [])


def paper_year(paper: AcademicPaper) -> int | None:
    return paper.year


def year_allowed(paper: AcademicPaper, args: argparse.Namespace) -> bool:
    year = paper_year(paper)
    if args.year_from and (year is None or year < args.year_from):
        return False
    if args.year_to and (year is None or year > args.year_to):
        return False
    return True


def dedupe_papers(papers: list[AcademicPaper]) -> list[AcademicPaper]:
    seen: set[str] = set()
    deduped: list[AcademicPaper] = []
    for paper in papers:
        arxiv_id = arxiv_id_from_url(paper.url) or str(paper.extra.get("arxiv_id") or "")
        key = arxiv_id or paper.doi or re.sub(r"\W+", " ", paper.title.lower()).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(paper)
    return deduped


async def search_arxiv(queries: list[str], args: argparse.Namespace) -> list[AcademicPaper]:
    searcher = ArxivSearcher()
    papers: list[AcademicPaper] = []
    last_request = 0.0
    for query in queries:
        elapsed = time.monotonic() - last_request
        if elapsed < args.request_delay:
            await asyncio.sleep(args.request_delay - elapsed)
        progress(args, f"arXiv search | {query}")
        last_request = time.monotonic()
        try:
            batch = await asyncio.to_thread(
                searcher.search,
                query,
                max_results=args.max_results,
                timeout=args.download_timeout,
                retries=0,
                request_delay=args.request_delay,
            )
        except Exception as exc:
            progress(args, f"arXiv search failed | {type(exc).__name__}: {exc}")
            continue
        papers.extend(paper for paper in batch if year_allowed(paper, args))
    return dedupe_papers(papers)


def select_papers(papers: list[AcademicPaper], queries: list[str], top_k: int) -> list[SelectedArxivPaper]:
    query_text = " ".join(queries).lower()
    query_terms = {
        token
        for token in re.findall(r"[a-zA-Z][a-zA-Z0-9-]{2,}", query_text)
        if token not in {"all", "and", "or", "the", "for", "with", "paper", "papers"}
    }
    selected: list[SelectedArxivPaper] = []
    for paper in papers:
        haystack = f"{paper.title} {paper.abstract} {' '.join(paper.authors)}".lower()
        overlap = sum(1 for term in query_terms if term.lower() in haystack)
        score = overlap
        score += 3.0 if paper.pdf_url else 0.0
        score += 1.0 if paper.year and paper.year >= 2024 else 0.0
        score += min(len(paper.abstract or "") / 1000.0, 1.0)
        selected.append(SelectedArxivPaper(paper=paper, score=round(score, 4)))
    selected.sort(key=lambda item: item.score, reverse=True)
    return [item for item in selected if item.paper.pdf_url][:top_k]


def safe_pdf_name(paper: AcademicPaper) -> str:
    arxiv_id = arxiv_id_from_url(paper.url) or str(paper.extra.get("arxiv_id") or "")
    stem = arxiv_id or re.sub(r"[^A-Za-z0-9_.-]+", "_", paper.title)[:90]
    return f"arxiv_{stem.strip('_')}.pdf"


def download_pdf(item: SelectedArxivPaper, args: argparse.Namespace) -> None:
    paper = item.paper
    if not paper.pdf_url:
        item.cache_status = "missing_pdf_url"
        return
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    path = args.cache_dir / safe_pdf_name(paper)
    if path.exists() and path.stat().st_size > 1024:
        item.cache_path = str(path)
        item.cache_status = "cache_hit"
        return

    last_error: Exception | None = None
    for attempt in range(max(args.download_retries, 0) + 1):
        try:
            request = urllib.request.Request(paper.pdf_url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=args.download_timeout) as response:
                payload = response.read()
            if len(payload) > args.max_pdf_mb * 1024 * 1024:
                raise ValueError(f"PDF too large: {len(payload) / 1024 / 1024:.1f} MB")
            if not payload.startswith(b"%PDF"):
                raise ValueError("download did not return a PDF")
            tmp_path = path.with_suffix(".pdf.tmp")
            tmp_path.write_bytes(payload)
            tmp_path.replace(path)
            item.cache_path = str(path)
            item.cache_status = "downloaded"
            return
        except (OSError, TimeoutError, urllib.error.URLError, ValueError) as exc:
            last_error = exc
            if attempt < args.download_retries:
                time.sleep(1.5 * (attempt + 1))
    item.cache_status = "failed"
    item.error = str(last_error) if last_error else "unknown download error"


def citation_for(paper: AcademicPaper) -> str:
    if paper.authors:
        first = paper.authors[0].split()[-1]
        author_text = f"{first} et al." if len(paper.authors) > 1 else paper.authors[0]
    else:
        author_text = "Unknown authors"
    return f"{author_text}, {paper.year or 'n.d.'}, {paper.title}"


def docname_for(paper: AcademicPaper) -> str:
    base = arxiv_id_from_url(paper.url) or paper.title
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", base)[:80] or "paper"


def configure_paperqa_settings(settings: Any, args: argparse.Namespace) -> None:
    if args.llm:
        settings.llm = normalize_openai_model(args.llm, args)
        settings.parsing.enrichment_llm = normalize_openai_model(args.llm, args)
    settings.summary_llm = normalize_openai_model(args.summary_llm or args.llm, args)
    settings.embedding = args.embedding
    settings.answer.evidence_k = args.evidence_k
    settings.answer.answer_max_sources = args.answer_max_sources
    settings.answer.answer_length = args.answer_length
    settings.parsing.multimodal = False
    settings.parsing.use_doc_details = False

    if args.openai_base_url and not args.disable_openai_compatible_config:
        api_key = api_key_for(args)
        llm_config = {
            "name": settings.llm,
            "model_list": [
                {
                    "model_name": settings.llm,
                    "litellm_params": {
                        "model": settings.llm,
                        "api_base": args.openai_base_url,
                        "temperature": settings.temperature,
                        **({"api_key": api_key} if api_key else {}),
                    },
                }
            ],
        }
        settings.llm_config = llm_config
        settings.summary_llm_config = llm_config
        settings.parsing.enrichment_llm_config = llm_config


async def answer_with_paperqa(question: str, selected: list[SelectedArxivPaper], args: argparse.Namespace) -> tuple[str, dict[str, Any]]:
    try:
        from paperqa import Docs, Settings
    except ImportError as exc:
        raise RuntimeError(
            "PaperQA is not installed. Run: python -m pip install paper-qa, "
            "or install the bundled dependency with python -m pip install -e ../third_party/paper-qa"
        ) from exc

    docs = Docs()
    settings = Settings()
    configure_paperqa_settings(settings, args)

    added = 0
    added_titles: list[str] = []
    for item in selected:
        if item.cache_status not in {"cache_hit", "downloaded"} or not item.cache_path:
            continue
        await docs.aadd(
            Path(item.cache_path),
            citation=citation_for(item.paper),
            docname=docname_for(item.paper),
            title=item.paper.title,
            doi=item.paper.doi,
            authors=item.paper.authors,
            settings=settings,
        )
        added += 1
        added_titles.append(item.paper.title)

    if added == 0:
        raise RuntimeError("No downloaded PDFs were available for PaperQA.")

    progress(args, f"PaperQA reading {added} PDFs")
    session = await docs.aquery(question, settings=settings)
    return str(getattr(session, "answer", session)), {
        "mode": "paperqa_docs",
        "added_pdfs": added,
        "added_titles": added_titles,
        "llm": settings.llm,
        "summary_llm": settings.summary_llm,
        "embedding": settings.embedding,
    }


def print_summary(queries: list[str], candidates: list[AcademicPaper], selected: list[SelectedArxivPaper]) -> None:
    print("\nSearch queries:")
    for idx, query in enumerate(queries, start=1):
        print(f"  {idx}. {query}")
    print(f"\nCandidates: {len(candidates)}")
    print(f"Selected PDFs: {len(selected)}\n")
    for idx, item in enumerate(selected, start=1):
        paper = item.paper
        authors = ", ".join(paper.authors[:4])
        if len(paper.authors) > 4:
            authors += ", ..."
        print(f"{idx}. {paper.title} ({paper.year or '?'})")
        print(f"   authors: {authors or 'Unknown'}")
        print(f"   score={item.score} cache={item.cache_status}")
        print(f"   url={paper.url}")
        print(f"   pdf={paper.pdf_url}")
        if item.error:
            print(f"   error={item.error}")


def write_run_json(
    path: Path,
    *,
    question: str,
    queries: list[str],
    candidates: list[AcademicPaper],
    selected: list[SelectedArxivPaper],
    answer: str | None,
    paperqa_trace: dict[str, Any] | None,
) -> None:
    payload = {
        "question": question,
        "queries": queries,
        "candidate_count": len(candidates),
        "selected_count": len(selected),
        "candidates": [paper.to_dict() for paper in candidates],
        "selected": [item.to_dict() for item in selected],
        "answer": answer,
        "paperqa_trace": paperqa_trace or {},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


async def async_main() -> None:
    args = parse_args()
    queries = await generate_arxiv_queries(args.question, args)
    candidates = await search_arxiv(queries, args)
    if not candidates:
        retry_queries = [
            query
            for query in fallback_arxiv_queries(args.question)
            if query.lower() not in {existing.lower() for existing in queries}
        ]
        if retry_queries:
            progress(
                args,
                f"no arXiv candidates; retrying with broader queries | count={len(retry_queries)}",
            )
            retry_candidates = await search_arxiv(retry_queries, args)
            queries = merge_queries(queries, retry_queries, limit=12)
            candidates = dedupe_papers([*candidates, *retry_candidates])
    selected = select_papers(candidates, queries, args.paper_k)

    if not args.dry_run:
        last_download = 0.0
        for item in selected:
            elapsed = time.monotonic() - last_download
            if elapsed < args.request_delay:
                time.sleep(args.request_delay - elapsed)
            progress(args, f"download PDF | {shorten(item.paper.title, width=80, placeholder='...')}")
            download_pdf(item, args)
            last_download = time.monotonic()

    print_summary(queries, candidates, selected)

    answer: str | None = None
    paperqa_trace: dict[str, Any] | None = None
    if not selected and not args.dry_run:
        write_run_json(
            args.run_json,
            question=args.question,
            queries=queries,
            candidates=candidates,
            selected=selected,
            answer=None,
            paperqa_trace={"error": "no_selected_pdfs"},
        )
        raise RuntimeError(
            "No arXiv PDFs were selected. Try a broader --query, for example "
            '--query \'all:"vision-language-action" AND all:"robot"\' or '
            '--query \'all:"robot manipulation" AND all:"vision-language"\'.'
        )
    if not args.dry_run and not args.download_only:
        answer, paperqa_trace = await answer_with_paperqa(args.question, selected, args)
        print("\nAnswer:\n")
        print(answer)

    write_run_json(
        args.run_json,
        question=args.question,
        queries=queries,
        candidates=candidates,
        selected=selected,
        answer=answer,
        paperqa_trace=paperqa_trace,
    )
    print(f"\nRun record: {args.run_json}")


def main() -> None:
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
