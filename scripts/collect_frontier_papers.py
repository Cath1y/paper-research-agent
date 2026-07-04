#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sqlite3
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from embodiedai_kb.data_collection.scorer import score_paper
from embodiedai_kb.storage.database import PaperDatabase, normalize_title
from embodiedai_kb.storage.schemas import PaperMetadata, now_iso


DEFAULT_SOURCE_DBS = [ROOT / "data/db/papers_recent_3y.sqlite"]
DEFAULT_FRONTIER_DB = ROOT / "data/db/frontier_papers.sqlite"
DEFAULT_JSONL = ROOT / "data/metadata/frontier_papers_2026_quality.jsonl"
DEFAULT_SUMMARY = ROOT / "data/metadata/frontier_papers_2026_quality.summary.json"
DEFAULT_CACHE_DIR = ROOT / "data/cache/frontier"

S2_BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"
S2_RECOMMEND_URL = "https://api.semanticscholar.org/recommendations/v1/papers"
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

DEFAULT_SEED_ARXIV_IDS = [
    "2212.06817",  # RT-1
    "2303.03378",  # PaLM-E
    "2306.10007",  # RPT
    "2307.05973",  # VoxPoser
    "2307.15818",  # RT-2
    "2405.12213",  # Octo
    "2406.09246",  # OpenVLA
    "2410.24164",  # pi0
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a separate 2026 frontier paper set using impact/community "
            "signals rather than a hard relevance-score threshold."
        )
    )
    parser.add_argument(
        "--source-db",
        action="append",
        type=Path,
        help=(
            "SQLite metadata DB to read candidates from. Can be repeated. "
            "Defaults to data/db/papers_recent_3y.sqlite."
        ),
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=DEFAULT_FRONTIER_DB,
        help="Output SQLite DB for selected frontier papers.",
    )
    parser.add_argument(
        "--jsonl-path",
        type=Path,
        default=DEFAULT_JSONL,
        help="Output JSONL file for selected frontier papers.",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=DEFAULT_SUMMARY,
        help="Output JSON summary path.",
    )
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional candidate limit for smoke tests.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete the output frontier DB before writing this run.",
    )
    parser.add_argument(
        "--skip-s2",
        action="store_true",
        help="Skip Semantic Scholar citation/recommendation enrichment.",
    )
    parser.add_argument(
        "--skip-hf",
        action="store_true",
        help="Skip Hugging Face paper-page enrichment.",
    )
    parser.add_argument(
        "--s2-batch-size",
        type=int,
        default=100,
        help="Semantic Scholar batch detail request size.",
    )
    parser.add_argument(
        "--s2-delay",
        type=float,
        default=1.0,
        help="Seconds to wait between Semantic Scholar requests.",
    )
    parser.add_argument(
        "--hf-delay",
        type=float,
        default=0.15,
        help="Seconds to wait between Hugging Face requests.",
    )
    parser.add_argument(
        "--recommendation-limit",
        type=int,
        default=500,
        help="Max papers requested from Semantic Scholar recommendations.",
    )
    parser.add_argument(
        "--seed-arxiv-id",
        action="append",
        help=(
            "Additional Semantic Scholar recommendation seed arXiv ID. "
            "Defaults include RT-1/RT-2/PaLM-E/OpenVLA/Octo/pi0-style seeds."
        ),
    )
    return parser.parse_args()


def json_loads(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def strip_arxiv_version(value: str | None) -> str | None:
    if not value:
        return None
    match = re.search(r"(\d{4}\.\d{4,5})(?:v\d+)?", str(value))
    if match:
        return match.group(1)
    return str(value).strip().removeprefix("arXiv:").removeprefix("ARXIV:") or None


def safe_cache_name(key: str) -> str:
    return urllib.parse.quote(key, safe="")


def cache_read(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def cache_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json_dumps(payload), encoding="utf-8")


def http_json(
    url: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    api_key: str | None = None,
    max_retries: int = 4,
) -> Any:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"User-Agent": "EmbodiedAI-KB/0.1 (frontier selection)"}
    if payload is not None:
        headers["Content-Type"] = "application/json"
    if api_key:
        headers["x-api-key"] = api_key
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            last_error = exc
            if exc.code not in {429, 500, 502, 503, 504} or attempt >= max_retries:
                raise
            retry_after = exc.headers.get("Retry-After")
            try:
                delay = float(retry_after) if retry_after else 2.0 * (attempt + 1)
            except ValueError:
                delay = 2.0 * (attempt + 1)
            time.sleep(delay)
        except (TimeoutError, urllib.error.URLError) as exc:
            last_error = exc
            if attempt >= max_retries:
                raise
            time.sleep(2.0 * (attempt + 1))
    if last_error:
        raise last_error
    raise RuntimeError(f"Request failed without exception: {url}")


def row_to_paper(row: sqlite3.Row) -> PaperMetadata:
    return PaperMetadata(
        paper_id=row["paper_id"],
        title=row["title"],
        authors=json_loads(row["authors"], []),
        year=row["year"],
        venue=row["venue"],
        abstract=row["abstract"],
        paper_url=row["paper_url"],
        pdf_url=row["pdf_url"],
        code_url=row["code_url"],
        project_url=row["project_url"],
        doi=row["doi"],
        arxiv_id=row["arxiv_id"],
        semantic_scholar_id=row["semantic_scholar_id"],
        citation_count=row["citation_count"],
        influential_citation_count=row["influential_citation_count"],
        fields_of_study=json_loads(row["fields_of_study"], []),
        keywords=json_loads(row["keywords"], []),
        categories=json_loads(row["categories"], []),
        sources=json_loads(row["sources"], []),
        source_queries=json_loads(row["source_queries"], []),
        relevance_score=float(row["relevance_score"] or 0),
        relevance_reasons=json_loads(row["relevance_reasons"], []),
        decision=row["decision"],
        raw_metadata=json_loads(row["raw_metadata"], {}),
        collected_at=row["collected_at"],
        updated_at=row["updated_at"],
    )


def candidate_key(paper: PaperMetadata) -> str:
    arxiv_id = strip_arxiv_version(paper.arxiv_id)
    if arxiv_id:
        return f"arxiv:{arxiv_id.lower()}"
    if paper.doi:
        return f"doi:{paper.doi.lower()}"
    if paper.semantic_scholar_id:
        return f"s2:{paper.semantic_scholar_id}"
    return f"title:{normalize_title(paper.title)}"


def merge_candidate(existing: PaperMetadata, incoming: PaperMetadata) -> PaperMetadata:
    if len(incoming.title) > len(existing.title):
        existing.title = incoming.title
    if incoming.abstract and (
        not existing.abstract or len(incoming.abstract) > len(existing.abstract)
    ):
        existing.abstract = incoming.abstract
    for attr in (
        "paper_url",
        "pdf_url",
        "code_url",
        "project_url",
        "doi",
        "arxiv_id",
        "semantic_scholar_id",
        "venue",
    ):
        if not getattr(existing, attr) and getattr(incoming, attr):
            setattr(existing, attr, getattr(incoming, attr))
    existing.authors = sorted(set(existing.authors + incoming.authors))
    existing.fields_of_study = sorted(
        set(existing.fields_of_study + incoming.fields_of_study)
    )
    existing.keywords = sorted(set(existing.keywords + incoming.keywords))
    existing.categories = sorted(set(existing.categories + incoming.categories))
    existing.sources = sorted(set(existing.sources + incoming.sources))
    existing.source_queries = sorted(set(existing.source_queries + incoming.source_queries))
    existing.relevance_reasons = sorted(
        set(existing.relevance_reasons + incoming.relevance_reasons)
    )
    existing.relevance_score = max(existing.relevance_score, incoming.relevance_score)
    if incoming.citation_count is not None:
        existing.citation_count = max(existing.citation_count or 0, incoming.citation_count)
    if incoming.influential_citation_count is not None:
        existing.influential_citation_count = max(
            existing.influential_citation_count or 0,
            incoming.influential_citation_count,
        )
    existing.raw_metadata.setdefault("merged_sources", []).append(incoming.raw_metadata)
    existing.updated_at = now_iso()
    return existing


def load_candidates(
    source_dbs: Iterable[Path], year: int, limit: int | None
) -> list[PaperMetadata]:
    candidates: dict[str, PaperMetadata] = {}
    for db_path in source_dbs:
        if not db_path.exists():
            raise FileNotFoundError(f"Source DB not found: {db_path}")
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        query = """
            SELECT *
            FROM papers
            WHERE year = ?
              AND abstract IS NOT NULL
              AND length(trim(abstract)) >= 40
              AND pdf_url IS NOT NULL
              AND length(trim(pdf_url)) > 0
            ORDER BY relevance_score DESC, citation_count DESC, title
        """
        rows = conn.execute(query, (year,)).fetchall()
        conn.close()
        for row in rows:
            paper = row_to_paper(row)
            paper.raw_metadata.setdefault("frontier_source_db", str(db_path))
            key = candidate_key(paper)
            if key in candidates:
                candidates[key] = merge_candidate(candidates[key], paper)
            else:
                candidates[key] = paper
            if limit and len(candidates) >= limit:
                return list(candidates.values())
    return list(candidates.values())


def s2_identifier(paper: PaperMetadata) -> str | None:
    arxiv_id = strip_arxiv_version(paper.arxiv_id)
    if arxiv_id:
        return f"ARXIV:{arxiv_id}"
    if paper.doi:
        return f"DOI:{paper.doi}"
    if paper.semantic_scholar_id:
        return paper.semantic_scholar_id
    return None


def chunks(values: list[str], size: int) -> Iterable[list[str]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def fetch_s2_details(
    papers: list[PaperMetadata],
    cache_dir: Path,
    batch_size: int,
    delay: float,
) -> dict[str, dict[str, Any] | None]:
    cache_base = cache_dir / "semantic_scholar_details"
    api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    identifiers = sorted({identifier for paper in papers if (identifier := s2_identifier(paper))})
    results: dict[str, dict[str, Any] | None] = {}
    missing: list[str] = []
    for identifier in identifiers:
        cached = cache_read(cache_base / f"{safe_cache_name(identifier)}.json")
        if cached is not None:
            results[identifier] = cached.get("item")
        else:
            missing.append(identifier)
    params = urllib.parse.urlencode({"fields": S2_FIELDS})
    url = f"{S2_BATCH_URL}?{params}"
    for index, batch in enumerate(chunks(missing, max(1, batch_size)), start=1):
        print(f"  S2 details batch {index}: {len(batch)} ids", flush=True)
        payload = {"ids": batch}
        try:
            data = http_json(url, method="POST", payload=payload, api_key=api_key)
        except Exception as exc:
            print(f"    S2 batch error: {exc}", flush=True)
            data = [None] * len(batch)
        if not isinstance(data, list):
            data = [None] * len(batch)
        for identifier, item in zip(batch, data, strict=False):
            results[identifier] = item if isinstance(item, dict) else None
            cache_write(
                cache_base / f"{safe_cache_name(identifier)}.json",
                {"fetched_at": now_iso(), "item": results[identifier]},
            )
        if delay:
            time.sleep(delay)
    return results


def fetch_s2_recommendations(
    seed_arxiv_ids: list[str],
    cache_dir: Path,
    limit: int,
    delay: float,
) -> list[dict[str, Any]]:
    cache_key = "|".join(sorted(seed_arxiv_ids)) + f"|limit={limit}"
    cache_path = cache_dir / "semantic_scholar_recommendations" / (
        safe_cache_name(cache_key) + ".json"
    )
    cached = cache_read(cache_path)
    if cached is not None:
        items = cached.get("recommendedPapers", [])
        return items if isinstance(items, list) else []

    api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    params = urllib.parse.urlencode({"fields": S2_FIELDS, "limit": limit})
    url = f"{S2_RECOMMEND_URL}?{params}"
    positive_ids = [f"ARXIV:{strip_arxiv_version(value)}" for value in seed_arxiv_ids]
    positive_ids = [value for value in positive_ids if value != "ARXIV:None"]
    payload = {"positivePaperIds": positive_ids}
    try:
        data = http_json(url, method="POST", payload=payload, api_key=api_key)
    except Exception as exc:
        print(f"  S2 recommendation error: {exc}", flush=True)
        data = {"recommendedPapers": []}
    if delay:
        time.sleep(delay)
    cache_write(cache_path, {"fetched_at": now_iso(), **(data or {})})
    items = (data or {}).get("recommendedPapers", [])
    return items if isinstance(items, list) else []


def s2_match_keys(item: dict[str, Any] | None) -> set[str]:
    if not item:
        return set()
    keys: set[str] = set()
    paper_id = item.get("paperId")
    if paper_id:
        keys.add(f"s2:{paper_id}")
    external = item.get("externalIds") or {}
    arxiv_id = strip_arxiv_version(external.get("ArXiv"))
    if arxiv_id:
        keys.add(f"arxiv:{arxiv_id.lower()}")
    doi = external.get("DOI")
    if doi:
        keys.add(f"doi:{str(doi).lower()}")
    title = item.get("title")
    if title:
        keys.add(f"title:{normalize_title(title)}")
    return keys


def fetch_hf_paper(
    arxiv_id: str,
    cache_dir: Path,
    delay: float,
) -> dict[str, Any] | None:
    clean_id = strip_arxiv_version(arxiv_id)
    if not clean_id:
        return None
    cache_path = cache_dir / "huggingface_papers" / f"{safe_cache_name(clean_id)}.json"
    cached = cache_read(cache_path)
    if cached is not None:
        return cached.get("item")
    url = f"https://huggingface.co/api/papers/{urllib.parse.quote(clean_id, safe='')}"
    try:
        item = http_json(url)
    except Exception as exc:
        print(f"    HF paper error {clean_id}: {exc}", flush=True)
        item = None
    cache_write(cache_path, {"fetched_at": now_iso(), "item": item})
    if delay:
        time.sleep(delay)
    return item if isinstance(item, dict) else None


def apply_s2_details(paper: PaperMetadata, item: dict[str, Any] | None) -> None:
    if not item:
        return
    external = item.get("externalIds") or {}
    paper.semantic_scholar_id = paper.semantic_scholar_id or item.get("paperId")
    paper.citation_count = item.get("citationCount", paper.citation_count)
    paper.influential_citation_count = item.get(
        "influentialCitationCount", paper.influential_citation_count
    )
    if not paper.abstract and item.get("abstract"):
        paper.abstract = item.get("abstract")
    if not paper.paper_url and item.get("url"):
        paper.paper_url = item.get("url")
    if not paper.doi and external.get("DOI"):
        paper.doi = str(external["DOI"])
    if not paper.arxiv_id and external.get("ArXiv"):
        paper.arxiv_id = str(external["ArXiv"])
    open_pdf = item.get("openAccessPdf") or {}
    if not paper.pdf_url and open_pdf.get("url"):
        paper.pdf_url = open_pdf.get("url")
    fields = [str(value) for value in item.get("fieldsOfStudy") or [] if value]
    for field in item.get("s2FieldsOfStudy") or []:
        category = field.get("category")
        if category:
            fields.append(str(category))
    if fields:
        paper.fields_of_study = sorted(set(paper.fields_of_study + fields))


def hf_signal_summary(item: dict[str, Any] | None) -> dict[str, Any]:
    if not item:
        return {}
    return {
        "id": item.get("id"),
        "upvotes": int(item.get("upvotes") or 0),
        "submitted_on_daily_at": item.get("submittedOnDailyAt"),
        "github_repo": item.get("githubRepo"),
        "github_stars": int(item.get("githubStars") or 0),
        "num_total_models": int(item.get("numTotalModels") or 0),
        "num_total_datasets": int(item.get("numTotalDatasets") or 0),
        "num_total_spaces": int(item.get("numTotalSpaces") or 0),
        "ai_keywords": item.get("ai_keywords") or [],
        "ai_summary": item.get("ai_summary"),
    }


def quality_signals(
    paper: PaperMetadata,
    *,
    s2_recommended: bool,
    hf: dict[str, Any],
) -> list[str]:
    signals: list[str] = []
    if (paper.citation_count or 0) > 0:
        signals.append("s2_citations")
    if (paper.influential_citation_count or 0) > 0:
        signals.append("s2_influential_citations")
    if s2_recommended:
        signals.append("s2_recommended_from_seed")
    if hf.get("upvotes", 0) > 0:
        signals.append("hf_upvotes")
    if hf.get("submitted_on_daily_at"):
        signals.append("hf_daily_papers")
    if paper.code_url or hf.get("github_repo"):
        signals.append("github_or_code_link")
    if paper.project_url:
        signals.append("project_link")
    if hf.get("num_total_models", 0) > 0:
        signals.append("hf_linked_models")
    if hf.get("num_total_datasets", 0) > 0:
        signals.append("hf_linked_datasets")
    if hf.get("num_total_spaces", 0) > 0:
        signals.append("hf_linked_spaces")
    return signals


def frontier_score(
    paper: PaperMetadata,
    *,
    signals: list[str],
    hf: dict[str, Any],
) -> float:
    score = 0.0
    citations = paper.citation_count or 0
    influential = paper.influential_citation_count or 0
    upvotes = int(hf.get("upvotes") or 0)
    github_stars = int(hf.get("github_stars") or 0)
    score += math.log1p(citations) * 1.4
    score += math.log1p(influential) * 2.0
    score += math.log1p(upvotes) * 1.5
    score += math.log1p(github_stars) * 1.2
    score += 2.5 if "s2_recommended_from_seed" in signals else 0.0
    score += 1.0 if "hf_daily_papers" in signals else 0.0
    score += 1.4 if "github_or_code_link" in signals else 0.0
    score += 1.0 if "project_link" in signals else 0.0
    score += 0.8 if "hf_linked_models" in signals else 0.0
    score += 0.8 if "hf_linked_datasets" in signals else 0.0
    score += 0.8 if "hf_linked_spaces" in signals else 0.0
    score += min(paper.relevance_score, 20.0) * 0.05
    return round(score, 3)


def ensure_frontier_metadata(
    paper: PaperMetadata,
    *,
    signals: list[str],
    score: float,
    hf: dict[str, Any],
    s2_item: dict[str, Any] | None,
    s2_recommended: bool,
) -> PaperMetadata:
    if hf.get("github_repo") and not paper.code_url:
        paper.code_url = hf["github_repo"]
    if hf.get("ai_keywords"):
        paper.keywords = sorted(set(paper.keywords + [str(v) for v in hf["ai_keywords"]]))
    paper.sources = sorted(set(paper.sources + ["frontier_quality"]))
    paper.categories = sorted(set(paper.categories + ["frontier_2026"]))
    paper.keywords = sorted(set(paper.keywords + signals))
    paper.decision = "frontier_selected"
    paper.raw_metadata = {
        **paper.raw_metadata,
        "frontier": {
            "selected_at": now_iso(),
            "frontier_score": score,
            "quality_signals": signals,
            "s2_recommended_from_seed": s2_recommended,
            "semantic_scholar": {
                "paperId": (s2_item or {}).get("paperId"),
                "url": (s2_item or {}).get("url"),
                "citationCount": paper.citation_count,
                "influentialCitationCount": paper.influential_citation_count,
            },
            "huggingface": hf,
        },
    }
    paper.updated_at = now_iso()
    scored = score_paper(paper)
    scored.decision = "frontier_selected"
    return scored


def paper_to_export_record(paper: PaperMetadata) -> dict[str, Any]:
    record = asdict(paper)
    frontier = paper.raw_metadata.get("frontier", {})
    record["frontier_score"] = frontier.get("frontier_score", 0)
    record["quality_signals"] = frontier.get("quality_signals", [])
    record["hf_upvotes"] = frontier.get("huggingface", {}).get("upvotes", 0)
    record["hf_github_stars"] = frontier.get("huggingface", {}).get("github_stars", 0)
    record["s2_recommended_from_seed"] = frontier.get("s2_recommended_from_seed", False)
    return record


def write_jsonl(path: Path, papers: list[PaperMetadata]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for paper in sorted(
            papers,
            key=lambda p: (
                p.raw_metadata.get("frontier", {}).get("frontier_score", 0),
                p.citation_count or 0,
                p.relevance_score,
            ),
            reverse=True,
        ):
            handle.write(json_dumps(paper_to_export_record(paper)) + "\n")


def write_summary(
    path: Path,
    *,
    args: argparse.Namespace,
    candidates: list[PaperMetadata],
    selected: list[PaperMetadata],
    signal_counts: dict[str, int],
    errors: list[str],
) -> None:
    top = [
        {
            "title": paper.title,
            "arxiv_id": strip_arxiv_version(paper.arxiv_id),
            "frontier_score": paper.raw_metadata.get("frontier", {}).get("frontier_score"),
            "quality_signals": paper.raw_metadata.get("frontier", {}).get("quality_signals", []),
            "citation_count": paper.citation_count,
            "influential_citation_count": paper.influential_citation_count,
            "hf_upvotes": paper.raw_metadata.get("frontier", {})
            .get("huggingface", {})
            .get("upvotes", 0),
            "code_url": paper.code_url,
            "pdf_url": paper.pdf_url,
        }
        for paper in sorted(
            selected,
            key=lambda p: p.raw_metadata.get("frontier", {}).get("frontier_score", 0),
            reverse=True,
        )[:20]
    ]
    summary = {
        "created_at": now_iso(),
        "year": args.year,
        "source_dbs": [str(path) for path in args.source_db],
        "candidate_count": len(candidates),
        "selected_count": len(selected),
        "signal_counts": dict(sorted(signal_counts.items())),
        "db_path": str(args.db_path),
        "jsonl_path": str(args.jsonl_path),
        "top": top,
        "errors": errors,
        "criteria": {
            "hard_filters": [
                f"year == {args.year}",
                "has abstract",
                "has PDF URL",
            ],
            "quality_gate": (
                "At least one of: S2 citations/influential citations, "
                "S2 recommendation from seed papers, HF upvotes/daily papers, "
                "GitHub/code/project/model/dataset/Space link."
            ),
            "relevance_score": "not a hard filter; used only as a small ranking feature",
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json_dumps(summary) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.source_db is None:
        args.source_db = DEFAULT_SOURCE_DBS
    else:
        args.source_db = [path.resolve() for path in args.source_db]

    if args.reset and args.db_path.exists():
        args.db_path.unlink()

    print("Loading candidates", flush=True)
    candidates = load_candidates(args.source_db, args.year, args.limit)
    print(f"  candidates with year/abstract/PDF: {len(candidates)}", flush=True)

    s2_details: dict[str, dict[str, Any] | None] = {}
    s2_recommendation_keys: set[str] = set()
    errors: list[str] = []
    if not args.skip_s2:
        print("Enriching Semantic Scholar details", flush=True)
        try:
            s2_details = fetch_s2_details(
                candidates,
                args.cache_dir,
                args.s2_batch_size,
                args.s2_delay,
            )
        except Exception as exc:
            errors.append(f"s2_details:{exc}")
            print(f"  S2 details failed: {exc}", flush=True)
        print("Fetching Semantic Scholar seed recommendations", flush=True)
        seeds = DEFAULT_SEED_ARXIV_IDS + list(args.seed_arxiv_id or [])
        try:
            recommendations = fetch_s2_recommendations(
                seeds,
                args.cache_dir,
                args.recommendation_limit,
                args.s2_delay,
            )
            for item in recommendations:
                s2_recommendation_keys.update(s2_match_keys(item))
            print(
                f"  recommendation papers: {len(recommendations)}, match keys: {len(s2_recommendation_keys)}",
                flush=True,
            )
        except Exception as exc:
            errors.append(f"s2_recommendations:{exc}")
            print(f"  S2 recommendations failed: {exc}", flush=True)

    selected: list[PaperMetadata] = []
    signal_counts: dict[str, int] = {}
    print("Applying quality gate", flush=True)
    for index, paper in enumerate(candidates, start=1):
        identifier = s2_identifier(paper)
        s2_item = s2_details.get(identifier) if identifier else None
        apply_s2_details(paper, s2_item)
        match_keys = {candidate_key(paper)}
        match_keys.update(s2_match_keys(s2_item))
        s2_recommended = bool(match_keys & s2_recommendation_keys)

        hf_summary: dict[str, Any] = {}
        if not args.skip_hf and paper.arxiv_id:
            try:
                hf_item = fetch_hf_paper(paper.arxiv_id, args.cache_dir, args.hf_delay)
                hf_summary = hf_signal_summary(hf_item)
            except Exception as exc:
                errors.append(f"hf:{paper.arxiv_id}:{exc}")
                print(f"    HF failed for {paper.arxiv_id}: {exc}", flush=True)

        signals = quality_signals(paper, s2_recommended=s2_recommended, hf=hf_summary)
        if not signals:
            continue
        score = frontier_score(paper, signals=signals, hf=hf_summary)
        paper = ensure_frontier_metadata(
            paper,
            signals=signals,
            score=score,
            hf=hf_summary,
            s2_item=s2_item,
            s2_recommended=s2_recommended,
        )
        selected.append(paper)
        for signal in signals:
            signal_counts[signal] = signal_counts.get(signal, 0) + 1
        if index % 50 == 0:
            print(f"  processed {index}/{len(candidates)}, selected {len(selected)}", flush=True)

    print("Writing frontier DB and exports", flush=True)
    db = PaperDatabase(args.db_path)
    try:
        for paper in selected:
            db.upsert_paper(paper)
    finally:
        db.close()
    write_jsonl(args.jsonl_path, selected)
    write_summary(
        args.summary_path,
        args=args,
        candidates=candidates,
        selected=selected,
        signal_counts=signal_counts,
        errors=errors,
    )

    print("\nFrontier selection summary")
    print(f"  candidates: {len(candidates)}")
    print(f"  selected: {len(selected)}")
    print(f"  db_path: {args.db_path}")
    print(f"  jsonl_path: {args.jsonl_path}")
    print(f"  summary_path: {args.summary_path}")
    if signal_counts:
        print("  signals:")
        for signal, count in sorted(signal_counts.items(), key=lambda item: (-item[1], item[0])):
            print(f"    {signal}: {count}")
    if errors:
        print("  errors:")
        for error in errors[:20]:
            print(f"    {error}")


if __name__ == "__main__":
    main()
