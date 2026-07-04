#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from embodiedai_kb.data_collection.arxiv_client import ArxivClient
from embodiedai_kb.data_collection.openalex_client import OpenAlexClient
from embodiedai_kb.data_collection.scorer import score_paper
from embodiedai_kb.data_collection.semantic_scholar_client import SemanticScholarClient
from embodiedai_kb.storage.database import PaperDatabase


DEFAULT_QUERIES = [
    "vision language action robot",
    "vision-language-action",
    "VLA robot",
    "embodied AI robot",
    "embodied agent robotics",
    "robot foundation model",
    "robotic manipulation language",
    "language guided robotics",
    "vision language navigation",
    "VLN embodied",
    "long horizon robot manipulation",
    "mobile manipulation language",
    "robot learning foundation model",
    "world model robotics embodied",
    "generalist robot policy",
    "open vocabulary robotic manipulation",
    "large language model robot planning",
    "multimodal robot learning",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect embodied AI paper metadata into SQLite."
    )
    parser.add_argument(
        "--db-path",
        default="data/db/papers.sqlite",
        help="SQLite database path.",
    )
    parser.add_argument(
        "--jsonl-path",
        default="data/metadata/papers_score_gte_4.jsonl",
        help="JSONL export path for accepted papers.",
    )
    parser.add_argument(
        "--query",
        action="append",
        help="Search query. Can be repeated. Defaults to embodied AI/VLA query set.",
    )
    parser.add_argument(
        "--query-file",
        help="Optional newline-delimited query file.",
    )
    parser.add_argument(
        "--sources",
        default="arxiv,openalex",
        help="Comma-separated sources: arxiv,openalex,semantic_scholar.",
    )
    parser.add_argument(
        "--arxiv-limit",
        type=int,
        default=50,
        help="Max arXiv results per query.",
    )
    parser.add_argument(
        "--semantic-limit",
        type=int,
        default=50,
        help="Max Semantic Scholar results per query.",
    )
    parser.add_argument(
        "--openalex-limit",
        type=int,
        default=50,
        help="Max OpenAlex results per query.",
    )
    parser.add_argument(
        "--arxiv-page-size",
        type=int,
        default=100,
        help="arXiv page size per API request.",
    )
    parser.add_argument(
        "--arxiv-request-delay",
        type=float,
        default=3.1,
        help="Seconds to wait between arXiv API requests.",
    )
    parser.add_argument(
        "--date-from",
        help="Earliest submission/publication date, YYYY-MM-DD.",
    )
    parser.add_argument(
        "--date-to",
        help="Latest submission/publication date, YYYY-MM-DD.",
    )
    parser.add_argument(
        "--recent-years",
        type=int,
        help="Convenience option: set date range from today minus N years to today.",
    )
    parser.add_argument(
        "--arxiv-sort-by",
        default="relevance",
        choices=["relevance", "lastUpdatedDate", "submittedDate"],
        help="arXiv sort field.",
    )
    parser.add_argument(
        "--year-windowed",
        action="store_true",
        help="Split the date range into calendar-year windows before querying.",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=4,
        help="Minimum deterministic score to store.",
    )
    parser.add_argument(
        "--include-rejected",
        action="store_true",
        help="Store all scored papers, not only min-score papers.",
    )
    return parser.parse_args()


def load_queries(args: argparse.Namespace) -> list[str]:
    queries = list(args.query or [])
    if args.query_file:
        with Path(args.query_file).open(encoding="utf-8") as f:
            queries.extend(
                line.strip()
                for line in f
                if line.strip() and not line.lstrip().startswith("#")
            )
    if not queries:
        queries = DEFAULT_QUERIES
    seen: set[str] = set()
    unique: list[str] = []
    for query in queries:
        key = query.lower()
        if key not in seen:
            seen.add(key)
            unique.append(query)
    return unique


def resolve_date_range(args: argparse.Namespace) -> tuple[str | None, str | None]:
    if args.recent_years is None:
        return args.date_from, args.date_to
    today = date.today()
    try:
        start = today.replace(year=today.year - args.recent_years)
    except ValueError:
        start = today.replace(month=2, day=28, year=today.year - args.recent_years)
    return args.date_from or start.isoformat(), args.date_to or today.isoformat()


def build_date_windows(
    date_from: str | None, date_to: str | None, year_windowed: bool
) -> list[tuple[str | None, str | None]]:
    if not year_windowed or not date_from or not date_to:
        return [(date_from, date_to)]
    start = date.fromisoformat(date_from)
    end = date.fromisoformat(date_to)
    windows: list[tuple[str | None, str | None]] = []
    current_year = start.year
    while current_year <= end.year:
        window_start = max(start, date(current_year, 1, 1))
        window_end = min(end, date(current_year, 12, 31))
        windows.append((window_start.isoformat(), window_end.isoformat()))
        current_year += 1
    return windows


def main() -> None:
    args = parse_args()
    sources = {source.strip() for source in args.sources.split(",") if source.strip()}
    queries = load_queries(args)
    date_from, date_to = resolve_date_range(args)
    date_windows = build_date_windows(date_from, date_to, args.year_windowed)
    db = PaperDatabase(args.db_path)
    arxiv = ArxivClient(request_delay=args.arxiv_request_delay) if "arxiv" in sources else None
    openalex = OpenAlexClient() if "openalex" in sources else None
    semantic = SemanticScholarClient() if "semantic_scholar" in sources else None

    fetched = 0
    accepted = 0
    errors: list[str] = []
    try:
        for idx, query in enumerate(queries, start=1):
            print(f"[{idx}/{len(queries)}] query={query!r}", flush=True)
            batches = []
            if arxiv:
                for window_from, window_to in date_windows:
                    window_label = (
                        f"{window_from}..{window_to}" if window_from or window_to else "all"
                    )
                    try:
                        papers = arxiv.search(
                            query,
                            limit=args.arxiv_limit,
                            date_from=window_from,
                            date_to=window_to,
                            sort_by=args.arxiv_sort_by,
                            sort_order="descending",
                            page_size=args.arxiv_page_size,
                        )
                        batches.append(("arxiv", papers))
                        print(
                            f"  arxiv[{window_label}]: {len(papers)} results",
                            flush=True,
                        )
                    except Exception as exc:
                        errors.append(f"arxiv:{query}:{window_label}:{exc}")
                        print(f"  arxiv[{window_label}] error: {exc}", flush=True)
            if openalex:
                try:
                    papers = openalex.search(query, limit=args.openalex_limit)
                    batches.append(("openalex", papers))
                    print(f"  openalex: {len(papers)} results", flush=True)
                except Exception as exc:
                    errors.append(f"openalex:{query}:{exc}")
                    print(f"  openalex error: {exc}", flush=True)
            if semantic:
                try:
                    papers = semantic.search(query, limit=args.semantic_limit)
                    batches.append(("semantic_scholar", papers))
                    print(f"  semantic_scholar: {len(papers)} results", flush=True)
                except Exception as exc:
                    errors.append(f"semantic_scholar:{query}:{exc}")
                    print(f"  semantic_scholar error: {exc}", flush=True)

            for _, papers in batches:
                for paper in papers:
                    fetched += 1
                    scored = score_paper(paper)
                    if args.include_rejected or scored.relevance_score >= args.min_score:
                        db.upsert_paper(scored)
                        if scored.relevance_score >= args.min_score:
                            accepted += 1

        exported = db.export_jsonl(args.jsonl_path, min_score=args.min_score)
        summary = db.summary()
        print("\nCollection summary")
        print(f"  fetched_records: {fetched}")
        print(f"  accepted_records_before_dedup: {accepted}")
        print(f"  db_total_rows: {summary['total']}")
        print(f"  db_score_gte_4: {summary['score_gte_4']}")
        print(f"  exported_jsonl_rows: {exported}")
        print(f"  date_from: {date_from}")
        print(f"  date_to: {date_to}")
        print(f"  db_path: {args.db_path}")
        print(f"  jsonl_path: {args.jsonl_path}")
        if errors:
            print("\nErrors")
            for error in errors:
                print(f"  - {error}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
