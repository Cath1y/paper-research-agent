#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from textwrap import shorten

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from embodiedai_kb.search.metadata_search import (
    DEFAULT_ARXIV_DB,
    DEFAULT_FRONTIER_DB,
    DEFAULT_TOPCONF_DB,
    MetadataSearchEngine,
    SearchCorpus,
    SearchFilters,
    score_distribution,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Search paper metadata with SQLite FTS5. This is the first-stage "
            "candidate retriever before PDF/PaperQA reranking."
        )
    )
    parser.add_argument("query", help="Natural-language search query.")
    parser.add_argument(
        "--candidate-k",
        "--top-k",
        type=int,
        default=30,
        help="Number of first-stage metadata candidates to return.",
    )
    parser.add_argument(
        "--topconf-db",
        type=Path,
        default=DEFAULT_TOPCONF_DB,
        help="Top-conference SQLite metadata DB.",
    )
    parser.add_argument(
        "--arxiv-db",
        type=Path,
        default=DEFAULT_ARXIV_DB,
        help="arXiv SQLite metadata DB, used only with --include-arxiv.",
    )
    parser.add_argument(
        "--frontier-db",
        type=Path,
        default=DEFAULT_FRONTIER_DB,
        help="Frontier/high-signal SQLite metadata DB, used only with --include-frontier.",
    )
    parser.add_argument(
        "--include-arxiv",
        action="store_true",
        help="Also search the separate arXiv metadata DB.",
    )
    parser.add_argument(
        "--include-frontier",
        action="store_true",
        help="Also search the separate 2026 frontier/high-signal metadata DB.",
    )
    parser.add_argument("--min-score", type=float, default=4.0)
    parser.add_argument("--year-from", type=int)
    parser.add_argument("--year-to", type=int)
    parser.add_argument(
        "--venues",
        help="Comma-separated venue prefixes, e.g. CVPR,ICLR,AAAI,IJCAI.",
    )
    parser.add_argument(
        "--allow-missing-abstract",
        action="store_true",
        help="Include papers without abstracts.",
    )
    parser.add_argument(
        "--require-pdf",
        action="store_true",
        help="Only return records with a known PDF URL.",
    )
    parser.add_argument(
        "--no-dedupe",
        action="store_true",
        help="Do not deduplicate by DOI/arXiv/title across corpora.",
    )
    parser.add_argument(
        "--jsonl",
        type=Path,
        help="Optional path to write full search results as JSONL.",
    )
    parser.add_argument(
        "--format",
        choices=("table", "json"),
        default="table",
        help="Print format.",
    )
    return parser.parse_args()


def parse_venues(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(item.strip() for item in value.split(",") if item.strip())


def print_table(results: list) -> None:
    if not results:
        print("No results.")
        return
    header = f"{'#':>2} {'score':>7} {'year':>4} {'corpus':<9} {'venue':<12} {'pdf':<3} title"
    print(header)
    print("-" * len(header))
    for result in results:
        pdf = "yes" if result.pdf_url else "no"
        venue = shorten(result.venue or "-", width=12, placeholder="..")
        title = shorten(result.title, width=96, placeholder="...")
        print(
            f"{result.rank:>2} {result.hybrid_score:>7.2f} "
            f"{str(result.year or '-'):>4} {result.corpus:<9} {venue:<12} {pdf:<3} {title}"
        )
        terms = ", ".join(result.matched_terms[:8]) or "-"
        print(
            f"   id={result.paper_id} rel={result.relevance_score:.2f} "
            f"frontier={result.frontier_score:.2f} cites={result.citation_count or 0} terms={terms}"
        )
        if result.quality_signals:
            print(f"   signals={', '.join(result.quality_signals[:6])}")


def write_jsonl(path: Path, results: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    corpora = [SearchCorpus("topconf", args.topconf_db)]
    if args.include_frontier:
        corpora.append(SearchCorpus("frontier", args.frontier_db))
    if args.include_arxiv:
        corpora.append(SearchCorpus("arxiv", args.arxiv_db))
    filters = SearchFilters(
        min_score=args.min_score,
        year_from=args.year_from,
        year_to=args.year_to,
        venues=parse_venues(args.venues),
        require_abstract=not args.allow_missing_abstract,
        require_pdf=args.require_pdf,
    )

    with MetadataSearchEngine(corpora=corpora, filters=filters) as engine:
        results = engine.search(
            args.query,
            candidate_k=args.candidate_k,
            dedupe=not args.no_dedupe,
        )

    if args.jsonl:
        write_jsonl(args.jsonl, results)

    if args.format == "json":
        print(
            json.dumps(
                {
                    "query": args.query,
                    "candidate_k": args.candidate_k,
                    "distribution": score_distribution(results),
                    "results": [result.to_dict() for result in results],
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print_table(results)
        print()
        print(f"Returned {len(results)} metadata candidates.")
        if args.jsonl:
            print(f"Wrote JSONL: {args.jsonl}")


if __name__ == "__main__":
    main()
