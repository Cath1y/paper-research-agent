#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sqlite3
import sys
import urllib.error
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from embodiedai_kb.data_collection.scorer import score_paper
from embodiedai_kb.data_collection.topconf_clients import (
    AAAIOAIClient,
    CrossrefAAAIClient,
    CrossrefIROSClient,
    CVFClient,
    ECVAClient,
    HuggingFaceICRAClient,
    IJCAIClient,
    NeurIPSClient,
    OpenReviewClient,
    PaperceptIROSClient,
    PMLRClient,
    RSSClient,
)
from embodiedai_kb.storage.database import PaperDatabase
from embodiedai_kb.storage.schemas import PaperMetadata


DEFAULT_SOURCES = (
    "cvf",
    "aaai_oai",
    "crossref_aaai",
    "crossref_iros",
    "ecva",
    "hf_icra",
    "ijcai",
    "pmlr",
    "neurips",
    "openreview",
    "papercept_iros",
    "rss",
)
DEFAULT_YEARS = (2023, 2024, 2025)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect official top-conference proceedings metadata into a separate "
            "SQLite database. This script intentionally does not read or write the "
            "arXiv metadata database."
        )
    )
    parser.add_argument(
        "--db-path",
        default="data/db/topconf_papers.sqlite",
        help="SQLite database path for top-conference papers only.",
    )
    parser.add_argument(
        "--all-jsonl-path",
        default="data/metadata/topconf_papers_all.jsonl",
        help="JSONL export path for all collected top-conference papers.",
    )
    parser.add_argument(
        "--relevant-jsonl-path",
        default="data/metadata/topconf_papers_score_gte_4.jsonl",
        help="JSONL export path for top-conference papers with score >= min-score.",
    )
    parser.add_argument(
        "--sources",
        default=",".join(DEFAULT_SOURCES),
        help=(
            "Comma-separated official proceeding sources: "
            "aaai_oai,crossref_aaai,cvf,crossref_iros,ecva,hf_icra,ijcai,pmlr,neurips,openreview,papercept_iros,rss."
        ),
    )
    parser.add_argument(
        "--years",
        default=",".join(str(year) for year in DEFAULT_YEARS),
        help="Comma-separated years to collect.",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=4,
        help="Minimum deterministic score for the relevant JSONL export.",
    )
    parser.add_argument(
        "--request-delay",
        type=float,
        default=1.0,
        help="Seconds to wait between requests for clients that support throttling.",
    )
    return parser.parse_args()


def parse_sources(value: str) -> list[str]:
    sources = [source.strip().lower() for source in value.split(",") if source.strip()]
    unknown = sorted(set(sources) - set(DEFAULT_SOURCES))
    if unknown:
        raise ValueError(
            f"Unknown top-conference source(s): {', '.join(unknown)}. "
            f"Allowed: {', '.join(DEFAULT_SOURCES)}"
        )
    return sources


def parse_years(value: str) -> list[int]:
    years: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        years.append(int(item))
    return sorted(set(years))


def collect_source(
    source: str, years: list[int], request_delay: float
) -> tuple[list[PaperMetadata], list[str]]:
    papers: list[PaperMetadata] = []
    errors: list[str] = []
    if source == "cvf":
        client = CVFClient(request_delay=request_delay)
        for year in years:
            for conf in client._confs_for_year(year):
                label = f"{conf} {year}"
                print(f"  fetching {label} from CVF...", flush=True)
                try:
                    batch = client.collect_conference(conf, year)
                except urllib.error.HTTPError as exc:
                    if exc.code == 404:
                        print(f"    skipped: 404", flush=True)
                        continue
                    errors.append(f"cvf:{label}:{exc}")
                    print(f"    error: {exc}", flush=True)
                    continue
                except (TimeoutError, urllib.error.URLError) as exc:
                    errors.append(f"cvf:{label}:{exc}")
                    print(f"    error: {exc}", flush=True)
                    continue
                papers.extend(batch)
                print(f"    records: {len(batch)}", flush=True)
        return papers, errors
    if source == "aaai_oai":
        client = AAAIOAIClient(request_delay=request_delay)
        print("  fetching AAAI OAI-PMH records...", flush=True)
        try:
            papers = client.collect(years)
        except (TimeoutError, urllib.error.URLError) as exc:
            return [], [f"aaai_oai:{exc}"]
        with_abstract = sum(1 for paper in papers if paper.abstract)
        print(f"    records: {len(papers)} with_abstract: {with_abstract}", flush=True)
        return papers, errors
    if source == "crossref_aaai":
        client = CrossrefAAAIClient(request_delay=request_delay)
        for year in years:
            print(f"  fetching AAAI {year} from Crossref/OpenAlex...", flush=True)
            try:
                batch = client.collect_year(year)
            except (TimeoutError, urllib.error.URLError) as exc:
                errors.append(f"crossref_aaai:{year}:{exc}")
                print(f"    error: {exc}", flush=True)
                continue
            papers.extend(batch)
            with_abstract = sum(1 for paper in batch if paper.abstract)
            print(
                f"    records: {len(batch)} with_abstract: {with_abstract}",
                flush=True,
            )
        return papers, errors
    if source == "crossref_iros":
        client = CrossrefIROSClient(request_delay=request_delay)
        for year in years:
            print(f"  fetching IROS {year} from Crossref/OpenAlex...", flush=True)
            try:
                batch = client.collect_year(year)
            except (TimeoutError, urllib.error.URLError) as exc:
                errors.append(f"crossref_iros:{year}:{exc}")
                print(f"    error: {exc}", flush=True)
                continue
            papers.extend(batch)
            with_abstract = sum(1 for paper in batch if paper.abstract)
            print(
                f"    records: {len(batch)} with_abstract: {with_abstract}",
                flush=True,
            )
        return papers, errors
    if source == "ijcai":
        client = IJCAIClient(request_delay=request_delay)
        for year in years:
            print(f"  fetching IJCAI {year} from official proceedings...", flush=True)
            try:
                batch = client.collect_year(year)
            except (TimeoutError, urllib.error.URLError) as exc:
                errors.append(f"ijcai:{year}:{exc}")
                print(f"    error: {exc}", flush=True)
                continue
            papers.extend(batch)
            with_abstract = sum(1 for paper in batch if paper.abstract)
            print(
                f"    records: {len(batch)} with_abstract: {with_abstract}",
                flush=True,
            )
        return papers, errors
    if source == "pmlr":
        client = PMLRClient(request_delay=request_delay)
        print("  discovering PMLR volumes...", flush=True)
        try:
            volumes = client.discover_volumes(years)
        except (TimeoutError, urllib.error.URLError) as exc:
            return papers, [f"pmlr:index:{exc}"]
        for conf, year, volume in volumes:
            label = f"{conf} {year} ({volume})"
            print(f"  fetching {label} from PMLR...", flush=True)
            try:
                batch = client.collect_volume(conf, year, volume)
            except urllib.error.HTTPError as exc:
                if exc.code == 404:
                    print(f"    skipped: 404", flush=True)
                    continue
                errors.append(f"pmlr:{label}:{exc}")
                print(f"    error: {exc}", flush=True)
                continue
            except (TimeoutError, urllib.error.URLError) as exc:
                errors.append(f"pmlr:{label}:{exc}")
                print(f"    error: {exc}", flush=True)
                continue
            papers.extend(batch)
            print(f"    records: {len(batch)}", flush=True)
        return papers, errors
    if source == "ecva":
        client = ECVAClient()
        for year in years:
            if year % 2 == 1:
                print(f"  skipping ECCV {year}: ECCV is biennial", flush=True)
                continue
            print(f"  fetching ECCV {year} from ECVA...", flush=True)
            try:
                batch = client.collect_year(year)
            except (TimeoutError, urllib.error.URLError) as exc:
                errors.append(f"ecva:{year}:{exc}")
                print(f"    error: {exc}", flush=True)
                continue
            papers.extend(batch)
            print(f"    records: {len(batch)}", flush=True)
        return papers, errors
    if source == "neurips":
        client = NeurIPSClient()
        for year in years:
            print(f"  fetching NeurIPS {year}...", flush=True)
            try:
                batch = client.collect_year(year)
            except urllib.error.HTTPError as exc:
                if exc.code == 404:
                    print(f"    skipped: 404", flush=True)
                    continue
                errors.append(f"neurips:{year}:{exc}")
                print(f"    error: {exc}", flush=True)
                continue
            except (TimeoutError, urllib.error.URLError) as exc:
                errors.append(f"neurips:{year}:{exc}")
                print(f"    error: {exc}", flush=True)
                continue
            papers.extend(batch)
            print(f"    records: {len(batch)}", flush=True)
        return papers, errors
    if source == "hf_icra":
        client = HuggingFaceICRAClient(request_delay=request_delay)
        for year in years:
            print(f"  fetching ICRA {year} from ai-conferences/Hugging Face...", flush=True)
            try:
                batch = client.collect_year(year)
            except (TimeoutError, urllib.error.URLError) as exc:
                errors.append(f"hf_icra:{year}:{exc}")
                print(f"    error: {exc}", flush=True)
                continue
            papers.extend(batch)
            with_abstract = sum(1 for paper in batch if paper.abstract)
            print(
                f"    records: {len(batch)} abstracts_from_arxiv: {with_abstract}",
                flush=True,
            )
        return papers, errors
    if source == "openreview":
        client = OpenReviewClient(request_delay=request_delay)
        for year in years:
            print(f"  fetching ICLR {year} from OpenReview...", flush=True)
            try:
                batch = client.collect_iclr_year(year)
            except (TimeoutError, urllib.error.URLError) as exc:
                errors.append(f"openreview:iclr:{year}:{exc}")
                print(f"    error: {exc}", flush=True)
                continue
            papers.extend(batch)
            print(f"    accepted records: {len(batch)}", flush=True)
        return papers, errors
    if source == "papercept_iros":
        client = PaperceptIROSClient(request_delay=request_delay)
        for year in years:
            print(f"  fetching IROS {year} from Papercept...", flush=True)
            try:
                batch = client.collect_year(year)
            except urllib.error.HTTPError as exc:
                if exc.code == 403:
                    print(f"    skipped: 403", flush=True)
                    continue
                errors.append(f"papercept_iros:{year}:{exc}")
                print(f"    error: {exc}", flush=True)
                continue
            except (TimeoutError, urllib.error.URLError) as exc:
                errors.append(f"papercept_iros:{year}:{exc}")
                print(f"    error: {exc}", flush=True)
                continue
            papers.extend(batch)
            with_abstract = sum(1 for paper in batch if paper.abstract)
            print(
                f"    records: {len(batch)} with_abstract: {with_abstract}",
                flush=True,
            )
        return papers, errors
    if source == "rss":
        client = RSSClient(request_delay=request_delay)
        for year in years:
            rss_number = client.year_to_rss_number.get(year)
            if rss_number is None and year == 2026:
                print(f"  fetching RSS {year} accepted papers...", flush=True)
            elif rss_number is None:
                print(f"  skipping RSS {year}: no proceedings mapping", flush=True)
                continue
            else:
                print(f"  fetching RSS {year} (rss{rss_number})...", flush=True)
            try:
                batch = (
                    client.collect_accepted_year(year)
                    if rss_number is None
                    else client.collect_year(year, rss_number)
                )
            except urllib.error.HTTPError as exc:
                if exc.code == 404:
                    print(f"    skipped: 404", flush=True)
                    continue
                errors.append(f"rss:{year}:{exc}")
                print(f"    error: {exc}", flush=True)
                continue
            except (TimeoutError, urllib.error.URLError) as exc:
                errors.append(f"rss:{year}:{exc}")
                print(f"    error: {exc}", flush=True)
                continue
            papers.extend(batch)
            with_abstract = sum(1 for paper in batch if paper.abstract)
            print(
                f"    records: {len(batch)} with_abstract: {with_abstract}",
                flush=True,
            )
        return papers, errors
    raise ValueError(f"Unsupported source: {source}")


def count_rows(db_path: str | Path, where: str = "1=1") -> int:
    conn = sqlite3.connect(db_path)
    try:
        return int(conn.execute(f"SELECT COUNT(*) FROM papers WHERE {where}").fetchone()[0])
    finally:
        conn.close()


def grouped_counts(db_path: str | Path, group_column: str) -> list[tuple[str, int]]:
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            f"""
            SELECT COALESCE({group_column}, 'unknown') AS key, COUNT(*) AS n
            FROM papers
            GROUP BY key
            ORDER BY n DESC, key ASC
            """
        ).fetchall()
        return [(str(row[0]), int(row[1])) for row in rows]
    finally:
        conn.close()


def print_grouped(label: str, rows: list[tuple[str, int]], limit: int = 30) -> None:
    print(label)
    for key, count in rows[:limit]:
        print(f"  {key}: {count}")
    if len(rows) > limit:
        print(f"  ... {len(rows) - limit} more")


def main() -> None:
    args = parse_args()
    sources = parse_sources(args.sources)
    years = parse_years(args.years)
    db = PaperDatabase(args.db_path)

    fetched_by_source: Counter[str] = Counter()
    errors: list[str] = []

    try:
        for source in sources:
            print(f"[topconf] source={source} years={years}", flush=True)
            papers, source_errors = collect_source(source, years, args.request_delay)
            errors.extend(source_errors)
            fetched_by_source[source] += len(papers)
            print(f"  fetched_records: {len(papers)}", flush=True)

            for paper in papers:
                scored = score_paper(paper)
                if "topconf" not in scored.sources:
                    scored.sources = ["topconf", *scored.sources]
                db.upsert_paper(scored)

        exported_all = db.export_jsonl(args.all_jsonl_path, min_score=0)
        exported_relevant = db.export_jsonl(
            args.relevant_jsonl_path, min_score=args.min_score
        )

        print("\nTop-conference collection summary")
        print(f"  db_path: {args.db_path}")
        print(f"  years: {years}")
        print(f"  sources: {sources}")
        print(f"  fetched_by_source: {dict(sorted(fetched_by_source.items()))}")
        print(f"  db_total_rows: {count_rows(args.db_path)}")
        print(
            f"  db_score_gte_{args.min_score:g}: "
            f"{count_rows(args.db_path, f'relevance_score >= {args.min_score}')}"
        )
        print(f"  exported_all_jsonl_rows: {exported_all}")
        print(f"  all_jsonl_path: {args.all_jsonl_path}")
        print(f"  exported_relevant_jsonl_rows: {exported_relevant}")
        print(f"  relevant_jsonl_path: {args.relevant_jsonl_path}")
        print_grouped("  by_venue:", grouped_counts(args.db_path, "venue"))
        print_grouped("  by_year:", grouped_counts(args.db_path, "year"))
        if errors:
            print("\nErrors")
            for error in errors:
                print(f"  - {error}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
