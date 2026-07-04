#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from embodiedai_kb.data_collection.scorer import score_paper
from embodiedai_kb.storage.database import PaperDatabase
from embodiedai_kb.storage.schemas import PaperMetadata, now_iso


USER_AGENT = "EmbodiedAI-KB/0.1 (top conference abstract enrichment)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Enrich top-conference metadata with abstracts from official detail "
            "pages, then rescore and re-export JSONL files."
        )
    )
    parser.add_argument(
        "--db-path",
        default="data/db/topconf_papers.sqlite",
        help="Top-conference SQLite database path.",
    )
    parser.add_argument(
        "--all-jsonl-path",
        default="data/metadata/topconf_papers_all.jsonl",
        help="JSONL export path for all top-conference papers.",
    )
    parser.add_argument(
        "--relevant-jsonl-path",
        default="data/metadata/topconf_papers_score_gte_4.jsonl",
        help="JSONL export path for papers with score >= export-min-score.",
    )
    parser.add_argument(
        "--selection-min-score",
        type=float,
        default=1.0,
        help="Enrich missing-abstract papers above this current score.",
    )
    parser.add_argument(
        "--export-min-score",
        type=float,
        default=4.0,
        help="Minimum score for the relevant JSONL export.",
    )
    parser.add_argument(
        "--all-missing",
        action="store_true",
        help="Try to enrich every missing abstract in the database.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum number of papers to attempt.",
    )
    parser.add_argument(
        "--request-delay",
        type=float,
        default=0.2,
        help="Seconds to wait between detail-page requests.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="HTTP timeout per detail page.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Print progress every N attempts.",
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


def get_url(url: str, timeout: int) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", "ignore")


def clean_html(value: str) -> str:
    value = re.sub(r"<script\b.*?</script>", " ", value, flags=re.DOTALL | re.I)
    value = re.sub(r"<style\b.*?</style>", " ", value, flags=re.DOTALL | re.I)
    value = re.sub(r"<[^>]+>", " ", value)
    value = html.unescape(value)
    value = value.replace("\x00", " ")
    value = re.sub(r"\s+", " ", value).strip()
    if len(value) >= 2 and value[0] == value[-1] == '"':
        value = value[1:-1].strip()
    return value


def extract_abstract(page: str) -> str | None:
    patterns = (
        r'<div\s+id=["\']abstract["\'][^>]*>(.*?)</div>',
        r'<p\s+class=["\']paper-abstract["\'][^>]*>(.*?)</section>',
        r"<b>\s*Abstract:\s*</b>\s*</p>\s*<p[^>]*>(.*?)</p>",
        r"<h4>\s*Abstract\s*</h4>\s*<div[^>]*>(.*?)</div>",
    )
    for pattern in patterns:
        match = re.search(pattern, page, flags=re.DOTALL | re.I)
        if not match:
            continue
        abstract = clean_html(match.group(1))
        if len(abstract) >= 80:
            return abstract

    meta_patterns = (
        r'<meta\s+name=["\']citation_abstract["\']\s+content=["\'](.*?)["\']',
        r'<meta\s+property=["\']og:description["\']\s+content=["\'](.*?)["\']',
        r'<meta\s+name=["\']twitter:description["\']\s+content=["\'](.*?)["\']',
    )
    for pattern in meta_patterns:
        match = re.search(pattern, page, flags=re.DOTALL | re.I)
        if not match:
            continue
        abstract = clean_html(match.group(1))
        if len(abstract) >= 120 and not abstract.endswith("..."):
            return abstract
    return None


def row_to_paper(row: Any, abstract: str) -> PaperMetadata:
    return PaperMetadata(
        paper_id=row["paper_id"],
        title=row["title"],
        authors=json_loads(row["authors"], []),
        year=row["year"],
        venue=row["venue"],
        abstract=abstract,
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
        updated_at=now_iso(),
    )


def select_rows(db: PaperDatabase, args: argparse.Namespace) -> list[Any]:
    missing = "(abstract IS NULL OR length(trim(abstract)) = 0)"
    if args.all_missing:
        where = f"{missing} AND paper_url IS NOT NULL"
        params: tuple[Any, ...] = ()
    else:
        where = (
            f"{missing} AND paper_url IS NOT NULL AND ("
            "relevance_score >= ? OR venue LIKE 'CoRL%' OR venue LIKE 'RSS%'"
            ")"
        )
        params = (args.selection_min_score,)
    limit_sql = " LIMIT ?" if args.limit else ""
    if args.limit:
        params = (*params, args.limit)
    return db.conn.execute(
        f"""
        SELECT *
        FROM papers
        WHERE {where}
        ORDER BY
            CASE
                WHEN venue LIKE 'CoRL%' THEN 0
                WHEN venue LIKE 'RSS%' THEN 1
                ELSE 2
            END,
            relevance_score DESC,
            year DESC
        {limit_sql}
        """,
        params,
    ).fetchall()


def update_paper(db: PaperDatabase, paper: PaperMetadata) -> None:
    db.conn.execute(
        """
        UPDATE papers SET
            abstract = ?,
            relevance_score = ?,
            relevance_reasons = ?,
            decision = ?,
            categories = ?,
            keywords = ?,
            updated_at = ?
        WHERE paper_id = ?
        """,
        (
            paper.abstract,
            paper.relevance_score,
            json_dumps(paper.relevance_reasons),
            paper.decision,
            json_dumps(paper.categories),
            json_dumps(paper.keywords),
            paper.updated_at,
            paper.paper_id,
        ),
    )


def main() -> None:
    args = parse_args()
    db = PaperDatabase(args.db_path)
    rows = select_rows(db, args)
    print(f"Selected missing-abstract papers: {len(rows)}", flush=True)

    attempted = 0
    enriched = 0
    failed = 0
    last_request = 0.0
    failures: list[str] = []

    try:
        for row in rows:
            attempted += 1
            elapsed = time.monotonic() - last_request
            if elapsed < args.request_delay:
                time.sleep(args.request_delay - elapsed)
            last_request = time.monotonic()

            try:
                page = get_url(row["paper_url"], timeout=args.timeout)
                abstract = extract_abstract(page)
            except (TimeoutError, urllib.error.URLError, UnicodeError) as exc:
                failed += 1
                failures.append(f"{row['paper_id']}: {exc}")
                abstract = None

            if abstract:
                scored = score_paper(row_to_paper(row, abstract))
                update_paper(db, scored)
                enriched += 1
            else:
                failed += 1

            if attempted % args.progress_every == 0:
                db.conn.commit()
                print(
                    f"  attempted={attempted} enriched={enriched} failed={failed}",
                    flush=True,
                )

        db.conn.commit()
        exported_all = db.export_jsonl(args.all_jsonl_path, min_score=0)
        exported_relevant = db.export_jsonl(
            args.relevant_jsonl_path, min_score=args.export_min_score
        )
        score_gte = db.conn.execute(
            "SELECT COUNT(*) FROM papers WHERE relevance_score >= ?",
            (args.export_min_score,),
        ).fetchone()[0]
        with_abs = db.conn.execute(
            """
            SELECT COUNT(*)
            FROM papers
            WHERE abstract IS NOT NULL AND length(trim(abstract)) > 0
            """
        ).fetchone()[0]
        print("\nEnrichment summary")
        print(f"  attempted: {attempted}")
        print(f"  enriched: {enriched}")
        print(f"  failed_or_not_found: {failed}")
        print(f"  db_with_abstract: {with_abs}")
        print(f"  db_score_gte_{args.export_min_score:g}: {score_gte}")
        print(f"  exported_all_jsonl_rows: {exported_all}")
        print(f"  exported_relevant_jsonl_rows: {exported_relevant}")
        if failures:
            print("\nFirst failures")
            for failure in failures[:10]:
                print(f"  - {failure}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
