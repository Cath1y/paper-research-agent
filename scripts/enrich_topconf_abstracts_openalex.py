#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from embodiedai_kb.data_collection.scorer import score_paper
from embodiedai_kb.storage.database import PaperDatabase
from embodiedai_kb.storage.schemas import PaperMetadata, now_iso


OPENALEX_WORKS_URL = "https://api.openalex.org/works"
USER_AGENT = "EmbodiedAI-KB/0.1 (top conference abstract enrichment via OpenAlex)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enrich missing top-conference abstracts via OpenAlex DOI lookup."
    )
    parser.add_argument("--db-path", default="data/db/topconf_papers.sqlite")
    parser.add_argument(
        "--all-jsonl-path", default="data/metadata/topconf_papers_all.jsonl"
    )
    parser.add_argument(
        "--relevant-jsonl-path",
        default="data/metadata/topconf_papers_score_gte_4.jsonl",
    )
    parser.add_argument(
        "--selection-min-score",
        type=float,
        default=4.0,
        help="Only enrich missing-abstract papers at or above this current score.",
    )
    parser.add_argument("--export-min-score", type=float, default=4.0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--request-delay", type=float, default=1.0)
    parser.add_argument("--progress-every", type=int, default=10)
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


def normalize_doi(doi: str | None) -> str | None:
    if not doi:
        return None
    doi = doi.strip()
    doi = re.sub(r"^https?://(dx\.)?doi\.org/", "", doi, flags=re.I)
    return doi or None


def abstract_from_inverted_index(index: dict[str, list[int]] | None) -> str | None:
    if not index:
        return None
    words: list[tuple[int, str]] = []
    for word, positions in index.items():
        for position in positions:
            words.append((int(position), word))
    abstract = " ".join(word for _, word in sorted(words))
    return abstract if len(abstract) >= 80 else None


def openalex_by_doi(doi: str) -> dict[str, Any] | None:
    quoted = urllib.parse.quote(f"https://doi.org/{doi}", safe="")
    url = f"{OPENALEX_WORKS_URL}/{quoted}"
    last_error: Exception | None = None
    for attempt in range(4):
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code == 404:
                return None
            if exc.code not in {429, 503} or attempt == 3:
                raise
            retry_after = exc.headers.get("Retry-After")
            try:
                delay = float(retry_after) if retry_after else 10.0 * (attempt + 1)
            except ValueError:
                delay = 10.0 * (attempt + 1)
            time.sleep(delay)
        except urllib.error.URLError as exc:
            last_error = exc
            if attempt == 3:
                raise
            time.sleep(5.0 * (attempt + 1))
    if last_error:
        raise last_error
    return None


def row_to_paper(row: Any, abstract: str, item: dict[str, Any]) -> PaperMetadata:
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
    raw = json_loads(row["raw_metadata"], {})
    enrichments = raw.get("abstract_enrichments", [])
    if not isinstance(enrichments, list):
        enrichments = [enrichments]
    enrichments.append(
        {
            "source": "openalex",
            "openalex_id": item.get("id"),
            "doi": normalize_doi(item.get("doi")) or row["doi"],
        }
    )
    raw["abstract_enrichments"] = enrichments

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
        citation_count=item.get("cited_by_count") or row["citation_count"],
        influential_citation_count=row["influential_citation_count"],
        fields_of_study=sorted(
            set(json_loads(row["fields_of_study"], []) + concepts + topics)
        ),
        keywords=json_loads(row["keywords"], []),
        categories=json_loads(row["categories"], []),
        sources=json_loads(row["sources"], []),
        source_queries=json_loads(row["source_queries"], []),
        relevance_score=float(row["relevance_score"] or 0),
        relevance_reasons=json_loads(row["relevance_reasons"], []),
        decision=row["decision"],
        raw_metadata=raw,
        collected_at=row["collected_at"],
        updated_at=now_iso(),
    )


def select_rows(db: PaperDatabase, args: argparse.Namespace) -> list[Any]:
    limit_sql = " LIMIT ?" if args.limit else ""
    params: tuple[Any, ...] = (args.selection_min_score,)
    if args.limit:
        params = (*params, args.limit)
    return db.conn.execute(
        f"""
        SELECT *
        FROM papers
        WHERE relevance_score >= ?
          AND (abstract IS NULL OR length(trim(abstract)) = 0)
          AND doi IS NOT NULL
          AND length(trim(doi)) > 0
        ORDER BY relevance_score DESC, year DESC
        {limit_sql}
        """,
        params,
    ).fetchall()


def update_paper(db: PaperDatabase, paper: PaperMetadata) -> None:
    db.conn.execute(
        """
        UPDATE papers SET
            abstract = ?,
            citation_count = ?,
            fields_of_study = ?,
            categories = ?,
            keywords = ?,
            relevance_score = ?,
            relevance_reasons = ?,
            decision = ?,
            raw_metadata = ?,
            updated_at = ?
        WHERE paper_id = ?
        """,
        (
            paper.abstract,
            paper.citation_count,
            json_dumps(paper.fields_of_study),
            json_dumps(paper.categories),
            json_dumps(paper.keywords),
            paper.relevance_score,
            json_dumps(paper.relevance_reasons),
            paper.decision,
            json_dumps(paper.raw_metadata),
            paper.updated_at,
            paper.paper_id,
        ),
    )


def main() -> None:
    args = parse_args()
    db = PaperDatabase(args.db_path)
    rows = select_rows(db, args)
    print(f"Selected DOI-backed missing abstracts: {len(rows)}", flush=True)

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
            doi = normalize_doi(row["doi"])
            if not doi:
                failed += 1
                continue
            try:
                item = openalex_by_doi(doi)
                abstract = abstract_from_inverted_index(
                    item.get("abstract_inverted_index") if item else None
                )
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                item = None
                abstract = None
                failures.append(f"{row['paper_id']}: {exc}")
            if abstract and item:
                scored = score_paper(row_to_paper(row, abstract, item))
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
        missing_relevant = db.conn.execute(
            """
            SELECT COUNT(*)
            FROM papers
            WHERE relevance_score >= ?
              AND (abstract IS NULL OR length(trim(abstract)) = 0)
            """,
            (args.export_min_score,),
        ).fetchone()[0]
        print("\nOpenAlex enrichment summary")
        print(f"  attempted: {attempted}")
        print(f"  enriched: {enriched}")
        print(f"  failed_or_not_found: {failed}")
        print(f"  db_score_gte_{args.export_min_score:g}: {score_gte}")
        print(f"  missing_abstract_score_gte_{args.export_min_score:g}: {missing_relevant}")
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
