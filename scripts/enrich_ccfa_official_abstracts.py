#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import html
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from embodiedai_kb.data_collection.scorer import score_paper
from embodiedai_kb.storage.database import PaperDatabase
from embodiedai_kb.storage.schemas import PaperMetadata, now_iso


USER_AGENT = "EmbodiedAI-KB/0.1 (CCF-A official abstract enrichment)"
AAAI_OAI_URL = "https://ojs.aaai.org/index.php/AAAI/oai"
OAI_NS = {
    "oai": "http://www.openarchives.org/OAI/2.0/",
    "dc": "http://purl.org/dc/elements/1.1/",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Enrich missing abstracts for CCF-A AAAI/IJCAI papers from official "
            "conference pages, then rescore and re-export the top-conference JSONLs."
        )
    )
    parser.add_argument("--db-path", default="data/db/topconf_papers.sqlite")
    parser.add_argument(
        "--all-jsonl-path", default="data/metadata/topconf_papers_all.jsonl"
    )
    parser.add_argument(
        "--relevant-jsonl-path",
        default="data/metadata/topconf_papers_score_gte_4.jsonl",
    )
    parser.add_argument("--selection-min-score", type=float, default=4.0)
    parser.add_argument("--export-min-score", type=float, default=4.0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--request-delay", type=float, default=0.5)
    parser.add_argument("--timeout", type=int, default=30)
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


def get_url(url: str, timeout: int) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = response.read()
        encoding = response.headers.get("Content-Encoding", "").lower()
        if encoding == "gzip" or payload.startswith(b"\x1f\x8b"):
            payload = gzip.decompress(payload)
        return payload.decode("utf-8", "ignore")


def clean_html(value: str) -> str:
    value = re.sub(r"<script\b.*?</script>", " ", value, flags=re.DOTALL | re.I)
    value = re.sub(r"<style\b.*?</style>", " ", value, flags=re.DOTALL | re.I)
    value = re.sub(r"<[^>]+>", " ", value)
    value = html.unescape(value)
    value = value.replace("\x00", " ")
    return re.sub(r"\s+", " ", value).strip().strip('"')


def aaai_article_id(row: Any) -> str | None:
    candidates = [row["doi"], row["paper_url"], row["pdf_url"]]
    for value in candidates:
        if not value:
            continue
        article_match = re.search(r"/article/(?:view|download)/(\d+)", str(value))
        if article_match:
            return article_match.group(1)
        doi_match = re.search(r"\.(\d+)$", str(value).strip())
        if doi_match and "aaai." in str(value).lower():
            return doi_match.group(1)
    return None


def extract_aaai_abstract(row: Any, timeout: int) -> tuple[str | None, str | None]:
    article_id = aaai_article_id(row)
    if not article_id:
        return None, None
    params = {
        "verb": "GetRecord",
        "metadataPrefix": "oai_dc",
        "identifier": f"oai:ojs.aaai.org:article/{article_id}",
    }
    url = f"{AAAI_OAI_URL}?{urllib.parse.urlencode(params)}"
    page = get_url(url, timeout=timeout)
    root = ET.fromstring(page)
    descriptions = [
        elem.text.strip()
        for elem in root.findall(".//dc:description", OAI_NS)
        if elem.text and elem.text.strip()
    ]
    for description in descriptions:
        abstract = clean_html(description)
        if len(abstract) >= 80:
            return abstract, url
    return None, url


def extract_ijcai_abstract(row: Any, timeout: int) -> tuple[str | None, str | None]:
    url = row["paper_url"]
    if not url:
        return None, None
    page = get_url(url, timeout=timeout)
    patterns = (
        r'<hr>\s*<div class="row">\s*<div class="col-md-12">\s*'
        r"(.*?)\s*</div>\s*<div class=\"col-md-12\">\s*<div class=\"keywords\">",
        r'<meta\s+name=["\']description["\']\s+content=["\'](.*?)["\']',
    )
    for pattern in patterns:
        match = re.search(pattern, page, flags=re.DOTALL | re.I)
        if not match:
            continue
        abstract = clean_html(match.group(1))
        if len(abstract) >= 80:
            return abstract, url
    return None, url


def row_to_paper(row: Any, abstract: str, source: str, url: str | None) -> PaperMetadata:
    raw = json_loads(row["raw_metadata"], {})
    enrichments = raw.get("abstract_enrichments", [])
    if not isinstance(enrichments, list):
        enrichments = [enrichments]
    enrichments.append({"source": source, "url": url})
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
          AND (venue LIKE 'AAAI%' OR venue LIKE 'IJCAI%')
        ORDER BY relevance_score DESC, year DESC, venue
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
            raw_metadata = ?,
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
            json_dumps(paper.raw_metadata),
            paper.updated_at,
            paper.paper_id,
        ),
    )


def main() -> None:
    args = parse_args()
    db = PaperDatabase(args.db_path)
    rows = select_rows(db, args)
    print(f"Selected CCF-A AAAI/IJCAI missing abstracts: {len(rows)}", flush=True)

    attempted = 0
    enriched = 0
    failed = 0
    failures: list[str] = []
    last_request = 0.0

    try:
        for row in rows:
            attempted += 1
            elapsed = time.monotonic() - last_request
            if elapsed < args.request_delay:
                time.sleep(args.request_delay - elapsed)
            last_request = time.monotonic()

            try:
                venue = row["venue"] or ""
                if venue.startswith("AAAI"):
                    abstract, source_url = extract_aaai_abstract(row, args.timeout)
                    source = "aaai_oai_official"
                elif venue.startswith("IJCAI"):
                    abstract, source_url = extract_ijcai_abstract(row, args.timeout)
                    source = "ijcai_official_detail"
                else:
                    abstract, source_url, source = None, None, "unsupported"
            except (ET.ParseError, TimeoutError, urllib.error.URLError, UnicodeError) as exc:
                abstract, source_url, source = None, None, "error"
                failures.append(f"{row['paper_id']}: {exc}")

            if abstract:
                scored = score_paper(row_to_paper(row, abstract, source, source_url))
                update_paper(db, scored)
                enriched += 1
            else:
                failed += 1
                failures.append(f"{row['paper_id']}: abstract not found")

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
        print("\nCCF-A official enrichment summary")
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
