#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = ROOT / "data/metadata"
DEFAULT_OUTPUT_DIR = ROOT / "data/hf_dataset"

DATASET_FILES = {
    "topconf_all": "topconf_papers_all.jsonl",
    "frontier_2026_quality": "frontier_papers_2026_quality.jsonl",
    "arxiv_recent_3y_score_gte_4": "papers_recent_3y_score_gte_4.jsonl",
}

UNION_SPLITS = (
    "topconf_all",
    "frontier_2026_quality",
    "arxiv_recent_3y_score_gte_4",
)


def _clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _as_str_list(value: Any) -> list[str]:
    return [_clean_text(item) for item in _as_list(value) if _clean_text(item)]


def _as_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _paper_key(record: dict[str, Any]) -> str:
    paper_id = _clean_text(record.get("paper_id"))
    if paper_id:
        return paper_id.lower()
    arxiv_id = _clean_text(record.get("arxiv_id"))
    if arxiv_id:
        return f"arxiv:{arxiv_id.lower()}"
    doi = _clean_text(record.get("doi"))
    if doi:
        return f"doi:{doi.lower()}"
    title = _clean_text(record.get("title")).lower()
    year = _clean_text(record.get("year"))
    return f"title:{title}|{year}"


def normalize_record(record: dict[str, Any], *, split: str, source_file: str) -> dict[str, Any]:
    """Keep a compact, stable schema that HF Datasets can preview easily.

    We intentionally drop raw provider payloads such as raw_metadata because
    they are large, inconsistent across sources, and not needed by the app.
    """

    return {
        "paper_id": _clean_text(record.get("paper_id")),
        "title": _clean_text(record.get("title")),
        "authors": _as_str_list(record.get("authors")),
        "year": _as_int(record.get("year")),
        "venue": _clean_text(record.get("venue")),
        "corpus": _clean_text(record.get("corpus")),
        "abstract": _clean_text(record.get("abstract")),
        "keywords": _as_str_list(record.get("keywords")),
        "categories": _as_str_list(record.get("categories")),
        "fields_of_study": _as_str_list(record.get("fields_of_study")),
        "paper_url": _clean_text(record.get("paper_url")),
        "pdf_url": _clean_text(record.get("pdf_url")),
        "doi": _clean_text(record.get("doi")),
        "arxiv_id": _clean_text(record.get("arxiv_id")),
        "semantic_scholar_id": _clean_text(record.get("semantic_scholar_id")),
        "code_url": _clean_text(record.get("code_url")),
        "project_url": _clean_text(record.get("project_url")),
        "sources": _as_str_list(record.get("sources")),
        "source_queries": _as_str_list(record.get("source_queries")),
        "quality_signals": _as_str_list(record.get("quality_signals")),
        "decision": _clean_text(record.get("decision")),
        "relevance_score": _as_float(record.get("relevance_score")),
        "frontier_score": _as_float(record.get("frontier_score")),
        "metadata_score": _as_float(record.get("metadata_score")),
        "hybrid_score": _as_float(record.get("hybrid_score")),
        "citation_count": _as_int(record.get("citation_count")),
        "influential_citation_count": _as_int(record.get("influential_citation_count")),
        "collected_at": _clean_text(record.get("collected_at")),
        "updated_at": _clean_text(record.get("updated_at")),
        "dataset_split": split,
        "source_file": source_file,
    }


def iter_jsonl(path: Path) -> tuple[int, list[dict[str, Any]]]:
    bad_lines = 0
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                bad_lines += 1
                continue
            if isinstance(payload, dict):
                records.append(payload)
    return bad_lines, records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    years = Counter(record.get("year") for record in records if record.get("year") is not None)
    venues = Counter(record.get("venue") or record.get("corpus") or "unknown" for record in records)
    return {
        "record_count": len(records),
        "with_abstract": sum(bool(record.get("abstract")) for record in records),
        "with_pdf_url": sum(bool(record.get("pdf_url")) for record in records),
        "with_paper_url": sum(bool(record.get("paper_url")) for record in records),
        "year_distribution": dict(sorted(years.items())),
        "top_venues": dict(venues.most_common(20)),
    }


def build_dataset_card(manifest: dict[str, Any]) -> str:
    split_rows = "\n".join(
        f"| `{name}` | {info['record_count']} | {info['with_abstract']} | {info['with_pdf_url']} |"
        for name, info in manifest["splits"].items()
    )
    generated_at = manifest["generated_at"]
    return f"""---
license: mit
task_categories:
- text-retrieval
- question-answering
language:
- en
- zh
tags:
- embodied-ai
- robotics
- vision-language-action
- literature-search
- rag
pretty_name: Embodied AI Literature Metadata
---

# Embodied AI Literature Metadata

This dataset contains normalized paper metadata collected for an Embodied AI /
Vision-Language-Action literature assistant. It is intended for metadata search,
paper triage, and PDF retrieval before PaperQA-style evidence reading.

Generated at: `{generated_at}`

## Splits

| Split | Records | With abstract | With PDF URL |
|---|---:|---:|---:|
{split_rows}

## Files

- `data/all_curated.jsonl`: de-duplicated union of the main curated sources.
- `data/topconf_all.jsonl`: top conference / venue metadata collected by the project.
- `data/frontier_2026_quality.jsonl`: 2026 frontier / trending-quality candidates.
- `data/arxiv_recent_3y_score_gte_4.jsonl`: recent arXiv-style metadata subset.
- `metadata_manifest.json`: counts, coverage, and generation provenance.

## Schema

Each JSONL row uses a compact schema:

```text
paper_id, title, authors, year, venue, corpus, abstract, keywords, categories,
fields_of_study, paper_url, pdf_url, doi, arxiv_id, semantic_scholar_id,
code_url, project_url, sources, source_queries, quality_signals, decision,
relevance_score, frontier_score, metadata_score, hybrid_score, citation_count,
influential_citation_count, collected_at, updated_at, dataset_split, source_file
```

Provider-specific raw payloads were intentionally removed to keep the dataset
small and easy to preview.

## Intended Use

Use this dataset as a lightweight metadata index for:

- literature search and candidate recall;
- LLM-based paper triage;
- retrieving PDF URLs for downstream PaperQA reading;
- reproducing the demo workflow in the repository.

The metadata can contain duplicates across non-union splits because the splits
represent different collection policies. Use `all_curated` for a de-duplicated
view.
"""


def prepare_dataset(input_dir: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    (output_dir / "data").mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "splits": {},
        "source_files": {},
    }
    normalized_by_split: dict[str, list[dict[str, Any]]] = {}

    for split, filename in DATASET_FILES.items():
        source_path = input_dir / filename
        if not source_path.exists():
            manifest["source_files"][filename] = {"status": "missing"}
            normalized_by_split[split] = []
            continue
        bad_lines, raw_records = iter_jsonl(source_path)
        normalized = [
            normalize_record(record, split=split, source_file=filename)
            for record in raw_records
        ]
        normalized_by_split[split] = normalized
        output_path = output_dir / "data" / f"{split}.jsonl"
        write_jsonl(output_path, normalized)
        split_summary = summarize_records(normalized)
        split_summary.update({"filename": str(output_path.relative_to(output_dir)), "bad_lines": bad_lines})
        manifest["splits"][split] = split_summary
        manifest["source_files"][filename] = {
            "status": "ok",
            "raw_records": len(raw_records),
            "bad_lines": bad_lines,
        }

    union: dict[str, dict[str, Any]] = {}
    union_sources: defaultdict[str, list[str]] = defaultdict(list)
    for split in UNION_SPLITS:
        for record in normalized_by_split.get(split, []):
            key = _paper_key(record)
            if not key:
                continue
            if key not in union:
                union[key] = dict(record)
            union_sources[key].append(split)
    all_curated = []
    for key, record in union.items():
        merged = dict(record)
        merged["dataset_split"] = "all_curated"
        merged["source_file"] = "union:" + ",".join(sorted(set(union_sources[key])))
        merged["union_sources"] = sorted(set(union_sources[key]))
        all_curated.append(merged)
    all_curated.sort(key=lambda item: (-(item.get("year") or 0), item.get("title") or ""))
    write_jsonl(output_dir / "data" / "all_curated.jsonl", all_curated)
    manifest["splits"]["all_curated"] = {
        **summarize_records(all_curated),
        "filename": "data/all_curated.jsonl",
        "bad_lines": 0,
        "deduped_from": list(UNION_SPLITS),
    }

    manifest["schema_fields"] = list(normalize_record({}, split="", source_file="").keys())
    (output_dir / "metadata_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "README.md").write_text(build_dataset_card(manifest), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a Hugging Face dataset folder for metadata.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = prepare_dataset(args.input_dir, args.output_dir)
    print(f"Wrote Hugging Face dataset folder: {args.output_dir}")
    for split, info in manifest["splits"].items():
        print(
            f"- {split}: {info['record_count']} records, "
            f"{info['with_abstract']} abstracts, {info['with_pdf_url']} PDF URLs"
        )


if __name__ == "__main__":
    main()
