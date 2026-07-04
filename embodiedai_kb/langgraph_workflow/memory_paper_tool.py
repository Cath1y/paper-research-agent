from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from textwrap import shorten
from typing import Any

from embodiedai_kb.search.metadata_search import SearchResult
from scripts.ask_literature import (
    SelectedPaper,
    normalize_openai_compatible_model,
    openai_compatible_api_key,
)

from .memory import memory_path, memory_store_dir


def _clean_text(value: Any, *, width: int = 900) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if not text:
        return ""
    return shorten(text, width=width, placeholder="...")


def _stable_hash(*parts: Any, length: int = 12) -> str:
    raw = "\n".join(str(part or "") for part in parts)
    return hashlib.sha1(raw.encode("utf-8", errors="ignore")).hexdigest()[:length]


def _memory_paper_model(args: Any) -> str | None:
    model = (
        getattr(args, "memory_llm", None)
        or getattr(args, "agent_llm", None)
        or getattr(args, "router_llm", None)
        or getattr(args, "llm", None)
    )
    if not model:
        return None
    if getattr(args, "openai_base_url", None) and not getattr(
        args, "disable_openai_compatible_config", False
    ):
        return normalize_openai_compatible_model(model, args, args.openai_base_url)
    return model


def _read_jsonl(path: Path, *, limit: int | None = None) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            records.append(payload)
    if limit is not None and limit > 0:
        return records[-limit:]
    return records


def _paper_key(item: dict[str, Any]) -> str:
    for field in ("pdf_url", "paper_url", "paper_id", "title"):
        value = str(item.get(field) or "").strip().lower()
        if value:
            return " ".join(value.split())
    return ""


def _merge_nonempty(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in extra.items():
        if value in (None, "", [], {}):
            continue
        if key not in merged or merged.get(key) in (None, "", [], {}):
            merged[key] = value
        elif key == "sources":
            values: list[str] = []
            for source in [*merged.get("sources", []), *value]:
                text = str(source or "").strip()
                if text and text not in values:
                    values.append(text)
            merged[key] = values
    return merged


def _candidate_from_card(card: dict[str, Any]) -> dict[str, Any]:
    return {
        "card_id": card.get("card_id"),
        "paper_id": card.get("paper_id"),
        "episode_id": card.get("episode_id"),
        "title": card.get("title"),
        "authors": card.get("authors") or [],
        "year": card.get("year"),
        "venue": card.get("venue"),
        "abstract": card.get("summary") or card.get("core_idea"),
        "paper_url": card.get("paper_url"),
        "pdf_url": card.get("pdf_url"),
        "cache_status": card.get("cache_status"),
        "cache_path": card.get("cache_path"),
        "sources": card.get("sources") or [],
        "source_question": card.get("source_question"),
        "status": card.get("status"),
        "confidence": card.get("confidence"),
        "evidence_ids": card.get("evidence_ids") or [],
        "evidence_snippets": card.get("evidence_snippets") or [],
        "method_keywords": card.get("method_keywords") or [],
        "core_idea": card.get("core_idea"),
    }


def _candidate_from_record_paper(
    paper: dict[str, Any],
    record: dict[str, Any],
) -> dict[str, Any]:
    return {
        "card_id": None,
        "paper_id": paper.get("paper_id")
        or f"memory_{_stable_hash(paper.get('title'), paper.get('pdf_url'))}",
        "episode_id": record.get("episode_id"),
        "title": paper.get("title"),
        "authors": paper.get("authors") or [],
        "year": paper.get("year"),
        "venue": paper.get("venue"),
        "abstract": paper.get("abstract"),
        "paper_url": paper.get("paper_url"),
        "pdf_url": paper.get("pdf_url"),
        "cache_status": paper.get("cache_status"),
        "cache_path": paper.get("cache_path"),
        "sources": paper.get("sources") or [],
        "source_question": record.get("question"),
        "status": "read_pdf"
        if (record.get("paperqa") or {}).get("evidence_count")
        else "selected",
        "confidence": "high"
        if (record.get("paperqa") or {}).get("evidence_count")
        else "medium",
        "evidence_ids": [],
        "evidence_snippets": [],
        "method_keywords": paper.get("keywords") or paper.get("categories") or [],
        "core_idea": paper.get("abstract"),
    }


def _load_memory_paper_candidates(args: Any) -> list[dict[str, Any]]:
    """Load thread-local paper memory from cards and richer run records."""

    store_dir = memory_store_dir(args)
    card_records = _read_jsonl(store_dir / "PAPER_CARDS.jsonl")
    record_limit = int(getattr(args, "memory_paper_record_limit", 30) or 30)
    run_records = _read_jsonl(memory_path(args), limit=record_limit)

    merged: dict[str, dict[str, Any]] = {}
    for card in card_records:
        candidate = _candidate_from_card(card)
        key = _paper_key(candidate)
        if key:
            merged[key] = _merge_nonempty(merged.get(key, {}), candidate)

    for record in run_records:
        for paper in record.get("selected_papers") or []:
            if not isinstance(paper, dict):
                continue
            candidate = _candidate_from_record_paper(paper, record)
            key = _paper_key(candidate)
            if key:
                merged[key] = _merge_nonempty(merged.get(key, {}), candidate)

    by_title: dict[str, dict[str, Any]] = {}
    untitled: list[dict[str, Any]] = []
    for item in merged.values():
        title_key = " ".join(str(item.get("title") or "").lower().split())
        if not title_key:
            untitled.append(item)
            continue
        by_title[title_key] = _merge_nonempty(by_title.get(title_key, {}), item)

    candidates = [*by_title.values(), *untitled]
    candidates.sort(
        key=lambda item: (
            bool(item.get("cache_path") and Path(str(item.get("cache_path"))).exists()),
            bool(item.get("evidence_ids")),
            str(item.get("episode_id") or ""),
        ),
        reverse=True,
    )
    limit = int(getattr(args, "memory_paper_candidate_limit", 50) or 50)
    return [
        item
        for item in candidates
        if item.get("title") and (item.get("pdf_url") or item.get("cache_path"))
    ][:limit]


def _candidate_prompt(candidates: list[dict[str, Any]]) -> tuple[str, dict[str, dict[str, Any]]]:
    key_to_item: dict[str, dict[str, Any]] = {}
    lines: list[str] = []
    abstract_chars = int(500)
    for idx, item in enumerate(candidates, start=1):
        key = f"M{idx:03d}"
        key_to_item[key] = item
        cache_path = str(item.get("cache_path") or "")
        cache_exists = bool(cache_path and Path(cache_path).exists())
        evidence = item.get("evidence_snippets") or []
        evidence_preview = ""
        if evidence:
            evidence_preview = _clean_text(
                evidence[0].get("snippet") if isinstance(evidence[0], dict) else "",
                width=360,
            )
        lines.append(
            "\n".join(
                [
                    key,
                    f"title: {_clean_text(item.get('title'), width=220)}",
                    f"authors: {_clean_text(', '.join(str(x) for x in (item.get('authors') or [])[:12]), width=260)}",
                    f"year: {item.get('year') or 'unknown'}",
                    f"venue: {_clean_text(item.get('venue'), width=90) or 'unknown'}",
                    f"source_question: {_clean_text(item.get('source_question'), width=220)}",
                    f"status: {item.get('status') or 'unknown'} confidence: {item.get('confidence') or 'unknown'}",
                    f"cache_status: {item.get('cache_status') or 'unknown'} cache_exists: {cache_exists}",
                    f"paper_url: {item.get('paper_url') or ''}",
                    f"pdf_url: {item.get('pdf_url') or ''}",
                    f"summary: {_clean_text(item.get('abstract') or item.get('core_idea'), width=abstract_chars)}",
                    f"evidence_preview: {evidence_preview or 'N/A'}",
                ]
            )
        )
    return "\n\n".join(lines), key_to_item


def _extract_json_object(content: str) -> dict[str, Any]:
    text = (content or "").strip()
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return {}
    parsed = json.loads(match.group(0))
    return parsed if isinstance(parsed, dict) else {}


def _arxiv_id_from_url(url: str | None) -> str | None:
    text = str(url or "")
    match = re.search(r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5}(?:v\d+)?)", text)
    return match.group(1) if match else None


def _search_result_from_memory(item: dict[str, Any], *, rank: int) -> SearchResult:
    score = float(max(1, 100 - rank))
    paper_id = str(item.get("paper_id") or f"memory_{_stable_hash(item.get('title'))}")
    pdf_url = str(item.get("pdf_url") or "").strip() or None
    paper_url = str(item.get("paper_url") or "").strip() or None
    arxiv_id = _arxiv_id_from_url(pdf_url) or _arxiv_id_from_url(paper_url)
    sources = ["memory_paper_tool"]
    for source in item.get("sources") or []:
        text = str(source or "").strip()
        if text and text not in sources:
            sources.append(text)
    return SearchResult(
        rank=rank,
        hybrid_score=score,
        retrieval_score=score,
        metadata_score=score,
        paper_id=paper_id,
        corpus="memory",
        title=str(item.get("title") or "Untitled"),
        authors=[str(author) for author in (item.get("authors") or [])],
        year=int(item["year"]) if str(item.get("year") or "").isdigit() else None,
        venue=str(item.get("venue") or "memory"),
        abstract=str(item.get("abstract") or item.get("core_idea") or "") or None,
        paper_url=paper_url,
        pdf_url=pdf_url,
        code_url=None,
        project_url=None,
        doi=None,
        arxiv_id=arxiv_id,
        citation_count=None,
        influential_citation_count=None,
        frontier_score=0.0,
        relevance_score=score,
        decision="memory_reuse_candidate",
        keywords=[str(x) for x in (item.get("method_keywords") or [])],
        categories=[],
        sources=sources,
        quality_signals=["thread_memory", str(item.get("status") or "memory")],
        relevance_reasons=[
            _clean_text(item.get("source_question"), width=180),
        ]
        if item.get("source_question")
        else [],
        matched_terms=[],
    )


def _selected_from_memory_items(items: list[dict[str, Any]]) -> list[SelectedPaper]:
    selected: list[SelectedPaper] = []
    for rank, item in enumerate(items, start=1):
        result = _search_result_from_memory(item, rank=rank)
        selected.append(
            SelectedPaper(
                result=result,
                selection_score=float(result.hybrid_score),
                cache_path=str(item.get("cache_path") or "") or None,
                cache_status=str(item.get("cache_status") or "not_downloaded"),
            )
        )
    return selected


def _empty_trace(reason: str) -> dict[str, Any]:
    return {
        "mode": "memory_paper_tool",
        "candidate_count": 0,
        "selected_count": 0,
        "sufficient_for_question": False,
        "selected_keys": [],
        "reason": reason,
    }


async def run_memory_paper_tool(
    *,
    question: str,
    research_plan: list[dict[str, Any]],
    research_questions: list[dict[str, Any]],
    args: Any,
) -> tuple[list[SelectedPaper], dict[str, Any]]:
    """Select reusable PDFs from the current thread's paper memory.

    This is a local ExpertResearchAgent tool. It does not search the web and it
    does not answer the user. It exposes already remembered/read papers to an
    LLM selector, then returns SelectedPaper objects that the normal PaperQA
    reader can consume.
    """

    if getattr(args, "no_memory", False):
        return [], _empty_trace("memory disabled")
    if getattr(args, "disable_memory_paper_tool", False):
        return [], _empty_trace("memory paper tool disabled")

    candidates = _load_memory_paper_candidates(args)
    if not candidates:
        return [], _empty_trace("no reusable memory paper candidates")

    model = _memory_paper_model(args)
    if not model:
        trace = _empty_trace("no memory/agent llm configured")
        trace["candidate_count"] = len(candidates)
        return [], trace

    candidate_text, key_to_item = _candidate_prompt(candidates)
    plan_context = "\n".join(
        f"- {item.get('perspective') or item.get('agent') or 'plan'}: "
        f"{item.get('research_question') or item.get('question') or item.get('description') or ''}"
        for item in (research_plan or [])[:8]
    )
    question_context = "\n".join(
        f"- {item.get('perspective') or ''}: "
        f"{item.get('question') or item.get('research_question') or ''}"
        for item in (research_questions or [])[:8]
    )

    raw_content = ""
    try:
        import litellm

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are MemoryPaperToolSelector inside ExpertResearchAgent. "
                        "Your task is to decide whether previously remembered PDFs are useful "
                        "for the current user question. Select only memory papers that should be "
                        "read or reused by PaperQA now. Do not answer the user. Do not invent papers. "
                        "If remembered papers are enough for the current question, set "
                        "sufficient_for_question=true so the workflow can skip external paper search. "
                        "If the question asks for newer/fresh papers or the remembered papers only "
                        "partially match, set sufficient_for_question=false."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"User question:\n{question}\n\n"
                        f"Research plan:\n{plan_context or 'N/A'}\n\n"
                        f"Research questions:\n{question_context or 'N/A'}\n\n"
                        f"Memory paper candidates:\n{candidate_text}\n\n"
                        "Return compact JSON only:\n"
                        "{\n"
                        '  "selected": ["M001", "M004"],\n'
                        '  "sufficient_for_question": false,\n'
                        '  "reason": "short Chinese reason",\n'
                        '  "coverage_notes": ["what memory covers", "what still needs search"]\n'
                        "}"
                    ),
                },
            ],
            "temperature": 0,
            "max_tokens": int(getattr(args, "memory_paper_max_tokens", 1000) or 1000),
            "timeout": float(getattr(args, "llm_timeout", 180.0)),
            "response_format": {"type": "json_object"},
        }
        if getattr(args, "openai_base_url", None) and not getattr(
            args, "disable_openai_compatible_config", False
        ):
            kwargs["api_base"] = args.openai_base_url
            api_key = openai_compatible_api_key(
                args,
                args.openai_base_url,
                args.openai_api_key_env,
            )
            if api_key:
                kwargs["api_key"] = api_key
        response = await litellm.acompletion(**kwargs)
        raw_content = (response.choices[0].message.content or "").strip()
        payload = _extract_json_object(raw_content)
    except Exception as exc:
        return [], {
            "mode": "memory_paper_tool",
            "model": model,
            "candidate_count": len(candidates),
            "selected_count": 0,
            "sufficient_for_question": False,
            "selected_keys": [],
            "reason": f"memory paper selector failed: {_clean_text(exc, width=220)}",
            "raw_content": raw_content,
            "error": _clean_text(exc, width=500),
        }

    top_k = int(getattr(args, "paperqa_k", 8) or 8)
    selected_keys = [
        str(key).strip()
        for key in payload.get("selected", [])
        if str(key).strip() in key_to_item
    ][:top_k]
    selected_items = [key_to_item[key] for key in selected_keys]
    selected = _selected_from_memory_items(selected_items)
    return selected, {
        "mode": "memory_paper_tool",
        "model": model,
        "candidate_count": len(candidates),
        "selected_count": len(selected),
        "selected_keys": selected_keys,
        "selected_titles": [item.result.title for item in selected],
        "sufficient_for_question": bool(payload.get("sufficient_for_question"))
        and bool(selected),
        "reason": _clean_text(payload.get("reason"), width=320),
        "coverage_notes": payload.get("coverage_notes") or [],
        "candidate_prompt_char_count": len(candidate_text),
        "raw_content": raw_content,
    }
