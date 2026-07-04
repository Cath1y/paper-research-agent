from __future__ import annotations

import json
import re
from textwrap import shorten
from typing import Any

from embodiedai_kb.search.metadata_search import SearchResult
from scripts.ask_literature import (
    SelectedPaper,
    normalize_openai_compatible_model,
    openai_compatible_api_key,
)

from .progress import emit_progress


def _triage_model(args: Any) -> str | None:
    model = (
        getattr(args, "paper_triage_llm", None)
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


def _extract_json_object(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        raise ValueError("No JSON object found in PaperTriageAgent output.")
    parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("PaperTriageAgent output must be a JSON object.")
    return parsed


def _dedupe_selected_candidates(candidates: list[SelectedPaper]) -> list[SelectedPaper]:
    seen: set[str] = set()
    deduped: list[SelectedPaper] = []
    for item in candidates:
        result = item.result
        key = (
            result.doi
            or result.arxiv_id
            or result.pdf_url
            or result.paper_url
            or result.title
        )
        key = " ".join(str(key or "").lower().split())
        if not key or key in seen or not (result.pdf_url or item.cache_path):
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _balanced_candidate_view(
    candidates: list[SelectedPaper],
    limit: int,
) -> list[SelectedPaper]:
    """Keep the LLM candidate list broad instead of dominated by one source."""

    groups: dict[str, list[SelectedPaper]] = {"academic": [], "metadata": [], "web": []}
    for item in candidates:
        result = item.result
        sources = set(result.sources or [])
        if sources & {"web_paper_discovery", "web_profile_llm", "web_profile"}:
            groups["web"].append(item)
        elif (
            result.corpus in {"academic", "paper_search"}
            or "academic_paper_search" in sources
            or sources & {"arxiv", "openalex", "openreview", "crossref"}
        ):
            groups["academic"].append(item)
        else:
            groups["metadata"].append(item)

    for values in groups.values():
        values.sort(key=lambda item: item.selection_score, reverse=True)

    ordered: list[SelectedPaper] = []
    seen: set[int] = set()
    group_order = ["web", "academic", "metadata"]
    while len(ordered) < limit:
        progressed = False
        for group_name in group_order:
            values = groups[group_name]
            index = sum(1 for item in ordered if id(item) in {id(v) for v in values})
            if index >= len(values):
                continue
            item = values[index]
            if id(item) not in seen:
                seen.add(id(item))
                ordered.append(item)
                progressed = True
                if len(ordered) >= limit:
                    break
        if not progressed:
            break
    return ordered


def _clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _join_or_dash(values: list[Any] | tuple[Any, ...] | None) -> str:
    cleaned = [_clean_text(value) for value in values or [] if _clean_text(value)]
    return ", ".join(cleaned) if cleaned else "-"


def _candidate_debug_record(key: str, item: SelectedPaper) -> dict[str, Any]:
    r = item.result
    return {
        "key": key,
        "title": r.title,
        "authors": r.authors,
        "year": r.year,
        "venue": r.venue,
        "corpus": r.corpus,
        "paper_id": r.paper_id,
        "paper_url": r.paper_url,
        "pdf_url": r.pdf_url,
        "doi": r.doi,
        "arxiv_id": r.arxiv_id,
        "citation_count": r.citation_count,
        "influential_citation_count": r.influential_citation_count,
        "selection_score": item.selection_score,
        "hybrid_score": r.hybrid_score,
        "retrieval_score": r.retrieval_score,
        "metadata_score": r.metadata_score,
        "frontier_score": r.frontier_score,
        "relevance_score": r.relevance_score,
        "decision": r.decision,
        "sources": r.sources,
        "quality_signals": r.quality_signals,
        "keywords": r.keywords,
        "categories": r.categories,
        "matched_terms": r.matched_terms,
        "relevance_reasons": r.relevance_reasons,
        "abstract": r.abstract,
    }


def _candidate_lines(
    candidates: list[SelectedPaper],
    *,
    abstract_max_chars: int = 1800,
    key_offset: int = 0,
    keys: list[str] | None = None,
    score_records: dict[str, dict[str, Any]] | None = None,
    compact: bool = False,
) -> tuple[str, dict[str, SelectedPaper], list[dict[str, Any]]]:
    key_to_item: dict[str, SelectedPaper] = {}
    debug_records: list[dict[str, Any]] = []
    lines: list[str] = []
    for idx, item in enumerate(candidates, start=1):
        key = keys[idx - 1] if keys and idx - 1 < len(keys) else f"P{idx + key_offset:03d}"
        key_to_item[key] = item
        r = item.result
        debug_records.append(_candidate_debug_record(key, item))
        authors = _join_or_dash(r.authors) or "Unknown authors"
        sources = _join_or_dash(r.sources) or r.corpus
        relevance_reasons = _join_or_dash(r.relevance_reasons)
        abstract = _clean_text(r.abstract)
        if abstract_max_chars > 0 and len(abstract) > abstract_max_chars:
            abstract = shorten(abstract, width=abstract_max_chars, placeholder="...")
        score_record = (score_records or {}).get(key, {})
        score_lines = []
        if score_record:
            score_lines = [
                f"screen_final_score: {score_record.get('final_score', 0)}",
                f"screen_entity_match: {score_record.get('entity_match', 0)}",
                f"screen_topic_match: {score_record.get('topic_match', 0)}",
                f"screen_year_match: {score_record.get('year_match', 0)}",
                f"screen_evidence_value: {score_record.get('evidence_value', 0)}",
                f"screen_should_read: {score_record.get('should_read', False)}",
                f"screen_reason: {_clean_text(score_record.get('reason'))}",
            ]
        if compact:
            lines.append(
                "\n".join(
                    [
                        f"{key}",
                        f"title: {r.title}",
                        f"authors_full: {authors}",
                        f"year: {r.year or 'unknown'}",
                        f"venue: {r.venue or r.corpus}",
                        f"sources: {sources}",
                        f"paper_url: {r.paper_url or ''}",
                        f"pdf_url: {r.pdf_url or ''}",
                        f"discovered_by_query: {relevance_reasons}",
                        f"abstract: {abstract or 'N/A'}",
                        *score_lines,
                    ]
                )
            )
            continue
        signals = _join_or_dash(r.quality_signals)
        keywords = _join_or_dash(r.keywords)
        categories = _join_or_dash(r.categories)
        matched_terms = _join_or_dash(r.matched_terms)
        lines.append(
            "\n".join(
                [
                    f"{key}",
                    f"title: {r.title}",
                    f"authors_full: {authors}",
                    f"year: {r.year or 'unknown'}",
                    f"venue/corpus: {r.venue or r.corpus}",
                    f"paper_id: {r.paper_id}",
                    f"doi: {r.doi or ''}",
                    f"sources: {sources}",
                    f"paper_url: {r.paper_url or ''}",
                    f"pdf_url: {r.pdf_url or ''}",
                    f"citations: {r.citation_count or 0}",
                    f"influential_citations: {r.influential_citation_count or 0}",
                    f"candidate_score: {item.selection_score:.4f}",
                    f"decision: {r.decision}",
                    f"quality_signals: {signals}",
                    f"keywords: {keywords}",
                    f"categories: {categories}",
                    f"matched_terms: {matched_terms}",
                    f"relevance_reasons: {relevance_reasons}",
                    f"abstract: {abstract or 'N/A'}",
                    *score_lines,
                ]
            )
        )
    return "\n\n".join(lines), key_to_item, debug_records


def _format_web_context(items: list[dict[str, Any]], limit: int = 8) -> str:
    lines: list[str] = []
    for idx, item in enumerate(items[:limit], start=1):
        title = item.get("title") or "Untitled"
        url = item.get("url") or ""
        snippet = shorten(
            str(item.get("snippet") or "").replace("\n", " "),
            width=450,
            placeholder="...",
        )
        lines.append(f"[{idx}] {title}\nurl: {url}\nsnippet: {snippet}")
    return "\n\n".join(lines)


def _coerce_score(value: Any, *, default: float = 0.0, max_value: float = 100.0) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(score, max_value))


def _coerce_bool(value: Any, *, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1"}:
            return True
        if lowered in {"false", "no", "0"}:
            return False
    if value is None:
        return default
    return bool(value)


def _normalize_screen_record(raw: dict[str, Any], key_to_item: dict[str, SelectedPaper]) -> dict[str, Any] | None:
    key = str(raw.get("paper_key") or raw.get("key") or raw.get("id") or "").strip()
    if key not in key_to_item:
        return None
    entity = _coerce_score(raw.get("entity_match"), max_value=5.0)
    topic = _coerce_score(raw.get("topic_match"), max_value=5.0)
    year = _coerce_score(raw.get("year_match"), max_value=5.0)
    evidence = _coerce_score(raw.get("evidence_value"), max_value=5.0)
    final_score = _coerce_score(raw.get("final_score"), max_value=100.0)
    if final_score <= 0:
        final_score = round((entity * 0.35 + topic * 0.30 + year * 0.15 + evidence * 0.20) * 20, 2)
    return {
        "key": key,
        "entity_match": entity,
        "topic_match": topic,
        "year_match": year,
        "evidence_value": evidence,
        "final_score": final_score,
        "should_read": _coerce_bool(raw.get("should_read"), default=final_score >= 60),
        "reason": _clean_text(raw.get("reason") or raw.get("rationale")),
    }


async def _screen_candidate_batches(
    *,
    question: str,
    candidates: list[SelectedPaper],
    web_context: str,
    plan_context: str,
    model: str,
    args: Any,
    batch_size: int,
    abstract_max_chars: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, SelectedPaper], list[dict[str, Any]]]:
    """Score every candidate in compact batches before final triage selection."""

    all_key_to_item: dict[str, SelectedPaper] = {}
    all_debug_records: list[dict[str, Any]] = []
    screen_records: list[dict[str, Any]] = []
    batch_traces: list[dict[str, Any]] = []
    prompt_char_count = 0

    for start in range(0, len(candidates), batch_size):
        batch = candidates[start : start + batch_size]
        candidate_text, key_to_item, debug_records = _candidate_lines(
            batch,
            abstract_max_chars=abstract_max_chars,
            key_offset=start,
            compact=True,
        )
        all_key_to_item.update(key_to_item)
        all_debug_records.extend(debug_records)
        prompt_char_count += len(candidate_text)
        emit_progress(
            args,
            "PaperTriageAgent",
            "screen batch start",
            batch=f"{len(batch_traces) + 1}/{(len(candidates) + batch_size - 1) // batch_size}",
            candidates=len(batch),
        )

        try:
            import litellm

            kwargs: dict[str, Any] = {
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are PaperTriageAgent.Screener. Score every candidate PDF "
                            "for whether it should be read by PaperQA for the exact user question. "
                            "Return compact JSON only. You must output one score object for every "
                            "candidate key in this batch. Do not invent papers. For author/person "
                            "questions, entity_match should be high only when authors_full or web "
                            "context supports the target person/team; remember the target author "
                            "may be a middle or last author. Topic-only similarity is not enough "
                            "for person questions. For topic questions, entity_match can be neutral."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"User question:\n{question}\n\n"
                            f"Research plan/context:\n{plan_context or 'N/A'}\n\n"
                            f"Web evidence for disambiguation:\n{web_context or 'N/A'}\n\n"
                            "Candidate PDFs in this batch:\n"
                            f"{candidate_text}\n\n"
                            "Return JSON schema:\n"
                            "{\n"
                            '  "scores": [\n'
                            '    {"paper_key": "P001", "entity_match": 0-5, '
                            '"topic_match": 0-5, "year_match": 0-5, '
                            '"evidence_value": 0-5, "final_score": 0-100, '
                            '"should_read": true, "reason": "short reason"}\n'
                            "  ]\n"
                            "}"
                        ),
                    },
                ],
                "temperature": 0,
                "max_tokens": max(1200, min(3600, 220 + 180 * len(batch))),
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
            batch_records: list[dict[str, Any]] = []
            for raw in payload.get("scores") or payload.get("candidates") or []:
                if not isinstance(raw, dict):
                    continue
                normalized = _normalize_screen_record(raw, key_to_item)
                if normalized:
                    batch_records.append(normalized)
            scored_keys = {item["key"] for item in batch_records}
            for key, item in key_to_item.items():
                if key in scored_keys:
                    continue
                batch_records.append(
                    {
                        "key": key,
                        "entity_match": 0.0,
                        "topic_match": 0.0,
                        "year_match": 0.0,
                        "evidence_value": 0.0,
                        "final_score": 0.0,
                        "should_read": False,
                        "reason": "missing from screener response",
                    }
                )
            screen_records.extend(batch_records)
            emit_progress(
                args,
                "PaperTriageAgent",
                "screen batch done",
                scored=len(batch_records),
            )
            batch_traces.append(
                {
                    "batch": len(batch_traces) + 1,
                    "candidate_count": len(batch),
                    "scored_count": len(batch_records),
                    "error": None,
                }
            )
        except Exception as exc:
            emit_progress(
                args,
                "PaperTriageAgent",
                "screen batch error",
                error=shorten(str(exc), width=120, placeholder="..."),
            )
            for key in key_to_item:
                screen_records.append(
                    {
                        "key": key,
                        "entity_match": 0.0,
                        "topic_match": 0.0,
                        "year_match": 0.0,
                        "evidence_value": 0.0,
                        "final_score": 0.0,
                        "should_read": False,
                        "reason": "screener batch failed",
                    }
                )
            batch_traces.append(
                {
                    "batch": len(batch_traces) + 1,
                    "candidate_count": len(batch),
                    "scored_count": 0,
                    "error": shorten(str(exc), width=400, placeholder="..."),
                }
            )

    screen_records.sort(key=lambda item: float(item.get("final_score") or 0.0), reverse=True)
    return screen_records, {
        "mode": "llm_batch_screen",
        "batch_size": batch_size,
        "abstract_max_chars": abstract_max_chars,
        "screened_count": len(screen_records),
        "prompt_char_count": prompt_char_count,
        "batches": batch_traces,
    }, all_key_to_item, all_debug_records


async def triage_papers_for_reading(
    *,
    question: str,
    candidates: list[SelectedPaper],
    web_evidence: list[dict[str, Any]],
    planning_web_evidence: list[dict[str, Any]],
    research_plan: list[dict[str, Any]],
    args: Any,
) -> tuple[list[SelectedPaper], dict[str, Any]]:
    """Let an LLM choose the PDFs to read from a candidate list.

    Existing deep-research projects generally retrieve broadly, then let the
    research model/compression step decide which sources are actually relevant.
    This node applies that pattern before PaperQA reads PDFs.
    """

    deduped = _dedupe_selected_candidates(candidates)
    limit = int(getattr(args, "paper_triage_candidate_limit", 30))
    if not deduped:
        return [], {"mode": "empty", "candidate_count": 0, "selected_count": 0}

    mode = getattr(args, "paper_selection_mode", "llm")
    model = _triage_model(args)
    if mode == "llm" and not model:
        return [], {
            "mode": "no_llm_no_selection",
            "model": None,
            "candidate_count": len(deduped),
            "selected_count": 0,
            "selected_keys": [],
            "reason": "paper_selection_mode=llm requires a triage LLM; refusing score fallback",
        }

    if mode == "score" or (mode == "auto" and not model):
        fallback = sorted(deduped, key=lambda item: item.selection_score, reverse=True)[
            : int(getattr(args, "paperqa_k", 8))
        ]
        return fallback, {
            "mode": "score_fallback",
            "model": model,
            "candidate_count": len(deduped),
            "selected_count": len(fallback),
            "selected_keys": [],
            "reason": "paper_selection_mode=score or no triage LLM configured",
        }

    final_abstract_max_chars = int(getattr(args, "paper_triage_abstract_max_chars", 400))
    screen_abstract_max_chars = int(
        getattr(args, "paper_triage_screen_abstract_max_chars", 350)
    )
    screen_batch_size = max(1, int(getattr(args, "paper_triage_screen_batch_size", 20)))
    screen_top_n = max(1, int(getattr(args, "paper_triage_screen_top_n", limit)))
    web_context = _format_web_context([*planning_web_evidence, *web_evidence], limit=10)
    plan_context = "\n".join(
        f"- {item.get('perspective') or item.get('agent') or 'plan'}: "
        f"{item.get('research_question') or item.get('objective') or item.get('description') or ''}"
        for item in research_plan[:8]
    )
    screen_records, screen_trace, all_key_to_item, screen_debug_records = (
        await _screen_candidate_batches(
            question=question,
            candidates=deduped,
            web_context=web_context,
            plan_context=plan_context,
            model=model,
            args=args,
            batch_size=screen_batch_size,
            abstract_max_chars=screen_abstract_max_chars,
        )
    )
    score_records = {str(item.get("key")): item for item in screen_records}
    effective_screen_records = [
        item
        for item in screen_records
        if float(item.get("final_score") or 0.0) > 0 or bool(item.get("should_read"))
    ]
    final_keys = [
        str(item.get("key"))
        for item in effective_screen_records
        if str(item.get("key")) in all_key_to_item
    ][: min(limit, screen_top_n)]
    final_candidates = [all_key_to_item[key] for key in final_keys]
    if not final_candidates:
        final_candidates = sorted(deduped, key=lambda item: item.selection_score, reverse=True)[
            : min(limit, screen_top_n)
        ]
        score_records = {}
    candidate_text, key_to_item, candidate_debug_records = _candidate_lines(
        final_candidates,
        abstract_max_chars=final_abstract_max_chars,
        keys=final_keys if len(final_keys) == len(final_candidates) else None,
        score_records=score_records,
        compact=True,
    )
    top_k = int(getattr(args, "paperqa_k", 8))
    raw_content = ""

    try:
        import litellm
        emit_progress(
            args,
            "PaperTriageAgent",
            "final selection start",
            candidates=len(final_candidates),
            top_k=top_k,
        )

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are PaperTriageAgent. Your only job is to choose which candidate "
                        "PDFs should be read by a downstream PaperQA reader for the user's question. "
                        "The candidates below are the top results from a previous batch screener "
                        "that scored all retrieved PDFs. Use those screen scores as hints, but make "
                        "the final decision from the compact metadata itself. "
                        "Do not invent papers. Do not select papers just because they are highly cited, "
                        "recent, or in a local domain database. Select only papers whose title/authors/"
                        "abstract/source evidence make them useful for answering the exact question. "
                        "Read the full authors_full field carefully; do not assume a target author is absent "
                        "just because they are not first author. "
                        "For author/person questions, select a paper only if the candidate metadata or "
                        "web context supports that it is by the target person/team; otherwise reject it. "
                        "For author/person questions with an explicit year or recent-year range, strongly "
                        "prioritize candidates that both match the target author/team and fall within the "
                        "requested year range. Do not select old papers unless the user asks for historical "
                        "background or there are no in-range target-author papers. "
                        "For author/person questions, never select a generic background survey or a paper "
                        "by unrelated authors merely because its topic matches the person's field. "
                        "If fewer than top_k papers are clearly tied to the target person/team, select fewer. "
                        "For topic questions, choose papers that jointly cover the user's requested angles. "
                        "It is acceptable to return fewer than the requested number, or zero, if the "
                        "candidate list is not relevant enough. Return compact JSON only. "
                        "Do not write a rejection reason for every candidate; include at most 10 "
                        "representative rejected_reasons entries for important or ambiguous cases."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"User question:\n{question}\n\n"
                        f"Research plan/context:\n{plan_context or 'N/A'}\n\n"
                        f"Web evidence that may help disambiguate authors or sources:\n"
                        f"{web_context or 'N/A'}\n\n"
                        f"Candidate PDFs. Choose up to {top_k} keys.\n\n"
                        f"{candidate_text}\n\n"
                        "Return JSON with this schema:\n"
                        "{\n"
                        '  "selected": ["P001", "P005"],\n'
                        '  "reject_all_if_needed": false,\n'
                        '  "rationale": "short reason",\n'
                        '  "coverage_notes": ["what is covered", "what is missing"],\n'
                        '  "rejected_reasons": {"P002": "not about target author"}\n'
                        "}"
                    ),
                },
            ],
            "temperature": 0,
            "max_tokens": 2200,
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
        selected_keys = [
            str(key).strip()
            for key in payload.get("selected", [])
            if str(key).strip() in key_to_item
        ][:top_k]
        selected: list[SelectedPaper] = []
        for rank, key in enumerate(selected_keys, start=1):
            original = key_to_item[key]
            selected.append(
                SelectedPaper(
                    result=original.result,
                    selection_score=float(top_k - rank + 1),
                    cache_path=original.cache_path,
                    cache_status=original.cache_status,
                    error=original.error,
                )
            )
        emit_progress(
            args,
            "PaperTriageAgent",
            "final selection done",
            selected=len(selected),
        )
        return selected, {
            "mode": "llm",
            "model": model,
            "candidate_count": len(deduped),
            "final_candidate_count": len(final_candidates),
            "candidate_records": candidate_debug_records,
            "screen_candidate_records": screen_debug_records,
            "screen_records": screen_records[: max(50, len(final_candidates))],
            "screen_trace": screen_trace,
            "candidate_prompt_char_count": len(candidate_text),
            "abstract_max_chars": final_abstract_max_chars,
            "selected_count": len(selected),
            "selected_keys": selected_keys,
            "selected_titles": [item.result.title for item in selected],
            "rationale": payload.get("rationale"),
            "coverage_notes": payload.get("coverage_notes") or [],
            "rejected_reasons": payload.get("rejected_reasons") or {},
        }
    except Exception as exc:
        emit_progress(
            args,
            "PaperTriageAgent",
            "final selection error",
            error=shorten(str(exc), width=120, placeholder="..."),
        )
        if mode == "llm":
            return [], {
                "mode": "llm_error_no_selection",
                "model": model,
                "candidate_count": len(deduped),
                "candidate_records": candidate_debug_records,
                "screen_candidate_records": screen_debug_records,
                "screen_records": screen_records[: max(50, len(final_candidates))],
                "screen_trace": screen_trace,
                "candidate_prompt_char_count": len(candidate_text),
                "abstract_max_chars": final_abstract_max_chars,
                "selected_count": 0,
                "error": shorten(str(exc), width=500, placeholder="..."),
                "raw_response_preview": shorten(
                    raw_content,
                    width=800,
                    placeholder="...",
                ),
            }
        fallback = sorted(deduped, key=lambda item: item.selection_score, reverse=True)[:top_k]
        return fallback, {
            "mode": "auto_fallback_after_llm_error",
            "model": model,
            "candidate_count": len(deduped),
            "candidate_records": candidate_debug_records,
            "screen_candidate_records": screen_debug_records,
            "screen_records": screen_records[: max(50, len(final_candidates))],
            "screen_trace": screen_trace,
            "candidate_prompt_char_count": len(candidate_text),
            "abstract_max_chars": final_abstract_max_chars,
            "selected_count": len(fallback),
            "error": shorten(str(exc), width=500, placeholder="..."),
        }
