from __future__ import annotations

from pathlib import Path
from textwrap import shorten
from typing import Any

from embodiedai_kb.search.metadata_search import (
    MetadataSearchEngine,
    SearchCorpus,
    SearchFilters,
    SearchResult,
    reciprocal_rank_fusion,
)
from scripts.ask_literature import (
    SelectedPaper,
    cache_selected_pdfs,
    normalize_openai_compatible_model,
    openai_compatible_api_key,
    parse_venues,
    select_for_paperqa,
)

from .memory_paper_tool import run_memory_paper_tool
from .paperqa_bridge import run_paperqa_reader_adapter
from .paper_search_agent import run_paper_search_agent
from .paper_triage import triage_papers_for_reading
from .progress import emit_progress
from .web_search import run_web_search_agent


PAPERQA_READING_TASKS = {
    "literature_review",
    "paper_deep_read",
    "learning_plan",
    "idea_generation",
}


def _triage_feedback_for_search(
    *,
    triage_trace: dict[str, Any],
    selected: list[SelectedPaper],
    args: Any,
) -> tuple[bool, str]:
    """Convert PaperTriageAgent's judgment into a follow-up search instruction."""

    selected_count = len(selected)
    min_selected = int(getattr(args, "paper_search_triage_min_selected", 1) or 1)
    sufficient_for_reading = triage_trace.get("sufficient_for_reading")
    requires_followup_search = triage_trace.get("requires_followup_search")
    if sufficient_for_reading is True and selected_count > 0:
        return False, ""
    if requires_followup_search is False and selected_count >= min_selected:
        return False, ""
    if sufficient_for_reading is not False and selected_count >= min_selected:
        return False, ""

    rationale = str(triage_trace.get("rationale") or "").strip()
    coverage_notes = triage_trace.get("coverage_notes") or []
    rejected_reasons = triage_trace.get("rejected_reasons") or {}
    screen_records = triage_trace.get("screen_records") or []
    top_rejections: list[str] = []
    if isinstance(rejected_reasons, dict):
        for key, reason in list(rejected_reasons.items())[:8]:
            top_rejections.append(f"{key}: {reason}")
    if not top_rejections and isinstance(screen_records, list):
        for record in screen_records[:8]:
            key = record.get("key") or record.get("paper_key") or "candidate"
            reason = record.get("reason") or ""
            score = record.get("final_score")
            top_rejections.append(f"{key}: score={score}; {reason}")

    feedback_parts = [
        "PaperTriageAgent judged the current paper candidates insufficient.",
        f"selected_count={selected_count}; required_min_selected={min_selected}.",
        f"sufficient_for_reading={sufficient_for_reading}; "
        f"requires_followup_search={requires_followup_search}.",
    ]
    if rationale:
        feedback_parts.append(f"Triage rationale: {rationale}")
    if coverage_notes:
        feedback_parts.append(
            "Coverage notes: "
            + "; ".join(str(item).strip() for item in coverage_notes[:6] if str(item).strip())
        )
    if top_rejections:
        feedback_parts.append("Representative rejection reasons: " + "; ".join(top_rejections[:8]))
    feedback_parts.append(
        "Next search should address these gaps directly. Prefer official/profile/"
        "publication-page evidence, exact paper-title lookup, and PDF-backed papers. "
        "Avoid repeating broad queries that only produced same-name or weakly related papers."
    )
    return True, "\n".join(feedback_parts)


def format_web_evidence(items: list[dict[str, Any]], limit: int = 6) -> str:
    lines: list[str] = []
    for idx, item in enumerate(items[:limit], start=1):
        title = item.get("title") or "Untitled"
        url = item.get("url") or ""
        snippet = item.get("snippet") or ""
        if snippet:
            lines.append(f"{idx}. {title}: {snippet}\n   {url}")
        else:
            lines.append(f"{idx}. {title}\n   {url}")
    return "\n".join(lines)


def web_queries_for_research(
    question: str,
    queries: list[str],
    research_plan: list[dict[str, Any]] | None = None,
) -> list[str]:
    """Convert perspective-planned queries into web-search-friendly queries."""

    web_queries: list[str] = []
    seen: set[str] = set()

    def add(query: str) -> None:
        clean = " ".join(str(query).split())
        key = clean.lower()
        if clean and key not in seen:
            seen.add(key)
            web_queries.append(clean)

    if "vla" in question.lower() or "视觉语言动作" in question:
        add("vision-language-action VLA robot manipulation 2026 paper github project")
        add("VLA robot manipulation 2026 arxiv hugging face papers")
        add("vision language action model robot manipulation ICRA 2026 OpenReview")

    add(question)
    for item in research_plan or []:
        subquestion = str(
            item.get("research_question")
            or item.get("question")
            or ""
        )
        if subquestion:
            add(subquestion)
        for query in item.get("queries") or []:
            add(str(query))

    noisy_markers = (
        "basic fact writer",
        "model architecture writer",
        "data benchmark writer",
        "deployment systems writer",
        "focusing broadly",
        "focuses backbones",
    )
    for query in queries:
        lowered = query.lower()
        if any(marker in lowered for marker in noisy_markers):
            continue
        if len(query.split()) < 3 and "vla" not in lowered:
            continue
        add(query)

    return web_queries


def _paperqa_question_for_research(
    question: str,
    research_questions: list[dict[str, Any]],
    research_plan: list[dict[str, Any]],
) -> str:
    """Give PaperQA the user's question plus the planned coverage checklist."""

    planned_questions: list[str] = []
    for item in research_questions or []:
        q = str(item.get("question") or item.get("research_question") or "").strip()
        if q:
            perspective = str(item.get("perspective") or "").strip()
            planned_questions.append(f"- {perspective}: {q}" if perspective else f"- {q}")
    if not planned_questions:
        for item in research_plan or []:
            q = str(item.get("research_question") or item.get("question") or "").strip()
            if q:
                perspective = str(item.get("perspective") or "").strip()
                planned_questions.append(f"- {perspective}: {q}" if perspective else f"- {q}")

    if not planned_questions:
        return question
    checklist = "\n".join(planned_questions[:6])
    return (
        "原始用户问题（最高优先级，所有证据都要服务于这个问题）：\n"
        f"{question}\n\n"
        "ResearchPlanningAgent 规划出的调研角度（用于补全覆盖面，但不能偏离原始问题）：\n"
        f"{checklist}"
        "\n\n"
        "请在检索证据和回答时同时判断：证据是否直接支持原始用户问题；如果某个角度证据不足，"
        "请明确说明不足，不要用相邻主题替代。"
    )


def _should_run_paperqa_reader(
    *,
    route: dict[str, Any],
    selected: list[SelectedPaper],
    args: Any,
) -> tuple[bool, str]:
    """Expert-owned policy for calling PaperQA reader tools.

    Router chooses whether ExpertResearchAgent should run. Once inside the
    expert, PDF reading is a tool-level decision and should not be hard-disabled
    by a possibly mistaken router boolean.
    """

    if not selected:
        return False, "no_selected_papers"
    if getattr(args, "dry_run", False):
        return False, "dry_run"
    if getattr(args, "download_only", False):
        return False, "download_only"
    if not route.get("need_paper_search", True):
        return False, "paper_search_not_requested"

    task_types = set(route.get("task_types") or [])
    task_type = route.get("task_type")
    if task_type:
        task_types.add(str(task_type))
    if task_types & PAPERQA_READING_TASKS:
        return True, "expert_policy_task_requires_evidence"
    if route.get("need_pdf_reading", False):
        return True, "router_requested_pdf_reading"

    return False, "task_does_not_require_pdf_reading"


def _metadata_search_tool(
    queries: list[str],
    args: Any,
) -> tuple[list[SearchResult], list[SelectedPaper], dict[str, Any]]:
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
        require_pdf=not args.allow_no_pdf,
    )
    with MetadataSearchEngine(corpora=corpora, filters=filters) as engine:
        result_groups = [
            engine.search(query, candidate_k=args.per_query_k)
            for query in queries
        ]
    candidates = reciprocal_rank_fusion(result_groups, top_k=args.candidate_k)
    selected = select_for_paperqa(candidates, top_k=args.paperqa_k)
    trace = {
        "query_count": len(queries),
        "candidate_count": len(candidates),
        "selected_count": len(selected),
        "corpora": [corpus.name for corpus in corpora],
    }
    return candidates, selected, trace


def _selected_key(item: SelectedPaper) -> str:
    result = item.result
    return (
        result.doi
        or result.arxiv_id
        or result.pdf_url
        or result.paper_url
        or normalize_title_safe(result.title)
    ).lower()


def normalize_title_safe(title: str) -> str:
    return " ".join(str(title or "").lower().split())


def _merge_selected_papers(
    *,
    memory_selected: list[SelectedPaper],
    web_selected: list[SelectedPaper],
    metadata_selected: list[SelectedPaper],
    top_k: int,
) -> list[SelectedPaper]:
    """Merge selected papers from external discovery and local metadata."""

    merged: list[SelectedPaper] = []
    seen: set[str] = set()
    for group in (memory_selected, web_selected, metadata_selected):
        for item in group:
            key = _selected_key(item)
            if not key or key in seen:
                continue
            seen.add(key)
            merged.append(item)
    merged.sort(key=lambda item: item.selection_score, reverse=True)
    return merged[:top_k]


def _selected_from_result_dicts(items: list[dict[str, Any]]) -> list[SelectedPaper]:
    selected: list[SelectedPaper] = []
    for item in items:
        try:
            result = SearchResult(**item)
        except TypeError:
            continue
        if not result.pdf_url:
            continue
        selected.append(
            SelectedPaper(
                result=result,
                selection_score=float(result.hybrid_score or result.relevance_score or 0.0),
            )
        )
    return selected


def _selected_from_search_results(results: list[SearchResult]) -> list[SelectedPaper]:
    selected: list[SelectedPaper] = []
    for result in results:
        if not result.pdf_url:
            continue
        selected.append(
            SelectedPaper(
                result=result,
                selection_score=float(result.hybrid_score or result.relevance_score or 0.0),
            )
        )
    return selected


def _cache_pdfs_tool(
    selected: list[SelectedPaper],
    args: Any,
) -> dict[str, Any]:
    already_cached: list[SelectedPaper] = []
    needs_download: list[SelectedPaper] = []
    for item in selected:
        if item.cache_path and Path(item.cache_path).exists():
            item.cache_status = item.cache_status or "cache_hit"
            if item.cache_status not in {"cache_hit", "downloaded"}:
                item.cache_status = "cache_hit"
            already_cached.append(item)
        else:
            needs_download.append(item)

    cache_selected_pdfs(
        needs_download,
        cache_dir=args.pdf_cache_dir,
        timeout=args.download_timeout,
        request_delay=args.request_delay,
        max_pdf_mb=args.max_pdf_mb,
        retries=args.download_retries,
    )
    cache_counts: dict[str, int] = {}
    for item in [*already_cached, *needs_download]:
        cache_counts[item.cache_status] = cache_counts.get(item.cache_status, 0) + 1
    return {"cache_counts": cache_counts}


def _expert_llm_model(args: Any) -> str | None:
    model = args.agent_llm or args.router_llm or args.llm
    if not model:
        return None
    if args.openai_base_url and not args.disable_openai_compatible_config:
        return normalize_openai_compatible_model(model, args, args.openai_base_url)
    return model


async def _summarize_web_findings(
    question: str,
    web_evidence: list[dict[str, Any]],
    args: Any,
) -> tuple[str, dict[str, Any]]:
    if not web_evidence:
        return "", {"mode": "empty", "error": None}

    fallback = (
        "web_search_tool 找到的补充线索：\n"
        f"{format_web_evidence(web_evidence, limit=5)}"
    )
    model = _expert_llm_model(args)
    if not model:
        return fallback, {"mode": "fallback_no_llm", "error": None}

    source_text = "\n".join(
        (
            f"[{idx}] title={item.get('title', '')}\n"
            f"url={item.get('url', '')}\n"
            f"snippet={item.get('snippet', '')}"
        )
        for idx, item in enumerate(web_evidence[:10], start=1)
    )
    try:
        import litellm

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are the web-reading part of an ExpertResearchAgent. "
                        "Given search snippets and URLs, extract only useful fresh context "
                        "for a literature review. Do not overclaim beyond snippets. "
                        "Mention URLs inline when they are useful. Answer in Chinese."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"用户问题：{question}\n\n"
                        "Web search results:\n"
                        f"{source_text}\n\n"
                        "请总结：1) 发现了哪些可能相关的新论文/项目/代码/榜单；"
                        "2) 哪些只能作为线索、不能当论文证据；3) 后续应优先读哪些。"
                    ),
                },
            ],
            "temperature": 0,
            "max_tokens": 900,
            "timeout": float(getattr(args, "llm_timeout", 180.0)),
        }
        if args.openai_base_url and not args.disable_openai_compatible_config:
            kwargs["api_base"] = args.openai_base_url
            api_key = openai_compatible_api_key(
                args,
                args.openai_base_url,
                args.openai_api_key_env,
            )
            if api_key:
                kwargs["api_key"] = api_key
        response = await litellm.acompletion(**kwargs)
        content = (response.choices[0].message.content or "").strip()
        return content or fallback, {"mode": "llm", "model": model, "error": None}
    except Exception as exc:
        return fallback, {
            "mode": "fallback_after_llm_error",
            "model": model,
            "error": shorten(str(exc), width=300, placeholder="..."),
        }


async def run_expert_research_agent(
    *,
    question: str,
    queries: list[str],
    research_plan: list[dict[str, Any]] | None = None,
    research_questions: list[dict[str, Any]] | None = None,
    perspectives: list[dict[str, Any]] | None = None,
    planning_web_evidence: list[dict[str, Any]] | None = None,
    route: dict[str, Any],
    args: Any,
) -> dict[str, Any]:
    """Run the enhanced STORM-style expert over retrieval and reading tools.

    Perspective generation and question planning happen before this agent. This
    agent receives those planned queries and acts as the research expert that
    decides how to use retrieval tools: web search, metadata search, PDF caching,
    and PaperQA's reader tools.
    """

    expert_trace: list[dict[str, Any]] = []
    web_evidence: list[dict[str, Any]] = []
    web_search_trace: dict[str, Any] = {}
    web_findings = ""
    candidates: list[SearchResult] = []
    memory_selected: list[SelectedPaper] = []
    memory_paper_trace: dict[str, Any] = {}
    metadata_selected: list[SelectedPaper] = []
    academic_selected: list[SelectedPaper] = []
    academic_paper_search_trace: dict[str, Any] = {}
    web_selected: list[SelectedPaper] = []
    selected: list[SelectedPaper] = []
    triage_trace: dict[str, Any] = {}
    paperqa_answer = ""
    paperqa_trace: dict[str, Any] = {}
    web_paper_discovery_trace: dict[str, Any] = {}
    paper_search_trace: dict[str, Any] = {}
    paper_search_web_evidence: list[dict[str, Any]] = {}
    research_plan = research_plan or []
    research_questions = research_questions or []
    planning_web_evidence = planning_web_evidence or []

    should_try_memory_papers = bool(
        route.get("need_paper_search", True) or route.get("need_pdf_reading", False)
    )
    if should_try_memory_papers:
        emit_progress(args, "MemoryPaperTool", "start")
        memory_selected, memory_paper_trace = await run_memory_paper_tool(
            question=question,
            research_plan=research_plan,
            research_questions=research_questions,
            args=args,
        )
        emit_progress(
            args,
            "MemoryPaperTool",
            "done",
            candidates=memory_paper_trace.get("candidate_count"),
            selected=memory_paper_trace.get("selected_count"),
            sufficient=memory_paper_trace.get("sufficient_for_question"),
        )
        expert_trace.append(
            {
                "tool": "memory_paper_tool",
                "candidate_count": memory_paper_trace.get("candidate_count"),
                "selected_count": memory_paper_trace.get("selected_count"),
                "sufficient_for_question": memory_paper_trace.get("sufficient_for_question"),
                "selected_titles": memory_paper_trace.get("selected_titles"),
                "reason": memory_paper_trace.get("reason"),
                "error": memory_paper_trace.get("error"),
            }
        )

    memory_papers_sufficient = bool(
        memory_selected and memory_paper_trace.get("sufficient_for_question")
    )

    if (
        route.get("need_web_search", False)
        and not args.disable_web_search
        and not memory_papers_sufficient
    ):
        web_queries = web_queries_for_research(question, queries, research_plan)
        emit_progress(
            args,
            "ExpertResearchAgent",
            "web search start",
            queries=len(web_queries),
        )
        web_evidence, web_search_trace = await run_web_search_agent(web_queries, args)
        emit_progress(
            args,
            "ExpertResearchAgent",
            "web search done",
            results=web_search_trace.get("result_count"),
            errors=len(web_search_trace.get("errors", [])),
            provider=web_search_trace.get("provider"),
        )
        expert_trace.append(
            {
                "tool": "web_search_tool",
                "query_count": web_search_trace.get("query_count"),
                "result_count": web_search_trace.get("result_count"),
                "error_count": len(web_search_trace.get("errors", [])),
            }
        )
        emit_progress(args, "ExpertResearchAgent", "web synthesis start")
        web_findings, web_findings_trace = await _summarize_web_findings(
            question,
            web_evidence,
            args,
        )
        emit_progress(
            args,
            "ExpertResearchAgent",
            "web synthesis done",
            mode=web_findings_trace.get("mode"),
            error=web_findings_trace.get("error"),
        )
        expert_trace.append(
            {
                "tool": "web_synthesis",
                "mode": web_findings_trace.get("mode"),
                "error": web_findings_trace.get("error"),
            }
        )

    triage_candidates = [
        *memory_selected,
        *_selected_from_search_results(candidates),
    ]
    if memory_papers_sufficient:
        selected = memory_selected[: int(getattr(args, "paperqa_k", 8))]
        triage_trace = {
            "mode": "memory_paper_tool_sufficient",
            "candidate_count": len(memory_selected),
            "selected_count": len(selected),
            "selected_titles": [item.result.title for item in selected],
            "rationale": memory_paper_trace.get("reason"),
        }
        expert_trace.append(
            {
                "tool": "paper_triage_agent",
                "mode": triage_trace.get("mode"),
                "candidate_count": triage_trace.get("candidate_count"),
                "selected_count": triage_trace.get("selected_count"),
                "rationale": triage_trace.get("rationale"),
            }
        )
    elif route.get("need_paper_search", True):
        max_search_triage_rounds = max(
            1,
            int(getattr(args, "paper_search_triage_rounds", 2) or 2),
        )
        current_queries = list(queries)
        current_research_plan = list(research_plan)
        feedback_history: list[dict[str, Any]] = []
        seen_candidate_keys: set[str] = set()
        all_candidates: list[SearchResult] = []

        for round_index in range(1, max_search_triage_rounds + 1):
            emit_progress(
                args,
                "PaperSearchAgent",
                "start",
                round=round_index,
                research_queries=len(current_queries),
            )
            (
                round_candidates,
                metadata_selected,
                paper_search_web_evidence,
                round_paper_search_trace,
            ) = await run_paper_search_agent(
                question=question,
                queries=current_queries,
                research_plan=current_research_plan,
                planning_web_evidence=planning_web_evidence,
                args=args,
            )
            paper_search_trace = round_paper_search_trace
            emit_progress(
                args,
                "PaperSearchAgent",
                "done",
                round=round_index,
                candidates=paper_search_trace.get("candidate_count"),
                workflow_candidates=paper_search_trace.get("workflow_candidate_count"),
                metadata_candidates=paper_search_trace.get("metadata_candidate_count"),
                stop=paper_search_trace.get("stop_reason"),
            )
            academic_paper_search_trace = paper_search_trace
            if paper_search_web_evidence:
                seen_urls = {str(item.get("url") or "") for item in web_evidence}
                for item in paper_search_web_evidence:
                    url = str(item.get("url") or "")
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        web_evidence.append(item)

            for candidate in round_candidates:
                key = (
                    candidate.doi
                    or candidate.arxiv_id
                    or candidate.pdf_url
                    or candidate.paper_url
                    or candidate.title
                )
                key = " ".join(str(key or "").lower().split())
                if key and key not in seen_candidate_keys:
                    seen_candidate_keys.add(key)
                    all_candidates.append(candidate)
            candidates = all_candidates
            academic_selected = _selected_from_search_results(candidates)
            triage_candidates = [
                *memory_selected,
                *_selected_from_search_results(candidates),
            ]
            expert_trace.append(
                {
                    "tool": "paper_search_agent",
                    "round": round_index,
                    "task_signals": (paper_search_trace.get("query_analysis") or {}).get("task_signals"),
                    "authors": (paper_search_trace.get("query_analysis") or {}).get("authors"),
                    "called_tools": (paper_search_trace.get("workflow_trace") or {}).get("called_tools"),
                    "candidate_count": paper_search_trace.get("candidate_count"),
                    "accumulated_candidate_count": len(candidates),
                    "workflow_candidate_count": paper_search_trace.get("workflow_candidate_count"),
                    "metadata_candidate_count": paper_search_trace.get("metadata_candidate_count"),
                    "selected_count": paper_search_trace.get("selected_count"),
                }
            )

            emit_progress(
                args,
                "PaperTriageAgent",
                "start",
                round=round_index,
                candidates=len(triage_candidates),
            )
            selected, triage_trace = await triage_papers_for_reading(
                question=question,
                candidates=triage_candidates,
                web_evidence=web_evidence,
                planning_web_evidence=planning_web_evidence,
                research_plan=current_research_plan,
                args=args,
            )
            emit_progress(
                args,
                "PaperTriageAgent",
                "done",
                round=round_index,
                selected=triage_trace.get("selected_count"),
                mode=triage_trace.get("mode"),
                error=triage_trace.get("error"),
            )
            expert_trace.append(
                {
                    "tool": "paper_triage_agent",
                    "round": round_index,
                    "mode": triage_trace.get("mode"),
                    "candidate_count": triage_trace.get("candidate_count"),
                    "selected_count": triage_trace.get("selected_count"),
                    "model": triage_trace.get("model"),
                    "sufficient_for_reading": triage_trace.get("sufficient_for_reading"),
                    "requires_followup_search": triage_trace.get("requires_followup_search"),
                    "rationale": triage_trace.get("rationale"),
                    "coverage_notes": triage_trace.get("coverage_notes"),
                    "error": triage_trace.get("error"),
                }
            )
            should_continue, feedback = _triage_feedback_for_search(
                triage_trace=triage_trace,
                selected=selected,
                args=args,
            )
            feedback_history.append(
                {
                    "round": round_index,
                    "should_continue": should_continue,
                    "feedback": feedback,
                    "selected_count": len(selected),
                }
            )
            if not should_continue or round_index >= max_search_triage_rounds:
                break

            emit_progress(
                args,
                "PaperSearchAgent",
                "triage feedback",
                round=round_index,
                feedback=shorten(feedback, width=160, placeholder="..."),
            )
            feedback_query = (
                "Follow-up paper search based on PaperTriageAgent feedback: "
                + feedback
            )
            current_queries = [feedback_query, *current_queries]
            current_research_plan = [
                {
                    "agent": "PaperTriageAgentFeedback",
                    "research_question": feedback,
                    "depends_on": "paper_search_and_triage",
                },
                *current_research_plan,
            ]

        if paper_search_trace:
            paper_search_trace = {
                **paper_search_trace,
                "search_triage_loop": {
                    "mode": "paper_search_triage_feedback_loop",
                    "round_count": len(feedback_history),
                    "max_rounds": max_search_triage_rounds,
                    "feedback_history": feedback_history,
                    "final_selected_count": len(selected),
                },
            }
            academic_paper_search_trace = paper_search_trace
        web_paper_discovery_trace = {
            "mode": "removed_from_main_chain",
            "reason": "heuristic web-page PDF extraction was replaced by PaperSearchAgent workflows",
            "candidate_count": 0,
            "selected_count": 0,
        }
    else:
        selected = _merge_selected_papers(
            memory_selected=memory_selected,
            web_selected=[*academic_selected, *web_selected],
            metadata_selected=metadata_selected,
            top_k=int(getattr(args, "paperqa_k", 8)),
        )
    if memory_selected or academic_selected or web_selected or metadata_selected:
        expert_trace.append(
            {
                "tool": "paper_selection_merge",
                "memory_selected_count": len(memory_selected),
                "academic_selected_count": len(academic_selected),
                "heuristic_web_selected_count": len(web_selected),
                "metadata_selected_count": len(metadata_selected),
                "triage_candidate_count": len(triage_candidates),
                "final_selected_count": len(selected),
            }
        )

    should_read_pdf, read_reason = _should_run_paperqa_reader(
        route=route,
        selected=selected,
        args=args,
    )

    if selected and not args.dry_run:
        emit_progress(args, "PDFCacheTool", "start", selected=len(selected))
        cache_trace = _cache_pdfs_tool(selected, args)
        emit_progress(
            args,
            "PDFCacheTool",
            "done",
            cache_counts=cache_trace.get("cache_counts"),
        )
        expert_trace.append({"tool": "pdf_cache_tool", **cache_trace})

        if should_read_pdf:
            try:
                emit_progress(
                    args,
                    "PaperQAReader",
                    "start",
                    selected=len(selected),
                    reason=read_reason,
                )
                paperqa_question = _paperqa_question_for_research(
                    question,
                    research_questions,
                    research_plan,
                )
                paperqa_answer, paperqa_trace = await run_paperqa_reader_adapter(
                    question=paperqa_question,
                    user_question=question,
                    selected=selected,
                    metadata_queries=queries,
                    args=args,
                )
                emit_progress(
                    args,
                    "PaperQAReader",
                    "done",
                    mode=paperqa_trace.get("mode"),
                    evidence=paperqa_trace.get("evidence_count"),
                    status=paperqa_trace.get("status"),
                )
                expert_trace.append(
                    {
                        "tool": "paperqa_reader_adapter",
                        "mode": paperqa_trace.get("mode"),
                        "adapter_mode": paperqa_trace.get("adapter_mode"),
                        "fallback_used": paperqa_trace.get("fallback_used"),
                        "status": paperqa_trace.get("status"),
                        "decision": read_reason,
                        "tool_steps": len(paperqa_trace.get("tool_trace", [])),
                    }
                )
                if paperqa_question != question:
                    paperqa_trace["user_question"] = question
                    paperqa_trace["reader_question"] = paperqa_question
            except Exception as exc:
                paperqa_trace = {
                    "mode": "langgraph-paperqa-tools",
                    "status": "error",
                    "error": shorten(str(exc), width=500, placeholder="..."),
                    "tool_trace": [],
                }
                expert_trace.append(
                    {
                        "tool": "paperqa_reader_adapter",
                        "status": "error",
                        "error": paperqa_trace["error"],
                    }
                )
                emit_progress(
                    args,
                    "PaperQAReader",
                    "error",
                    error=paperqa_trace["error"],
                )
        else:
            emit_progress(args, "PaperQAReader", "skipped", reason=read_reason)
            expert_trace.append(
                {
                    "tool": "paperqa_reader_adapter",
                    "skipped": read_reason,
                }
            )
    elif selected:
        emit_progress(args, "PDFCacheTool", "skipped dry_run", selected=len(selected))
        expert_trace.append(
            {
                "tool": "pdf_cache_tool",
                "skipped": "dry_run",
                "selected_count": len(selected),
            }
        )

    return {
        "web_evidence": web_evidence,
        "web_search_trace": web_search_trace,
        "web_findings": web_findings,
        "research_plan": research_plan,
        "memory_paper_candidates": [item.to_dict() for item in memory_selected],
        "memory_paper_trace": memory_paper_trace,
        "academic_paper_candidates": [item.to_dict() for item in academic_selected],
        "academic_paper_search_trace": academic_paper_search_trace,
        "paper_search_trace": paper_search_trace,
        "paper_triage_trace": triage_trace,
        "web_paper_candidates": [item.to_dict() for item in web_selected],
        "web_paper_discovery_trace": web_paper_discovery_trace,
        "metadata_candidates": [result.to_dict() for result in candidates],
        "candidates": [result.to_dict() for result in candidates],
        "selected": [item.to_dict() for item in selected],
        "paperqa_answer": paperqa_answer,
        "paperqa_trace": paperqa_trace,
        "expert_trace": expert_trace,
    }
