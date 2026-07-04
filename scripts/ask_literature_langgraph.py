#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from textwrap import shorten
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from embodiedai_kb.langgraph_workflow import build_literature_graph
from embodiedai_kb.search.metadata_search import SearchResult
from scripts.ask_literature import (
    DEFAULT_RUN_JSON,
    SelectedPaper,
    parse_args,
    print_search_summary,
)


DEFAULT_LANGGRAPH_RUN_JSON = ROOT / "data/metadata/ask_literature_langgraph_last.json"


def _selected_from_dicts(items: list[dict[str, Any]]) -> list[SelectedPaper]:
    """Convert JSON-like LangGraph state back into helper dataclasses."""

    selected: list[SelectedPaper] = []
    for item in items:
        selected.append(
            SelectedPaper(
                result=SearchResult(**item["result"]),
                selection_score=item["selection_score"],
                cache_path=item.get("cache_path"),
                cache_status=item.get("cache_status", "not_downloaded"),
                error=item.get("error"),
            )
        )
    return selected


def write_langgraph_run(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _candidate_debug_rows(items: list[dict[str, Any]], *, limit: int | None = None) -> list[dict[str, Any]]:
    """Compact paper candidate view for debugging retrieval/triage."""

    rows: list[dict[str, Any]] = []
    for idx, item in enumerate(items[:limit] if limit else items, start=1):
        result = item.get("result", item)
        authors = result.get("authors") or []
        if isinstance(authors, str):
            authors = [authors]
        rows.append(
            {
                "debug_rank": idx,
                "rank": result.get("rank"),
                "title": result.get("title"),
                "authors": authors[:10],
                "year": result.get("year"),
                "venue": result.get("venue"),
                "corpus": result.get("corpus"),
                "sources": result.get("sources") or [],
                "paper_id": result.get("paper_id"),
                "arxiv_id": result.get("arxiv_id"),
                "doi": result.get("doi"),
                "pdf_url": result.get("pdf_url"),
                "paper_url": result.get("paper_url"),
                "hybrid_score": result.get("hybrid_score"),
                "citation_count": result.get("citation_count"),
            }
        )
    return rows


def _print_candidate_debug(title: str, rows: list[dict[str, Any]], *, limit: int = 30) -> None:
    if not rows:
        return
    print(f"{title}: {len(rows)}")
    for row in rows[:limit]:
        authors = ", ".join(row.get("authors") or []) or "Unknown authors"
        venue = row.get("venue") or row.get("corpus") or "?"
        sources = ", ".join(row.get("sources") or [])
        print(
            f"  {row['debug_rank']:02d}. {row.get('year') or '?'} {venue} | "
            f"{row.get('title') or 'Untitled'}"
        )
        print(f"      authors={authors}")
        if sources:
            print(f"      sources={sources}")
        if row.get("pdf_url"):
            print(f"      pdf={row.get('pdf_url')}")
    print()


def _one_line(value: Any, *, width: int = 180) -> str:
    return shorten(str(value or "").replace("\n", " "), width=width, placeholder="...")


def _paper_title_from_record(record: dict[str, Any]) -> str:
    return str(record.get("title") or Path(str(record.get("path") or "")).name or "Untitled")


def _print_paperqa_reader_debug(trace: dict[str, Any]) -> None:
    """Print the high-signal PaperQA adapter trace for terminal debugging."""

    print("\nPaperQA reader debug:")
    if not trace:
        print("  no paperqa_trace recorded")
        return

    print(
        "  "
        f"mode={trace.get('mode')} "
        f"adapter={trace.get('adapter_mode')} "
        f"fallback={trace.get('fallback_used')} "
        f"status={trace.get('status')} "
        f"answer_mode={trace.get('answer_mode')} "
        f"evidence={trace.get('evidence_count')} "
        f"sufficient={trace.get('sufficient')} "
        f"reason={trace.get('sufficiency_reason') or trace.get('fallback_reason')}"
    )
    if trace.get("error"):
        print(f"  error={_one_line(trace.get('error'), width=260)}")

    native_trace = trace.get("native_agent_trace")
    if native_trace:
        print(
            "  native_agent: "
            f"status={native_trace.get('status')} "
            f"sufficient={native_trace.get('sufficient')} "
            f"reason={native_trace.get('sufficiency_reason')} "
            f"evidence={native_trace.get('evidence_count')} "
            f"relevant_papers={native_trace.get('relevant_paper_count')} "
            f"self_success={native_trace.get('has_successful_answer')}"
        )
        if native_trace.get("error"):
            print(f"    error={_one_line(native_trace.get('error'), width=260)}")
    elif trace.get("mode") == "paperqa-agent":
        print(
            "  native_agent: "
            f"agent_type={trace.get('agent_type')} "
            f"evidence={trace.get('evidence_count')} "
            f"relevant_papers={trace.get('relevant_paper_count')} "
            f"self_success={trace.get('has_successful_answer')}"
        )

    papers = trace.get("agent_papers") or []
    if papers:
        print(f"  pdfs={len(papers)}")
        for idx, record in enumerate(papers[:10], start=1):
            print(f"    {idx}. {_paper_title_from_record(record)}")

    if trace.get("paper_search_query_limit") is not None:
        print(
            "  explicit_tools_config: "
            f"search_query_limit={trace.get('paper_search_query_limit')} "
            f"title_query_count={trace.get('paperqa_title_query_count')} "
            f"per_paper_evidence={trace.get('paperqa_per_paper_evidence_count')} "
            f"evidence_contexts={len(trace.get('evidence_contexts') or [])}"
        )

    tool_trace = trace.get("tool_trace") or []
    if not tool_trace:
        return
    print(f"  tool_steps={len(tool_trace)}")
    for idx, step in enumerate(tool_trace[:40], start=1):
        tool = step.get("tool")
        if tool == "agent_action":
            print(
                f"    {idx:02d}. native action | {step.get('status')} | "
                f"{_one_line(step.get('summary'), width=220)}"
            )
        elif tool == "env_step":
            print(
                f"    {idx:02d}. native step | done={step.get('done')} "
                f"truncated={step.get('truncated')} reward={step.get('reward')} | "
                f"{_one_line(step.get('output_preview'), width=220)}"
            )
        elif tool == "paper_search":
            print(
                f"    {idx:02d}. paper_search | {step.get('status')} | "
                f"query={_one_line(step.get('query'), width=160)}"
            )
            if step.get("output_preview"):
                print(f"         output={_one_line(step.get('output_preview'), width=240)}")
        elif tool == "gather_evidence":
            scope = step.get("scope") or "unknown"
            title = step.get("paper_title")
            title_part = f" | paper={_one_line(title, width=80)}" if title else ""
            print(
                f"    {idx:02d}. gather_evidence[{scope}] | {step.get('status')}"
                f"{title_part}"
            )
            if step.get("question_preview"):
                print(f"         question={_one_line(step.get('question_preview'), width=240)}")
            if step.get("output_preview"):
                print(f"         output={_one_line(step.get('output_preview'), width=240)}")
        elif tool == "paperqa_agent_reader":
            print(
                f"    {idx:02d}. native_agent_result | status={step.get('status')} "
                f"sufficient={step.get('sufficient')} reason={step.get('reason')}"
            )
        else:
            print(
                f"    {idx:02d}. {tool} | {step.get('status')} | "
                f"{_one_line(step.get('output_preview') or step, width=240)}"
            )
    if len(tool_trace) > 40:
        print(f"    ... {len(tool_trace) - 40} more tool steps omitted")


async def async_main() -> None:
    # 主入口只负责三件事：
    # 1. 解析命令行参数；
    # 2. 构建并启动 LangGraph；
    # 3. 把 LangGraph 返回的 state 打印/保存下来。
    # 具体每一步怎么搜索、下载、读 PDF，主要在 graph.py、expert_research.py
    # 和 paperqa_bridge.py 里。
    # Reuse the original ask_literature CLI so both demos accept the same flags.
    args = parse_args()

    # Keep the LangGraph run record separate from the older PaperQA-only script.
    if args.run_json == DEFAULT_RUN_JSON:
        args.run_json = DEFAULT_LANGGRAPH_RUN_JSON

    # The default PaperQA embedding is remote OpenAI. For our local demo, sparse
    # retrieval is cheaper and avoids needing an embedding API key.
    if not args.embedding:
        args.embedding = "sparse"

    # build_literature_graph 会返回一个已经 compile() 过的 LangGraph runnable。
    # ainvoke() 这一行是真正“启动工作流”的地方：LangGraph 会从初始 state
    # 出发，由 graph.py 里的 Planner/Supervisor 循环派发 worker agent，
    # 直到进入最终 SynthesisAgent。
    graph = build_literature_graph(args)
    state = await graph.ainvoke({"question": args.question, "trace": []})

    # 从这里开始主要是展示和落盘，不再执行新的 agent 逻辑。
    # 因为 LangGraph state 为了方便保存成 JSON，里面的候选论文/选中论文都是 dict；
    # 这里转回 dataclass，只是为了复用原 ask_literature.py 里的打印函数。
    metadata_candidate_dicts = state.get("metadata_candidates") or state.get("candidates", [])
    candidates = [SearchResult(**item) for item in metadata_candidate_dicts]
    selected = _selected_from_dicts(state.get("selected", []))

    academic_candidate_dicts = (
        state.get("academic_paper_search_trace", {}).get("candidates") or []
    )
    paper_candidate_debug = {
        "metadata_top_candidates": _candidate_debug_rows(
            metadata_candidate_dicts,
            limit=int(getattr(args, "candidate_k", 30)),
        ),
        "academic_search_candidates": _candidate_debug_rows(
            academic_candidate_dicts,
            limit=int(getattr(args, "paper_triage_candidate_limit", 60)),
        ),
        "metadata_candidate_count": len(metadata_candidate_dicts),
        "academic_search_candidate_count": len(academic_candidate_dicts),
    }
    state["paper_candidate_debug"] = paper_candidate_debug

    memory_load_trace = state.get("memory_load_trace", {})
    if memory_load_trace:
        print("Thread memory:")
        print(
            "  "
            f"thread_id={memory_load_trace.get('thread_id')} "
            f"mode={memory_load_trace.get('mode')} "
            f"loaded={memory_load_trace.get('loaded_count')} "
            f"used={memory_load_trace.get('used_count', 0)} "
            f"reset={memory_load_trace.get('reset')} "
            f"path={memory_load_trace.get('path')}"
        )
        packet_trace = memory_load_trace.get("packet") or {}
        if packet_trace:
            print(
                "  "
                f"packet_episodes={packet_trace.get('recent_episode_count')} "
                f"latest_papers={packet_trace.get('latest_selected_paper_count')} "
                f"recent_papers={packet_trace.get('recent_selected_paper_count')}"
            )
        if state.get("memory_context"):
            print("  compact memory_context loaded for follow-up resolution")
        print()

    route = state.get("route", {})
    if route:
        print("Planner decision:")
        print(
            "  "
            f"task={route.get('task_type')} "
            f"tasks={route.get('task_types')} "
            f"paper={route.get('need_paper_search')} "
            f"pdf={route.get('need_pdf_reading')} "
            f"web={route.get('need_web_search')} "
            f"learning={route.get('need_learning_plan')} "
            f"idea={route.get('need_idea_generation')} "
            f"source={route.get('source')} "
            f"confidence={route.get('confidence')}"
        )
        if route.get("reasoning"):
            print(f"  reasoning={route.get('reasoning')}")
        print()

    plan = state.get("plan", {})
    if plan:
        print("Planner workflow:")
        if plan.get("workflow_goal"):
            print(f"  goal={plan.get('workflow_goal')}")
        for idx, step in enumerate(plan.get("steps", []), start=1):
            depends_on = step.get("depends_on") or []
            depends = ",".join(depends_on) if depends_on else "-"
            print(
                "  "
                f"{idx}. {step.get('id')} -> {step.get('agent')} "
                f"[depends_on={depends}, "
                f"granularity={step.get('granularity', 'coarse')}, "
                f"status={step.get('status', 'pending')}]"
            )
            if step.get("objective"):
                print(f"     {step.get('objective')}")
            for substep in step.get("substeps") or []:
                print(f"     - {substep}")
        if plan.get("next_step"):
            print(f"  next_step={plan.get('next_step')}")
        print()

    research_plan = state.get("research_plan", [])
    if research_plan:
        print("ResearchPlanningAgent plan:")
        for idx, item in enumerate(research_plan, start=1):
            print(
                f"  {idx}. {item.get('perspective')}: "
                f"{item.get('description')} [{item.get('source')}]"
            )
            if item.get("research_question"):
                print(f"     question={item.get('research_question')}")
            targets = item.get("coverage_targets") or []
            if targets:
                print(f"     targets={', '.join(str(target) for target in targets[:5])}")
            signals = item.get("signal_sources") or []
            if signals:
                print(f"     signals={', '.join(str(signal) for signal in signals[:5])}")
            for query in item.get("queries", []):
                print(f"     - {query}")
        print()

    planning_web_evidence = state.get("planning_web_evidence", [])
    if planning_web_evidence:
        trace = state.get("planning_web_trace", {})
        print(
            "ResearchPlanningAgent web scouting: "
            f"{len(planning_web_evidence)} results "
            f"from {trace.get('query_count', 0)} queries "
            f"via {trace.get('provider_sequence') or trace.get('provider', 'unknown')}"
        )
        for idx, item in enumerate(planning_web_evidence[:6], start=1):
            title = item.get("title") or "Untitled"
            url = item.get("url") or ""
            snippet = item.get("snippet") or ""
            print(f"  {idx}. {title}")
            if snippet:
                print(f"     {snippet[:260]}")
            print(f"     {url}")
        print()

    research_questions = state.get("research_questions", [])
    if research_questions:
        print("Compatibility research questions:")
        for idx, item in enumerate(research_questions, start=1):
            print(f"  {idx}. ({item.get('perspective')}) {item.get('question')}")
            for query in item.get("queries", []):
                print(f"     - {query}")
        print()

    web_evidence = state.get("web_evidence", [])
    if web_evidence:
        trace = state.get("web_search_trace", {})
        print(
            "ExpertResearchAgent web evidence: "
            f"{len(web_evidence)} results "
            f"from {trace.get('query_count', 0)} queries "
            f"via {trace.get('provider_sequence') or trace.get('provider', 'unknown')}"
        )
        for idx, item in enumerate(web_evidence[:8], start=1):
            title = item.get("title") or "Untitled"
            url = item.get("url") or ""
            snippet = item.get("snippet") or ""
            print(f"  {idx}. {title}")
            if snippet:
                print(f"     {snippet}")
            print(f"     {url}")
        errors = trace.get("errors") or []
        if errors:
            print(f"  web_search_errors={len(errors)}")
        print()

    if state.get("web_findings"):
        print("ExpertResearchAgent web findings:")
        print(state["web_findings"])
        print()

    paper_search_trace = state.get("paper_search_trace", {})
    if paper_search_trace:
        analysis = paper_search_trace.get("query_analysis") or {}
        print("PaperSearchAgent:")
        print(
            "  "
            f"task_signals={analysis.get('task_signals')} "
            f"authors={analysis.get('authors')} "
            f"mode={analysis.get('mode')} "
            f"workflow_candidates={paper_search_trace.get('workflow_candidate_count')} "
            f"metadata_candidates={paper_search_trace.get('metadata_candidate_count')} "
            f"total_candidates={paper_search_trace.get('candidate_count')}"
        )
        if analysis.get("error"):
            print(f"  query_analyzer_error={analysis.get('error')}")
        if analysis.get("raw_response_preview"):
            print("  query_analyzer_raw_preview:")
            print(f"    {analysis.get('raw_response_preview')}")
        if analysis.get("search_queries"):
            print("  search_queries:")
            for query in analysis.get("search_queries", [])[:10]:
                print(f"    - {query}")
        if analysis.get("tool_queries"):
            print("  tool_queries:")
            for tool, tool_queries in (analysis.get("tool_queries") or {}).items():
                if not tool_queries:
                    continue
                print(f"    {tool}:")
                for query in tool_queries[:8]:
                    print(f"      - {query}")
        if analysis.get("platform_queries"):
            print("  platform_queries:")
            for source, source_queries in (analysis.get("platform_queries") or {}).items():
                if not source_queries:
                    continue
                print(f"    {source}:")
                for query in source_queries[:8]:
                    print(f"      - {query}")
        if analysis.get("web_queries"):
            print("  web_queries:")
            for query in analysis.get("web_queries", [])[:10]:
                print(f"    - {query}")
        workflow_trace = paper_search_trace.get("workflow_trace") or {}
        if workflow_trace:
            print(f"  workflow={workflow_trace.get('mode')}")
            if workflow_trace.get("called_tools"):
                print(f"  called_tools={workflow_trace.get('called_tools')}")
            if workflow_trace.get("iteration_count") is not None:
                print(
                    "  search_loop="
                    f"{workflow_trace.get('iteration_count')}/"
                    f"{workflow_trace.get('max_iterations')} "
                    f"stop={workflow_trace.get('stop_reason')}"
                )
                for item in workflow_trace.get("iterations", [])[:4]:
                    print(
                        "    iter "
                        f"{item.get('iteration')}: "
                        f"tools={item.get('called_tools')} "
                        f"candidates={item.get('workflow_candidate_count')} "
                        f"pdf={item.get('pdf_candidate_count')} "
                        f"reason={item.get('reason')}"
                    )
                for item in workflow_trace.get("reflections", [])[:3]:
                    print(
                        "    reflect "
                        f"after_iter={item.get('after_iteration')} "
                        f"mode={item.get('mode')} "
                        f"trigger={item.get('trigger')}"
                    )
            academic_trace = workflow_trace.get("academic") or {}
            if academic_trace.get("source_results"):
                print(f"  source_results={academic_trace.get('source_results')}")
            errors = [
                *(workflow_trace.get("errors") or []),
                *(academic_trace.get("errors") or []),
                *((workflow_trace.get("web_profile") or {}).get("errors") or []),
            ]
            if errors:
                print(f"  workflow_errors={len(errors)}")
        print()

    if paper_candidate_debug.get("metadata_top_candidates") or paper_candidate_debug.get(
        "academic_search_candidates"
    ):
        print("Paper candidate debug:")
        _print_candidate_debug(
            "  Metadata top candidates",
            paper_candidate_debug["metadata_top_candidates"],
            limit=30,
        )
        _print_candidate_debug(
            "  Academic search candidates",
            paper_candidate_debug["academic_search_candidates"],
            limit=30,
        )

    paper_triage_trace = state.get("paper_triage_trace", {})
    if paper_triage_trace:
        print(
            "PaperTriageAgent: "
            f"mode={paper_triage_trace.get('mode')} "
            f"candidates={paper_triage_trace.get('candidate_count')} "
            f"selected={paper_triage_trace.get('selected_count')} "
            f"model={paper_triage_trace.get('model')}"
        )
        if paper_triage_trace.get("candidate_records"):
            print(
                "  candidate_records_saved="
                f"{len(paper_triage_trace.get('candidate_records') or [])} "
                f"prompt_chars={paper_triage_trace.get('candidate_prompt_char_count')} "
                f"abstract_max_chars={paper_triage_trace.get('abstract_max_chars')}"
            )
        if paper_triage_trace.get("rationale"):
            print(f"  rationale={paper_triage_trace.get('rationale')}")
        notes = paper_triage_trace.get("coverage_notes") or []
        for note in notes[:5]:
            print(f"  note={note}")
        if paper_triage_trace.get("selected_titles"):
            print("  selected_titles:")
            for title in paper_triage_trace.get("selected_titles", [])[:10]:
                print(f"    - {title}")
        if paper_triage_trace.get("error"):
            print(f"  error={paper_triage_trace.get('error')}")
        print()

    memory_paper_trace = state.get("memory_paper_trace", {})
    if memory_paper_trace:
        print(
            "MemoryPaperTool: "
            f"candidates={memory_paper_trace.get('candidate_count')} "
            f"selected={memory_paper_trace.get('selected_count')} "
            f"sufficient={memory_paper_trace.get('sufficient_for_question')}"
        )
        if memory_paper_trace.get("reason"):
            print(f"  reason={memory_paper_trace.get('reason')}")
        if memory_paper_trace.get("selected_titles"):
            print("  selected_titles:")
            for title in memory_paper_trace.get("selected_titles", [])[:8]:
                print(f"    - {title}")
        if memory_paper_trace.get("error"):
            print(f"  error={memory_paper_trace.get('error')}")
        print()

    academic_paper_candidates = state.get("academic_paper_candidates", [])
    if academic_paper_candidates:
        trace = state.get("academic_paper_search_trace", {})
        print(
            "Academic paper search: "
            f"{len(academic_paper_candidates)} selected OA/PDF candidates "
            f"from {trace.get('candidate_count', 0)} candidates "
            f"via {trace.get('sources_used', [])}"
        )
        if trace.get("source_results"):
            print(f"  source_results={trace.get('source_results')}")
        print(
            "  "
            f"resolved_pdf_count={trace.get('resolved_pdf_count', 0)} "
            f"unpaywall_enabled={trace.get('unpaywall_enabled')}"
        )
        if trace.get("queries"):
            print("  queries:")
            for query in trace.get("queries", [])[:12]:
                print(f"    - {query}")
        for idx, item in enumerate(academic_paper_candidates[:10], start=1):
            result = item.get("result", item)
            title = result.get("title") or "Untitled"
            venue = result.get("venue") or result.get("corpus") or "?"
            pdf_url = result.get("pdf_url") or ""
            source_url = result.get("paper_url") or ""
            sources = ", ".join(result.get("sources") or [])
            print(f"  {idx}. {title} ({result.get('year') or '?'}, {venue})")
            print(f"     sources={sources}")
            print(f"     pdf={pdf_url}")
            print(f"     source={source_url}")
        print()

    web_paper_candidates = state.get("web_paper_candidates", [])
    if web_paper_candidates:
        trace = state.get("web_paper_discovery_trace", {})
        print(
            "Web paper discovery: "
            f"{len(web_paper_candidates)} selected PDF candidates "
            f"from {trace.get('candidate_count', 0)} discovered candidates"
        )
        for idx, item in enumerate(web_paper_candidates[:10], start=1):
            result = item.get("result", item)
            title = result.get("title") or "Untitled"
            venue = result.get("venue") or result.get("corpus") or "?"
            pdf_url = result.get("pdf_url") or ""
            source_url = result.get("project_url") or result.get("paper_url") or ""
            print(f"  {idx}. {title} ({result.get('year') or '?'}, {venue})")
            print(f"     pdf={pdf_url}")
            print(f"     source={source_url}")
        print()

    if state.get("queries") or candidates or selected:
        print_search_summary(state.get("queries", []), candidates, selected)

    print("\nLangGraph trace:")
    for step in state.get("trace", []):
        print(f"  - {step.get('node')}: {step}")

    print("\nExpertResearchAgent tool trace:")
    for step in state.get("expert_trace", []):
        print(f"  - {step.get('tool')}: {step}")

    _print_paperqa_reader_debug(state.get("paperqa_trace", {}))

    if state.get("synthesis_trace"):
        print("\nSynthesisAgent trace:")
        print(f"  - {state.get('synthesis_trace')}")

    answer = state.get("final_answer", "")
    if answer:
        print("\n" + answer)

    memory_write_trace = state.get("memory_write_trace", {})
    if memory_write_trace:
        print(
            "\nMemory record: "
            f"written={memory_write_trace.get('written')} "
            f"thread_id={memory_write_trace.get('thread_id')} "
            f"path={memory_write_trace.get('path')}"
        )

    write_langgraph_run(args.run_json, dict(state))
    print(f"\nRun record: {args.run_json}")


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
