from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path
from textwrap import shorten
from typing import Any

from scripts.ask_literature import (
    PAPERQA_SRC,
    SelectedPaper,
    _jsonable,
    configure_openai_compatible_settings,
    prepare_agent_paper_dir,
    print_agent_progress,
    state_status,
    summarize_tool_request,
    validate_openai_compatible_config,
)


def configure_paperqa_settings(args: Any) -> Any:
    """Create PaperQA Settings from our CLI args.

    This mirrors the configuration used by scripts/ask_literature.py so the
    LangGraph path and the earlier PaperQA demo path behave the same way.
    """
    # 这里把我们 CLI 里的模型、embedding、base_url 等配置翻译成 PaperQA Settings。
    # 这样 LangGraph 版本和原来的 ask_literature.py 版本会使用同一套 LLM 配置。
    if str(PAPERQA_SRC) not in sys.path:
        sys.path.insert(0, str(PAPERQA_SRC))

    from paperqa import Settings

    settings = Settings()
    if args.llm:
        settings.llm = args.llm
    if args.summary_llm or args.llm:
        settings.summary_llm = args.summary_llm or args.llm
    if args.embedding:
        settings.embedding = args.embedding
    if args.agent_llm or args.llm:
        settings.agent.agent_llm = args.agent_llm or args.llm
    if args.llm:
        settings.parsing.enrichment_llm = args.llm

    validate_openai_compatible_config(args)
    configure_openai_compatible_settings(settings, args)

    if not args.parse_pdf_media:
        settings.parsing.multimodal = False
    settings.parsing.use_doc_details = args.allow_paperqa_metadata_lookup
    settings.answer.answer_max_sources = args.answer_max_sources
    settings.answer.evidence_k = args.evidence_k
    settings.answer.answer_length = args.answer_length
    settings.agent.search_count = args.agent_search_count
    settings.agent.timeout = args.agent_timeout
    settings.agent.return_paper_metadata = True
    if args.agent_max_timesteps is not None:
        settings.agent.max_timesteps = args.agent_max_timesteps
    return settings


def configure_paperqa_index(settings: Any, args: Any) -> None:
    """Point PaperQA's local search index at this run's PDF directory."""

    # PaperQA 的 paper_search 不是直接扫我们的 metadata 数据库，
    # 而是读取一个本地 PDF 目录并建立/打开索引。这里指定本轮运行的 PDF 目录、
    # manifest.csv，以及独立的 index 目录，避免不同问题之间索引串在一起。
    settings.agent.index.paper_directory = str(args.agent_paper_dir.resolve())
    settings.agent.index.use_absolute_paper_directory = True
    settings.agent.index.manifest_file = str(
        (args.agent_paper_dir / "manifest.csv").resolve()
    )
    index_directory = args.agent_paper_dir.parent / "indexes" / args.agent_paper_dir.name
    if index_directory.exists():
        shutil.rmtree(index_directory)
    index_directory.mkdir(parents=True, exist_ok=True)
    settings.agent.index.index_directory = str(index_directory.resolve())
    settings.agent.index.name = "papers"
    settings.agent.rebuild_index = True


def _positive_context_count(session: Any) -> int:
    return sum(1 for context in getattr(session, "contexts", []) if getattr(context, "score", 0) > 0)


def _relevant_doc_count(session: Any) -> int:
    doc_keys: set[str] = set()
    for context in getattr(session, "contexts", []):
        if getattr(context, "score", 0) <= 0:
            continue
        text = getattr(context, "text", None)
        doc = getattr(text, "doc", None)
        key = (
            getattr(doc, "dockey", None)
            or getattr(doc, "docname", None)
            or getattr(doc, "citation", None)
        )
        if key:
            doc_keys.add(str(key))
    return len(doc_keys)


def _paperqa_agent_sufficient(session: Any, status: Any, args: Any) -> tuple[bool, str]:
    answer = str(getattr(session, "answer", "") or "").strip()
    if not answer:
        return False, "empty_answer"

    status_text = str(status).lower()
    if "fail" in status_text:
        return False, f"agent_status={status}"

    min_evidence = int(getattr(args, "paperqa_min_evidence_count", 3))
    evidence_count = _positive_context_count(session)
    if evidence_count < min_evidence:
        return False, f"evidence_count<{min_evidence}"

    min_papers = int(getattr(args, "paperqa_min_relevant_papers", 1))
    relevant_papers = _relevant_doc_count(session)
    if relevant_papers < min_papers:
        return False, f"relevant_papers<{min_papers}"

    return True, "sufficient"


def _paper_search_queries(
    question: str,
    metadata_queries: list[str],
    selected: list[SelectedPaper],
    limit: int = 8,
    title_query_count: int = 0,
) -> list[str]:
    """Choose a few local full-text search queries for PaperQA paper_search.

    Title queries make sure selected PDFs enter PaperQA's EnvironmentState.
    ResearchPlanningAgent queries then provide broader topical recall.
    """

    queries: list[str] = []
    seen: set[str] = set()

    def add(query: str) -> None:
        if len(queries) >= limit:
            return
        clean = _refine_paper_search_query(query)
        key = clean.lower()
        if clean and key not in seen:
            seen.add(key)
            queries.append(clean)

    # 1. Use exact-title follow-up for selected papers. PaperQA's later
    # evidence step can only read documents that PaperSearch added to state.docs.
    for item in selected[: max(0, title_query_count)]:
        add(item.result.title)
        add(_short_title_query(item.result.title))

    # 2. Use refined planning queries. These encode the research angles chosen
    # by ResearchPlanningAgent and broaden recall after title-level coverage.
    for query in metadata_queries:
        add(query)

    # 3. Last fallback: the original question, trimmed for local PDF search.
    add(question)
    return queries or [question]


def _refine_paper_search_query(query: str) -> str:
    """Turn LLM/planner text into a concise local full-text search query."""

    clean = re.sub(r"\s+", " ", str(query)).strip().strip('"')
    clean = clean.replace("Vision-Language-Action", "vision language action")
    clean = clean.replace("vision-language-action", "vision language action")
    clean = re.sub(r"[“”‘’]", "", clean)
    clean = re.sub(r"\bplease\b|\bcompare\b|\bsummarize\b", "", clean, flags=re.I)
    clean = re.sub(r"\s+", " ", clean).strip(" ,.;:")
    words = clean.split()
    # Long natural-language questions hurt local retrieval; keep the most
    # specific front part unless this is an exact title.
    if len(words) > 14 and not re.search(r"\bVLA\b|-\b", clean):
        clean = " ".join(words[:14])
    return clean


def _short_title_query(title: str) -> str:
    """Make a title-like query that keeps names/acronyms and drops subtitles."""

    title = re.split(r":|\(|—|–", title, maxsplit=1)[0]
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-]*", title)
    keep: list[str] = []
    stop = {
        "a",
        "an",
        "the",
        "for",
        "with",
        "and",
        "of",
        "to",
        "in",
        "on",
        "model",
        "models",
    }
    for token in tokens:
        if token.lower() in stop and token.upper() != token:
            continue
        keep.append(token)
    return " ".join(keep[:8]) or title


def _selected_paper_line(item: SelectedPaper) -> str:
    result = item.result
    title = result.title or "Untitled"
    year = result.year or "unknown year"
    venue = result.venue or result.corpus or "unknown venue"
    authors = ", ".join((result.authors or [])[:4]) or "Unknown authors"
    return f"- {title} ({year}, {venue}; {authors})"


def _selected_paper_lines(selected: list[SelectedPaper], limit: int = 12) -> str:
    return "\n".join(_selected_paper_line(item) for item in selected[:limit])


def _paperqa_evidence_question(
    question: str,
    selected: list[SelectedPaper],
    metadata_queries: list[str],
    user_question: str | None = None,
) -> str:
    """Build an English retrieval task for PaperQA's GatherEvidence.

    The final answer can still be Chinese. This string is for retrieving and
    summarizing English PDF chunks, so it deliberately adds English evidence
    instructions and selected-paper titles.
    """

    query_hints = "; ".join(
        _refine_paper_search_query(query) for query in metadata_queries[:8] if query
    )
    paper_lines = _selected_paper_lines(selected)
    original_question = (user_question or question).strip()
    reader_context = question if question.strip() != original_question else "N/A"
    return (
        "Evidence retrieval task for English scientific PDFs.\n"
        "Use the selected papers to answer the ORIGINAL Chinese user question. "
        "The original question is the highest-priority relevance criterion. "
        "Retrieve concrete evidence, not general background. Cover only papers and "
        "chunks that are directly relevant to the original question or to a planned "
        "coverage angle that clearly supports it. For each relevant paper, extract: "
        "research problem, method or model design, data/benchmark/experiment setup, "
        "main findings, limitations, and how it relates to the original question. "
        "If a paper is merely adjacent, mark the evidence as weak or skip it.\n\n"
        f"Original user question in Chinese:\n{original_question}\n\n"
        f"Reader coverage checklist/context:\n{reader_context}\n\n"
        f"Research/query hints:\n{query_hints or 'N/A'}\n\n"
        f"Selected papers:\n{paper_lines or 'N/A'}"
    )


def _paper_specific_evidence_question(
    question: str,
    item: SelectedPaper,
    user_question: str | None = None,
) -> str:
    result = item.result
    title = result.title or "Untitled"
    abstract = (result.abstract or "").strip()
    if abstract:
        abstract = shorten(re.sub(r"\s+", " ", abstract), width=900, placeholder="...")
    original_question = (user_question or question).strip()
    reader_context = question if question.strip() != original_question else "N/A"
    return (
        f'Evidence retrieval task for the paper "{title}".\n'
        "Read this paper's PDF evidence that is relevant to the ORIGINAL Chinese "
        "user question. Extract concrete details about the problem, proposed method, "
        "datasets/benchmarks/experiments, key findings, limitations, and relation "
        "to the original question. Also state when this paper does not contain "
        "enough evidence for the requested angle. Prefer exact terminology from "
        "the paper.\n\n"
        f"Original user question in Chinese:\n{original_question}\n\n"
        f"Reader coverage checklist/context:\n{reader_context}\n\n"
        f"Paper metadata:\n{_selected_paper_line(item)}\n"
        f"Abstract hint:\n{abstract or 'N/A'}"
    )


def _context_record(context: Any) -> dict[str, Any]:
    text = getattr(context, "text", None)
    doc = getattr(text, "doc", None)
    return {
        "id": getattr(context, "id", None),
        "score": getattr(context, "score", None),
        "question": getattr(context, "question", None),
        "context": getattr(context, "context", None),
        "text_name": getattr(text, "name", None),
        "docname": getattr(doc, "docname", None),
        "dockey": str(getattr(doc, "dockey", "") or ""),
        "citation": (
            getattr(doc, "formatted_citation", None)
            or getattr(doc, "citation", None)
            or None
        ),
    }


def _evidence_contexts_from_state(state: Any, limit: int = 24) -> list[dict[str, Any]]:
    try:
        contexts = state.get_relevant_contexts()
    except AttributeError:
        contexts = [
            context
            for context in getattr(getattr(state, "session", None), "contexts", [])
            if getattr(context, "score", 0) > 0
        ]
    contexts = sorted(
        contexts,
        key=lambda context: getattr(context, "score", 0),
        reverse=True,
    )
    return [_context_record(context) for context in contexts[:limit]]


def _evidence_only_answer(
    evidence_contexts: list[dict[str, Any]],
    user_question: str | None,
    reader_question: str,
) -> str:
    lines = [
        "PaperQA evidence contexts extracted from selected PDFs.",
        "These are not a final answer. SynthesisAgent should use them together "
        "with web evidence and metadata to answer the original user question.",
        "",
        f"Original user question: {user_question or reader_question}",
    ]
    if user_question and reader_question.strip() != user_question.strip():
        lines.extend(["", f"Reader coverage checklist/context: {reader_question}"])
    lines.append("")
    if not evidence_contexts:
        lines.append("No positive PaperQA evidence contexts were extracted.")
        return "\n".join(lines)

    for idx, record in enumerate(evidence_contexts, start=1):
        source = record.get("citation") or record.get("docname") or "unknown source"
        lines.extend(
            [
                f"[PaperQA/PDF evidence {idx}] score={record.get('score')} source={source}",
                f"question={record.get('question') or 'N/A'}",
                str(record.get("context") or "").strip(),
                "",
            ]
        )
    return "\n".join(lines).strip()


async def run_paperqa_agent_reader(
    question: str,
    selected: list[SelectedPaper],
    args: Any,
) -> tuple[str, dict[str, Any]]:
    """Run PaperQA's native multi-step agent over the selected PDF directory."""

    if str(PAPERQA_SRC) not in sys.path:
        sys.path.insert(0, str(PAPERQA_SRC))

    from paperqa import Docs, agent_query

    settings = configure_paperqa_settings(args)
    paper_records = prepare_agent_paper_dir(selected, args.agent_paper_dir)
    if not paper_records:
        raise RuntimeError("No PDFs were available for PaperQA agent after download/cache.")
    configure_paperqa_index(settings, args)

    print_agent_progress(
        f"LangGraph native PaperQA agent prepared {len(paper_records)} PDFs in {args.agent_paper_dir}",
        args,
    )

    tool_trace: list[dict[str, Any]] = []
    action_trace: list[dict[str, Any]] = []

    async def on_gather_started(state: Any) -> None:
        print_agent_progress(f"native gather_evidence started | {state_status(state)}", args)

    async def on_gather_completed(state: Any) -> None:
        print_agent_progress(f"native gather_evidence completed | {state_status(state)}", args)

    async def on_answer_started(state: Any) -> None:
        print_agent_progress(f"native gen_answer started | {state_status(state)}", args)

    async def on_answer_completed(state: Any) -> None:
        print_agent_progress(f"native gen_answer completed | {state_status(state)}", args)

    settings.agent.callbacks.setdefault("gather_evidence_initialized", []).append(
        on_gather_started
    )
    settings.agent.callbacks.setdefault("gather_evidence_completed", []).append(
        on_gather_completed
    )
    settings.agent.callbacks.setdefault("gen_answer_initialized", []).append(on_answer_started)
    settings.agent.callbacks.setdefault("gen_answer_completed", []).append(on_answer_completed)

    async def on_agent_action_callback(action: Any, state: Any) -> None:
        summary = summarize_tool_request(action)
        status = state_status(state)
        print_agent_progress(f"native calling {summary} | {status}", args)
        action_record = {
            "action_type": type(action).__name__,
            "action": _jsonable(action),
            "state_type": type(state).__name__,
            "summary": summary,
            "status": status,
        }
        action_trace.append(action_record)
        tool_trace.append(
            {
                "tool": "agent_action",
                "summary": summary,
                "status": status,
            }
        )

    async def on_env_step_callback(
        obs: list[Any],
        reward: float,
        done: bool,
        truncated: bool,
    ) -> None:
        previews: list[str] = []
        for item in obs[:2]:
            content = getattr(item, "content", "")
            if content:
                previews.append(
                    shorten(str(content).replace("\n", " "), width=220, placeholder="...")
                )
        if previews:
            print_agent_progress(f"native tool response: {' | '.join(previews)}", args)
        print_agent_progress(
            f"native step done={done} truncated={truncated} reward={reward:g}",
            args,
        )
        tool_trace.append(
            {
                "tool": "env_step",
                "done": done,
                "truncated": truncated,
                "reward": reward,
                "status": "done" if done else "running",
                "output_preview": " | ".join(previews),
            }
        )

    agent_type = getattr(args, "paperqa_agent_type", None) or settings.agent.agent_type
    response = await agent_query(
        query=question,
        settings=settings,
        docs=Docs(),
        agent_type=agent_type,
        on_agent_action_callback=on_agent_action_callback,
        on_env_step_callback=on_env_step_callback,
    )
    session = response.session
    evidence_count = _positive_context_count(session)
    relevant_papers = _relevant_doc_count(session)
    sufficient, sufficiency_reason = _paperqa_agent_sufficient(session, response.status, args)
    trace = {
        "mode": "paperqa-agent",
        "agent_type": str(agent_type),
        "agent_paper_dir": str(args.agent_paper_dir),
        "agent_papers": paper_records,
        "agent_actions": action_trace,
        "tool_trace": tool_trace,
        "status": str(response.status),
        "has_successful_answer": getattr(session, "has_successful_answer", None),
        "evidence_count": evidence_count,
        "relevant_paper_count": relevant_papers,
        "sufficient": sufficient,
        "sufficiency_reason": sufficiency_reason,
    }
    return str(getattr(session, "answer", session)), trace


async def run_paperqa_tools_reader(
    question: str,
    selected: list[SelectedPaper],
    metadata_queries: list[str],
    args: Any,
    user_question: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Run PaperQA's internal tools directly as a LangGraph reader node.

    PaperQA's own agent loop uses Aviary's ToolSelector. For the LangGraph
    version we keep the same useful tools, but call them explicitly:

        PaperSearch -> GatherEvidence -> GenerateAnswer

    The important bit is that all three tools share the same EnvironmentState.
    PaperSearch adds matching papers to state.docs, GatherEvidence adds contexts
    to state.session, and GenerateAnswer reads that session to produce the final
    cited answer.
    """

    # 这个函数是 LangGraph ReaderAgent 的主体。
    # 我们没有把整个 PaperQA agent 黑盒包起来，而是直接复用 PaperQA 的三个工具：
    # 1. PaperSearch：在本轮 PDF 索引里找相关论文，并把文本加入 state.docs；
    # 2. GatherEvidence：从 state.docs 中抽取/总结证据片段；
    # 3. GenerateAnswer：基于证据生成带引用的最终回答。
    if str(PAPERQA_SRC) not in sys.path:
        sys.path.insert(0, str(PAPERQA_SRC))

    from paperqa import Docs
    from paperqa.agents.search import get_directory_index
    from paperqa.agents.tools import EnvironmentState, GatherEvidence, GenerateAnswer, PaperSearch
    from paperqa.types import PQASession

    settings = configure_paperqa_settings(args)

    # Prepare the exact PDF directory searched by PaperQA. The manifest carries
    # our metadata into PaperQA so it does not need Crossref/Semantic Scholar.
    paper_records = prepare_agent_paper_dir(selected, args.agent_paper_dir)
    if not paper_records:
        raise RuntimeError("No PDFs were available for PaperQA tools after download/cache.")
    configure_paperqa_index(settings, args)

    print_agent_progress(
        f"LangGraph reader prepared {len(paper_records)} PDFs in {args.agent_paper_dir}",
        args,
    )

    # 先构建本轮 PDF 目录的本地索引。后续 PaperSearch 会打开这个索引做全文检索。
    await get_directory_index(settings=settings, build=True)

    embedding_model = settings.get_embedding_model()
    summary_llm_model = settings.get_summary_llm()
    llm_model = settings.get_llm()
    state = EnvironmentState(
        docs=Docs(),
        session=PQASession(question=question, config_md5=settings.md5),
    )

    trace: list[dict[str, Any]] = []

    # Tool 1：在本地 PDF 索引中检索相关论文。
    # 注意这里用的是 metadata_search 阶段生成的 query，而不是重新让 LLM 随机想 query。
    # 检索到的论文文本会进入共享的 EnvironmentState，供后续证据抽取使用。
    search_tool = PaperSearch(settings=settings, embedding_model=embedding_model)
    search_query_limit = int(getattr(args, "paperqa_search_query_limit", 8))
    title_query_count = int(
        getattr(args, "paperqa_title_query_count", min(len(selected), 6))
    )
    for query in _paper_search_queries(
        question,
        metadata_queries,
        selected,
        limit=search_query_limit,
        title_query_count=title_query_count,
    ):
        print_agent_progress(f"LangGraph reader paper_search({query!r})", args)
        output = await search_tool.paper_search(
            query=query,
            min_year=args.year_from,
            max_year=args.year_to,
            state=state,
        )
        trace.append(
            {
                "tool": "paper_search",
                "query": query,
                "status": state_status(state),
                "output_preview": shorten(output.replace("\n", " "), width=300, placeholder="..."),
            }
        )

    # Tool 2：从已检索到的论文文本中抽证据。
    # 这一步会调用 summary_llm，把长文本片段压缩成和问题相关的 evidence。
    gather_tool = GatherEvidence(
        settings=settings,
        summary_llm_model=summary_llm_model,
        embedding_model=embedding_model,
    )
    answer_mode = str(getattr(args, "paperqa_answer_mode", "answer")).lower()
    evidence_question = _paperqa_evidence_question(
        question,
        selected,
        metadata_queries,
        user_question=user_question,
    )
    print_agent_progress(
        f"LangGraph reader gather_evidence(overall) | {state_status(state)}", args
    )
    evidence_output = await gather_tool.gather_evidence(
        question=evidence_question,
        state=state,
    )
    trace.append(
        {
            "tool": "gather_evidence",
            "scope": "overall",
            "question_preview": shorten(
                evidence_question.replace("\n", " "), width=400, placeholder="..."
            ),
            "status": state_status(state),
            "output_preview": shorten(
                evidence_output.replace("\n", " "), width=500, placeholder="..."
            ),
        }
    )

    per_paper_count = max(0, int(getattr(args, "paperqa_per_paper_evidence_count", 4)))
    for item in selected[:per_paper_count]:
        paper_question = _paper_specific_evidence_question(
            question,
            item,
            user_question=user_question,
        )
        title = item.result.title or "Untitled"
        print_agent_progress(
            f"LangGraph reader gather_evidence({shorten(title, width=70, placeholder='...')!r})"
            f" | {state_status(state)}",
            args,
        )
        paper_evidence_output = await gather_tool.gather_evidence(
            question=paper_question,
            state=state,
        )
        trace.append(
            {
                "tool": "gather_evidence",
                "scope": "per_paper",
                "paper_title": title,
                "question_preview": shorten(
                    paper_question.replace("\n", " "), width=400, placeholder="..."
                ),
                "status": state_status(state),
                "output_preview": shorten(
                    paper_evidence_output.replace("\n", " "),
                    width=500,
                    placeholder="...",
                ),
            }
        )

    evidence_contexts = _evidence_contexts_from_state(state)
    if answer_mode == "evidence-only":
        trace.append(
            {
                "tool": "gen_answer",
                "status": "skipped_evidence_only",
                "output_preview": (
                    "Skipped PaperQA GenerateAnswer; returning evidence contexts "
                    "for SynthesisAgent."
                ),
            }
        )
        return _evidence_only_answer(evidence_contexts, user_question, question), {
            "mode": "langgraph-paperqa-tools",
            "answer_mode": answer_mode,
            "agent_paper_dir": str(args.agent_paper_dir),
            "agent_papers": paper_records,
            "paper_search_query_limit": search_query_limit,
            "paperqa_title_query_count": title_query_count,
            "paperqa_per_paper_evidence_count": per_paper_count,
            "evidence_count": len(evidence_contexts),
            "evidence_contexts": evidence_contexts,
            "tool_trace": trace,
            "status": state_status(state),
        }

    # Tool 3：根据 evidence 生成带 citation 的回答。
    # 最终答案会写入 state.session.answer，并返回给 LangGraph 的 final_answer。
    answer_tool = GenerateAnswer(
        settings=settings,
        llm_model=llm_model,
        summary_llm_model=summary_llm_model,
        embedding_model=embedding_model,
    )
    print_agent_progress(f"LangGraph reader gen_answer | {state_status(state)}", args)
    answer_output = await answer_tool.gen_answer(state=state)
    trace.append(
        {
            "tool": "gen_answer",
            "status": state_status(state),
            "output_preview": shorten(
                answer_output.replace("\n", " "), width=500, placeholder="..."
            ),
        }
    )

    return state.session.answer, {
        "mode": "langgraph-paperqa-tools",
        "answer_mode": answer_mode,
        "agent_paper_dir": str(args.agent_paper_dir),
        "agent_papers": paper_records,
        "paper_search_query_limit": search_query_limit,
        "paperqa_title_query_count": title_query_count,
        "paperqa_per_paper_evidence_count": per_paper_count,
        "evidence_count": len(evidence_contexts),
        "evidence_contexts": evidence_contexts,
        "tool_trace": trace,
        "status": state_status(state),
    }


async def run_paperqa_reader_adapter(
    question: str,
    user_question: str | None,
    selected: list[SelectedPaper],
    metadata_queries: list[str],
    args: Any,
) -> tuple[str, dict[str, Any]]:
    """PaperQA reader adapter used by ExpertResearchAgent.

    Default policy:
      1. Run PaperQA's native multi-step agent over the selected PDFs.
      2. If the native agent fails or returns too little evidence, fall back to
         our explicit PaperSearch -> GatherEvidence -> GenerateAnswer chain.

    The outer LangGraph agent still sees this as one reader tool.
    """

    mode = str(getattr(args, "paperqa_reader_mode", "paperqa-agent")).lower()
    answer_mode = str(getattr(args, "paperqa_answer_mode", "answer")).lower()
    if mode == "explicit-tools" or answer_mode == "evidence-only":
        answer, trace = await run_paperqa_tools_reader(
            question=question,
            selected=selected,
            metadata_queries=metadata_queries,
            args=args,
            user_question=user_question,
        )
        trace["adapter_mode"] = "explicit-tools"
        trace["fallback_used"] = False
        if answer_mode == "evidence-only" and mode != "explicit-tools":
            trace["forced_explicit_tools_reason"] = "paperqa_answer_mode=evidence-only"
        return answer, trace

    agent_trace: dict[str, Any] | None = None
    try:
        answer, agent_trace = await run_paperqa_agent_reader(
            question=question,
            selected=selected,
            args=args,
        )
        if agent_trace.get("sufficient", False) or mode == "agent-only":
            agent_trace["adapter_mode"] = mode
            agent_trace["fallback_used"] = False
            return answer, agent_trace
        fallback_reason = agent_trace.get("sufficiency_reason", "insufficient_evidence")
    except Exception as exc:
        if mode == "agent-only":
            raise
        fallback_reason = f"agent_error:{type(exc).__name__}"
        agent_trace = {
            "mode": "paperqa-agent",
            "status": "error",
            "error": shorten(str(exc), width=700, placeholder="..."),
            "tool_trace": [],
            "sufficient": False,
            "sufficiency_reason": fallback_reason,
        }

    print_agent_progress(
        f"native PaperQA agent fallback to explicit tools ({fallback_reason})",
        args,
    )
    fallback_answer, fallback_trace = await run_paperqa_tools_reader(
        question=question,
        selected=selected,
        metadata_queries=metadata_queries,
        args=args,
        user_question=user_question,
    )
    fallback_trace["adapter_mode"] = mode
    fallback_trace["fallback_used"] = True
    fallback_trace["fallback_reason"] = fallback_reason
    fallback_trace["native_agent_trace"] = agent_trace
    fallback_trace["tool_trace"] = [
        {
            "tool": "paperqa_agent_reader",
            "status": agent_trace.get("status") if agent_trace else "error",
            "sufficient": agent_trace.get("sufficient") if agent_trace else False,
            "reason": fallback_reason,
        },
        *fallback_trace.get("tool_trace", []),
    ]
    return fallback_answer, fallback_trace
