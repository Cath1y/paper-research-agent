from __future__ import annotations

import re
from textwrap import shorten
from typing import Any

from scripts.ask_literature import (
    normalize_openai_compatible_model,
    openai_compatible_api_key,
)


def _synthesis_model(args: Any) -> str | None:
    model = (
        getattr(args, "synthesis_llm", None)
        or getattr(args, "llm", None)
        or getattr(args, "agent_llm", None)
        or getattr(args, "router_llm", None)
    )
    if not model:
        return None
    if args.openai_base_url and not args.disable_openai_compatible_config:
        return normalize_openai_compatible_model(model, args, args.openai_base_url)
    return model


def _synthesis_max_tokens(args: Any) -> int:
    value = getattr(args, "synthesis_max_tokens", 4096)
    try:
        return max(512, int(value))
    except (TypeError, ValueError):
        return 4096


def _episode_summary_max_tokens(args: Any) -> int:
    value = getattr(args, "episode_summary_max_tokens", 700)
    try:
        return max(256, int(value))
    except (TypeError, ValueError):
        return 700


def _paper_brief(item: dict[str, Any], idx: int) -> str:
    result = item.get("result", item)
    authors = result.get("authors") or []
    authors_text = ", ".join(authors[:3])
    if len(authors) > 3:
        authors_text += ", et al."
    fields = [
        f"{idx}. {result.get('title', 'Untitled')}",
        f"   year={result.get('year')} venue={result.get('venue') or result.get('corpus')}",
        f"   authors={authors_text or 'Unknown'}",
        f"   id={result.get('paper_id')} citations={result.get('citation_count')} influential={result.get('influential_citation_count')}",
        f"   code={result.get('code_url') or result.get('project_url') or 'N/A'}",
        f"   paper_url={result.get('paper_url') or 'N/A'}",
        f"   pdf_url={result.get('pdf_url') or 'N/A'}",
        f"   sources={', '.join(result.get('sources') or []) or 'N/A'}",
        f"   signals={', '.join(result.get('quality_signals') or []) or 'N/A'}",
        f"   selection_score={item.get('selection_score', 'N/A')} cache={item.get('cache_status', 'N/A')}",
        f"   abstract={shorten((result.get('abstract') or '').replace(chr(10), ' '), width=700, placeholder='...')}",
    ]
    return "\n".join(fields)


def _question_brief(item: dict[str, Any], idx: int) -> str:
    queries = item.get("queries") or []
    query_text = "; ".join(str(query) for query in queries[:3])
    targets = item.get("coverage_targets") or []
    target_text = "; ".join(str(target) for target in targets[:4])
    return (
        f"{idx}. perspective={item.get('perspective', '')}\n"
        f"   question={item.get('question') or item.get('research_question', '')}\n"
        f"   coverage_targets={target_text}\n"
        f"   queries={query_text}"
    )


def _fallback_answer(state: dict[str, Any]) -> str:
    route = state.get("route", {})
    if route and not route.get("need_paper_search", True):
        return route.get("direct_answer") or (
            "Planner/Supervisor 判断这个问题暂时不需要论文检索，但当前没有可用的 "
            "SynthesisAgent LLM，因此没有生成更完整的短答。"
        )

    selected = state.get("selected", [])
    web_findings = state.get("web_findings", "")
    paperqa_answer = state.get("paperqa_answer", "")
    paperqa_trace = state.get("paperqa_trace", {})

    lines = ["## 调研结果草稿"]
    if selected:
        lines.append("\n### 代表性论文候选")
        for idx, item in enumerate(selected[:10], start=1):
            result = item.get("result", item)
            title = result.get("title", "Untitled")
            year = result.get("year", "?")
            venue = result.get("venue") or result.get("corpus") or "?"
            score = item.get("selection_score", "N/A")
            lines.append(f"- {title} ({year}, {venue})，selection_score={score}")
    if paperqa_answer:
        lines.append("\n### PaperQA 证据回答")
        lines.append(paperqa_answer)
    elif paperqa_trace.get("status") == "error":
        lines.append("\n### PaperQA 阅读状态")
        lines.append(f"PaperQA PDF 阅读失败：{paperqa_trace.get('error')}")
    else:
        lines.append("\n### PaperQA 阅读状态")
        lines.append("当前运行没有生成 PaperQA 证据回答，可能处于 dry-run/download-only。")
    if web_findings:
        lines.append("\n### Web 补充线索")
        lines.append(web_findings)
    lines.append(
        "\n### 覆盖提示\n"
        "当前 fallback 没有执行 LLM 级别的最终综合。建议配置 SynthesisAgent LLM 后，"
        "生成论文表、趋势-论文映射、覆盖不足和阅读顺序。"
    )
    return "\n".join(lines)


def _fallback_episode_summary(state: dict[str, Any], answer: str) -> str:
    question = shorten(str(state.get("question") or ""), width=180, placeholder="...")
    selected = state.get("selected", []) or []
    selected_titles = []
    for item in selected[:5]:
        result = item.get("result", item)
        title = result.get("title") if isinstance(result, dict) else None
        if title:
            selected_titles.append(str(title))
    paper_bits = "；".join(selected_titles)
    answer_hint = shorten(str(answer or ""), width=520, placeholder="...")
    parts = [f"本轮问题：{question}。"]
    if paper_bits:
        parts.append(f"涉及论文：{shorten(paper_bits, width=260, placeholder='...')}。")
    if answer_hint:
        parts.append(f"回答要点：{answer_hint}")
    return " ".join(parts)


def _build_synthesis_prompt(state: dict[str, Any]) -> str:
    question = state.get("question", "")
    memory_context = state.get("memory_context", "")
    route = state.get("route", {})
    selected = state.get("selected", [])
    candidates = state.get("candidates", [])
    research_plan = state.get("research_plan", [])
    research_questions = state.get("research_questions", [])
    perspectives = state.get("perspectives", [])
    queries = state.get("queries", [])
    paperqa_answer = state.get("paperqa_answer", "")
    paperqa_trace = state.get("paperqa_trace", {})
    web_findings = state.get("web_findings", "")
    planning_web_evidence = state.get("planning_web_evidence", [])
    expert_trace = state.get("expert_trace", [])

    selected_text = "\n".join(
        _paper_brief(item, idx)
        for idx, item in enumerate(selected[:12], start=1)
    )
    candidate_titles = "\n".join(
        f"- {item.get('title', 'Untitled')} ({item.get('year')}, {item.get('venue') or item.get('corpus')})"
        for item in candidates[:12]
    )
    plan_text = "\n".join(
        _question_brief(item, idx)
        for idx, item in enumerate(research_plan[:8], start=1)
    )
    question_text = "\n".join(
        _question_brief(item, idx)
        for idx, item in enumerate(research_questions[:8], start=1)
    )
    perspective_text = "\n".join(
        f"- {item.get('name')}: {item.get('description')}"
        for item in perspectives[:8]
    )
    expert_trace_text = "\n".join(
        f"- {item}"
        for item in expert_trace
    )
    planning_web_text = "\n".join(
        (
            f"- {item.get('title', 'Untitled')}\n"
            f"  url={item.get('url', '')}\n"
            f"  snippet={shorten(str(item.get('snippet') or '').replace(chr(10), ' '), width=350, placeholder='...')}"
        )
        for item in planning_web_evidence[:8]
    )

    return (
        f"用户原始问题：{question}\n\n"
        "Compact thread memory (only for resolving follow-up references; not evidence)：\n"
        f"{memory_context or 'N/A'}\n\n"
        f"Planner route：{route}\n\n"
        "ResearchPlanningAgent plan：\n"
        f"{plan_text or 'N/A'}\n\n"
        "Compatibility perspective view：\n"
        f"{perspective_text or 'N/A'}\n\n"
        "Compatibility research questions view：\n"
        f"{question_text or 'N/A'}\n\n"
        "ResearchPlanningAgent web scouting context (for query planning, not final strong evidence)：\n"
        f"{planning_web_text or 'N/A'}\n\n"
        "Search queries：\n"
        + "\n".join(f"- {query}" for query in queries[:12])
        + "\n\nSelected papers metadata：\n"
        f"{selected_text or 'N/A'}\n\n"
        "Other high-ranked candidate titles：\n"
        f"{candidate_titles or 'N/A'}\n\n"
        "ExpertResearchAgent tool trace：\n"
        f"{expert_trace_text or 'N/A'}\n\n"
        "Web findings：\n"
        f"{shorten(web_findings.replace(chr(10), ' '), width=3000, placeholder='...') or 'N/A'}\n\n"
        "PaperQA trace：\n"
        f"{paperqa_trace}\n\n"
        "PaperQA evidence-based draft answer：\n"
        f"{shorten(paperqa_answer, width=9000, placeholder='...') or 'N/A'}"
    )


def _tagged_section(content: str, tag: str) -> str:
    match = re.search(
        rf"<{tag}>\s*(.*?)\s*</{tag}>",
        content,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return match.group(1).strip() if match else ""


def _strip_summary_tag(content: str) -> str:
    return re.sub(
        r"\s*<episode_summary>\s*.*?\s*</episode_summary>\s*",
        "",
        content,
        flags=re.IGNORECASE | re.DOTALL,
    ).strip()


def _parse_synthesis_output(
    content: str,
) -> tuple[str, str, str]:
    final_answer = _tagged_section(content, "final_answer")
    episode_summary = _tagged_section(content, "episode_summary")
    if final_answer:
        return final_answer, episode_summary, "tagged"
    fallback_answer = _strip_summary_tag(content)
    return fallback_answer, episode_summary, "untagged"


async def run_synthesis_agent(
    state: dict[str, Any],
    args: Any,
) -> tuple[str, dict[str, Any]]:
    """Generate the final user-facing answer from the shared LangGraph state."""

    fallback = _fallback_answer(state)
    model = _synthesis_model(args)
    if not model:
        episode_summary = _fallback_episode_summary(state, fallback)
        return fallback, {
            "mode": "fallback_no_llm",
            "model": None,
            "error": None,
            "episode_summary": episode_summary,
            "episode_summary_trace": {"mode": "fallback_no_llm"},
        }

    prompt = _build_synthesis_prompt(state)
    try:
        import litellm

        max_tokens = _synthesis_max_tokens(args)
        summary_budget = _episode_summary_max_tokens(args)
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are SynthesisAgent for a literature-learning assistant. "
                        "Your job is to answer the user's original question using the shared state "
                        "from Planner, ResearchPlanningAgent, ExpertResearchAgent, "
                        "and PaperQA. Answer in Chinese. Do not invent citations. Preserve citations "
                        "that already appear in the PaperQA draft.\n\n"
                        "Do not force a fixed report template. Decide the structure from the "
                        "user's wording and the available evidence. A table, outline, timeline, "
                        "or short direct answer is acceptable only when it helps answer the "
                        "question.\n\n"
                        "Evidence discipline:\n"
                        "- Compact thread memory is conversation context only. Never cite it as "
                        "evidence and never use it to override fresh PaperQA/web evidence.\n"
                        "- Exception: if Planner route answer_style is memory_paper_list, the "
                        "user is asking for the previous turn's paper list/links. In that case, "
                        "answer directly from Compact thread memory, do not demand new evidence, "
                        "and do not start a literature review format.\n"
                        "- Treat PaperQA evidence and web evidence as equal evidence sources. "
                        "Use whichever directly supports the user's question, and ignore evidence "
                        "that is merely adjacent.\n"
                        "- Always label the source type when using evidence: for example "
                        "[PaperQA/PDF], [Web], [Metadata], or [Planning web scout]. Preserve "
                        "PaperQA citations that already appear in the draft, and include web URLs "
                        "or page names when available.\n"
                        "- Metadata-only selected papers may be discussed as candidates, but mark "
                        "them as [Metadata] if PaperQA or web evidence has not confirmed them.\n"
                        "- If PaperQA snippets, web results, or selected papers do not actually "
                        "support the original question, say that the retrieval is weak or mismatched "
                        "instead of stretching them into a conclusion.\n"
                        "- Make uncertainty explicit and avoid filling gaps with prior knowledge.\n\n"
                        "Keep the answer concise but complete. Focus on the user's question, not on "
                        "debug logs.\n\n"
                        "Output contract:\n"
                        "Return exactly two XML-style sections. The first section is the user-visible "
                        "answer; the second section is a compact memory index summary and will not be "
                        "shown to the user.\n\n"
                        "<final_answer>\n"
                        "Write the user-facing answer here.\n"
                        "</final_answer>\n"
                        "<episode_summary>\n"
                        f"Write a Chinese 120-220 character memory summary here, under about "
                        f"{summary_budget} tokens. Include the user "
                        "question, what was actually done, main conclusion, key selected/read papers "
                        "or sources, and whether evidence was sufficient. Do not add new facts.\n"
                        "</episode_summary>"
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
            "max_tokens": max_tokens,
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
        choice = response.choices[0]
        content = (choice.message.content or "").strip()
        final_answer, episode_summary, parse_mode = _parse_synthesis_output(content)
        final_answer = final_answer or content or fallback
        episode_summary = episode_summary or _fallback_episode_summary(state, final_answer)
        return final_answer, {
            "mode": "llm",
            "model": model,
            "error": None,
            "finish_reason": getattr(choice, "finish_reason", None),
            "max_tokens": max_tokens,
            "output_parse_mode": parse_mode,
            "episode_summary": episode_summary,
            "episode_summary_trace": {
                "mode": "same_llm_call",
                "parse_mode": parse_mode,
            },
        }
    except Exception as exc:
        episode_summary = _fallback_episode_summary(state, fallback)
        return fallback, {
            "mode": "fallback_after_llm_error",
            "model": model,
            "error": shorten(str(exc), width=500, placeholder="..."),
            "episode_summary": episode_summary,
            "episode_summary_trace": {"mode": "fallback_after_llm_error"},
        }
