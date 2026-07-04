from __future__ import annotations

import json
import re
from argparse import Namespace
from dataclasses import asdict, dataclass
from textwrap import shorten
from typing import Any

from scripts.ask_literature import (
    normalize_openai_compatible_model,
    openai_compatible_api_key,
)


TASK_TYPES = {
    "quick_answer",
    "literature_review",
    "learning_plan",
    "idea_generation",
    "paper_deep_read",
}

TASK_PRIORITY = [
    "paper_deep_read",
    "idea_generation",
    "learning_plan",
    "literature_review",
    "quick_answer",
]

ALLOWED_AGENTS = {
    "ResearchPlanningAgent",
    "ExpertResearchAgent",
    "SynthesisAgent",
}

AGENT_ALIASES = {
    "PersonaGeneratorAgent": "ResearchPlanningAgent",
    "PerspectiveGeneratorAgent": "ResearchPlanningAgent",
    "PerspectiveAgent": "ResearchPlanningAgent",
    "QueryPlannerAgent": "ResearchPlanningAgent",
    "QuestionAskerAgent": "ResearchPlanningAgent",
    "QuestionPlannerAgent": "ResearchPlanningAgent",
    "PaperSearchAgent": "ExpertResearchAgent",
    "LiteratureSearchAgent": "ExpertResearchAgent",
    "LiteratureReviewAgent": "ExpertResearchAgent",
    "WebSearchAgent": "ExpertResearchAgent",
    "MetadataSearchAgent": "ExpertResearchAgent",
    "PaperTriageAgent": "ExpertResearchAgent",
    "PaperQAReaderAgent": "ExpertResearchAgent",
    "LearningPlanAgent": "SynthesisAgent",
    "CurriculumRefinementAgent": "SynthesisAgent",
    "CurriculumPlannerAgent": "SynthesisAgent",
    "ResearchIdeaAgent": "SynthesisAgent",
    "IdeaGenerationAgent": "SynthesisAgent",
    "AnswerAgent": "SynthesisAgent",
    "FinalAnswerAgent": "SynthesisAgent",
}

AGENT_DESCRIPTIONS = {
    "ResearchPlanningAgent": "generates STORM-style research perspectives and turns each one into research subquestions plus high-precision search queries",
    "ExpertResearchAgent": "enhanced STORM-style expert that uses web search, metadata search, PDF caching, and PaperQA tools to collect evidence",
    "SynthesisAgent": "integrates outputs from prior steps into the final response",
}


class RouterOutputParseError(ValueError):
    """Raised when the planner LLM returns text that cannot be parsed as JSON."""


@dataclass(slots=True)
class PlanStep:
    """One planned worker-agent step in the supervisor plan."""

    id: str
    agent: str
    objective: str
    depends_on: list[str]
    input_keys: list[str]
    output_keys: list[str]
    granularity: str = "coarse"
    substeps: list[str] | None = None
    status: str = "pending"


@dataclass(slots=True)
class RouteDecision:
    """Structured output produced by the LangGraph Planner/Supervisor agent."""

    # task_type is the primary label for compatibility; task_types is multi-label.
    task_type: str
    task_types: list[str]
    workflow_goal: str
    steps: list[PlanStep]
    next_step: str | None
    need_web_search: bool
    need_paper_search: bool
    need_pdf_reading: bool
    need_learning_plan: bool
    need_idea_generation: bool
    answer_style: str
    confidence: float
    reasoning: str
    direct_answer: str = ""
    source: str = "heuristic"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _step(
    step_id: str,
    agent: str,
    objective: str,
    depends_on: list[str],
    input_keys: list[str],
    output_keys: list[str],
) -> PlanStep:
    return PlanStep(
        id=step_id,
        agent=agent,
        objective=objective,
        depends_on=depends_on,
        input_keys=input_keys,
        output_keys=output_keys,
    )


def build_heuristic_steps(
    question: str,
    task_types: list[str],
    need_web_search: bool,
    need_paper_search: bool,
    need_pdf_reading: bool,
) -> list[PlanStep]:
    """Create a dependency-aware default plan from task labels.

    Keep the plan aligned with the currently executable outer LangGraph nodes.
    Tool-level work such as web search, metadata search, PDF caching, PaperQA
    reading, topic extraction, and paper triage happens inside ExpertResearchAgent.
    """

    steps: list[PlanStep] = []
    needs_research_planning = need_web_search or need_paper_search
    research_depends_on: list[str] = []
    if needs_research_planning:
        steps.append(
            _step(
                "research_planning",
                "ResearchPlanningAgent",
                "按照 STORM 思路生成多个研究视角，并为每个视角产出 research question 与可用于论文库/Web/PDF 阅读的 queries。",
                [],
                ["question"],
                ["research_plan", "perspectives", "research_questions", "queries"],
            )
        )
        research_depends_on = ["research_planning"]

    if need_web_search or need_paper_search:
        steps.append(
            _step(
                "expert_research",
                "ExpertResearchAgent",
                "作为加强版 STORM expert，基于 perspective questions 调用 web search、metadata search、PDF cache 和 PaperQA tools 收集证据。",
                research_depends_on,
                ["question", "research_questions", "queries", "perspectives"],
                [
                    "web_evidence",
                    "web_findings",
                    "candidates",
                    "selected",
                    "paperqa_answer",
                    "paperqa_trace",
                ],
            )
        )

    if "learning_plan" in task_types:
        # LearningPlannerAgent is planned for a later iteration. For now the
        # final synthesis may mention that a learning roadmap is requested, but
        # the executable research path remains the same.
        pass

    if "idea_generation" in task_types:
        # IdeaThinkAgent is also a later worker. Keep it out of the executable
        # plan until we wire the node.
        pass

    if not steps:
        steps.append(
            _step(
                "answer_directly",
                "SynthesisAgent",
                "由 Planner/Synthesis 直接回答用户的概念性问题，并在必要时指出可继续检索的方向。",
                [],
                ["question"],
                ["final_answer"],
            )
        )

    final_depends_on = [steps[-1].id] if steps else []
    if steps[-1].id != "answer_directly":
        steps.append(
            _step(
                "synthesize_final_answer",
                "SynthesisAgent",
                "整合所有已完成步骤的证据，生成面向学习助手场景的最终回答。",
                final_depends_on,
                ["question", "paperqa_answer", "web_evidence", "learning_plan", "research_ideas"],
                ["final_answer"],
            )
        )

    return steps


def heuristic_direct_answer(question: str) -> str:
    """Tiny fallback for common demo concepts when no router LLM is configured."""

    text = question.lower()
    if "vla" in text or "视觉语言动作" in question:
        return (
            "在机器人领域，VLA 通常指 Vision-Language-Action model，"
            "也就是把视觉观测、语言指令和机器人动作统一建模的策略模型。"
            "它的目标是让机器人根据图像/视频和自然语言指令直接输出可执行动作，"
            "常见应用包括机械臂操作、抓取、移动操作和通用机器人任务。"
        )
    if "rag" in text:
        return (
            "RAG 是 Retrieval-Augmented Generation，即检索增强生成。"
            "系统先从外部知识库或网页中找相关材料，再把检索结果放进提示词，"
            "让 LLM 基于证据回答，从而降低幻觉并补充模型参数中没有的新知识。"
        )
    if "langgraph" in text:
        return (
            "LangGraph 是 LangChain 生态里的有状态工作流框架，适合把多个 agent、"
            "工具调用和条件分支组织成图。相比简单链式调用，它更适合多轮规划、"
            "循环、状态累积和 supervisor/router 控制流。"
        )
    if "多智能体" in question or "multi-agent" in text:
        return (
            "多智能体系统通常把一个复杂任务拆给多个角色化 agent，例如 planner、"
            "searcher、reader、critic 和 synthesizer。关键不是 agent 数量越多越好，"
            "而是每个 agent 是否有清晰输入输出、共享状态和可验证的工具能力。"
        )
    return ""


def heuristic_route(question: str) -> RouteDecision:
    """Keyword fallback for routing when the LLM router is unavailable."""

    text = question.lower()
    learning_terms = (
        "学习路线",
        "怎么学",
        "如何学",
        "入门",
        "课程",
        "学习计划",
        "roadmap",
        "learn",
        "tutorial",
    )
    idea_terms = (
        "idea",
        "创新",
        "选题",
        "课题",
        "研究方向",
        "项目",
        "proposal",
        "future work",
    )
    deep_read_terms = ("精读", "解读这篇", "这篇论文", "paper reading", "deep read")
    literature_terms = (
        "论文",
        "文献",
        "综述",
        "调研",
        "热点",
        "趋势",
        "代表性工作",
        "比较",
        "survey",
        "literature",
        "papers",
        "state of the art",
        "sota",
    )
    freshness_terms = (
        "最新",
        "最近",
        "2026",
        "2025",
        "recent",
        "latest",
        "github",
        "hugging face",
        "hf",
        "开源",
        "代码",
        "项目主页",
        "leaderboard",
        "benchmark",
    )
    author_profile_terms = (
        "老师",
        "教授",
        "导师",
        "课题组",
        "个人主页",
        "主页",
        "publication",
        "publications",
        "google scholar",
        "scholar",
        "dblp",
        "openreview",
    )

    need_learning_plan = any(term in text for term in learning_terms)
    need_idea_generation = any(term in text for term in idea_terms)
    is_deep_read = any(term in text for term in deep_read_terms)
    is_literature = any(term in text for term in literature_terms)
    need_web_search = any(term in text for term in freshness_terms) or any(
        term in text for term in author_profile_terms
    )

    task_types: list[str] = []
    if is_deep_read:
        task_types.append("paper_deep_read")
    if need_idea_generation:
        task_types.append("idea_generation")
    if need_learning_plan:
        task_types.append("learning_plan")
    if is_literature:
        task_types.append("literature_review")
    if not task_types:
        task_types.append("quick_answer")

    if need_idea_generation:
        task_type = "idea_generation"
        need_paper_search = True
        need_pdf_reading = True
        answer_style = "research_ideas"
        need_web_search = True
    elif need_learning_plan:
        task_type = "learning_plan"
        need_paper_search = is_literature or "具身" in question or "robot" in text
        need_pdf_reading = need_paper_search
        answer_style = "learning_plan"
    elif is_deep_read:
        task_type = "paper_deep_read"
        need_paper_search = True
        need_pdf_reading = True
        answer_style = "deep_reading"
    elif is_literature:
        task_type = "literature_review"
        need_paper_search = True
        need_pdf_reading = True
        answer_style = "survey"
    else:
        task_type = "quick_answer"
        need_paper_search = False
        need_pdf_reading = False
        answer_style = "concise_explanation"
    direct_answer = heuristic_direct_answer(question) if task_type == "quick_answer" else ""

    steps = build_heuristic_steps(
        question=question,
        task_types=task_types,
        need_web_search=need_web_search,
        need_paper_search=need_paper_search,
        need_pdf_reading=need_pdf_reading,
    )

    return RouteDecision(
        task_type=task_type,
        task_types=task_types,
        workflow_goal=f"根据用户问题规划并执行任务：{question}",
        steps=steps,
        next_step=steps[0].id if steps else None,
        need_web_search=need_web_search,
        need_paper_search=need_paper_search,
        need_pdf_reading=need_pdf_reading,
        need_learning_plan=need_learning_plan,
        need_idea_generation=need_idea_generation,
        answer_style=answer_style,
        confidence=0.62,
        reasoning="heuristic keyword routing",
        direct_answer=direct_answer,
        source="heuristic",
    )


def _router_model(args: Namespace) -> str | None:
    model = args.router_llm or args.agent_llm or args.llm
    if not model:
        return None
    if args.openai_base_url and not args.disable_openai_compatible_config:
        return normalize_openai_compatible_model(model, args, args.openai_base_url)
    return model


def _extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError as first_exc:
        first_error = first_exc

    for candidate in _balanced_json_candidates(text):
        try:
            parsed = json.loads(candidate)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            continue

    preview = shorten(text.replace("\n", " "), width=500, placeholder="...")
    raise RouterOutputParseError(
        f"Planner LLM returned invalid JSON: {first_error}. Output preview: {preview}"
    )


def _balanced_json_candidates(text: str) -> list[str]:
    """Return balanced top-level JSON object substrings from an LLM response."""

    candidates: list[str] = []
    start: int | None = None
    depth = 0
    in_string = False
    escape = False
    for idx, char in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = idx
            depth += 1
        elif char == "}":
            if depth:
                depth -= 1
                if depth == 0 and start is not None:
                    candidates.append(text[start : idx + 1])
                    start = None
    return candidates


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    if value is None:
        return default
    return bool(value)


def _coerce_task_types(value: Any, fallback: list[str]) -> list[str]:
    if isinstance(value, str):
        raw_items = re.split(r"[,;/\n]+", value)
    elif isinstance(value, list):
        raw_items = value
    else:
        raw_items = fallback

    task_types: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        task_type = str(item).strip()
        if task_type in TASK_TYPES and task_type not in seen:
            seen.add(task_type)
            task_types.append(task_type)

    return task_types or fallback or ["quick_answer"]


def _primary_task_type(task_types: list[str], fallback: str) -> str:
    for task_type in TASK_PRIORITY:
        if task_type in task_types:
            return task_type
    return fallback if fallback in TASK_TYPES else "quick_answer"


def normalize_agent_name(agent: str) -> str:
    """Map planner free-form agent names into the supported agent registry."""

    clean = re.sub(r"\s+", "", str(agent or "")).strip()
    if clean in ALLOWED_AGENTS:
        return clean
    if clean in AGENT_ALIASES:
        return AGENT_ALIASES[clean]
    lowered = clean.lower()
    for name in ALLOWED_AGENTS:
        if lowered == name.lower():
            return name
    for alias, name in AGENT_ALIASES.items():
        if lowered == alias.lower():
            return name
    if "paperqa" in lowered or "reader" in lowered or "pdf" in lowered:
        return "ExpertResearchAgent"
    if "metadata" in lowered or "papersearch" in lowered or "literaturesearch" in lowered:
        return "ExpertResearchAgent"
    if "triage" in lowered or "rank" in lowered or "select" in lowered:
        return "ExpertResearchAgent"
    if "web" in lowered or "github" in lowered or "project" in lowered:
        return "ExpertResearchAgent"
    if "expert" in lowered or "research" in lowered:
        return "ExpertResearchAgent"
    if "learn" in lowered or "curriculum" in lowered or "roadmap" in lowered:
        return "SynthesisAgent"
    if "idea" in lowered or "innovation" in lowered:
        return "SynthesisAgent"
    if "topic" in lowered or "extract" in lowered:
        return "ExpertResearchAgent"
    if "persona" in lowered or "perspective" in lowered:
        return "ResearchPlanningAgent"
    if "question" in lowered or "queryplanner" in lowered:
        return "ResearchPlanningAgent"
    if "clarif" in lowered or "scope" in lowered:
        return "SynthesisAgent"
    if "synth" in lowered or "final" in lowered or "summary" in lowered:
        return "SynthesisAgent"
    if "tutor" in lowered or "answer" in lowered:
        return "SynthesisAgent"
    return "SynthesisAgent"


def _coerce_steps(value: Any, fallback: list[PlanStep], question: str) -> list[PlanStep]:
    if not isinstance(value, list):
        return fallback

    steps: list[PlanStep] = []
    seen: set[str] = set()
    for idx, item in enumerate(value, start=1):
        if not isinstance(item, dict):
            continue
        raw_id = str(item.get("id") or f"step_{idx}").strip()
        step_id = re.sub(r"[^a-zA-Z0-9_]+", "_", raw_id).strip("_").lower()
        if not step_id or step_id in seen:
            step_id = f"step_{idx}"
        seen.add(step_id)
        agent = normalize_agent_name(str(item.get("agent") or "SynthesisAgent").strip())
        objective = str(item.get("objective") or item.get("task") or "").strip()
        if not objective:
            objective = f"执行第 {idx} 个计划步骤。"
        depends_on = [
            str(dep).strip()
            for dep in item.get("depends_on", [])
            if str(dep).strip()
        ]
        input_keys = [
            str(key).strip()
            for key in item.get("input_keys", item.get("inputs", []))
            if str(key).strip()
        ] or ["question"]
        output_keys = [
            str(key).strip()
            for key in item.get("output_keys", item.get("outputs", []))
            if str(key).strip()
        ] or [f"{step_id}_output"]
        granularity = str(item.get("granularity") or "coarse").strip().lower()
        if granularity not in {"coarse", "refined"}:
            granularity = "coarse"
        raw_substeps = item.get("substeps")
        substeps = None
        if isinstance(raw_substeps, list):
            substeps = [
                str(substep).strip()
                for substep in raw_substeps
                if str(substep).strip()
            ] or None
        status = str(item.get("status") or "pending").strip() or "pending"
        steps.append(
            PlanStep(
                id=step_id,
                agent=agent,
                objective=objective,
                depends_on=depends_on,
                input_keys=input_keys,
                output_keys=output_keys,
                granularity=granularity,
                substeps=substeps,
                status=status,
            )
        )

    return steps or fallback or build_heuristic_steps(
        question=question,
        task_types=["quick_answer"],
        need_web_search=False,
        need_paper_search=False,
        need_pdf_reading=False,
    )


def _decision_from_payload(
    payload: dict[str, Any],
    fallback: RouteDecision,
    question: str,
) -> RouteDecision:
    task_types = _coerce_task_types(payload.get("task_types"), fallback.task_types)
    task_type = str(payload.get("task_type") or "").strip()
    if task_type not in TASK_TYPES or task_type not in task_types:
        task_type = _primary_task_type(task_types, fallback.task_type)

    confidence_raw = payload.get("confidence", fallback.confidence)
    try:
        confidence = max(0.0, min(1.0, float(confidence_raw)))
    except (TypeError, ValueError):
        confidence = fallback.confidence

    need_paper_search = _coerce_bool(
        payload.get("need_paper_search"),
        fallback.need_paper_search,
    )
    need_pdf_reading = _coerce_bool(
        payload.get("need_pdf_reading"),
        fallback.need_pdf_reading,
    )
    if need_pdf_reading:
        need_paper_search = True

    need_learning_plan = _coerce_bool(
        payload.get("need_learning_plan"),
        fallback.need_learning_plan,
    ) or "learning_plan" in task_types
    need_idea_generation = _coerce_bool(
        payload.get("need_idea_generation"),
        fallback.need_idea_generation,
    ) or "idea_generation" in task_types

    need_web_search = _coerce_bool(
        payload.get("need_web_search"),
        fallback.need_web_search,
    )
    default_steps = build_heuristic_steps(
        question=question,
        task_types=task_types,
        need_web_search=need_web_search,
        need_paper_search=need_paper_search,
        need_pdf_reading=need_pdf_reading,
    )
    steps = _coerce_steps(
        payload.get("steps"),
        default_steps,
        question,
    )
    next_step = str(payload.get("next_step") or "").strip() or None
    if next_step not in {step.id for step in steps}:
        next_step = steps[0].id if steps else None

    return RouteDecision(
        task_type=task_type,
        task_types=task_types,
        workflow_goal=str(payload.get("workflow_goal") or fallback.workflow_goal),
        steps=steps,
        next_step=next_step,
        need_web_search=need_web_search,
        need_paper_search=need_paper_search,
        need_pdf_reading=need_pdf_reading,
        need_learning_plan=need_learning_plan,
        need_idea_generation=need_idea_generation,
        answer_style=str(payload.get("answer_style") or fallback.answer_style),
        confidence=confidence,
        reasoning=str(payload.get("reasoning") or fallback.reasoning),
        direct_answer=str(payload.get("direct_answer") or ""),
        source="llm",
    )


def _agent_prompt_catalog() -> str:
    return "\n".join(
        f"- {name}: {description}"
        for name, description in AGENT_DESCRIPTIONS.items()
    )


async def _repair_router_json(
    broken_text: str,
    base_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Ask the same LLM provider to convert malformed planner text to valid JSON."""

    repair_kwargs = {
        key: value
        for key, value in base_kwargs.items()
        if key not in {"messages", "max_tokens"}
    }
    repair_kwargs["max_tokens"] = 2200
    repair_kwargs["messages"] = [
        {
            "role": "system",
            "content": (
                "You repair malformed JSON. Return ONLY one valid JSON object. "
                "Do not add markdown, comments, or explanations. Preserve the original "
                "meaning and required keys where possible."
            ),
        },
        {
            "role": "user",
            "content": (
                "Repair this planner output into valid JSON with keys task_type, "
                "task_types, workflow_goal, steps, next_step, need_web_search, "
                "need_paper_search, need_pdf_reading, need_learning_plan, "
                "need_idea_generation, answer_style, confidence, reasoning, "
                "direct_answer:\n\n"
                f"{broken_text}"
            ),
        },
    ]

    import litellm

    response = await litellm.acompletion(**repair_kwargs)
    content = response.choices[0].message.content or ""
    return _extract_json_object(content)


async def route_question(
    question: str,
    args: Namespace,
    memory_context: str = "",
) -> RouteDecision:
    """Plan which agents should run and how their outputs should connect.

    This is the first SupervisorAgent layer. It still returns route booleans for
    current graph compatibility, but the important output is a dependency-aware
    list of planned worker steps.
    """

    fallback = heuristic_route(question)
    if args.router_mode == "heuristic":
        return fallback

    model = _router_model(args)
    if not model:
        return fallback

    try:
        import litellm

        agent_catalog = _agent_prompt_catalog()
        memory_block = memory_context.strip() or "N/A"

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a router for a multi-agent research learning assistant. "
                        "Act as a Planner/Supervisor. Classify the user's request, decide which "
                        "agent capabilities are needed, and produce a dependency-aware workflow plan. "
                        "You may use compact thread memory only to resolve follow-up references "
                        "such as 'this paper', 'that direction', or 'continue'. Do not treat "
                        "memory as factual evidence and do not let it override the current user request. "
                        "Planning guidance for memory follow-ups: if the user asks to list, send, "
                        "show, or provide links/PDFs for papers from the previous turn, inspect "
                        "compact memory. If memory already contains enough selected_papers to answer, "
                        "you may classify it as quick_answer with need_web_search=false, "
                        "need_paper_search=false, and need_pdf_reading=false. If the user asks to "
                        "search, update, find more, or investigate new papers beyond memory, plan a "
                        "new research workflow instead. "
                        "For simple quick_answer requests, put the actual concise answer in "
                        "direct_answer and do not invent a separate answering worker. "
                        "This is a multi-label decision: the user may need literature review, "
                        "learning planning, and research idea generation at the same time. "
                        "Return ONLY a JSON object with keys: task_type, task_types, workflow_goal, "
                        "steps, next_step, need_web_search, "
                        "need_paper_search, need_pdf_reading, need_learning_plan, "
                        "need_idea_generation, answer_style, confidence, reasoning, direct_answer. "
                        "need_pdf_reading is only a planner hint for reporting; ExpertResearchAgent "
                        "owns the final decision of whether to call internal PaperQA reader tools. "
                        "task_type is the primary label. task_types is an array of all applicable labels. "
                        "Allowed labels are quick_answer, literature_review, learning_plan, "
                        "idea_generation, paper_deep_read. Do not include quick_answer if any "
                        "more specific label applies. steps must be an array of objects with id, agent, "
                        "objective, depends_on, input_keys, output_keys, optional granularity, "
                        "optional substeps, and optional status. "
                        "You may ONLY use these agent names in steps:\n"
                        f"{agent_catalog}\n"
                        "Only plan agents that are currently implemented in the outer LangGraph. "
                        "Do NOT include PerspectiveAgent, QuestionPlannerAgent, ScopeClarificationAgent, "
                        "TopicExtractionAgent, LearningPlannerAgent, IdeaThinkAgent, WebSearchAgent, "
                        "MetadataSearchAgent, PaperTriageAgent, or PaperQAReaderAgent as separate steps. If scope is "
                        "unclear, make a reasonable assumption in reasoning instead of adding a "
                        "clarification step. If topic extraction, learning planning, or idea "
                        "generation seems useful, mention it in reasoning or answer_style, but "
                        "do not add an unimplemented worker step yet. "
                        "Prefer coarse-grained worker-agent steps. Set granularity='coarse' by default. "
                        "Do NOT expand internal tool calls for normal requests. "
                        "ResearchPlanningAgent must stay before ExpertResearchAgent "
                        "when a literature task needs broad coverage. ExpertResearchAgent is the enhanced "
                        "STORM expert: it internally handles web search, metadata search, paper triage, "
                        "PDF caching, PaperQA paper_search, gather_evidence, and answer generation. "
                        "Do not plan those internal tools as separate worker agents. Only set "
                        "granularity='refined' and add "
                        "short substeps when the user's request is unusual, ambiguous, or requires a "
                        "nonstandard strategy inside that worker. "
                        "When one task depends on another, make that dependency explicit. For example, "
                        "if the user asks to analyze a target paper and then research related directions, "
                        "schedule ResearchPlanningAgent if broad coverage is needed, "
                        "then ExpertResearchAgent; the expert internally handles target-paper reading, "
                        "related-topic extraction, web search, metadata search, and PDF evidence. "
                        "Keep objectives concise. Return valid JSON only: double-quoted strings, "
                        "no trailing commas, no markdown, no comments."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Current user request:\n{question}\n\n"
                        "Compact thread memory:\n"
                        f"{memory_block}"
                    ),
                },
            ],
            "temperature": 0,
            "max_tokens": 2200,
            "timeout": float(getattr(args, "llm_timeout", 180.0)),
            "response_format": {"type": "json_object"},
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
        content = response.choices[0].message.content or ""
        try:
            payload = _extract_json_object(content)
        except RouterOutputParseError:
            payload = await _repair_router_json(content, kwargs)
        return _decision_from_payload(payload, fallback, question)
    except Exception as exc:
        if args.router_mode == "llm":
            raise RuntimeError(f"RouterAgent LLM call failed: {exc}") from exc
        fallback.reasoning = f"LLM router unavailable, used heuristic fallback: {exc}"
        fallback.source = "heuristic_fallback"
        return fallback
