from __future__ import annotations

import re
import json
from dataclasses import asdict, dataclass
from html.parser import HTMLParser
from textwrap import shorten
from typing import Any

import requests

from scripts.ask_literature import build_search_queries, openai_compatible_api_key

from .router import _router_model
from .web_search import run_web_search_agent


DEFAULT_PERSONA = (
    "Basic fact writer: Basic fact writer focusing on broadly covering the basic "
    "facts about the topic."
)


@dataclass(slots=True)
class Perspective:
    name: str
    description: str
    persona: str
    source: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ResearchQuestion:
    perspective: str
    question: str
    queries: list[str]
    coverage_targets: list[str] | None = None
    signal_sources: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ResearchPlanItem:
    perspective: str
    description: str
    persona: str
    research_question: str
    queries: list[str]
    coverage_targets: list[str]
    signal_sources: list[str]
    source: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class _WikiOutlineParser(HTMLParser):
    """Small HTML parser for STORM-style related-page title and outline extraction."""

    def __init__(self) -> None:
        super().__init__()
        self._current_tag: str | None = None
        self._buffer: list[str] = []
        self.title = ""
        self.headings: list[tuple[int, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            self._current_tag = tag
            self._buffer = []

    def handle_data(self, data: str) -> None:
        if self._current_tag:
            self._buffer.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag != self._current_tag:
            return
        text = " ".join("".join(self._buffer).split())
        text = text.replace("[edit]", "").strip().replace("\xa0", " ")
        if tag == "h1" and not self.title:
            self.title = text
        elif tag and tag.startswith("h") and len(tag) == 2 and tag[1:].isdigit():
            level = int(tag[1:])
            if text:
                self.headings.append((level, text))
        self._current_tag = None
        self._buffer = []


def get_wiki_page_title_and_toc(url: str, timeout: int = 8) -> tuple[str, str]:
    """Mirror STORM's title/TOC extraction without adding BeautifulSoup as a dependency."""

    response = requests.get(url, timeout=timeout, headers={"User-Agent": "EmbodiedAI-KB/0.1"})
    response.raise_for_status()
    parser = _WikiOutlineParser()
    parser.feed(response.text)

    excluded_sections = {
        "Contents",
        "See also",
        "Notes",
        "References",
        "External links",
    }
    levels: list[int] = []
    toc_lines: list[str] = []
    for level, title in parser.headings:
        if level < 2 or title in excluded_sections:
            continue
        while levels and level <= levels[-1]:
            levels.pop()
        levels.append(level)
        toc_lines.append(f"{'  ' * (len(levels) - 1)}{title}")

    return parser.title or url, "\n".join(toc_lines).strip()


def _parse_numbered_items(text: str) -> list[str]:
    items: list[str] = []
    for line in text.splitlines():
        clean = line.strip()
        match = re.match(r"^\s*(?:\d+[\.)]|[-*])\s*(.+)$", clean)
        if match:
            clean = match.group(1).strip()
        if clean:
            items.append(clean)
    return items


def _parse_urls(text: str) -> list[str]:
    seen: set[str] = set()
    urls: list[str] = []
    for url in re.findall(r"https?://[^\s)<>\]]+", text):
        url = url.rstrip(".,;")
        if url not in seen:
            seen.add(url)
            urls.append(url)
    return urls


def _parse_bullet_queries(text: str, limit: int) -> list[str]:
    queries: list[str] = []
    seen: set[str] = set()
    for line in text.splitlines():
        clean = re.sub(r"^\s*(?:[-*]|\d+[\.)])\s*", "", line).strip().strip('"')
        clean = re.sub(r"\s+", " ", clean)
        if clean.count('"') % 2:
            clean = clean.replace('"', "")
        key = clean.lower()
        if clean and key not in seen:
            seen.add(key)
            queries.append(clean)
        if len(queries) >= limit:
            break
    return queries


def _extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in ResearchPlanningAgent output.")
    parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("ResearchPlanningAgent output must be a JSON object.")
    return parsed


def _planning_scout_queries(topic: str) -> list[str]:
    """Broad web queries used before ResearchPlanningAgent writes final queries."""

    normalized = " ".join(topic.split())
    lowered = normalized.lower()
    queries: list[str] = []

    def add(query: str) -> None:
        query = re.sub(r"\s+", " ", query).strip()
        if query and query.lower() not in {item.lower() for item in queries}:
            queries.append(query)

    if "vla" in lowered or "vision-language-action" in lowered or "视觉" in normalized:
        add("2026 VLA robot manipulation OpenReview ICLR ICRA")
        add("2026 vision language action robot manipulation HuggingFace Papers")
        add("VLA robot manipulation 2026 arXiv GitHub project")
    else:
        add(f"{normalized} latest papers 2026 OpenReview arXiv")
        add(f"{normalized} survey benchmark GitHub")

    add(normalized)
    return queries[:3]


def _format_planning_web_context(items: list[dict[str, Any]], limit: int = 10) -> str:
    lines: list[str] = []
    for idx, item in enumerate(items[:limit], start=1):
        title = re.sub(r"\s+", " ", str(item.get("title") or "Untitled")).strip()
        url = str(item.get("url") or "").strip()
        snippet = re.sub(r"\s+", " ", str(item.get("snippet") or "")).strip()
        if snippet:
            snippet = shorten(snippet, width=450, placeholder="...")
            lines.append(f"{idx}. {title}\n   url={url}\n   snippet={snippet}")
        else:
            lines.append(f"{idx}. {title}\n   url={url}")
    return "\n".join(lines)


async def _planning_web_scout(topic: str, args: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if getattr(args, "disable_web_search", False):
        return [], {
            "provider": "disabled",
            "query_count": 0,
            "result_count": 0,
            "errors": [],
            "scout_queries": [],
        }

    scout_queries = _planning_scout_queries(topic)
    try:
        evidence, trace = await run_web_search_agent(scout_queries, args)
        trace = {**trace, "scout_queries": scout_queries}
        return evidence, trace
    except Exception as exc:
        return [], {
            "provider": "error",
            "query_count": len(scout_queries),
            "result_count": 0,
            "errors": [{"query": " | ".join(scout_queries), "error": str(exc)}],
            "scout_queries": scout_queries,
        }


async def _acompletion_text(
    args: Any,
    messages: list[dict[str, str]],
    max_tokens: int = 900,
) -> str:
    model = _router_model(args)
    if not model:
        return ""

    import litellm

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
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
    return response.choices[0].message.content or ""


def fallback_perspectives(topic: str, max_num_persona: int) -> list[Perspective]:
    personas = [
        DEFAULT_PERSONA,
        "Model architecture writer: focuses on VLA backbones, action heads, multimodal fusion, and reasoning mechanisms.",
        "Data and benchmark writer: focuses on datasets, evaluation protocols, benchmark gaps, and scaling signals.",
        "Deployment and systems writer: focuses on real robot deployment, latency, safety, generalization, and open-source implementations.",
        "Learning roadmap writer: focuses on prerequisites, study sequence, reproduction practice, and milestones.",
    ]
    selected = personas[: max(1, max_num_persona + 1)]
    return [_perspective_from_persona(persona, "fallback") for persona in selected]


def _perspective_from_persona(persona: str, source: str) -> Perspective:
    if ":" in persona:
        name, description = persona.split(":", 1)
    else:
        name, description = persona, persona
    return Perspective(
        name=name.strip(),
        description=description.strip(),
        persona=persona.strip(),
        source=source,
    )


async def generate_perspectives(topic: str, args: Any) -> tuple[list[Perspective], dict[str, Any]]:
    """STORM-style PerspectiveAgent.

    This mirrors STORM's logic:
    1. Find related pages/topics for inspiration.
    2. Fetch their title/table-of-contents examples.
    3. Generate Wikipedia-writer personas.
    4. Prepend STORM's default Basic fact writer persona.
    """

    max_num_persona = int(getattr(args, "storm_max_perspectives", 3))
    if getattr(args, "router_mode", "auto") == "heuristic" or not _router_model(args):
        return fallback_perspectives(topic, max_num_persona), {
            "related_topics": "",
            "examples": ["N/A"],
            "source": "fallback",
        }

    try:
        related_topics = await _acompletion_text(
            args,
            [
                {
                    "role": "system",
                    "content": (
                        "I'm writing a Wikipedia-style research page for a topic. "
                        "Identify and recommend closely related Wikipedia pages or canonical "
                        "overview pages that can inspire useful perspectives. "
                        "Please list URLs in separate lines only."
                    ),
                },
                {"role": "user", "content": f"Topic of interest: {topic}"},
            ],
            max_tokens=500,
        )

        examples: list[str] = []
        for url in _parse_urls(related_topics)[:5]:
            try:
                title, toc = get_wiki_page_title_and_toc(url)
                examples.append(f"Title: {title}\nTable of Contents: {toc}")
            except Exception:
                continue
        if not examples:
            examples.append("N/A")

        persona_text = await _acompletion_text(
            args,
            [
                {
                    "role": "system",
                    "content": (
                        "You need to select a group of Wikipedia editors who will work "
                        "together to create a comprehensive article on the topic. Each editor "
                        "represents a different perspective, role, or affiliation related to this topic. "
                        "Use the related page outlines for inspiration. For each editor, add a "
                        "description of what they will focus on. Give your answer in this format:\n"
                        "1. short summary of editor 1: description\n"
                        "2. short summary of editor 2: description\n..."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Topic of interest: {topic}\n\n"
                        "Wiki page outlines of related topics for inspiration:\n"
                        + "\n----------\n".join(examples)
                    ),
                },
            ],
            max_tokens=900,
        )
        personas = _parse_numbered_items(persona_text)
        considered = [DEFAULT_PERSONA] + personas[:max_num_persona]
        if len(considered) == 1:
            return fallback_perspectives(topic, max_num_persona), {
                "related_topics": related_topics,
                "examples": examples,
                "source": "fallback_after_empty_llm_personas",
            }
        return [_perspective_from_persona(persona, "llm") for persona in considered], {
            "related_topics": related_topics,
            "examples": examples,
            "source": "llm",
        }
    except Exception as exc:
        if getattr(args, "router_mode", "auto") == "llm":
            raise RuntimeError(f"PerspectiveAgent LLM call failed: {exc}") from exc
        return fallback_perspectives(topic, max_num_persona), {
            "related_topics": "",
            "examples": ["N/A"],
            "source": f"fallback_after_error:{type(exc).__name__}",
        }


def fallback_research_questions(
    topic: str,
    perspectives: list[Perspective],
    args: Any,
) -> list[ResearchQuestion]:
    questions: list[ResearchQuestion] = []
    query_limit = int(getattr(args, "storm_search_queries_per_question", 2))
    author_like = any(
        marker in topic.lower()
        for marker in (
            "老师",
            "教授",
            "导师",
            "课题组",
            "个人主页",
            "publication",
            "publications",
            "google scholar",
            "scholar",
            "dblp",
            "openreview",
        )
    )
    for perspective in perspectives:
        name = perspective.name.lower()
        if author_like:
            question = f"围绕用户问题“{topic}”，先完成作者消歧、近期论文收集和研究主题聚类。"
            queries = [
                f"{topic} publications Google Scholar OpenReview arXiv",
                f"{topic} 个人主页 论文 研究方向",
            ]
        elif name.startswith("basic fact"):
            question = f"围绕用户问题“{topic}”，有哪些代表性论文、核心概念、时间线和主要趋势？"
            queries = [
                "2026 vision language action robot manipulation representative papers",
                "2026 VLA robot manipulation arxiv ICRA ICLR code",
            ]
        elif "architecture" in name or "model" in name:
            question = f"围绕用户问题“{topic}”，模型架构路线有哪些差异，例如 VLM backbone、action head、reasoning-control bridge？"
            queries = [
                "2026 VLA robot manipulation model architecture action head",
                "vision language action robot manipulation transformer diffusion policy 2026",
            ]
        elif "data" in name or "benchmark" in name:
            question = f"围绕用户问题“{topic}”，关键数据集、真实机器人评测和 benchmark 设置是什么？"
            queries = [
                "2026 VLA robot manipulation dataset benchmark real robot evaluation",
                "vision language action manipulation benchmark dataset 2026",
            ]
        elif "deployment" in name or "system" in name:
            question = f"围绕用户问题“{topic}”，真实部署、实时推理、安全、开源系统和跨具身泛化有哪些代表工作？"
            queries = [
                "2026 VLA real robot deployment open source system latency safety",
                "vision language action cross embodiment generalist robot 2026",
            ]
        else:
            question = f"从 {perspective.name} 角度调研“{topic}”时，需要覆盖哪些论文、证据和局限？"
            queries = build_search_queries(f"{topic} {perspective.description}", None)
        questions.append(
            ResearchQuestion(
                perspective=perspective.name,
                question=question,
                queries=queries[:query_limit],
                coverage_targets=[
                    "representative works",
                    "core methods",
                    "datasets and benchmarks",
                    "limitations",
                ],
            )
        )
    return questions


def _plan_items_from_parts(
    perspectives: list[Perspective],
    questions: list[ResearchQuestion],
    source: str,
) -> list[ResearchPlanItem]:
    perspective_by_name = {item.name: item for item in perspectives}
    plan: list[ResearchPlanItem] = []
    for question in questions:
        perspective = perspective_by_name.get(question.perspective)
        plan.append(
            ResearchPlanItem(
                perspective=question.perspective,
                description=perspective.description if perspective else question.perspective,
                persona=perspective.persona if perspective else question.perspective,
                research_question=question.question,
                queries=question.queries,
                coverage_targets=question.coverage_targets
                or [
                    "representative works",
                    "technical trend",
                    "evidence from papers",
                ],
                signal_sources=question.signal_sources or ["user_question"],
                source=source,
            )
        )
    return plan


def _normalize_plan_item(
    raw: dict[str, Any],
    idx: int,
    query_limit: int,
) -> ResearchPlanItem | None:
    perspective = str(
        raw.get("perspective")
        or raw.get("perspective_name")
        or raw.get("name")
        or f"Perspective {idx}"
    ).strip()
    question = str(
        raw.get("research_question")
        or raw.get("question")
        or raw.get("subquestion")
        or ""
    ).strip()
    if not perspective or not question:
        return None

    raw_queries = raw.get("queries") or raw.get("search_queries") or []
    queries: list[str] = []
    if isinstance(raw_queries, str):
        queries = _parse_bullet_queries(raw_queries, query_limit)
    elif isinstance(raw_queries, list):
        for query in raw_queries:
            clean = re.sub(r"\s+", " ", str(query)).strip().strip('"')
            if clean.count('"') % 2:
                clean = clean.replace('"', "")
            if clean:
                queries.append(clean)
            if len(queries) >= query_limit:
                break
    if not queries:
        queries = build_search_queries(question, None)[:query_limit]

    raw_targets = raw.get("coverage_targets") or raw.get("targets") or []
    coverage_targets: list[str] = []
    if isinstance(raw_targets, str):
        coverage_targets = _parse_numbered_items(raw_targets)
    elif isinstance(raw_targets, list):
        coverage_targets = [
            re.sub(r"\s+", " ", str(item)).strip()
            for item in raw_targets
            if str(item).strip()
        ]
    if not coverage_targets:
        coverage_targets = [
            "representative works",
            "topic-specific evidence",
            "limitations or open gaps",
        ]

    raw_signal_sources = (
        raw.get("signal_sources")
        or raw.get("signals")
        or raw.get("evidence_signals")
        or raw.get("source_signals")
        or []
    )
    signal_sources: list[str] = []
    if isinstance(raw_signal_sources, str):
        signal_sources = _parse_numbered_items(raw_signal_sources)
    elif isinstance(raw_signal_sources, list):
        signal_sources = [
            re.sub(r"\s+", " ", str(item)).strip()
            for item in raw_signal_sources
            if str(item).strip()
        ]
    if not signal_sources:
        signal_sources = ["user_question"]

    description = str(
        raw.get("description")
        or raw.get("focus")
        or raw.get("role")
        or perspective
    ).strip()
    persona = str(raw.get("persona") or f"{perspective}: {description}").strip()
    return ResearchPlanItem(
        perspective=perspective,
        description=description,
        persona=persona,
        research_question=question,
        queries=queries,
        coverage_targets=coverage_targets,
        signal_sources=signal_sources,
        source="llm",
    )


async def plan_research(
    topic: str,
    args: Any,
    memory_context: str = "",
) -> dict[str, Any]:
    """ResearchPlanningAgent.

    This merges the earlier PerspectiveAgent and QuestionPlannerAgent into one
    worker: it creates coverage perspectives and immediately turns each
    perspective into a concrete research question plus search queries for the
    ExpertResearchAgent.
    """

    max_num_persona = int(getattr(args, "storm_max_perspectives", 3))
    query_limit = int(getattr(args, "storm_search_queries_per_question", 2))
    max_queries = int(getattr(args, "storm_max_search_queries", 8))

    def fallback(source: str = "fallback") -> dict[str, Any]:
        perspectives = fallback_perspectives(topic, max_num_persona)
        questions = fallback_research_questions(topic, perspectives, args)
        plan = _plan_items_from_parts(perspectives, questions, source)
        return _format_research_plan_response(
            topic=topic,
            plan=plan,
            max_queries=max_queries,
            trace={"source": source, "mode": "fallback"},
            planning_web_evidence=[],
            planning_web_trace={},
        )

    if getattr(args, "router_mode", "auto") == "heuristic" or not _router_model(args):
        return fallback()

    try:
        planning_web_evidence, planning_web_trace = await _planning_web_scout(topic, args)
        planning_web_context = _format_planning_web_context(planning_web_evidence)
        memory_block = memory_context.strip() or "N/A"
        content = await _acompletion_text(
            args,
            [
                {
                    "role": "system",
                    "content": (
                        "You are ResearchPlanningAgent for a literature-learning assistant. "
                        "Generate a compact multi-perspective research plan. Each perspective "
                        "must become one concrete research subquestion and 1-3 high-precision "
                        "English keyword queries for web search, paper metadata search, and PDF "
                        "reading. Keep each query concise: prefer 5-10 keywords, no full "
                        "sentences, no polite phrases, no unmatched quotation marks. Include "
                        "topic-specific terms in every query; avoid generic queries like "
                        "'robotics robot learning' or 'vision language action papers'. Include "
                        "paper names/acronyms when the user names or implies a known work. "
                        "The user is not available for clarification, so make reasonable "
                        "scope assumptions inside the plan. Do not use a fixed checklist of "
                        "method/data/evaluation/deployment/safety perspectives. Derive the "
                        "perspectives from the actual signals available here: the user's "
                        "wording, compact thread memory, web scouting context, and necessary "
                        "domain knowledge. Use method, data, evaluation, deployment, safety, "
                        "or learning-path lenses only when they are directly useful for this "
                        "specific request. Make perspectives non-overlapping and task-specific. "
                        "Use web scouting context to notice fresh hotspots, venues, project pages, "
                        "people, and paper names, but do not treat snippets as final evidence. "
                        "Use compact thread memory only to resolve follow-up phrases like "
                        "'this paper' or 'that direction'; do not treat memory as evidence and "
                        "do not overfit to stale papers. For every perspective, include "
                        "signal_sources naming why it exists, using short labels such as "
                        "user_question, memory, web_scout, paper_title, author_identity, or "
                        "domain_knowledge. Return ONLY valid JSON, no markdown. Schema:\n"
                        "{\n"
                        '  "research_plan": [\n'
                        "    {\n"
                        '      "perspective": "short name",\n'
                        '      "description": "what this perspective covers",\n'
                        '      "research_question": "one concrete question in Chinese",\n'
                        '      "queries": ["query 1", "query 2"],\n'
                        '      "coverage_targets": ["target 1", "target 2"],\n'
                        '      "signal_sources": ["user_question", "web_scout"]\n'
                        "    }\n"
                        "  ]\n"
                        "}\n"
                        f"Generate exactly {max_num_persona + 1} perspectives. "
                        "Include a broad overview only when it helps orient the answer."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"User question/topic: {topic}\n\n"
                        "Compact thread memory for follow-up resolution:\n"
                        f"{memory_block}\n\n"
                        "Web scouting context for query planning only:\n"
                        f"{planning_web_context or 'N/A'}"
                    ),
                },
            ],
            max_tokens=1800,
        )
        payload = _extract_json_object(content)
        raw_items = payload.get("research_plan")
        if not isinstance(raw_items, list):
            return fallback("fallback_after_missing_research_plan")
        plan: list[ResearchPlanItem] = []
        for idx, raw in enumerate(raw_items, start=1):
            if not isinstance(raw, dict):
                continue
            item = _normalize_plan_item(raw, idx, query_limit)
            if item:
                plan.append(item)
        if not plan:
            return fallback("fallback_after_empty_research_plan")
        plan = plan[: max(1, max_num_persona + 1)]
        return _format_research_plan_response(
            topic=topic,
            plan=plan,
            max_queries=max_queries,
            trace={"source": "llm", "mode": "signal_driven_research_planning"},
            planning_web_evidence=planning_web_evidence,
            planning_web_trace=planning_web_trace,
        )
    except Exception as exc:
        if getattr(args, "router_mode", "auto") == "llm":
            raise RuntimeError(f"ResearchPlanningAgent LLM call failed: {exc}") from exc
        return fallback(f"fallback_after_error:{type(exc).__name__}")


def _format_research_plan_response(
    *,
    topic: str,
    plan: list[ResearchPlanItem],
    max_queries: int,
    trace: dict[str, Any],
    planning_web_evidence: list[dict[str, Any]] | None = None,
    planning_web_trace: dict[str, Any] | None = None,
) -> dict[str, Any]:
    queries: list[str] = []
    seen: set[str] = set()
    for item in plan:
        for query in item.queries:
            clean = re.sub(r"\s+", " ", query).strip()
            key = clean.lower()
            if clean and key not in seen:
                seen.add(key)
                queries.append(clean)

    for query in build_search_queries(topic, None):
        key = query.lower()
        if query and key not in seen:
            seen.add(key)
            queries.append(query)

    perspectives = [
        Perspective(
            name=item.perspective,
            description=item.description,
            persona=item.persona,
            source=item.source,
        ).to_dict()
        for item in plan
    ]
    research_questions = [
        ResearchQuestion(
            perspective=item.perspective,
            question=item.research_question,
            queries=item.queries,
            coverage_targets=item.coverage_targets,
            signal_sources=item.signal_sources,
        ).to_dict()
        for item in plan
    ]
    planning_trace = {
        **trace,
        "agent": "ResearchPlanningAgent",
        "perspective_count": len(plan),
        "query_count": min(len(queries), max_queries),
        "web_scout_result_count": len(planning_web_evidence or []),
    }
    return {
        "research_plan": [item.to_dict() for item in plan],
        "perspectives": perspectives,
        "research_questions": research_questions,
        "planning_web_evidence": planning_web_evidence or [],
        "planning_web_trace": planning_web_trace or {},
        "queries": queries[:max_queries],
        "planning_trace": planning_trace,
        # Compatibility alias for older display/synthesis code.
        "perspective_trace": planning_trace,
    }


async def generate_research_questions(
    topic: str,
    perspectives: list[Perspective],
    args: Any,
) -> list[ResearchQuestion]:
    """STORM-style QuestionPlannerAgent.

    For each persona, produce one retrieval-oriented research subquestion, then
    convert that subquestion into search queries. This keeps STORM's
    perspective-guided coverage, but avoids interactive clarification questions
    because our workflow has no human expert turn here.
    """

    if getattr(args, "router_mode", "auto") == "heuristic" or not _router_model(args):
        return fallback_research_questions(topic, perspectives, args)

    query_limit = int(getattr(args, "storm_search_queries_per_question", 2))
    planned: list[ResearchQuestion] = []
    try:
        for perspective in perspectives:
            question = await _acompletion_text(
                args,
                [
                    {
                        "role": "system",
                        "content": (
                            "You are QuestionPlannerAgent in a literature research workflow. "
                            "The user is NOT available for clarification. Given a topic and a "
                            "perspective/persona, write ONE concrete research subquestion that "
                            "ExpertResearchAgent can answer using web search, paper metadata search, "
                            "and PDF reading. Do not ask the user for preferences. Do not write "
                            "questions like 'do you want...', 'should we include...', or 'please "
                            "clarify...'. If scope is ambiguous, make a reasonable assumption inside "
                            "the question, such as focusing on public 2026 papers, robot manipulation, "
                            "real/sim evaluation, representative works, code availability, or technical "
                            "trends. Prefer Chinese for the subquestion. Return only the question."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Topic: {topic}\n"
                            f"Perspective/persona: {perspective.persona}\n"
                            "Output one retrieval-oriented research subquestion for ExpertResearchAgent."
                        ),
                    },
                ],
                max_tokens=350,
            )
            question = " ".join(question.split()).strip()
            if not question or question.startswith("Thank you so much for your help!"):
                continue

            query_text = await _acompletion_text(
                args,
                [
                    {
                        "role": "system",
                        "content": (
                            "Convert the research subquestion into search engine queries for finding "
                            "papers, project pages, code, benchmarks, and metadata. Return only a "
                            "bullet list. Use concise English keyword queries. Include years, venues, "
                            "and technical synonyms when useful. Do not use unmatched quotation marks. "
                            "Write the queries in the following format:\n"
                            "- query 1\n- query 2\n...\n- query n"
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Topic you are discussing about: {topic}\n"
                            f"Question you want to answer: {question}"
                        ),
                    },
                ],
                max_tokens=450,
            )
            queries = _parse_bullet_queries(query_text, query_limit)
            if not queries:
                queries = build_search_queries(question, None)[:query_limit]
            planned.append(
                ResearchQuestion(
                    perspective=perspective.name,
                    question=question,
                    queries=queries,
                )
            )
        return planned or fallback_research_questions(topic, perspectives, args)
    except Exception as exc:
        if getattr(args, "router_mode", "auto") == "llm":
            raise RuntimeError(f"QuestionPlannerAgent LLM call failed: {exc}") from exc
        return fallback_research_questions(topic, perspectives, args)


async def plan_storm_research(topic: str, args: Any) -> dict[str, Any]:
    perspectives, metadata = await generate_perspectives(topic, args)
    questions = await generate_research_questions(topic, perspectives, args)

    queries: list[str] = []
    seen: set[str] = set()
    for question in questions:
        for query in question.queries:
            clean = re.sub(r"\s+", " ", query).strip()
            key = clean.lower()
            if clean and key not in seen:
                seen.add(key)
                queries.append(clean)

    for query in build_search_queries(topic, getattr(args, "search_query", None)):
        key = query.lower()
        if query and key not in seen:
            seen.add(key)
            queries.append(query)

    max_queries = int(getattr(args, "storm_max_search_queries", 8))
    return {
        "perspectives": [perspective.to_dict() for perspective in perspectives],
        "research_questions": [question.to_dict() for question in questions],
        "queries": queries[:max_queries],
        "perspective_trace": {
            "related_topics": metadata.get("related_topics", ""),
            "related_topic_examples": [
                shorten(example.replace("\n", " "), width=300, placeholder="...")
                for example in metadata.get("examples", [])
            ],
            "source": metadata.get("source", "unknown"),
        },
    }
