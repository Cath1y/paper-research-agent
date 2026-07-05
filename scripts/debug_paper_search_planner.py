from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from embodiedai_kb.langgraph_workflow.paper_search_agent import (
    analyze_paper_search_query,
)


DEFAULT_QUESTIONS = [
    "帮我调研上海交通大学赵波老师2026年在研究的方向及其代表论文",
    "调研北大唐浩老师的2026年研究方向和代表论文",
    "帮我调研卢宗青老师最近在研究的论文及其关注点",
    "帮我调研 UC Berkeley Pieter Abbeel 最近的机器人学习论文",
    "帮我调研 Stanford Chelsea Finn 最近的机器人学习和具身智能论文",
]


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Debug PaperSearchAgent query planning only. This does not download PDFs "
            "or call arXiv/OpenAlex; it only asks the LLM to produce platform-specific queries."
        )
    )
    parser.add_argument(
        "questions",
        nargs="*",
        help="Questions to test. If omitted, a small author-disambiguation regression set is used.",
    )
    parser.add_argument(
        "--llm",
        default=os.getenv("PAPER_SEARCH_LLM")
        or os.getenv("AGENT_LLM")
        or os.getenv("ROUTER_LLM")
        or os.getenv("LLM")
        or "openai/gpt-5.2",
        help="LiteLLM model name for the query planner.",
    )
    parser.add_argument(
        "--openai-base-url",
        default=os.getenv("OPENAI_BASE_URL"),
        help="OpenAI-compatible API base URL.",
    )
    parser.add_argument(
        "--openai-api-key-env",
        default="OPENAI_API_KEY",
        help="Environment variable name that stores the API key.",
    )
    parser.add_argument(
        "--llm-timeout",
        type=float,
        default=90.0,
        help="Timeout for each query-planner LLM call.",
    )
    return parser.parse_args()


async def run_one(question: str, args: argparse.Namespace) -> None:
    workflow_args = SimpleNamespace(
        paper_search_llm=args.llm,
        agent_llm=None,
        router_llm=None,
        llm=None,
        openai_base_url=args.openai_base_url,
        disable_openai_compatible_config=False,
        openai_api_key_env=args.openai_api_key_env,
        llm_timeout=args.llm_timeout,
    )
    plan, trace = await analyze_paper_search_query(
        question=question,
        queries=[question],
        research_plan=[{"perspective": "paper search", "research_question": question}],
        args=workflow_args,
    )

    print("\n" + "=" * 100)
    print("QUESTION:", question)
    print("TRACE:", json.dumps(trace, ensure_ascii=False, indent=2))
    print("AUTHORS:", json.dumps(plan.authors, ensure_ascii=False))
    print("ARXIV:", json.dumps(plan.tool_queries.get("arxiv_search", []), ensure_ascii=False, indent=2))
    print("OPENALEX:", json.dumps(plan.tool_queries.get("openalex_search", []), ensure_ascii=False, indent=2))
    profile_tools = {
        key: value
        for key, value in plan.tool_queries.items()
        if key
        in {
            "author_homepage",
            "publication_page",
            "google_scholar_profile",
            "dblp_profile",
            "openreview_profile",
            "web_search",
        }
    }
    print("PROFILE/WEB:", json.dumps(profile_tools, ensure_ascii=False, indent=2))
    print("REASON:", plan.reasoning)


async def async_main() -> None:
    args = build_args()
    questions = args.questions or DEFAULT_QUESTIONS
    for question in questions:
        await run_one(question, args)


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
