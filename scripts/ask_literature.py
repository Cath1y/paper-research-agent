#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import csv
import gzip
import json
import os
import re
import shutil
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from textwrap import shorten
from typing import Any
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from embodiedai_kb.search.metadata_search import (
    DEFAULT_ARXIV_DB,
    DEFAULT_FRONTIER_DB,
    DEFAULT_TOPCONF_DB,
    MetadataSearchEngine,
    SearchCorpus,
    SearchFilters,
    SearchResult,
    reciprocal_rank_fusion,
)
from embodiedai_kb.storage.database import normalize_title


PAPERQA_SRC = ROOT / "third_party/paper-qa/src"
DEFAULT_PDF_CACHE_DIR = ROOT / "data/pdf_cache/literature"
DEFAULT_AGENT_PAPER_DIR = ROOT / "data/paperqa_agent_runs/last"
DEFAULT_RUN_JSON = ROOT / "data/metadata/ask_literature_last.json"
DEFAULT_MEMORY_DIR = ROOT / "data/memory"
USER_AGENT = "EmbodiedAI-KB/0.1 (ask_literature PDF cache)"

EN_STOPWORDS = {
    "a",
    "about",
    "and",
    "are",
    "can",
    "compare",
    "discuss",
    "for",
    "from",
    "give",
    "how",
    "in",
    "is",
    "me",
    "of",
    "on",
    "paper",
    "papers",
    "recent",
    "related",
    "survey",
    "the",
    "to",
    "using",
    "what",
    "with",
}

CN_QUERY_EXPANSIONS = {
    "具身": "embodied AI embodied agent embodied intelligence",
    "智能体": "agent embodied agent",
    "机器人": "robot robotics robot learning",
    "导航": "navigation vision-language navigation VLN embodied navigation",
    "视觉语言导航": "vision-language navigation VLN",
    "语言导航": "vision-language navigation VLN",
    "操作": "robot manipulation mobile manipulation robotic manipulation",
    "操控": "robot manipulation mobile manipulation robotic manipulation",
    "抓取": "robot manipulation grasping",
    "移动操作": "mobile manipulation open-vocabulary mobile manipulation",
    "世界模型": "world model robot world model",
    "空间智能": "spatial intelligence",
    "多智能体": "multi-agent embodied agent collaboration",
    "大模型": "large language model VLM robot foundation model",
    "基础模型": "robot foundation model generalist robot",
    "视觉语言动作": "vision-language-action VLA",
    "安全": "safety benchmark embodied agent",
    "压缩": "compression efficient embodied AI",
    "仿真": "simulation sim-to-real embodied AI",
    "真实": "real robot sim-to-real",
    "基准": "benchmark embodied AI",
    "综述": "survey systematic review",
}


def load_dotenv_files(*paths: Path) -> None:
    for path in paths:
        if not path.exists():
            continue
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


@dataclass(slots=True)
class SelectedPaper:
    result: SearchResult
    selection_score: float
    cache_path: str | None = None
    cache_status: str = "not_downloaded"
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["result"] = self.result.to_dict()
        return data


def parse_args() -> argparse.Namespace:
    load_dotenv_files(ROOT / ".env", ROOT / ".env.local")
    parser = argparse.ArgumentParser(
        description=(
            "End-to-end literature QA demo: plan metadata search queries, retrieve "
            "top candidates, select a small PDF set, cache PDFs, and ask PaperQA."
        )
    )
    parser.add_argument("question", help="Literature question to answer.")
    parser.add_argument("--candidate-k", type=int, default=30)
    parser.add_argument("--paperqa-k", type=int, default=8)
    parser.add_argument("--per-query-k", type=int, default=30)
    parser.add_argument("--include-arxiv", action="store_true")
    parser.add_argument("--include-frontier", action="store_true")
    parser.add_argument("--min-score", type=float, default=4.0)
    parser.add_argument("--year-from", type=int)
    parser.add_argument("--year-to", type=int)
    parser.add_argument(
        "--venues",
        help="Comma-separated venue prefixes, e.g. CVPR,ICLR,AAAI,IJCAI.",
    )
    parser.add_argument(
        "--allow-missing-abstract",
        action="store_true",
        help="Allow metadata records without abstracts in first-stage search.",
    )
    parser.add_argument(
        "--allow-no-pdf",
        action="store_true",
        help="Do not require PDF URLs during metadata search.",
    )
    parser.add_argument(
        "--search-query",
        action="append",
        help="Add an explicit metadata search query. Can be passed multiple times.",
    )
    parser.add_argument("--pdf-cache-dir", type=Path, default=DEFAULT_PDF_CACHE_DIR)
    parser.add_argument("--download-timeout", type=int, default=60)
    parser.add_argument("--download-retries", type=int, default=2)
    parser.add_argument("--request-delay", type=float, default=1.5)
    parser.add_argument("--max-pdf-mb", type=float, default=80.0)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only run query planning, metadata search, and paper selection.",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Run search/selection/download, then stop before PaperQA.",
    )
    parser.add_argument("--run-json", type=Path, default=DEFAULT_RUN_JSON)
    parser.add_argument(
        "--paper-search-log-json",
        type=Path,
        default=None,
        help=(
            "Write a focused JSON debug log for PaperSearchAgent/PaperTriageAgent. "
            "If omitted, the LangGraph runner writes one under data/metadata/paper_search_logs/."
        ),
    )
    parser.add_argument(
        "--thread-id",
        default=os.environ.get("LITERATURE_THREAD_ID", "default"),
        help="Thread/session id for lightweight local JSONL memory.",
    )
    parser.add_argument(
        "--memory-dir",
        type=Path,
        default=Path(os.environ.get("LITERATURE_MEMORY_DIR", DEFAULT_MEMORY_DIR)),
        help="Directory for lightweight thread memory JSONL files.",
    )
    parser.add_argument(
        "--memory-recent-turns",
        type=int,
        default=int(os.environ.get("LITERATURE_MEMORY_RECENT_TURNS", "3")),
        help="How many recent compact memory records to load into graph prompts.",
    )
    parser.add_argument(
        "--no-memory",
        action="store_true",
        help="Disable lightweight thread memory load/write.",
    )
    parser.add_argument(
        "--reset-memory",
        action="store_true",
        help="Delete the current thread memory before this run, then write the new turn.",
    )
    parser.add_argument(
        "--disable-memory-paper-tool",
        action="store_true",
        help=(
            "Disable ExpertResearchAgent's local memory_paper_tool. By default, "
            "the expert first checks whether already remembered PDFs in this "
            "thread can answer the question before running external paper search."
        ),
    )
    parser.add_argument(
        "--memory-paper-candidate-limit",
        type=int,
        default=int(os.environ.get("MEMORY_PAPER_CANDIDATE_LIMIT", "50")),
        help="Maximum remembered paper cards exposed to memory_paper_tool.",
    )
    parser.add_argument(
        "--memory-paper-record-limit",
        type=int,
        default=int(os.environ.get("MEMORY_PAPER_RECORD_LIMIT", "30")),
        help="Number of recent thread JSONL records used to enrich memory paper cards.",
    )
    parser.add_argument(
        "--memory-paper-max-tokens",
        type=int,
        default=int(os.environ.get("MEMORY_PAPER_MAX_TOKENS", "1000")),
        help="Maximum LLM output tokens for memory_paper_tool selection.",
    )
    parser.add_argument("--topconf-db", type=Path, default=DEFAULT_TOPCONF_DB)
    parser.add_argument("--arxiv-db", type=Path, default=DEFAULT_ARXIV_DB)
    parser.add_argument("--frontier-db", type=Path, default=DEFAULT_FRONTIER_DB)
    parser.add_argument(
        "--paperqa-mode",
        choices=["docs", "agent", "fake-agent"],
        default="docs",
        help=(
            "docs manually adds selected PDFs to Docs.aquery; agent/fake-agent "
            "lets PaperQA call its SearchPapers/GatherEvidence/GenerateAnswer tools "
            "over a run-specific paper directory."
        ),
    )
    parser.add_argument("--agent-paper-dir", type=Path, default=DEFAULT_AGENT_PAPER_DIR)
    parser.add_argument("--agent-search-count", type=int, default=8)
    parser.add_argument(
        "--paperqa-search-query-limit",
        type=int,
        default=int(os.environ.get("PAPERQA_SEARCH_QUERY_LIMIT", "12")),
        help=(
            "Maximum local PaperQA paper_search queries. Higher values improve "
            "coverage across selected PDFs at the cost of a little more search time."
        ),
    )
    parser.add_argument(
        "--paperqa-title-query-count",
        type=int,
        default=int(os.environ.get("PAPERQA_TITLE_QUERY_COUNT", "6")),
        help=(
            "Number of selected-paper title queries sent to PaperQA paper_search "
            "before broader ResearchPlanningAgent queries. This helps selected "
            "PDFs enter PaperQA's evidence state."
        ),
    )
    parser.add_argument(
        "--paperqa-reader-mode",
        choices=["paperqa-agent", "explicit-tools", "agent-only"],
        default=os.environ.get("PAPERQA_READER_MODE", "explicit-tools"),
        help=(
            "LangGraph PaperQA reader adapter mode. The default explicit-tools "
            "path runs PaperSearch/GatherEvidence directly for more predictable "
            "evidence extraction. paperqa-agent runs PaperQA's native multi-step "
            "agent first and falls back to explicit tools when it fails or "
            "returns too little evidence."
        ),
    )
    parser.add_argument(
        "--paperqa-answer-mode",
        choices=["answer", "evidence-only"],
        default=os.environ.get("PAPERQA_ANSWER_MODE", "evidence-only"),
        help=(
            "How the LangGraph PaperQA adapter should return PDF-reading output. "
            "'answer' lets PaperQA generate a cited draft answer; 'evidence-only' "
            "uses explicit PaperQA tools to gather evidence contexts and leaves "
            "final answer writing to SynthesisAgent."
        ),
    )
    parser.add_argument(
        "--paperqa-agent-type",
        default=os.environ.get("PAPERQA_AGENT_TYPE"),
        help=(
            "Optional PaperQA native agent type override. Leave unset to use "
            "PaperQA's default ToolSelector agent; use 'fake' only for debugging."
        ),
    )
    parser.add_argument(
        "--paperqa-min-evidence-count",
        type=int,
        default=int(os.environ.get("PAPERQA_MIN_EVIDENCE_COUNT", "4")),
        help="Minimum positive evidence contexts required before skipping fallback.",
    )
    parser.add_argument(
        "--paperqa-min-relevant-papers",
        type=int,
        default=int(os.environ.get("PAPERQA_MIN_RELEVANT_PAPERS", "1")),
        help="Minimum distinct relevant papers required before skipping fallback.",
    )
    parser.add_argument(
        "--paperqa-per-paper-evidence-count",
        type=int,
        default=int(os.environ.get("PAPERQA_PER_PAPER_EVIDENCE_COUNT", "4")),
        help=(
            "Run targeted GatherEvidence passes for the first N selected papers "
            "after the overall evidence pass. Set to 0 to disable; increase to "
            "paperqa-k for stronger coverage at higher LLM cost."
        ),
    )
    parser.add_argument("--agent-max-timesteps", type=int)
    parser.add_argument("--agent-timeout", type=float, default=500.0)
    parser.add_argument(
        "--llm-timeout",
        type=float,
        default=float(os.environ.get("LITERATURE_LLM_TIMEOUT", "180")),
        help="Per-request timeout in seconds for LiteLLM calls.",
    )
    parser.add_argument(
        "--parse-pdf-media",
        action="store_true",
        help=(
            "Let PaperQA parse images/tables from PDFs. Disabled by default for "
            "faster text-only literature QA."
        ),
    )
    parser.add_argument(
        "--allow-paperqa-metadata-lookup",
        action="store_true",
        help=(
            "Allow PaperQA to query Crossref/Semantic Scholar for document metadata. "
            "Disabled by default because this project already supplies metadata."
        ),
    )
    parser.add_argument(
        "--quiet-agent-progress",
        action="store_true",
        help="Suppress PaperQA agent progress messages.",
    )
    parser.add_argument(
        "--quiet-workflow-progress",
        action="store_true",
        help="Suppress live LangGraph/agent progress messages.",
    )
    parser.add_argument(
        "--router-mode",
        choices=["auto", "llm", "heuristic"],
        default=os.environ.get("LITERATURE_ROUTER_MODE", "auto"),
        help=(
            "LangGraph router mode. auto uses an LLM when a router/agent/answer "
            "model is configured, then falls back to heuristic routing."
        ),
    )
    parser.add_argument(
        "--router-llm",
        default=os.environ.get("LITERATURE_ROUTER_LLM"),
        help="Optional LLM name for the LangGraph RouterAgent. Defaults to --agent-llm or --llm.",
    )
    parser.add_argument(
        "--storm-max-perspectives",
        type=int,
        default=int(os.environ.get("STORM_MAX_PERSPECTIVES", "5")),
        help="Number of non-default STORM-style perspectives/personas to generate.",
    )
    parser.add_argument(
        "--storm-search-queries-per-question",
        type=int,
        default=int(os.environ.get("STORM_SEARCH_QUERIES_PER_QUESTION", "2")),
        help="Maximum search queries generated from each perspective-guided question.",
    )
    parser.add_argument(
        "--storm-max-search-queries",
        type=int,
        default=int(os.environ.get("STORM_MAX_SEARCH_QUERIES", "12")),
        help="Maximum merged STORM-style queries passed to metadata search.",
    )
    parser.add_argument(
        "--disable-web-search",
        action="store_true",
        help="Disable the LangGraph WebSearchAgent even when the router requests web context.",
    )
    parser.add_argument(
        "--web-search-provider",
        choices=["auto", "tavily", "duckduckgo"],
        default=os.environ.get("WEB_SEARCH_PROVIDER", "auto"),
        help=(
            "Web search provider for ExpertResearchAgent. auto uses Tavily when "
            "TAVILY_API_KEY is available and falls back to DuckDuckGo HTML."
        ),
    )
    parser.add_argument(
        "--tavily-api-key-env",
        default=os.environ.get("TAVILY_API_KEY_ENV", "TAVILY_API_KEY"),
        help="Environment variable name that stores the Tavily API key.",
    )
    parser.add_argument(
        "--tavily-search-depth",
        choices=["basic", "advanced", "fast", "ultra-fast"],
        default=os.environ.get("TAVILY_SEARCH_DEPTH", "basic"),
        help="Tavily search_depth value passed to the Search API.",
    )
    parser.add_argument(
        "--web-max-queries",
        type=int,
        default=int(os.environ.get("WEB_SEARCH_MAX_QUERIES", "4")),
        help="Maximum planned queries sent to the WebSearchAgent.",
    )
    parser.add_argument(
        "--web-results-per-query",
        type=int,
        default=int(os.environ.get("WEB_SEARCH_RESULTS_PER_QUERY", "4")),
        help="Maximum web results collected for each web search query.",
    )
    parser.add_argument(
        "--web-search-timeout",
        type=float,
        default=float(os.environ.get("WEB_SEARCH_TIMEOUT", "12")),
        help="HTTP timeout in seconds for each web search request.",
    )
    parser.add_argument(
        "--web-search-delay",
        type=float,
        default=float(os.environ.get("WEB_SEARCH_DELAY", "0.4")),
        help="Delay in seconds between web search requests.",
    )
    parser.add_argument(
        "--disable-web-paper-discovery",
        action="store_true",
        help=(
            "Disable discovery of paper PDFs from web results, project pages, "
            "author publication pages, arXiv, OpenReview, and CVF."
        ),
    )
    parser.add_argument(
        "--disable-academic-paper-search",
        action="store_true",
        help=(
            "Disable the multi-source academic paper connector tool. When enabled, "
            "the workflow falls back to local metadata search only."
        ),
    )
    parser.add_argument(
        "--academic-paper-sources",
        default=os.environ.get(
            "ACADEMIC_PAPER_SOURCES",
            "arxiv,openalex,openreview",
        ),
        help=(
            "Comma-separated academic sources for the PaperSearch tool. Supported: "
            "arxiv, openalex, openreview, crossref. Crossref is supported but "
            "not enabled by default because it is best used as a DOI/title fallback."
        ),
    )
    parser.add_argument(
        "--academic-paper-k",
        type=int,
        default=int(os.environ.get("ACADEMIC_PAPER_K", "6")),
        help="Maximum OA/PDF candidates selected by the academic paper search tool.",
    )
    parser.add_argument(
        "--academic-paper-max-queries",
        type=int,
        default=int(os.environ.get("ACADEMIC_PAPER_MAX_QUERIES", "6")),
        help="Maximum planned queries sent to each academic source connector.",
    )
    parser.add_argument(
        "--academic-paper-results-per-source",
        type=int,
        default=int(os.environ.get("ACADEMIC_PAPER_RESULTS_PER_SOURCE", "8")),
        help="Maximum results requested from each source for each query.",
    )
    parser.add_argument(
        "--academic-paper-request-delay",
        type=float,
        default=float(os.environ.get("ACADEMIC_PAPER_REQUEST_DELAY", "1.5")),
        help="Delay used to stagger academic connector requests and reduce rate-limit errors.",
    )
    parser.add_argument(
        "--academic-paper-max-workers",
        type=int,
        default=int(os.environ.get("ACADEMIC_PAPER_MAX_WORKERS", "2")),
        help="Maximum concurrent academic connector requests.",
    )
    parser.add_argument(
        "--academic-paper-min-score",
        type=float,
        default=float(os.environ.get("ACADEMIC_PAPER_MIN_SCORE", "2.5")),
        help="Minimum internal score for academic connector candidates.",
    )
    parser.add_argument(
        "--disable-oa-pdf-resolution",
        action="store_true",
        help="Do not use OpenAlex/Unpaywall DOI fallback to resolve OA PDF URLs.",
    )
    parser.add_argument(
        "--paper-selection-mode",
        choices=["llm", "auto", "score"],
        default=os.environ.get("PAPER_SELECTION_MODE", "llm"),
        help=(
            "How to choose PDFs for PaperQA from retrieved candidates. llm lets "
            "PaperTriageAgent select papers from the candidate list; score uses "
            "the old numeric fallback."
        ),
    )
    parser.add_argument(
        "--paper-triage-llm",
        default=os.environ.get("PAPER_TRIAGE_LLM"),
        help="Optional LLM for PaperTriageAgent. Defaults to --agent-llm, --router-llm, or --llm.",
    )
    parser.add_argument(
        "--paper-triage-candidate-limit",
        type=int,
        default=int(os.environ.get("PAPER_TRIAGE_CANDIDATE_LIMIT", "30")),
        help=(
            "Maximum screened PDF candidates shown to the final PaperTriageAgent selection step. "
            "All retrieved candidates are first scored in compact batches."
        ),
    )
    parser.add_argument(
        "--paper-triage-abstract-max-chars",
        type=int,
        default=int(os.environ.get("PAPER_TRIAGE_ABSTRACT_MAX_CHARS", "400")),
        help=(
            "Maximum abstract characters shown per candidate to PaperTriageAgent. "
            "Set to 0 to pass full abstracts."
        ),
    )
    parser.add_argument(
        "--paper-triage-screen-batch-size",
        type=int,
        default=int(os.environ.get("PAPER_TRIAGE_SCREEN_BATCH_SIZE", "20")),
        help="Number of PDF candidates scored per PaperTriageAgent screening batch.",
    )
    parser.add_argument(
        "--paper-triage-screen-top-n",
        type=int,
        default=int(os.environ.get("PAPER_TRIAGE_SCREEN_TOP_N", "30")),
        help="How many highest-scoring screened candidates are passed to final triage.",
    )
    parser.add_argument(
        "--paper-triage-screen-abstract-max-chars",
        type=int,
        default=int(os.environ.get("PAPER_TRIAGE_SCREEN_ABSTRACT_MAX_CHARS", "350")),
        help="Maximum abstract characters shown per candidate in the batch screening step.",
    )
    parser.add_argument(
        "--paper-search-llm",
        default=os.environ.get("PAPER_SEARCH_LLM"),
        help=(
            "Optional LLM for PaperSearchAgent query analysis and profile-page "
            "publication extraction. Defaults to --agent-llm, --router-llm, or --llm."
        ),
    )
    parser.add_argument(
        "--paper-search-author-paper-limit",
        type=int,
        default=int(os.environ.get("PAPER_SEARCH_AUTHOR_PAPER_LIMIT", "60")),
        help="Maximum papers fetched from each resolved author profile.",
    )
    parser.add_argument(
        "--paper-search-profile-pages",
        type=int,
        default=int(os.environ.get("PAPER_SEARCH_PROFILE_PAGES", "8")),
        help="Maximum official/profile pages fetched by PaperSearchAgent for paper extraction.",
    )
    parser.add_argument(
        "--paper-search-web-queries",
        type=int,
        default=int(os.environ.get("PAPER_SEARCH_WEB_QUERIES", "5")),
        help="Maximum PaperSearchAgent-specific web queries added before profile extraction.",
    )
    parser.add_argument(
        "--paper-search-loop-iterations",
        type=int,
        default=int(os.environ.get("PAPER_SEARCH_LOOP_ITERATIONS", "1")),
        help=(
            "Maximum internal PaperSearchAgent search-loop iterations. "
            "The outer PaperSearchAgent/PaperTriageAgent feedback loop usually "
            "handles refinement, so the default internal loop is one search pass."
        ),
    )
    parser.add_argument(
        "--paper-search-min-pdf-candidates",
        type=int,
        default=int(os.environ.get("PAPER_SEARCH_MIN_PDF_CANDIDATES", "3")),
        help="Minimum PDF-backed online candidates before PaperSearchAgent stops refining.",
    )
    parser.add_argument(
        "--paper-search-min-candidates",
        type=int,
        default=int(os.environ.get("PAPER_SEARCH_MIN_CANDIDATES", "8")),
        help="Minimum total online candidates before PaperSearchAgent stops refining.",
    )
    parser.add_argument(
        "--academic-paper-request-timeout",
        type=float,
        default=float(os.environ.get("ACADEMIC_PAPER_REQUEST_TIMEOUT", "12")),
        help=(
            "Per-request timeout in seconds for academic connectors such as arXiv, "
            "OpenAlex, OpenReview, and Crossref."
        ),
    )
    parser.add_argument(
        "--academic-paper-request-retries",
        type=int,
        default=int(os.environ.get("ACADEMIC_PAPER_REQUEST_RETRIES", "0")),
        help=(
            "Retry count for each academic connector HTTP request. Keep this low for "
            "interactive web demos so one rate-limited source cannot block the run."
        ),
    )
    parser.add_argument(
        "--academic-paper-search-timeout",
        type=float,
        default=float(os.environ.get("ACADEMIC_PAPER_SEARCH_TIMEOUT", "75")),
        help=(
            "Overall timeout in seconds for the academic connector batch inside "
            "PaperSearchAgent."
        ),
    )
    parser.add_argument(
        "--disable-paper-search-loop",
        action="store_true",
        help="Disable iterative PaperSearchAgent refinement and run only the initial search plan.",
    )
    parser.add_argument(
        "--disable-paper-search-reflection",
        action="store_true",
        help=(
            "Disable the PaSa-lite LLM reflection step inside PaperSearchAgent. "
            "When disabled, loop refinement falls back to simple candidate-count checks."
        ),
    )
    parser.add_argument(
        "--paper-search-triage-rounds",
        type=int,
        default=int(os.environ.get("PAPER_SEARCH_TRIAGE_ROUNDS", "2")),
        help=(
            "Maximum outer rounds of PaperSearchAgent -> PaperTriageAgent feedback. "
            "If triage selects too few PDFs, its rationale is fed back to PaperSearchAgent."
        ),
    )
    parser.add_argument(
        "--paper-search-triage-min-selected",
        type=int,
        default=int(os.environ.get("PAPER_SEARCH_TRIAGE_MIN_SELECTED", "1")),
        help=(
            "Minimum number of PDFs PaperTriageAgent must select before the "
            "PaperSearchAgent/Triage feedback loop stops."
        ),
    )
    parser.add_argument(
        "--disable-metadata-paper-search",
        action="store_true",
        help="Disable local metadata search inside PaperSearchAgent.",
    )
    parser.add_argument(
        "--enable-heuristic-web-paper-discovery",
        action="store_true",
        help=(
            "Deprecated compatibility flag. The old heuristic webpage parser is no "
            "longer used in the main LangGraph workflow; PaperSearchAgent handles "
            "paper discovery."
        ),
    )
    parser.add_argument(
        "--web-paper-discovery-k",
        type=int,
        default=int(os.environ.get("WEB_PAPER_DISCOVERY_K", "6")),
        help="Maximum web-discovered PDF candidates to prefer before metadata candidates.",
    )
    parser.add_argument(
        "--web-paper-discovery-max-queries",
        type=int,
        default=int(os.environ.get("WEB_PAPER_DISCOVERY_MAX_QUERIES", "4")),
        help="Maximum site-specific web queries used to find extra paper/PDF pages.",
    )
    parser.add_argument(
        "--web-paper-discovery-results-per-query",
        type=int,
        default=int(os.environ.get("WEB_PAPER_DISCOVERY_RESULTS_PER_QUERY", "5")),
        help="Maximum results per site-specific paper discovery query.",
    )
    parser.add_argument(
        "--web-paper-discovery-max-pages",
        type=int,
        default=int(os.environ.get("WEB_PAPER_DISCOVERY_MAX_PAGES", "8")),
        help="Maximum web pages fetched to extract PDF/arXiv/OpenReview links.",
    )
    parser.add_argument(
        "--web-paper-discovery-min-score",
        type=float,
        default=float(os.environ.get("WEB_PAPER_DISCOVERY_MIN_SCORE", "30")),
        help=(
            "Minimum internal relevance score for web-discovered PDFs. Higher "
            "values avoid random arXiv hits for author/team-specific questions."
        ),
    )
    parser.add_argument(
        "--llm",
        default=os.environ.get("PAPERQA_LLM"),
        help="PaperQA answer LLM. Defaults to PaperQA's own setting.",
    )
    parser.add_argument(
        "--synthesis-llm",
        default=os.environ.get("LITERATURE_SYNTHESIS_LLM"),
        help="Optional LLM for the final LangGraph SynthesisAgent. Defaults to --llm.",
    )
    parser.add_argument(
        "--synthesis-max-tokens",
        type=int,
        default=int(os.environ.get("LITERATURE_SYNTHESIS_MAX_TOKENS", "4096")),
        help=(
            "Maximum tokens for the final LangGraph SynthesisAgent answer. "
            "Increase this when long reports are cut off."
        ),
    )
    parser.add_argument(
        "--episode-summary-max-tokens",
        type=int,
        default=int(os.environ.get("LITERATURE_EPISODE_SUMMARY_MAX_TOKENS", "700")),
        help=(
            "Maximum tokens for the SynthesisAgent memory episode summary. "
            "This is a short post-answer summary used by thread memory indexes."
        ),
    )
    parser.add_argument(
        "--memory-llm",
        default=os.environ.get("LITERATURE_MEMORY_LLM"),
        help=(
            "Optional LLM for MemoryRecallAgent. Defaults to --router-llm or --llm. "
            "It only selects memory ids from indexes before Planner runs."
        ),
    )
    parser.add_argument(
        "--memory-recall-max-tokens",
        type=int,
        default=int(os.environ.get("LITERATURE_MEMORY_RECALL_MAX_TOKENS", "900")),
        help="Maximum output tokens for MemoryRecallAgent index selection.",
    )
    parser.add_argument(
        "--memory-detail-max-chars",
        type=int,
        default=int(os.environ.get("LITERATURE_MEMORY_DETAIL_MAX_CHARS", "9000")),
        help="Maximum characters of detailed memory blocks recalled into state.",
    )
    parser.add_argument(
        "--summary-llm",
        default=os.environ.get("PAPERQA_SUMMARY_LLM"),
        help="PaperQA evidence-summary LLM. Defaults to --llm or PaperQA default.",
    )
    parser.add_argument(
        "--embedding",
        default=os.environ.get("PAPERQA_EMBEDDING"),
        help="PaperQA embedding model. Defaults to PaperQA's own setting.",
    )
    parser.add_argument(
        "--agent-llm",
        default=os.environ.get("PAPERQA_AGENT_LLM"),
        help="PaperQA agent LLM. Defaults to PaperQA's own setting.",
    )
    parser.add_argument(
        "--openai-base-url",
        default=(
            os.environ.get("OPENAI_BASE_URL")
            or os.environ.get("OPENAI_API_BASE")
            or os.environ.get("DEEPSEEK_BASE_URL")
        ),
        help=(
            "OpenAI-compatible API base URL passed to LiteLLM as api_base. "
            "Defaults to OPENAI_BASE_URL, OPENAI_API_BASE, or DEEPSEEK_BASE_URL."
        ),
    )
    parser.add_argument(
        "--openai-api-key-env",
        default="OPENAI_API_KEY",
        help="Environment variable that stores the OpenAI-compatible API key.",
    )
    parser.add_argument(
        "--embedding-base-url",
        default=os.environ.get("OPENAI_EMBEDDING_BASE_URL")
        or os.environ.get("EMBEDDING_BASE_URL"),
        help=(
            "Optional OpenAI-compatible base URL for the embedding model. "
            "If omitted, embeddings use PaperQA/LiteLLM defaults."
        ),
    )
    parser.add_argument(
        "--embedding-api-key-env",
        default=os.environ.get("OPENAI_EMBEDDING_API_KEY_ENV", "OPENAI_API_KEY"),
        help="Environment variable that stores the embedding API key.",
    )
    parser.add_argument(
        "--disable-openai-compatible-config",
        action="store_true",
        help="Do not inject OPENAI_BASE_URL/OPENAI_API_KEY into PaperQA LiteLLM configs.",
    )
    parser.add_argument("--answer-max-sources", type=int, default=6)
    parser.add_argument("--evidence-k", type=int, default=12)
    parser.add_argument("--answer-length", default="about 500 words")
    return parser.parse_args()


def parse_venues(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(item.strip() for item in value.split(",") if item.strip())


def english_keywords(question: str, limit: int = 12) -> list[str]:
    tokens: list[str] = []
    seen: set[str] = set()
    for token in re.findall(r"[A-Za-z][A-Za-z0-9-]+", question):
        lowered = token.lower()
        if len(lowered) < 3 or lowered in EN_STOPWORDS or lowered in seen:
            continue
        seen.add(lowered)
        tokens.append(token)
        if len(tokens) >= limit:
            break
    return tokens


def chinese_expansions(question: str) -> list[str]:
    expansions = [
        expansion
        for phrase, expansion in CN_QUERY_EXPANSIONS.items()
        if phrase in question
    ]
    return expansions


def build_search_queries(question: str, explicit_queries: list[str] | None) -> list[str]:
    queries: list[str] = []
    seen: set[str] = set()

    def add(query: str) -> None:
        clean = re.sub(r"\s+", " ", query).strip()
        key = clean.lower()
        if clean and key not in seen:
            seen.add(key)
            queries.append(clean)

    for query in explicit_queries or []:
        add(query)

    expansions = chinese_expansions(question)
    keywords = english_keywords(question)
    if keywords:
        add(" ".join(keywords))
    for expansion in expansions[:4]:
        add(expansion)
    if expansions:
        add(" ".join(expansions[:3]))
    add(question)

    if not queries:
        add("embodied AI robot learning vision-language-action")
    return queries[:6]


def select_for_paperqa(results: list[SearchResult], top_k: int) -> list[SelectedPaper]:
    selected: list[SelectedPaper] = []
    for result in results:
        if not result.pdf_url:
            continue
        score = result.hybrid_score
        score += result.relevance_score * 0.12
        score += result.frontier_score * 0.18
        score += min(result.citation_count or 0, 50) * 0.025
        score += min(result.influential_citation_count or 0, 10) * 0.08
        score += 0.6 if result.corpus == "topconf" else 0.0
        score += 0.35 if result.corpus == "frontier" else 0.0
        score += 0.25 if result.year and result.year >= 2024 else 0.0
        score += 0.2 if result.code_url or result.project_url else 0.0
        selected.append(SelectedPaper(result=result, selection_score=round(score, 4)))

    selected.sort(key=lambda item: item.selection_score, reverse=True)
    diverse: list[SelectedPaper] = []
    venue_counts: dict[str, int] = {}
    title_seen: set[str] = set()
    for item in selected:
        title_key = normalize_title(item.result.title)
        if title_key in title_seen:
            continue
        venue = item.result.venue or item.result.corpus
        if venue_counts.get(venue, 0) >= 3 and len(diverse) >= max(3, top_k // 2):
            continue
        diverse.append(item)
        title_seen.add(title_key)
        venue_counts[venue] = venue_counts.get(venue, 0) + 1
        if len(diverse) >= top_k:
            break
    return diverse


def safe_pdf_name(result: SearchResult) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", result.paper_id).strip("_")
    if not stem:
        stem = normalize_title(result.title).replace(" ", "_")[:80]
    return f"{stem}.pdf"


def fetch_pdf(
    url: str,
    path: Path,
    timeout: int,
    max_pdf_mb: float,
    retries: int,
) -> str:
    if path.exists() and path.stat().st_size > 1024:
        return "cache_hit"

    last_error: Exception | None = None
    for attempt in range(max(retries, 0) + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = response.read()
                if response.headers.get("Content-Encoding", "").lower() == "gzip":
                    payload = gzip.decompress(payload)
                if len(payload) > max_pdf_mb * 1024 * 1024:
                    raise ValueError(
                        f"PDF too large: {len(payload) / 1024 / 1024:.1f} MB"
                    )
                if not payload.startswith(b"%PDF"):
                    content_type = response.headers.get("Content-Type", "")
                    raise ValueError(f"download did not return a PDF: {content_type}")
            break
        except (OSError, TimeoutError, urllib.error.URLError, ValueError) as exc:
            last_error = exc
            if attempt >= retries:
                raise
            time.sleep(1.5 * (attempt + 1))
    else:
        raise RuntimeError(f"PDF download failed: {last_error}")

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_bytes(payload)
    tmp_path.replace(path)
    return "downloaded"


def cache_selected_pdfs(
    selected: list[SelectedPaper],
    cache_dir: Path,
    timeout: int,
    request_delay: float,
    max_pdf_mb: float,
    retries: int,
) -> None:
    last_request = 0.0
    for item in selected:
        result = item.result
        if not result.pdf_url:
            item.cache_status = "missing_pdf_url"
            continue
        elapsed = time.monotonic() - last_request
        if elapsed < request_delay:
            time.sleep(request_delay - elapsed)
        last_request = time.monotonic()
        path = cache_dir / safe_pdf_name(result)
        try:
            item.cache_status = fetch_pdf(
                result.pdf_url,
                path=path,
                timeout=timeout,
                max_pdf_mb=max_pdf_mb,
                retries=retries,
            )
            item.cache_path = str(path)
        except (OSError, TimeoutError, urllib.error.URLError, ValueError) as exc:
            item.cache_status = "failed"
            item.error = str(exc)


def citation_for(result: SearchResult) -> str:
    if result.authors:
        first = result.authors[0].split()[-1]
        author_text = f"{first} et al." if len(result.authors) > 1 else result.authors[0]
    else:
        author_text = "Unknown"
    year = result.year or "n.d."
    return f"{author_text}, {year}, {result.title}"


def print_search_summary(
    queries: list[str],
    candidates: list[SearchResult],
    selected: list[SelectedPaper],
) -> None:
    print("Search queries:")
    for idx, query in enumerate(queries, start=1):
        print(f"  {idx}. {query}")
    print()
    print(f"Metadata candidates: {len(candidates)}")
    print(f"Selected for PaperQA: {len(selected)}")
    print()
    header = f"{'#':>2} {'sel':>6} {'year':>4} {'corpus':<9} {'venue':<12} {'cache':<12} title"
    print(header)
    print("-" * len(header))
    for idx, item in enumerate(selected, start=1):
        result = item.result
        title = shorten(result.title, width=88, placeholder="...")
        venue = shorten(result.venue or result.corpus, width=12, placeholder="..")
        print(
            f"{idx:>2} {item.selection_score:>6.2f} "
            f"{str(result.year or '-'):>4} {result.corpus:<9} {venue:<12} {item.cache_status:<12} {title}"
        )
        print(
            f"   id={result.paper_id} rank={result.rank} "
            f"metadata={result.hybrid_score:.2f} rel={result.relevance_score:.2f} "
            f"frontier={result.frontier_score:.2f} cites={result.citation_count or 0}"
        )
        if result.quality_signals:
            print(f"   signals={', '.join(result.quality_signals[:6])}")
        if item.error:
            print(f"   error={item.error}")


def write_run_json(
    path: Path,
    question: str,
    queries: list[str],
    candidates: list[SearchResult],
    selected: list[SelectedPaper],
    answer: str | None = None,
    paperqa_trace: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "question": question,
        "queries": queries,
        "candidate_count": len(candidates),
        "selected_count": len(selected),
        "candidates": [result.to_dict() for result in candidates],
        "selected": [item.to_dict() for item in selected],
        "answer": answer,
        "paperqa_trace": paperqa_trace or {},
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return str(value)


def prepare_agent_paper_dir(
    selected: list[SelectedPaper],
    paper_dir: Path,
) -> list[dict[str, Any]]:
    if paper_dir.exists():
        shutil.rmtree(paper_dir)
    paper_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    for idx, item in enumerate(selected, start=1):
        if item.cache_status not in {"cache_hit", "downloaded"} or not item.cache_path:
            continue
        source = Path(item.cache_path).resolve()
        title_slug = re.sub(r"[^a-zA-Z0-9]+", "_", item.result.title).strip("_")[:90]
        destination = paper_dir / f"{idx:02d}_{title_slug or safe_pdf_name(item.result)}.pdf"
        try:
            destination.symlink_to(source)
            link_type = "symlink"
        except OSError:
            shutil.copy2(source, destination)
            link_type = "copy"
        records.append(
            {
                "title": item.result.title,
                "source": str(source),
                "path": str(destination),
                "link_type": link_type,
                "corpus": item.result.corpus,
                "selection_score": item.selection_score,
            }
        )
        manifest_rows.append(
            {
                "file_location": str(destination.absolute()),
                "title": item.result.title,
                "doi": item.result.doi or "",
                "citation": citation_for(item.result),
                "authors": repr(item.result.authors or []),
                "year": item.result.year or "",
                "journal": item.result.venue or item.result.corpus,
                "pdf_url": item.result.pdf_url or "",
                "url": item.result.paper_url or "",
            }
        )
    if manifest_rows:
        manifest_path = paper_dir / "manifest.csv"
        with manifest_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "file_location",
                    "title",
                    "doi",
                    "citation",
                    "authors",
                    "year",
                    "journal",
                    "pdf_url",
                    "url",
                ],
            )
            writer.writeheader()
            writer.writerows(manifest_rows)
    return records


def _tool_call_arguments(call: Any) -> dict[str, Any]:
    function = getattr(call, "function", None)
    arguments = getattr(function, "arguments", None)
    if arguments is None:
        arguments = getattr(call, "arguments", None)
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {"arguments": shorten(arguments, width=120, placeholder="...")}
    return arguments if isinstance(arguments, dict) else {}


def _tool_call_name(call: Any) -> str:
    function = getattr(call, "function", None)
    name = getattr(function, "name", None) or getattr(call, "name", None)
    return str(name or type(call).__name__)


def summarize_tool_request(action: Any) -> str:
    calls = getattr(action, "tool_calls", None) or []
    parts: list[str] = []
    for call in calls:
        name = _tool_call_name(call)
        arguments = _tool_call_arguments(call)
        if name == "paper_search":
            query = shorten(str(arguments.get("query", "")), width=80, placeholder="...")
            parts.append(
                f"paper_search(query={query!r}, min_year={arguments.get('min_year')}, max_year={arguments.get('max_year')})"
            )
        elif name == "gather_evidence":
            question = shorten(str(arguments.get("question", "")), width=80, placeholder="...")
            parts.append(f"gather_evidence(question={question!r})")
        elif name == "gen_answer":
            parts.append("gen_answer()")
        elif name == "complete":
            parts.append("complete()")
        else:
            arg_preview = shorten(json.dumps(arguments, ensure_ascii=False), width=90, placeholder="...")
            parts.append(f"{name}({arg_preview})")
    return ", ".join(parts) if parts else type(action).__name__


def state_status(state: Any) -> str:
    status = getattr(state, "status", "")
    if status:
        return str(status)
    messages = getattr(state, "messages", None)
    tools = getattr(state, "tools", None)
    if messages is not None and tools is not None:
        return f"agent ledger: messages={len(messages)} tools={len(tools)}"
    return type(state).__name__


def print_agent_progress(message: str, args: argparse.Namespace) -> None:
    if not args.quiet_agent_progress:
        print(f"[PaperQA] {message}", flush=True)


def _model_has_provider_prefix(model: str) -> bool:
    return "/" in model and not model.startswith(("http://", "https://"))


def _is_special_embedding_model(model: str) -> bool:
    return model == "sparse" or model.startswith(("hybrid-", "st-"))


def normalize_openai_compatible_model(
    model: str,
    args: argparse.Namespace,
    base_url: str | None = None,
) -> str:
    """Make LiteLLM route unknown bare model names through the OpenAI provider."""

    model = model.strip()
    if (
        not model
        or args.disable_openai_compatible_config
        or not (base_url or args.openai_base_url)
        or _model_has_provider_prefix(model)
    ):
        return model
    return f"openai/{model}"


def openai_compatible_chat_config(
    model: str,
    args: argparse.Namespace,
    temperature: float,
) -> tuple[str, dict[str, Any] | None]:
    if args.disable_openai_compatible_config or not args.openai_base_url:
        return model, None

    normalized_model = normalize_openai_compatible_model(model, args, args.openai_base_url)
    litellm_params: dict[str, Any] = {
        "model": normalized_model,
        "api_base": args.openai_base_url,
        "temperature": temperature,
        "cache_control_injection_points": [
            {"location": "message", "role": "system"},
        ],
    }
    api_key = openai_compatible_api_key(args, args.openai_base_url, args.openai_api_key_env)
    if api_key:
        litellm_params["api_key"] = api_key
    return normalized_model, {
        "name": normalized_model,
        "model_list": [
            {
                "model_name": normalized_model,
                "litellm_params": litellm_params,
            }
        ],
    }


def openai_compatible_embedding_config(args: argparse.Namespace) -> dict[str, Any] | None:
    if (
        args.disable_openai_compatible_config
        or not args.embedding_base_url
        or (args.embedding and _is_special_embedding_model(args.embedding))
    ):
        return None
    kwargs: dict[str, Any] = {"api_base": args.embedding_base_url}
    api_key = openai_compatible_api_key(
        args,
        args.embedding_base_url,
        args.embedding_api_key_env,
    )
    if api_key:
        kwargs["api_key"] = api_key
    return {"kwargs": kwargs}


def openai_compatible_api_key(
    args: argparse.Namespace,
    base_url: str | None,
    api_key_env: str,
) -> str | None:
    api_key = os.getenv(api_key_env)
    if api_key:
        return api_key
    if base_url and "deepseek.com" in base_url:
        return os.getenv("DEEPSEEK_API_KEY")
    return None


def validate_openai_compatible_config(args: argparse.Namespace) -> None:
    if args.disable_openai_compatible_config:
        return
    for label, value in (
        ("--openai-base-url / OPENAI_BASE_URL", args.openai_base_url),
        ("--embedding-base-url / EMBEDDING_BASE_URL", args.embedding_base_url),
    ):
        if not value:
            continue
        parsed = urlparse(value)
        if value.startswith(("sk-", "sk_", "sk-")) or not parsed.scheme or not parsed.netloc:
            raise ValueError(
                f"{label} must be an HTTP(S) API base URL, but got a value that "
                "does not look like a URL. Did you put an API key in the base URL "
                "variable by mistake?"
            )


def redact_url(value: str | None) -> str:
    if not value:
        return "None"
    parsed = urlparse(value)
    if parsed.scheme and parsed.netloc:
        return f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
    return "<invalid-url>"


def configure_openai_compatible_settings(settings: Any, args: argparse.Namespace) -> None:
    if args.disable_openai_compatible_config or not args.openai_base_url:
        return

    settings.llm, settings.llm_config = openai_compatible_chat_config(
        settings.llm,
        args,
        settings.temperature,
    )
    settings.summary_llm, settings.summary_llm_config = openai_compatible_chat_config(
        settings.summary_llm,
        args,
        settings.temperature,
    )
    settings.agent.agent_llm, settings.agent.agent_llm_config = openai_compatible_chat_config(
        settings.agent.agent_llm,
        args,
        settings.temperature,
    )
    (
        settings.parsing.enrichment_llm,
        settings.parsing.enrichment_llm_config,
    ) = openai_compatible_chat_config(
        settings.parsing.enrichment_llm,
        args,
        settings.temperature,
    )
    if args.embedding:
        settings.embedding = args.embedding
    if args.embedding_base_url and not _is_special_embedding_model(settings.embedding):
        settings.embedding = normalize_openai_compatible_model(
            settings.embedding,
            args,
            args.embedding_base_url,
        )
    embedding_config = openai_compatible_embedding_config(args)
    if embedding_config:
        settings.embedding_config = embedding_config


async def run_paperqa(
    question: str,
    selected: list[SelectedPaper],
    args: argparse.Namespace,
) -> tuple[str, dict[str, Any]]:
    if str(PAPERQA_SRC) not in sys.path:
        sys.path.insert(0, str(PAPERQA_SRC))
    try:
        from paperqa import Docs, Settings, agent_query
    except ImportError as exc:
        raise RuntimeError(
            "PaperQA dependencies are not installed. Install them with: "
            "python -m pip install -e third_party/paper-qa"
        ) from exc

    docs = Docs()
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
    if args.openai_base_url and not args.disable_openai_compatible_config:
        print_agent_progress(
            (
                "using OpenAI-compatible LiteLLM config "
                f"api_base={redact_url(args.openai_base_url)!r}, "
                f"llm={settings.llm!r}, summary_llm={settings.summary_llm!r}, "
                f"agent_llm={settings.agent.agent_llm!r}, "
                f"enrichment_llm={settings.parsing.enrichment_llm!r}, "
                f"embedding={settings.embedding!r}"
            ),
            args,
        )
    settings.answer.answer_max_sources = args.answer_max_sources
    settings.answer.evidence_k = args.evidence_k
    settings.answer.answer_length = args.answer_length
    settings.agent.search_count = args.agent_search_count
    settings.agent.timeout = args.agent_timeout
    settings.agent.return_paper_metadata = True
    if args.agent_max_timesteps is not None:
        settings.agent.max_timesteps = args.agent_max_timesteps

    if args.paperqa_mode in {"agent", "fake-agent"}:
        paper_records = prepare_agent_paper_dir(selected, args.agent_paper_dir)
        if not paper_records:
            raise RuntimeError("No PDFs were available for PaperQA agent after download/cache.")

        print_agent_progress(
            f"prepared {len(paper_records)} PDFs in {args.agent_paper_dir}",
            args,
        )
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

        async def on_gather_started(state: Any) -> None:
            print_agent_progress(f"gather_evidence started | {state_status(state)}", args)

        async def on_gather_completed(state: Any) -> None:
            print_agent_progress(f"gather_evidence completed | {state_status(state)}", args)

        async def on_answer_started(state: Any) -> None:
            print_agent_progress(f"gen_answer started | {state_status(state)}", args)

        async def on_answer_completed(state: Any) -> None:
            print_agent_progress(f"gen_answer completed | {state_status(state)}", args)

        settings.agent.callbacks.setdefault("gather_evidence_initialized", []).append(
            on_gather_started
        )
        settings.agent.callbacks.setdefault("gather_evidence_completed", []).append(
            on_gather_completed
        )
        settings.agent.callbacks.setdefault("gen_answer_initialized", []).append(
            on_answer_started
        )
        settings.agent.callbacks.setdefault("gen_answer_completed", []).append(
            on_answer_completed
        )
        action_trace: list[dict[str, Any]] = []

        async def on_agent_action_callback(action: Any, state: Any) -> None:
            print_agent_progress(
                f"calling {summarize_tool_request(action)} | {state_status(state)}",
                args,
            )
            action_trace.append(
                {
                    "action_type": type(action).__name__,
                    "action": _jsonable(action),
                    "state_type": type(state).__name__,
                }
            )

        async def on_env_step_callback(
            obs: list[Any],
            reward: float,
            done: bool,
            truncated: bool,
        ) -> None:
            previews = []
            for item in obs[:2]:
                content = getattr(item, "content", "")
                if content:
                    previews.append(shorten(str(content).replace("\n", " "), width=180, placeholder="..."))
            if previews:
                print_agent_progress(
                    f"tool response: {' | '.join(previews)}",
                    args,
                )
            print_agent_progress(
                f"step done={done} truncated={truncated} reward={reward:g}",
                args,
            )

        agent_type = "fake" if args.paperqa_mode == "fake-agent" else settings.agent.agent_type
        response = await agent_query(
            query=question,
            settings=settings,
            docs=Docs(),
            agent_type=agent_type,
            on_agent_action_callback=on_agent_action_callback,
            on_env_step_callback=on_env_step_callback,
        )
        session = response.session
        trace = {
            "mode": args.paperqa_mode,
            "agent_type": agent_type,
            "agent_paper_dir": str(args.agent_paper_dir),
            "agent_papers": paper_records,
            "agent_actions": action_trace,
            "status": str(response.status),
        }
        return str(getattr(session, "answer", session)), trace

    added = 0
    for item in selected:
        if item.cache_status not in {"cache_hit", "downloaded"} or not item.cache_path:
            continue
        result = item.result
        path = Path(item.cache_path)
        await docs.aadd(
            path,
            citation=citation_for(result),
            docname=normalize_title(result.title).replace(" ", "_")[:80],
            title=result.title,
            doi=result.doi,
            authors=result.authors,
            settings=settings,
        )
        added += 1

    if added == 0:
        raise RuntimeError("No PDFs were available for PaperQA after download/cache.")

    session = await docs.aquery(question, settings=settings)
    trace = {
        "mode": "docs",
        "added_pdfs": added,
    }
    return str(getattr(session, "answer", session)), trace


def run_metadata_search(args: argparse.Namespace) -> tuple[list[str], list[SearchResult], list[SelectedPaper]]:
    queries = build_search_queries(args.question, args.search_query)
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
    return queries, candidates, selected


async def async_main() -> None:
    args = parse_args()
    queries, candidates, selected = run_metadata_search(args)

    if not args.dry_run:
        cache_selected_pdfs(
            selected,
            cache_dir=args.pdf_cache_dir,
            timeout=args.download_timeout,
            request_delay=args.request_delay,
            max_pdf_mb=args.max_pdf_mb,
            retries=args.download_retries,
        )

    print_search_summary(queries, candidates, selected)

    answer: str | None = None
    paperqa_trace: dict[str, Any] | None = None
    if not args.dry_run and not args.download_only:
        print("\nRunning PaperQA...\n")
        answer, paperqa_trace = await run_paperqa(args.question, selected, args)
        print(answer)

    write_run_json(
        args.run_json,
        args.question,
        queries,
        candidates,
        selected,
        answer,
        paperqa_trace,
    )
    print(f"\nRun record: {args.run_json}")


def main() -> None:
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except RuntimeError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
