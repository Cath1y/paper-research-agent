from __future__ import annotations

from typing import Any, TypedDict


class LiteratureGraphState(TypedDict, total=False):
    """Shared state carried through the LangGraph literature workflow.

    In LangGraph, every node receives the current state and returns a partial
    state update. The graph runtime merges each returned dict into this shared
    state before moving to the next node.
    """

    # User input.
    question: str

    # Lightweight thread memory. The current version loads/saves local JSONL
    # records for demos and future multi-turn expansion.
    memory_records: list[dict[str, Any]]
    memory_packet: dict[str, Any]
    memory_context: str
    memory_details: str
    memory_load_trace: dict[str, Any]
    memory_recall_trace: dict[str, Any]
    memory_write_trace: dict[str, Any]

    # RouterAgent output. It decides which downstream capabilities should run.
    route: dict[str, Any]
    plan: dict[str, Any]
    completed_steps: list[str]
    current_step: str | None
    current_agent: str | None
    next_node: str | None
    supervisor_iterations: int

    # ResearchPlanningAgent output. The first field is the canonical plan; the
    # next fields are compatibility views used by older printers/prompts.
    research_plan: list[dict[str, Any]]
    perspectives: list[dict[str, Any]]
    research_questions: list[dict[str, Any]]
    planning_web_evidence: list[dict[str, Any]]
    planning_web_trace: dict[str, Any]
    planning_trace: dict[str, Any]
    perspective_trace: dict[str, Any]

    # Outputs from the metadata search part of the graph.
    queries: list[str]
    web_evidence: list[dict[str, Any]]
    web_search_trace: dict[str, Any]
    web_findings: str
    memory_paper_candidates: list[dict[str, Any]]
    memory_paper_trace: dict[str, Any]
    academic_paper_candidates: list[dict[str, Any]]
    academic_paper_search_trace: dict[str, Any]
    paper_search_trace: dict[str, Any]
    web_paper_candidates: list[dict[str, Any]]
    web_paper_discovery_trace: dict[str, Any]
    metadata_candidates: list[dict[str, Any]]
    paper_candidate_debug: dict[str, Any]
    candidates: list[dict[str, Any]]
    selected: list[dict[str, Any]]
    expert_trace: list[dict[str, Any]]
    paper_triage_trace: dict[str, Any]

    # Outputs from the PaperQA tools reader and final synthesis nodes.
    paperqa_answer: str
    final_answer: str
    episode_summary: str
    paperqa_trace: dict[str, Any]
    synthesis_trace: dict[str, Any]

    # Human-readable trace of graph node execution for demos/debugging.
    trace: list[dict[str, Any]]
