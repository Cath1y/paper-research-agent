from __future__ import annotations

from argparse import Namespace
from typing import Any

from langgraph.graph import END, START, StateGraph

from .expert_research import run_expert_research_agent
from .memory import append_thread_memory, load_thread_memory
from .memory_recall import run_memory_recall_agent
from .perspective import plan_research
from .progress import emit_progress
from .router import route_question
from .state import LiteratureGraphState
from .synthesis import run_synthesis_agent


def _trace_step(name: str, **payload: Any) -> dict[str, Any]:
    return {"node": name, **payload}


def _step_id(step: dict[str, Any]) -> str:
    return str(step.get("id") or "").strip()


def _step_agent(step: dict[str, Any]) -> str:
    return str(step.get("agent") or "").strip()


def _step_to_node(step: dict[str, Any]) -> str:
    """Map planner-level agents onto the worker nodes currently implemented."""

    agent = _step_agent(step)
    if agent == "ResearchPlanningAgent":
        return "research_planning"
    if agent == "ExpertResearchAgent":
        return "expert_research"
    return "synthesize_answer"


def _step_is_done(
    dep: str,
    *,
    completed: set[str],
    steps: list[dict[str, Any]],
) -> bool:
    """Check dependencies by step id, worker node name, or agent name."""

    if dep in completed:
        return True
    for step in steps:
        step_id = _step_id(step)
        if step_id not in completed:
            continue
        if dep == _step_to_node(step) or dep == _step_agent(step):
            return True
    return False


def _next_planned_step(
    steps: list[dict[str, Any]],
    completed_steps: list[str],
) -> dict[str, Any] | None:
    """Return the next pending step whose dependencies are already satisfied."""

    completed = set(completed_steps)
    for step in steps:
        step_id = _step_id(step)
        if not step_id or step_id in completed:
            continue
        depends_on = [
            str(dep).strip()
            for dep in step.get("depends_on", [])
            if str(dep).strip()
        ]
        if all(
            _step_is_done(dep, completed=completed, steps=steps)
            for dep in depends_on
        ):
            return step
    return None


def _mark_current_step_done(
    state: LiteratureGraphState,
    *,
    fallback_step: str,
) -> list[str]:
    completed = list(state.get("completed_steps", []))
    step_id = str(state.get("current_step") or fallback_step).strip()
    if step_id and step_id not in completed:
        completed.append(step_id)
    return completed


def _plan_with_status(
    plan: dict[str, Any] | None,
    *,
    completed_steps: list[str],
    current_step: str | None,
) -> dict[str, Any]:
    """Return a copy of the planner output annotated with live step status."""

    if not plan:
        return {"steps": [], "next_step": current_step}
    completed = set(completed_steps)
    steps: list[dict[str, Any]] = []
    for step in plan.get("steps", []):
        step_copy = dict(step)
        step_id = _step_id(step_copy)
        if step_id in completed:
            status = "completed"
        elif step_id and step_id == current_step:
            status = "in_progress"
        else:
            status = "pending"
        step_copy["status"] = status
        steps.append(step_copy)
    return {
        **plan,
        "steps": steps,
        "next_step": current_step,
    }


def build_literature_graph(args: Namespace):
    """Build the LangGraph literature assistant workflow.

    The outer graph is now a minimal dynamic Supervisor loop:

        Planner/Supervisor -> selected worker -> Planner/Supervisor -> ...

    The initial planner still produces a dependency-aware plan. After each
    worker updates the shared state, the supervisor re-checks completed steps
    and dispatches the next executable agent. Internal tool decisions still live
    inside ExpertResearchAgent.
    """

    async def load_memory(state: LiteratureGraphState) -> dict[str, Any]:
        # MemoryReader：从本地 JSONL 读取最近几轮摘要，压缩成短上下文放入
        # shared state。Planner / ResearchPlanning / Synthesis 会用它解析追问，
        # 但它不是论文证据，也不让 agent 任意访问文件系统。
        emit_progress(args, "MemoryReader", "start", thread_id=getattr(args, "thread_id", "default"))
        records, context, memory_trace, memory_packet = load_thread_memory(args)
        emit_progress(
            args,
            "MemoryReader",
            "done",
            loaded=memory_trace.get("loaded_count"),
            used=memory_trace.get("used_count", 0),
            papers=(memory_trace.get("packet") or {}).get("recent_selected_paper_count"),
            path=memory_trace.get("path"),
        )
        return {
            "memory_records": records,
            "memory_context": context,
            "memory_packet": memory_packet,
            "memory_load_trace": memory_trace,
            "trace": [
                *state.get("trace", []),
                _trace_step(
                    "load_memory",
                    mode=memory_trace.get("mode"),
                    thread_id=memory_trace.get("thread_id"),
                    loaded_count=memory_trace.get("loaded_count"),
                    used_count=memory_trace.get("used_count", 0),
                    reset=memory_trace.get("reset"),
                ),
            ],
        }

    async def planner_supervisor(state: LiteratureGraphState) -> dict[str, Any]:
        # 节点 0：Planner/SupervisorAgent。
        # 第一轮先作为 memory tool user 查看轻量索引并按需召回详情，
        # 再调用 router LLM/heuristic 产出 plan；之后每一轮都基于 shared
        # state 中的 completed_steps 选择下一个可执行 worker。
        trace = list(state.get("trace", []))
        route = state.get("route")
        plan = state.get("plan")
        completed_steps = list(state.get("completed_steps", []))
        iterations = int(state.get("supervisor_iterations", 0)) + 1

        updates: dict[str, Any] = {"supervisor_iterations": iterations}
        if not route or not plan:
            memory_context = state.get("memory_context", "")
            memory_details = state.get("memory_details", "")
            recall_trace = state.get("memory_recall_trace") or {}
            if not recall_trace:
                emit_progress(
                    args,
                    "PlannerSupervisor",
                    "memory recall start",
                    iteration=iterations,
                )
                memory_details, recall_trace = await run_memory_recall_agent(
                    question=state["question"],
                    memory_packet=state.get("memory_packet", {}),
                    args=args,
                )
                if memory_details:
                    memory_context = (
                        f"{memory_context}\n\n"
                        "Planner memory tool decision:\n"
                        f"- memory_sufficient_to_answer={recall_trace.get('memory_sufficient_to_answer')}\n"
                        f"- reason={recall_trace.get('reason')}\n\n"
                        "Detailed memory recalled by Planner memory tool "
                        "(context only, not fresh evidence):\n"
                        f"{memory_details}"
                    ).strip()
                trace.append(
                    _trace_step(
                        "planner_memory_recall",
                        mode=recall_trace.get("mode"),
                        need_memory_detail=recall_trace.get("need_memory_detail"),
                        memory_sufficient_to_answer=recall_trace.get(
                            "memory_sufficient_to_answer"
                        ),
                        episode_ids=recall_trace.get("episode_ids"),
                        paper_ids=recall_trace.get("paper_ids"),
                        card_ids=recall_trace.get("card_ids"),
                        detail_chars=len(memory_details or ""),
                        reason=recall_trace.get("reason"),
                    )
                )
                updates.update(
                    {
                        "memory_details": memory_details,
                        "memory_context": memory_context,
                        "memory_recall_trace": recall_trace,
                    }
                )
                emit_progress(
                    args,
                    "PlannerSupervisor",
                    "memory recall done",
                    need=recall_trace.get("need_memory_detail"),
                    details=len(memory_details or ""),
                    sufficient=recall_trace.get("memory_sufficient_to_answer"),
                )
            emit_progress(args, "PlannerSupervisor", "routing start", iteration=iterations)
            decision = await route_question(
                state["question"],
                args,
                memory_context=memory_context,
            )
            route = decision.to_dict()
            plan = {
                "workflow_goal": route.get("workflow_goal"),
                "steps": route.get("steps", []),
                "next_step": route.get("next_step"),
            }
            completed_steps = []
            trace.append(
                _trace_step(
                    "route_request",
                    task_type=decision.task_type,
                    task_types=decision.task_types,
                    step_count=len(decision.steps),
                    next_step=decision.next_step,
                    need_web_search=decision.need_web_search,
                    need_paper_search=decision.need_paper_search,
                    need_pdf_reading=decision.need_pdf_reading,
                    need_learning_plan=decision.need_learning_plan,
                    need_idea_generation=decision.need_idea_generation,
                    source=decision.source,
                    confidence=decision.confidence,
                )
            )
            updates.update(
                {
                    "route": route,
                    "completed_steps": completed_steps,
                }
            )
            emit_progress(
                args,
                "PlannerSupervisor",
                "routing done",
                task=decision.task_type,
                steps=len(decision.steps),
                source=decision.source,
            )

        steps = list((plan or {}).get("steps", []))
        max_iterations = max(len(steps) + 4, 6)
        if iterations > max_iterations:
            trace.append(
                _trace_step(
                    "planner_supervisor",
                    action="force_synthesis",
                    reason="max_iterations_reached",
                    completed_steps=completed_steps,
                    max_iterations=max_iterations,
                )
            )
            next_node = "synthesize_answer"
            current_step = "synthesize_final_answer"
            current_agent = "SynthesisAgent"
        else:
            next_step = _next_planned_step(steps, completed_steps)
            if next_step is None:
                if state.get("final_answer"):
                    next_node = "end"
                    current_step = None
                    current_agent = None
                else:
                    next_node = "synthesize_answer"
                    current_step = "synthesize_final_answer"
                    current_agent = "SynthesisAgent"
            else:
                current_step = _step_id(next_step)
                current_agent = _step_agent(next_step)
                next_node = _step_to_node(next_step)

        plan = _plan_with_status(
            plan,
            completed_steps=completed_steps,
            current_step=current_step,
        )
        trace.append(
            _trace_step(
                "planner_supervisor",
                action="dispatch",
                current_step=current_step,
                current_agent=current_agent,
                next_node=next_node,
                completed_steps=completed_steps,
                iteration=iterations,
            )
        )
        emit_progress(
            args,
            "PlannerSupervisor",
            "dispatch",
            next_node=next_node,
            current_agent=current_agent,
            completed=len(completed_steps),
        )
        updates.update(
            {
                "plan": plan,
                "current_step": current_step,
                "current_agent": current_agent,
                "next_node": next_node,
                "trace": trace,
            }
        )
        return updates

    async def research_planning(state: LiteratureGraphState) -> dict[str, Any]:
        # ResearchPlanningAgent：生成多角度 research_plan，并为每个角度产出
        # research question 和 queries。完成后回到 Supervisor，而不是固定进入
        # ExpertResearchAgent。
        emit_progress(args, "ResearchPlanningAgent", "start")
        research_plan = await plan_research(
            state["question"],
            args,
            memory_context=state.get("memory_context", ""),
        )
        queries = research_plan["queries"]
        completed_steps = _mark_current_step_done(
            state,
            fallback_step="research_planning",
        )
        emit_progress(
            args,
            "ResearchPlanningAgent",
            "done",
            perspectives=len(research_plan["perspectives"]),
            questions=len(research_plan["research_questions"]),
            queries=len(queries),
            source=research_plan["planning_trace"].get("source"),
        )
        return {
            "queries": queries,
            "research_plan": research_plan["research_plan"],
            "perspectives": research_plan["perspectives"],
            "research_questions": research_plan["research_questions"],
            "planning_web_evidence": research_plan.get("planning_web_evidence", []),
            "planning_web_trace": research_plan.get("planning_web_trace", {}),
            "planning_trace": research_plan["planning_trace"],
            "perspective_trace": research_plan["perspective_trace"],
            "completed_steps": completed_steps,
            "current_step": None,
            "current_agent": None,
            "next_node": None,
            "trace": [
                *state.get("trace", []),
                _trace_step(
                    "research_planning",
                    completed_plan_step=completed_steps[-1] if completed_steps else None,
                    perspective_count=len(research_plan["perspectives"]),
                    research_question_count=len(research_plan["research_questions"]),
                    query_count=len(queries),
                    source=research_plan["planning_trace"].get("source"),
                    web_scout_result_count=research_plan["planning_trace"].get(
                        "web_scout_result_count"
                    ),
                    queries=queries,
                ),
            ],
        }

    async def expert_research(state: LiteratureGraphState) -> dict[str, Any]:
        # ExpertResearchAgent：基于已有 research questions/queries 调用
        # web_search_tool、metadata_search_tool、pdf_cache_tool 和 PaperQA tools。
        emit_progress(
            args,
            "ExpertResearchAgent",
            "start",
            query_count=len(state.get("queries", [])),
            plan_items=len(state.get("research_plan", [])),
        )
        result = await run_expert_research_agent(
            question=state["question"],
            queries=state.get("queries", []),
            research_plan=state.get("research_plan", []),
            research_questions=state.get("research_questions", []),
            perspectives=state.get("perspectives", []),
            planning_web_evidence=state.get("planning_web_evidence", []),
            route=state.get("route", {}),
            args=args,
        )
        expert_trace = result.get("expert_trace", [])
        completed_steps = _mark_current_step_done(
            state,
            fallback_step="expert_research",
        )
        emit_progress(
            args,
            "ExpertResearchAgent",
            "done",
            candidates=len(result.get("candidates", [])),
            selected=len(result.get("selected", [])),
            web_results=len(result.get("web_evidence", [])),
            paperqa=bool(result.get("paperqa_answer")),
        )
        return {
            **result,
            "completed_steps": completed_steps,
            "current_step": None,
            "current_agent": None,
            "next_node": None,
            "trace": [
                *state.get("trace", []),
                _trace_step(
                    "expert_research",
                    completed_plan_step=completed_steps[-1] if completed_steps else None,
                    tool_count=len(expert_trace),
                    candidate_count=len(result.get("candidates", [])),
                    web_paper_candidate_count=len(
                        result.get("web_paper_candidates", [])
                    ),
                    selected_count=len(result.get("selected", [])),
                    web_result_count=len(result.get("web_evidence", [])),
                    has_paperqa_answer=bool(result.get("paperqa_answer")),
                ),
            ],
        }

    async def synthesize_answer(state: LiteratureGraphState) -> dict[str, Any]:
        # SynthesisAgent：读取整个 shared state，综合 PaperQA 和 web evidence。
        emit_progress(args, "SynthesisAgent", "start")
        final_answer, synthesis_trace = await run_synthesis_agent(dict(state), args)
        completed_steps = _mark_current_step_done(
            state,
            fallback_step="synthesize_final_answer",
        )
        plan = _plan_with_status(
            state.get("plan", {}),
            completed_steps=completed_steps,
            current_step=None,
        )
        emit_progress(
            args,
            "SynthesisAgent",
            "done",
            mode=synthesis_trace.get("mode"),
            chars=len(final_answer or ""),
        )
        return {
            "final_answer": final_answer,
            "episode_summary": synthesis_trace.get("episode_summary", ""),
            "synthesis_trace": synthesis_trace,
            "plan": plan,
            "completed_steps": completed_steps,
            "current_step": None,
            "current_agent": None,
            "next_node": None,
            "trace": [
                *state.get("trace", []),
                _trace_step(
                    "synthesize_answer",
                    completed_plan_step=completed_steps[-1] if completed_steps else None,
                    strategy=synthesis_trace.get("mode"),
                    model=synthesis_trace.get("model"),
                    error=synthesis_trace.get("error"),
                ),
            ],
        }

    async def write_memory(state: LiteratureGraphState) -> dict[str, Any]:
        # MemoryWriter：在单轮任务结束后，把本轮高层摘要追加到 thread
        # JSONL 文件。写入内容比 prompt 里的 memory_context 更完整，方便
        # debug；下一轮读取时会再压缩。
        emit_progress(args, "MemoryWriter", "start", thread_id=getattr(args, "thread_id", "default"))
        memory_trace = append_thread_memory(dict(state), args)
        emit_progress(
            args,
            "MemoryWriter",
            "done",
            written=memory_trace.get("written"),
            path=memory_trace.get("path"),
        )
        return {
            "memory_write_trace": memory_trace,
            "trace": [
                *state.get("trace", []),
                _trace_step(
                    "write_memory",
                    mode=memory_trace.get("mode"),
                    thread_id=memory_trace.get("thread_id"),
                    written=memory_trace.get("written"),
                    selected_paper_count=memory_trace.get("selected_paper_count"),
                ),
            ],
        }

    def route_after_supervisor(state: LiteratureGraphState) -> str:
        return state.get("next_node") or "synthesize_answer"

    graph = StateGraph(LiteratureGraphState)
    graph.add_node("load_memory", load_memory)
    graph.add_node("planner_supervisor", planner_supervisor)
    graph.add_node("research_planning", research_planning)
    graph.add_node("expert_research", expert_research)
    graph.add_node("synthesize_answer", synthesize_answer)
    graph.add_node("write_memory", write_memory)

    graph.add_edge(START, "load_memory")
    graph.add_edge("load_memory", "planner_supervisor")
    graph.add_conditional_edges(
        "planner_supervisor",
        route_after_supervisor,
        {
            "research_planning": "research_planning",
            "expert_research": "expert_research",
            "synthesize_answer": "synthesize_answer",
            "end": END,
        },
    )
    graph.add_edge("research_planning", "planner_supervisor")
    graph.add_edge("expert_research", "planner_supervisor")
    graph.add_edge("synthesize_answer", "write_memory")
    graph.add_edge("write_memory", END)

    return graph.compile()
