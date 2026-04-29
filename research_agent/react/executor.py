"""
ReAct pipeline — simplified 3-stage architecture.

New pipeline (feature/supervisor-parallel-researchers):
    Brain → Analyzer Supervisor → Writer ↔ Critic

Key changes vs old pipeline:
  - Executor TAOR loop removed: no more rollouts, reflects, or repair cycles
  - Analyzer Supervisor is now the main ReAct agent (with conduct_research)
  - Analyzer spawns parallel SubResearchers via conduct_research tool
  - Critic prewrite gate kept; Writer↔Critic loop unchanged
"""
from __future__ import annotations

import os

from research_agent.agents.analyzer_agent import analyzer_node
from research_agent.agents.brain_agent import brain_node
from research_agent.agents.critic_agent import critic_node
from research_agent.agents.writer_agent import writer_node
from research_agent.observability.trace_logger import init_logger, get_logger
from research_agent.state import ResearchState

_SEP = "─" * 60


def _phase(name: str, detail: str = "") -> None:
    suffix = f"  {detail}" if detail else ""
    print(f"\n{_SEP}")
    print(f"▶ [{name}]{suffix}")


def _phase_done(name: str, detail: str = "") -> None:
    suffix = f"  {detail}" if detail else ""
    print(f"✓ [{name}] done{suffix}")


_CLARIFY_CONFIDENCE_THRESHOLD = 0.80


def run_react_pipeline(
    initial_state: ResearchState,
    clarification_callback=None,
) -> ResearchState:
    """
    Brain → Analyzer Supervisor → Writer ↔ Critic

    Brain:    minimal planner — produces research_brief + plan
    Analyzer: supervisor ReAct loop with parallel sub-researchers
    Writer:   drafts the document (up to writer_head_max_rounds)
    Critic:   reviews and accepts or requests revision

    clarification_callback: optional callable(questions: str, state: dict) -> str | None
      Called when Brain confidence < _CLARIFY_CONFIDENCE_THRESHOLD.
      Return value:
        None  → user declined; abort pipeline
        ""    → user confirmed defaults; proceed as-is
        <str> → user supplement; re-run Brain with merged input
    """
    # ── 0. Observability: init logger ─────────────────────────────────────────
    session_id = str(initial_state.get("session_id", "default") or "default")
    logger = init_logger(session_id)

    # ── 1. Brain ──────────────────────────────────────────────────────────────
    _phase("Brain")
    with logger.span("brain", "plan"):
        state = brain_node(initial_state)
    _phase_done("Brain", f"plan={state.get('brain_plan', [])}")
    logger.event("brain", "done",
                 plan=state.get("brain_plan", []),
                 task_profile=state.get("task_profile", ""),
                 sub_questions=len(state.get("research_brief", {}).get("sub_questions", [])))

    if not state.get("brain_plan"):
        # Conversational / clarification response — no further processing needed
        return state

    # ── 1b. Human clarification gate ─────────────────────────────────────────
    brief_confidence = float(state.get("research_brief_confidence", 1.0) or 1.0)
    if clarification_callback is not None and brief_confidence < _CLARIFY_CONFIDENCE_THRESHOLD:
        questions = (state.get("assistant_response") or "").strip()
        user_reply = clarification_callback(questions, state)
        if user_reply is None:
            # User declined — return Brain state with empty plan so CLI treats as abort
            print("  [Brain] user declined clarification — aborting pipeline")
            return {**state, "brain_plan": []}
        if user_reply:
            # Merge user supplement into raw_input and re-run Brain once
            merged_input = str(state.get("raw_input", "") or "") + "\n\n[用户补充] " + user_reply
            print(f"  [Brain] re-running with user supplement: {user_reply[:80]}")
            _phase("Brain (re-plan)")
            state = brain_node({**state, "raw_input": merged_input})
            _phase_done("Brain (re-plan)", f"plan={state.get('brain_plan', [])}")
            if not state.get("brain_plan"):
                return state
        # user_reply == "" → proceed with current plan as-is

    constraints = state.get("user_constraints", {}) or {}
    economy_mode = bool(
        constraints.get("economy_mode", False)
        or os.getenv("RESEARCH_AGENT_ECONOMY_MODE", "").strip().lower() in {"1", "true", "yes", "on"}
    )
    critic_gate = bool(constraints.get("critic_gate_before_write", False))  # temporarily disabled
    writer_head_max_rounds = max(
        1,
        int(
            constraints.get("writer_head_max_rounds", 2 if not economy_mode else 1)
            or (1 if economy_mode else 2)
        ),
    )

    # ── 2. Analyzer Supervisor (ReAct + parallel sub-researchers) ─────────────
    plan = list(state.get("brain_plan", []))
    # Any of these plan keywords triggers the Analyzer Supervisor
    _RESEARCH_TRIGGERS = {"research", "search", "literature_review", "read", "evidence_evaluate", "github"}
    if any(k in plan for k in _RESEARCH_TRIGGERS):
        _phase("Analyzer Supervisor", "supervisor ReAct loop + parallel sub-researchers")
        with logger.span("analyzer", "supervisor_loop"):
            try:
                state = analyzer_node(state)
            except Exception as e:
                state = {**state, "error_log": list(state.get("error_log", []) or []) + [f"[analyzer] {e}"]}
                print(f"  [Analyzer] exception: {e}")

        found = len(state.get("found_papers", []) or [])
        read = len(state.get("paper_summaries", []) or [])
        evidence_count = len(state.get("evidence", []) or [])
        _phase_done("Analyzer", f"found={found} read={read} stop={state.get('analyzer_stop_reason', '?')}")
        logger.event("analyzer", "done",
                     found=found, read=read,
                     evidence_cards=evidence_count,
                     stop=state.get("analyzer_stop_reason", ""))

    # ── 3. Writer ↔ Critic iterative output head ─────────────────────────────
    allow_write_without_papers = bool(constraints.get("allow_write_without_papers", False))
    has_content = bool(
        state.get("paper_summaries")
        or state.get("assistant_response")
        or state.get("analysis_brief")
        or allow_write_without_papers
    )

    if "write" in plan and has_content:
        state = {
            **state,
            "writer_head_round": 0,
            "critique_accepted": False,
            "writer_revision_brief": "",
            "writer_revision_suggestions": [],
        }

        for round_id in range(1, writer_head_max_rounds + 1):
            state = {**state, "writer_head_round": round_id}
            _phase(f"Writer (round {round_id}/{writer_head_max_rounds})")
            try:
                state = writer_node(state)
            except Exception as e:
                state = {**state, "error_log": list(state.get("error_log", []) or []) + [f"[write] {e}"]}
                print(f"  [Writer] exception: {e}")
                break
            _phase_done(f"Writer#{round_id}")

            if not critic_gate:
                state = {**state, "critique_accepted": True, "critic_gate_passed": True}
                break

            _phase(f"Critic (round {round_id}/{writer_head_max_rounds})")
            try:
                state = critic_node(state)
            except Exception as e:
                state = {**state, "error_log": list(state.get("error_log", []) or []) + [f"[critic] {e}"]}
                print(f"  [Critic] exception: {e}")
                break

            score = float(state.get("critique_score", 0.0) or 0.0)
            threshold = float(state.get("critique_accept_threshold", 7.0) or 7.0)
            accepted = bool(state.get("critique_accepted", False))
            _phase_done(f"Critic#{round_id}", f"score={score:.1f}/{threshold:.1f} accepted={accepted}")
            if accepted:
                break

        state = {**state, "critic_gate_passed": bool(state.get("critique_accepted", False))}

    # ── Final observability flush ─────────────────────────────────────────────
    logger.flush_summary(
        papers_found=len(state.get("found_papers", []) or []),
        papers_read=len(state.get("paper_summaries", []) or []),
        evidence_cards=len(state.get("evidence", []) or []),
        critic_accepted=bool(state.get("critique_accepted", False)),
    )
    tokens = logger.token_summary()
    print(f"\n[Pipeline] Token usage — "
          f"prompt={tokens['prompt_tokens']:,} "
          f"completion={tokens['completion_tokens']:,} "
          f"total={tokens['total_tokens']:,} "
          f"calls={tokens['call_count']}")

    return state
