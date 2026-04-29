from __future__ import annotations

from research_agent.memory.conversation_memory import ConversationMemory
from research_agent.protocols import persist_evolution_report
from research_agent.react import run_react_pipeline
from research_agent.state import ResearchState


def run_pipeline(
    raw_input: str,
    human_in_loop: bool = False,
    hitl_stages: list[str] | None = None,
    engine: str = "react",
    max_steps: int = 12,
    max_errors: int = 4,
    executor_rollouts: int = 1,
    executor_parallelism: int = 1,
    session_id: str = "default",
    input_paths: list[str] | None = None,
    local_context: str = "",
    target_tex_path: str = "",
    output_format: str = "auto",
    writer_rules_path: str = "",
    frontdesk_decision: dict | None = None,
    clarification_callback=None,
) -> ResearchState:
    frontdesk_decision = frontdesk_decision or {}
    initial_state: ResearchState = {
        "topic_mode": "B",
        "raw_input": raw_input,
        "research_questions": [],
        "selected_question": "",
        "session_id": session_id,
        "task_profile": "general_research",
        "user_constraints": {},
        "context_package": {},
        "input_paths": input_paths or [],
        "local_context": local_context,
        "target_tex_path": target_tex_path,
        "output_format": output_format,
        "writer_rules_path": writer_rules_path,
        "paper_store_dir": f"./paper_index/{session_id}",
        "frontdesk_intent": str(frontdesk_decision.get("intent", "")),
        "frontdesk_confidence": float(frontdesk_decision.get("confidence", 0.0) or 0.0),
        "frontdesk_next_action": str(frontdesk_decision.get("next_action", "")),
        "search_queries": [],
        "found_papers": [],
        "confirmed_papers": [],
        "paper_summaries": [],
        "key_paper": "",
        "reasoning_notes": "",
        "math_derivations": [],
        "github_repo_url": "",
        "reproduced_code": "",
        "improvement_plan": "",
        "improved_code": "",
        "experiment_results": {},
        "doc_type": "survey",
        "draft": {},
        "raw_draft": {},
        "draft_md_path": "",
        "draft_latex_path": "",
        "critique": "",
        "critique_score": 0.0,
        "critique_accepted": False,
        "critique_accept_threshold": 7.0,
        "revision_count": 0,
        "human_feedback": "",
        "human_in_loop": human_in_loop,
        "hitl_stages": hitl_stages or [],
        "current_stage": "",
        "next": "",
        "brain_plan": [],
        "brain_plan_index": 0,
        "error_log": [],
        "messages": [],
        "engine": "react",
        "tasks": [],
        "artifacts": [],
        "evidence_ids": [],
        "action_history": [],
        "analysis_brief": "",
        "analysis_key_points": [],
        "analysis_open_risks": [],
        "analysis_writer_focus": "",
        "analysis_critic_focus": "",
        "selection_reason": "",
        "executor_goal": "",
        "executor_rollouts": max(1, executor_rollouts),
        "executor_parallelism": max(1, executor_parallelism),
        "selected_rollout_id": 0,
        "rollout_summaries": [],
        "executor_trace": [],
        "last_observation": "",
        "last_reflection": "",
        "stop_reason": "",
        "run_budget": {
            "max_steps": max_steps,
            "max_errors": max_errors,
            "used_steps": 0,
            "used_errors": 0,
        },
        "replan_reason": "",
        "replan_next": "",
        "execute_actions": [],
        "assistant_response": "",
        "confidence_score": 0.0,
        "confidence_label": "low",
        "quality_objectives": {},
        "evidence_quality": {},
        "optimized_queries": [],
        "optimization_round": 0,
        "evidence_repair_rounds": 0,
        "critic_gate_passed": False,
        "relevance_filtered_count": 0,
        "analyzer_supervisor_round": 0,
        "analyzer_supervisor_max_rounds": 2,
        "writer_head_round": 0,
        "writer_head_max_rounds": 2,
        "writer_revision_brief": "",
        "writer_revision_suggestions": [],
        "research_brief": {},
        "research_brief_confidence": 0.0,
        "brief_ready": False,
        "brief_missing_fields": [],
          "agent_capabilities": {},
          "analyzer_trace": [],
          "analyzer_stop_reason": "",
    }

    final_state = run_react_pipeline(initial_state, clarification_callback=clarification_callback)
    try:
        persist_evolution_report(final_state)
    except Exception:
        pass
    _persist_session_memory(final_state)
    return final_state


def _persist_session_memory(state: ResearchState):
    """将本轮关键信息写入会话记忆（长记忆层）。"""
    session_id = state.get("session_id", "default")
    mem = ConversationMemory(session_id)

    q = state.get("selected_question", "")
    if q:
        mem.pin_goal(q)
        mem.add_fact(f"latest_topic={q[:180]}")
    if state.get("paper_store_dir"):
        mem.add_fact(f"latest_paper_store_dir={state.get('paper_store_dir')}")

    plan = state.get("brain_plan", [])
    if plan:
        mem.add_decision(f"executed_plan={'->'.join(plan)}")
    stop_reason = (state.get("stop_reason") or "").strip()
    if stop_reason:
        mem.add_decision(f"executor_stop_reason={stop_reason[:240]}")

    if state.get("draft_md_path"):
        mem.add_fact(f"latest_draft={state.get('draft_md_path')}")
        mem.add_turn("assistant", f"已产出草稿：{state.get('draft_md_path')}")
    if state.get("assistant_response"):
        mem.add_turn("assistant", state.get("assistant_response", "")[:1200])

    constraints = state.get("user_constraints", {}) or {}
    pinned = []
    if constraints.get("single_paragraph"):
        pinned.append("prefer single paragraph output")
    if constraints.get("target_words"):
        pinned.append(f"target words around {constraints.get('target_words')}")
    if pinned:
        mem.set_constraints(pinned)

    # Build rolling_summary so Brain has compressed context on next run.
    # rolling_summary is already passed through build_context_package → Brain,
    # but was never written — this fills that gap.
    summary_parts: list[str] = []
    if q:
        summary_parts.append(f"上次主题：{q[:120]}")
    if plan:
        summary_parts.append(f"执行计划：{'→'.join(plan)}")
    papers_found = len(state.get("found_papers", []) or [])
    papers_read = len(state.get("paper_summaries", []) or [])
    if papers_found:
        summary_parts.append(f"检索论文 {papers_found} 篇，精读 {papers_read} 篇")
    key_points = list(state.get("analysis_key_points", []) or [])[:3]
    if key_points:
        summary_parts.append(f"核心要点：{'; '.join(str(p)[:60] for p in key_points)}")
    draft = state.get("draft", {}) or {}
    abstract = str(draft.get("abstract", "") or "").strip()
    if abstract:
        summary_parts.append(f"草稿摘要：{abstract[:200]}")
    elif state.get("draft_md_path"):
        summary_parts.append(f"草稿路径：{state.get('draft_md_path')}")
    score = float(state.get("critique_score", 0.0) or 0.0)
    if score > 0:
        accepted = bool(state.get("critique_accepted", False))
        summary_parts.append(f"Critic 评分：{score:.1f}（{'通过' if accepted else '未通过'}）")
    if summary_parts:
        mem.data["rolling_summary"] = "\n".join(summary_parts)

    mem.save()
