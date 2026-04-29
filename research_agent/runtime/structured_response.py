from __future__ import annotations

from typing import Any

from research_agent.state import ResearchState


def _confidence_from_state(state: ResearchState, eval_summary: dict[str, Any] | None) -> tuple[float, str]:
    if eval_summary and isinstance(eval_summary.get("overall_score"), (int, float)):
        score = max(0.0, min(1.0, float(eval_summary["overall_score"])))
    else:
        score = 0.25
        if len(state.get("found_papers", [])) >= 6:
            score += 0.2
        if len(state.get("paper_summaries", [])) >= 4:
            score += 0.2
        if state.get("draft_md_path") or state.get("draft_latex_path"):
            score += 0.15
        if float(state.get("critique_score", 0.0) or 0.0) >= 7.0:
            score += 0.15
        if state.get("error_log"):
            score -= 0.15
        score = max(0.0, min(1.0, score))

    if score >= 0.75:
        return score, "high"
    if score >= 0.5:
        return score, "medium"
    return score, "low"


def build_structured_response(
    state: ResearchState,
    eval_summary: dict[str, Any] | None = None,
) -> ResearchState:
    confidence_score, label = _confidence_from_state(state, eval_summary)

    lines: list[str] = []
    question = state.get("selected_question", "") or state.get("raw_input", "")
    plan = state.get("brain_plan", []) or []
    budget = state.get("run_budget", {}) or {}

    lines.append("## Task")
    lines.append(question or "N/A")
    lines.append("")

    lines.append("## Plan")
    lines.append(" -> ".join(plan) if plan else "N/A")
    lines.append("")

    lines.append("## Execution")
    lines.append(f"- Executor goal: {state.get('executor_goal', '') or question or 'N/A'}")
    lines.append(f"- Steps used: {budget.get('used_steps', 0)} / {budget.get('max_steps', 0)}")
    lines.append(f"- Errors: {budget.get('used_errors', 0)} / {budget.get('max_errors', 0)}")
    if state.get("stop_reason"):
        lines.append(f"- Stop reason: {state.get('stop_reason')}")
    if state.get("analysis_brief"):
        lines.append(f"- Analyzer brief: {state.get('analysis_brief')}")
    lines.append("")

    lines.append("## Evidence")
    lines.append(f"- Papers found: {len(state.get('found_papers', []))}")
    lines.append(f"- Papers read: {len(state.get('paper_summaries', []))}")
    if state.get("github_repo_url"):
        lines.append(f"- GitHub repo: {state.get('github_repo_url')}")
    top_titles = [p.get("title", "") for p in (state.get("found_papers", []) or [])[:5] if p.get("title")]
    if top_titles:
        lines.append("- Top sources:")
        for title in top_titles:
            lines.append(f"  - {title}")
    lines.append("")

    # ── Quality Gate ──────────────────────────────────────────────────────────
    lines.append("## Quality Gate")
    critic_score = float(state.get("critique_score", 0.0) or 0.0)
    threshold = float(state.get("critique_accept_threshold", 7.0) or 7.0)
    gate_passed = bool(state.get("critic_gate_passed", False))
    repair_rounds = int(state.get("evidence_repair_rounds", 0) or 0)
    filtered = int(state.get("relevance_filtered_count", 0) or 0)

    lines.append(f"- Critic score: {critic_score:.1f} / {threshold:.1f}  ({'PASS ✓' if gate_passed else 'FAIL ✗'})")
    if filtered:
        lines.append(f"- Relevance filter: {filtered} off-topic paper(s) removed before analysis")
    if repair_rounds:
        lines.append(f"- Evidence repair: {repair_rounds} cycle(s) triggered")
    else:
        lines.append("- Evidence repair: not triggered")
    if not gate_passed:
        replan = str(state.get("replan_reason", "") or "").strip()
        lines.append(
            f"- Gate result: FAILED — {'repair budget exhausted' if repair_rounds else 'no repair configured'}. "
            + (f"Reason: {replan[:140]}" if replan else "Draft produced with reduced confidence.")
        )
    else:
        lines.append("- Gate result: PASSED — draft reflects verified evidence")

    if state.get("critique"):
        lines.append("- Critic guidance (summary):")
        for row in str(state.get("critique", "")).splitlines()[:6]:
            row = row.strip()
            if row:
                lines.append(f"  {row}")
    lines.append("")

    lines.append("## Deliverables")
    deliverables = []
    if state.get("draft_md_path"):
        deliverables.append(("Markdown", state.get("draft_md_path")))
    if state.get("draft_latex_path"):
        deliverables.append(("LaTeX", state.get("draft_latex_path")))
    for artifact in state.get("artifacts", []) or []:
        ref = artifact.get("content_ref", "")
        title = artifact.get("title", "artifact")
        if ref:
            deliverables.append((title, ref))
    if deliverables:
        seen = set()
        for title, ref in deliverables:
            key = (title, ref)
            if key in seen:
                continue
            seen.add(key)
            lines.append(f"- {title}: {ref}")
    else:
        lines.append("- No materialized deliverables")
    lines.append("")

    lines.append("## Confidence")
    lines.append(f"- Label: {label}")
    lines.append(f"- Score: {confidence_score:.0%}")
    if eval_summary and eval_summary.get("summary"):
        lines.append(f"- Evaluator summary: {eval_summary.get('summary')}")

    if state.get("error_log"):
        lines.append("")
        lines.append("## Errors")
        seen_errors = []
        for item in state.get("error_log", [])[:6]:
            msg = str(item).split("\n")[0][:180]
            if msg not in seen_errors:
                seen_errors.append(msg)
        for item in seen_errors:
            lines.append(f"- {item}")

    return {
        **state,
        "assistant_response": "\n".join(lines).strip(),
        "confidence_score": confidence_score,
        "confidence_label": label,  # type: ignore[typeddict-item]
    }
