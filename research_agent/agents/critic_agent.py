from __future__ import annotations

import json
from typing import Any

from research_agent.llm.router import call_llm_json, system
from research_agent.prompts import render_prompt
from research_agent.state import ResearchState
from research_agent.utils.json_parser import parse_json


def _critic_threshold(state: ResearchState) -> float:
    constraints = state.get("user_constraints", {}) or {}
    profile = state.get("task_profile", "general_research")
    threshold = 7.0
    if profile in {"literature_review", "paper_from_implementation", "math_reasoning"}:
        threshold = 7.3
    if constraints.get("fast_mode"):
        threshold = max(6.6, threshold - 0.4)
    if constraints.get("need_critic"):
        threshold = max(threshold, 7.4)
    return round(threshold, 1)


def _sections_text(state: ResearchState) -> str:
    draft = state.get("draft", {}) or {}
    ordered_sections = [
        ("abstract", draft.get("abstract", "")),
        ("intro", draft.get("intro", "")),
        ("related_work", draft.get("related_work", "")),
        ("method", draft.get("method", "")),
        ("experiments", draft.get("experiments", "")),
        ("conclusion", draft.get("conclusion", "")),
    ]
    chunks: list[str] = []
    for name, text in ordered_sections:
        content = str(text or "").strip()
        if not content:
            continue
        chunks.append(f"## {name}\n{content[:1400]}")
    return "\n\n".join(chunks)


def _build_input_json(state: ResearchState, accept_threshold: float) -> str:
    payload = {
        "question": state.get("selected_question", "") or state.get("raw_input", ""),
        "task_profile": state.get("task_profile", "general_research"),
        "doc_type": state.get("doc_type", "report"),
        "selected_rollout_id": state.get("selected_rollout_id", 0),
        "analysis_brief": state.get("analysis_brief", ""),
        "analysis_key_points": state.get("analysis_key_points", []),
        "analysis_open_risks": state.get("analysis_open_risks", []),
        "papers_found": len(state.get("found_papers", [])),
        "papers_read": len(state.get("paper_summaries", [])),
        "repo_url": state.get("github_repo_url", ""),
        "sections_text": _sections_text(state),
        "artifacts": [
            {
                "title": item.get("title", ""),
                "path": item.get("content_ref", ""),
            }
            for item in (state.get("artifacts", []) or [])[:8]
        ],
        "stop_reason": state.get("stop_reason", ""),
        "accept_threshold": accept_threshold,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _fallback_result(state: ResearchState, accept_threshold: float, error: Exception | None = None) -> dict[str, Any]:
    draft = state.get("draft", {}) or {}
    non_empty_sections = sum(1 for key in ["abstract", "intro", "related_work", "method", "experiments", "conclusion"] if str(draft.get(key, "") or "").strip())
    has_sources = bool(state.get("found_papers") or state.get("github_repo_url") or state.get("artifacts"))
    score = 6.8 if non_empty_sections >= 4 and has_sources else 5.8
    comment = "草稿结构基本完整，仍需修正证据锚点与措辞。" if score >= 6.5 else "草稿结构或证据不足，建议先补齐关键章节后再收口。"
    issues = []
    if non_empty_sections < 4:
        issues.append({"section": "structure", "problem": "核心章节不完整", "suggestion": "优先补齐 intro/method/experiments/conclusion 四段。"})
    if not has_sources:
        issues.append({"section": "evidence", "problem": "证据链较弱", "suggestion": "在 related_work 与 experiments 中显式引用已观测来源。"})
    if error is not None:
        issues.append({"section": "runtime", "problem": f"Critic 降级：{str(error)[:120]}", "suggestion": "使用保守措辞完成交付。"})
    return {
        "scores": {"coverage": score, "evidence": score, "structure": score, "writing": score},
        "total": score,
        "overall_comment": comment,
        "issues": issues,
        "accept": score >= accept_threshold,
    }


def critic_node(state: ResearchState) -> ResearchState:
    accept_threshold = _critic_threshold(state)
    print(f"\n[Critic Agent] 审阅 Writer 草稿并生成修订意见（threshold={accept_threshold}）...")

    try:
        response = call_llm_json(
            "critique",
            [system(render_prompt(
                "agents.critic.minimal",
                doc_type=state.get("doc_type", "report"),
                question=state.get("selected_question", "") or state.get("raw_input", ""),
                accept_threshold=f"{accept_threshold:.1f}",
                sections_text=_sections_text(state),
            ))],
            max_tokens=32768,
        )
        data = parse_json(response)
        if not isinstance(data, dict):
            raise ValueError("critic response is not a dict")
    except Exception as e:
        data = _fallback_result(state, accept_threshold, e)

    score = float(data.get("total", 0.0) or 0.0)
    if score <= 0:
        scores = data.get("scores", {}) or {}
        if isinstance(scores, dict):
            nums = []
            for k in ("coverage", "evidence", "structure", "writing"):
                try:
                    nums.append(float(scores.get(k, 0.0) or 0.0))
                except Exception:
                    pass
            if nums:
                score = sum(nums) / len(nums)
    overall_comment = str(data.get("overall_comment", "") or "").strip()
    accepted = bool(data.get("accept", False)) or score >= accept_threshold
    issues = data.get("issues", []) or []
    if not isinstance(issues, list):
        issues = []

    issue_lines: list[str] = []
    for item in issues[:8]:
        if not isinstance(item, dict):
            continue
        section = str(item.get("section", "global") or "global").strip().lower()
        problem = str(item.get("problem", "") or "").strip()
        suggestion = str(item.get("suggestion", "") or "").strip()
        if not (problem or suggestion):
            continue
        issue_lines.append(f"[{section}] {problem} -> {suggestion}".strip())

    critique_parts = [
        f"Readiness: {score:.1f}/{accept_threshold:.1f}",
        overall_comment or "Critic 未返回总体判断。",
    ]
    if issue_lines:
        critique_parts.append("Issues:")
        critique_parts.extend(f"- {line}" for line in issue_lines)
    critique_text = "\n".join(part for part in critique_parts if part).strip()

    revision_suggestions = issue_lines[:6]
    writer_revision_brief = "\n".join(revision_suggestions) if revision_suggestions else (overall_comment or "保守表达并补充证据锚点。")

    return {
        **state,
        "current_stage": "critic",
        "critique": critique_text,
        "critique_score": score,
        "critique_accepted": accepted,
        "critique_accept_threshold": accept_threshold,
        "writer_revision_brief": writer_revision_brief,
        "writer_revision_suggestions": revision_suggestions,
        "revision_count": int(state.get("revision_count", 0) or 0) + 1,
    }
