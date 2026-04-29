from __future__ import annotations

import json
import os
from typing import Any

from research_agent.state import ResearchState


def build_evolution_report(state: ResearchState) -> dict[str, Any]:
    action_history = state.get("action_history", []) or []
    error_log = state.get("error_log", []) or []
    papers = state.get("found_papers", []) or []
    summaries = state.get("paper_summaries", []) or []
    plan = state.get("brain_plan", []) or []
    critique_score = float(state.get("critique_score", 0.0) or 0.0)

    issues: list[str] = []
    proposals: list[dict[str, str]] = []

    if not plan:
        issues.append("planner did not produce an actionable plan")
        proposals.append(
            {
                "scope": "planner",
                "proposal": "tighten planner prompt so it always emits at least one executable stage or an explicit final answer",
            }
        )
    if "search" in plan and len(papers) < 5:
        issues.append("paper retrieval coverage is still thin")
        proposals.append(
            {
                "scope": "search",
                "proposal": "increase query diversity and trigger another search optimization round before writing",
            }
        )
    if "read" in plan and papers and not summaries:
        issues.append("reader failed to turn retrieved sources into usable evidence")
        proposals.append(
            {
                "scope": "reader",
                "proposal": "fallback to abstract-only extraction when PDF parsing fails",
            }
        )
    if "critic" in plan and critique_score < 7.0:
        issues.append("final draft quality did not clear the current critic threshold")
        proposals.append(
            {
                "scope": "writer",
                "proposal": "feed critic findings back into a targeted revision pass instead of a full rewrite",
            }
        )
    if error_log:
        issues.append("runtime errors were observed during execution")
        proposals.append(
            {
                "scope": "runtime",
                "proposal": "add explicit fallback behavior for the most common tool and parsing failures",
            }
        )
    if not proposals:
        proposals.append(
            {
                "scope": "system",
                "proposal": "current run is stable; next evolution step should focus on benchmark-driven optimization rather than urgent bug fixes",
            }
        )

    return {
        "protocol": "SEPL-lite",
        "status": "needs_iteration" if issues else "stable",
        "issues": issues,
        "proposals": proposals,
        "signals": {
            "steps": len(action_history),
            "errors": len(error_log),
            "papers_found": len(papers),
            "papers_read": len(summaries),
            "critique_score": critique_score,
        },
    }


def persist_evolution_report(state: ResearchState, output_dir: str = "./output") -> str:
    os.makedirs(output_dir, exist_ok=True)
    session_id = state.get("session_id", "default") or "default"
    path = os.path.join(output_dir, f"sepl_report_{session_id}.json")
    report = build_evolution_report(state)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return path
