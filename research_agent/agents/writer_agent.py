"""
Writer Agent — narrow interface.

Input:   Analyzer + Critic + evidence bundle (from state)
Output:  structured Draft + Markdown file saved to disk

No replan, no revision loop, no rules bundle.
Just: take the aggregated context and write a clean structured memo.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any

from research_agent.llm.router import call_llm_json, system
from research_agent.prompts import render_prompt
from research_agent.state import Draft, ResearchState
from research_agent.utils.json_parser import parse_json
from research_agent.writer.latex_writer import render_latex, save_latex
from research_agent.writer.markdown_writer import render_markdown, save_markdown


def _make_cite_key(s: dict, used: set) -> str:
    """Generate an author+year cite key (e.g. kwon2023) for a paper summary."""
    authors = s.get("authors", []) or []
    year = int(s.get("year", 0) or 0)
    if authors:
        last_name = str(authors[0]).split()[-1].lower()
        last_name = re.sub(r"[^a-z]", "", last_name)
    else:
        title_words = str(s.get("title", "") or "").split()
        first_word = title_words[0] if title_words else "unknown"
        last_name = re.sub(r"[^a-z]", "", first_word.lower()[:10])
    if not last_name:
        last_name = "unknown"
    year_str = str(year) if year else ""
    base = f"{last_name}{year_str}"
    key = base
    suffix = ord("a")
    while key in used:
        key = base + chr(suffix)
        suffix += 1
    used.add(key)
    return key


def _build_paper_summaries_with_keys(state: ResearchState) -> list[dict]:
    """Build paper summaries list with author+year cite keys.
    Cap at 24 papers and trim field sizes to keep input_json manageable.
    """
    used: set[str] = set()
    result = []
    for s in (state.get("paper_summaries", []) or [])[:24]:
        key = _make_cite_key(s, used)
        result.append({
            "cite_key": key,
            "title": s.get("title", ""),
            "authors": s.get("authors", []),
            "year": s.get("year", 0),
            "problem": (s.get("problem", "") or "")[:300],
            "method": (s.get("method", "") or "")[:400],
            "result": (s.get("result", "") or "")[:400],
            "limitations": (s.get("limitations", "") or "")[:200],
        })
    return result


def _build_input_json(state: ResearchState) -> str:
    constraints = state.get("user_constraints", {}) or {}
    revision_round = int(state.get("writer_head_round", 1) or 1)
    revision_feedback = str(state.get("writer_revision_brief", "") or "").strip()
    revision_suggestions = list(state.get("writer_revision_suggestions", []) or [])
    payload = {
        "question": state.get("selected_question", "") or state.get("raw_input", ""),
        "task_profile": state.get("task_profile", "general_research"),
        "doc_type": state.get("doc_type", "report"),
        "output_language": constraints.get("output_language", "auto"),
        "revision_round": revision_round,
        "revision_feedback": revision_feedback,
        "revision_suggestions": revision_suggestions[:8],
        "previous_draft": state.get("draft", {}),
        # Analyzer outputs
        "analysis_brief": state.get("analysis_brief", ""),
        "analysis_key_points": (state.get("analysis_key_points", []) or [])[:8],
        "analysis_open_risks": (state.get("analysis_open_risks", []) or [])[:6],
        "analysis_writer_focus": state.get("analysis_writer_focus", ""),
        "selection_reason": state.get("selection_reason", ""),
        "selected_rollout_id": state.get("selected_rollout_id", 0),
        # Critic outputs
        "critique": state.get("critique", ""),
        "critique_score": state.get("critique_score", 0.0),
        "critique_accepted": state.get("critique_accepted", False),
        # Evidence bundle — cite keys in author+year format (e.g. kwon2023)
        "papers_found": len(state.get("found_papers", []) or []),
        "paper_summaries": _build_paper_summaries_with_keys(state),
        "github_repo_url": state.get("github_repo_url", ""),
        "artifacts": [
            {"title": a.get("title", ""), "path": a.get("content_ref", "")}
            for a in (state.get("artifacts", []) or [])[:6]
        ],
        "stop_reason": state.get("stop_reason", ""),
        "assistant_response": (state.get("assistant_response", "") or "")[:800],
        "local_context_preview": (state.get("local_context", "") or "")[:1200],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _fallback_draft(state: ResearchState) -> dict[str, Any]:
    """Deterministic fallback when LLM fails: build a structured English draft from raw evidence."""
    q = state.get("selected_question", "") or state.get("raw_input", "")
    analysis_brief = str(state.get("analysis_brief", "") or "").strip()
    key_points = [str(p) for p in (state.get("analysis_key_points", []) or []) if str(p).strip()]
    open_risks = [str(r) for r in (state.get("analysis_open_risks", []) or []) if str(r).strip()]
    summaries = list(state.get("paper_summaries", []) or [])

    kp_text = "\n".join(f"- {p}" for p in key_points[:8]) or "No key points extracted."
    risk_text = "\n".join(f"- {r}" for r in open_risks[:6]) or "No major risks identified."

    # Build related_work from summaries
    rw_lines = []
    for s in summaries[:12]:
        title = s.get("title", "Untitled")
        method = str(s.get("method", "") or "").strip()[:200]
        result = str(s.get("result", "") or "").strip()[:200]
        rw_lines.append(f"**{title}** — {method}" + (f" Results: {result}" if result else ""))
    rw_text = "\n\n".join(rw_lines) if rw_lines else "No papers were successfully summarized."

    return {
        "title": (q[:100] or "Research Survey"),
        "abstract": (
            f"This survey examines {q}. "
            + (analysis_brief[:400] if analysis_brief else f"We surveyed {len(summaries)} papers on the topic.")
        ),
        "intro": (
            f"This report addresses the following research question: {q}\n\n"
            + (analysis_brief[:800] if analysis_brief else "Evidence gathering is in progress.")
        ),
        "related_work": rw_text,
        "method": f"Key findings from the surveyed literature:\n\n{kp_text}",
        "experiments": f"Coverage gaps and open risks:\n\n{risk_text}",
        "conclusion": (
            f"Based on {len(summaries)} surveyed papers, the main contributions identified are:\n\n{kp_text}"
            if key_points else
            f"The survey gathered {len(summaries)} papers on {q}. Further analysis is recommended."
        ),
    }


def writer_node(state: ResearchState) -> ResearchState:
    round_id = int(state.get("writer_head_round", 1) or 1)
    print(f"\n[Writer Agent] Round {round_id}: 基于 Analyzer + Critic 生成/修订结构化草稿...")

    raw: dict[str, Any]
    input_json = _build_input_json(state)
    try:
        response = call_llm_json(
            "writing",
            [system(render_prompt("agents.writer.main", input_json=input_json))],
            max_tokens=32000,
        )
        parsed = parse_json(response)
        if not isinstance(parsed, dict):
            raise ValueError(f"writer response is not a dict: {type(parsed)}")
        # Require at least one non-empty section before accepting
        non_empty = sum(1 for k in ("abstract", "intro", "related_work", "method") if str(parsed.get(k, "") or "").strip())
        if non_empty == 0:
            raise ValueError("writer response has no section content")
        raw = parsed
    except Exception as e:
        import traceback
        print(f"  [Writer Agent] LLM failed (round {round_id}): {e}")
        print(f"  [Writer Agent] traceback: {traceback.format_exc()[-600:]}")
        print(f"  [Writer Agent] input_json size: {len(input_json)} chars")
        raw = _fallback_draft(state)

    title = str(raw.get("title", "") or state.get("selected_question", "") or "Research Output").strip()
    draft: Draft = {
        "abstract":     str(raw.get("abstract", "") or "").strip(),
        "intro":        str(raw.get("intro", "") or "").strip(),
        "related_work": str(raw.get("related_work", "") or "").strip(),
        "method":       str(raw.get("method", "") or "").strip(),
        "experiments":  str(raw.get("experiments", "") or "").strip(),
        "conclusion":   str(raw.get("conclusion", "") or "").strip(),
    }

    paper_summaries = list(state.get("paper_summaries", []) or [])
    session_id = state.get("session_id", "default")

    # Render and save Markdown
    md_content = render_markdown(
        title=title,
        draft=draft,
        experiment_results=state.get("experiment_results", {}),
        paper_summaries=paper_summaries,
    )
    md_path = os.path.join("./output", session_id, "paper.md")
    save_markdown(md_content, md_path)
    print(f"  [Writer Agent] Markdown saved → {md_path}")

    # Render and save LaTeX
    latex_path = ""
    try:
        templates_dir = os.path.join(os.path.dirname(__file__), "..", "..", "templates")
        latex_content = render_latex(
            title=title,
            draft=draft,
            experiment_results=state.get("experiment_results", {}),
            paper_summaries=paper_summaries,
            templates_dir=os.path.abspath(templates_dir),
        )
        latex_path = os.path.join("./output", session_id, "paper.tex")
        save_latex(latex_content, latex_path)
        print(f"  [Writer Agent] LaTeX saved → {latex_path}")
    except Exception as e:
        print(f"  [Writer Agent] LaTeX render failed (non-fatal): {e}")

    artifacts = list(state.get("artifacts", []) or [])
    artifacts.append({
        "id": f"draft-md-{len(artifacts) + 1}",
        "kind": "markdown",
        "title": title,
        "source_stage": "write",
        "content_ref": md_path,
        "evidence_ids": list(state.get("evidence_ids", []) or []),
    })
    if latex_path:
        artifacts.append({
            "id": f"draft-tex-{len(artifacts) + 1}",
            "kind": "latex",
            "title": title,
            "source_stage": "write",
            "content_ref": latex_path,
            "evidence_ids": list(state.get("evidence_ids", []) or []),
        })

    return {
        **state,
        "current_stage": "write",
        "draft": draft,
        "raw_draft": dict(raw),
        "draft_md_path": md_path,
        "draft_latex_path": latex_path,
        "artifacts": artifacts,
        "writer_head_round": round_id,
    }
