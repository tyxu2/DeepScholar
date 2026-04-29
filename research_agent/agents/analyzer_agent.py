"""
Analyzer Supervisor Agent

Architecture (mirrors open_deep_research supervisor):
  - Runs a ReAct loop with three meta-tools:
      • think_tool       — pure strategic reflection (no-op externally)
      • conduct_research — spawns parallel SubResearcher workers
      • research_complete — end the research phase
  - Also has direct access to search_papers / read_papers for lightweight tasks
  - When conduct_research fires: ThreadPoolExecutor runs SubResearchers in parallel,
    returns compressed_evidence as tool results
  - When research_complete fires (or budget exhausted): exits loop and does
    final semantic compression for Writer/Critic

Key savings vs old Executor TAOR:
  - No per-step reflect LLM call
  - No multi-rollout fan-out
  - SubResearchers are cheap (no reflect, focused, compressed output)
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from research_agent.runtime.evidence import Evidence, deduplicate_evidence

from research_agent.llm.router import call_llm_json, call_llm_with_tools, system
from research_agent.prompts import render_prompt
from research_agent.react.tools import get_tool_by_name, to_litellm_tool
from research_agent.runtime.sub_researcher import run_sub_researcher
from research_agent.state import ResearchState
from research_agent.utils.json_parser import parse_json


# ── Meta-tool schemas (handled by supervisor, not tool registry) ───────────────

_THINK_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "think_tool",
        "description": (
            "Strategic reflection tool. Use this to reason about what to research next, "
            "evaluate current evidence gaps, or plan research delegation. "
            "Does not execute any external calls."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "reflection": {
                    "type": "string",
                    "description": "Your strategic reasoning about the current research state.",
                }
            },
            "required": ["reflection"],
        },
    },
}

_CONDUCT_RESEARCH_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "conduct_research",
        "description": (
            "Delegate a focused research task to a sub-researcher. "
            "The sub-researcher will search and read papers on the given topic, "
            "then return a compressed evidence summary. "
            "You can call this multiple times in one turn to run researchers in parallel. "
            "Use this for substantial sub-topics that need dedicated search effort."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "research_topic": {
                    "type": "string",
                    "description": (
                        "A specific, well-scoped research topic for the sub-researcher. "
                        "Be precise — include key terms, system names, and what evidence you need."
                    ),
                },
                "max_steps": {
                    "type": "integer",
                    "description": "Max tool-call steps for this sub-researcher (default 5).",
                    "default": 5,
                },
            },
            "required": ["research_topic"],
        },
    },
}

_RESEARCH_COMPLETE_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "research_complete",
        "description": (
            "Signal that research is complete and hand off to the Writer. "
            "Call this when you have sufficient evidence to support a comprehensive survey."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "Brief (1-2 sentence) summary of what was gathered.",
                }
            },
            "required": ["summary"],
        },
    },
}

# Direct research tools the supervisor can use without delegation
_SUPERVISOR_DIRECT_TOOL_NAMES = [
    "search_papers",
    "read_papers",
    "optimize_queries",
    "paper_store_query",
    "session_context",
]


# ── Tool helpers ───────────────────────────────────────────────────────────────

def _build_supervisor_tools() -> list[dict[str, Any]]:
    """All litellm-format tools available to the supervisor."""
    tools: list[dict[str, Any]] = [
        _THINK_TOOL_SCHEMA,
        _CONDUCT_RESEARCH_SCHEMA,
        _RESEARCH_COMPLETE_SCHEMA,
    ]
    for name in _SUPERVISOR_DIRECT_TOOL_NAMES:
        spec = get_tool_by_name(name)
        if spec is not None:
            tools.append(to_litellm_tool(spec))
    return tools


def _compact_obs(output: dict) -> str:
    obs: dict[str, Any] = {}
    for k, v in output.items():
        if k == "papers" and isinstance(v, list):
            obs["papers_count"] = len(v)
            obs["papers_sample"] = [p.get("title", "") for p in v[:4]]
        elif k == "summaries" and isinstance(v, list):
            obs["summaries_count"] = len(v)
            obs["summaries_sample"] = [s.get("title", "") for s in v[:4]]
        else:
            obs[k] = v
    return json.dumps(obs, ensure_ascii=False)[:800]


# ── Parallel sub-researcher execution ─────────────────────────────────────────

def _run_parallel_sub_researchers(
    conduct_calls: list[dict[str, Any]],
    base_state: ResearchState,
    max_concurrent: int = 6,
) -> list[dict[str, Any]]:
    """
    Execute conduct_research calls in parallel using ThreadPoolExecutor.
    Returns list of results in the same order as conduct_calls.
    """
    results: dict[int, dict[str, Any]] = {}
    allowed = conduct_calls[:max_concurrent]
    overflow = conduct_calls[max_concurrent:]

    with ThreadPoolExecutor(max_workers=max_concurrent) as pool:
        future_to_idx = {}
        for idx, tc in enumerate(allowed):
            args = tc.get("args", {}) if isinstance(tc.get("args"), dict) else {}
            topic = str(args.get("research_topic", "") or "general research")
            steps = int(args.get("max_steps", 5) or 5)
            future = pool.submit(
                run_sub_researcher,
                topic,
                base_state,
                steps,
                f"worker-{idx}",
            )
            future_to_idx[future] = idx

        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                results[idx] = {
                    "compressed_evidence": f"Sub-researcher failed: {e}",
                    "evidence": [],
                    "paper_summaries": [],
                    "found_papers": [],
                    "steps_used": 0,
                }

    # Overflow gets an error message
    for idx, tc in enumerate(overflow, start=len(allowed)):
        results[idx] = {
            "compressed_evidence": (
                f"Skipped: exceeded max_concurrent ({max_concurrent}) sub-researchers. "
                "Please reduce concurrent conduct_research calls."
            ),
            "evidence": [],
            "paper_summaries": [],
            "found_papers": [],
            "steps_used": 0,
        }

    return [results[i] for i in range(len(conduct_calls))]


def _merge_sub_researcher_results(
    state: ResearchState,
    results: list[dict[str, Any]],
) -> ResearchState:
    """Merge paper_summaries, found_papers, and Evidence cards from sub-researchers."""
    found = list(state.get("found_papers", []) or [])
    summaries = list(state.get("paper_summaries", []) or [])
    existing_evidence: list[Evidence] = list(state.get("evidence", []) or [])
    existing_found_titles = {str(p.get("title", "")).lower() for p in found}
    existing_summary_titles = {str(s.get("title", "")).lower() for s in summaries}

    new_evidence: list[Evidence] = []
    for r in results:
        for p in r.get("found_papers", []):
            t = str(p.get("title", "")).lower()
            if t and t not in existing_found_titles:
                existing_found_titles.add(t)
                found.append(p)
        for s in r.get("paper_summaries", []):
            t = str(s.get("title", "")).lower()
            if t and t not in existing_summary_titles:
                existing_summary_titles.add(t)
                summaries.append(s)
        new_evidence.extend(r.get("evidence", []) or [])

    all_evidence = deduplicate_evidence(existing_evidence + new_evidence)
    return {**state, "found_papers": found, "paper_summaries": summaries, "evidence": all_evidence}


# ── Supervisor context compression ────────────────────────────────────────────

_SUPERVISOR_COMPRESS_EVERY = 2   # compress after every N conduct_research rounds

def _compress_supervisor_messages(
    messages: list[dict[str, Any]],
    state: ResearchState,
) -> list[dict[str, Any]]:
    """
    Collapse old tool-call/result turns into a rolling summary user message.
    Keeps system + initial user goal + last 2 COMPLETE assistant+tool pairs.
    Prevents O(rounds) context growth in long supervisor loops.

    Critical invariant: every 'tool' message must be preceded by an 'assistant'
    message that contains the matching tool_call id — we never orphan tool msgs.
    """
    if len(messages) <= 4:
        return messages

    system_msgs = [m for m in messages if m.get("role") == "system"]
    non_system = [m for m in messages if m.get("role") != "system"]

    # Walk backwards to collect 2 complete assistant+tool groups
    recent: list[dict[str, Any]] = []
    groups_kept = 0
    i = len(non_system) - 1
    while i >= 0 and groups_kept < 2:
        msg = non_system[i]
        if msg.get("role") == "tool":
            # Collect all consecutive tool messages for this group
            group_tools: list[dict[str, Any]] = []
            while i >= 0 and non_system[i].get("role") == "tool":
                group_tools.insert(0, non_system[i])
                i -= 1
            # The message before them must be the assistant with tool_calls
            if i >= 0 and non_system[i].get("role") == "assistant":
                recent = [non_system[i]] + group_tools + recent
                i -= 1
                groups_kept += 1
            else:
                # Orphaned tool messages — skip them (shouldn't happen normally)
                break
        elif msg.get("role") == "assistant":
            recent = [msg] + recent
            i -= 1
            groups_kept += 1
        else:
            i -= 1

    # Build summary from state
    found = len(state.get("found_papers", []) or [])
    read = len(state.get("paper_summaries", []) or [])
    evidence_count = len(state.get("evidence", []) or [])
    brief_titles = [s.get("title", "") for s in (state.get("paper_summaries", []) or [])[:8]]
    summary_content = (
        "[RESEARCH PROGRESS SUMMARY — previous steps compressed]\n"
        f"Papers found: {found} | Summaries read: {read} | Evidence cards: {evidence_count}\n"
        "Topics covered:\n"
        + "\n".join(f"  - {t}" for t in brief_titles if t)
    )

    # Reconstruct: system + first user goal + summary + recent complete pairs
    first_user = next((m for m in non_system if m.get("role") == "user"), None)
    compressed: list[dict[str, Any]] = system_msgs[:]
    if first_user and first_user not in recent:
        compressed.append(first_user)
    compressed.append({"role": "user", "content": summary_content})
    compressed.extend(recent)
    return compressed


# ── Cross-paper synthesis (for paper comparison analysis) ──────────────────────

def _synthesize_comparison(evidence_list: list[Evidence], research_topic: str) -> dict[str, Any]:
    """
    Analyze evidence across papers to generate comparison synthesis.
    Returns method taxonomy, performance matrix, SOTA trajectory, conflicts, gaps.
    """
    if not evidence_list:
        return {
            "method_taxonomy": {},
            "performance_matrix": {},
            "sota_trajectory": [],
            "conflict_flags": [],
            "gap_analysis": [],
            "recommendation": "Insufficient evidence for cross-paper analysis.",
        }

    # Convert Evidence list to JSON for LLM
    evidence_json = json.dumps([
        {
            "title": ev.get("title", ""),
            "problem": ev.get("problem", ""),
            "method": ev.get("method", ""),
            "result": ev.get("result", ""),
            "limitations": ev.get("limitations", ""),
            "topic": ev.get("topic", ""),
        }
        for ev in evidence_list
    ])

    try:
        response = call_llm_json(
            "summarization",
            [system(render_prompt(
                "agents.analyzer.synthesize_comparison",
                evidence_list_json=evidence_json,
            ))],
            max_tokens=4000,
        )
        parsed = parse_json(response)
        if isinstance(parsed, dict):
            return parsed
    except Exception as e:
        print(f"  [synthesize_comparison] LLM failed: {e}")

    # Fallback: return empty structure
    return {
        "method_taxonomy": {},
        "performance_matrix": {},
        "sota_trajectory": [],
        "conflict_flags": [],
        "gap_analysis": [],
        "recommendation": f"Comparison synthesis failed (error: {str(e)[:50]}). Please review evidence manually.",
    }


# ── Supervisor TAOR loop ───────────────────────────────────────────────────────

def _run_supervisor_loop(state: ResearchState) -> tuple[ResearchState, str]:
    """
    Main ReAct supervisor loop.

    Returns (updated_state, stop_reason).
    stop_reason is one of:
      "research_complete", "no_tool_calls", "budget_exhausted", "error_budget"
    """
    caps = (state.get("agent_capabilities", {}) or {}).get("analyzer", {}) or {}
    max_steps = int(caps.get("max_steps", 6) or 6)
    max_concurrent = int(caps.get("max_concurrent_researchers", 6) or 6)
    max_errors = int(caps.get("max_errors", 3) or 3)

    tools = _build_supervisor_tools()
    goal = str(state.get("selected_question", "") or state.get("raw_input", "")).strip()
    research_brief = state.get("research_brief", {}) or {}

    # Derive minimum summary target from task_profile and constraints
    constraints = state.get("user_constraints", {}) or {}
    task_profile = str(state.get("task_profile", "general_research") or "general_research")
    explicit_limit = int(constraints.get("paper_limit", 0) or 0)
    if explicit_limit >= 3:
        min_summaries = explicit_limit
    elif task_profile == "literature_review":
        min_summaries = 10
    elif task_profile in {"repo_research", "general_task"}:
        min_summaries = 4
    else:
        min_summaries = 6

    system_msg = render_prompt(
        "agents.analyzer.supervisor",
        goal=goal,
        active_tools=", ".join(t["function"]["name"] for t in tools),
        research_brief_json=json.dumps(research_brief, ensure_ascii=False, indent=2),
        analyzer_capability_json=json.dumps(caps, ensure_ascii=False, indent=2),
        min_summaries=min_summaries,
    )

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_msg},
        {
            "role": "user",
            "content": (
                f"Begin research supervision. Goal: {goal}\n\n"
                f"Current evidence: {len(state.get('found_papers', []) or [])} papers found, "
                f"{len(state.get('paper_summaries', []) or [])} read.\n"
                f"Research brief:\n{json.dumps(research_brief, ensure_ascii=False)[:800]}"
            ),
        },
    ]

    errors = 0
    stop_reason = "budget_exhausted"
    conduct_rounds = 0   # track conduct_research calls for compression trigger

    for step in range(1, max_steps + 1):
        try:
            thought, tool_calls = call_llm_with_tools(
                "planning",
                messages,
                tools,
                max_tokens=3000,
            )
        except Exception as e:
            errors += 1
            print(f"  [Supervisor] step={step} llm error: {e}")
            if errors >= max_errors:
                stop_reason = "error_budget"
                break
            continue

        asst_msg: dict[str, Any] = {"role": "assistant", "content": thought or ""}
        if tool_calls:
            asst_msg["tool_calls"] = tool_calls
        messages.append(asst_msg)

        if thought:
            print(f"  [Supervisor] step={step} T: {thought[:120]}")

        if not tool_calls:
            stop_reason = "no_tool_calls"
            break

        # ── Classify tool calls ────────────────────────────────────────────────
        think_calls = [tc for tc in tool_calls if tc.get("function", {}).get("name") == "think_tool"]
        conduct_calls = [tc for tc in tool_calls if tc.get("function", {}).get("name") == "conduct_research"]
        complete_calls = [tc for tc in tool_calls if tc.get("function", {}).get("name") == "research_complete"]
        direct_calls = [
            tc for tc in tool_calls
            if tc.get("function", {}).get("name") not in {"think_tool", "conduct_research", "research_complete"}
        ]

        tool_result_msgs: list[dict[str, Any]] = []

        # 1. think_tool — pure reflection, acknowledge and continue
        for tc in think_calls:
            reflection = ""
            try:
                args = json.loads(tc.get("function", {}).get("arguments", "{}"))
                reflection = str(args.get("reflection", ""))
            except Exception:
                pass
            print(f"  [Supervisor] think: {reflection[:120]}")
            tool_result_msgs.append({
                "role": "tool",
                "tool_call_id": str(tc.get("id", "") or "unknown"),
                "name": "think_tool",
                "content": "Reflection recorded.",
            })

        # 2. conduct_research — run sub-researchers in parallel
        if conduct_calls:
            parsed_calls = []
            for tc in conduct_calls:
                try:
                    args = json.loads(tc.get("function", {}).get("arguments", "{}"))
                except Exception:
                    args = {}
                parsed_calls.append({"tc": tc, "args": args})

            print(f"  [Supervisor] step={step} conduct_research × {len(parsed_calls)}")
            sub_inputs = [{"args": pc["args"]} for pc in parsed_calls]
            sub_results = _run_parallel_sub_researchers(sub_inputs, state, max_concurrent)

            # Merge new papers/summaries into state
            state = _merge_sub_researcher_results(state, sub_results)
            conduct_rounds += 1
            if conduct_rounds % _SUPERVISOR_COMPRESS_EVERY == 0:
                messages = _compress_supervisor_messages(messages, state)
                print(f"  [Supervisor] context compressed after {conduct_rounds} conduct_research rounds")

            for pc, result in zip(parsed_calls, sub_results):
                tc = pc["tc"]
                compressed = result.get("compressed_evidence", "No findings.")
                n_papers = len(result.get("found_papers", []))
                n_summaries = len(result.get("paper_summaries", []))
                content = (
                    f"Research complete ({result.get('steps_used', 0)} steps, "
                    f"{n_papers} papers found, {n_summaries} summaries).\n\n"
                    f"Evidence summary:\n{compressed}"
                )
                tool_result_msgs.append({
                    "role": "tool",
                    "tool_call_id": str(tc.get("id", "") or "unknown"),
                    "name": "conduct_research",
                    "content": content[:3000],
                })

        # 3. Direct tool calls (search_papers, read_papers, etc.)
        for tc in direct_calls:
            tool_name = str(tc.get("function", {}).get("name", "") or "")
            args_raw = tc.get("function", {}).get("arguments", "{}")
            try:
                tool_input = json.loads(args_raw)
                if not isinstance(tool_input, dict):
                    tool_input = {}
            except Exception:
                tool_input = {}

            spec = get_tool_by_name(tool_name)
            if spec is None:
                obs_str = json.dumps({"error": f"unknown tool: {tool_name}"})
                errors += 1
            else:
                try:
                    output = spec.fn(tool_input, state)
                    state = spec.state_updater(state, output)
                    obs_str = _compact_obs(output)
                except Exception as e:
                    obs_str = json.dumps({"error": str(e)[:200]})
                    errors += 1

            tool_result_msgs.append({
                "role": "tool",
                "tool_call_id": str(tc.get("id", "") or "unknown"),
                "name": tool_name,
                "content": obs_str,
            })

        # 4. research_complete — end loop
        if complete_calls:
            current_read = len(state.get("paper_summaries", []) or [])
            if current_read < min_summaries:
                msg = (
                    f"Not enough evidence yet: {current_read}/{min_summaries} summaries. "
                    "Continue research and fill remaining sub-topic gaps before completion."
                )
                print(f"  [Supervisor] research_complete rejected: {msg}")
                for tc in complete_calls:
                    tool_result_msgs.append({
                        "role": "tool",
                        "tool_call_id": str(tc.get("id", "") or "unknown"),
                        "name": "research_complete",
                        "content": msg,
                    })
                messages.extend(tool_result_msgs)
                continue

            for tc in complete_calls:
                try:
                    args = json.loads(tc.get("function", {}).get("arguments", "{}"))
                    summary = str(args.get("summary", ""))
                except Exception:
                    summary = ""
                tool_result_msgs.append({
                    "role": "tool",
                    "tool_call_id": str(tc.get("id", "") or "unknown"),
                    "name": "research_complete",
                    "content": "Research phase complete.",
                })
                print(f"  [Supervisor] research_complete: {summary[:120]}")
            stop_reason = "research_complete"
            messages.extend(tool_result_msgs)
            break

        messages.extend(tool_result_msgs)

        found = len(state.get("found_papers", []) or [])
        read = len(state.get("paper_summaries", []) or [])
        print(f"  [Supervisor] step={step} state: found={found} read={read}")

        if errors >= max_errors:
            stop_reason = "error_budget"
            break

    return state, stop_reason


# ── Final compression (Analyzer main) ─────────────────────────────────────────

def _build_compression_input(state: ResearchState) -> str:
    payload = {
        "goal": state.get("selected_question", "") or state.get("raw_input", ""),
        "task_profile": state.get("task_profile", "general_research"),
        "papers_found": len(state.get("found_papers", []) or []),
        "papers_read": len(state.get("paper_summaries", []) or []),
        "top_summaries": [
            {
                "title": s.get("title", ""),
                "problem": (s.get("problem", "") or "")[:500],
                "method":  (s.get("method",  "") or "")[:500],
                "result":  (s.get("result",  "") or "")[:500],
            }
            for s in (state.get("paper_summaries", []) or [])[:12]
        ],
        "research_brief": state.get("research_brief", {}),
        "stop_reason": state.get("analyzer_stop_reason", ""),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _fallback_analysis(state: ResearchState) -> dict[str, Any]:
    q = str(state.get("selected_question", "") or state.get("raw_input", "") or "").strip()
    found = len(state.get("found_papers", []) or [])
    read = len(state.get("paper_summaries", []) or [])
    return {
        "selected_rollout_id": 0,
        "selection_reason": "heuristic fallback",
        "analysis_brief": (
            f"Research gathered {found} papers ({read} read) on topic: {q[:120]}. "
            "Writer should ground claims in available summaries and flag uncertainty."
        ),
        "key_points": [f"Found {found} papers, read {read}"],
        "merged_evidence": [s.get("title", "") for s in (state.get("paper_summaries", []) or [])[:6]],
        "open_risks": (
            ["Insufficient evidence — fewer than 4 papers read"] if read < 4 else []
        ),
        "writer_focus": "Ground every claim in paper_summaries; include comparison tables.",
        "critic_focus": "Check citation coverage and quantitative results tables.",
    }


# ── Public entry point ─────────────────────────────────────────────────────────

def analyzer_node(state: ResearchState) -> ResearchState:
    print("\n[Analyzer Supervisor] Starting supervisor loop...")

    # 1. Run supervisor ReAct loop
    state, stop_reason = _run_supervisor_loop(state)
    state = {**state, "analyzer_stop_reason": stop_reason}

    found = len(state.get("found_papers", []) or [])
    read = len(state.get("paper_summaries", []) or [])
    print(f"[Analyzer Supervisor] Loop ended ({stop_reason}). found={found} read={read}")

    # 1.5 Synthesize cross-paper comparison analysis
    print(f"[Analyzer Supervisor] Generating cross-paper synthesis...")
    evidence_list = list(state.get("evidence", []) or [])
    comparison_synthesis = _synthesize_comparison(
        evidence_list,
        research_topic=str(state.get("selected_question", "") or "").strip()
    )
    print(f"  [synthesize_comparison] Generated: "
          f"taxonomy={len(comparison_synthesis.get('method_taxonomy', {}))} categories, "
          f"conflicts={len(comparison_synthesis.get('conflict_flags', []))}, "
          f"gaps={len(comparison_synthesis.get('gap_analysis', []))}")

    # 2. Final semantic compression for Writer/Critic
    data: dict[str, Any]
    try:
        response = call_llm_json(
            "summarization",
            [system(render_prompt("agents.analyzer.main", input_json=_build_compression_input(state)))],
            max_tokens=6000,
        )
        parsed = parse_json(response)
        if not isinstance(parsed, dict):
            raise ValueError("analyzer response is not a dict")
        data = parsed
    except Exception as e:
        print(f"  [Analyzer] compression LLM failed: {e}, using fallback")
        data = _fallback_analysis(state)

    selected_rollout_id = int(data.get("selected_rollout_id", 0) or 0)
    selection_reason = str(data.get("selection_reason", "") or "").strip()
    analysis_brief = str(data.get("analysis_brief", "") or "").strip()
    key_points = [str(p).strip() for p in (data.get("key_points", []) or []) if str(p).strip()][:8]
    merged_evidence = [str(p).strip() for p in (data.get("merged_evidence", []) or []) if str(p).strip()][:12]
    open_risks = [str(p).strip() for p in (data.get("open_risks", []) or []) if str(p).strip()][:8]
    writer_focus = str(data.get("writer_focus", "") or "").strip()
    critic_focus = str(data.get("critic_focus", "") or "").strip()

    if not analysis_brief:
        fb = _fallback_analysis(state)
        analysis_brief = str(fb["analysis_brief"])

    context_package = dict(state.get("context_package", {}) or {})
    context_package["analyzer"] = {
        "selected_rollout_id": selected_rollout_id,
        "selection_reason": selection_reason,
        "analysis_brief": analysis_brief,
        "key_points": key_points,
        "merged_evidence": merged_evidence,
        "open_risks": open_risks,
        "writer_focus": writer_focus,
        "critic_focus": critic_focus,
        "stop_reason": stop_reason,
        "comparison_synthesis": comparison_synthesis,
    }

    return {
        **state,
        "selected_rollout_id": selected_rollout_id,
        "selection_reason": selection_reason,
        "analysis_brief": analysis_brief,
        "analysis_key_points": key_points,
        "analysis_open_risks": open_risks,
        "analysis_writer_focus": writer_focus,
        "analysis_critic_focus": critic_focus,
        "comparison_synthesis": comparison_synthesis,
        "context_package": context_package,
        "analyzer_stop_reason": stop_reason,
        "current_stage": "analyzer",
        "evidence": deduplicate_evidence(list(state.get("evidence", []) or [])),
        # Compatibility fields
        "rollout_summaries": state.get("rollout_summaries", []),
        "analyzer_supervisor_round": 1,
    }
