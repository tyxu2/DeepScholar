"""
Sub-Researcher — parallel worker spawned by Analyzer Supervisor.

Three mechanisms:

  1. TAOR LOOP (Thought → Action → Observation → Reflection)
     After each tool-call step, a lightweight reflect LLM call decides
     whether to continue, finish, or handoff.  This mirrors the Executor
     TAOR loop and prevents the researcher from drifting off-topic.

  2. ROLLING CONTEXT COMPRESSION
     After COMPRESS_AFTER_STEPS turns, old assistant/tool messages are folded
     into a single rolling-summary user message. The LLM at step N only ever
     sees: system + rolling-summary + last KEEP_RECENT_TURNS turns.
     This keeps context O(constant) instead of O(steps).

  3. STRUCTURED EVIDENCE OUTPUT
     compress() returns List[Evidence] cards (title/method/result/cite_key)
     instead of free prose. The Supervisor receives a structured JSON array
     it can cite directly, not a paragraph it has to re-parse.
     Evidence objects flow: SubResearcher → Supervisor state → Writer [refN].

Design (mirrors open_deep_research researcher subgraph):
  - Given a single research_topic string
  - Loops: call_llm_with_tools → execute all tools → reflect → check stop → loop
  - Returns: {"compressed_evidence": str, "evidence": List[Evidence], ...}
"""
from __future__ import annotations

import json
from typing import Any

from research_agent.agents.executor_agent import _reflect_step
from research_agent.llm.router import call_llm, call_llm_with_tools, system, user
from research_agent.react.tools import get_tool_by_name, to_litellm_tool
from research_agent.runtime.evidence import Evidence, deduplicate_evidence, evidence_from_summary
from research_agent.state import ResearchState

# ── Config ─────────────────────────────────────────────────────────────────────

# After this many tool-call rounds, compress the old turns into a rolling summary
COMPRESS_AFTER_STEPS = 3
# How many recent turns to keep verbatim after compression
KEEP_RECENT_TURNS = 2

_SUB_RESEARCHER_TOOL_NAMES = [
    "search_papers",
    "quick_relevant_check",
    "read_papers",
    "refine_evidence",
    "optimize_queries",
    "arxiv_search",
    "paper_search",
    "paper_store_query",
    "fetch_url",
    "web_search",
]

# ── Prompts ────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a focused Sub-Researcher. Your only job is to gather evidence on the given topic.

Research topic: {research_topic}

STRICT WORKFLOW — follow this order every time:
1. Run ONE focused search (search_papers or arxiv_search) to find relevant papers.
2. (OPTIONAL) Call quick_relevant_check to filter found papers by relevance — this saves reading time.
3. IMMEDIATELY call read_papers on the results — do NOT run a second search before reading.
4. (OPTIONAL) For top papers, call refine_evidence with stage="qualitative" to deepen extraction.
5. If evidence is still insufficient after reading, run ONE more search then read again.
6. Stop once you have read ≥ 4 papers OR have exhausted useful search angles.

HARD RULES:
- You MUST call read_papers after every search that returns results.
- Never run 2 consecutive searches without a read_papers call in between.
- Do NOT explain your plan — call tools immediately.
- When you have enough evidence, stop calling tools (return a message with no tool calls).
- If you use quick_relevant_check, ensure it returns ≥5 papers before reading (the filter should return top-K high-relevance papers).
"""

_ROLLING_SUMMARY_TEMPLATE = (
    "[Rolling summary of prior steps — details already processed]\n"
    "Papers found so far: {found}\n"
    "Papers read (summaries extracted): {read}\n"
    "Queries used: {queries}\n"
    "Key titles seen: {titles}\n"
    "[Continue searching for remaining gaps in the topic below]"
)

_COMPRESS_PROMPT = """\
You are a research evidence extractor. Given raw tool observations from a researcher,
extract structured evidence cards and a brief synthesis.

Research topic: {research_topic}

Raw observations:
{raw_findings}

Return ONLY valid JSON in this exact format (no markdown, no explanation):
{{
  "evidence": [
    {{
      "title": "exact paper title",
      "method": "core technique in 1-2 sentences",
      "result": "key quantitative or qualitative result",
      "limitations": "main caveat or 'Not mentioned'",
      "problem": "what problem it solves in 1 sentence"
    }}
  ],
  "synthesis": "2-3 sentence synthesis covering the topic, highlighting gaps"
}}

Rules:
- Include only papers that were actually retrieved (grounded in observations).
- result field must contain numbers/metrics when available (latency ms, throughput, speedup ×).
- If no papers were retrieved, return empty evidence list and note gaps in synthesis.
"""

# ── Helpers ────────────────────────────────────────────────────────────────────

def _collect_tools(base_state: ResearchState) -> list[Any]:
    specs = []
    for name in _SUB_RESEARCHER_TOOL_NAMES:
        spec = get_tool_by_name(name)
        if spec is not None:
            specs.append(spec)
    return specs


def _compact_obs(tool_name: str, output: dict) -> str:
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
    return json.dumps(obs, ensure_ascii=False)[:600]


def _build_rolling_summary(
    state: ResearchState,
    papers_before: int,
    summaries_before: int,
    used_queries: list[str],
) -> str:
    """Compact summary of what has been gathered so far, injected as a user message."""
    found = len(state.get("found_papers", []) or []) - papers_before
    read = len(state.get("paper_summaries", []) or []) - summaries_before
    titles = [
        s.get("title", "")
        for s in (state.get("paper_summaries", []) or [])[summaries_before:summaries_before + 6]
    ]
    return _ROLLING_SUMMARY_TEMPLATE.format(
        found=found,
        read=read,
        queries=", ".join(used_queries[-4:]),
        titles="; ".join(t for t in titles if t) or "none yet",
    )


def _find_turn_boundary(messages: list[dict[str, Any]], keep_turns: int) -> int:
    """
    Walk backwards to find the start index of the Nth-from-last complete turn.
    A turn starts at an assistant message that has tool_calls.
    Returns the index of the first message to keep (always an assistant msg),
    or 1 (keep everything after system) if not enough turns found.
    """
    turns_seen = 0
    i = len(messages) - 1
    while i >= 1:
        msg = messages[i]
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            turns_seen += 1
            if turns_seen >= keep_turns:
                return i
        i -= 1
    return 1  # fallback: keep all non-system messages


def _compress_messages(
    messages: list[dict[str, Any]],
    state: ResearchState,
    papers_before: int,
    summaries_before: int,
    used_queries: list[str],
) -> list[dict[str, Any]]:
    """
    Rolling compression: replace all turns except the last KEEP_RECENT_TURNS
    with a single rolling-summary user message.

    Turn boundaries are found by walking backwards to assistant messages that
    have tool_calls, so we never start a slice on an orphaned tool result.
    """
    if len(messages) <= 1 + KEEP_RECENT_TURNS * 2:
        return messages

    boundary = _find_turn_boundary(messages, KEEP_RECENT_TURNS)
    if boundary <= 1:
        return messages  # not enough history to compress safely

    rolling = _build_rolling_summary(state, papers_before, summaries_before, used_queries)
    recent = messages[boundary:]
    return [
        messages[0],                              # system
        {"role": "user", "content": rolling},     # rolling summary
        *recent,
    ]


# ── Compression / evidence extraction ─────────────────────────────────────────

def _extract_evidence(
    raw_observations: list[str],
    research_topic: str,
    new_summaries: list[dict],
    topic_label: str,
) -> tuple[str, list[Evidence]]:
    """
    Call LLM to extract structured Evidence cards from raw observations.
    Falls back to building Evidence directly from paper_summaries if LLM fails.
    """
    from research_agent.utils.json_parser import parse_json

    raw_findings = "\n\n".join(raw_observations) if raw_observations else "No tool results."

    try:
        raw = call_llm(
            "summarization",
            [
                system(_COMPRESS_PROMPT.format(
                    research_topic=research_topic,
                    raw_findings=raw_findings[:6000],
                )),
                user("Extract evidence cards now."),
            ],
            max_tokens=2000,
        )
        parsed = parse_json(raw)
        if not isinstance(parsed, dict):
            raise ValueError("not a dict")

        synthesis = str(parsed.get("synthesis", "") or "").strip()
        raw_ev_list = parsed.get("evidence", []) or []

        evidence_list: list[Evidence] = []
        for i, item in enumerate(raw_ev_list[:20]):
            if not isinstance(item, dict):
                continue
            title = str(item.get("title", "") or "").strip()
            if not title:
                continue
            from research_agent.runtime.evidence import make_evidence_id
            method = str(item.get("method", "") or "")
            ev = Evidence(
                id=make_evidence_id(title, method),
                cite_key=f"ref{i+1}",
                title=title,
                source_type="paper",
                problem=str(item.get("problem", "") or "")[:300],
                method=method[:300],
                result=str(item.get("result", "") or "")[:300],
                limitations=str(item.get("limitations", "") or "")[:150],
                topic=topic_label,
            )
            evidence_list.append(ev)

        # Fallback: also convert any paper_summaries not caught by LLM
        seen_titles = {e["title"].lower() for e in evidence_list}
        for i, s in enumerate(new_summaries):
            title = str(s.get("title", "") or "").strip()
            if title.lower() not in seen_titles:
                ev = evidence_from_summary(s, f"ref{len(evidence_list)+1}", topic=topic_label)
                evidence_list.append(ev)
                seen_titles.add(title.lower())

        compressed_text = _format_evidence_as_text(evidence_list, synthesis)
        return compressed_text, deduplicate_evidence(evidence_list)

    except Exception as e:
        # Hard fallback: build from paper_summaries directly, no LLM
        evidence_list = [
            evidence_from_summary(s, f"ref{i+1}", topic=topic_label)
            for i, s in enumerate(new_summaries[:20])
        ]
        compressed_text = (
            f"[Evidence extraction failed: {e}. "
            f"Gathered {len(new_summaries)} summaries on topic: {research_topic}]"
        )
        return compressed_text, deduplicate_evidence(evidence_list)


def _format_evidence_as_text(evidence_list: list[Evidence], synthesis: str) -> str:
    """
    Render Evidence list as a structured text block for Supervisor's tool result.
    Supervisor reads this; it has named fields so it can cite [refN] directly.
    """
    if not evidence_list:
        return synthesis or "No evidence gathered."

    lines = []
    if synthesis:
        lines.append(f"SYNTHESIS: {synthesis}\n")
    lines.append("EVIDENCE CARDS:")
    for ev in evidence_list:
        lines.append(
            f"[{ev.get('cite_key','?')}] {ev.get('title','?')}\n"
            f"  Problem: {ev.get('problem','N/A')}\n"
            f"  Method:  {ev.get('method','N/A')}\n"
            f"  Result:  {ev.get('result','N/A')}\n"
            f"  Limits:  {ev.get('limitations','N/A')}"
        )
    return "\n".join(lines)


# ── Main entry point ───────────────────────────────────────────────────────────

def run_sub_researcher(
    research_topic: str,
    base_state: ResearchState,
    max_steps: int = 6,
    rollout_label: str = "",
) -> dict[str, Any]:
    """
    Run a lightweight researcher loop on a single topic.

    Returns:
      compressed_evidence  str             — structured text for Supervisor tool result
      evidence             List[Evidence]  — typed evidence cards for Writer citation
      paper_summaries      list            — new summaries gathered
      found_papers         list            — new papers found
      steps_used           int
    """
    label = rollout_label or research_topic[:40]
    print(f"  [SubResearcher:{label}] start — '{research_topic[:70]}'")

    state: ResearchState = dict(base_state)  # type: ignore[assignment]
    state = {**state, "_log_caller": f"SubResearcher:{label}"}

    tools = _collect_tools(state)
    if not tools:
        return {
            "compressed_evidence": f"No tools available for: {research_topic}",
            "evidence": [],
            "paper_summaries": [],
            "found_papers": [],
            "steps_used": 0,
        }

    litellm_tools = [to_litellm_tool(spec) for spec in tools]
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _SYSTEM_PROMPT.format(research_topic=research_topic)},
        {"role": "user", "content": f"Begin research on: {research_topic}"},
    ]

    raw_observations: list[str] = []
    used_queries: list[str] = []
    papers_before = len(state.get("found_papers", []) or [])
    summaries_before = len(state.get("paper_summaries", []) or [])
    steps_used = 0

    for step in range(1, max_steps + 1):
        steps_used = step

        # ── Rolling compression: fold old turns before calling LLM ────────────
        if step > COMPRESS_AFTER_STEPS:
            messages = _compress_messages(
                messages, state, papers_before, summaries_before, used_queries
            )

        try:
            thought, tool_calls = call_llm_with_tools(
                "planning",
                messages,
                litellm_tools,
                max_tokens=2000,
            )
        except Exception as e:
            print(f"  [SubResearcher:{label}] step={step} llm error: {e}")
            break

        asst_msg: dict[str, Any] = {"role": "assistant", "content": thought or ""}
        if tool_calls:
            asst_msg["tool_calls"] = tool_calls
        messages.append(asst_msg)

        if not tool_calls:
            print(f"  [SubResearcher:{label}] step={step} — no more tools, stopping")
            break

        # ── Execute all tool calls ─────────────────────────────────────────────
        tool_result_msgs: list[dict[str, Any]] = []
        for tc in tool_calls:
            tool_name = str(tc.get("function", {}).get("name", "") or "")
            args_raw = tc.get("function", {}).get("arguments", "{}")
            try:
                tool_input = json.loads(args_raw)
                if not isinstance(tool_input, dict):
                    tool_input = {}
            except Exception:
                tool_input = {}

            # Track queries for rolling summary
            if tool_name in ("search_papers", "arxiv_search", "paper_search"):
                q = str(tool_input.get("query", "") or "")
                if q:
                    used_queries.append(q)

            spec = get_tool_by_name(tool_name)
            if spec is None:
                obs_str = json.dumps({"error": f"unknown tool: {tool_name}"})
            else:
                try:
                    output = spec.fn(tool_input, state)
                    state = spec.state_updater(state, output)
                    obs_str = _compact_obs(tool_name, output)
                except Exception as e:
                    obs_str = json.dumps({"error": str(e)[:200]})

            raw_observations.append(f"[{tool_name}] {obs_str}")
            tool_result_msgs.append({
                "role": "tool",
                "tool_call_id": str(tc.get("id", "") or "unknown"),
                "name": tool_name,
                "content": obs_str,
            })

        messages.extend(tool_result_msgs)

        total_found = len(state.get("found_papers", []) or [])
        total_read = len(state.get("paper_summaries", []) or [])
        new_read = total_read - summaries_before
        new_found = total_found - papers_before
        print(f"  [SubResearcher:{label}] step={step} +papers={new_found} +summaries={new_read}")

        # ── Mandatory status injection (mirrors Executor status_msg) ──────────
        if total_found > 0 and total_read == summaries_before:
            # Papers found but nothing read yet — force the agent to read
            status = (
                f"[SYSTEM step={step}/{max_steps}] "
                f"found={total_found} papers, read=0 summaries. "
                f"You MUST call read_papers immediately — no more searches until you read."
            )
        else:
            status = (
                f"[SYSTEM step={step}/{max_steps}] "
                f"found={total_found}, read={total_read} summaries."
            )
        messages.append({"role": "user", "content": status})

        # ── TAOR: Reflect ─────────────────────────────────────────────────────
        step_gap = {
            "tool_count": len(tool_calls),
            "new_papers": new_found,
            "new_summaries": new_read,
            "step_errors": 0,
            "done_called": False,
            "total_found": len(state.get("found_papers", []) or []),
            "total_read": total_read,
        }
        reflect = _reflect_step(
            step=step,
            thought=thought or "",
            action=", ".join(
                str(tc.get("function", {}).get("name", "")) for tc in tool_calls
            ),
            observation=raw_observations[-1] if raw_observations else "",
            previous_step_gap=step_gap,
            plan=["search", "read"],
            used_errors=0,
            max_errors=3,
        )
        if reflect.get("reflection"):
            print(f"  [SubResearcher:{label}] R: {str(reflect['reflection'])[:120]}")

        if total_read >= 6:
            print(f"  [SubResearcher:{label}] sufficient ({total_read} summaries), stopping")
            break

        if reflect.get("should_finish") or reflect.get("should_handoff"):
            stop_r = str(reflect.get("stop_reason", "") or "reflect_finish")
            print(f"  [SubResearcher:{label}] reflect says stop: {stop_r}")
            break

    # ── Extract structured Evidence ────────────────────────────────────────────
    new_summaries = list(state.get("paper_summaries", []) or [])[summaries_before:]
    new_found_papers = list(state.get("found_papers", []) or [])[papers_before:]

    compressed_text, evidence_list = _extract_evidence(
        raw_observations, research_topic, new_summaries, topic_label=label
    )

    print(
        f"  [SubResearcher:{label}] done — "
        f"{steps_used} steps, {len(new_found_papers)} found, "
        f"{len(new_summaries)} summaries, {len(evidence_list)} evidence cards"
    )

    return {
        "compressed_evidence": compressed_text,
        "evidence": evidence_list,
        "paper_summaries": new_summaries,
        "found_papers": new_found_papers,
        "steps_used": steps_used,
    }
