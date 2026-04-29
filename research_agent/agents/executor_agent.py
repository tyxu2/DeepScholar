"""
Parallel Rollout Executor Agent

Fan-out strategy (inspired by DeepResearch):
  - Run `executor_rollouts` independent ReAct loops in parallel
  - Each rollout: independent messages / tool_calls / trace / stop_reason / score
  - Slight temperature offset per rollout to avoid full homogenization
  - If base rollouts show high uncertainty, one extra rollout is added automatically

Hot path:
  Brain → run_parallel_executor → Analyzer → Critic → Writer
"""
from __future__ import annotations

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from research_agent.llm.router import call_llm_json, call_llm_with_tools, system
from research_agent.prompts import render_prompt
from research_agent.protocols import build_resource_snapshot, render_resource_contract
from research_agent.react.tools import get_tools_for_plan, get_tool_by_name, to_litellm_tool
from research_agent.state import ExecutorTurn, ResearchState
from research_agent.utils.json_parser import parse_json


_TRACE_WRITE_LOCK = threading.Lock()


# ── Rollout summary schema ─────────────────────────────────────────────────────

def _empty_rollout_summary(rollout_id: int) -> dict[str, Any]:
    return {
        "rollout_id": rollout_id,
        "steps_used": 0,
        "errors": 0,
        "done_called": False,
        "papers_found": 0,
        "papers_read": 0,
        "repo_url": "",
        "artifacts": [],
        "key_observations": [],
        "stop_reason": "not_started",
        "score": 0.0,
    }


def _score_rollout(summary: dict[str, Any]) -> float:
    base = float(summary.get("score", 0.0) or 0.0)
    base += min(int(summary.get("papers_read", 0) or 0), 8) * 0.3
    base += min(int(summary.get("papers_found", 0) or 0), 12) * 0.08
    if summary.get("repo_url"):
        base += 0.8
    if summary.get("done_called"):
        base += 0.5
    base -= min(int(summary.get("errors", 0) or 0), 5) * 0.7
    return round(base, 3)


def _compact_obs(tool_name: str, output: dict) -> str:
    """Compact observation string sent back to LLM as tool result."""
    obs: dict = {}
    for k, v in output.items():
        if k == "papers":
            obs["papers_count"] = len(v)
            obs["papers_sample"] = [p.get("title", "") for p in v[:3]]
        elif k == "summaries":
            obs["summaries_count"] = len(v)
            obs["summaries_sample"] = [s.get("title", "") for s in v[:3]]
        elif k == "repos":
            obs["repos_count"] = len(v)
            obs["repos_sample"] = [r.get("full_name", "") for r in v[:3]]
        else:
            obs[k] = v
    return json.dumps(obs, ensure_ascii=False)[:600]


def _get_agent_allowed_tools(state: ResearchState, agent_name: str) -> set[str] | None:
    caps = state.get("agent_capabilities", {}) or {}
    if not isinstance(caps, dict):
        return None
    agent_caps = caps.get(agent_name, {}) or {}
    if not isinstance(agent_caps, dict):
        return None
    allowed = agent_caps.get("allowed_tools")
    if not isinstance(allowed, list):
        return None
    names = {str(item).strip() for item in allowed if str(item).strip()}
    return names


def _trace_jsonl_path(session_id: str) -> str:
    sid = str(session_id or "default").strip() or "default"
    return os.path.join("./output", sid, "trace.jsonl")


def _reset_trace_jsonl(session_id: str):
    path = _trace_jsonl_path(session_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with _TRACE_WRITE_LOCK:
        with open(path, "w", encoding="utf-8") as f:
            f.write("")


def _append_trace_jsonl(session_id: str, record: dict[str, Any]):
    path = _trace_jsonl_path(session_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    line = json.dumps(record, ensure_ascii=False)
    with _TRACE_WRITE_LOCK:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def _reflect_step(
    *,
    step: int,
    thought: str,
    action: str,
    observation: str,
    previous_step_gap: dict[str, Any] | None,
    plan: list[str],
    used_errors: int,
    max_errors: int,
) -> dict[str, Any]:
    """Mandatory TAOR reflection step; fail-closed when reflection is unavailable."""
    payload = {
        "step": step,
        "thought": thought[:300],
        "action": action,
        "observation": observation[:600],
        "previous_step_gap": previous_step_gap or {},
        "plan": plan,
        "error_budget": {"used": used_errors, "max": max_errors},
    }
    try:
        response = call_llm_json(
            "planning",
            [system(render_prompt("agents.executor.reflect", input_json=json.dumps(payload, ensure_ascii=False)))],
            max_tokens=10240,
        )
        data = parse_json(response)
        if not isinstance(data, dict):
            raise ValueError("reflect response is not a dict")
    except Exception as e:
        # Reflection failure must NOT stop the TAOR loop — continue searching
        return {
            "reflection": f"reflect_unavailable: {str(e)[:80]}",
            "progress": "partial",
            "next_focus": "continue_searching",
            "should_continue": True,
            "should_handoff": False,
            "should_finish": False,
            "stop_reason": "",
        }

    return {
        "reflection": str(data.get("reflection", "") or "").strip(),
        "progress": str(data.get("progress", "partial") or "partial").strip(),
        "next_focus": str(data.get("next_focus", "") or "").strip(),
        "should_continue": bool(data.get("should_continue", True)),
        "should_handoff": bool(data.get("should_handoff", False)),
        "should_finish": bool(data.get("should_finish", False)),
        "stop_reason": str(data.get("stop_reason", "") or "").strip(),
    }


# ── Single rollout ─────────────────────────────────────────────────────────────

def _run_single_rollout(
    base_state: ResearchState,
    rollout_id: int,
    rollout_bias: str,
    plan: list[str],
    max_steps: int,
    max_errors: int,
    temperature_offset: float,
    resource_contract: str,
    session_id: str,
) -> tuple[ResearchState, dict[str, Any], list[ExecutorTurn]]:
    """
    One independent TAOR (Thought→Action→Observation→Reflection) loop.
    Returns (updated_state_snapshot, rollout_summary, trace).
    """
    # Shallow copy — state_updaters always return new dicts so this is safe
    state: ResearchState = dict(base_state)  # type: ignore[assignment]

    active_tools = get_tools_for_plan(plan)
    executor_allowed = _get_agent_allowed_tools(state, "executor")
    if executor_allowed is not None:
        active_tools = [tool for tool in active_tools if tool.name in executor_allowed]
    if not active_tools:
        summary = {**_empty_rollout_summary(rollout_id), "stop_reason": "no_tools"}
        return state, summary, []

    litellm_tools = [to_litellm_tool(spec) for spec in active_tools]
    selected_topic = (state.get("selected_question", "") or state.get("raw_input", "")).strip()
    max_search = max(1, max_steps // 3)
    local_context = (state.get("local_context", "") or "")[:2000]
    research_brief_json = json.dumps(state.get("research_brief", {}) or {}, ensure_ascii=False, indent=2)
    executor_capability_json = json.dumps(
        ((state.get("agent_capabilities", {}) or {}).get("executor", {}) or {}),
        ensure_ascii=False,
        indent=2,
    )

    system_msg = render_prompt(
        "react.executor.system",
        goal=state.get("raw_input", ""),
        executor_goal=state.get("executor_goal", ""),
        selected_topic=selected_topic or "N/A",
        task_profile=state.get("task_profile", "general_research"),
        plan=" → ".join(plan),
        rollout_id=rollout_id,
        rollout_bias=rollout_bias,
        active_tools=", ".join(t.name for t in active_tools),
        research_brief_json=research_brief_json,
        executor_capability_json=executor_capability_json,
        local_context=local_context or "(none)",
        recent_history="(rollout start)",
        resource_contract=resource_contract,
        max_search=max_search,
    )

    tool_call_only = set(plan).issubset({"tool_call"})
    messages: list[dict] = [
        {"role": "system", "content": system_msg},
        {
            "role": "user",
            "content": (
                f"原始请求: {state.get('raw_input', '')}\n"
                f"当前主题: {selected_topic or state.get('raw_input', '')}\n"
                + (
                    "请优先调用通用工具完成任务，结束时必须调用 done。"
                    if tool_call_only
                    else "检索时请围绕当前主题构造英文查询词。"
                )
            ),
        },
    ]

    trace: list[ExecutorTurn] = []
    errors = 0
    done_called = False
    last_observation = ""
    stop_reason = "budget_exhausted"
    temperature = max(0.0, min(1.0, 0.2 + temperature_offset))

    for step in range(1, max_steps + 1):

        # ── LLM turn ──────────────────────────────────────────────────────────
        try:
            text_content, tool_calls = call_llm_with_tools(
                "planning", messages, litellm_tools, temperature=temperature
            )
        except Exception as e:
            errors += 1
            print(f"  [Rollout#{rollout_id}] step={step} LLM error: {e}")
            turn = {
                "step": step, "thought": "", "action": "llm_error",
                "observation": str(e), "reflection": "llm turn failed before action",
                "decision": "error", "status": "error",
            }
            trace.append(turn)
            _append_trace_jsonl(session_id, {
                "rollout_id": rollout_id,
                **turn,
                "plan": plan,
                "temperature": temperature,
            })
            if errors >= max_errors:
                stop_reason = "error_budget_exhausted"
                break
            continue

        assistant_msg: dict = {"role": "assistant", "content": text_content}
        if tool_calls:
            assistant_msg["tool_calls"] = tool_calls
        messages.append(assistant_msg)

        if not tool_calls:
            reflect = _reflect_step(
                step=step,
                thought=text_content,
                action="end_turn",
                observation="no more tools",
                previous_step_gap={
                    "reason": "no_tool_calls",
                    "tool_count": 0,
                    "new_papers": 0,
                    "new_summaries": 0,
                },
                plan=plan,
                used_errors=errors,
                max_errors=max_errors,
            )
            stop_reason = reflect.get("stop_reason") or "end_turn"
            turn = {
                "step": step,
                "thought": text_content,
                "action": "end_turn",
                "observation": "no more tools",
                "reflection": str(reflect.get("reflection", "") or ""),
                "decision": "finish",
                "status": "finish",
            }
            trace.append(turn)
            _append_trace_jsonl(session_id, {
                "rollout_id": rollout_id,
                **turn,
                "plan": plan,
                "temperature": temperature,
                "stop_reason": stop_reason,
            })
            break

        # ── Execute tools ──────────────────────────────────────────────────────
        tool_result_msgs: list[dict] = []
        step_observations: list[dict[str, Any]] = []
        papers_before = len(state.get("found_papers", []) or [])
        summaries_before = len(state.get("paper_summaries", []) or [])
        errors_before = errors

        if text_content:
            print(f"  [Rollout#{rollout_id}] T: {text_content[:160]}")

        for tc in tool_calls:
            tool_name: str = tc["function"]["name"]
            try:
                tool_input: dict = json.loads(tc["function"]["arguments"])
            except Exception:
                tool_input = {}

            print(f"  [Rollout#{rollout_id}] step={step} tool={tool_name}")

            spec = get_tool_by_name(tool_name)
            if spec is None:
                obs_str = json.dumps({"error": f"unknown tool: {tool_name}"})
                errors += 1
            else:
                try:
                    output = spec.fn(tool_input, state)
                    state = spec.state_updater(state, output)
                    obs_str = _compact_obs(tool_name, output)
                    print(f"    → {obs_str[:100]}")
                except Exception as e:
                    errors += 1
                    obs_str = json.dumps({"error": str(e)[:200]})
                    print(f"    ✗ {obs_str[:100]}")

            step_observations.append({
                "tool": tool_name,
                "tool_input": tool_input,
                "observation": obs_str,
            })
            tool_result_msgs.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "name": tool_name,
                "content": obs_str,
            })
            if tool_name == "done":
                done_called = True

        messages.extend(tool_result_msgs)
        last_observation = json.dumps(step_observations, ensure_ascii=False)

        total_found = len(state.get("found_papers", []) or [])
        total_read = len(state.get("paper_summaries", []) or [])

        # ── Inject per-step state snapshot so LLM knows cumulative progress ──
        hints: list[str] = []
        if total_found > 0 and total_read == 0:
            hints.append("⚠ found>0 且 read=0 → 下一步必须调用 read_papers，禁止 handoff")
        elif total_found > 0 and total_read < 4:
            hints.append(f"已读 {total_read}/{total_found} 篇，建议继续调用 read_papers 直到 ≥4 篇")
        status_msg = (
            f"[系统状态 step={step}/{max_steps}] "
            f"found={total_found}, read={total_read}, errors={errors}/{max_errors}"
            + (f" | {' | '.join(hints)}" if hints else "")
        )
        messages.append({"role": "user", "content": status_msg})

        step_gap = {
            "tool_count": len(tool_calls),
            "new_papers": total_found - papers_before,
            "new_summaries": total_read - summaries_before,
            "step_errors": errors - errors_before,
            "done_called": done_called,
            "total_found": total_found,
            "total_read": total_read,
        }

        reflect = _reflect_step(
            step=step,
            thought=text_content,
            action=", ".join(tc["function"]["name"] for tc in tool_calls),
            observation=last_observation,
            previous_step_gap=step_gap,
            plan=plan,
            used_errors=errors,
            max_errors=max_errors,
        )
        if reflect.get("reflection"):
            print(f"  [Rollout#{rollout_id}] R: {str(reflect['reflection'])[:160]}")
        should_handoff = bool(reflect.get("should_handoff", False))
        should_finish = bool(reflect.get("should_finish", False))
        should_continue = bool(reflect.get("should_continue", True))
        if done_called:
            decision = "finish"
        elif should_handoff:
            decision = "handoff"
        elif should_finish or not should_continue:
            decision = "finish"
        else:
            decision = "continue"

        turn = {
            "step": step,
            "thought": text_content,
            "action": ", ".join(tc["function"]["name"] for tc in tool_calls),
            "observation": last_observation,
            "reflection": str(reflect.get("reflection", "") or ""),
            "decision": decision,
            "status": "success" if errors == 0 else "error",
        }
        trace.append(turn)
        _append_trace_jsonl(session_id, {
            "rollout_id": rollout_id,
            **turn,
            "plan": plan,
            "temperature": temperature,
            "previous_step_gap": step_gap,
            "reflect_decision": {
                "should_continue": bool(reflect.get("should_continue", True)),
                "should_handoff": bool(reflect.get("should_handoff", False)),
                "should_finish": bool(reflect.get("should_finish", False)),
                "stop_reason": str(reflect.get("stop_reason", "") or ""),
            },
        })

        if errors >= max_errors:
            stop_reason = "error_budget_exhausted"
            break
        if done_called:
            stop_reason = "done_called"
            break
        if should_handoff:
            stop_reason = str(reflect.get("stop_reason", "") or "handoff")
            break
        if should_finish or not should_continue:
            stop_reason = str(reflect.get("stop_reason", "") or "reflect_finish")
            break

    # Build rollout summary
    summary: dict[str, Any] = {
        "rollout_id": rollout_id,
        "steps_used": len(trace),
        "errors": errors,
        "done_called": done_called,
        "papers_found": len(state.get("found_papers", [])),
        "papers_read": len(state.get("paper_summaries", [])),
        "repo_url": str(state.get("github_repo_url", "") or ""),
        "artifacts": [
            str(a.get("content_ref", ""))
            for a in (state.get("artifacts", []) or [])[:4]
        ],
        "key_observations": [
            str(t.get("observation", ""))[:120]
            for t in trace[-4:]
            if t.get("observation")
        ],
        "stop_reason": stop_reason,
        "score": 0.0,
    }
    summary["score"] = _score_rollout(summary)

    return state, summary, trace


# ── High-uncertainty check ─────────────────────────────────────────────────────

def _is_high_uncertainty(summaries: list[dict]) -> bool:
    if not summaries:
        return False
    total_papers = sum(s.get("papers_found", 0) for s in summaries)
    done_count = sum(1 for s in summaries if s.get("done_called"))
    total_errors = sum(s.get("errors", 0) for s in summaries)
    avg_score = sum(_score_rollout(s) for s in summaries) / len(summaries)
    return (
        total_papers == 0
        or avg_score < 2.0
        or (total_errors > len(summaries) * 2 and done_count == 0)
    )


# ── Evidence merger ────────────────────────────────────────────────────────────

def _merge_rollout_state(merged: ResearchState, rollout: ResearchState) -> ResearchState:
    """Union-merge evidence from a rollout snapshot into the accumulated state."""
    merged = dict(merged)  # type: ignore[assignment]

    # found_papers union
    existing = list(merged.get("found_papers", []))
    seen = {str(p.get("title", "")).lower() for p in existing}
    for p in rollout.get("found_papers", []) or []:
        key = str(p.get("title", "")).lower()
        if key and key not in seen:
            seen.add(key)
            existing.append(p)
    merged["found_papers"] = existing

    # paper_summaries union
    existing_s = list(merged.get("paper_summaries", []))
    seen_s = {str(s.get("title", "")).lower() for s in existing_s}
    for s in rollout.get("paper_summaries", []) or []:
        key = str(s.get("title", "")).lower()
        if key and key not in seen_s:
            seen_s.add(key)
            existing_s.append(s)
    merged["paper_summaries"] = existing_s

    # github_repo_url: first non-empty wins
    if not merged.get("github_repo_url") and rollout.get("github_repo_url"):
        merged["github_repo_url"] = rollout["github_repo_url"]

    # artifacts union by content_ref
    existing_a = list(merged.get("artifacts", []))
    seen_a = {str(a.get("content_ref", "")) for a in existing_a}
    for a in rollout.get("artifacts", []) or []:
        ref = str(a.get("content_ref", ""))
        if ref and ref not in seen_a:
            seen_a.add(ref)
            existing_a.append(a)
    merged["artifacts"] = existing_a

    # assistant_response: first non-empty
    if not merged.get("assistant_response") and rollout.get("assistant_response"):
        merged["assistant_response"] = rollout["assistant_response"]

    return merged  # type: ignore[return-value]


# ── Main entry ─────────────────────────────────────────────────────────────────

def run_parallel_executor(state: ResearchState) -> ResearchState:
    """
    Fan-out parallel rollout executor.

    Runs `executor_rollouts` independent ReAct loops (up to `executor_parallelism`
    concurrent) and merges evidence back into state.

    Outputs:
        rollout_summaries, selected_rollout_id, executor_trace,
        last_observation, last_reflection, stop_reason
    """
    plan = list(state.get("brain_plan", []) or [])
    if not plan:
        return state

    # Skip tool execution for write/critic-only plans
    exec_stages = set(plan) - {"write", "critic", "coding_plan"}
    if not exec_stages:
        return {**state, "stop_reason": "no_exec_stages", "current_stage": "executor"}

    constraints = state.get("user_constraints", {}) or {}
    deep_mode = bool(constraints.get("deep_research_mode", False))
    n_rollouts = max(1, int(state.get("executor_rollouts", 1) or 1))
    parallelism = max(1, int(state.get("executor_parallelism", 1) or 1))
    if deep_mode:
        n_rollouts = max(n_rollouts, int(constraints.get("executor_rollouts_deep", 3) or 3))
        parallelism = max(parallelism, int(constraints.get("executor_parallelism_deep", 2) or 2))
    budget = state.get("run_budget") or {"max_steps": 12, "max_errors": 4, "used_steps": 0, "used_errors": 0}
    max_steps = int(budget.get("max_steps", 12))
    max_errors = int(budget.get("max_errors", 4))
    if deep_mode:
        max_steps = max(max_steps, 18)
    session_id = str(state.get("session_id", "default") or "default")
    if int(budget.get("used_steps", 0) or 0) == 0:
        _reset_trace_jsonl(session_id)

    # Compute resource contract once (shared across rollouts)
    try:
        resource_contract = render_resource_contract(build_resource_snapshot()[:24])
    except Exception:
        resource_contract = "(resource snapshot unavailable)"

    # Rollout configs: (rollout_id, bias_label, temperature_offset)
    _BIASES = [
        ("neutral",      0.0),
        ("explore",      0.25),
        ("conservative", -0.1),
    ]
    rollout_configs = [
        (i, _BIASES[i % len(_BIASES)][0], _BIASES[i % len(_BIASES)][1])
        for i in range(n_rollouts)
    ]

    caller_ctx = state.get("_log_caller", "Executor")
    print(f"  [{caller_ctx}] fan-out: {n_rollouts} rollout(s) | parallelism={parallelism} | plan={plan}")

    rollout_summaries: list[dict] = []
    all_traces: list[ExecutorTurn] = []
    merged_state: ResearchState = dict(state)  # type: ignore[assignment]

    def _run(cfg: tuple) -> tuple:
        rid, bias, temp = cfg
        return _run_single_rollout(
            state, rid, bias, plan, max_steps, max_errors, temp, resource_contract, session_id
        )

    with ThreadPoolExecutor(max_workers=min(parallelism, n_rollouts)) as ex:
        futures = {ex.submit(_run, cfg): cfg for cfg in rollout_configs}
        for future in as_completed(futures):
            cfg = futures[future]
            try:
                rollout_state, summary, trace = future.result()
                rollout_summaries.append(summary)
                all_traces.extend(trace)
                merged_state = _merge_rollout_state(merged_state, rollout_state)
            except Exception as e:
                print(f"  [Executor Agent] rollout#{cfg[0]} exception: {e}")
                rollout_summaries.append(
                    {**_empty_rollout_summary(cfg[0]), "stop_reason": f"exception: {str(e)[:80]}"}
                )

    # Dynamic extra rollout: if uncertainty is high, run one more focused pass
    if _is_high_uncertainty(rollout_summaries):
        extra_id = n_rollouts
        print(f"  [Executor Agent] High uncertainty → extra rollout #{extra_id}")
        try:
            extra_state, extra_summary, extra_trace = _run_single_rollout(
                merged_state, extra_id, "explore", plan,
                max_steps, max_errors, 0.3, resource_contract, session_id,
            )
            rollout_summaries.append(extra_summary)
            all_traces.extend(extra_trace)
            merged_state = _merge_rollout_state(merged_state, extra_state)
        except Exception as e:
            print(f"  [Executor Agent] extra rollout failed: {e}")

    # Pick best rollout
    best_summary = (
        max(rollout_summaries, key=_score_rollout) if rollout_summaries else {}
    )
    best_rollout_id = int(best_summary.get("rollout_id", 0))
    stop_reason = str(best_summary.get("stop_reason", "budget_exhausted") or "budget_exhausted")

    # Extract last observation / reflection from trace
    last_obs = ""
    last_ref = ""
    for turn in reversed(all_traces):
        if not last_obs and turn.get("observation"):
            last_obs = str(turn["observation"])[:400]
        if not last_ref and turn.get("reflection"):
            last_ref = str(turn["reflection"])[:400]
        if last_obs and last_ref:
            break

    total_steps = sum(s.get("steps_used", 0) for s in rollout_summaries)

    print(
        f"  [Executor Agent] Done. rollouts={len(rollout_summaries)} "
        f"best=#{best_rollout_id} score={best_summary.get('score', 0):.2f} "
        f"stop={stop_reason}"
    )

    return {
        **merged_state,
        "rollout_summaries": rollout_summaries,
        "selected_rollout_id": best_rollout_id,
        "executor_trace": all_traces,
        "last_observation": last_obs,
        "last_reflection": last_ref,
        "stop_reason": stop_reason,
        "run_budget": {**budget, "used_steps": total_steps},
        "current_stage": "executor",
    }
