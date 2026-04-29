"""
LLM Router

Key behavior:
1) Unified retries with exponential backoff.
2) Task-based model routing.
3) JSON helper with re-prompt on parse failure.
4) Tool-calling helper with message normalization.
"""

from __future__ import annotations

import os
import time

import litellm
from dotenv import load_dotenv
from litellm import completion

load_dotenv()

litellm.modify_params = True
litellm.suppress_debug_info = True

PRIMARY_MODEL = os.getenv("RESEARCH_AGENT_MODEL_PRIMARY", "gpt-5")
LIGHT_MODEL = os.getenv("RESEARCH_AGENT_MODEL_LIGHT", PRIMARY_MODEL)
SAFE_FALLBACK_MODEL = os.getenv("RESEARCH_AGENT_MODEL_FALLBACK", PRIMARY_MODEL)
COST_PROFILE = os.getenv("RESEARCH_AGENT_COST_PROFILE", "quality").strip().lower()
if COST_PROFILE not in {"quality", "balanced", "budget"}:
    COST_PROFILE = "quality"

default_planning_model = LIGHT_MODEL if COST_PROFILE in {"balanced", "budget"} else PRIMARY_MODEL
default_coding_model = LIGHT_MODEL if COST_PROFILE == "budget" else PRIMARY_MODEL
default_writing_model = PRIMARY_MODEL
default_critique_model = LIGHT_MODEL if COST_PROFILE == "budget" else PRIMARY_MODEL
default_tool_call_model = LIGHT_MODEL if COST_PROFILE in {"balanced", "budget"} else PRIMARY_MODEL

PLANNING_MODEL = os.getenv("RESEARCH_AGENT_MODEL_PLANNING", default_planning_model)
CODING_MODEL = os.getenv("RESEARCH_AGENT_MODEL_CODING", default_coding_model)
WRITING_MODEL = os.getenv("RESEARCH_AGENT_MODEL_WRITING", default_writing_model)
CRITIQUE_MODEL = os.getenv("RESEARCH_AGENT_MODEL_CRITIQUE", default_critique_model)
TOOL_CALL_MODEL = os.getenv("RESEARCH_AGENT_MODEL_TOOL_CALL", default_tool_call_model)
GPT5_REASONING_EFFORT = os.getenv("RESEARCH_AGENT_GPT5_REASONING_EFFORT", "low").strip().lower()
ALLOW_MODEL_DOWNGRADE = os.getenv("RESEARCH_AGENT_ALLOW_DOWNGRADE", "0").strip().lower() in {
    "1", "true", "yes", "on"
}

LLM_ROUTING: dict[str, str] = {
    "planning": PLANNING_MODEL,
    "coding": CODING_MODEL,
    "summarization": LIGHT_MODEL,
    "writing": WRITING_MODEL,
    "critique": CRITIQUE_MODEL,
    "fallback": SAFE_FALLBACK_MODEL,
}

if COST_PROFILE == "budget":
    LLM_MAX_TOKENS: dict[str, int] = {
        "planning": 4096,
        "coding": 4096,
        "summarization": 2500,
        "writing": 9000,
        "critique": 4000,
        "fallback": 4000,
    }
elif COST_PROFILE == "balanced":
    LLM_MAX_TOKENS = {
        "planning": 8192,
        "coding": 8192,
        "summarization": 4000,
        "writing": 12000,
        "critique": 6000,
        "fallback": 6000,
    }
else:
    LLM_MAX_TOKENS = {
        "planning": 16384,
        "coding": 16384,
        "summarization": 8000,
        "writing": 32000,
        "critique": 8000,
        "fallback": 8000,
    }

default_retries = 2 if COST_PROFILE == "budget" else 3
_MAX_RETRIES = max(1, int(os.getenv("RESEARCH_AGENT_MAX_RETRIES", str(default_retries)) or default_retries))
_BASE_DELAY = 2.0
_MAX_TOKEN_GROWTH_CAP = max(2048, int(os.getenv("RESEARCH_AGENT_MAX_TOKEN_CAP", "16384" if COST_PROFILE == "budget" else "32768") or 32768))


def _is_rate_limit_error(e: Exception) -> bool:
    msg = str(e).lower()
    return any(kw in msg for kw in ("rate limit", "ratelimit", "429", "too many requests"))


def _is_context_error(e: Exception) -> bool:
    msg = str(e).lower()
    return any(kw in msg for kw in ("context length", "maximum context", "token limit"))


def _is_unsupported_params_error(e: Exception) -> bool:
    msg = str(e).lower()
    return "unsupportedparamserror" in msg or "unsupported" in msg


def _is_schema_bad_request_error(e: Exception) -> bool:
    msg = str(e).lower()
    return "badrequesterror" in msg and "one of" in msg


def _is_output_limit_error(e: Exception) -> bool:
    msg = str(e).lower()
    return any(
        kw in msg
        for kw in (
            "could not finish the message because max_tokens",
            "model output limit was reached",
            "output limit was reached",
        )
    )


def _should_grow_tokens_for_choice(choice: object) -> bool:
    finish_reason = str(getattr(choice, "finish_reason", "") or "").lower()
    msg = getattr(choice, "message", None)
    content = getattr(msg, "content", "") if msg is not None else ""
    content_text = content if isinstance(content, str) else str(content or "")
    return finish_reason == "length" and not content_text.strip()


def _normalize_messages(messages: list[dict]) -> list[dict]:
    normalized = [m for m in (messages or []) if isinstance(m, dict)]
    if not normalized:
        return [{"role": "user", "content": "Please continue with the task."}]
    has_user = any(str(m.get("role", "")).lower() == "user" for m in normalized)
    if not has_user:
        normalized.append({"role": "user", "content": "Please continue with the task."})
    return normalized


def _sanitize_kwargs_for_model(model: str, kwargs: dict) -> dict:
    sanitized = dict(kwargs)
    if model.startswith("gpt-5"):
        for k in ("temperature", "top_p", "presence_penalty", "frequency_penalty"):
            sanitized.pop(k, None)
        if "reasoning_effort" not in sanitized and GPT5_REASONING_EFFORT in {"minimal", "low", "medium", "high"}:
            sanitized["reasoning_effort"] = GPT5_REASONING_EFFORT
    return sanitized


def call_llm(task: str, messages: list[dict], max_tokens: int | None = None, **kwargs) -> str:
    model = LLM_ROUTING.get(task, LLM_ROUTING["fallback"])
    effective_max_tokens = max_tokens or LLM_MAX_TOKENS.get(task, 1500)
    normalized_messages = _normalize_messages(messages)
    sanitized_kwargs = _sanitize_kwargs_for_model(model, kwargs)

    last_exc: Exception | None = None

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            response = completion(
                model=model,
                messages=normalized_messages,
                max_tokens=effective_max_tokens,
                **sanitized_kwargs,
            )
            choice = response.choices[0]
            if _should_grow_tokens_for_choice(choice) and attempt < _MAX_RETRIES:
                grown = min(max(effective_max_tokens * 2, effective_max_tokens + 512), _MAX_TOKEN_GROWTH_CAP)
                if grown > effective_max_tokens:
                    effective_max_tokens = grown
                    print(f"    [router] empty length-capped response; retry with max_tokens={effective_max_tokens}")
                    continue
            # Token accounting — non-fatal
            try:
                usage = getattr(response, "usage", None)
                if usage:
                    from research_agent.observability.trace_logger import get_logger
                    get_logger().budget.add(
                        int(getattr(usage, "prompt_tokens", 0) or 0),
                        int(getattr(usage, "completion_tokens", 0) or 0),
                    )
            except Exception:
                pass
            content = choice.message.content
            return content if isinstance(content, str) else str(content or "")
        except Exception as e:
            last_exc = e

            if _is_output_limit_error(e) and attempt < _MAX_RETRIES:
                grown = min(max(effective_max_tokens * 2, effective_max_tokens + 512), _MAX_TOKEN_GROWTH_CAP)
                if grown > effective_max_tokens:
                    effective_max_tokens = grown
                    print(f"    [router] output capped; retry with max_tokens={effective_max_tokens}")
                    continue

            should_downgrade = ALLOW_MODEL_DOWNGRADE and model != LLM_ROUTING["fallback"]
            if should_downgrade and (_is_unsupported_params_error(e) or _is_schema_bad_request_error(e) or _is_context_error(e)):
                model = LLM_ROUTING["fallback"]
                sanitized_kwargs = _sanitize_kwargs_for_model(model, {})
                print(f"    [router] downgrade to fallback model: {model}")
                continue

            if attempt < _MAX_RETRIES:
                delay = _BASE_DELAY * (3 ** attempt) if _is_rate_limit_error(e) else _BASE_DELAY * attempt
                print(f"    [router] call_llm failed ({attempt}/{_MAX_RETRIES}): {str(e)[:120]}, retry in {delay:.0f}s")
                time.sleep(delay)
            else:
                raise

    raise last_exc  # type: ignore[misc]


def call_llm_json(task: str, messages: list[dict], max_retries: int = 2, **kwargs) -> str:
    from research_agent.utils.json_parser import parse_json

    for attempt in range(1, max_retries + 1):
        content = call_llm(task, messages, **kwargs)
        try:
            parse_json(content)
            return content
        except Exception:
            if attempt < max_retries:
                retry_msg = user("Please return strict JSON only, without markdown code blocks.")
                messages = messages + [assistant(content), retry_msg]
            else:
                return content


def call_llm_with_tools(
    task: str,
    messages: list[dict],
    tools: list[dict],
    max_tokens: int | None = None,
    **kwargs,
) -> tuple[str, list[dict]]:
    if not tools:
        return call_llm(task, messages, max_tokens=max_tokens, **kwargs), []

    model = TOOL_CALL_MODEL or LLM_ROUTING.get(task, LLM_ROUTING["fallback"])
    effective_max_tokens = max_tokens or 2000
    normalized_messages = _normalize_messages(messages)
    sanitized_kwargs = _sanitize_kwargs_for_model(model, kwargs)

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            response = completion(
                model=model,
                messages=normalized_messages,
                tools=tools,
                max_tokens=effective_max_tokens,
                **sanitized_kwargs,
            )
            choice = response.choices[0]
            msg = choice.message
            msg_tool_calls = getattr(msg, "tool_calls", None)
            if _should_grow_tokens_for_choice(choice) and not msg_tool_calls and attempt < _MAX_RETRIES:
                grown = min(max(effective_max_tokens * 2, effective_max_tokens + 512), _MAX_TOKEN_GROWTH_CAP)
                if grown > effective_max_tokens:
                    effective_max_tokens = grown
                    print(f"    [router] tools empty length-capped response; retry with max_tokens={effective_max_tokens}")
                    continue

            # Token accounting — non-fatal
            try:
                usage = getattr(response, "usage", None)
                if usage:
                    from research_agent.observability.trace_logger import get_logger
                    get_logger().budget.add(
                        int(getattr(usage, "prompt_tokens", 0) or 0),
                        int(getattr(usage, "completion_tokens", 0) or 0),
                    )
            except Exception:
                pass
            content: str = msg.content or ""
            tool_calls: list[dict] = []
            if msg_tool_calls:
                for tc in msg_tool_calls:
                    tool_calls.append(
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                    )
            return content, tool_calls
        except Exception as e:
            if _is_output_limit_error(e) and attempt < _MAX_RETRIES:
                grown = min(max(effective_max_tokens * 2, effective_max_tokens + 512), _MAX_TOKEN_GROWTH_CAP)
                if grown > effective_max_tokens:
                    effective_max_tokens = grown
                    print(f"    [router] tools output capped; retry with max_tokens={effective_max_tokens}")
                    continue

            should_downgrade = ALLOW_MODEL_DOWNGRADE and model != LLM_ROUTING["fallback"]
            if should_downgrade and (_is_unsupported_params_error(e) or _is_schema_bad_request_error(e) or _is_context_error(e)):
                model = LLM_ROUTING["fallback"]
                sanitized_kwargs = _sanitize_kwargs_for_model(model, {})
                print(f"    [router] tools downgrade to fallback model: {model}")
                continue

            if attempt < _MAX_RETRIES:
                delay = _BASE_DELAY * (3 ** attempt) if _is_rate_limit_error(e) else _BASE_DELAY * attempt
                print(f"    [router] call_llm_with_tools failed ({attempt}/{_MAX_RETRIES}): {str(e)[:120]}, retry in {delay:.0f}s")
                time.sleep(delay)
            else:
                raise

    return "", []


def system(content: str) -> dict:
    return {"role": "system", "content": content}


def user(content: str) -> dict:
    return {"role": "user", "content": content}


def assistant(content: str) -> dict:
    return {"role": "assistant", "content": content}
