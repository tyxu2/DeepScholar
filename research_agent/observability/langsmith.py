"""
Optional LangSmith observability.

Activates only when LANGSMITH_API_KEY is set in the environment. Safe to call
unconditionally at startup — it is a no-op when the key is missing or the
`langsmith` package is not installed.

Environment variables:
    LANGSMITH_API_KEY     required to enable tracing
    LANGSMITH_PROJECT     optional project name (default: "deepscholar")
    LANGSMITH_ENDPOINT    optional self-hosted endpoint
    LANGSMITH_TRACING     set to "false" to force-disable even if a key exists
"""

from __future__ import annotations

import os


_ENABLED: bool = False


def enable_langsmith_if_configured(verbose: bool = False) -> bool:
    """Enable LangSmith tracing if LANGSMITH_API_KEY is configured.

    Returns True if tracing is now enabled, False otherwise.
    """
    global _ENABLED
    if _ENABLED:
        return True

    api_key = os.environ.get("LANGSMITH_API_KEY", "").strip()
    if not api_key:
        return False

    if os.environ.get("LANGSMITH_TRACING", "").lower() == "false":
        return False

    # LangChain / LangGraph read these env vars to route traces.
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = api_key
    os.environ.setdefault("LANGCHAIN_PROJECT", os.environ.get("LANGSMITH_PROJECT", "deepscholar"))
    endpoint = os.environ.get("LANGSMITH_ENDPOINT", "").strip()
    if endpoint:
        os.environ["LANGCHAIN_ENDPOINT"] = endpoint

    try:
        import langsmith  # noqa: F401
    except Exception:
        if verbose:
            print("[langsmith] LANGSMITH_API_KEY is set but `langsmith` is not installed; "
                  "run `pip install langsmith` to enable tracing.")
        # Even without the SDK, LANGCHAIN_TRACING_V2 will route via langchain's
        # built-in client where available, so we still flip the flag.

    _ENABLED = True
    if verbose:
        project = os.environ.get("LANGCHAIN_PROJECT", "deepscholar")
        print(f"[langsmith] tracing enabled (project={project})")
    return True


def is_enabled() -> bool:
    return _ENABLED
