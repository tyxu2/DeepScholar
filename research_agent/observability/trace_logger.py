"""
TraceLogger — lightweight per-run observability.

Borrows from "Observability / Tracing" in production agent runtimes:
  - trace_id per run (session_id + timestamp)
  - step-level events with agent / action / tokens / latency / status
  - timeline.jsonl for human-readable post-run analysis
  - token_budget accumulated across all agents in the run

Design choices:
  - No external deps (no OpenTelemetry, no Langsmith) — just JSON to disk
  - Thread-safe for parallel SubResearchers (threading.Lock on write)
  - Global singleton per session_id, created by run_react_pipeline
  - Agents call logger.event(...) — one line, no boilerplate

Usage:
    from research_agent.observability.trace_logger import get_logger, init_logger

    init_logger(session_id="default")          # called once at pipeline start
    logger = get_logger()

    with logger.span("supervisor", "conduct_research") as span:
        result = do_work()
        span.set_tokens(prompt=1200, completion=300)
        span.set_meta(topics=["ZeRO", "tensor parallel"])

    # Or fire-and-forget:
    logger.event("sub_researcher", "compress_done", evidence_count=4, steps=5)
"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Generator, Optional


# ── Token budget accumulator ──────────────────────────────────────────────────

@dataclass
class TokenBudget:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    call_count: int = 0

    def add(self, prompt: int, completion: int) -> None:
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.call_count += 1

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "call_count": self.call_count,
        }


# ── Span context manager ───────────────────────────────────────────────────────

class Span:
    """Tracks a single timed operation. Used via logger.span()."""

    def __init__(self, logger: "TraceLogger", agent: str, action: str):
        self._logger = logger
        self._agent = agent
        self._action = action
        self._start = time.monotonic()
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._meta: dict[str, Any] = {}
        self._status = "ok"
        self._error: str = ""

    def set_tokens(self, prompt: int = 0, completion: int = 0) -> None:
        self._prompt_tokens = prompt
        self._completion_tokens = completion

    def set_meta(self, **kwargs: Any) -> None:
        self._meta.update(kwargs)

    def fail(self, error: str) -> None:
        self._status = "error"
        self._error = error[:300]

    def finish(self) -> None:
        latency_ms = int((time.monotonic() - self._start) * 1000)
        self._logger.event(
            agent=self._agent,
            action=self._action,
            status=self._status,
            latency_ms=latency_ms,
            prompt_tokens=self._prompt_tokens,
            completion_tokens=self._completion_tokens,
            error=self._error or None,
            **self._meta,
        )
        if self._prompt_tokens or self._completion_tokens:
            self._logger.budget.add(self._prompt_tokens, self._completion_tokens)


# ── Core logger ───────────────────────────────────────────────────────────────

class TraceLogger:
    """
    Per-session trace logger. Thread-safe.

    Writes two files:
      output/{session_id}/timeline.jsonl  — one JSON event per line
      (token totals emitted as the last event on flush())
    """

    def __init__(self, session_id: str, output_dir: str = "./output"):
        self.session_id = session_id
        self.trace_id = f"{session_id}-{uuid.uuid4().hex[:8]}"
        self.budget = TokenBudget()
        self._lock = threading.Lock()
        self._path = os.path.join(output_dir, session_id, "timeline.jsonl")
        self._start_wall = time.time()
        os.makedirs(os.path.dirname(self._path), exist_ok=True)
        # Reset file at the start of each run
        with open(self._path, "w", encoding="utf-8") as f:
            f.write("")
        self.event("pipeline", "start", trace_id=self.trace_id)

    def event(
        self,
        agent: str,
        action: str,
        status: str = "ok",
        latency_ms: int = 0,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        error: Optional[str] = None,
        **meta: Any,
    ) -> None:
        record: dict[str, Any] = {
            "t": round(time.time() - self._start_wall, 3),
            "agent": agent,
            "action": action,
            "status": status,
        }
        if latency_ms:
            record["latency_ms"] = latency_ms
        if prompt_tokens or completion_tokens:
            record["tokens"] = {"prompt": prompt_tokens, "completion": completion_tokens}
        if error:
            record["error"] = error
        if meta:
            record.update(meta)

        line = json.dumps(record, ensure_ascii=False)
        with self._lock:
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    @contextmanager
    def span(self, agent: str, action: str) -> Generator[Span, None, None]:
        s = Span(self, agent, action)
        try:
            yield s
        except Exception as e:
            s.fail(str(e))
            raise
        finally:
            s.finish()

    def flush_summary(self, **extra: Any) -> None:
        """Write final token summary event. Call at pipeline end."""
        self.event(
            "pipeline",
            "end",
            **self.budget.to_dict(),
            **extra,
        )

    def token_summary(self) -> dict[str, Any]:
        return {"trace_id": self.trace_id, **self.budget.to_dict()}


# ── Global singleton (one per process, keyed by session_id) ───────────────────

_loggers: dict[str, TraceLogger] = {}
_global_lock = threading.Lock()
_default_logger: Optional[TraceLogger] = None


def init_logger(session_id: str, output_dir: str = "./output") -> TraceLogger:
    """Create (or replace) the logger for a session. Call at pipeline start."""
    global _default_logger
    logger = TraceLogger(session_id=session_id, output_dir=output_dir)
    with _global_lock:
        _loggers[session_id] = logger
        _default_logger = logger
    return logger


def get_logger(session_id: Optional[str] = None) -> TraceLogger:
    """
    Get the active logger. Returns a no-op logger if none is initialized,
    so agents don't need to guard against missing loggers.
    """
    with _global_lock:
        if session_id and session_id in _loggers:
            return _loggers[session_id]
        if _default_logger is not None:
            return _default_logger
    # No-op fallback — writes nothing, never crashes
    return _NoopLogger()


class _NoopLogger(TraceLogger):
    """Silent logger used when no session is active (tests, imports, etc.)."""

    def __init__(self) -> None:  # type: ignore[override]
        self.session_id = "noop"
        self.trace_id = "noop"
        self.budget = TokenBudget()
        self._lock = threading.Lock()

    def event(self, *args: Any, **kwargs: Any) -> None:
        pass

    def flush_summary(self, **extra: Any) -> None:
        pass
