"""
TraceLogger — 结构化 CoT 可观测性

每个 Agent 在执行前后调用此模块，记录：
  - Thought（推理过程）
  - Action（执行的操作）
  - Observation（执行结果摘要）
  - Duration（耗时）

输出到：
  - console（可选，受 --verbose 控制）
  - output/trace.jsonl（持久化，每行一条 JSON 记录）
"""

from __future__ import annotations
import json
import os
import time
from datetime import datetime
from typing import Any

_TRACE_PATH = "./output/trace.jsonl"
_verbose = False


def set_verbose(v: bool):
    global _verbose
    _verbose = v


class StepTracer:
    """
    上下文管理器，在 with 块内自动计时并记录 trace。

    用法：
        with StepTracer("search", thought="需要检索 LLM inference 相关论文") as t:
            result = search_node(state)
            t.observe(f"找到 {len(result['found_papers'])} 篇论文")
    """

    def __init__(self, stage: str, thought: str = "", step: int = 0):
        self.stage = stage
        self.thought = thought
        self.step = step
        self._start: float = 0.0
        self._observation: str = ""
        self._status: str = "success"

    def observe(self, msg: str):
        self._observation = msg

    def error(self, msg: str):
        self._observation = msg
        self._status = "error"

    def __enter__(self) -> "StepTracer":
        self._start = time.time()
        if _verbose and self.thought:
            print(f"\n  💭 [{self.stage}] {self.thought}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self._start
        if exc_type is not None:
            self._observation = f"Exception: {exc_val}"
            self._status = "error"

        record = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "step": self.step,
            "stage": self.stage,
            "thought": self.thought,
            "observation": self._observation,
            "status": self._status,
            "duration_s": round(duration, 2),
        }

        if _verbose:
            icon = "✓" if self._status == "success" else "✗"
            print(f"  {icon} [{self.stage}] {self._observation[:100]} ({duration:.1f}s)")

        _append_trace(record)
        return False  # 不吞异常


def _append_trace(record: dict):
    os.makedirs(os.path.dirname(_TRACE_PATH) or ".", exist_ok=True)
    with open(_TRACE_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def log_thought(stage: str, thought: str, step: int = 0):
    """快速记录一条 Thought（无 Action/Observation）。"""
    if _verbose:
        print(f"\n  💭 [{stage}] {thought}")
    _append_trace({
        "ts": datetime.utcnow().isoformat() + "Z",
        "step": step,
        "stage": stage,
        "thought": thought,
        "observation": "",
        "status": "thought",
        "duration_s": 0,
    })


def read_trace(path: str = _TRACE_PATH) -> list[dict]:
    """读取 trace 文件，返回所有记录列表。"""
    if not os.path.exists(path):
        return []
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def summarize_trace(path: str = _TRACE_PATH) -> str:
    """生成 trace 摘要（用于 eval report）。"""
    records = read_trace(path)
    if not records:
        return "无 trace 记录"
    total_steps = len([r for r in records if r.get("status") != "thought"])
    errors = [r for r in records if r.get("status") == "error"]
    total_time = sum(r.get("duration_s", 0) for r in records)
    lines = [
        f"总步骤：{total_steps}，错误：{len(errors)}，总耗时：{total_time:.1f}s",
    ]
    for r in records[-5:]:  # 最近 5 条
        lines.append(
            f"  [{r.get('stage', '?')}] {r.get('observation', '')[:60]} ({r.get('status', '?')})"
        )
    return "\n".join(lines)
