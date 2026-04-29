"""
BaseTool — MCP 兼容的工具抽象基类。

每个工具实现三件事：
  1. 声明 name / description / input_schema（JSON Schema）
  2. 实现 run(**kwargs) -> str（同步执行）
  3. 可选：实现 arun(**kwargs) 异步版本

工具可被：
  - ReActExecutor 调用（Thought→Action→Observation 循环）
  - MCP Server 通过 stdio/HTTP 暴露给外部客户端
  - 任何 LLM provider 通过 tool_call / function_calling 调用
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any


class BaseTool(ABC):
    # 子类必须定义这三个类属性
    name: str = ""
    description: str = ""
    input_schema: dict = {}   # JSON Schema（type: object, properties, required）

    # ── 执行接口 ──────────────────────────────────────────────────────────
    @abstractmethod
    def run(self, **kwargs: Any) -> str:
        """同步执行工具，返回字符串结果（Observation）。"""
        ...

    async def arun(self, **kwargs: Any) -> str:
        """异步执行，默认包装同步版本。高 I/O 工具可覆盖此方法。"""
        return self.run(**kwargs)

    # ── 规范输出（供 MCP / OpenAI / Anthropic 使用）────────────────────────
    def to_mcp_spec(self) -> dict:
        """返回 MCP tools/list 格式的工具规范。"""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": {
                "type": "object",
                **self.input_schema,
            },
        }

    def to_openai_spec(self) -> dict:
        """返回 OpenAI function_calling 格式的工具规范。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    **self.input_schema,
                },
            },
        }

    def to_anthropic_spec(self) -> dict:
        """返回 Anthropic tool_use 格式的工具规范。"""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                **self.input_schema,
            },
        }

    def __repr__(self) -> str:
        return f"<Tool:{self.name}>"
