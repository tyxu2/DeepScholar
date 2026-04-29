"""
ToolRegistry — 全局工具注册表。

用法：
    # 注册工具（在工具文件顶层使用装饰器）
    @register_tool
    class MyTool(BaseTool):
        name = "my_tool"
        ...

    # 查询工具
    tool = get_tool("my_tool")
    result = tool.run(param="value")

    # 列出所有工具
    specs = get_all_specs("openai")   # or "mcp", "anthropic"
"""

from __future__ import annotations
from typing import Type
from research_agent.tools.base import BaseTool

# 全局注册表：tool_name → BaseTool 实例
_REGISTRY: dict[str, BaseTool] = {}


def register_tool(cls: Type[BaseTool]) -> Type[BaseTool]:
    """类装饰器：实例化工具类并注册到全局注册表。"""
    instance = cls()
    if not instance.name:
        raise ValueError(f"{cls.__name__} must define a non-empty `name`")
    _REGISTRY[instance.name] = instance
    return cls


def get_tool(name: str) -> BaseTool:
    """根据名称获取工具实例，找不到时抛出 KeyError。"""
    if name not in _REGISTRY:
        raise KeyError(f"Tool '{name}' not registered. Available: {list(_REGISTRY)}")
    return _REGISTRY[name]


def list_tools() -> list[BaseTool]:
    """返回所有已注册工具的列表。"""
    return list(_REGISTRY.values())


def get_tools_by_names(names: list[str]) -> list[BaseTool]:
    """根据名称列表批量获取工具（跳过未注册的）。"""
    return [_REGISTRY[n] for n in names if n in _REGISTRY]


def get_all_specs(format: str = "openai") -> list[dict]:
    """
    获取所有工具的规范描述，供 LLM 调用。

    Args:
        format: "openai" | "mcp" | "anthropic"
    """
    tools = list_tools()
    if format == "mcp":
        return [t.to_mcp_spec() for t in tools]
    elif format == "anthropic":
        return [t.to_anthropic_spec() for t in tools]
    else:
        return [t.to_openai_spec() for t in tools]


def get_specs_by_names(names: list[str], format: str = "openai") -> list[dict]:
    """获取指定工具子集的规范描述。"""
    tools = get_tools_by_names(names)
    if format == "mcp":
        return [t.to_mcp_spec() for t in tools]
    elif format == "anthropic":
        return [t.to_anthropic_spec() for t in tools]
    else:
        return [t.to_openai_spec() for t in tools]


def registry_summary() -> str:
    """返回注册表摘要，用于调试和日志。"""
    lines = [f"  {name}: {tool.description[:60]}" for name, tool in _REGISTRY.items()]
    return f"已注册工具 ({len(_REGISTRY)} 个):\n" + "\n".join(lines)
