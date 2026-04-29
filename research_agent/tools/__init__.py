"""
Tools package entry.

导入 builtin 模块以触发工具自动注册。
"""

from research_agent.tools.registry import (
    get_all_specs,
    get_specs_by_names,
    get_tool,
    get_tools_by_names,
    list_tools,
    register_tool,
    registry_summary,
)

# side-effect: register builtin tools
from . import builtin as _builtin  # noqa: F401

__all__ = [
    "register_tool",
    "get_tool",
    "list_tools",
    "get_all_specs",
    "get_specs_by_names",
    "get_tools_by_names",
    "registry_summary",
]
