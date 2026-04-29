from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

from research_agent.prompts import list_prompt_ids
from research_agent.react.skill_registry import get_default_registry
from research_agent.skills.catalog import list_skills
from research_agent.tools import list_tools


ResourceKind = Literal["agent", "tool", "prompt", "skill"]
LifecycleState = Literal["active", "experimental", "deprecated"]


@dataclass
class ResourceRecord:
    kind: ResourceKind
    name: str
    description: str
    version: str = "1.0"
    lifecycle: LifecycleState = "active"
    capabilities: list[str] = field(default_factory=list)
    source: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _agent_records() -> list[ResourceRecord]:
    records: list[ResourceRecord] = []
    for spec in get_default_registry().all():
        records.append(
            ResourceRecord(
                kind="agent",
                name=spec.name,
                description=f"Stage agent `{spec.name}` with failure policy `{spec.failure_policy}`.",
                capabilities=list(dict.fromkeys(spec.output_signals + spec.required_inputs)),
                source="research_agent.react.skill_registry",
            )
        )
    return records


def _tool_records() -> list[ResourceRecord]:
    records: list[ResourceRecord] = []
    for tool in list_tools():
        records.append(
            ResourceRecord(
                kind="tool",
                name=tool.name,
                description=tool.description,
                capabilities=sorted((tool.input_schema or {}).get("properties", {}).keys()),
                source=tool.__class__.__module__,
            )
        )
    return records


def _prompt_records() -> list[ResourceRecord]:
    return [
        ResourceRecord(
            kind="prompt",
            name=prompt_id,
            description=f"Prompt template `{prompt_id}`",
            capabilities=["template"],
            source="research_agent.prompts.defaults",
        )
        for prompt_id in list_prompt_ids()
    ]


def _skill_records() -> list[ResourceRecord]:
    return [
        ResourceRecord(
            kind="skill",
            name=name,
            description=desc,
            capabilities=["template"],
            source="research_agent.skills.catalog",
        )
        for name, desc in list_skills()
    ]


def build_resource_snapshot() -> list[dict]:
    records = _agent_records() + _tool_records() + _prompt_records() + _skill_records()
    records.sort(key=lambda item: (item.kind, item.name))
    return [record.to_dict() for record in records]


def render_resource_contract(resources: list[dict]) -> str:
    if not resources:
        return "(no registered resources)"

    lines: list[str] = []
    grouped: dict[str, list[dict]] = {}
    for resource in resources:
        grouped.setdefault(resource.get("kind", "unknown"), []).append(resource)

    for kind in ("agent", "tool", "prompt", "skill"):
        items = grouped.get(kind, [])
        if not items:
            continue
        lines.append(f"[{kind.upper()}]")
        for item in items:
            caps = ", ".join(item.get("capabilities", [])[:6])
            suffix = f" | caps: {caps}" if caps else ""
            lines.append(f"- {item.get('name', '')}: {item.get('description', '')}{suffix}")
        lines.append("")
    return "\n".join(lines).strip()
