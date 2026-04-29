from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SkillSpec:
    name: str
    required_inputs: list[str]
    output_signals: list[str]
    estimated_cost: float
    failure_policy: str
    description: str = ""


class SkillRegistry:
    def __init__(self):
        self._skills: dict[str, SkillSpec] = {}
        self._order: list[str] = []

    def register(self, spec: SkillSpec):
        self._skills[spec.name] = spec
        if spec.name not in self._order:
            self._order.append(spec.name)

    def get(self, name: str) -> SkillSpec | None:
        return self._skills.get(name)

    def names(self) -> list[str]:
        return [n for n in self._order if n in self._skills]

    def all(self) -> list[SkillSpec]:
        return [self._skills[n] for n in self.names()]


_DEFAULT_REGISTRY = SkillRegistry()


_DEFAULT_SPECS = [
    SkillSpec("search", ["selected_question|raw_input"], ["found_papers"], 0.35, "retry_then_end", "Search academic sources for the current topic."),
    SkillSpec("evidence_evaluate", ["found_papers|paper_summaries"], ["evidence_quality"], 0.02, "continue", "Evaluate evidence coverage and reliability."),
    SkillSpec("query_optimize", ["raw_input|selected_question", "evidence_quality"], ["optimized_queries"], 0.08, "fallback_to_search", "Generate better retrieval queries when evidence is weak."),
    SkillSpec("read", ["found_papers"], ["paper_summaries"], 0.45, "skip_failed_items", "Read and summarize retrieved papers."),
    SkillSpec("github", ["key_paper"], ["github_repo_url"], 0.15, "continue_without_repo", "Find implementation repositories related to the topic."),
    SkillSpec("tool_call", ["raw_input|selected_question"], ["assistant_response|artifacts"], 0.22, "continue", "Execute general tools for non-paper tasks."),
    SkillSpec("write", ["paper_summaries"], ["draft_md_path|draft"], 0.50, "retry_section", "Draft the user-facing report or article."),
    SkillSpec("critic", ["draft"], ["critique_score"], 0.25, "bounded_revisions", "Review draft quality and recommend the next revision move."),
]


def register_default_skills() -> SkillRegistry:
    if _DEFAULT_REGISTRY.names():
        return _DEFAULT_REGISTRY
    for spec in _DEFAULT_SPECS:
        _DEFAULT_REGISTRY.register(spec)
    return _DEFAULT_REGISTRY


def get_default_registry() -> SkillRegistry:
    return register_default_skills()
