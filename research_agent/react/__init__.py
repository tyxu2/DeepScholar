from __future__ import annotations

from research_agent.state import ResearchState


def run_react_pipeline(initial_state: ResearchState, clarification_callback=None) -> ResearchState:
    # 延迟导入，避免 brain_agent -> react.skill_registry 时触发循环导入
    from research_agent.react.executor import run_react_pipeline as _run_react_pipeline

    return _run_react_pipeline(initial_state, clarification_callback=clarification_callback)


__all__ = ["run_react_pipeline"]
