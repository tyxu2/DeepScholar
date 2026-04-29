from __future__ import annotations

from research_agent.state import ResearchState


def resolve_replan_next(state: ResearchState, reason: str) -> str:
    """
    根据反思/评审信号选择下一步。
    这里只做轻量规则判断，避免再引入第二个调度器。
    """
    reason_lower = (reason or "").lower()

    if any(k in reason_lower for k in ["引用不足", "缺乏文献", "需要更多论文", "paper", "evidence", "coverage"]):
        return "query_optimize"
    if any(k in reason_lower for k in ["证据质量", "source credibility", "reproducibility", "可信"]):
        return "query_optimize"
    if any(k in reason_lower for k in ["代码", "实现计划", "prompt", "codex", "claude"]):
        return "coding_plan"
    if any(k in reason_lower for k in ["仓库", "github", "repo"]):
        return "github"
    if any(k in reason_lower for k in ["重写", "结构", "写作", "draft", "section"]):
        return "write"

    plan = state.get("brain_plan", []) or []
    for stage in ("query_optimize", "github", "coding_plan", "write"):
        if stage in plan:
            return stage
    return "end"
