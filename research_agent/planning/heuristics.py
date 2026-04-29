from __future__ import annotations

from research_agent.state import ResearchState


def assess_difficulty(state: ResearchState) -> int:
    """
    轻量任务难度评估，只作为 Brain 的 seed planning 参考。
    """
    score = 0
    raw_input = (state.get("raw_input") or "").lower()
    doc_type = state.get("doc_type", "survey")
    task_profile = state.get("task_profile", "general_research")

    if doc_type == "experimental":
        score += 3
    elif doc_type == "survey":
        score += 2
    else:
        score += 1

    if task_profile == "paper_from_implementation":
        score += 2
    elif task_profile == "repo_research":
        score += 1
    elif task_profile == "math_reasoning":
        score += 2
    elif task_profile == "general_task":
        score += 1

    heavy_signals = ["数学推导", "derivation", "proof", "深入分析", "system design"]
    coding_plan_signals = ["implement", "代码", "code", "codex", "claude code", "工程计划", "prompt"]
    write_signals = ["综述", "survey", "写", "write", "report", "总结", "overview", "调研"]
    light_signals = ["找", "搜", "search", "find", "查", "看看有没有"]
    general_signals = ["文件", "file", "目录", "folder", "readme", "path", "url", "网页", "project", "workspace"]

    if any(w in raw_input for w in heavy_signals):
        score += 3
    elif any(w in raw_input for w in coding_plan_signals):
        score += 2
    elif any(w in raw_input for w in write_signals):
        score += 2
    elif any(w in raw_input for w in general_signals):
        score += 1
    elif any(w in raw_input for w in light_signals):
        score += 1
    else:
        score += 2

    words = len(raw_input.split())
    if words > 30:
        score += 2
    elif words > 15:
        score += 1

    return min(score, 9)


def recommend_plan(state: ResearchState) -> list[str]:
    difficulty = assess_difficulty(state)
    doc_type = state.get("doc_type", "survey")
    task_profile = state.get("task_profile", "general_research")

    if task_profile == "paper_from_implementation":
        return ["search", "read", "github", "coding_plan", "write", "critic"]
    if task_profile == "repo_research":
        return ["search", "read", "github", "coding_plan"]
    if task_profile == "math_reasoning":
        return ["search", "read", "write", "critic"]
    if task_profile == "general_task":
        return ["tool_call"]

    if difficulty <= 3:
        return ["search"]
    if difficulty <= 5:
        return ["search", "read"]
    if difficulty <= 7:
        if doc_type == "survey":
            return ["search", "read", "write", "critic"]
        return ["search", "read", "write"]
    return ["search", "read", "github", "coding_plan", "write", "critic"]
