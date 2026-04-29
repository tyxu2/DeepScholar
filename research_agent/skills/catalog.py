from __future__ import annotations

from typing import Dict


SKILLS: Dict[str, dict] = {
    "literature_scout": {
        "description": "快速侦察某主题的代表性论文、方法分支和空白点。",
        "template": (
            "任务：围绕主题“{topic}”做文献侦察。\n"
            "要求：\n"
            "1) 给出 3-5 个主要研究分支\n"
            "2) 每个分支列出代表论文线索与关键词\n"
            "3) 总结 3 个可切入研究空白\n"
            "附加约束：{constraints}\n"
        ),
    },
    "llm_infra_review": {
        "description": "生成 LLM Infra 方向的结构化综述提纲或一段式总结。",
        "template": (
            "任务：撰写 LLM Infra 研究综述，主题“{topic}”。\n"
            "要求：\n"
            "1) 覆盖训练、推理、系统优化、评测四个面向\n"
            "2) 给出趋势判断与挑战\n"
            "3) 输出可直接给写作 agent 的提纲或段落\n"
            "附加约束：{constraints}\n"
        ),
    },
    "multi_agent_plan": {
        "description": "生成通用 Multi-Agent 执行方案（Plan/Execute/Evaluate）。",
        "template": (
            "任务：为“{topic}”设计 Multi-Agent 执行方案。\n"
            "要求：\n"
            "1) 定义角色分工、输入输出契约\n"
            "2) 给出 ReAct 路由与错误恢复策略\n"
            "3) 给出质量评估指标与人工介入点\n"
            "附加约束：{constraints}\n"
        ),
    },
}


def list_skills() -> list[tuple[str, str]]:
    return [(k, v.get("description", "")) for k, v in SKILLS.items()]


def get_skill(name: str) -> dict:
    if name not in SKILLS:
        raise KeyError(f"skill '{name}' not found")
    return SKILLS[name]


def render_skill(name: str, topic: str, constraints: str = "") -> str:
    skill = get_skill(name)
    tpl = skill.get("template", "")
    return tpl.format(topic=topic, constraints=constraints or "无")
