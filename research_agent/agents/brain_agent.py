from __future__ import annotations

import json
import hashlib
import os
import re

from research_agent.llm.router import call_llm_json, system
from research_agent.memory.conversation_memory import ConversationMemory
from research_agent.planning import assess_difficulty, recommend_plan
from research_agent.prompts import render_prompt
from research_agent.react.skill_registry import get_default_registry
from research_agent.react.tools import get_tools_for_plan
from research_agent.state import ResearchState, TaskItem
from research_agent.utils.json_parser import parse_json

PLANNABLE_STAGES = get_default_registry().names()
PLANNABLE_SET = set(PLANNABLE_STAGES)
RESEARCH_BRIEF_THRESHOLD = 0.95
BRIEF_GATE_ENV = "RESEARCH_AGENT_BRIEF_GATE"


def _build_tasks_from_plan(plan: list[str], topic: str) -> list[TaskItem]:
    tasks: list[TaskItem] = []
    for i, stage in enumerate(plan, 1):
        depends = [f"task-{i - 1}"] if i > 1 else []
        tasks.append(
            TaskItem(
                id=f"task-{i}",
                stage=stage,
                goal=f"{stage} for topic: {topic[:120]}",
                status="pending",
                depends_on=depends,
            )
        )
    return tasks


def _should_clarify_only(raw_input: str) -> bool:
    """判断是否应进入澄清模式（寒暄/信息不足），避免误触发重流程。"""
    text = (raw_input or "").strip()
    if not text:
        return True
    lower = text.lower()
    capability_patterns = [
        "你会", "你能", "can you", "are you able", "what can you do",
        "你可以做什么", "你能做什么", "会写代码吗", "写代码吗", "你是谁",
    ]
    if any(p in lower for p in capability_patterns):
        return True
    greetings = {
        "hi", "hello", "hey", "yo", "你好", "您好", "在吗", "嗨", "早", "早上好", "晚上好",
    }
    if lower in greetings:
        return True
    # 过短且无研究意图词，默认先澄清
    if len(text) <= 8:
        intent_tokens = [
            "research", "paper", "survey", "综述", "论文", "找", "搜", "review", "report",
            "llm", "infra", "github", "代码", "tex", "latex",
        ]
        if not any(tok in lower for tok in intent_tokens):
            return True
    return False


def _extract_memory_topic(context_package: dict) -> str:
    guard = (context_package or {}).get("guard", {}) or {}
    pinned_goal = str(guard.get("pinned_goal", "") or "").strip()
    if pinned_goal:
        return pinned_goal

    facts = ((context_package or {}).get("long_memory", {}) or {}).get("facts", []) or []
    for fact in reversed(facts):
        text = str(fact or "")
        if text.startswith("latest_topic="):
            return text.split("=", 1)[1].strip()
    return ""


def _extract_fact_value(context_package: dict, key: str) -> str:
    facts = ((context_package or {}).get("long_memory", {}) or {}).get("facts", []) or []
    prefix = f"{key}="
    for fact in reversed(facts):
        text = str(fact or "")
        if text.startswith(prefix):
            return text.split("=", 1)[1].strip()
    return ""


def _is_followup_instruction(raw_input: str) -> bool:
    text = (raw_input or "").strip().lower()
    if not text:
        return False
    followup_tokens = [
        "生成", "写", "改", "继续", "按这个", "基于这个", "这个主题", "上面", "这一版",
        "generate", "write", "revise", "continue", "based on", "this topic", "same topic",
    ]
    return any(t in text for t in followup_tokens)


def _detect_edit_existing_markdown_request(raw_input: str) -> dict:
    text = (raw_input or "").strip().lower()
    md_paths = re.findall(r"[\w./-]+\.md", raw_input or "")
    source_md_path = md_paths[0] if md_paths else "./output/paper.md"
    edit_tokens = ["改", "修改", "重写", "rewrite", "revise", "edit", "polish", "整理", "转成", "变成"]
    md_tokens = [".md", "paper md", "paper.md", "markdown", "md文件", "md 文档"]
    is_edit = any(t in text for t in edit_tokens) and any(t in text for t in md_tokens)
    return {
        "is_edit_existing_markdown": is_edit,
        "source_markdown_path": source_md_path,
    }


def _infer_constraints(raw_input: str) -> dict:
    text = raw_input.lower()
    edit_detect = _detect_edit_existing_markdown_request(raw_input)
    deep_tokens = [
        "deep", "deeper", "systematic", "comprehensive", "thorough",
        "深度", "深入", "系统综述", "系统性", "全面", "详细", "深挖", "深研",
    ]
    deep_mode = any(k in text for k in deep_tokens)
    constraints = {
        "paper_limit": 0,
        "single_paragraph": any(k in text for k in ["single paragraph", "一段", "单段"]),
        "remove_headings": any(k in text for k in ["不要标题", "无标题", "不用标题", "no heading", "without heading"]),
        "target_words": 0,
        "need_math_reasoning": any(k in text for k in ["数学", "math", "derivation", "proof"]),
        "need_repo_search": any(k in text for k in ["github", "repo", "开源仓库", "代码仓"]),
        "need_critic": any(k in text for k in ["评审", "critic", "严格", "严谨", "quality", "审稿"]),
        "fast_mode": any(k in text for k in ["快速", "quick", "简要", "brainstorm", "方向", "总结", "摘要", "概述", "summary", "overview", "简介"]),
        "output_format": "auto",
        "prefer_modify_tex": any(k in text for k in ["tex", "latex", "模板", "框架格式"]),
        "allow_write_without_papers": False,
        "need_write": any(k in text for k in ["write", "survey", "综述", "report", "撰写", "写一份", "写个"]),
        "edit_existing_draft": bool(edit_detect.get("is_edit_existing_markdown", False)),
        "source_markdown_path": str(edit_detect.get("source_markdown_path", "") or ""),
        "prefer_tool_call": any(
            k in text
            for k in [
                "文件", "file", "path", "目录", "folder", "url", "网页", "网页链接",
                "readme", "代码库", "仓库结构", "本地项目", "project", "workspace",
                "读一下", "看看这个目录", "分析这个文件",
            ]
        ),
        "output_language": "en" if any(k in text for k in ["全英文", "英文", "english", "in english"]) else (
            "zh" if any(k in text for k in ["中文", "chinese", "in chinese"]) else "auto"
        ),
        "deep_research_mode": deep_mode,
        "economy_mode": False,  # disabled — use RESEARCH_AGENT_COST_PROFILE=budget env var instead
        "min_papers_found": 0,
        "min_papers_read": 0,
        "min_supervisor_rounds": 1,
        "evidence_repair_max_rounds": 2,
        "analyzer_supervisor_max_rounds": 2,
        "writer_head_max_rounds": 2,
        "executor_rollouts_deep": 3,
        "executor_parallelism_deep": 2,
        "enforce_brief_gate": False,
    }
    if constraints["edit_existing_draft"]:
        constraints["allow_write_without_papers"] = True
    if deep_mode:
        constraints["fast_mode"] = False
        constraints["need_critic"] = True
        constraints["paper_limit"] = max(int(constraints.get("paper_limit", 0) or 0), 20)
        constraints["min_papers_found"] = max(int(constraints.get("min_papers_found", 0) or 0), 16)
        constraints["min_papers_read"] = max(int(constraints.get("min_papers_read", 0) or 0), 10)
        constraints["min_supervisor_rounds"] = max(int(constraints.get("min_supervisor_rounds", 1) or 1), 2)
        constraints["evidence_repair_max_rounds"] = max(int(constraints.get("evidence_repair_max_rounds", 1) or 1), 3)
        constraints["analyzer_supervisor_max_rounds"] = max(int(constraints.get("analyzer_supervisor_max_rounds", 2) or 2), 4)
        constraints["writer_head_max_rounds"] = max(int(constraints.get("writer_head_max_rounds", 2) or 2), 3)

    if constraints.get("economy_mode"):
        # Economy mode trims loop depth while keeping a quality gate via critic.
        constraints["fast_mode"] = True
        constraints["min_supervisor_rounds"] = 1
        constraints["evidence_repair_max_rounds"] = min(int(constraints.get("evidence_repair_max_rounds", 2) or 2), 1)
        constraints["analyzer_supervisor_max_rounds"] = min(int(constraints.get("analyzer_supervisor_max_rounds", 2) or 2), 1)
        constraints["writer_head_max_rounds"] = min(int(constraints.get("writer_head_max_rounds", 2) or 2), 2)
        constraints["executor_rollouts_deep"] = 1
        constraints["executor_parallelism_deep"] = 1
        if int(constraints.get("paper_limit", 0) or 0) > 12:
            constraints["paper_limit"] = 12
        if int(constraints.get("min_papers_found", 0) or 0) > 10:
            constraints["min_papers_found"] = 10
        if int(constraints.get("min_papers_read", 0) or 0) > 6:
            constraints["min_papers_read"] = 6
    # 简单抽取 words 目标（支持“400字 / 400 words”）
    for token in re.findall(r"\d{2,5}", text):
        val = int(token)
        if 100 <= val <= 20000:
            constraints["target_words"] = val
            break
    # 字数较少时自动开启 fast_mode
    if 0 < constraints.get("target_words", 0) <= 600:
        constraints["fast_mode"] = True
    # 简单抽取论文上限
    if "top" in text:
        parts = text.split("top")
        if len(parts) > 1:
            nxt = parts[1].strip().split(" ")[0]
            if nxt.isdigit():
                constraints["paper_limit"] = min(max(int(nxt), 5), 80)
    return constraints


def _build_research_brief(
    raw_input: str,
    context_package: dict,
    memory_topic: str,
    task_profile: str,
    doc_type: str,
    constraints: dict,
) -> tuple[dict, dict]:
    """
    Build a normalized brief + explicitness map used by the 95% gate.
    Required dimensions:
      - user_goal/topic
      - genre (doc_type/task_profile)
      - target_words
      - output_structure
      - output_format
    """
    text = (raw_input or "").strip()
    lower = text.lower()

    topic = memory_topic or _extract_fact_value(context_package, "brief_topic") or text
    explicit_topic = bool(memory_topic) or len(text) >= 10

    genre_signals = ["综述", "survey", "report", "报告", "实验", "experimental"]
    explicit_genre = any(k in lower for k in genre_signals) or bool(_extract_fact_value(context_package, "brief_doc_type"))

    mem_words = _extract_fact_value(context_package, "brief_target_words")
    target_words = int(constraints.get("target_words", 0) or 0)
    if target_words <= 0 and mem_words.isdigit():
        target_words = int(mem_words)
    explicit_words = target_words > 0

    format_hint = str(constraints.get("output_format", "auto") or "auto").lower()
    if format_hint == "auto":
        format_hint = _extract_fact_value(context_package, "brief_output_format") or "auto"
    explicit_format = format_hint in {"markdown", "tex", "both"} or any(
        k in lower for k in ["markdown", "md", "latex", "tex", "both", "双格式"]
    )

    if constraints.get("single_paragraph"):
        output_structure = "single_paragraph"
    elif constraints.get("remove_headings"):
        output_structure = "no_headings"
    elif any(k in lower for k in ["结构", "章节", "section", "outline", "提纲"]):
        output_structure = "structured_sections"
    else:
        output_structure = _extract_fact_value(context_package, "brief_output_structure") or "structured_sections"
    explicit_structure = bool(
        constraints.get("single_paragraph")
        or constraints.get("remove_headings")
        or any(k in lower for k in ["结构", "章节", "section", "outline", "提纲", "单段", "标题"])
        or _extract_fact_value(context_package, "brief_output_structure")
    )

    brief = {
        "topic": topic.strip()[:220],
        "task_profile": task_profile,
        "doc_type": doc_type,
        "target_words": target_words,
        "output_structure": output_structure,
        "output_format": format_hint if format_hint in {"auto", "markdown", "tex", "both"} else "auto",
        "output_language": str(constraints.get("output_language", "auto") or "auto"),
    }
    explicit = {
        "topic": explicit_topic,
        "genre": explicit_genre,
        "target_words": explicit_words,
        "output_structure": explicit_structure,
        "output_format": explicit_format,
    }
    return brief, explicit


def _score_research_brief(brief: dict, explicit: dict) -> tuple[float, list[str]]:
    weights = {
        "topic": 0.30,
        "genre": 0.20,
        "target_words": 0.20,
        "output_structure": 0.15,
        "output_format": 0.15,
    }
    score = 0.0
    missing: list[str] = []
    for key, weight in weights.items():
        if bool(explicit.get(key, False)):
            score += weight
        else:
            missing.append(key)

    # Guardrails: even if explicit flag is true, empty value still counts as missing.
    if not str(brief.get("topic", "")).strip() and "topic" not in missing:
        missing.append("topic")
        score -= weights["topic"]
    if int(brief.get("target_words", 0) or 0) <= 0 and "target_words" not in missing:
        missing.append("target_words")
        score -= weights["target_words"]

    return max(0.0, min(1.0, round(score, 3))), missing


def _missing_label(field: str) -> str:
    return {
        "topic": "研究目标/题材",
        "genre": "文档类型（survey/report/experimental）",
        "target_words": "字数目标",
        "output_structure": "输出结构（单段/分章节/是否要标题）",
        "output_format": "输出格式（markdown/tex/both）",
    }.get(field, field)


def _build_brief_clarification(brief: dict, missing_fields: list[str], confidence: float) -> str:
    known = [
        f"- 主题: {brief.get('topic', '') or '未确认'}",
        f"- 类型: {brief.get('doc_type', 'report')}",
        f"- 字数: {brief.get('target_words', 0) or '未确认'}",
        f"- 结构: {brief.get('output_structure', 'structured_sections')}",
        f"- 格式: {brief.get('output_format', 'auto')}",
    ]
    asks = [f"- {_missing_label(item)}" for item in missing_fields]
    return (
        "我先不启动 Research，先把需求确认到可执行阈值（>=95%）。\n\n"
        f"当前 Brief 置信度：{confidence:.0%}\n"
        "已识别信息：\n"
        + "\n".join(known)
        + "\n\n请补充以下缺失项（可一次性回复）：\n"
        + "\n".join(asks)
    )


def _persist_brief_facts(mem: ConversationMemory, brief: dict):
    if brief.get("topic"):
        mem.add_fact(f"brief_topic={str(brief.get('topic'))[:220]}")
    if brief.get("doc_type"):
        mem.add_fact(f"brief_doc_type={brief.get('doc_type')}")
    if int(brief.get("target_words", 0) or 0) > 0:
        mem.add_fact(f"brief_target_words={int(brief.get('target_words', 0) or 0)}")
    if brief.get("output_structure"):
        mem.add_fact(f"brief_output_structure={brief.get('output_structure')}")
    if brief.get("output_format") and brief.get("output_format") != "auto":
        mem.add_fact(f"brief_output_format={brief.get('output_format')}")


def _build_executor_goal(brief: dict, task_profile: str) -> str:
    topic = str(brief.get("topic", "") or "").strip()
    words = int(brief.get("target_words", 0) or 0)
    structure = str(brief.get("output_structure", "structured_sections") or "structured_sections")
    if not topic:
        return "collect high-signal evidence and handoff to analyzer"
    if words > 0:
        return (
            f"collect evidence for '{topic}', prepare analyzer-ready package, "
            f"target output around {words} words with structure={structure} (profile={task_profile})"
        )
    return (
        f"collect evidence for '{topic}', prepare analyzer-ready package, "
        f"respect structure={structure} (profile={task_profile})"
    )


def _build_agent_capabilities(
    plan: list[str],
    constraints: dict,
    task_profile: str,
    brief: dict,
) -> dict:
    executor_tools = [spec.name for spec in get_tools_for_plan(plan)]
    if not executor_tools and task_profile == "general_task":
        executor_tools = [
            "session_context",
            "list_local_files",
            "read_local_file",
            "fetch_url",
            "web_search",
            "paper_search",
            "github_repo_search",
            "save_text_artifact",
            "done",
        ]

    # Analyzer keeps a supervisor tool belt for evidence verification and patching.
    analyzer_tools = [
        "optimize_queries",
        "search_papers",
        "read_papers",
        "find_github_repo",
        "paper_store_query",
        "explain_with_evidence",
        "list_local_files",
        "read_local_file",
        "fetch_url",
        "web_search",
        "arxiv_search",
        "paper_search",
        "github_repo_search",
        "session_context",
    ]

    max_analyzer_steps = 2 if constraints.get("fast_mode") else 4
    return {
        "executor": {
            "role": "evidence_collection_taor",
            "allowed_tools": list(dict.fromkeys(executor_tools)),
            "constraints": {
                "must_ground_observation": True,
                "stop_when_enough_for_handoff": True,
            },
        },
        "researcher": {
            "role": "semantic_compression",
            "owner": "analyzer",
            "deliverables": [
                "analysis_brief",
                "analysis_key_points",
                "analysis_open_risks",
            ],
            "brief_topic": str(brief.get("topic", "") or ""),
        },
        "analyzer": {
            "role": "supervisor_taor_with_tools",
            "allowed_tools": analyzer_tools,
            "max_steps": max_analyzer_steps,
            "max_errors": 2,
        },
        "writer": {
            "role": "final_draft_generation",
            "allowed_tools": [],
        },
        "critic": {
            "role": "final_draft_review",
            "allowed_tools": [],
        },
    }


def _normalize_plan(
    plan: list[str],
    constraints: dict,
    task_profile: str,
    doc_type: str,
) -> list[str]:
    """对 LLM 计划做策略后处理：去无效阶段、补依赖、任务感知 critic。"""
    valid_stages = PLANNABLE_SET
    ordered = [s for s in plan if s in valid_stages and s != "experiment"]
    allow_write_without_papers = bool((constraints or {}).get("allow_write_without_papers", False))
    prefer_tool_call = bool((constraints or {}).get("prefer_tool_call", False))
    need_write = bool((constraints or {}).get("need_write", False))
    deep_mode = bool((constraints or {}).get("deep_research_mode", False))
    if task_profile == "general_task" and not ordered:
        ordered = ["tool_call"]
    if not ordered:
        if allow_write_without_papers:
            ordered = ["write"]
        elif prefer_tool_call or task_profile == "general_task":
            ordered = ["tool_call"]
        else:
            ordered = ["search", "read"]
    if allow_write_without_papers:
        ordered = [s for s in ordered if s in {"write", "critic"}]
        if "write" not in ordered:
            ordered.insert(0, "write")

    # 基础依赖：后续任何阶段都应先有 search/read
    research_downstream = {"read", "github", "coding_plan", "write", "critic"}
    if (not allow_write_without_papers) and "search" not in ordered and any(
        s in ordered for s in ["read", "github", "coding_plan", "write", "critic"]
    ) and "tool_call" not in ordered:
        ordered.insert(0, "search")
    if (not allow_write_without_papers) and "read" not in ordered and any(
        s in ordered for s in ["github", "coding_plan", "write", "critic"]
    ) and "tool_call" not in ordered:
        insert_idx = 1 if ordered and ordered[0] == "search" else 0
        ordered.insert(insert_idx, "read")
    # evidence_evaluate 只在综述/文献综述路径下强制插入，快速模式和轻量任务跳过
    need_evidence_eval = (
        doc_type == "survey" or task_profile == "literature_review"
    ) and not constraints.get("fast_mode", False)
    if "read" in ordered and "evidence_evaluate" not in ordered and need_evidence_eval:
        ordered.insert(ordered.index("read") + 1, "evidence_evaluate")

    # 需要仓库检索则补 github（放到 read 后）
    if constraints.get("need_repo_search"):
        if "github" not in ordered and "read" in ordered:
            read_idx = ordered.index("read")
            ordered.insert(read_idx + 1, "github")

    # coding_plan 依赖 read / github
    if "coding_plan" in ordered and "read" not in ordered:
        ordered.insert(0, "read")

    # 用户明确要求写作时，强制包含 write（除非是纯 general_task 工具流）
    if need_write and task_profile != "general_task" and "write" not in ordered:
        ordered.append("write")

    # 任务感知 critic：报告/综述/数学推理默认更需要质检；快速探索可跳过
    need_critic = constraints.get("need_critic", False)
    fast_mode = constraints.get("fast_mode", False)
    if task_profile in {"literature_review", "paper_from_implementation", "math_reasoning"}:
        need_critic = True if not fast_mode else need_critic
    if doc_type == "report" and not fast_mode:
        need_critic = True if not constraints.get("single_paragraph") else need_critic

    if "write" in ordered and need_critic and "critic" not in ordered:
        ordered.append("critic")
    if fast_mode and "critic" in ordered and not constraints.get("need_critic"):
        ordered = [s for s in ordered if s != "critic"]
    if "critic" in ordered and "write" not in ordered:
        ordered.insert(max(len(ordered) - 1, 0), "write")

    if deep_mode and not allow_write_without_papers:
        if "query_optimize" not in ordered:
            ordered.insert(0, "query_optimize")
        if "evidence_evaluate" not in ordered:
            if "read" in ordered:
                ordered.insert(ordered.index("read") + 1, "evidence_evaluate")
            else:
                ordered.append("evidence_evaluate")
        if "critic" not in ordered and "write" in ordered:
            ordered.append("critic")

    if task_profile == "general_task" and "tool_call" not in ordered:
        ordered.insert(0, "tool_call")
    if task_profile == "general_task":
        keep = {"tool_call", "write", "critic"}
        if any(stage in ordered for stage in keep):
            ordered = [stage for stage in ordered if stage in keep]

    # 统一顺序，避免 LLM 给出不合理拓扑
    stage_order = [s for s in PLANNABLE_STAGES if s != "query_optimize"] + ["query_optimize"]
    unique = []
    seen = set()
    for s in ordered:
        if s not in seen:
            seen.add(s)
            unique.append(s)
    unique.sort(key=lambda x: stage_order.index(x))
    return unique


def brain_node(state: ResearchState) -> ResearchState:
    raw_input = state.get("raw_input", "")
    session_id = state.get("session_id", "default")
    mem = ConversationMemory(session_id)
    mem.add_turn("user", raw_input)

    context_package = mem.build_context_package(current_input=raw_input, char_budget=5200)
    memory_topic = _extract_memory_topic(context_package)
    followup_with_memory = _is_followup_instruction(raw_input) and bool(memory_topic)
    if _should_clarify_only(raw_input) and not followup_with_memory:
        lower = (raw_input or "").strip().lower()
        if any(k in lower for k in ["写代码", "code", "你会", "你能", "can you"]):
            clarification = (
                "会的。我可以直接改代码、修流程、写计划和提示词。"
                "你给我一个具体目标（改哪个文件/实现什么功能）我就直接开做。"
            )
        else:
            clarification = (
                "你好，我在。告诉我你想研究的主题或目标产物（如：LLM Infra方向、"
                "一段式综述、基于你的TeX模板改写），我就开始执行。"
            )
        print("  检测到输入信息不足，进入澄清模式（不触发检索流程）。")
        mem.add_turn("assistant", clarification)
        mem.add_decision("clarify_only_no_pipeline")
        mem.save()
        return {
            **state,
            "selected_question": "",
            "task_profile": "general_research",
            "doc_type": "report",
            "user_constraints": {},
            "context_package": context_package,
            "brain_plan": [],
            "brain_plan_index": 0,
            "tasks": [],
            "next": "end",
            "current_stage": "brain",
            "assistant_response": clarification,
            "confidence_score": 0.95,
            "confidence_label": "high",
            "research_brief": {},
            "research_brief_confidence": 0.0,
            "brief_ready": False,
            "brief_missing_fields": ["topic", "genre", "target_words", "output_structure", "output_format"],
        }

    effective_input = raw_input
    if followup_with_memory and memory_topic and memory_topic.lower() not in raw_input.lower():
        effective_input = f"{raw_input}\n\n会话已确认主题：{memory_topic}"

    heuristic_constraints = _infer_constraints(raw_input)
    if state.get("output_format"):
        heuristic_constraints["output_format"] = state.get("output_format")
    if state.get("target_tex_path"):
        heuristic_constraints["prefer_modify_tex"] = True
        if heuristic_constraints.get("output_format") == "auto":
            heuristic_constraints["output_format"] = "tex"
    if state.get("writer_rules_path"):
        heuristic_constraints["writer_rules_path"] = state.get("writer_rules_path")
    heuristic_profile = state.get("task_profile", "general_research")
    heuristic_doc_type = state.get("doc_type", "survey")

    if "literature" in effective_input.lower() or "综述" in effective_input:
        heuristic_profile = "literature_review"
        heuristic_doc_type = "survey"
    if any(k in effective_input.lower() for k in ["codex", "claude code", "实现计划", "prompt", "repo"]):
        heuristic_profile = "paper_from_implementation"
        heuristic_doc_type = "report"
    general_task_signals = [
        "文件", "file", "path", "目录", "folder", "readme", "workspace", "project",
        "网页", "url", "link", "整理一下", "总结这个页面", "读一下这个",
    ]
    if state.get("input_paths") or any(k in effective_input.lower() for k in general_task_signals):
        heuristic_profile = "general_task"
        heuristic_doc_type = "report"
        heuristic_constraints["prefer_tool_call"] = True
    edit_detect = _detect_edit_existing_markdown_request(raw_input)
    if edit_detect.get("is_edit_existing_markdown"):
        heuristic_constraints["edit_existing_draft"] = True
        heuristic_constraints["allow_write_without_papers"] = True
        heuristic_constraints["source_markdown_path"] = edit_detect.get("source_markdown_path", "./output/paper.md")
        heuristic_profile = "general_research"
        heuristic_doc_type = "report"

    heuristic_state = {
        **state,
        "raw_input": effective_input,
        "task_profile": heuristic_profile,
        "doc_type": heuristic_doc_type,
    }
    heuristic_difficulty = assess_difficulty(heuristic_state)
    heuristic_plan = recommend_plan(heuristic_state)

    # Only inject memory topic when user input is clearly a follow-up instruction.
    brief_memory_topic = memory_topic if followup_with_memory else ""
    research_brief, explicit_map = _build_research_brief(
        raw_input=raw_input,
        context_package=context_package,
        memory_topic=brief_memory_topic,
        task_profile=heuristic_profile,
        doc_type=heuristic_doc_type,
        constraints=heuristic_constraints,
    )
    brief_confidence, brief_missing = _score_research_brief(research_brief, explicit_map)
    env_gate = os.getenv(BRIEF_GATE_ENV, "").strip().lower() in {"1", "true", "yes", "on"}
    brief_gate_enabled = bool(heuristic_constraints.get("enforce_brief_gate", False) or env_gate)
    brief_ready = (brief_confidence >= RESEARCH_BRIEF_THRESHOLD) or (not brief_gate_enabled)
    provisional_caps = _build_agent_capabilities(
        heuristic_plan,
        heuristic_constraints,
        heuristic_profile,
        research_brief,
    )
    provisional_executor_goal = _build_executor_goal(research_brief, heuristic_profile)
    _persist_brief_facts(mem, research_brief)

    if not brief_ready:
        clarification = _build_brief_clarification(research_brief, brief_missing, brief_confidence)
        print(f"  Research Brief 置信度不足：{brief_confidence:.0%} (< {RESEARCH_BRIEF_THRESHOLD:.0%})，先澄清再执行。")
        mem.add_turn("assistant", clarification)
        mem.add_decision(f"brief_gate_blocked confidence={brief_confidence:.2f} missing={brief_missing}")
        mem.save()
        return {
            **state,
            "selected_question": research_brief.get("topic", ""),
            "task_profile": heuristic_profile,
            "doc_type": heuristic_doc_type,
            "user_constraints": heuristic_constraints,
            "context_package": context_package,
            "brain_plan": [],
            "brain_plan_index": 0,
            "tasks": [],
            "next": "end",
            "current_stage": "brain",
            "assistant_response": clarification,
            "confidence_score": 0.95,
            "confidence_label": "high",
            "research_brief": research_brief,
            "research_brief_confidence": brief_confidence,
            "brief_ready": False,
            "brief_missing_fields": brief_missing,
            "executor_goal": provisional_executor_goal,
            "agent_capabilities": provisional_caps,
        }

    input_json = json.dumps(
        {
            "user_input": effective_input,
            "original_user_input": raw_input,
            "memory_topic": memory_topic,
            "research_brief": research_brief,
            "brief_confidence": brief_confidence,
            "session_id": session_id,
            "heuristic": {
                "difficulty": heuristic_difficulty,
                "task_profile": heuristic_profile,
                "doc_type": heuristic_doc_type,
                "plan": heuristic_plan,
                "constraints": heuristic_constraints,
            },
            "context_package": context_package,
            "local_context_preview": (state.get("local_context", "") or "")[:3500],
            "system_policy": {
                "disable_reproduction_default": True,
                "prefer_coding_plan_handoff": True,
            },
        },
        ensure_ascii=False,
        indent=2,
    )

    print(f"\n[Brain Agent] 任务理解与动态规划：{raw_input[:80]}")
    try:
        response = call_llm_json(
            "planning",
            [
                system(
                    render_prompt(
                        "agents.brain.main",
                        input_json=input_json,
                        stage_list=", ".join(PLANNABLE_STAGES),
                    )
                )
            ],
            max_tokens=12000,
        )
        data = parse_json(response)

        valid_stages = PLANNABLE_SET
        raw_plan = data.get("plan", heuristic_plan)
        plan = [
            s for s in (
                [item.get("stage", item.get("name", "")) if isinstance(item, dict) else str(item)
                 for item in raw_plan]
            )
            if s in valid_stages
        ]
        if not plan:
            plan = heuristic_plan

        task_profile = data.get("task_profile", heuristic_profile)
        doc_type = data.get("doc_type", heuristic_doc_type)
        topic = data.get("topic", raw_input)
        constraints = data.get("constraints", heuristic_constraints) or heuristic_constraints
        if "need_write" not in constraints:
            constraints["need_write"] = bool(heuristic_constraints.get("need_write", False))
        understanding = data.get("understanding", "")
        skip_reason = data.get("skip_reason", "")
        response_to_user = data.get("response_to_user", "")
        sub_questions = [
            str(q).strip() for q in (data.get("sub_questions", []) or [])
            if str(q).strip()
        ][:4]
        if sub_questions:
            research_brief["sub_questions"] = sub_questions

        if state.get("writer_rules_path") and not constraints.get("writer_rules_path"):
            constraints["writer_rules_path"] = state.get("writer_rules_path")
        if followup_with_memory and memory_topic:
            generic_topic_tokens = {"", raw_input.strip(), effective_input.strip(), "same topic", "this topic", "当前主题"}
            if str(topic).strip().lower() in {t.lower() for t in generic_topic_tokens if t}:
                topic = memory_topic
        if edit_detect.get("is_edit_existing_markdown"):
            task_profile = "general_research"
            doc_type = "report"
            constraints["edit_existing_draft"] = True
            constraints["allow_write_without_papers"] = True
            constraints["source_markdown_path"] = edit_detect.get("source_markdown_path", "./output/paper.md")
            constraints["need_critic"] = False
            plan = ["write"]
        elif heuristic_constraints.get("prefer_tool_call") and task_profile in {"general_research", "general_task"} and not plan:
            task_profile = "general_task"
            plan = ["tool_call"]

        # 策略后处理：按任务画像/约束动态调整 plan
        plan = _normalize_plan(plan, constraints, task_profile, doc_type)
        agent_capabilities = _build_agent_capabilities(plan, constraints, task_profile, research_brief)
        executor_goal = _build_executor_goal(research_brief, task_profile)

        print(f"  理解：{understanding}")
        print(f"  任务画像：{task_profile} | 文档类型：{doc_type}")
        print(f"  计划：{' → '.join(plan)}")
        print(
            f"  Brief 置信度：{brief_confidence:.0%} "
            f"(gate={'on' if brief_gate_enabled else 'off'}, threshold={RESEARCH_BRIEF_THRESHOLD:.0%})"
        )
        if skip_reason:
            print(f"  跳过说明：{skip_reason}")
        if response_to_user:
            print(f"  {response_to_user}")

        mem.pin_goal(topic)
        _persist_brief_facts(mem, research_brief)
        pinned_constraints = []
        if constraints.get("single_paragraph"):
            pinned_constraints.append("output should be single paragraph when requested")
        if constraints.get("target_words", 0):
            pinned_constraints.append(f"target words about {constraints['target_words']}")
        mem.set_constraints(pinned_constraints)
        mem.add_decision(f"plan={plan}; profile={task_profile}; doc_type={doc_type}")
        mem.save()

        tasks = _build_tasks_from_plan(plan, topic)
        topic_key = hashlib.md5(f"{session_id}:{topic}".encode("utf-8")).hexdigest()[:12]
        return {
            **state,
            "selected_question": topic,
            "task_profile": task_profile,
            "doc_type": doc_type,
            "user_constraints": constraints,
            "context_package": context_package,
            "paper_store_dir": f"./paper_index/{session_id}/{topic_key}",
            "brain_plan": plan,
            "brain_plan_index": 0,
            "tasks": tasks,
            "next": plan[0] if plan else "end",
            "current_stage": "brain",
            "research_brief": research_brief,
            "research_brief_confidence": brief_confidence,
            "brief_ready": True,
            "brief_missing_fields": [],
            "executor_goal": executor_goal,
            "agent_capabilities": agent_capabilities,
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        heuristic_plan = _normalize_plan(
            heuristic_plan,
            heuristic_constraints,
            heuristic_profile,
            heuristic_doc_type,
        )
        print(f"  ⚠ Brain 失败，降级使用启发式计划：{heuristic_plan} | {e}")
        mem.add_decision(f"fallback plan={heuristic_plan}")
        _persist_brief_facts(mem, research_brief)
        mem.save()
        fallback_topic = memory_topic if (followup_with_memory and memory_topic) else raw_input
        topic_key = hashlib.md5(f"{session_id}:{fallback_topic}".encode("utf-8")).hexdigest()[:12]
        return {
            **state,
            "selected_question": fallback_topic,
            "task_profile": heuristic_profile,
            "doc_type": heuristic_doc_type,
            "user_constraints": heuristic_constraints,
            "context_package": context_package,
            "paper_store_dir": f"./paper_index/{session_id}/{topic_key}",
            "brain_plan": heuristic_plan,
            "brain_plan_index": 0,
            "tasks": _build_tasks_from_plan(heuristic_plan, fallback_topic),
            "next": heuristic_plan[0] if heuristic_plan else "end",
            "current_stage": "brain",
            "error_log": state.get("error_log", []) + [f"[brain] {str(e)}"],
            "research_brief": research_brief,
            "research_brief_confidence": brief_confidence,
            "brief_ready": True,
            "brief_missing_fields": [],
              "executor_goal": provisional_executor_goal,
              "agent_capabilities": provisional_caps,
        }
