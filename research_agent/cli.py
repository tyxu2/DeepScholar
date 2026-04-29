import typer
import os
import json
from typing import Optional
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.prompt import Prompt
from research_agent.graph import run_pipeline
from research_agent.eval.evaluator import evaluate
from research_agent.llm.router import call_llm, system
from research_agent.runtime import build_structured_response
from research_agent.tools import get_all_specs, get_tool, list_tools
from research_agent.tools.mcp_stdio import serve_stdio
from research_agent.skills.catalog import get_skill, list_skills, render_skill
from research_agent.utils import tracer as _tracer
from research_agent.memory.conversation_memory import ConversationMemory
from research_agent.context.local_context import build_local_context
from research_agent.observability.langsmith import enable_langsmith_if_configured

# Activate LangSmith tracing if LANGSMITH_API_KEY is set (no-op otherwise).
enable_langsmith_if_configured(verbose=os.environ.get("DEEPSCHOLAR_VERBOSE") == "1")

try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.history import FileHistory
except Exception:
    PromptSession = None  # type: ignore[assignment]
    FileHistory = None  # type: ignore[assignment]

app = typer.Typer(help="全链路科研 Multi-Agent 系统")
tools_app = typer.Typer(help="工具链（registry / MCP）")
skills_app = typer.Typer(help="可复用 skill 模板")
app.add_typer(tools_app, name="tools")
app.add_typer(skills_app, name="skills")
console = Console()

EXAMPLES = """\
示例指令：
  "帮我找关于 few-shot segmentation 的论文"
  "调研 transformer attention 改进方向，写一份综述"
  "找 diffusion model 最新进展，写一段式综述"
"""


def _print_banner():
    console.print(Panel(
        Text("Research Agent  v2.0", style="bold cyan"),
        subtitle="Brain Agent 驱动 · 自然语言指令",
        expand=False,
    ))


def _build_prompt_session(session_id: str):
    """Create a history-enabled interactive prompt when prompt_toolkit is available."""
    if PromptSession is None or FileHistory is None:
        return None
    history_dir = "./output/sessions"
    os.makedirs(history_dir, exist_ok=True)
    history_path = os.path.join(history_dir, f".cli_history_{session_id}.txt")
    return PromptSession(history=FileHistory(history_path))


def _ask_text(prompt_text: str, session=None, default: str = "") -> str:
    if session is not None:
        value = session.prompt(f"{prompt_text} ")
        if not value.strip() and default:
            return default
        return value
    if default:
        return Prompt.ask(prompt_text, default=default)
    return Prompt.ask(prompt_text)


def _ask_choice(prompt_text: str, choices: list[str], default: str, session=None) -> str:
    choices_norm = [c.strip().lower() for c in choices]
    default_norm = default.strip().lower()
    while True:
        value = _ask_text(
            f"{prompt_text} ({'/'.join(choices_norm)})",
            session=session,
            default=default_norm,
        ).strip().lower()
        if value in choices_norm:
            return value
        console.print(f"[yellow]请输入有效选项：{', '.join(choices_norm)}[/yellow]")


def _print_summary(final_state: dict):
    console.print("\n[bold green]✓ 完成[/bold green]")
    budget = final_state.get("run_budget", {})
    items = [
        ("执行引擎", final_state.get("engine", "classic")),
        ("任务画像", final_state.get("task_profile", "")),
        ("研究问题", final_state.get("selected_question", "")),
        ("执行计划", " → ".join(final_state.get("brain_plan", []))),
        ("ReAct步数", str(budget.get("used_steps", 0)) if final_state.get("engine") == "react" else ""),
        ("找到论文", f"{len(final_state.get('found_papers', []))} 篇"),
        ("精读论文", f"{len(final_state.get('paper_summaries', []))} 篇"),
        ("论文草稿", final_state.get("draft_md_path", "")),
        ("LaTeX",    final_state.get("draft_latex_path", "")),
    ]
    for label, val in items:
        if val and val != "0 篇":
            console.print(f"  [dim]{label}:[/dim] {val}")

    artifacts = final_state.get("artifacts", [])
    tex_art = next((a for a in artifacts if a.get("id") == "artifact-template-tex"), None)
    if tex_art:
        console.print(f"  [dim]TemplateTeX:[/dim] {tex_art.get('content_ref')}")


def _confidence_cn(label: str) -> str:
    return {"high": "高", "medium": "中", "low": "低"}.get(label, "中")


def _print_agent_reply(final_state: dict):
    response = (final_state.get("assistant_response", "") or "").strip()
    if not response:
        return
    label = _confidence_cn(final_state.get("confidence_label", "medium"))
    score = float(final_state.get("confidence_score", 0.0))
    title = f"Agent 回复 · 可信度 {label} ({score:.0%})"
    console.print()
    console.print(Panel.fit(response, title=title, border_style="green"))


def _save_assistant_reply(session_id: str, response: str):
    text = (response or "").strip()
    if not text:
        return
    try:
        mem = ConversationMemory(session_id)
        mem.add_turn("assistant", text[:1500])
        mem.save()
    except Exception:
        pass


def _format_last_outputs(state: dict) -> list[str]:
    lines: list[str] = []
    md_path = state.get("draft_md_path", "")
    tex_path = state.get("draft_latex_path", "")
    if md_path:
        lines.append(f"- Markdown: {md_path}")
    if tex_path:
        lines.append(f"- LaTeX: {tex_path}")

    for art in state.get("artifacts", []):
        ref = art.get("content_ref", "")
        title = art.get("title", "") or art.get("id", "artifact")
        if ref:
            lines.append(f"- {title}: {ref}")

    # 去重，保持顺序
    return list(dict.fromkeys(lines))


def _reply_for_session_followup(user_text: str, last_state: dict | None) -> str | None:
    if not last_state:
        return None

    text = user_text.strip().lower()
    path_signals = ["写在哪里", "写在哪", "在哪个文件", "输出在哪", "保存在哪", "路径", "where"]
    progress_signals = ["进度", "做到哪", "写到哪", "现在状态", "status", "progress"]
    error_signals = ["报错", "错误", "失败", "为什么失败", "为什么报错", "why fail"]

    if any(k in text for k in path_signals):
        outputs = _format_last_outputs(last_state)
        if outputs:
            return "上一轮生成文件在这些位置：\n" + "\n".join(outputs)
        return "上一轮还没有落盘文件输出。"

    if any(k in text for k in progress_signals):
        plan = " → ".join(last_state.get("brain_plan", [])) or "（无）"
        return (
            "上一轮进度：\n"
            f"- 计划: {plan}\n"
            f"- 找到论文: {len(last_state.get('found_papers', []))} 篇\n"
            f"- 精读论文: {len(last_state.get('paper_summaries', []))} 篇\n"
            f"- 草稿: {'有' if last_state.get('draft_md_path') else '无'}\n"
            f"- ReAct 步数: {last_state.get('run_budget', {}).get('used_steps', 0)}"
        )

    if any(k in text for k in error_signals):
        errors = last_state.get("error_log", [])
        if not errors:
            return "上一轮没有记录到错误。"
        unique = list(dict.fromkeys(e.split("\n")[0][:160] for e in errors))
        return "上一轮主要错误是：\n" + "\n".join(f"- {e}" for e in unique[:5])

    explain_signals = [
        "为什么", "为啥", "怎么", "如何", "原理", "关系", "区别", "是什么",
        "why", "how", "what", "reason", "principle",
    ]
    if any(k in text for k in explain_signals):
        summaries = last_state.get("paper_summaries", [])
        if summaries:
            evidence_lines = []
            for s in summaries[:8]:
                evidence_lines.append(
                    f"- {s.get('title','')}\n  method: {str(s.get('method',''))[:140]}\n  result: {str(s.get('result',''))[:120]}"
                )
            evidence = "\n".join(evidence_lines)
            prompt = f"""你是研究助手。仅基于证据回答用户追问，不要发散检索。
用户追问：{user_text}

已有证据（上一轮已读论文）：
{evidence}

请输出中文回答：
1) 直接解释用户问题
2) 说明与多智能体/系统机制的关联
3) 用1-2句提示证据边界（不确定就说不确定）
"""
            try:
                ans = call_llm("summarization", [system(prompt)], max_tokens=7000).strip()
                if ans:
                    return ans
            except Exception:
                pass
            return "可以基于上一轮已读论文解释这个问题；如果你愿意，我可以给你一版更短的口语化解释。"

    return None


def _make_brain_clarification_callback(prompt_session=None):
    """
    Returns a callback for run_pipeline's clarification_callback hook.

    When Brain confidence < 80%, the pipeline pauses and shows the user:
      [Y] proceed with defaults  [N] abort  [Other] type a supplement

    Return value mirrors the contract in executor.py:
      None  → abort
      ""    → proceed as-is
      <str> → re-run Brain with this supplement merged into the topic
    """
    def _callback(questions: str, state: dict) -> "str | None":
        confidence = float(state.get("research_brief_confidence", 0.0) or 0.0)
        console.print()
        console.print(Panel.fit(
            questions or "Brain 对任务理解存在歧义，请确认是否继续。",
            title=f"[bold yellow]Brain 需要确认  (置信度 {confidence:.0%})[/bold yellow]",
            border_style="yellow",
        ))
        choice = _ask_choice(
            "如何继续？",
            ["yes", "no", "other"],
            default="yes",
            session=prompt_session,
        )
        if choice == "no":
            console.print("[yellow]已中止，请重新输入更详细的指令。[/yellow]")
            return None
        if choice == "yes":
            console.print("[dim]按默认值继续执行…[/dim]")
            return ""
        supplement = _ask_text("请补充说明（字数目标 / 语言 / 范围 等）", session=prompt_session)
        return supplement.strip()

    return _callback


def _maybe_clarify_format(
    topic: str,
    output_format: str,
    template_tex: str,
    clarify: bool,
    prompt_session=None,
) -> tuple[str, str, str]:
    """
    在格式不明确时，询问用户 yes/no/other。
    返回 (final_output_format, final_template_tex, extra_instruction)
    """
    fmt = (output_format or "auto").strip().lower()
    tex_path = (template_tex or "").strip()
    extra_instruction = ""

    if not clarify:
        return fmt, tex_path, extra_instruction
    if fmt != "auto":
        return fmt, tex_path, extra_instruction
    if tex_path:
        return "tex", tex_path, extra_instruction

    topic_l = topic.lower()
    likely_needs_format = any(
        k in topic_l
        for k in ["论文", "paper", "tex", "latex", "模板", "框架", "format", "写作", "report"]
    )
    if not likely_needs_format:
        return fmt, tex_path, extra_instruction

    console.print("\n[bold]格式不确定，先确认一下：[/bold]")
    choice = _ask_choice(
        "是否要在已有 TeX 模板中生成/修改？",
        ["yes", "no", "other"],
        default="yes",
        session=prompt_session,
    )
    if choice == "yes":
        tex_path = _ask_text("请输入 TeX 模板路径（.tex）", session=prompt_session)
        return "tex", tex_path, extra_instruction
    if choice == "no":
        return "markdown", "", extra_instruction

    custom = _ask_text("请输入你希望的格式要求（markdown/tex/both 或自由描述）", session=prompt_session)
    custom_l = custom.strip().lower()
    if custom_l in {"markdown", "tex", "both"}:
        return custom_l, tex_path, extra_instruction
    extra_instruction = f"\n\n输出格式要求（用户指定）：{custom.strip()}"
    return "auto", tex_path, extra_instruction


def _load_run_config(path: str) -> dict:
    if not path.strip():
        return {}
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    lower = path.lower()
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    if lower.endswith(".json"):
        data = json.loads(text)
    elif lower.endswith((".yaml", ".yml")):
        try:
            import yaml  # type: ignore
        except Exception as e:
            raise RuntimeError("读取 YAML 需要安装 pyyaml") from e
        data = yaml.safe_load(text) or {}
    else:
        # 默认按 JSON 解析
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("配置文件顶层必须是对象")
    return data


@app.command()
def run(
    topic: str = typer.Option("", "--topic", "-t", help="研究指令（自然语言）"),
    hitl: bool = typer.Option(False, "--hitl", help="开启人工确认节点"),
    hitl_stages: Optional[str] = typer.Option(
        None, "--hitl-stages", help="指定人工确认阶段，如 search,read,write"
    ),
    chat: bool = typer.Option(False, "--chat", help="交互对话模式"),
    engine: str = typer.Option("react", "--engine", help="执行引擎：react（primary）或 classic（compat alias）"),
    max_steps: int = typer.Option(12, "--max-steps", help="react 模式最大迭代步数"),
    max_errors: int = typer.Option(4, "--max-errors", help="react 模式最大错误数"),
    executor_rollouts: int = typer.Option(1, "--executor-rollouts", help="Executor 并行 rollout 数"),
    executor_parallelism: int = typer.Option(1, "--executor-parallelism", help="Executor rollout 最大并发数"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="打印详细 CoT 推理过程"),
    session_id: str = typer.Option("default", "--session-id", help="会话 ID（用于多轮记忆）"),
    input_path: Optional[list[str]] = typer.Option(
        None, "--input-path", "-i", help="输入文件或目录路径，可重复传入"
    ),
    template_tex: str = typer.Option("", "--template-tex", help="已有 TeX 模板路径（在该框架中改写）"),
    output_format: str = typer.Option("auto", "--output-format", help="输出格式：auto|markdown|tex|both"),
    writer_rules: str = typer.Option("", "--writer-rules", help="Writer 规则文件路径（json/yaml）"),
    clarify: bool = typer.Option(True, "--clarify/--no-clarify", help="格式不确定时询问 yes/no/other"),
    config: str = typer.Option("", "--config", help="运行配置文件（json/yaml）"),
):
    """运行科研 Agent。支持自然语言指令，Brain Agent 自动规划执行步骤。"""
    _print_banner()

    cfg = {}
    if config:
        try:
            cfg = _load_run_config(config)
        except Exception as e:
            console.print(f"[red]读取配置失败: {e}[/red]")
            raise typer.Exit(1)

    if not topic and isinstance(cfg.get("topic"), str):
        topic = cfg["topic"]
    if engine == "react" and isinstance(cfg.get("engine"), str):
        engine = cfg["engine"]
    if max_steps == 12 and isinstance(cfg.get("max_steps"), int):
        max_steps = cfg["max_steps"]
    if max_errors == 4 and isinstance(cfg.get("max_errors"), int):
        max_errors = cfg["max_errors"]
    if executor_rollouts == 1 and isinstance(cfg.get("executor_rollouts"), int):
        executor_rollouts = cfg["executor_rollouts"]
    if executor_parallelism == 1 and isinstance(cfg.get("executor_parallelism"), int):
        executor_parallelism = cfg["executor_parallelism"]
    if output_format == "auto" and isinstance(cfg.get("output_format"), str):
        output_format = cfg["output_format"]
    if not writer_rules and isinstance(cfg.get("writer_rules_path"), str):
        writer_rules = cfg["writer_rules_path"]
    if not template_tex and isinstance(cfg.get("template_tex"), str):
        template_tex = cfg["template_tex"]
    if not input_path and isinstance(cfg.get("input_paths"), list):
        input_path = [str(p) for p in cfg["input_paths"]]
    if not hitl and bool(cfg.get("hitl", False)):
        hitl = True
    if not hitl_stages and isinstance(cfg.get("hitl_stages"), list):
        hitl_stages = ",".join(str(s) for s in cfg["hitl_stages"])

    if chat or not topic:
        # 交互对话模式
        console.print(f"\n[dim]{EXAMPLES}[/dim]")
        prompt_session = _build_prompt_session(session_id)
        topic = _ask_text("请输入研究指令", session=prompt_session)
    else:
        prompt_session = _build_prompt_session(session_id)

    if not topic.strip():
        console.print("[red]请提供研究指令[/red]")
        raise typer.Exit(1)

    # 默认不让 FrontDesk 拦截，统一交给 Brain 判断任务意图。
    frontdesk = {
        "intent": "research_task",
        "confidence": 1.0,
        "next_action": "run_research",
        "response": "",
        "reason": "bypass_frontdesk_default",
    }
    if verbose:
        console.print("[dim]FrontDesk: bypassed, delegated to Brain[/dim]")

    output_format, template_tex, extra_instruction = _maybe_clarify_format(
        topic=topic,
        output_format=output_format,
        template_tex=template_tex,
        clarify=clarify,
        prompt_session=prompt_session,
    )
    if extra_instruction:
        topic = topic + extra_instruction

    engine = engine.strip().lower()
    if engine not in {"classic", "react"}:
        console.print("[red]--engine 只支持 classic 或 react[/red]")
        raise typer.Exit(1)
    if engine == "classic":
        console.print("[dim]classic 已降级为兼容别名，实际执行仍使用 react 内核[/dim]")

    format_allowed = {"auto", "markdown", "tex", "both"}
    if output_format not in format_allowed:
        console.print(f"[red]--output-format 仅支持 {sorted(format_allowed)}[/red]")
        raise typer.Exit(1)

    stages = hitl_stages.split(",") if hitl_stages else []
    if hitl and not stages:
        stages = ["search", "read", "write"]

    raw_paths = list(input_path or [])
    if template_tex:
        raw_paths.append(template_tex)
    resolved_paths, local_context = build_local_context(raw_paths)

    # 初始化 tracer
    _tracer.set_verbose(verbose)
    trace_path = "./output/trace.jsonl"
    if os.path.exists(trace_path):
        os.remove(trace_path)   # 每次 run 清空上次 trace

    console.print(f"\n[bold]指令：[/bold]{topic}")
    console.print(f"[bold]会话：[/bold]{session_id}")
    console.print(f"[bold]引擎：[/bold]{engine}")
    console.print(f"[bold]Executor Rollouts：[/bold]{executor_rollouts} (parallel={executor_parallelism})")
    if stages:
        console.print(f"[bold]人工介入：[/bold]{stages}")
    if verbose:
        console.print(f"[bold]详细模式：[/bold]开启（CoT 推理可见）")
    if resolved_paths:
        console.print(f"[bold]本地输入：[/bold]{len(resolved_paths)} 个文件")
    if template_tex:
        console.print(f"[bold]模板 TeX：[/bold]{template_tex}")
    console.print(f"[bold]输出格式：[/bold]{output_format}")
    if writer_rules:
        console.print(f"[bold]Writer规则：[/bold]{writer_rules}")
    console.print()

    final_state = run_pipeline(
        raw_input=topic,
        human_in_loop=bool(stages),
        hitl_stages=stages,
        engine=engine,
        max_steps=max_steps,
        max_errors=max_errors,
        executor_rollouts=executor_rollouts,
        executor_parallelism=executor_parallelism,
        session_id=session_id,
        input_paths=resolved_paths,
        local_context=local_context,
        target_tex_path=template_tex,
        output_format=output_format,
        writer_rules_path=writer_rules,
        frontdesk_decision=frontdesk,
        clarification_callback=_make_brain_clarification_callback(prompt_session),
    )

    if final_state.get("assistant_response") and not final_state.get("brain_plan"):
        _print_agent_reply(final_state)
        _save_assistant_reply(session_id, final_state.get("assistant_response", ""))
        return

    eval_report = evaluate(final_state, print_report=verbose)
    final_state = build_structured_response(final_state, eval_report.to_dict())
    _print_agent_reply(final_state)
    _save_assistant_reply(session_id, final_state.get("assistant_response", ""))

    if verbose:
        _print_summary(final_state)
    else:
        artifacts = final_state.get("artifacts", [])
        if artifacts or final_state.get("draft_md_path") or final_state.get("draft_latex_path"):
            console.print("\n[dim]可交付文件：[/dim]")
            if final_state.get("draft_md_path"):
                console.print(f"  - {final_state.get('draft_md_path')}")
            if final_state.get("draft_latex_path"):
                console.print(f"  - {final_state.get('draft_latex_path')}")
            for a in artifacts[:6]:
                ref = a.get("content_ref", "")
                if ref:
                    console.print(f"  - {ref}")

    if verbose:
        from research_agent.utils.tracer import summarize_trace
        console.print(f"\n[dim]CoT Trace 摘要：[/dim]")
        console.print(f"  {summarize_trace()}")
        console.print(f"  完整 trace: {trace_path}")

    if final_state.get("error_log"):
        # 只打印非重复错误
        unique_errors = list(dict.fromkeys(
            e.split("\n")[0][:100] for e in final_state["error_log"]
        ))
        console.print(f"\n[yellow]⚠ 错误（{len(unique_errors)} 类）：[/yellow]")
        for e in unique_errors[:5]:
            console.print(f"  {e}")

    # 评估报告已在上方执行并保存


@app.command()
def session(
    session_id: str = typer.Option("default", "--session-id", help="会话 ID"),
    engine: str = typer.Option("react", "--engine", help="执行引擎：react（primary）或 classic（compat alias）"),
    max_steps: int = typer.Option(12, "--max-steps", help="react 模式最大迭代步数"),
    max_errors: int = typer.Option(4, "--max-errors", help="react 模式最大错误数"),
    executor_rollouts: int = typer.Option(1, "--executor-rollouts", help="Executor 并行 rollout 数"),
    executor_parallelism: int = typer.Option(1, "--executor-parallelism", help="Executor rollout 最大并发数"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="打印详细 CoT 推理过程"),
):
    """进入多轮会话模式：持续提问、持续记忆、持续迭代。"""
    _print_banner()
    console.print(f"[bold]会话模式[/bold] | session_id={session_id}")
    console.print("[dim]输入 /exit 退出，/reset 清空会话记忆[/dim]\n")

    _tracer.set_verbose(verbose)
    mem = ConversationMemory(session_id)
    last_state: dict | None = None
    prompt_session = _build_prompt_session(session_id)

    while True:
        topic = _ask_text("你", session=prompt_session)
        if not topic.strip():
            continue
        cmd = topic.strip().lower()
        if cmd in {"/exit", "/quit"}:
            console.print("[green]会话结束。[/green]")
            break
        if cmd == "/reset":
            mem.clear()
            console.print("[yellow]已清空该会话记忆。[/yellow]")
            last_state = None
            continue

        # 会话模式同样默认直达 Brain；仅对明确“查状态”追问做本地回复快捷路径。
        followup_reply = _reply_for_session_followup(topic, last_state)
        if followup_reply:
            console.print(f"[cyan]{followup_reply}[/cyan]")
            _save_assistant_reply(session_id, followup_reply)
            continue

        frontdesk = {
            "intent": "research_task",
            "confidence": 1.0,
            "next_action": "run_research",
            "response": "",
            "reason": "bypass_frontdesk_default",
        }
        if verbose:
            console.print("[dim]FrontDesk: bypassed, delegated to Brain[/dim]")

        final_state = run_pipeline(
            raw_input=topic,
            human_in_loop=False,
            hitl_stages=[],
            engine=engine,
            max_steps=max_steps,
            max_errors=max_errors,
            executor_rollouts=executor_rollouts,
            executor_parallelism=executor_parallelism,
            session_id=session_id,
            frontdesk_decision=frontdesk,
            clarification_callback=_make_brain_clarification_callback(prompt_session),
        )
        last_state = final_state
        if final_state.get("assistant_response") and not final_state.get("brain_plan"):
            _print_agent_reply(final_state)
            _save_assistant_reply(session_id, final_state.get("assistant_response", ""))
            continue
        eval_report = evaluate(final_state, print_report=False)
        final_state = build_structured_response(final_state, eval_report.to_dict())
        _print_agent_reply(final_state)
        _save_assistant_reply(session_id, final_state.get("assistant_response", ""))
        if verbose:
            _print_summary(final_state)
        if final_state.get("error_log"):
            unique_errors = list(dict.fromkeys(e.split("\n")[0][:120] for e in final_state["error_log"]))
            console.print(f"[yellow]错误（{len(unique_errors)} 类）[/yellow]")
            for e in unique_errors[:3]:
                console.print(f"  {e}")


@app.command()
def status():
    """查看最新运行的评估报告。"""
    import json, os
    path = "./output/eval_report.json"
    if not os.path.exists(path):
        console.print("[yellow]暂无评估报告，请先运行 research-agent run[/yellow]")
        return
    with open(path) as f:
        report = json.load(f)
    console.print(f"\n总分：{report['overall_score']:.1%}  "
                  f"{'✓ 通过' if report['passed'] else '✗ 未通过'}")
    console.print(f"总结：{report['summary']}")
    for s in report.get("stages", []):
        status_icon = "✓" if s["passed"] else "✗"
        console.print(f"  {status_icon} [{s['stage']:<12}] {s['score']:.1%}")


@tools_app.command("list")
def tools_list_cmd(
    format: str = typer.Option("plain", "--format", help="plain|openai|mcp|anthropic"),
):
    """列出可用工具。"""
    fmt = format.strip().lower()
    if fmt == "plain":
        tools = list_tools()
        if not tools:
            console.print("[yellow]当前没有注册工具[/yellow]")
            return
        console.print(f"[bold]已注册工具（{len(tools)}）[/bold]")
        for t in tools:
            console.print(f"- {t.name}: {t.description}")
        return
    if fmt not in {"openai", "mcp", "anthropic"}:
        console.print("[red]--format 仅支持 plain/openai/mcp/anthropic[/red]")
        raise typer.Exit(1)
    console.print(json.dumps(get_all_specs(fmt), ensure_ascii=False, indent=2))


@tools_app.command("describe")
def tools_describe_cmd(
    name: str = typer.Option(..., "--name", help="工具名"),
    format: str = typer.Option("mcp", "--format", help="openai|mcp|anthropic"),
):
    """查看某个工具的 schema。"""
    fmt = format.strip().lower()
    if fmt not in {"openai", "mcp", "anthropic"}:
        console.print("[red]--format 仅支持 openai/mcp/anthropic[/red]")
        raise typer.Exit(1)
    try:
        tool = get_tool(name)
    except KeyError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)
    if fmt == "openai":
        spec = tool.to_openai_spec()
    elif fmt == "anthropic":
        spec = tool.to_anthropic_spec()
    else:
        spec = tool.to_mcp_spec()
    console.print(json.dumps(spec, ensure_ascii=False, indent=2))


@tools_app.command("call")
def tools_call_cmd(
    name: str = typer.Option(..., "--name", help="工具名"),
    args_json: str = typer.Option("{}", "--args", help='JSON 参数，如 {"query":"llm infra"}'),
):
    """直接调用一个工具。"""
    try:
        tool = get_tool(name)
    except KeyError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)
    try:
        args = json.loads(args_json)
        if not isinstance(args, dict):
            raise ValueError("args must be a JSON object")
    except Exception as e:
        console.print(f"[red]--args 解析失败: {e}[/red]")
        raise typer.Exit(1)
    try:
        result = tool.run(**args)
    except Exception as e:
        console.print(f"[red]工具执行失败: {e}[/red]")
        raise typer.Exit(1)
    console.print(result)


@tools_app.command("mcp-serve")
def tools_mcp_serve_cmd():
    """启动轻量 MCP stdio 服务（JSON-RPC 按行输入输出）。"""
    serve_stdio()


@skills_app.command("list")
def skills_list_cmd():
    """列出内置 skill。"""
    items = list_skills()
    if not items:
        console.print("[yellow]暂无可用 skill[/yellow]")
        return
    console.print(f"[bold]Skills（{len(items)}）[/bold]")
    for name, desc in items:
        console.print(f"- {name}: {desc}")


@skills_app.command("show")
def skills_show_cmd(
    name: str = typer.Option(..., "--name", help="skill 名称"),
):
    """查看 skill 定义。"""
    try:
        s = get_skill(name)
    except KeyError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)
    console.print(json.dumps(s, ensure_ascii=False, indent=2))


@skills_app.command("render")
def skills_render_cmd(
    name: str = typer.Option(..., "--name", help="skill 名称"),
    topic: str = typer.Option(..., "--topic", help="主题"),
    constraints: str = typer.Option("", "--constraints", help="附加约束"),
):
    """渲染 skill 模板为可直接使用的任务 prompt。"""
    try:
        prompt = render_skill(name=name, topic=topic, constraints=constraints)
    except KeyError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)
    console.print(prompt)


def main():
    app()


if __name__ == "__main__":
    main()
