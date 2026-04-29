from research_agent.utils.json_parser import parse_json
import json
import os
from dataclasses import dataclass, field, asdict
from research_agent.state import ResearchState
from research_agent.llm.router import call_llm, system
from research_agent.prompts import render_prompt

# ── 数据结构 ───────────────────────────────────────────────────────────────────

@dataclass
class StageResult:
    stage: str
    passed: bool
    score: float          # 0.0 - 1.0
    details: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)


@dataclass
class EvalReport:
    overall_score: float
    passed: bool
    stage_results: list[StageResult] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> dict:
        return {
            "overall_score": round(self.overall_score, 3),
            "passed": self.passed,
            "summary": self.summary,
            "stages": [asdict(r) for r in self.stage_results],
        }

    def print(self):
        bar = "=" * 50
        print(f"\n{bar}")
        print("  评估报告")
        print(bar)
        print(f"  总分：{self.overall_score:.1%}  {'✓ 通过' if self.passed else '✗ 未通过'}")
        print(f"  总结：{self.summary}")
        print()
        for r in self.stage_results:
            status = "✓" if r.passed else "✗"
            print(f"  {status} [{r.stage:<12}] {r.score:.1%}", end="")
            if r.warnings:
                print(f"  ⚠ {r.warnings[0]}", end="")
            print()
            for k, v in r.details.items():
                print(f"       {k}: {v}")
        print(bar)


# ── 各阶段评估函数 ─────────────────────────────────────────────────────────────

def eval_search(state: ResearchState) -> StageResult:
    """评估找论文阶段：数量、下载率。"""
    found = state.get("found_papers", [])
    confirmed = state.get("confirmed_papers", [])
    downloaded = [p for p in found if p.get("pdf_path") and os.path.exists(p["pdf_path"])]

    count_ok = len(found) >= 5
    download_rate = len(downloaded) / len(found) if found else 0
    download_ok = download_rate >= 0.5

    score = (
        0.4 * (min(len(found), 20) / 20) +
        0.4 * download_rate +
        0.2 * (1.0 if confirmed else 0.5)
    )
    warnings = []
    if not count_ok:
        warnings.append(f"论文数量偏少（{len(found)} 篇，建议 ≥ 5）")
    if download_rate < 0.5:
        warnings.append(f"PDF 下载率低（{download_rate:.0%}）")

    return StageResult(
        stage="search",
        passed=count_ok and download_ok,
        score=score,
        details={
            "找到论文数": len(found),
            "PDF下载率": f"{download_rate:.0%}",
            "确认精读数": len(confirmed),
        },
        warnings=warnings,
    )


def eval_read(state: ResearchState) -> StageResult:
    """评估读论文阶段：摘要完整性与覆盖率。"""
    summaries = state.get("paper_summaries", [])
    found = state.get("found_papers", [])

    if not summaries:
        return StageResult("read", False, 0.0, {}, ["没有生成任何摘要"])

    # 完整性：每个摘要包含必要字段
    complete = [
        s for s in summaries
        if s.get("problem") and s.get("method") and s.get("result")
    ]
    completeness = len(complete) / len(summaries)

    # 覆盖率：成功读取的论文 / 尝试读取的论文
    coverage = len(summaries) / max(len(found), 1)

    score = 0.6 * completeness + 0.4 * coverage
    warnings = []
    if completeness < 0.7:
        warnings.append(f"摘要完整率偏低（{completeness:.0%}）")

    return StageResult(
        stage="read",
        passed=completeness >= 0.6,
        score=score,
        details={
            "成功解析": f"{len(summaries)}/{len(found)}",
            "摘要完整率": f"{completeness:.0%}",
            "建议深入论文数": sum(1 for s in summaries if s.get("worth_reproducing")),
        },
        warnings=warnings,
    )


def eval_coding_plan(state: ResearchState) -> StageResult:
    """评估 coding_plan 阶段：是否产出给 Coding Agent 的可执行交接物。"""
    artifacts = state.get("artifacts", [])
    ids = {a.get("id", "") for a in artifacts}
    has_plan = "artifact-coding-plan" in ids
    has_codex_prompt = "artifact-codex-prompt" in ids
    has_claude_prompt = "artifact-claude-prompt" in ids

    score = (
        0.4 * (1.0 if has_plan else 0.0)
        + 0.3 * (1.0 if has_codex_prompt else 0.0)
        + 0.3 * (1.0 if has_claude_prompt else 0.0)
    )

    warnings = []
    if not has_plan:
        warnings.append("缺少 coding plan 文档")
    if not has_codex_prompt:
        warnings.append("缺少 Codex Prompt")
    if not has_claude_prompt:
        warnings.append("缺少 Claude Code Prompt")

    return StageResult(
        stage="coding_plan",
        passed=has_plan and has_codex_prompt and has_claude_prompt,
        score=score,
        details={
            "coding_plan": "已生成" if has_plan else "未生成",
            "codex_prompt": "已生成" if has_codex_prompt else "未生成",
            "claude_prompt": "已生成" if has_claude_prompt else "未生成",
        },
        warnings=warnings,
    )


def eval_github(state: ResearchState) -> StageResult:
    """评估 github 阶段：是否检索到可用开源仓库线索。"""
    repo = (state.get("github_repo_url") or "").strip()
    passed = bool(repo)
    score = 1.0 if passed else 0.35
    warnings = [] if passed else ["未找到可用仓库（可能因关键词过窄或API受限）"]
    return StageResult(
        stage="github",
        passed=passed,
        score=score,
        details={"repo_url": repo or "未命中"},
        warnings=warnings,
    )


def eval_experiment(state: ResearchState) -> StageResult:
    """评估实验阶段：代码执行成功、结果有效。"""
    results = state.get("experiment_results", {})
    improved_code = state.get("improved_code", "")

    baseline = results.get("baseline", {})
    improved = results.get("improved", {})

    has_baseline = bool(baseline) and "stdout" not in baseline
    has_improved = bool(improved) and "stdout" not in improved
    has_improvement = bool(improved_code)

    # 检查是否有数值指标
    numeric_baseline = {
        k: v for k, v in baseline.items()
        if isinstance(v, (int, float)) and k not in ("timestamp",)
    }
    has_numeric = bool(numeric_baseline)

    score = (
        0.4 * (1.0 if has_baseline else 0.0) +
        0.25 * (1.0 if has_improved else 0.0) +
        0.25 * (1.0 if has_numeric else 0.0) +
        0.1 * (1.0 if has_improvement else 0.0)
    )

    warnings = []
    if not has_baseline:
        warnings.append("基线实验未成功执行")
    if not has_numeric:
        warnings.append("未解析到数值指标，结果可能存储在 stdout 中")

    return StageResult(
        stage="experiment",
        passed=has_baseline,
        score=score,
        details={
            "基线实验": "成功" if has_baseline else "失败",
            "改进实验": "成功" if has_improved else "未运行",
            "数值指标": str(numeric_baseline) if numeric_baseline else "无",
        },
        warnings=warnings,
    )


def eval_writing(state: ResearchState) -> StageResult:
    """评估写论文阶段：章节完整性、内容质量、LLM深度评估。"""
    draft = state.get("draft", {})
    md_path = state.get("draft_md_path", "")
    latex_path = state.get("draft_latex_path", "")
    critic_score = state.get("critique_score", 0.0)
    plan = state.get("brain_plan", [])
    critic_expected = "critic" in plan
    critic_done = state.get("revision_count", 0) > 0

    required_sections = ["intro", "related_work", "method", "experiments", "conclusion", "abstract"]
    present = [s for s in required_sections if draft.get(s) and len(draft[s]) > 50]
    completeness = len(present) / len(required_sections)

    # 字数统计
    total_words = sum(len(draft.get(s, "").split()) for s in required_sections)

    # LLM 深度评估
    llm_scores = {"structure": 0, "depth": 0, "language": 0}
    llm_comment = ""
    try:
        response = call_llm(
            "critique",
            [
                system(
                    render_prompt(
                        "eval.paper_quality",
                        abstract=draft.get("abstract", "")[:400],
                        intro=draft.get("intro", "")[:300],
                        method=draft.get("method", "")[:300],
                    )
                )
            ]
        )
        data = parse_json(response)
        llm_scores = {k: data.get(k, 0) for k in ("structure", "depth", "language")}
        llm_comment = data.get("comment", "")
    except Exception:
        pass

    llm_avg = sum(llm_scores.values()) / 3 / 10 if any(llm_scores.values()) else 0.5

    if critic_expected and critic_done:
        score = (
            0.3 * completeness +
            0.3 * (critic_score / 10) +
            0.4 * llm_avg
        )
        passed = completeness >= 0.8 and critic_score >= 7.0
    elif critic_expected and not critic_done:
        score = 0.4 * completeness + 0.6 * llm_avg
        passed = completeness >= 0.75 and llm_avg >= 0.65
    else:
        score = 0.45 * completeness + 0.55 * llm_avg
        passed = completeness >= 0.75 and llm_avg >= 0.6

    warnings = []
    missing = [s for s in required_sections if s not in present]
    if missing:
        warnings.append(f"缺少章节：{', '.join(missing)}")
    if total_words < 800:
        warnings.append(f"论文字数偏少（{total_words} 词）")
    if critic_expected and not critic_done:
        warnings.append("计划包含 critic 但本次未执行，采用降级写作评分")

    return StageResult(
        stage="writing",
        passed=passed,
        score=score,
        details={
            "完整章节": f"{len(present)}/{len(required_sections)}",
            "总词数": total_words,
            "Critic评分": f"{critic_score}/10" if critic_done else "N/A（未执行）",
            "Critic计划": "已启用" if critic_expected else "未启用",
            "修改轮次": state.get("revision_count", 0),
            "LLM评估": f"结构{llm_scores['structure']} 深度{llm_scores['depth']} 语言{llm_scores['language']}",
            "LLM评语": llm_comment[:60] if llm_comment else "N/A",
            "Markdown": md_path if os.path.exists(md_path) else "未生成",
            "LaTeX": latex_path if latex_path and os.path.exists(latex_path) else "未生成",
        },
        warnings=warnings,
    )


# ── 主入口 ─────────────────────────────────────────────────────────────────────

def evaluate(
    state: ResearchState,
    save_path: str = "./output/eval_report.json",
    print_report: bool = True,
) -> EvalReport:
    """
    对完整流水线的运行结果进行评估。

    Args:
        state: 流水线最终状态
        save_path: 评估报告保存路径

    Returns:
        EvalReport 对象
    """
    if print_report:
        print("\n[Evaluator] 开始评估...")

    # 根据 brain_plan 决定哪些阶段需要评估
    plan = state.get("brain_plan", [])
    need_github = "github" in plan
    need_write = "write" in plan or "critic" in plan

    stage_results = [eval_search(state), eval_read(state)]

    if need_github:
        stage_results.append(eval_github(state))
    if need_write:
        stage_results.append(eval_writing(state))

    # 动态权重：按实际执行的阶段分配
    if need_github and need_write:
        weights = [0.22, 0.28, 0.20, 0.30]
    elif need_github:
        weights = [0.28, 0.37, 0.35]
    elif need_write:
        weights = [0.25, 0.35, 0.40]
    else:
        weights = [0.40, 0.60]

    overall = sum(r.score * w for r, w in zip(stage_results, weights))
    passed = overall >= 0.6 and all(r.passed for r in stage_results)

    # LLM 生成总结
    details_str = "\n".join([
        f"{r.stage}: {r.score:.0%} ({'通过' if r.passed else '未通过'}) "
        f"| {r.details}"
        for r in stage_results
    ])
    try:
        summary = call_llm(
            "summarization",
            [system(render_prompt("eval.summary", details=details_str))]
        )
    except Exception:
        passing = sum(1 for r in stage_results if r.passed)
        summary = f"{passing}/{len(stage_results)} 个阶段通过，总体得分 {overall:.0%}。"

    report = EvalReport(
        overall_score=overall,
        passed=passed,
        stage_results=stage_results,
        summary=summary,
    )

    # 保存报告
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)

    if print_report:
        report.print()
        print(f"\n  报告已保存：{save_path}")
    return report
