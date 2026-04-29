# Research Agent — Architecture Rules Index

> This file is the authoritative reference for how this agent system is structured.
> Sub-documents below define binding contracts. When in doubt, these rules win.

## Sub-documents

| File | What it governs |
|------|-----------------|
| [rules/state_contract.md](rules/state_contract.md) | Which fields each stage owns, reads, and may write |
| [rules/evidence_contract.md](rules/evidence_contract.md) | Evidence TypedDict schema, flow path, dedup rules |
| [rules/supervisor_rules.md](rules/supervisor_rules.md) | Supervisor loop invariants, tool use, stopping conditions |
| [rules/writer_rules.md](rules/writer_rules.md) | Output format requirements (tables, citations, structure) |

---

## Pipeline Overview

```
Brain → Analyzer Supervisor → Writer ↔ Critic
```

| Stage | Agent | Key output |
|-------|-------|------------|
| Brain | `brain_node` | `research_brief` + `brain_plan` + `sub_questions` |
| Analyzer | `analyzer_node` | `paper_summaries` + `evidence` + `analysis_brief` |
| Writer | `writer_node` | `draft` + `paper.md` + `paper.tex` |
| Critic | `critic_node` | `critique_score` + `critique_accepted` |

## Hard Rules

1. **State ownership is exclusive.** Only the owning stage writes to its fields (see `state_contract.md`).
2. **Evidence flows forward only.** SubResearcher → Analyzer → Writer. Never backwards.
3. **No raw paper text in Supervisor context.** Compress to Evidence cards before passing to Supervisor.
4. **Writer must produce at least one comparison table.** If LLM omits it, `markdown_writer.py` injects a fallback.
5. **Supervisor never outputs JSON text.** It outputs tool calls. `STRICT_JSON_RULE` does not apply.
6. **Context compression is mandatory at scale.** SubResearcher compresses after 3 steps; Supervisor compresses every 2 conduct_research rounds.

## Model Routing

Configured via env vars. See `research_agent/llm/router.py`.

```
RESEARCH_AGENT_MODEL_PRIMARY=gpt-5
RESEARCH_AGENT_COST_PROFILE=quality|balanced|budget
RESEARCH_AGENT_ECONOMY_MODE=1   # forces budget + single writer round
```

## Observability

Per-session `timeline.jsonl` written to `output/{session_id}/`.
Token summary printed at pipeline end.
See `research_agent/observability/trace_logger.py`.
