# Supervisor Rules

## Tools Available

| Tool | Purpose | Rule |
|------|---------|------|
| `think_tool` | Strategic reasoning only | No external calls; use before conduct_research |
| `conduct_research` | Spawn parallel SubResearchers | Topic must be ≥2 sentences with specific terms |
| `research_complete` | End research phase | Rejected if read < min_summaries |
| `search_papers` | Lightweight direct search | For gap-filling, not primary research |
| `read_papers` | Read specific papers | For targeted deep-reads |

## Loop Invariants

- `max_steps` default: 6 (set via `agent_capabilities.analyzer.max_steps`)
- `max_concurrent` default: 3 parallel sub-researchers per conduct_research call
- `max_errors` default: 3 before `error_budget` stop
- `min_summaries`: 10 (literature_review) | 6 (default) | 4 (repo/task) | explicit `paper_limit`

## Stopping Conditions

| `stop_reason` | Meaning |
|---------------|---------|
| `research_complete` | LLM called `research_complete` and evidence threshold met |
| `no_tool_calls` | LLM returned text with no tool calls (consider this an error if early) |
| `budget_exhausted` | Reached `max_steps` without completing |
| `error_budget` | Hit `max_errors` consecutive tool failures |

## Context Compression

- Compresses after every `_SUPERVISOR_COMPRESS_EVERY = 2` conduct_research rounds
- Keeps: system prompt + initial user goal + last 4 messages
- Injects: rolling summary with paper counts + covered topics
- Purpose: prevents O(rounds) context growth

## Prompt Injection Points

The supervisor system prompt is rendered from `agents.analyzer.supervisor` with:
- `{goal}` — the research question
- `{research_brief_json}` — Brain's full brief including sub_questions
- `{min_summaries}` — the evidence threshold
- `{active_tools}` — comma-separated available tools

## What Supervisor Must NOT Do

- Output bare JSON text (it outputs tool calls)
- Call `research_complete` before meeting `min_summaries`
- Delegate more than `max_concurrent` parallel researchers in one step
- Use `think_tool` as a substitute for actual research
