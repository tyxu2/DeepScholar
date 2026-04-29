# State Contract

Every field in `ResearchState` has exactly one **owner** stage.
Only the owner may write it. Other stages may read it.

## Ownership Table

| Field(s) | Owner | Readers |
|----------|-------|---------|
| `raw_input`, `selected_question`, `session_id` | Brain | All |
| `task_profile`, `user_constraints`, `input_paths` | Brain | All |
| `research_brief`, `brain_plan`, `doc_type` | Brain | Analyzer, Writer |
| `found_papers`, `confirmed_papers` | Analyzer | Writer |
| `paper_summaries` | Analyzer (via SubResearcher) | Writer, Critic |
| `search_queries` | Analyzer | — |
| `analysis_brief`, `analysis_key_points` | Analyzer | Writer, Critic |
| `analysis_open_risks`, `analysis_writer_focus` | Analyzer | Writer, Critic |
| `analyzer_stop_reason`, `context_package` | Analyzer | Critic |
| `evidence` | SubResearcher → Analyzer | Writer |
| `evidence_ids` | Writer | — |
| `draft`, `raw_draft`, `draft_md_path` | Writer | Critic |
| `artifacts` | Writer (appends) | All |
| `writer_head_round`, `writer_revision_brief` | Executor (executor.py) | Writer |
| `critique`, `critique_score`, `critique_accepted` | Critic | Executor |
| `critic_gate_passed` | Executor | — |
| `error_log` | Any (append-only) | — |
| `current_stage`, `next` | Each stage on exit | Executor |

## Invariants

- `brain_plan` is set once by Brain and never modified downstream.
- `paper_summaries` is **append-only**; stages may add but not remove entries.
- `evidence` is deduplicated by `deduplicate_evidence()` before being written to state.
- `artifacts` is **append-only**; each write stage appends, never replaces.
- `writer_head_round` is managed exclusively by `executor.py`, not by `writer_node`.

## Deprecated Fields

All fields marked `# deprecated` in `state.py` are kept for legacy `graph.py` / `tools.py`
compatibility. New agents must not read or write them.
