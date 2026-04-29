# Writer Rules

## Mandatory Output Requirements

Every draft **must** contain:

1. **Comparison table** in `related_work` or `method` section
   - Columns: Paper | Method | Key Result | Limitation
   - At least 4 rows (one per paper)

2. **Quantitative results table** in `experiments` section
   - Columns: Method | Metric | Value | Dataset/Setting
   - If no experimental data: use qualitative comparison table

3. **Structured sections** — all six fields required:
   `abstract`, `intro`, `related_work`, `method`, `experiments`, `conclusion`

4. **In-text citations** — every factual claim must reference `[refN]`
   where N corresponds to `paper_summaries[N-1].cite_key`

## Enforcement

- Prompt: `agents.writer.main` contains TABLE REQUIREMENTS section
- Fallback: `markdown_writer._build_paper_table()` auto-injects a paper comparison
  table into `related_work` if no Markdown table (`|`) is detected
- LaTeX: tables rendered via Jinja2 template `templates/paper.tex.j2`

## Section Length Targets (quality profile)

| Section | Min tokens |
|---------|-----------|
| abstract | 150 |
| intro | 400 |
| related_work | 600 (must include table) |
| method | 500 |
| experiments | 400 (must include table) |
| conclusion | 200 |

## Revision Loop

- Max rounds: `writer_head_max_rounds` (default 2, economy mode 1)
- Critic accept threshold: `critique_accept_threshold` (default 7.0/10)
- `writer_revision_brief` + `writer_revision_suggestions` injected on round > 1
- If critic rejects after max rounds: use last draft anyway
