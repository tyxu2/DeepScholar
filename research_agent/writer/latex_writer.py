import os
import re
from jinja2 import Environment, FileSystemLoader
from research_agent.state import Draft


# ── Character escaping ──────────────────────────────────────────────────────────

def _escape_latex_chars(text: str) -> str:
    """Escape LaTeX special characters in plain text (no Markdown conversion)."""
    if not text:
        return ""
    replacements = [
        ("\\", r"\textbackslash{}"),
        ("&",  r"\&"),
        ("%",  r"\%"),
        ("$",  r"\$"),
        ("#",  r"\#"),
        ("_",  r"\_"),
        ("{",  r"\{"),
        ("}",  r"\}"),
        ("~",  r"\textasciitilde{}"),
        ("^",  r"\textasciicircum{}"),
    ]
    for char, escaped in replacements:
        text = text.replace(char, escaped)
    return text


# ── Markdown table → LaTeX tabular ─────────────────────────────────────────────

def _parse_md_row(line: str) -> list[str]:
    """Parse a Markdown pipe table row into a list of cell strings."""
    line = line.strip().strip("|")
    return [c.strip() for c in line.split("|")]


def _convert_md_table(rows: list[str]) -> str:
    """Convert a list of Markdown table row strings to a LaTeX booktabs tabular."""
    if len(rows) < 2:
        return "\n".join(rows)

    header_cells = _parse_md_row(rows[0])
    # rows[1] is the separator line — skip it
    data_rows = [_parse_md_row(r) for r in rows[2:]]

    n_cols = max(len(header_cells), max((len(r) for r in data_rows), default=0))
    col_spec = "l" * n_cols

    def esc(c: str) -> str:
        return _escape_latex_chars(c)

    header_line = " & ".join(esc(c) for c in header_cells)
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        f"{header_line} \\\\",
        r"\midrule",
    ]
    for row in data_rows:
        padded = row + [""] * (n_cols - len(row))
        cells_line = " & ".join(esc(c) for c in padded[:n_cols])
        lines.append(f"{cells_line} \\\\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def _extract_md_tables(text: str, sentinels: dict[str, str]) -> str:
    """
    Scan text for Markdown pipe tables, convert each to LaTeX tabular,
    and replace the original table with a sentinel token.
    The sentinels dict is mutated in-place.
    """
    lines = text.split("\n")
    result: list[str] = []
    i = 0
    idx = len(sentinels)  # unique offset per call

    while i < len(lines):
        line = lines[i]
        # Detect potential table header: line with | ... |
        if (
            re.match(r"^\s*\|.+\|\s*$", line)
            and i + 1 < len(lines)
            and re.match(r"^\s*\|[\s\-|:]+\|\s*$", lines[i + 1])
        ):
            # Collect all consecutive pipe-table rows
            table_rows = [line]
            j = i + 1
            while j < len(lines) and re.match(r"^\s*\|.+\|\s*$", lines[j]):
                table_rows.append(lines[j])
                j += 1
            latex_table = _convert_md_table(table_rows)
            tok = f"\x00TABLE{idx}\x00"
            sentinels[tok] = latex_table
            idx += 1
            result.append(tok)
            i = j
        else:
            result.append(line)
            i += 1

    return "\n".join(result)


# ── Main escape + Markdown → LaTeX ────────────────────────────────────────────

# LaTeX commands that should be preserved verbatim (not char-escaped)
_LATEX_CMD_RE = re.compile(
    r'\\(?:cite|citep|citet|citealt|ref|label|footnote|url|href|emph)\{[^}]*\}'
)


def _latex_escape(text: str) -> str:
    """
    Convert Markdown prose (with embedded \\cite{} references and pipe tables)
    to LaTeX-safe text.

    Processing order:
      1. Protect \\cite{...} and similar commands from char-escaping
      2. Convert Markdown pipe tables → LaTeX tabular (also protected)
      3. Line-by-line: Markdown headings → \\subsection / \\subsubsection,
         then escape special chars, then bold/italic inline markers
      4. Restore all protected segments
    """
    if not text:
        return ""

    sentinels: dict[str, str] = {}
    counter = [0]

    # Step 1 — protect LaTeX commands
    def _protect_cmd(m: re.Match) -> str:
        tok = f"\x00CMD{counter[0]}\x00"
        sentinels[tok] = m.group(0)
        counter[0] += 1
        return tok

    text = _LATEX_CMD_RE.sub(_protect_cmd, text)

    # Step 2 — convert Markdown tables (also becomes sentinels)
    text = _extract_md_tables(text, sentinels)

    # Step 3 — line-by-line processing
    lines = text.split("\n")
    out: list[str] = []
    skip_rest = False

    for line in lines:
        # Strip trailing ## References block
        if re.match(r"^#{1,6}\s*[Rr]eferences", line):
            skip_rest = True
            continue
        if skip_rest:
            continue

        # Markdown headings → LaTeX section commands
        m = re.match(r"^(#{1,6})\s+(.+)$", line)
        if m:
            level = len(m.group(1))
            heading = _escape_latex_chars(m.group(2))
            cmd = "\\subsection" if level <= 2 else "\\subsubsection"
            out.append(f"{cmd}*{{{heading}}}")
            continue

        # Escape special chars in plain text
        escaped = _escape_latex_chars(line)

        # Inline formatting (after char-escape so inserted braces aren't re-escaped)
        escaped = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", escaped)
        escaped = re.sub(
            r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"\\textit{\1}", escaped
        )

        out.append(escaped)

    result = "\n".join(out)

    # Step 4 — restore protected segments
    for tok, original in sentinels.items():
        result = result.replace(tok, original)

    return result


# ── Reference builders ─────────────────────────────────────────────────────────

def _make_cite_key(s: dict, used: set) -> str:
    """Generate an author+year cite key (e.g. kwon2023) for a paper summary."""
    authors = s.get("authors", []) or []
    year = int(s.get("year", 0) or 0)
    if authors:
        last_name = str(authors[0]).split()[-1].lower()
        last_name = re.sub(r"[^a-z]", "", last_name)
    else:
        title_words = str(s.get("title", "") or "").split()
        first_word = title_words[0] if title_words else "unknown"
        last_name = re.sub(r"[^a-z]", "", first_word.lower()[:10])
    if not last_name:
        last_name = "unknown"
    year_str = str(year) if year else ""
    base = f"{last_name}{year_str}"
    key = base
    suffix = ord("a")
    while key in used:
        key = base + chr(suffix)
        suffix += 1
    used.add(key)
    return key


def _build_results_table(experiment_results: dict) -> list[dict]:
    baseline = experiment_results.get("baseline", {})
    improved = experiment_results.get("improved", {})
    plan = experiment_results.get("improvement_plan", "Proposed")

    rows = []
    for k, v in baseline.items():
        if k not in ("timestamp", "stdout"):
            rows.append({"method": "Baseline", "metric": k, "value": str(v)})
    for k, v in improved.items():
        if k not in ("timestamp", "stdout"):
            rows.append({"method": plan, "metric": k, "value": str(v)})
    return rows


def _build_references(paper_summaries: list) -> list[dict]:
    """Build reference list with author+year keys for \\bibitem entries."""
    used: set[str] = set()
    refs = []
    for s in paper_summaries:
        key = _make_cite_key(s, used)
        authors_list = s.get("authors", []) or []
        year = int(s.get("year", 0) or 0)
        title = s.get("title", "Unknown")

        if len(authors_list) == 0:
            author_str = "Unknown"
        elif len(authors_list) == 1:
            author_str = authors_list[0]
        elif len(authors_list) == 2:
            author_str = " and ".join(authors_list[:2])
        else:
            author_str = f"{authors_list[0]} et al."

        refs.append({
            "key": key,
            "title": title,
            "authors": author_str,
            "year": str(year) if year else "",
        })
    return refs


# ── Render entry points ────────────────────────────────────────────────────────

def render_latex(
    title: str,
    draft: Draft,
    experiment_results: dict,
    paper_summaries: list,
    authors: str = "Research Agent",
    templates_dir: str = "./templates",
) -> str:
    env = Environment(
        loader=FileSystemLoader(templates_dir),
        variable_start_string="{{",
        variable_end_string="}}",
        block_start_string="{%",
        block_end_string="%}",
    )
    env.filters["latex_escape"] = _latex_escape

    template = env.get_template("paper.tex.j2")
    return template.render(
        title=title,
        authors=authors,
        abstract=draft.get("abstract", ""),
        intro=draft.get("intro", ""),
        related_work=draft.get("related_work", ""),
        method=draft.get("method", ""),
        experiments=draft.get("experiments", ""),
        conclusion=draft.get("conclusion", ""),
        results_table=_build_results_table(experiment_results),
        references=_build_references(paper_summaries),
    )


def save_latex(content: str, output_path: str = "./output/paper.tex") -> str:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
    return output_path
