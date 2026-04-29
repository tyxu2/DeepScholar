import os
import re
from research_agent.state import Draft


def _citations_to_markdown(text: str) -> str:
    """Convert LaTeX \\cite{key} to Markdown [key] for readable output."""
    return re.sub(r"\\cite\{([^}]+)\}", r"[\1]", text)


def _make_cite_key(s: dict, used: set) -> str:
    """Generate an author+year cite key matching writer_agent logic."""
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


def render_markdown(
    title: str,
    draft: Draft,
    experiment_results: dict,
    paper_summaries: list,
    heading_style: str = "numbered",
    include_dividers: bool = True,
) -> str:
    results_table = _build_results_table(experiment_results)
    references = _build_references(paper_summaries)

    style = (heading_style or "numbered").strip().lower()
    if style == "plain":
        headings = {
            "abstract": "## Abstract",
            "intro": "## Introduction",
            "related_work": "## Related Work",
            "method": "## Method",
            "experiments": "## Results and Analysis",
            "conclusion": "## Conclusion",
            "references": "## References",
        }
    else:
        headings = {
            "abstract": "## Abstract",
            "intro": "## 1. Introduction",
            "related_work": "## 2. Related Work",
            "method": "## 3. Methodology",
            "experiments": "## 4. Experiments",
            "conclusion": "## 5. Conclusion",
            "references": "## References",
        }

    def render_section(text: str) -> str:
        return _citations_to_markdown(text or "")

    divider = "\n---\n" if include_dividers else "\n"
    parts = [
        f"# {title}",
        divider.strip(),
        f"{headings['abstract']}\n\n{render_section(draft.get('abstract', ''))}",
        divider.strip(),
        f"{headings['intro']}\n\n{render_section(draft.get('intro', ''))}",
        divider.strip(),
        f"{headings['related_work']}\n\n{render_section(draft.get('related_work', ''))}",
        divider.strip(),
        f"{headings['method']}\n\n{render_section(draft.get('method', ''))}",
        divider.strip(),
        f"{headings['experiments']}\n\n{render_section(draft.get('experiments', ''))}\n\n{results_table}".rstrip(),
        divider.strip(),
        f"{headings['conclusion']}\n\n{render_section(draft.get('conclusion', ''))}",
        divider.strip(),
        f"{headings['references']}\n\n{references}",
    ]
    return "\n\n".join(p for p in parts if p)


def _build_results_table(experiment_results: dict) -> str:
    baseline = experiment_results.get("baseline", {})
    improved = experiment_results.get("improved", {})
    plan = experiment_results.get("improvement_plan", "Proposed Method")

    if not baseline and not improved:
        return ""

    rows = ["| Method | Metric | Value |", "|--------|--------|-------|"]
    for k, v in baseline.items():
        if k not in ("timestamp", "stdout"):
            rows.append(f"| Baseline | {k} | {v} |")
    for k, v in improved.items():
        if k not in ("timestamp", "stdout"):
            rows.append(f"| {plan} | {k} | {v} |")

    return "\n".join(rows) + "\n"


def _build_references(paper_summaries: list) -> str:
    """Build a references section with author+year keys matching \\cite{} in the draft."""
    if not paper_summaries:
        return ""
    used: set[str] = set()
    lines = []
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

        year_part = f" ({year})" if year else ""
        lines.append(f"[{key}] {author_str}{year_part}. *{title}*.")
    return "\n".join(lines)


def save_markdown(content: str, output_path: str = "./output/paper.md") -> str:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
    return output_path
