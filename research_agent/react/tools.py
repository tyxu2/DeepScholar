"""
Open-Tool definitions for ReAct pipeline.

Each tool exposes:
  name / description / input_schema  → LLM sees this via tool_use API
  fn(tool_input, state) → dict       → actual execution, returns structured output
  state_updater(state, output) → state → merges output back into ResearchState

Write / critic remain as terminal stages (too stateful to be tools).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Callable

from research_agent.state import ResearchState
from research_agent.tools import get_tool as get_registered_tool
from research_agent.utils.json_parser import parse_json


@dataclass
class ToolSpec:
    name: str
    description: str
    input_schema: dict          # JSON Schema passed to LLM
    fn: Callable[[dict, ResearchState], dict]
    state_updater: Callable[[ResearchState, dict], ResearchState]
    estimated_cost: float = 0.1


def to_litellm_tool(spec: ToolSpec) -> dict:
    """Convert ToolSpec to OpenAI-compatible tool schema (litellm format)."""
    return {
        "type": "function",
        "function": {
            "name": spec.name,
            "description": spec.description,
            "parameters": spec.input_schema,
        },
    }


def _noop_updater(state: ResearchState, output: dict) -> ResearchState:
    return state


def _call_registered_tool(tool_name: str, tool_input: dict) -> dict:
    tool = get_registered_tool(tool_name)
    raw = tool.run(**tool_input)
    try:
        parsed = parse_json(raw)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    return {"text": raw}


# ── done ──────────────────────────────────────────────────────────────────────

def _done_fn(tool_input: dict, state: ResearchState) -> dict:
    return _call_registered_tool("done", tool_input)


def _done_updater(state: ResearchState, output: dict) -> ResearchState:
    label = str(output.get("confidence", "medium")).strip().lower()
    if label not in {"high", "medium", "low"}:
        label = "medium"
    score = {"high": 0.85, "medium": 0.65, "low": 0.4}[label]
    return {
        **state,
        "assistant_response": str(output.get("result", "")).strip(),
        "confidence_label": label,  # type: ignore[typeddict-item]
        "confidence_score": score,
    }


# ── save_text_artifact ────────────────────────────────────────────────────────

def _save_text_artifact_fn(tool_input: dict, state: ResearchState) -> dict:
    return _call_registered_tool("save_text_artifact", tool_input)


def _save_text_artifact_updater(state: ResearchState, output: dict) -> ResearchState:
    path = str(output.get("path", "")).strip()
    if not path:
        return state
    artifacts = list(state.get("artifacts", []))
    artifacts.append({
        "id": f"artifact-tool-{len(artifacts) + 1}",
        "kind": "text",
        "title": path.split("/")[-1],
        "source_stage": "tool_call",
        "content_ref": path,
        "evidence_ids": list(state.get("evidence_ids", [])),
    })
    return {**state, "artifacts": artifacts}


# ── search_papers helpers ──────────────────────────────────────────────────────

def _is_duplicate_query(query: str, used_queries: list[str]) -> str:
    """
    Return the conflicting query if this query is too similar to an already-used
    one, otherwise return empty string.
    Checks: exact normalized match, or near-identical token sets.

    Note: We intentionally avoid substring containment checks because they can
    over-prune useful expanded queries during repair rounds.
    """
    q = " ".join(re.findall(r"[a-z0-9]+", query.lower()))
    if not q:
        return ""
    q_tokens = set(q.split())

    for used in used_queries:
        u = " ".join(re.findall(r"[a-z0-9]+", str(used).lower()))
        if not u:
            continue
        if q == u:
            return used

        u_tokens = set(u.split())
        if not q_tokens or not u_tokens:
            continue

        overlap_ratio = len(q_tokens & u_tokens) / max(len(q_tokens), len(u_tokens))
        if overlap_ratio >= 0.9 and abs(len(q_tokens) - len(u_tokens)) <= 1:
            return used
    return ""


def _simplify_query_for_search(query: str) -> str:
    """Reduce boolean-heavy queries to plain keyword queries for brittle APIs."""
    tokens = re.findall(r"[a-zA-Z0-9-]{2,}", (query or "").lower())
    if not tokens:
        return ""
    drop = {
        "and", "or", "not", "site", "http", "https", "www", "com",
        "arxiv", "ieee", "acm", "osdi", "nsdi", "mlsys", "sosp", "eurosys", "atc",
    }
    slim: list[str] = []
    for t in tokens:
        if t in drop:
            continue
        if re.fullmatch(r"\d{4}", t):
            continue
        slim.append(t)
    return " ".join(slim[:14]).strip()


def _filter_by_relevance(
    papers: list[dict],
    topic: str,
    caller: str = "?",
) -> tuple[list[dict], int]:
    """
    Batch-score paper titles+abstracts against topic using a fast LLM call.
    Returns (relevant_papers, n_filtered_out).
    Threshold: score >= 3/10 to be kept.
    Falls back to keeping all papers on any error.
    """
    if not papers or not topic:
        return papers, 0

    from research_agent.llm.router import call_llm_json, system
    from research_agent.utils.json_parser import parse_json as _pj

    entries = "\n".join(
        f"{i + 1}. {p.get('title', '')} | {str(p.get('abstract', '') or '')[:120]}"
        for i, p in enumerate(papers)
    )
    prompt = (
        f"Rate each paper's relevance to the research topic on a 0–10 scale.\n"
        f"Topic: \"{topic}\"\n"
        f"0 = completely off-topic, 5 = borderline, 10 = directly relevant.\n"
        f"Only return JSON: {{\"scores\": [<int>, ...]}} — one integer per paper, same order.\n\n"
        f"{entries}"
    )
    try:
        resp = call_llm_json(
            "summarization",
            [{"role": "user", "content": prompt}],
            max_tokens=20480,
        )
        data = _pj(resp)
        scores = data.get("scores", [])
        if len(scores) == len(papers):
            kept = [p for p, s in zip(papers, scores) if int(s) >= 3]
            filtered = len(papers) - len(kept)
            if filtered:
                print(f"  [{caller}][Relevance] {filtered}/{len(papers)} filtered for: {topic[:50]}")
            return kept, filtered
    except Exception as e:
        print(f"  [{caller}][Relevance] scoring failed ({e}), keeping all")
    return papers, 0


# ── search_papers ──────────────────────────────────────────────────────────────

def _search_papers_fn(tool_input: dict, state: ResearchState) -> dict:
    """Search via registered paper_search tool (arXiv + Semantic Scholar)."""
    query = str(tool_input.get("query", "")).strip()
    max_results = min(int(tool_input.get("max_results", 10)), 20)
    if not query:
        return {"papers": [], "count": 0, "query_used": query}

    # ── 1. Query deduplication ─────────────────────────────────────────────────
    used_queries = list(state.get("search_queries", []) or [])
    conflict = _is_duplicate_query(query, used_queries)
    if conflict:
        msg = f"Query skipped — too similar to already-used query: '{conflict}'"
        print(f"  [search_papers] {msg}")
        return {
            "papers": [], "count": 0, "query_used": query,
            "skipped": True, "skip_reason": msg,
        }

    # ── 2. Fetch ───────────────────────────────────────────────────────────────
    query_used = query
    result = _call_registered_tool("paper_search", {"query": query, "limit": max_results})
    papers = [
        {
            "title": p.get("title", ""),
            "abstract": str(p.get("abstract", "") or "")[:300],
            "url": p.get("url", ""),
            "citations": p.get("citations", 0),
            "year": p.get("year", 0),
            "pdf_path": "",
        }
        for p in result.get("papers", [])
    ]

    # Some sources are sensitive to complex boolean syntax; retry once with a simplified query.
    if not papers:
        simplified = _simplify_query_for_search(query)
        if simplified and simplified.lower() != query.lower():
            conflict2 = _is_duplicate_query(simplified, used_queries + [query])
            if not conflict2:
                result2 = _call_registered_tool("paper_search", {"query": simplified, "limit": max_results})
                papers2 = [
                    {
                        "title": p.get("title", ""),
                        "abstract": str(p.get("abstract", "") or "")[:300],
                        "url": p.get("url", ""),
                        "citations": p.get("citations", 0),
                        "year": p.get("year", 0),
                        "pdf_path": "",
                    }
                    for p in result2.get("papers", [])
                ]
                if papers2:
                    papers = papers2
                    query_used = simplified

    # ── 3. Relevance filter (literature / research tasks) ─────────────────────
    constraints = state.get("user_constraints", {}) or {}
    profile = state.get("task_profile", "general_research")
    run_filter = constraints.get(
        "strict_literature_relevance",
        profile in {"literature_review", "paper_from_implementation", "general_research"},
    )
    filtered_out = 0
    if run_filter and papers:
        topic = str(state.get("selected_question", "") or state.get("raw_input", "")).strip()
        caller = state.get("_log_caller", state.get("current_stage", "Executor"))
        papers, filtered_out = _filter_by_relevance(papers, topic, caller=caller)

    return {
        "papers": papers,
        "count": len(papers),
        "query_used": query_used,
        "filtered_out": filtered_out,
    }


def _search_papers_updater(state: ResearchState, output: dict) -> ResearchState:
    # Skipped queries still record themselves so dedup keeps working
    queries = list(state.get("search_queries", []))
    q = str(output.get("query_used", "") or "").strip()
    if q and q not in queries:
        queries.append(q)

    if output.get("skipped"):
        return {**state, "search_queries": queries}

    existing = list(state.get("found_papers", []))
    seen = {str(p.get("title", "")).lower() for p in existing}
    for paper in output.get("papers", []):
        title_key = str(paper.get("title", "")).lower()
        if not title_key or title_key in seen:
            continue
        seen.add(title_key)
        existing.append({
            "title": paper.get("title", ""),
            "abstract": paper.get("abstract", ""),
            "url": paper.get("url", ""),
            "pdf_path": paper.get("pdf_path", ""),
            "year": paper.get("year", 0),
            "authors": paper.get("authors", []),
            "citations": paper.get("citations", 0),
        })
    existing.sort(key=lambda p: (p.get("citations", 0), p.get("year", 0)), reverse=True)

    total_filtered = int(state.get("relevance_filtered_count", 0) or 0) + int(output.get("filtered_out", 0) or 0)
    return {
        **state,
        "found_papers": existing,
        "search_queries": queries,
        "relevance_filtered_count": total_filtered,
    }


SEARCH_PAPERS = ToolSpec(
    name="search_papers",
    description=(
        "Search academic papers on arXiv and Semantic Scholar. "
        "Returns papers with title, abstract, citation count, and year. "
        "Use different queries to broaden coverage. Do NOT repeat an already-used query."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "English search query. Combine method + task + domain terms.",
            },
            "max_results": {
                "type": "integer",
                "description": "Max papers to retrieve (5–20). Default 10.",
                "default": 10,
            },
        },
        "required": ["query"],
    },
    fn=_search_papers_fn,
    state_updater=_search_papers_updater,
    estimated_cost=0.30,
)


# ── read_papers ────────────────────────────────────────────────────────────────

def _read_papers_fn(tool_input: dict, state: ResearchState) -> dict:
    """Extract structured summaries from found_papers using LLM."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from research_agent.llm.router import call_llm_json, system
    from research_agent.prompts import render_prompt

    papers = list(state.get("found_papers", []))
    limit = min(int(tool_input.get("limit", len(papers))), len(papers), 30)
    to_read = papers[:limit]

    def _summarize(paper: dict) -> dict | None:
        text = (
            f"Title: {paper.get('title', '')}\n"
            f"Abstract: {paper.get('abstract', '') or '(no abstract available)'}\n"
        )[:2000]
        try:
            resp = call_llm_json(
                "summarization",
                [system(render_prompt("agents.reader.extract", text=text))],
                max_tokens=4000,
            )
            data = parse_json(resp)
            if isinstance(data, dict):
                return {"title": paper.get("title", ""), **data}
        except Exception:
            pass
        return {
            "title": paper.get("title", ""),
            "problem": "(extraction failed)",
            "method": "",
            "result": "",
            "contributions": [],
            "limitations": "Not mentioned",
            "worth_reproducing": False,
            "code_url": "",
        }

    summaries: list[dict] = []
    errors = 0
    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = {ex.submit(_summarize, p): p for p in to_read}
        for future in as_completed(futures):
            try:
                result = future.result()
                if result:
                    summaries.append(result)
            except Exception:
                errors += 1

    return {"summaries": summaries, "count": len(summaries), "errors": errors}


def _read_papers_updater(state: ResearchState, output: dict) -> ResearchState:
    existing = list(state.get("paper_summaries", []))
    seen = {str(s.get("title", "")).lower() for s in existing}
    new_summaries = []
    for s in output.get("summaries", []):
        if str(s.get("title", "")).lower() not in seen:
            seen.add(str(s.get("title", "")).lower())
            existing.append(s)
            new_summaries.append(s)
    key_paper = state.get("key_paper", "")
    if not key_paper and existing:
        key_paper = existing[0].get("title", "")

    # Index new summaries into local PaperStore (ChromaDB + BM25)
    if new_summaries:
        try:
            from research_agent.memory.paper_store import PaperStore
            store = PaperStore(persist_dir="./paper_index")
            for s in new_summaries:
                store.add_paper(s)
            store.save()
            print(f"  [PaperStore] Indexed {len(new_summaries)} new paper(s) → ./paper_index (total={len(store)})")
        except Exception as e:
            print(f"  [PaperStore] Indexing failed (non-fatal): {e}")

    return {**state, "paper_summaries": existing, "key_paper": key_paper}


READ_PAPERS = ToolSpec(
    name="read_papers",
    description=(
        "Extract structured summaries (problem, method, results, limitations) from fetched papers. "
        "Requires search_papers to have been called first. "
        "Call once — no need to repeat unless new papers were added."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "limit": {
                "type": "integer",
                "description": "Number of papers to read from the found list. Default: all.",
            },
        },
        "required": [],
    },
    fn=_read_papers_fn,
    state_updater=_read_papers_updater,
    estimated_cost=0.45,
)


# ── optimize_queries ───────────────────────────────────────────────────────────

def _optimize_queries_fn(tool_input: dict, state: ResearchState) -> dict:
    """Generate improved queries using LLM based on current evidence gaps."""
    from research_agent.llm.router import call_llm_json, system
    from research_agent.prompts import render_prompt

    question = str(
        tool_input.get("question", "") or state.get("selected_question") or state.get("raw_input", "")
    ).strip()
    previous_queries = list(tool_input.get("previous_queries") or state.get("search_queries", []))
    papers = list(state.get("found_papers", []))

    # Strip output-format meta-terms so the LLM sees a clean topic, not "vllm survey 4000 words"
    _q_meta_stop = {
        "words", "word", "survey", "review", "report", "paper", "papers",
        "latex", "tex", "please", "write", "generate", "create",
    }
    _q_tokens = [t for t in re.findall(r"[a-zA-Z]{3,}", question.lower()) if t not in _q_meta_stop]
    clean_question = " ".join(_q_tokens[:6]).strip() or question

    input_data = {
        "question": clean_question,
        "previous_queries": previous_queries[:6],
        "papers_found": len(papers),
        "evidence_gap": "insufficient papers" if len(papers) < 5 else "low citation quality",
    }
    try:
        resp = call_llm_json(
            "planning",
            [system(render_prompt("agents.query_optimizer.optimize",
                                  input_json=json.dumps(input_data, ensure_ascii=False)))],
            max_tokens=4000,
        )
        data = parse_json(resp)
        queries = [str(q).strip() for q in (data.get("queries", []) or []) if str(q).strip()][:4]
    except Exception:
        # Strip meta/format terms so fallback queries use only the core topic
        _meta_stop = {
            "words", "word", "survey", "review", "report", "paper", "papers",
            "latex", "tex", "please", "write", "generate", "create",
        }
        _core_tokens = [
            t for t in re.findall(r"[a-zA-Z]{3,}", question.lower())
            if t not in _meta_stop and not t.isdigit()
        ][:5]
        _core = " ".join(_core_tokens).strip() or question
        queries = [
            f"{_core} survey",
            f"{_core} recent advances",
            f"{_core} benchmark",
        ]
    return {"queries": queries or [question], "strategy": "deficit-driven"}


def _optimize_queries_updater(state: ResearchState, output: dict) -> ResearchState:
    queries = output.get("queries", [])
    # Keep optimized suggestions separate; search_queries should only contain executed queries.
    return {**state, "optimized_queries": queries}


OPTIMIZE_QUERIES = ToolSpec(
    name="optimize_queries",
    description=(
        "Generate better search queries based on current evidence gaps "
        "(too few papers, low citation count, poor coverage). "
        "After calling this, call search_papers again with the returned queries."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "Research question. Leave empty to use current topic.",
            },
            "previous_queries": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Queries already tried, to avoid repetition.",
            },
        },
        "required": [],
    },
    fn=_optimize_queries_fn,
    state_updater=_optimize_queries_updater,
    estimated_cost=0.08,
)


# ── find_github_repo ───────────────────────────────────────────────────────────

def _find_github_repo_fn(tool_input: dict, state: ResearchState) -> dict:
    """Find GitHub repo for a paper using the registered github_repo_search tool."""
    paper_title = str(
        tool_input.get("paper_title", "") or state.get("key_paper", "")
    ).strip()
    if not paper_title:
        return {"repo_url": "", "confidence": 0.0, "reason": "no paper title"}

    result = _call_registered_tool("github_repo_search", {"query": paper_title, "limit": 5})
    repos = result.get("repos", []) or []
    if not repos:
        return {"repo_url": "", "confidence": 0.0, "reason": "no repos found"}

    best = repos[0]
    repo_url = str(best.get("url", "")).strip()
    confidence = 0.6 if repo_url else 0.0
    reason = f"top starred repo: {best.get('full_name', '')}"
    return {"repo_url": repo_url, "confidence": confidence, "reason": reason}


def _find_github_repo_updater(state: ResearchState, output: dict) -> ResearchState:
    repo_url = str(output.get("repo_url", "") or "").strip()
    return {**state, "github_repo_url": repo_url} if repo_url else state


FIND_GITHUB_REPO = ToolSpec(
    name="find_github_repo",
    description=(
        "Find the GitHub repository implementing a specific paper. "
        "Call after read_papers when the task involves code reproduction or repo analysis. "
        "Returns repo URL and confidence score."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "paper_title": {
                "type": "string",
                "description": "Title of the paper. Leave empty to use the key paper.",
            },
        },
        "required": [],
    },
    fn=_find_github_repo_fn,
    state_updater=_find_github_repo_updater,
    estimated_cost=0.15,
)


# ── github_repo_search updater ────────────────────────────────────────────────

def _github_repo_search_fn(tool_input: dict, state: ResearchState) -> dict:
    return _call_registered_tool("github_repo_search", tool_input)


def _github_repo_search_updater(state: ResearchState, output: dict) -> ResearchState:
    repos = output.get("repos", []) or []
    if not repos:
        return state
    repo_url = str((repos[0] or {}).get("url", "")).strip()
    return {**state, "github_repo_url": repo_url} if repo_url else state


def _paper_search_fn(tool_input: dict, state: ResearchState) -> dict:
    return _call_registered_tool("paper_search", tool_input)


def _paper_search_updater(state: ResearchState, output: dict) -> ResearchState:
    existing = list(state.get("found_papers", []))
    seen = {str(p.get("title", "")).lower() for p in existing}
    for paper in output.get("papers", []) or []:
        title_key = str(paper.get("title", "")).lower()
        if not title_key or title_key in seen:
            continue
        seen.add(title_key)
        existing.append({
            "title": paper.get("title", ""),
            "abstract": "",
            "url": paper.get("url", ""),
            "pdf_path": "",
            "year": paper.get("year", 0),
            "authors": paper.get("authors", []),
            "citations": paper.get("citations", 0),
        })
    queries = list(state.get("search_queries", []))
    q = str(output.get("query", "")).strip()
    if q and q not in queries:
        queries.append(q)
    return {**state, "found_papers": existing, "search_queries": queries}


def _session_context_fn(tool_input: dict, state: ResearchState) -> dict:
    if not tool_input.get("session_id"):
        tool_input = {**tool_input, "session_id": state.get("session_id", "default")}
    return _call_registered_tool("session_context", tool_input)


def _session_context_updater(state: ResearchState, output: dict) -> ResearchState:
    return {**state, "context_package": output} if output else state


def _registered_tool_spec(
    tool_name: str,
    estimated_cost: float = 0.06,
    state_updater: Callable[[ResearchState, dict], ResearchState] = _noop_updater,
) -> ToolSpec:
    tool = get_registered_tool(tool_name)
    return ToolSpec(
        name=tool.name,
        description=tool.description,
        input_schema={"type": "object", **tool.input_schema},
        fn=lambda tool_input, state, _name=tool.name: _call_registered_tool(_name, tool_input),
        state_updater=state_updater,
        estimated_cost=estimated_cost,
    )


# ── Named specs ────────────────────────────────────────────────────────────────

DONE = ToolSpec(
    name="done",
    description=get_registered_tool("done").description,
    input_schema={"type": "object", **get_registered_tool("done").input_schema},
    fn=_done_fn,
    state_updater=_done_updater,
    estimated_cost=0.01,
)

SESSION_CONTEXT = ToolSpec(
    name="session_context",
    description=get_registered_tool("session_context").description,
    input_schema={"type": "object", **get_registered_tool("session_context").input_schema},
    fn=_session_context_fn,
    state_updater=_session_context_updater,
    estimated_cost=0.02,
)

LIST_LOCAL_FILES = _registered_tool_spec("list_local_files", estimated_cost=0.02)
READ_LOCAL_FILE = _registered_tool_spec("read_local_file", estimated_cost=0.03)

SAVE_TEXT_ARTIFACT = ToolSpec(
    name="save_text_artifact",
    description=get_registered_tool("save_text_artifact").description,
    input_schema={"type": "object", **get_registered_tool("save_text_artifact").input_schema},
    fn=_save_text_artifact_fn,
    state_updater=_save_text_artifact_updater,
    estimated_cost=0.02,
)

FETCH_URL = _registered_tool_spec("fetch_url", estimated_cost=0.08)
WEB_SEARCH = _registered_tool_spec("web_search", estimated_cost=0.08)
ARXIV_SEARCH = _registered_tool_spec("arxiv_search", estimated_cost=0.08)

PAPER_SEARCH = ToolSpec(
    name="paper_search",
    description=get_registered_tool("paper_search").description,
    input_schema={"type": "object", **get_registered_tool("paper_search").input_schema},
    fn=_paper_search_fn,
    state_updater=_paper_search_updater,
    estimated_cost=0.08,
)

GITHUB_REPO_SEARCH = ToolSpec(
    name="github_repo_search",
    description=get_registered_tool("github_repo_search").description,
    input_schema={"type": "object", **get_registered_tool("github_repo_search").input_schema},
    fn=_github_repo_search_fn,
    state_updater=_github_repo_search_updater,
    estimated_cost=0.08,
)

# ── quick_relevant_check ──────────────────────────────────────────────────────

def _quick_relevant_check_fn(tool_input: dict, state: ResearchState) -> dict:
    """Rank papers by relevance to query using embedding + keyword similarity."""
    from research_agent.llm.router import call_llm_json, system
    from research_agent.prompts import render_prompt

    papers = list(state.get("found_papers", []))
    top_k = int(tool_input.get("top_k", 8))
    query = str(tool_input.get("query") or state.get("selected_question") or "").strip()

    if not papers or not query:
        return {"ranked_papers": papers[:top_k], "total_input": len(papers), "method": "fallback"}

    # Prepare papers list for LLM
    papers_json = json.dumps([
        {
            "title": p.get("title", ""),
            "abstract": p.get("abstract", "")[:500] or "(no abstract)"
        }
        for p in papers[:20]  # Limit input to first 20
    ])

    try:
        resp = call_llm_json(
            "planning",
            [system(render_prompt("react.tools.quick_relevant_check",
                                 query=query,
                                 papers_json=papers_json))],
            max_tokens=3000,
        )
        data = parse_json(resp)
        if isinstance(data, dict) and "ranked_papers" in data:
            ranked = data.get("ranked_papers", [])
            # Map back to original papers dict
            result_papers = []
            for rp in ranked[:top_k]:
                title = rp.get("title", "")
                score = rp.get("relevance_score", 0.5)
                matching_paper = next((p for p in papers if p.get("title", "") == title), None)
                if matching_paper:
                    result_papers.append({**matching_paper, "relevance_score": score})
            return {
                "ranked_papers": result_papers,
                "total_input": len(papers),
                "method": "llm_ranking"
            }
    except Exception as e:
        print(f"  [quick_relevant_check] LLM ranking failed (fallback to BM25): {e}")

    # Fallback: return top papers as-is
    return {"ranked_papers": papers[:top_k], "total_input": len(papers), "method": "fallback"}


def _quick_relevant_check_updater(state: ResearchState, output: dict) -> ResearchState:
    """Filter found_papers to top-K ranked by relevance."""
    ranked = output.get("ranked_papers", [])
    if ranked:
        return {**state, "found_papers": ranked}
    return state


QUICK_RELEVANT_CHECK = ToolSpec(
    name="quick_relevant_check",
    description=(
        "Rank found papers by relevance to the research query. "
        "Uses semantic similarity and keyword matching to filter low-relevance results. "
        "Returns top-K papers to focus reading on high-quality candidates."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "top_k": {
                "type": "integer",
                "description": "Number of top papers to return (default 8)",
                "default": 8,
            },
            "query": {
                "type": "string",
                "description": "Research query for relevance evaluation. If empty, uses current research goal.",
            },
        },
        "required": [],
    },
    fn=_quick_relevant_check_fn,
    state_updater=_quick_relevant_check_updater,
    estimated_cost=0.01,
)


# ── refine_evidence ────────────────────────────────────────────────────────────

def _refine_evidence_fn(tool_input: dict, state: ResearchState) -> dict:
    """Refine evidence extraction from a paper across multiple stages (qualitative/quantitative/limitations)."""
    from research_agent.llm.router import call_llm_json, system
    from research_agent.prompts import render_prompt

    paper_title = str(tool_input.get("paper_title", "")).strip()
    stage = str(tool_input.get("stage", "qualitative")).strip().lower()

    # Find the paper in summaries
    summaries = list(state.get("paper_summaries", []))
    paper = next((s for s in summaries if s.get("title", "").lower() == paper_title.lower()), None)

    if not paper:
        return {"error": f"Paper '{paper_title}' not found in summaries", "refined_evidence": {}}

    # Extract text from paper (or use existing summary if refinement)
    if stage == "qualitative":
        # First pass: extract from abstract/title
        text = (
            f"Title: {paper.get('title', '')}\n"
            f"Problem: {paper.get('problem', '')}\n"
            f"Method: {paper.get('method', '')}\n"
        )
        prompt_key = "react.tools.refine_evidence.qualitative"
    elif stage == "quantitative":
        # Second pass: extract numeric results
        text = (
            f"Result: {paper.get('result', '')}\n"
            f"Existing metrics in paper: {paper.get('quantitative_metrics', 'not yet extracted')}\n"
        )
        prompt_key = "react.tools.refine_evidence.quantitative"
    elif stage == "limitations":
        # Third pass: extract limitations and scope
        text = (
            f"Limitations: {paper.get('limitations', '')}\n"
            f"Existing findings: Problem={paper.get('problem', '')}, Method={paper.get('method', '')}, Result={paper.get('result', '')}\n"
        )
        prompt_key = "react.tools.refine_evidence.limitations"
    else:
        return {"error": f"Unknown stage: {stage}", "refined_evidence": {}}

    try:
        resp = call_llm_json(
            "summarization",
            [system(render_prompt(prompt_key, title=paper.get("title", ""), text=text))],
            max_tokens=2000,
        )
        data = parse_json(resp)
        if isinstance(data, dict):
            return {
                "stage": stage,
                "paper_title": paper_title,
                "refined_evidence": data,
                "status": "success"
            }
    except Exception as e:
        print(f"  [refine_evidence] LLM extraction failed for stage={stage}: {e}")

    return {
        "stage": stage,
        "paper_title": paper_title,
        "refined_evidence": {},
        "status": "failed"
    }


def _refine_evidence_updater(state: ResearchState, output: dict) -> ResearchState:
    """Update paper_summaries with refined evidence for specific stage."""
    if output.get("status") != "success":
        return state

    stage = output.get("stage", "")
    paper_title = output.get("paper_title", "")
    refined = output.get("refined_evidence", {})

    summaries = list(state.get("paper_summaries", []))
    updated = False

    for i, s in enumerate(summaries):
        if s.get("title", "").lower() == paper_title.lower():
            if stage == "qualitative":
                s["problem"] = refined.get("problem", s.get("problem", ""))
                s["method"] = refined.get("method", s.get("method", ""))
                s["result"] = refined.get("result", s.get("result", ""))
            elif stage == "quantitative":
                s["quantitative_metrics"] = refined
            elif stage == "limitations":
                s["limitations"] = refined.get("scope_boundaries", s.get("limitations", ""))
                s["failure_modes"] = refined.get("failure_modes", "")
                s["assumptions"] = refined.get("assumptions", "")
                s["comparison_notes"] = refined.get("comparison_notes", "")
            updated = True
            break

    if updated:
        return {**state, "paper_summaries": summaries}
    return state


REFINE_EVIDENCE = ToolSpec(
    name="refine_evidence",
    description=(
        "Refine/deepen evidence extraction from a specific paper across multiple stages. "
        "Stage 'qualitative': extract problem/method/result. "
        "Stage 'quantitative': force-extract numeric metrics (accuracy, latency, etc). "
        "Stage 'limitations': extract scope, failure modes, assumptions, conflicts with other papers."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "paper_title": {
                "type": "string",
                "description": "Title of the paper to refine (must exist in paper_summaries)",
            },
            "stage": {
                "type": "string",
                "enum": ["qualitative", "quantitative", "limitations"],
                "description": "Extraction stage. Call sequentially: qualitative → quantitative → limitations.",
                "default": "qualitative",
            },
        },
        "required": ["paper_title"],
    },
    fn=_refine_evidence_fn,
    state_updater=_refine_evidence_updater,
    estimated_cost=0.02,
)


PAPER_STORE_QUERY = _registered_tool_spec("paper_store_query", estimated_cost=0.05)
DOWNLOAD_PAPER = _registered_tool_spec("download_paper", estimated_cost=0.10)
EXPLAIN_WITH_EVIDENCE = _registered_tool_spec("explain_with_evidence", estimated_cost=0.06)


# ── Registry ───────────────────────────────────────────────────────────────────

_ALL_TOOLS: list[ToolSpec] = [
    SEARCH_PAPERS,
    READ_PAPERS,
    QUICK_RELEVANT_CHECK,
    REFINE_EVIDENCE,
    OPTIMIZE_QUERIES,
    FIND_GITHUB_REPO,
    DONE,
    SESSION_CONTEXT,
    LIST_LOCAL_FILES,
    READ_LOCAL_FILE,
    SAVE_TEXT_ARTIFACT,
    FETCH_URL,
    WEB_SEARCH,
    ARXIV_SEARCH,
    PAPER_SEARCH,
    GITHUB_REPO_SEARCH,
    PAPER_STORE_QUERY,
    DOWNLOAD_PAPER,
    EXPLAIN_WITH_EVIDENCE,
]

# Maps brain_plan stage names to tool names (for plan-based tool filtering)
_PLAN_STAGE_TO_TOOLS: dict[str, list[str]] = {
    "search": ["search_papers", "quick_relevant_check"],
    "read": ["read_papers", "refine_evidence", "download_paper", "paper_store_query"],
    "query_optimize": ["optimize_queries"],
    "github": ["find_github_repo"],
    "tool_call": [
        "done",
        "session_context",
        "list_local_files",
        "read_local_file",
        "save_text_artifact",
        "fetch_url",
        "web_search",
        "arxiv_search",
        "paper_search",
        "github_repo_search",
        "paper_store_query",
        "download_paper",
        "explain_with_evidence",
    ],
}


def get_tools_for_plan(plan: list[str]) -> list[ToolSpec]:
    """Return only the tools relevant to the given brain_plan stages."""
    tool_names: set[str] = set()
    for stage in plan:
        tool_names.update(_PLAN_STAGE_TO_TOOLS.get(stage, []))
    return [t for t in _ALL_TOOLS if t.name in tool_names]


def get_tool_by_name(name: str) -> ToolSpec | None:
    for spec in _ALL_TOOLS:
        if spec.name == name:
            return spec
    return None
