"""
Evidence model — lightweight typed contracts between agents.

Borrows from "Evidence & Citation Binding" in production agent runtimes:
  SubResearcher  →  List[Evidence]  →  Supervisor  →  Writer
  Each claim the Writer makes can reference evidence.id for grounding.

Kept as TypedDict (consistent with existing codebase) rather than Pydantic
to avoid adding a dependency and to stay serialization-transparent (json.dumps works).
"""
from __future__ import annotations

import hashlib
from typing import TypedDict


class Evidence(TypedDict, total=False):
    """
    A single grounded evidence unit produced by a SubResearcher.

    id          — deterministic hash of (title + method[:60]), stable across runs
    cite_key    — ref1, ref2, ... assigned by SubResearcher for Writer citations
    title       — paper / source title
    source_type — "paper" | "web" | "repo" | "unknown"
    problem     — what problem the source addresses (1-2 sentences)
    method      — core technique / approach
    result      — quantitative or qualitative result
    limitations — known caveats
    topic       — which research sub-topic this evidence belongs to
    raw_ref     — original paper dict from paper_store (optional, for debugging)
"""
    id: str
    cite_key: str
    title: str
    source_type: str
    problem: str
    method: str
    result: str
    limitations: str
    topic: str


def make_evidence_id(title: str, method: str = "") -> str:
    key = f"{title.strip().lower()}::{method.strip()[:60].lower()}"
    return "ev_" + hashlib.md5(key.encode()).hexdigest()[:10]


def evidence_from_summary(
    summary: dict,
    cite_key: str,
    topic: str = "",
) -> Evidence:
    """Build an Evidence from a paper_summary dict (reader agent output)."""
    title = str(summary.get("title", "") or "")
    method = str(summary.get("method", "") or "")
    return Evidence(
        id=make_evidence_id(title, method),
        cite_key=cite_key,
        title=title,
        source_type="paper",
        problem=str(summary.get("problem", "") or "")[:300],
        method=method[:300],
        result=str(summary.get("result", "") or "")[:300],
        limitations=str(summary.get("limitations", "") or "")[:150],
        topic=topic,
    )


def deduplicate_evidence(items: list[Evidence]) -> list[Evidence]:
    """Remove duplicates by id, keeping first occurrence."""
    seen: set[str] = set()
    out: list[Evidence] = []
    for e in items:
        eid = e.get("id", "")
        if eid and eid not in seen:
            seen.add(eid)
            out.append(e)
        elif not eid:
            out.append(e)
    return out
