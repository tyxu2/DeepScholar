# Evidence Contract

## Schema (`runtime/evidence.py`)

```python
class Evidence(TypedDict, total=False):
    id: str          # deterministic: "ev_" + MD5(title::method[:60])
    cite_key: str    # "ref1", "ref2", ... assigned by SubResearcher
    title: str
    source_type: str # "paper" | "repo" | "web"
    problem: str
    method: str      # max 700 chars in state payload
    result: str      # max 700 chars in state payload
    limitations: str # max 300 chars
    topic: str       # sub_question or research topic that produced this
```

## Flow Path

```
SubResearcher._extract_evidence()
    → List[Evidence]
    → returned in sub_researcher result dict

_merge_sub_researcher_results()
    → deduplicate_evidence(existing + new)
    → written to state["evidence"]

analyzer_node()
    → deduplicate_evidence(state["evidence"])
    → final state["evidence"]

writer_node()
    → reads state["evidence"] for citation binding (future: [refN] links)
```

## Deduplication

`make_evidence_id(title, method)` produces a deterministic hash.
`deduplicate_evidence(items)` keeps the first occurrence of each id.
Dedup runs at every merge boundary to prevent evidence inflation.

## Rules

1. `evidence` is **never overwritten** mid-pipeline — only appended + deduped.
2. SubResearcher must always run `_extract_evidence()` even if compress LLM fails (fallback from `paper_summaries`).
3. `cite_key` must be stable within a session; do not reassign.
4. `topic` field should be set to the `sub_question` that triggered this sub-researcher.
