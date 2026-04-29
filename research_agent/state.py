from typing import TYPE_CHECKING, TypedDict, Annotated, List, Literal
import operator

if TYPE_CHECKING:
    from research_agent.runtime.evidence import Evidence


class Paper(TypedDict):
    title: str
    abstract: str
    url: str
    pdf_path: str
    year: int
    authors: List[str]
    citations: int


class Summary(TypedDict):
    title: str
    problem: str
    method: str
    result: str
    contributions: List[str]
    limitations: str
    worth_reproducing: bool


class Draft(TypedDict):
    intro: str
    related_work: str
    method: str
    experiments: str
    conclusion: str
    abstract: str


class TaskItem(TypedDict):
    id: str
    stage: str
    goal: str
    status: Literal["pending", "running", "done", "failed", "skipped"]
    depends_on: List[str]


class Artifact(TypedDict):
    id: str
    kind: str
    title: str
    source_stage: str
    content_ref: str
    evidence_ids: List[str]


class ActionRecord(TypedDict):
    step: int
    stage: str
    thought: str
    observation: str
    status: Literal["success", "error", "skipped"]


class ExecutorTurn(TypedDict, total=False):
    step: int
    thought: str
    action: str
    observation: str
    reflection: str
    decision: Literal["continue", "finish", "handoff", "error"]
    status: Literal["success", "error", "finish"]


class RunBudget(TypedDict):
    max_steps: int
    max_errors: int
    used_steps: int
    used_errors: int


class ResearchState(TypedDict):
    # ═══════════════════════════════════════════════════════════
    # ACTIVE FIELDS (Supervisor + SubResearcher architecture)
    # See: docs/rules/state_contract.md for per-stage ownership
    # ═══════════════════════════════════════════════════════════

    # ── Brain (owner: brain_node) ──────────────────────────────
    raw_input: str
    selected_question: str
    session_id: str
    task_profile: Literal[
        "literature_review",
        "repo_research",
        "math_reasoning",
        "paper_from_implementation",
        "general_research",
        "general_task",
    ]
    user_constraints: dict
    input_paths: List[str]
    local_context: str
    target_tex_path: str
    output_format: str
    paper_store_dir: str
    research_brief: dict           # {goal, doc_type, word_count, output_language, sub_questions}
    research_brief_confidence: float
    brief_ready: bool
    brief_missing_fields: List[str]
    brain_plan: List[str]          # stages to execute: ["research", "write"]
    doc_type: Literal["survey", "experimental", "report"]

    # ── Analyzer Supervisor (owner: analyzer_node) ─────────────
    found_papers: List[Paper]
    confirmed_papers: List[Paper]
    paper_summaries: List[Summary]
    search_queries: List[str]
    github_repo_url: str
    experiment_results: dict
    analysis_brief: str
    analysis_key_points: List[str]
    analysis_open_risks: List[str]
    analysis_writer_focus: str
    analysis_critic_focus: str
    selection_reason: str
    analyzer_stop_reason: str
    analyzer_supervisor_round: int
    analyzer_supervisor_max_rounds: int
    analyzer_trace: List[ExecutorTurn]
    context_package: dict          # per-stage debug snapshots
    agent_capabilities: dict
    stop_reason: str

    # ── Evidence (owner: sub_researcher → analyzer_node) ───────
    # Structured Evidence cards; flows SubResearcher → Analyzer → Writer
    # See: docs/rules/evidence_contract.md
    evidence: List[dict]           # runtime type: List[Evidence]
    evidence_ids: List[str]

    # ── Writer (owner: writer_node) ────────────────────────────
    draft: Draft
    raw_draft: dict
    draft_md_path: str
    draft_latex_path: str
    writer_head_round: int
    writer_head_max_rounds: int
    writer_revision_brief: str
    writer_revision_suggestions: List[str]
    artifacts: List[Artifact]

    # ── Critic (owner: critic_node) ────────────────────────────
    critique: str
    critique_score: float
    critique_accepted: bool
    critique_accept_threshold: float
    critic_gate_passed: bool
    revision_count: int

    # ── Control flow ───────────────────────────────────────────
    current_stage: str
    next: str
    error_log: Annotated[List[str], operator.add]
    run_budget: RunBudget
    assistant_response: str
    tasks: List[TaskItem]
    action_history: List[ActionRecord]

    # ═══════════════════════════════════════════════════════════
    # DEPRECATED — legacy graph/executor/TAOR fields
    # Kept for graph.py / tools.py compatibility; do not use in new agents
    # ═══════════════════════════════════════════════════════════
    topic_mode: Literal["A", "B", "C"]       # deprecated: old A/B/C routing
    research_questions: List[str]             # deprecated: use selected_question
    frontdesk_intent: str                     # deprecated: graph.py only
    frontdesk_confidence: float               # deprecated: graph.py only
    frontdesk_next_action: str                # deprecated: graph.py only
    writer_rules_path: str                    # deprecated: use user_constraints
    human_in_loop: bool                       # deprecated: HITL not implemented
    hitl_stages: List[str]                    # deprecated: HITL not implemented
    human_feedback: str                       # deprecated: HITL not implemented
    messages: Annotated[List[dict], operator.add]  # deprecated
    engine: Literal["classic", "react"]       # deprecated: always react
    executor_goal: str                        # deprecated: use research_brief
    executor_rollouts: int                    # deprecated: no more rollouts
    executor_parallelism: int                 # deprecated: use agent_capabilities
    executor_trace: List[ExecutorTurn]        # deprecated
    execute_actions: List[dict]               # deprecated
    selected_rollout_id: int                  # deprecated: compat for critic/writer
    rollout_summaries: List[dict]             # deprecated
    last_observation: str                     # deprecated
    last_reflection: str                      # deprecated
    replan_reason: str                        # deprecated
    replan_next: str                          # deprecated
    confidence_score: float                   # deprecated: cli display only
    confidence_label: Literal["high", "medium", "low"]  # deprecated: cli display only
    quality_objectives: dict                  # deprecated
    evidence_quality: dict                    # deprecated
    evidence_repair_rounds: int               # deprecated
    relevance_filtered_count: int             # deprecated
    optimization_round: int                   # deprecated
    optimized_queries: List[str]              # deprecated
    brain_plan_index: int                     # deprecated
    key_paper: str                            # deprecated: repo_research only
    reasoning_notes: str                      # deprecated
    math_derivations: List[str]               # deprecated
    reproduced_code: str                      # deprecated
    improvement_plan: str                     # deprecated
    improved_code: str                        # deprecated
