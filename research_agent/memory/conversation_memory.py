"""
ConversationMemory — 会话级长短记忆 + 上下文防腐化守卫

目标：
1) 支持多轮对话（跨 run）
2) 控制上下文窗口（压缩 + 预算）
3) 降低上下文腐化（固定目标/约束 + 冲突规则）
"""

from __future__ import annotations

import json
import os
from datetime import datetime

_BASE_DIR = "./output/sessions"
_DEFAULT_CHAR_BUDGET = 5000


def _now() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _safe_session_id(session_id: str) -> str:
    keep = []
    for ch in session_id.strip():
        if ch.isalnum() or ch in "-_":
            keep.append(ch)
    return "".join(keep) or "default"


class ConversationMemory:
    """
    每个 session 一个 JSON 文件：
      - short_memory: 最近对话（滑动窗口）
      - long_memory: 结构化记忆（facts/preferences/decisions/open_items）
      - rolling_summary: 压缩摘要
      - guard: 防腐化规则（固定目标/约束/冲突处理）
    """

    def __init__(self, session_id: str):
        self.session_id = _safe_session_id(session_id)
        self.path = os.path.join(_BASE_DIR, f"{self.session_id}.json")
        self.data: dict = {
            "session_id": self.session_id,
            "created_at": _now(),
            "updated_at": _now(),
            "short_memory": [],
            "long_memory": {
                "facts": [],
                "preferences": [],
                "decisions": [],
                "open_items": [],
            },
            "rolling_summary": "",
            "guard": {
                "pinned_goal": "",
                "pinned_constraints": [],
                "conflict_rule": "latest_user_instruction_wins",
                "memory_trust_rule": "prefer_confirmed_facts_over_generated_claims",
            },
        }
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
            except Exception:
                pass

    def save(self):
        self.data["updated_at"] = _now()
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

    def clear(self):
        if os.path.exists(self.path):
            os.remove(self.path)
        self.__init__(self.session_id)

    def add_turn(self, role: str, content: str):
        if not content.strip():
            return
        turns = self.data.get("short_memory", [])
        turns.append(
            {
                "role": role,
                "content": content.strip()[:2500],
                "ts": _now(),
            }
        )
        self.data["short_memory"] = turns[-24:]

    def pin_goal(self, goal: str):
        if goal.strip():
            self.data["guard"]["pinned_goal"] = goal.strip()[:500]

    def set_constraints(self, constraints: list[str]):
        cleaned = [c.strip()[:160] for c in constraints if c.strip()]
        self.data["guard"]["pinned_constraints"] = cleaned[:20]

    def add_fact(self, fact: str):
        facts = self.data["long_memory"].get("facts", [])
        val = fact.strip()[:220]
        if val and val not in facts:
            facts.append(val)
        self.data["long_memory"]["facts"] = facts[-40:]

    def add_decision(self, decision: str):
        decisions = self.data["long_memory"].get("decisions", [])
        val = decision.strip()[:220]
        if val:
            decisions.append(val)
        self.data["long_memory"]["decisions"] = decisions[-30:]

    def _estimate_chars(self, payload: dict) -> int:
        return len(json.dumps(payload, ensure_ascii=False))

    def _compress_turns(self, turns: list[dict], max_chars: int) -> tuple[str, list[dict]]:
        """
        规则压缩：
        - 保留最近几轮完整短记忆
        - 旧轮压缩为摘要 bullet（无 LLM 调用，省 token）
        """
        if not turns:
            return "", []

        keep = turns[-6:]
        old = turns[:-6]
        bullets = []
        for t in old[-14:]:
            content = t.get("content", "").replace("\n", " ").strip()
            if len(content) > 90:
                content = content[:90] + "..."
            bullets.append(f"[{t.get('role','?')}] {content}")
        summary = " | ".join(bullets)
        if len(summary) > max_chars:
            summary = summary[:max_chars] + "..."
        return summary, keep

    def build_context_package(
        self,
        current_input: str,
        char_budget: int = _DEFAULT_CHAR_BUDGET,
    ) -> dict:
        """
        构建给 Planner/Brain 的 JSON 上下文包。
        包含 guard 规则 + 压缩后的记忆，不超过预算。
        """
        short_turns = self.data.get("short_memory", [])
        compressed_summary, short_tail = self._compress_turns(short_turns, max_chars=1200)

        package = {
            "session_id": self.session_id,
            "current_input": current_input[:1000],
            "guard": self.data.get("guard", {}),
            "rolling_summary": self.data.get("rolling_summary", ""),
            "compressed_older_turns": compressed_summary,
            "short_memory_tail": short_tail,
            "long_memory": {
                "facts": self.data.get("long_memory", {}).get("facts", [])[-16:],
                "preferences": self.data.get("long_memory", {}).get("preferences", [])[-10:],
                "decisions": self.data.get("long_memory", {}).get("decisions", [])[-10:],
                "open_items": self.data.get("long_memory", {}).get("open_items", [])[-10:],
            },
            "compression_rule": {
                "policy": "keep_recent_drop_old_detail",
                "budget_chars": char_budget,
                "conflict_resolution": "latest_user_instruction_wins",
            },
        }

        # 超预算时继续瘦身：先裁剪 short tail，再裁 long memory
        while self._estimate_chars(package) > char_budget and len(package["short_memory_tail"]) > 3:
            package["short_memory_tail"] = package["short_memory_tail"][1:]

        while self._estimate_chars(package) > char_budget and package["long_memory"]["facts"]:
            package["long_memory"]["facts"] = package["long_memory"]["facts"][1:]

        while self._estimate_chars(package) > char_budget and package["long_memory"]["decisions"]:
            package["long_memory"]["decisions"] = package["long_memory"]["decisions"][1:]

        return package

