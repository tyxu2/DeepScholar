"""
SessionMemory — 分层记忆架构（中期记忆层）

三层记忆架构：
  Layer 1 - Working Memory:  ResearchState（in-process TypedDict，当前 run 内有效）
  Layer 2 - Session Memory:  本文件（JSON 持久化，跨 run 共享，自动过期）
  Layer 3 - Knowledge Base:  PaperStore（LlamaIndex 向量库，长期论文知识存储）

Session Memory 存储内容：
  - 历史搜索词（避免重复搜索）
  - 已读论文 ID 集合（避免重复下载解析）
  - 草稿版本历史（支持回滚）
  - 研究方向笔记（跨 run 积累的洞察）
"""

from __future__ import annotations
import json
import os
from datetime import datetime, timedelta
from typing import Optional

_SESSION_PATH = "./output/session_memory.json"
_DRAFT_HISTORY_PATH = "./output/draft_history.json"
_MAX_DRAFT_VERSIONS = 5
_QUERY_CACHE_DAYS = 7     # 搜索词缓存有效期（天）
_PAPER_CACHE_DAYS = 30    # 已读论文缓存有效期（天）


class SessionMemory:
    """
    中期记忆：JSON 持久化，跨 run 共享。

    用法：
        mem = SessionMemory()
        # 记录搜索词
        mem.add_search_query("LLM inference optimization")
        # 检查是否已读过某论文
        if not mem.is_paper_read("arxiv:2401.12345"):
            ...
        # 保存草稿版本
        mem.save_draft_version(draft_dict, score=7.5)
        mem.save()
    """

    def __init__(self, path: str = _SESSION_PATH):
        self.path = path
        self._data: dict = {
            "search_queries": [],      # [{query, ts}]
            "read_paper_ids": [],      # [{id, title, ts}]
            "research_notes": [],      # [{topic, note, ts}]
            "run_history": [],         # [{ts, topic, score, plan}]
        }
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                    self._data.update(loaded)
            except Exception:
                pass

    def save(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)

    # ── 搜索词记忆 ────────────────────────────────────────────────────────────

    def add_search_query(self, query: str):
        """记录一个搜索词（带时间戳）。"""
        self._data["search_queries"].append({
            "query": query,
            "ts": _now(),
        })

    def get_recent_queries(self, days: int = _QUERY_CACHE_DAYS) -> list[str]:
        """返回最近 N 天内使用过的搜索词。"""
        cutoff = _days_ago(days)
        return [
            item["query"]
            for item in self._data["search_queries"]
            if item.get("ts", "") >= cutoff
        ]

    def is_query_seen(self, query: str, days: int = _QUERY_CACHE_DAYS) -> bool:
        recent = self.get_recent_queries(days)
        return query.lower() in [q.lower() for q in recent]

    # ── 已读论文记忆 ──────────────────────────────────────────────────────────

    def mark_paper_read(self, paper_id: str, title: str = ""):
        """标记一篇论文已被读取解析。"""
        if not self.is_paper_read(paper_id):
            self._data["read_paper_ids"].append({
                "id": paper_id,
                "title": title,
                "ts": _now(),
            })

    def is_paper_read(self, paper_id: str, days: int = _PAPER_CACHE_DAYS) -> bool:
        cutoff = _days_ago(days)
        for item in self._data["read_paper_ids"]:
            if item.get("id") == paper_id and item.get("ts", "") >= cutoff:
                return True
        return False

    def get_read_titles(self) -> list[str]:
        return [item.get("title", "") for item in self._data["read_paper_ids"]]

    # ── 研究笔记 ──────────────────────────────────────────────────────────────

    def add_note(self, topic: str, note: str):
        """添加关于某个研究方向的洞察笔记。"""
        self._data["research_notes"].append({
            "topic": topic,
            "note": note,
            "ts": _now(),
        })

    def get_notes(self, topic: str = "") -> list[str]:
        notes = self._data["research_notes"]
        if topic:
            notes = [n for n in notes if topic.lower() in n.get("topic", "").lower()]
        return [n["note"] for n in notes]

    # ── Run 历史 ──────────────────────────────────────────────────────────────

    def record_run(self, topic: str, plan: list[str], score: float):
        """记录一次完整 run 的摘要。"""
        self._data["run_history"].append({
            "ts": _now(),
            "topic": topic,
            "plan": plan,
            "score": score,
        })
        # 只保留最近 20 条
        self._data["run_history"] = self._data["run_history"][-20:]

    def get_run_history(self) -> list[dict]:
        return list(reversed(self._data["run_history"]))

    def context_for_brain(self, current_topic: str) -> str:
        """
        为 Brain Agent 提供历史上下文摘要。
        帮助 Brain Agent 避免重复已知工作、从过去的成败中学习。
        """
        lines = []
        recent_queries = self.get_recent_queries(days=3)
        if recent_queries:
            lines.append(f"近期已使用搜索词：{', '.join(recent_queries[:8])}")
        read_titles = self.get_read_titles()
        if read_titles:
            lines.append(f"已读论文（{len(read_titles)} 篇）：{', '.join(read_titles[:5])}{'...' if len(read_titles) > 5 else ''}")
        notes = self.get_notes(current_topic)
        if notes:
            lines.append(f"相关研究笔记：{notes[0][:200]}")
        history = self.get_run_history()
        if history:
            last = history[0]
            lines.append(f"上次运行：{last['topic'][:60]}（得分 {last['score']:.0%}）")
        return "\n".join(lines) if lines else "（无历史上下文）"


# ── 草稿版本历史 ──────────────────────────────────────────────────────────────

class DraftHistory:
    """
    草稿版本管理：保存每轮 write-critic 循环的草稿快照。
    支持查看历史版本、回滚到最佳版本。
    """

    def __init__(self, path: str = _DRAFT_HISTORY_PATH):
        self.path = path
        self._versions: list[dict] = []
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self._versions = json.load(f)
            except Exception:
                self._versions = []

    def save_version(self, draft: dict, score: float, revision: int):
        """保存一个草稿版本快照。"""
        self._versions.append({
            "ts": _now(),
            "revision": revision,
            "score": score,
            "draft": draft,
        })
        # 只保留最近 N 个版本
        if len(self._versions) > _MAX_DRAFT_VERSIONS:
            self._versions = self._versions[-_MAX_DRAFT_VERSIONS:]
        self._persist()

    def get_best_version(self) -> Optional[dict]:
        """返回历史上 critic score 最高的草稿版本。"""
        if not self._versions:
            return None
        return max(self._versions, key=lambda v: v.get("score", 0)).get("draft")

    def get_latest_version(self) -> Optional[dict]:
        if not self._versions:
            return None
        return self._versions[-1].get("draft")

    def _persist(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._versions, f, ensure_ascii=False, indent=2)

    def clear(self):
        self._versions = []
        if os.path.exists(self.path):
            os.remove(self.path)


# ── 工具函数 ──────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.utcnow().isoformat()


def _days_ago(days: int) -> str:
    return (datetime.utcnow() - timedelta(days=days)).isoformat()
